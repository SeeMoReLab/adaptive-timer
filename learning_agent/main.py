import argparse
import base64
import re
import statistics
import threading
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import grpc
from google.protobuf import empty_pb2
import torch

import agent_pb2
import agent_pb2_grpc
from models import (
    ActionDecoder,
    FeatureExtractor,
    LinearTorchModel,
    ReportFeatureExtractor,
    ScheduledTimeoutModel,
    TensorModel,
    TimeoutActionDecoder,
    timeout_ms_to_index,
)

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in ("true", "1", "yes", "y", "on"):
        return True
    if lowered in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def reward_engineering(throughput: int) -> float:
    """Quadratic shaping below 1000, linear above (prior-work style)."""
    value = float(max(0, throughput))
    if value < 1000.0:
        return (value / 1000.0) ** 2 * 1000.0
    return value


@dataclass(frozen=True)
class HostEntry:
    node_id: int
    host: str
    agent_port: int
    consensus_port: int


def load_hosts_config(path: Path) -> List[HostEntry]:
    entries: List[HostEntry] = []
    with path.open("r") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(
                    f"Invalid hosts config at line {lineno}: expected at least 6 columns"
                )
            try:
                node_id = int(parts[0])
                host = parts[1]
                agent_port = int(parts[4])
                consensus_port = int(parts[5])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid hosts config at line {lineno}: {line}"
                ) from exc
            entries.append(
                HostEntry(
                    node_id=node_id,
                    host=host,
                    agent_port=agent_port,
                    consensus_port=consensus_port,
                )
            )
    if not entries:
        raise ValueError(f"No hosts found in config: {path}")
    return entries


def compute_threshold(num_nodes: int) -> int:
    if num_nodes <= 0:
        raise ValueError("Number of nodes must be positive")
    f = (num_nodes - 1) // 3
    return 2 * f + 1


def _key_paths(key_dir: Path, node_id: int) -> Tuple[Path, Path]:
    private_path = key_dir / f"node_{node_id}.ecdsa"
    public_path = key_dir / f"node_{node_id}.ecdsa.pub"
    return private_path, public_path


DEFAULT_PRIVATE_KEY_DER_B64 = (
    "MD4CAQAwEAYHKoZIzj0CAQYFK4EEAAoEJzAlAgEBBCBnhIob4JXH+WpaNiL72BlbtUMAIBQo"
    "M852d+tKFBb7fg=="
)

DEFAULT_PUBLIC_KEY_DER_B64 = (
    "MFYwEAYHKoZIzj0CAQYFK4EEAAoDQgAEavNEKGRcmB7u49alxowlwCi1s24ANOpOQ9UiFBxg"
    "qnO/RfOl3BJm0qE2IJgCnvL7XUetwj5C/8MnMWi9ux2aeQ=="
)

def _load_default_private_key() -> ec.EllipticCurvePrivateKey:
    private_der = base64.b64decode(DEFAULT_PRIVATE_KEY_DER_B64)
    private_key = serialization.load_der_private_key(private_der, password=None)
    if not isinstance(private_key, ec.EllipticCurvePrivateKey):
        raise ValueError("Default private key is not an ECDSA key.")
    return private_key


def _load_default_public_key() -> ec.EllipticCurvePublicKey:
    public_der = base64.b64decode(DEFAULT_PUBLIC_KEY_DER_B64)
    public_key = serialization.load_der_public_key(public_der)
    if not isinstance(public_key, ec.EllipticCurvePublicKey):
        raise ValueError("Default public key is not an ECDSA key.")
    return public_key


def load_private_key(
    node_id: int, key_dir: Path, use_default_key: bool
) -> ec.EllipticCurvePrivateKey:
    if use_default_key:
        return _load_default_private_key()

    private_path, public_path = _key_paths(key_dir, node_id)
    if not private_path.exists() or not public_path.exists():
        raise ValueError(
            f"Missing key files for node {node_id} in {key_dir}. "
            "Generate keys before starting the agent."
        )

    private_bytes = private_path.read_bytes()
    private_key = serialization.load_pem_private_key(private_bytes, password=None)
    if not isinstance(private_key, ec.EllipticCurvePrivateKey):
        raise ValueError(f"Invalid ECDSA private key: {private_path}")
    return private_key


def load_trusted_public_keys(key_dir: Path) -> Dict[int, ec.EllipticCurvePublicKey]:
    keys: Dict[int, ec.EllipticCurvePublicKey] = {}
    if not key_dir.exists():
        return keys
    pattern = re.compile(r"node_(\d+)\.ecdsa\.pub$")
    for path in key_dir.glob("node_*.ecdsa.pub"):
        match = pattern.match(path.name)
        if not match:
            continue
        node_id = int(match.group(1))
        public_bytes = path.read_bytes()
        public_key = serialization.load_pem_public_key(public_bytes)
        if not isinstance(public_key, ec.EllipticCurvePublicKey):
            raise ValueError(f"Invalid ECDSA public key: {path}")
        keys[node_id] = public_key
    return keys


def serialize_report_for_signing(report: agent_pb2.ReportLocal) -> bytes:
    clone = agent_pb2.ReportLocal()
    clone.CopyFrom(report)
    clone.signature = b""
    return clone.SerializeToString(deterministic=True)


class LearningAgentServicer(agent_pb2_grpc.LearningAgentServicer):
    def __init__(
        self,
        node_id: int,
        peers: List[HostEntry],
        key_dir: Path,
        collect_wait_s: float,
        sharing: bool,
        model: TensorModel,
        feature_extractor: FeatureExtractor[agent_pb2.Report],
        action_decoder: ActionDecoder[agent_pb2.Timeout],
        consensus_stub: Optional[agent_pb2_grpc.ConsensusStub],
        share_timeout_s: Optional[float],
        consensus_timeout_s: Optional[float],
        signing_enabled: bool,
        use_default_keys: bool,
    ):
        self.node_id = node_id
        self._collect_wait_s = collect_wait_s
        self._sharing = sharing
        self._model = model
        self._feature_extractor = feature_extractor
        self._action_decoder = action_decoder
        self._consensus_stub = consensus_stub
        self._share_timeout_s = share_timeout_s
        self._consensus_timeout_s = consensus_timeout_s
        self._signing_enabled = signing_enabled
        self._use_default_keys = use_default_keys

        self._private_key: Optional[ec.EllipticCurvePrivateKey] = None
        if self._signing_enabled:
            self._private_key = load_private_key(node_id, key_dir, use_default_keys)

        self._trusted_public_keys: Dict[int, ec.EllipticCurvePublicKey] = {}
        self._default_public_key: Optional[ec.EllipticCurvePublicKey] = None
        if self._signing_enabled:
            if use_default_keys:
                self._default_public_key = _load_default_public_key()
            else:
                self._trusted_public_keys = load_trusted_public_keys(key_dir)

        self._peer_stubs: Dict[int, agent_pb2_grpc.LearningAgentStub] = {}
        if self._sharing:
            for peer in peers:
                if peer.node_id == node_id:
                    continue
                channel = grpc.insecure_channel(f"{peer.host}:{peer.agent_port}")
                self._peer_stubs[peer.node_id] = agent_pb2_grpc.LearningAgentStub(channel)

        if self._signing_enabled and not self._use_default_keys:
            missing_keys = [
                peer.node_id
                for peer in peers
                if peer.node_id not in self._trusted_public_keys
            ]
            if missing_keys:
                missing_str = ", ".join(str(node) for node in sorted(missing_keys))
                raise ValueError(
                    "Missing public keys for nodes: "
                    f"{missing_str}. Generate keys before starting the agent."
                )

        self._threshold = compute_threshold(len(peers))
        self._lock = threading.Lock()
        self._current_episode = 0
        self._local_accepted_episode: Optional[int] = None
        self._reports_by_episode: Dict[int, Dict[int, agent_pb2.ReportLocal]] = {}
        self._timers_by_episode: Dict[int, threading.Timer] = {}
        self._submitted_episodes: Set[int] = set()
        self._finalized_episodes: Set[int] = set()
        self._received_episodes: Set[int] = set()
        self._pending_episodes: Set[int] = set()
        self._timeouts_by_episode: Dict[int, agent_pb2.Timeout] = {}
        self._features_by_episode: Dict[int, torch.Tensor] = {}

    def SendReport(self, request: agent_pb2.ReportLocal, context) -> empty_pb2.Empty:
        local_report = agent_pb2.ReportLocal(
            node_id=self.node_id,
            episode=request.episode,
            state=request.state,
        )

        signed_report: Optional[agent_pb2.ReportLocal] = None
        with self._lock:
            self._apply_reward_update_locked(request)

            if local_report.episode < self._current_episode:
                context.abort(
                    grpc.StatusCode.OUT_OF_RANGE,
                    "Episode is older than current episode.",
                )
            if self._local_accepted_episode == local_report.episode:
                context.abort(
                    grpc.StatusCode.ALREADY_EXISTS,
                    "Local report already accepted for this episode.",
                )
            if local_report.episode > self._current_episode:
                self._current_episode = local_report.episode
                self._local_accepted_episode = None

            self._local_accepted_episode = local_report.episode
            self._received_episodes.add(local_report.episode)

            if not self._sharing:
                timeout = self._predict_timeout_locked(
                    local_report.episode, local_report.state
                )
                self._timeouts_by_episode[local_report.episode] = timeout
            else:
                self._pending_episodes.add(local_report.episode)
                signed_report = self._sign_report(local_report)
                self._store_report_locked(signed_report)

        if signed_report is not None:
            self._broadcast_signed_report_async(signed_report)

        return empty_pb2.Empty()

    def ShareReport(self, request: agent_pb2.ReportLocal, context) -> empty_pb2.Empty:
        if not self._sharing:
            return empty_pb2.Empty()
        if not self._verify_report(request):
            return empty_pb2.Empty()

        stored = agent_pb2.ReportLocal()
        stored.CopyFrom(request)

        with self._lock:
            if stored.episode < self._current_episode:
                return empty_pb2.Empty()
            if stored.episode in self._finalized_episodes:
                return empty_pb2.Empty()
            if stored.episode in self._submitted_episodes:
                return empty_pb2.Empty()
            self._store_report_locked(stored)

        return empty_pb2.Empty()

    def DeliverConsensus(
        self, request: agent_pb2.ReportBatch, context
    ) -> empty_pb2.Empty:
        if not request.reports:
            return empty_pb2.Empty()

        print(f"[episode {request.episode}] consensus batch {len(request.reports)} reports")
        for report in request.reports:
            local = report.state
            print(
                f"node={report.node_id} "
                f"processed={local.processed_transactions} "
                f"avg_delay={local.avg_message_delay:.6f} "
                f"max_delay={local.max_message_delay:.6f} "
                f"min_delay={local.min_message_delay:.6f} "
                f"std_delay={local.std_message_delay:.6f}"
            )

        median_report = self._median_report(request.reports)
        with self._lock:
            if request.episode in self._finalized_episodes:
                return empty_pb2.Empty()
            if request.episode > self._current_episode:
                self._current_episode = request.episode
                self._local_accepted_episode = None

            timer = self._timers_by_episode.pop(request.episode, None)
            if timer is not None:
                timer.cancel()
            self._reports_by_episode.pop(request.episode, None)
            self._submitted_episodes.add(request.episode)
            self._finalized_episodes.add(request.episode)
            self._pending_episodes.discard(request.episode)
            self._received_episodes.add(request.episode)

            timeout = self._predict_timeout_locked(request.episode, median_report)
            self._timeouts_by_episode[request.episode] = timeout

        return empty_pb2.Empty()

    def GetTimeout(
        self, request: agent_pb2.TimeoutRequest, context
    ) -> agent_pb2.TimeoutStatus:
        with self._lock:
            timeout = self._timeouts_by_episode.get(request.episode)
            if timeout is not None:
                return agent_pb2.TimeoutStatus(
                    episode=request.episode,
                    status=agent_pb2.TimeoutStatus.Status.READY,
                    timeout=timeout,
                )
            if request.episode in self._pending_episodes or request.episode in self._received_episodes:
                return agent_pb2.TimeoutStatus(
                    episode=request.episode,
                    status=agent_pb2.TimeoutStatus.Status.PENDING,
                )
            return agent_pb2.TimeoutStatus(
                episode=request.episode,
                status=agent_pb2.TimeoutStatus.Status.NOT_RECEIVED,
            )

    def Reset(self, request: empty_pb2.Empty, context) -> empty_pb2.Empty:
        with self._lock:
            for timer in self._timers_by_episode.values():
                timer.cancel()
            self._timers_by_episode.clear()
            self._reports_by_episode.clear()
            self._submitted_episodes.clear()
            self._finalized_episodes.clear()
            self._received_episodes.clear()
            self._pending_episodes.clear()
            self._timeouts_by_episode.clear()
            self._features_by_episode.clear()
            self._current_episode = 0
            self._local_accepted_episode = None
        return empty_pb2.Empty()

    def _store_report_locked(self, report: agent_pb2.ReportLocal) -> None:
        if report.episode in self._finalized_episodes:
            return
        if report.episode in self._submitted_episodes:
            return

        reports_for_episode = self._reports_by_episode.setdefault(report.episode, {})
        if report.node_id in reports_for_episode:
            return
        reports_for_episode[report.node_id] = report

        if (
            len(reports_for_episode) >= self._threshold
            and report.episode not in self._timers_by_episode
            and report.episode not in self._submitted_episodes
        ):
            timer = threading.Timer(
                self._collect_wait_s, self._finalize_episode, args=(report.episode,)
            )
            self._timers_by_episode[report.episode] = timer
            timer.daemon = True
            timer.start()

    def _finalize_episode(self, episode: int) -> None:
        with self._lock:
            if episode in self._finalized_episodes or episode in self._submitted_episodes:
                return
            reports_for_episode = self._reports_by_episode.get(episode, {})
            if not reports_for_episode:
                return
            snapshot = list(reports_for_episode.values())
            self._submitted_episodes.add(episode)
            self._reports_by_episode.pop(episode, None)
            self._timers_by_episode.pop(episode, None)

        if self._consensus_stub is None:
            print(f"[episode {episode}] consensus stub not configured")
            return

        batch = agent_pb2.ReportBatch(episode=episode, reports=snapshot)
        try:
            if self._consensus_timeout_s is None:
                self._consensus_stub.SubmitReportBatch(batch)
            else:
                self._consensus_stub.SubmitReportBatch(
                    batch, timeout=self._consensus_timeout_s
                )
        except grpc.RpcError as exc:
            print(f"SubmitReportBatch for episode {episode} failed: {exc}")

    def _sign_report(self, report: agent_pb2.ReportLocal) -> agent_pb2.ReportLocal:
        signed = agent_pb2.ReportLocal()
        signed.CopyFrom(report)
        if self._signing_enabled and self._private_key is not None:
            signed.signature = self._private_key.sign(
                serialize_report_for_signing(signed), ec.ECDSA(hashes.SHA256())
            )
        return signed

    def _verify_report(self, report: agent_pb2.ReportLocal) -> bool:
        if not self._signing_enabled:
            return True
        if not report.signature:
            return False

        if self._use_default_keys:
            public_key = self._default_public_key
        else:
            public_key = self._trusted_public_keys.get(report.node_id)
        if public_key is None:
            return False

        try:
            public_key.verify(
                report.signature,
                serialize_report_for_signing(report),
                ec.ECDSA(hashes.SHA256()),
            )
        except Exception:
            return False

        return True

    def _broadcast_signed_report_async(self, report: agent_pb2.ReportLocal) -> None:
        thread = threading.Thread(
            target=self._broadcast_signed_report, args=(report,), daemon=True
        )
        thread.start()

    def _broadcast_signed_report(self, report: agent_pb2.ReportLocal) -> None:
        for peer_id, stub in self._peer_stubs.items():
            try:
                if self._share_timeout_s is None:
                    stub.ShareReport(report)
                else:
                    stub.ShareReport(report, timeout=self._share_timeout_s)
            except grpc.RpcError as exc:
                print(f"ShareReport to node {peer_id} failed: {exc}")

    def _median_report(self, reports: List[agent_pb2.ReportLocal]) -> agent_pb2.Report:
        processed = int(
            round(
                statistics.median(
                    [entry.state.processed_transactions for entry in reports]
                )
            )
        )
        avg_delay = statistics.median(
            [entry.state.avg_message_delay for entry in reports]
        )
        max_delay = statistics.median(
            [entry.state.max_message_delay for entry in reports]
        )
        min_delay = statistics.median(
            [entry.state.min_message_delay for entry in reports]
        )
        std_delay = statistics.median(
            [entry.state.std_message_delay for entry in reports]
        )
        return agent_pb2.Report(
            processed_transactions=processed,
            avg_message_delay=avg_delay,
            max_message_delay=max_delay,
            min_message_delay=min_delay,
            std_message_delay=std_delay,
        )

    def _apply_reward_update_locked(self, request: agent_pb2.ReportLocal) -> None:
        if not request.HasField("reward"):
            return

        feedback = request.reward
        features = self._features_by_episode.get(feedback.episode)
        if features is None:
            print(
                "[learning] skipped reward update "
                f"(episode={feedback.episode}): no stored features for episode"
            )
            return

        timeout_ms = int(feedback.timeout_used.timeout_milliseconds)
        try:
            action_index = timeout_ms_to_index(timeout_ms)
        except ValueError as exc:
            print(f"[learning] skipped reward for episode {feedback.episode}: {exc}")
            return

        engineered_reward = reward_engineering(feedback.report.processed_transactions)
        action = torch.tensor(float(action_index), dtype=torch.float32)
        self._model.update_model(features, action, engineered_reward)
        # Reward for this episode has been consumed.
        self._features_by_episode.pop(feedback.episode, None)
        print(
            "[learning] applied reward update "
            f"(episode={feedback.episode}, timeout_ms={timeout_ms}, "
            f"throughput={feedback.report.processed_transactions}, "
            f"reward={engineered_reward:.2f})"
        )

    def _predict_timeout_locked(
        self, episode: int, report: agent_pb2.Report
    ) -> agent_pb2.Timeout:
        features = self._feature_extractor.featurize(report)
        self._features_by_episode[episode] = features.detach().clone()
        action = self._model.predict_action(features)
        return self._action_decoder.decode(action)


def serve(
    node_id: int,
    hosts_config: Path,
    key_dir: Optional[Path],
    collect_wait_ms: int,
    sharing: bool,
    share_timeout_ms: Optional[int],
    consensus_timeout_ms: Optional[int],
    host: str,
    port: Optional[int],
    signing_enabled: bool,
    model_type: str,
    schedule_file: Optional[Path],
):
    peers = load_hosts_config(hosts_config)
    entry = next((peer for peer in peers if peer.node_id == node_id), None)
    if entry is None:
        raise ValueError(f"Node id {node_id} not found in {hosts_config}")
    bind_port = port if port is not None else entry.agent_port

    share_timeout_s = None if share_timeout_ms is None else share_timeout_ms / 1000.0
    consensus_timeout_s = (
        None if consensus_timeout_ms is None else consensus_timeout_ms / 1000.0
    )

    use_default_keys = key_dir is None
    effective_key_dir = key_dir if key_dir is not None else Path(".")

    consensus_stub = None
    if sharing:
        consensus_channel = grpc.insecure_channel(
            f"{entry.host}:{entry.consensus_port}"
        )
        consensus_stub = agent_pb2_grpc.ConsensusStub(consensus_channel)

    if model_type == "linear":
        model = LinearTorchModel(n_features=5, lr=0.05)
    elif model_type == "scheduled":
        if schedule_file is None:
            raise ValueError("--schedule-file is required when --model-type=scheduled")
        model = ScheduledTimeoutModel.from_csv(schedule_file)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    feature_extractor = ReportFeatureExtractor()
    action_decoder = TimeoutActionDecoder()
    servicer = LearningAgentServicer(
        node_id=node_id,
        peers=peers,
        key_dir=effective_key_dir,
        collect_wait_s=collect_wait_ms / 1000.0,
        sharing=sharing,
        model=model,
        feature_extractor=feature_extractor,
        action_decoder=action_decoder,
        consensus_stub=consensus_stub,
        share_timeout_s=share_timeout_s,
        consensus_timeout_s=consensus_timeout_s,
        signing_enabled=signing_enabled,
        use_default_keys=use_default_keys,
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    agent_pb2_grpc.add_LearningAgentServicer_to_server(servicer, server)
    server.add_insecure_port(f"{host}:{bind_port}")
    server.start()

    print(
        "LearningAgent server running at "
        f"{host}:{bind_port} (node_id={node_id}, threshold={servicer._threshold})"
    )
    server.wait_for_termination()


def main():
    parser = argparse.ArgumentParser(description="LearningAgent server")
    parser.add_argument(
        "--node-id",
        type=int,
        required=True,
        help="Local node id (must exist in hosts config).",
    )
    parser.add_argument(
        "--hosts-config",
        type=Path,
        default=Path("config/hosts.config"),
        help="Path to hosts config file.",
    )
    parser.add_argument(
        "--key-dir",
        type=Path,
        default=None,
        help="Directory for ECDSA key material. If omitted, built-in default key is used.",
    )
    parser.add_argument(
        "--collect-wait-ms",
        type=int,
        default=1000,
        help="Wait time (ms) after reaching 2f+1 reports before consensus submit.",
    )
    parser.add_argument(
        "--share-timeout-ms",
        type=int,
        default=None,
        help="Timeout (ms) for ShareReport RPCs to peers.",
    )
    parser.add_argument(
        "--consensus-timeout-ms",
        type=int,
        default=None,
        help="Timeout (ms) for SubmitReportBatch RPCs.",
    )
    parser.add_argument(
        "--sharing",
        type=_parse_bool,
        default=False,
        help="Enable data sharing with peers: true or false (default false).",
    )
    parser.add_argument(
        "--signing",
        type=_parse_bool,
        default=True,
        help="Enable signing and verification: true or false (default true).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override port to bind the server (defaults to hosts config entry).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=("linear", "scheduled"),
        default="linear",
        help="linear: learned linear model; scheduled: timeout from a time-based schedule CSV.",
    )
    parser.add_argument(
        "--schedule-file",
        type=Path,
        default=None,
        help="Path to schedule CSV (time_s,timeout_ms). Required when --model-type=scheduled.",
    )
    args = parser.parse_args()

    serve(
        node_id=args.node_id,
        hosts_config=args.hosts_config,
        key_dir=args.key_dir,
        collect_wait_ms=args.collect_wait_ms,
        sharing=args.sharing,
        share_timeout_ms=args.share_timeout_ms,
        consensus_timeout_ms=args.consensus_timeout_ms,
        host=args.host,
        port=args.port,
        signing_enabled=args.signing,
        model_type=args.model_type,
        schedule_file=args.schedule_file,
    )


if __name__ == "__main__":
    main()
