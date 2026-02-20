import argparse
import base64
import csv
import random
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import grpc
from google.protobuf import empty_pb2
import torch

import agent_pb2
import agent_pb2_grpc
from aggregator import BaseAggregator, MedianAggregator, WeightedMedianAggregator
from models import (
    ActionDecoder,
    FeatureExtractor,
    LinearTorchModel,
    ProtocolReportFeatureExtractor,
    ProtocolTimeoutActionDecoder,
    ScheduledTimeoutModel,
    TensorModel,
    get_action_space_size,
    get_report_feature_dim,
    get_report_feature_field_names,
    timeout_to_action_index,
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


PROTOCOL_NAME_TO_ENUM = {
    "pbft": agent_pb2.PROTOCOL_PBFT,
    "two_pc_over_pbft": agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT,
    "sbft": agent_pb2.PROTOCOL_SBFT,
}


def _parse_protocol(value: str) -> int:
    normalized = value.strip().lower().replace("-", "_")
    if normalized == "2pc":
        normalized = "two_pc_over_pbft"
    protocol = PROTOCOL_NAME_TO_ENUM.get(normalized)
    if protocol is None:
        raise argparse.ArgumentTypeError(
            f"Invalid protocol '{value}'. Valid options: {', '.join(PROTOCOL_NAME_TO_ENUM.keys())}"
        )
    return protocol


def _extract_state(report: agent_pb2.ReportLocal, protocol: int) -> object:
    if protocol == agent_pb2.PROTOCOL_PBFT:
        if not report.HasField("pbft_state"):
            raise ValueError("ReportLocal missing pbft_state")
        return report.pbft_state
    if protocol == agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT:
        if not report.HasField("two_pc_over_pbft_state"):
            raise ValueError("ReportLocal missing two_pc_over_pbft_state")
        return report.two_pc_over_pbft_state
    if protocol == agent_pb2.PROTOCOL_SBFT:
        if not report.HasField("sbft_state"):
            raise ValueError("ReportLocal missing sbft_state")
        return report.sbft_state
    raise ValueError(f"Unsupported protocol: {protocol}")


def _set_state(report: agent_pb2.ReportLocal, protocol: int, state: object) -> None:
    if protocol == agent_pb2.PROTOCOL_PBFT:
        report.pbft_state.CopyFrom(state)
        return
    if protocol == agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT:
        report.two_pc_over_pbft_state.CopyFrom(state)
        return
    if protocol == agent_pb2.PROTOCOL_SBFT:
        report.sbft_state.CopyFrom(state)
        return
    raise ValueError(f"Unsupported protocol: {protocol}")


def _copy_state(state: object) -> object:
    cloned = state.__class__()
    cloned.CopyFrom(state)
    return cloned


def _timeout_from_status(protocol: int, timeout: agent_pb2.Timeout) -> object:
    if protocol == agent_pb2.PROTOCOL_PBFT:
        if not timeout.HasField("pbft"):
            raise ValueError("Timeout payload missing pbft")
        return timeout.pbft
    if protocol == agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT:
        if not timeout.HasField("two_pc_over_pbft"):
            raise ValueError("Timeout payload missing two_pc_over_pbft")
        return timeout.two_pc_over_pbft
    if protocol == agent_pb2.PROTOCOL_SBFT:
        if not timeout.HasField("sbft"):
            raise ValueError("Timeout payload missing sbft")
        return timeout.sbft
    raise ValueError(f"Unsupported protocol: {protocol}")


def _extract_reward_payload(protocol: int, reward: agent_pb2.Reward) -> Optional[Tuple[int, object, object]]:
    if protocol == agent_pb2.PROTOCOL_PBFT:
        if not reward.HasField("pbft"):
            return None
        payload = reward.pbft
        return payload.episode, payload.report, payload.timeout_used
    if protocol == agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT:
        if not reward.HasField("two_pc_over_pbft"):
            return None
        payload = reward.two_pc_over_pbft
        return payload.episode, payload.report, payload.timeout_used
    if protocol == agent_pb2.PROTOCOL_SBFT:
        if not reward.HasField("sbft"):
            return None
        payload = reward.sbft
        return payload.episode, payload.report, payload.timeout_used
    return None


def _report_total_transactions(report: object) -> int:
    return int(getattr(report, "total_transactions", 0))


@dataclass(frozen=True)
class HostEntry:
    node_id: int
    host: str
    agent_port: int
    consensus_port: int


@dataclass(frozen=True)
class AgentInjectionPhase:
    at_time_s: float
    dirty_report_replica_ids: Set[int]
    ignore_report_replica_ids: Set[int]


def _parse_required_int(parent: ET.Element, tag: str) -> int:
    node = parent.find(tag)
    if node is None or node.text is None:
        raise ValueError(f"Missing required tag <{tag}>")
    try:
        return int(node.text.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid integer in <{tag}>: {node.text!r}") from exc


def load_agent_injection_schedule(path: Path) -> List[AgentInjectionPhase]:
    root = ET.parse(path).getroot()
    phase_nodes = root.findall("./phases/phase")
    if not phase_nodes:
        raise ValueError(f"No <phase> entries found in {path}")

    has_at_time = any(node.find("atTime") is not None for node in phase_nodes)
    phases: List[AgentInjectionPhase] = []
    cumulative_start = 0.0

    for index, phase_node in enumerate(phase_nodes):
        if has_at_time:
            if phase_node.find("atTime") is not None:
                at_time_s = float(_parse_required_int(phase_node, "atTime"))
            else:
                at_time_s = float(_parse_required_int(phase_node, "time"))
        else:
            at_time_s = cumulative_start
            duration_s = float(_parse_required_int(phase_node, "time"))
            if duration_s < 0:
                raise ValueError("Legacy phase duration <time> must be >= 0")
            cumulative_start += duration_s

        if at_time_s < 0:
            raise ValueError("Phase trigger time must be >= 0")
        if index > 0 and at_time_s < phases[index - 1].at_time_s:
            raise ValueError("Phases must be non-decreasing by trigger time")

        dirty_ids: Set[int] = set()
        for node in phase_node.findall("./agent/dirtyReport/replicas/id"):
            if node.text is None:
                continue
            raw = node.text.strip()
            try:
                dirty_ids.add(int(raw))
            except ValueError as exc:
                raise ValueError(f"Invalid dirtyReport replica id: {raw!r}") from exc

        ignore_ids: Set[int] = set()
        for node in phase_node.findall("./agent/ignoreReport/replicas/id"):
            if node.text is None:
                continue
            raw = node.text.strip()
            try:
                ignore_ids.add(int(raw))
            except ValueError as exc:
                raise ValueError(f"Invalid ignoreReport replica id: {raw!r}") from exc

        phases.append(
            AgentInjectionPhase(
                at_time_s=at_time_s,
                dirty_report_replica_ids=dirty_ids,
                ignore_report_replica_ids=ignore_ids,
            )
        )

    if not has_at_time:
        print("[injection] no <atTime> found; using legacy duration semantics from <time>")

    return phases


def _compute_start_monotonic(start_unix_ms: Optional[int]) -> float:
    if start_unix_ms is None:
        return time.monotonic()
    return time.monotonic() + ((start_unix_ms / 1000.0) - time.time())


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
        protocol: int,
        node_id: int,
        peers: List[HostEntry],
        key_dir: Path,
        collect_wait_s: float,
        sharing: bool,
        model: TensorModel,
        feature_extractor: FeatureExtractor[object],
        action_decoder: ActionDecoder[agent_pb2.Timeout],
        aggregator: BaseAggregator,
        consensus_stub: Optional[agent_pb2_grpc.ConsensusStub],
        share_timeout_s: Optional[float],
        consensus_timeout_s: Optional[float],
        signing_enabled: bool,
        use_default_keys: bool,
        dataset_csv: Optional[Path],
        agent_injection_schedule: Optional[List[AgentInjectionPhase]],
        injection_start_mono: float,
    ):
        self._protocol = int(protocol)
        self.node_id = node_id
        self._collect_wait_s = collect_wait_s
        self._sharing = sharing
        self._model = model
        self._feature_extractor = feature_extractor
        self._action_decoder = action_decoder
        self._aggregator = aggregator
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
        self._rng = random.Random(node_id + 1)
        self._agent_injection_schedule = (
            list(agent_injection_schedule) if agent_injection_schedule else []
        )
        self._injection_start_mono = injection_start_mono
        self._active_injection_phase_index = -1
        self._current_episode = 0
        self._local_accepted_episode: Optional[int] = None
        self._reports_by_episode: Dict[int, Dict[int, agent_pb2.ReportLocal]] = {}
        self._timers_by_episode: Dict[int, threading.Timer] = {}
        self._submitted_episodes: Set[int] = set()
        self._finalized_episodes: Set[int] = set()
        self._received_episodes: Set[int] = set()
        self._pending_episodes: Set[int] = set()
        self._timeouts_by_episode: Dict[int, agent_pb2.Timeout] = {}
        self._state_by_episode: Dict[int, object] = {}
        self._phase_by_episode: Dict[int, int] = {}
        self._report_feature_field_names = get_report_feature_field_names(self._protocol)
        self._dataset_handle = None
        self._dataset_writer = None
        self._setup_dataset_csv(dataset_csv)

    def _setup_dataset_csv(self, dataset_csv: Optional[Path]) -> None:
        if dataset_csv is None:
            return

        dataset_csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not dataset_csv.exists() or dataset_csv.stat().st_size == 0
        self._dataset_handle = dataset_csv.open("a", newline="")
        self._dataset_writer = csv.writer(self._dataset_handle)
        if write_header:
            self._dataset_writer.writerow(
                [
                    "reward_episode",
                    "phase",
                    "report_node_id",
                    *self._report_feature_field_names,
                    "election_timeout_milliseconds_used",
                    "prepare_timeout_milliseconds_used",
                    "slow_path_timeout_milliseconds_used",
                    "batch_timeout_milliseconds_used",
                    "reward",
                    "matched_episode_features",
                ]
            )
            self._dataset_handle.flush()

    def _active_agent_injection_flags_locked(self) -> Tuple[bool, bool]:
        if not self._agent_injection_schedule:
            return False, False

        active_index = self._current_injection_phase_index_locked()
        elapsed_s = max(0.0, time.monotonic() - self._injection_start_mono)
        if active_index < 0:
            phase = None
        else:
            phase = self._agent_injection_schedule[active_index]

        if active_index != self._active_injection_phase_index:
            if phase is not None:
                dirty_active = self.node_id in phase.dirty_report_replica_ids
                ignore_active = self.node_id in phase.ignore_report_replica_ids
                print(
                    "[injection] active phase "
                    f"idx={active_index} at_time={phase.at_time_s:g}s "
                    f"(elapsed={elapsed_s:.3f}s, dirty_report={dirty_active}, "
                    f"ignore_report={ignore_active})"
                )
            else:
                print(
                    "[injection] no active phase yet "
                    f"(elapsed={elapsed_s:.3f}s)"
                )
            self._active_injection_phase_index = active_index

        if phase is None:
            return False, False

        return (
            self.node_id in phase.dirty_report_replica_ids,
            self.node_id in phase.ignore_report_replica_ids,
        )

    def _current_injection_phase_index_locked(self) -> int:
        if not self._agent_injection_schedule:
            return -1
        elapsed_s = max(0.0, time.monotonic() - self._injection_start_mono)
        active_index = -1
        for idx, phase in enumerate(self._agent_injection_schedule):
            if elapsed_s >= phase.at_time_s:
                active_index = idx
            else:
                break
        return active_index

    def _dirty_pbft_state(self, state: agent_pb2.PbftReport) -> agent_pb2.PbftReport:
        baseline_total_tx = max(int(state.total_transactions), 1)
        baseline_total_cons = max(int(state.total_consensus_instances), 1)
        dirty_report = agent_pb2.PbftReport()
        dirty_report.CopyFrom(state)
        dirty_report.total_transactions = self._rng.randint(0, baseline_total_tx * 3)
        dirty_report.total_consensus_instances = self._rng.randint(0, baseline_total_cons * 3)
        dirty_report.avg_consensus_latency_ms = float(
            max(0.0, float(state.avg_consensus_latency_ms) * self._rng.uniform(0.4, 2.0))
        )
        dirty_report.p95_consensus_latency_ms = float(
            max(
                dirty_report.avg_consensus_latency_ms,
                float(state.p95_consensus_latency_ms) * self._rng.uniform(0.4, 2.2),
            )
        )
        dirty_report.p99_consensus_latency_ms = float(
            max(
                dirty_report.p95_consensus_latency_ms,
                float(state.p99_consensus_latency_ms) * self._rng.uniform(0.4, 2.4),
            )
        )
        dirty_report.throughput_tps = float(
            max(0.0, float(state.throughput_tps) * self._rng.uniform(0.4, 1.8))
        )
        dirty_report.timeout_violation_rate = float(
            min(1.0, max(0.0, float(state.timeout_violation_rate) + self._rng.uniform(-0.1, 0.2)))
        )
        dirty_report.avg_batch_size = float(
            max(0.0, float(state.avg_batch_size) * self._rng.uniform(0.4, 1.6))
        )
        dirty_report.p95_batch_size = float(
            max(
                float(dirty_report.avg_batch_size),
                float(state.p95_batch_size) * self._rng.uniform(0.4, 1.8),
            )
        )
        dirty_report.leader_change_count = self._rng.randint(
            0, max(int(state.leader_change_count), 1) * 3
        )
        dirty_report.regency_change_count = self._rng.randint(
            0, max(int(state.regency_change_count), 1) * 3
        )
        dirty_report.phase_propose_avg_delay_ms = float(
            max(0.0, float(state.phase_propose_avg_delay_ms) * self._rng.uniform(0.4, 2.0))
        )
        dirty_report.phase_propose_p95_delay_ms = float(
            max(
                dirty_report.phase_propose_avg_delay_ms,
                float(state.phase_propose_p95_delay_ms) * self._rng.uniform(0.4, 2.2),
            )
        )
        dirty_report.phase_write_avg_delay_ms = float(
            max(0.0, float(state.phase_write_avg_delay_ms) * self._rng.uniform(0.4, 2.0))
        )
        dirty_report.phase_write_p95_delay_ms = float(
            max(
                dirty_report.phase_write_avg_delay_ms,
                float(state.phase_write_p95_delay_ms) * self._rng.uniform(0.4, 2.2),
            )
        )
        dirty_report.phase_accept_avg_delay_ms = float(
            max(0.0, float(state.phase_accept_avg_delay_ms) * self._rng.uniform(0.4, 2.0))
        )
        dirty_report.phase_accept_p95_delay_ms = float(
            max(
                dirty_report.phase_accept_avg_delay_ms,
                float(state.phase_accept_p95_delay_ms) * self._rng.uniform(0.4, 2.2),
            )
        )
        dirty_report.phase_post_decision_avg_delay_ms = float(
            max(
                0.0,
                float(state.phase_post_decision_avg_delay_ms) * self._rng.uniform(0.4, 2.0),
            )
        )
        dirty_report.phase_post_decision_p95_delay_ms = float(
            max(
                dirty_report.phase_post_decision_avg_delay_ms,
                float(state.phase_post_decision_p95_delay_ms) * self._rng.uniform(0.4, 2.2),
            )
        )
        return dirty_report

    def _dirty_sbft_state(self, state: agent_pb2.SbftReport) -> agent_pb2.SbftReport:
        baseline_total_tx = max(int(state.total_transactions), 1)
        baseline_total_cons = max(int(state.total_consensus_instances), 1)
        dirty_report = agent_pb2.SbftReport()
        dirty_report.CopyFrom(state)
        dirty_report.total_transactions = self._rng.randint(0, baseline_total_tx * 3)
        dirty_report.total_consensus_instances = self._rng.randint(0, baseline_total_cons * 3)
        dirty_report.avg_consensus_latency_ms = float(
            max(0.0, float(state.avg_consensus_latency_ms) * self._rng.uniform(0.4, 2.0))
        )
        dirty_report.p95_consensus_latency_ms = float(
            max(
                dirty_report.avg_consensus_latency_ms,
                float(state.p95_consensus_latency_ms) * self._rng.uniform(0.4, 2.2),
            )
        )
        dirty_report.p99_consensus_latency_ms = float(
            max(
                dirty_report.p95_consensus_latency_ms,
                float(state.p99_consensus_latency_ms) * self._rng.uniform(0.4, 2.4),
            )
        )
        dirty_report.throughput_tps = float(
            max(0.0, float(state.throughput_tps) * self._rng.uniform(0.4, 1.8))
        )
        dirty_report.timeout_violation_rate = float(
            min(1.0, max(0.0, float(state.timeout_violation_rate) + self._rng.uniform(-0.1, 0.2)))
        )
        dirty_report.avg_batch_size = float(
            max(0.0, float(state.avg_batch_size) * self._rng.uniform(0.4, 1.6))
        )
        dirty_report.p95_batch_size = float(
            max(
                float(dirty_report.avg_batch_size),
                float(state.p95_batch_size) * self._rng.uniform(0.4, 1.8),
            )
        )
        dirty_report.leader_change_count = self._rng.randint(
            0, max(int(state.leader_change_count), 1) * 3
        )
        dirty_report.regency_change_count = self._rng.randint(
            0, max(int(state.regency_change_count), 1) * 3
        )
        return dirty_report

    def _dirty_two_pc_over_pbft_state(
        self, state: agent_pb2.TwoPcOverPbftReport
    ) -> agent_pb2.TwoPcOverPbftReport:
        dirty = agent_pb2.TwoPcOverPbftReport()
        dirty.CopyFrom(state)
        dirty.pbft.CopyFrom(self._dirty_pbft_state(state.pbft))
        dirty.total_transactions = self._rng.randint(0, max(int(state.total_transactions), 1) * 3)
        dirty.single_shard_transactions = self._rng.randint(
            0, max(int(state.single_shard_transactions), 1) * 3
        )
        dirty.cross_shard_transactions = self._rng.randint(
            0, max(int(state.cross_shard_transactions), 1) * 3
        )
        dirty.prepare_ok_count = self._rng.randint(0, max(int(state.prepare_ok_count), 1) * 3)
        dirty.prepare_fail_count = self._rng.randint(0, max(int(state.prepare_fail_count), 1) * 3)
        dirty.commit_ok_count = self._rng.randint(0, max(int(state.commit_ok_count), 1) * 3)
        dirty.abort_count = self._rng.randint(0, max(int(state.abort_count), 1) * 3)
        dirty.in_doubt_count = self._rng.randint(0, max(int(state.in_doubt_count), 1) * 3)
        dirty.prepare_timeout_count = self._rng.randint(
            0, max(int(state.prepare_timeout_count), 1) * 3
        )
        dirty.commit_timeout_unknown_outcome_count = self._rng.randint(
            0, max(int(state.commit_timeout_unknown_outcome_count), 1) * 3
        )
        dirty.abort_best_effort_timeout_count = self._rng.randint(
            0, max(int(state.abort_best_effort_timeout_count), 1) * 3
        )
        dirty.lock_contention_count = self._rng.randint(
            0, max(int(state.lock_contention_count), 1) * 3
        )
        dirty.prepare_timeout_rate = float(
            min(1.0, max(0.0, float(state.prepare_timeout_rate) + self._rng.uniform(-0.1, 0.2)))
        )
        dirty.abort_rate = float(
            min(1.0, max(0.0, float(state.abort_rate) + self._rng.uniform(-0.1, 0.2)))
        )
        dirty.commit_rate = float(
            min(1.0, max(0.0, float(state.commit_rate) + self._rng.uniform(-0.1, 0.2)))
        )
        dirty.lock_contention_rate = float(
            min(
                1.0, max(0.0, float(state.lock_contention_rate) + self._rng.uniform(-0.1, 0.2))
            )
        )
        dirty.avg_end_to_end_latency_ms = float(
            max(0.0, float(state.avg_end_to_end_latency_ms) * self._rng.uniform(0.4, 2.0))
        )
        dirty.p95_end_to_end_latency_ms = float(
            max(
                dirty.avg_end_to_end_latency_ms,
                float(state.p95_end_to_end_latency_ms) * self._rng.uniform(0.4, 2.2),
            )
        )
        dirty.p99_end_to_end_latency_ms = float(
            max(
                dirty.p95_end_to_end_latency_ms,
                float(state.p99_end_to_end_latency_ms) * self._rng.uniform(0.4, 2.4),
            )
        )
        dirty.avg_prepare_latency_ms = float(
            max(0.0, float(state.avg_prepare_latency_ms) * self._rng.uniform(0.4, 2.0))
        )
        dirty.p95_prepare_latency_ms = float(
            max(
                dirty.avg_prepare_latency_ms,
                float(state.p95_prepare_latency_ms) * self._rng.uniform(0.4, 2.2),
            )
        )
        dirty.avg_commit_latency_ms = float(
            max(0.0, float(state.avg_commit_latency_ms) * self._rng.uniform(0.4, 2.0))
        )
        dirty.p95_commit_latency_ms = float(
            max(
                dirty.avg_commit_latency_ms,
                float(state.p95_commit_latency_ms) * self._rng.uniform(0.4, 2.2),
            )
        )
        dirty.avg_lock_wait_ms = float(
            max(0.0, float(state.avg_lock_wait_ms) * self._rng.uniform(0.4, 2.0))
        )
        dirty.p95_lock_wait_ms = float(
            max(
                dirty.avg_lock_wait_ms,
                float(state.p95_lock_wait_ms) * self._rng.uniform(0.4, 2.2),
            )
        )
        dirty.avg_lock_hold_ms = float(
            max(0.0, float(state.avg_lock_hold_ms) * self._rng.uniform(0.4, 2.0))
        )
        dirty.p95_lock_hold_ms = float(
            max(
                dirty.avg_lock_hold_ms,
                float(state.p95_lock_hold_ms) * self._rng.uniform(0.4, 2.2),
            )
        )
        dirty.reason_prepare_lock_busy_count = self._rng.randint(
            0, max(int(state.reason_prepare_lock_busy_count), 1) * 3
        )
        dirty.reason_prepare_validation_fail_count = self._rng.randint(
            0, max(int(state.reason_prepare_validation_fail_count), 1) * 3
        )
        dirty.reason_prepare_timeout_count = self._rng.randint(
            0, max(int(state.reason_prepare_timeout_count), 1) * 3
        )
        dirty.reason_prepare_other_count = self._rng.randint(
            0, max(int(state.reason_prepare_other_count), 1) * 3
        )
        dirty.reason_commit_timeout_unknown_outcome_count = self._rng.randint(
            0, max(int(state.reason_commit_timeout_unknown_outcome_count), 1) * 3
        )
        dirty.reason_abort_best_effort_timeout_count = self._rng.randint(
            0, max(int(state.reason_abort_best_effort_timeout_count), 1) * 3
        )
        dirty.reason_other_count = self._rng.randint(
            0, max(int(state.reason_other_count), 1) * 3
        )
        return dirty

    def _dirty_state(self, state: object) -> object:
        if self._protocol == agent_pb2.PROTOCOL_PBFT:
            return self._dirty_pbft_state(state)
        if self._protocol == agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT:
            return self._dirty_two_pc_over_pbft_state(state)
        if self._protocol == agent_pb2.PROTOCOL_SBFT:
            return self._dirty_sbft_state(state)
        raise ValueError(f"Unsupported protocol: {self._protocol}")

    def _log_reward_row_locked(
        self,
        phase: int,
        report_node_id: int,
        reward_episode: int,
        timeout_used: object,
        reward_report: object,
        matched_state: Optional[object],
    ) -> None:
        if self._dataset_writer is None or self._dataset_handle is None:
            return

        matched = matched_state is not None
        report = matched_state
        report_values = (
            [getattr(report, field_name) for field_name in self._report_feature_field_names]
            if report is not None
            else [""] * len(self._report_feature_field_names)
        )
        election_timeout_ms = getattr(timeout_used, "election_timeout_milliseconds", "")
        prepare_timeout_ms = getattr(timeout_used, "prepare_timeout_milliseconds", "")
        slow_path_timeout_ms = getattr(timeout_used, "slow_path_timeout_milliseconds", "")
        batch_timeout_ms = getattr(timeout_used, "batch_timeout_milliseconds", "")
        self._dataset_writer.writerow(
            [
                reward_episode,
                phase,
                report_node_id,
                *report_values,
                election_timeout_ms,
                prepare_timeout_ms,
                slow_path_timeout_ms,
                batch_timeout_ms,
                _report_total_transactions(reward_report),
                int(matched),
            ]
        )
        # Flush every row to keep memory usage minimal and avoid data loss.
        self._dataset_handle.flush()

    def SendReport(self, request: agent_pb2.ReportLocal, context) -> empty_pb2.Empty:
        if request.protocol != self._protocol:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Protocol mismatch in report.",
            )
        try:
            incoming_state = _extract_state(request, self._protocol)
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        local_report = agent_pb2.ReportLocal(
            node_id=self.node_id,
            episode=request.episode,
            protocol=self._protocol,
        )
        _set_state(local_report, self._protocol, incoming_state)

        signed_report: Optional[agent_pb2.ReportLocal] = None
        with self._lock:
            self._apply_reward_update_locked(request)
            dirty_active, _ = self._active_agent_injection_flags_locked()
            if dirty_active:
                state = _extract_state(local_report, self._protocol)
                _set_state(local_report, self._protocol, self._dirty_state(state))
                print(
                    "[injection] dirtyReport applied "
                    f"(node={self.node_id}, episode={local_report.episode})"
                )

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
                print(
                    f"[episode {local_report.episode}] accepted valid local report "
                    f"from node={self.node_id} (sharing=off)"
                )
                state = _extract_state(local_report, self._protocol)
                timeout = self._predict_timeout_locked(
                    local_report.episode, state
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
        if request.protocol != self._protocol:
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
        if request.protocol != self._protocol:
            return empty_pb2.Empty()

        print(f"[episode {request.episode}] consensus batch {len(request.reports)} reports")
        for report in request.reports:
            local = _extract_state(report, self._protocol)
            if self._protocol == agent_pb2.PROTOCOL_PBFT:
                print(
                    f"node={report.node_id} "
                    f"total_tx={local.total_transactions} "
                    f"throughput={local.throughput_tps:.6f} "
                    f"avg_latency_ms={local.avg_consensus_latency_ms:.6f} "
                    f"p95_latency_ms={local.p95_consensus_latency_ms:.6f} "
                    f"timeout_violation_rate={local.timeout_violation_rate:.6f}"
                )
            elif self._protocol == agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT:
                print(
                    f"node={report.node_id} "
                    f"total_tx={local.total_transactions} "
                    f"prepare_timeout_rate={local.prepare_timeout_rate:.6f} "
                    f"commit_rate={local.commit_rate:.6f} "
                    f"avg_e2e_latency_ms={local.avg_end_to_end_latency_ms:.6f}"
                )
            else:
                print(
                    f"node={report.node_id} "
                    f"total_tx={local.total_transactions} "
                    f"throughput={local.throughput_tps:.6f} "
                    f"avg_latency_ms={local.avg_consensus_latency_ms:.6f} "
                    f"p95_latency_ms={local.p95_consensus_latency_ms:.6f} "
                    f"timeout_violation_rate={local.timeout_violation_rate:.6f}"
                )

        aggregated_report = self._aggregator.aggregate(
            node_id=self.node_id, report_batch=request
        )
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

            timeout = self._predict_timeout_locked(request.episode, aggregated_report)
            self._timeouts_by_episode[request.episode] = timeout

        return empty_pb2.Empty()

    def GetTimeout(
        self, request: agent_pb2.TimeoutRequest, context
    ) -> agent_pb2.TimeoutStatus:
        if request.protocol != self._protocol:
            return agent_pb2.TimeoutStatus(
                episode=request.episode,
                status=agent_pb2.TimeoutStatus.Status.NOT_RECEIVED,
            )
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
            self._state_by_episode.clear()
            self._phase_by_episode.clear()
            self._current_episode = 0
            self._local_accepted_episode = None
        return empty_pb2.Empty()

    def _store_report_locked(self, report: agent_pb2.ReportLocal) -> bool:
        if report.episode in self._finalized_episodes:
            return False
        if report.episode in self._submitted_episodes:
            return False

        reports_for_episode = self._reports_by_episode.setdefault(report.episode, {})
        if report.node_id in reports_for_episode:
            return False
        reports_for_episode[report.node_id] = report
        reports_collected = len(reports_for_episode)

        print(
            f"[episode {report.episode}] accepted valid report "
            f"from node={report.node_id} ({reports_collected}/{self._threshold})"
        )

        if (
            reports_collected >= self._threshold
            and report.episode not in self._timers_by_episode
            and report.episode not in self._submitted_episodes
        ):
            timer = threading.Timer(
                self._collect_wait_s, self._finalize_episode, args=(report.episode,)
            )
            self._timers_by_episode[report.episode] = timer
            timer.daemon = True
            timer.start()
        return True

    def _finalize_episode(self, episode: int) -> None:
        with self._lock:
            if episode in self._finalized_episodes or episode in self._submitted_episodes:
                return
            reports_for_episode = self._reports_by_episode.get(episode, {})
            if not reports_for_episode:
                return
            snapshot = list(reports_for_episode.values())
            _, ignore_active = self._active_agent_injection_flags_locked()
            if ignore_active and len(snapshot) > self._threshold:
                snapshot = sorted(snapshot, key=lambda item: item.node_id)[: self._threshold]
                print(
                    "[injection] ignoreReport applied "
                    f"(node={self.node_id}, episode={episode}, kept={len(snapshot)})"
                )
            self._submitted_episodes.add(episode)
            self._reports_by_episode.pop(episode, None)
            self._timers_by_episode.pop(episode, None)

        if self._consensus_stub is None:
            print(f"[episode {episode}] consensus stub not configured")
            return

        batch = agent_pb2.ReportBatch(
            episode=episode,
            protocol=self._protocol,
            reports=snapshot,
        )
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

    def _apply_reward_update_locked(self, request: agent_pb2.ReportLocal) -> None:
        if not request.HasField("reward"):
            return

        payload = _extract_reward_payload(self._protocol, request.reward)
        if payload is None:
            return
        reward_episode, reward_report, timeout_used = payload

        matched_state = self._state_by_episode.get(reward_episode)
        phase = self._phase_by_episode.get(reward_episode, -1)
        self._log_reward_row_locked(
            phase=phase,
            report_node_id=request.node_id,
            reward_episode=reward_episode,
            timeout_used=timeout_used,
            reward_report=reward_report,
            matched_state=matched_state,
        )
        if matched_state is None:
            print(
                "[learning] skipped reward update "
                f"(episode={reward_episode}): no stored state for episode"
            )
            self._phase_by_episode.pop(reward_episode, None)
            return

        try:
            action_index = timeout_to_action_index(self._protocol, timeout_used)
        except ValueError as exc:
            print(f"[learning] skipped reward for episode {reward_episode}: {exc}")
            return

        total_transactions = _report_total_transactions(reward_report)
        engineered_reward = reward_engineering(total_transactions)
        features = self._feature_extractor.featurize(matched_state)
        action = torch.tensor(float(action_index), dtype=torch.float32)
        self._model.update_model(features, action, engineered_reward)
        # Reward for this episode has been consumed.
        self._state_by_episode.pop(reward_episode, None)
        self._phase_by_episode.pop(reward_episode, None)
        if self._protocol == agent_pb2.PROTOCOL_PBFT:
            reward_timeout = f"{timeout_used.election_timeout_milliseconds}"
        elif self._protocol == agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT:
            reward_timeout = (
                f"{timeout_used.election_timeout_milliseconds}/"
                f"{timeout_used.prepare_timeout_milliseconds}"
            )
        else:
            reward_timeout = (
                f"{timeout_used.election_timeout_milliseconds}/"
                f"{timeout_used.slow_path_timeout_milliseconds}/"
                f"{timeout_used.batch_timeout_milliseconds}"
            )
        print(
            "[learning] applied reward update "
            f"(episode={reward_episode}, timeout={reward_timeout}, "
            f"total_tx={total_transactions}, "
            f"reward={engineered_reward:.2f})"
        )

    def _predict_timeout_locked(
        self, episode: int, report: object
    ) -> agent_pb2.Timeout:
        cached_report = _copy_state(report)
        self._state_by_episode[episode] = cached_report
        self._phase_by_episode[episode] = self._current_injection_phase_index_locked()
        features = self._feature_extractor.featurize(report)
        action = self._model.predict_action(features)
        return self._action_decoder.decode(action)


def serve(
    protocol: int,
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
    aggregate_function: str,
    aggregate_weight: int,
    schedule_file: Optional[Path],
    dataset_csv: Optional[Path],
    failure_spec: Optional[Path],
    injection_start_unix_ms: Optional[int],
):
    use_explicit_bind = host is not None and port is not None
    if use_explicit_bind:
        if sharing:
            raise ValueError(
                "--sharing=true requires --hosts-config. "
                "Standalone --host/--port mode does not configure peer sharing."
            )
        bind_port = int(port)
        entry = HostEntry(
            node_id=node_id,
            host=host,
            agent_port=bind_port,
            consensus_port=bind_port,
        )
        peers = [entry]
    else:
        peers = load_hosts_config(hosts_config)
        entry = next((peer for peer in peers if peer.node_id == node_id), None)
        if entry is None:
            raise ValueError(f"Node id {node_id} not found in {hosts_config}")
        bind_port = entry.agent_port

    share_timeout_s = None if share_timeout_ms is None else share_timeout_ms / 1000.0
    consensus_timeout_s = (
        None if consensus_timeout_ms is None else consensus_timeout_ms / 1000.0
    )
    start_mono = _compute_start_monotonic(injection_start_unix_ms)

    use_default_keys = key_dir is None
    effective_key_dir = key_dir if key_dir is not None else Path(".")

    consensus_stub = None
    if sharing:
        consensus_channel = grpc.insecure_channel(
            f"{entry.host}:{entry.consensus_port}"
        )
        consensus_stub = agent_pb2_grpc.ConsensusStub(consensus_channel)

    if model_type == "linear":
        model = LinearTorchModel(
            action_dim=get_action_space_size(protocol),
            n_features=get_report_feature_dim(protocol),
            lr=0.05,
        )
    elif model_type == "scheduled":
        if schedule_file is None:
            raise ValueError("--schedule-file is required when --model-type=scheduled")
        model = ScheduledTimeoutModel.from_csv(
            protocol, schedule_file, start_monotonic=start_mono
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    feature_extractor = ProtocolReportFeatureExtractor(protocol)
    action_decoder = ProtocolTimeoutActionDecoder(protocol)
    if aggregate_function == "median":
        aggregator: BaseAggregator = MedianAggregator(protocol=protocol)
    elif aggregate_function == "weighted_median":
        aggregator = WeightedMedianAggregator(protocol=protocol, weights=aggregate_weight)
    else:
        raise ValueError(f"Unsupported aggregation function: {aggregate_function}")

    agent_injection_schedule = (
        load_agent_injection_schedule(failure_spec) if failure_spec is not None else None
    )
    if agent_injection_schedule:
        print(
            "[injection] loaded schedule "
            f"(phases={len(agent_injection_schedule)}, start_unix_ms={injection_start_unix_ms})"
        )
    servicer = LearningAgentServicer(
        protocol=protocol,
        node_id=node_id,
        peers=peers,
        key_dir=effective_key_dir,
        collect_wait_s=collect_wait_ms / 1000.0,
        sharing=sharing,
        model=model,
        feature_extractor=feature_extractor,
        action_decoder=action_decoder,
        aggregator=aggregator,
        consensus_stub=consensus_stub,
        share_timeout_s=share_timeout_s,
        consensus_timeout_s=consensus_timeout_s,
        signing_enabled=signing_enabled,
        use_default_keys=use_default_keys,
        dataset_csv=dataset_csv,
        agent_injection_schedule=agent_injection_schedule,
        injection_start_mono=start_mono,
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
        help=(
            "Local node id. Must exist in hosts config unless running in "
            "standalone mode with explicit --host and --port."
        ),
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
        help=(
            "Port to bind the server. If set together with --host, hosts-config "
            "loading is skipped."
        ),
    )
    parser.add_argument(
        "--protocol",
        type=_parse_protocol,
        default=agent_pb2.PROTOCOL_PBFT,
        help="Protocol for this agent: pbft, two_pc_over_pbft (alias: 2pc), or sbft.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=("linear", "scheduled"),
        default="linear",
        help="linear: learned linear model; scheduled: timeout from a time-based schedule CSV.",
    )

    parser.add_argument(
        "--function",
        "-f",
        type=str,
        choices=("median", "weighted_median"),
        default="median",
        help="Aggregation function to use (default: median)",
    )
    parser.add_argument(
        "--function-weight",
        type=int,
        default=3,
        help="Weight multiplier for local node when using weighted_median (default: 3).",
    )

    parser.add_argument(
        "--schedule-file",
        type=Path,
        default=None,
        help=(
            "Path to schedule CSV. PBFT format: time_s,election_timeout_ms. "
            "2PC-over-PBFT format: time_s,election_timeout_ms,prepare_timeout_ms. "
            "SBFT format: time_s,election_timeout_ms,slow_path_timeout_ms,batch_timeout_ms. "
            "Required when --model-type=scheduled."
        ),
    )
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV output path for reward-labeled samples. "
            "Appends one row per received reward and flushes every row."
        ),
    )
    parser.add_argument(
        "--failure-spec",
        type=Path,
        default=None,
        help=(
            "Optional failure spec XML path. "
            "If set, learning-agent injections use <agent><dirtyReport> and "
            "<agent><ignoreReport> schedules."
        ),
    )
    parser.add_argument(
        "--injection-start-unix-ms",
        type=int,
        default=None,
        help=(
            "Optional epoch start time in milliseconds for injection schedule "
            "and scheduled timeout model timeline. If omitted, both start when "
            "the server starts."
        ),
    )
    args = parser.parse_args()

    serve(
        protocol=args.protocol,
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
        aggregate_function=args.function,
        aggregate_weight=args.function_weight,
        schedule_file=args.schedule_file,
        dataset_csv=args.dataset_csv,
        failure_spec=args.failure_spec,
        injection_start_unix_ms=args.injection_start_unix_ms,
    )


if __name__ == "__main__":
    main()
