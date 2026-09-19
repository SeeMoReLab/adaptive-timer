import argparse
import base64
import builtins
import csv
import random
import re
import statistics
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent import futures
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import grpc
from google.protobuf import empty_pb2

import agent_pb2
import agent_pb2_grpc
from aggregator import BaseAggregator, MedianAggregator, WeightedMedianAggregator
from continuous_bandit import (
    DEFAULT_PROPORTIONAL_K,
    ContinuousThompsonSamplingModel,
    ContinuousHeuristicTimeoutModel,
    ContinuousTimeoutActionCodec,
    NormalizedLatencyFloor,
)
from latency_floor import (
    DEFAULT_LATENCY_FLOOR_K,
    DEFAULT_LATENCY_FLOOR_QUANTILE,
    LatencyFloor,
    LatencyFloorPolicy,
    latency_feature_indices,
)
from models import (
    ActionCodec,
    CMAB,
    DiscountedLinUCB,
    FeatureExtractor,
    LinearModel,
    PavlovTimeoutModel,
    StarvationFreeThompsonSamplingModel,
    SurvivalGatedThompsonSamplingModel,
    ThompsonSamplingModel,
    ThompsonSamplingModelV2,
    ProtocolReportFeatureExtractor,
    ProtocolTimeoutActionCodec,
    RandomTimeoutModel,
    ScheduledTimeoutActionCodec,
    ScheduledTimeoutModel,
    TensorModel,
    get_action_space_size,
    get_report_feature_dim,
    get_report_feature_field_names,
    get_timeout_values_ms,
    report_to_feature_values,
)
from reward_utils import (
    DEFAULT_REWARD_DESIGN,
    REWARD_DESIGNS,
    calculate_autobahn_reward,
    calculate_hotstuff_reward,
    calculate_pbft_reward,
    calculate_sbft_reward,
    calculate_smartbft_reward,
    calculate_tendermint_reward,
    set_reward_design,
)

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec


_ORIGINAL_PRINT = builtins.print


def _timestamped_print(*args, **kwargs) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if args:
        first, *rest = args
        _ORIGINAL_PRINT(f"[{timestamp}] {first}", *rest, **kwargs)
        return
    _ORIGINAL_PRINT(f"[{timestamp}]", **kwargs)


def _install_timestamped_print() -> None:
    builtins.print = _timestamped_print


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in ("true", "1", "yes", "y", "on"):
        return True
    if lowered in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _tendermint_applied_timeouts(timeout_used: object) -> dict:
    """The applied Tendermint timeout triple, as reward calculator kwargs.

    Empty when no Timeout was supplied, which leaves the calculator's defaults
    in place: reward_tps does not need them and reward_mixed says so itself.
    """
    if timeout_used is None:
        return {}
    return {
        "propose_timeout_ms": int(timeout_used.propose_timeout_milliseconds),
        "prevote_timeout_ms": int(timeout_used.prevote_timeout_milliseconds),
        "precommit_timeout_ms": int(timeout_used.precommit_timeout_milliseconds),
    }


def reward_engineering(
    protocol: int, report: object, timeout_used: object = None
) -> float:
    """Calculate a protocol-specific reward from a complete reward report.

    ``timeout_used`` is the Timeout message the episode actually ran with. It
    is only read for protocols whose report does not carry its own timeout
    field - Tendermint - and is ignored everywhere else.
    """
    if protocol in (agent_pb2.PROTOCOL_PBFT, agent_pb2.PROTOCOL_SMARTBFT):
        # SmartbftReport has the PbftReport field layout but its own reward.
        calculator = (
            calculate_pbft_reward
            if protocol == agent_pb2.PROTOCOL_PBFT
            else calculate_smartbft_reward
        )
        return calculator(
            total_transactions=int(report.total_transactions),
            total_consensus_instances=int(report.total_consensus_instances),
            avg_consensus_latency_ms=float(report.avg_consensus_latency_ms),
            p50_consensus_latency_ms=float(report.p50_consensus_latency_ms),
            p95_consensus_latency_ms=float(report.p95_consensus_latency_ms),
            throughput_tps=float(report.throughput_tps),
            avg_batch_size=float(report.avg_batch_size),
            p95_batch_size=float(report.p95_batch_size),
            leader_change_count=int(report.leader_change_count),
            regency_change_count=int(report.regency_change_count),
            timeout_ms=int(report.timeout_ms),
            avg_inter_commit_gap_ms=float(report.avg_inter_commit_gap_ms),
            p50_inter_commit_gap_ms=float(report.p50_inter_commit_gap_ms),
            p95_inter_commit_gap_ms=float(report.p95_inter_commit_gap_ms),
            view_change_count=int(report.view_change_count),
            no_progress_view_change_count=int(
                report.no_progress_view_change_count
            ),
        )
    if protocol == agent_pb2.PROTOCOL_SBFT:
        return calculate_sbft_reward(
            total_transactions=int(report.total_transactions),
            total_consensus_instances=int(report.total_consensus_instances),
            avg_consensus_latency_ms=float(report.avg_consensus_latency_ms),
            p95_consensus_latency_ms=float(report.p95_consensus_latency_ms),
            p99_consensus_latency_ms=float(report.p99_consensus_latency_ms),
            throughput_tps=float(report.throughput_tps),
            timeout_violation_rate=float(report.timeout_violation_rate),
            avg_batch_size=float(report.avg_batch_size),
            p95_batch_size=float(report.p95_batch_size),
            leader_change_count=int(report.leader_change_count),
            regency_change_count=int(report.regency_change_count),
        )
    if protocol == agent_pb2.PROTOCOL_TENDERMINT:
        return calculate_tendermint_reward(
            total_transactions=int(report.total_transactions),
            total_consensus_instances=int(report.total_consensus_instances),
            avg_consensus_latency_ms=float(report.avg_consensus_latency_ms),
            p50_consensus_latency_ms=float(report.p50_consensus_latency_ms),
            p90_consensus_latency_ms=float(report.p90_consensus_latency_ms),
            throughput_tps=float(report.throughput_tps),
            timeout_violation_rate=float(report.timeout_violation_rate),
            avg_batch_size=float(report.avg_batch_size),
            p50_batch_size=float(report.p50_batch_size),
            p90_batch_size=float(report.p90_batch_size),
            leader_change_count=int(report.leader_change_count),
            regency_change_count=int(report.regency_change_count),
            propose_latency_ms=float(report.propose_latency_ms),
            prevote_latency_ms=float(report.prevote_latency_ms),
            precommit_latency_ms=float(report.precommit_latency_ms),
            **_tendermint_applied_timeouts(timeout_used),
        )
    if protocol == agent_pb2.PROTOCOL_HOTSTUFF:
        return calculate_hotstuff_reward(
            total_transactions=int(report.total_transactions),
            total_consensus_instances=int(report.total_consensus_instances),
            avg_consensus_latency_ms=float(report.avg_consensus_latency_ms),
            p50_consensus_latency_ms=float(report.p50_consensus_latency_ms),
            p95_consensus_latency_ms=float(report.p95_consensus_latency_ms),
            throughput_tps=float(report.throughput_tps),
            avg_batch_size=float(report.avg_batch_size),
            p95_batch_size=float(report.p95_batch_size),
            leader_change_count=int(report.leader_change_count),
            regency_change_count=int(report.regency_change_count),
            timeout_ms=int(report.timeout_ms),
            avg_inter_commit_gap_ms=float(report.avg_inter_commit_gap_ms),
            p50_inter_commit_gap_ms=float(report.p50_inter_commit_gap_ms),
            p95_inter_commit_gap_ms=float(report.p95_inter_commit_gap_ms),
            view_change_count=int(report.view_change_count),
            no_progress_view_change_count=int(
                report.no_progress_view_change_count
            ),
        )
    if protocol == agent_pb2.PROTOCOL_AUTOBAHN:
        return calculate_autobahn_reward(
            total_transactions=int(report.total_transactions),
            total_consensus_instances=int(report.total_consensus_instances),
            avg_consensus_latency_ms=float(report.avg_consensus_latency_ms),
            p50_consensus_latency_ms=float(report.p50_consensus_latency_ms),
            p95_consensus_latency_ms=float(report.p95_consensus_latency_ms),
            throughput_tps=float(report.throughput_tps),
            avg_batch_size=float(report.avg_batch_size),
            p95_batch_size=float(report.p95_batch_size),
            leader_change_count=int(report.leader_change_count),
            regency_change_count=int(report.regency_change_count),
            timeout_ms=int(report.timeout_ms),
            avg_inter_commit_gap_ms=float(report.avg_inter_commit_gap_ms),
            p50_inter_commit_gap_ms=float(report.p50_inter_commit_gap_ms),
            p95_inter_commit_gap_ms=float(report.p95_inter_commit_gap_ms),
            view_change_count=int(report.view_change_count),
            no_progress_view_change_count=int(
                report.no_progress_view_change_count
            ),
        )
    raise ValueError(f"Unsupported protocol: {protocol}")


PROTOCOL_NAME_TO_ENUM = {
    "pbft": agent_pb2.PROTOCOL_PBFT,
    "sbft": agent_pb2.PROTOCOL_SBFT,
    "tendermint": agent_pb2.PROTOCOL_TENDERMINT,
    "hotstuff": agent_pb2.PROTOCOL_HOTSTUFF,
    "autobahn": agent_pb2.PROTOCOL_AUTOBAHN,
    "smartbft": agent_pb2.PROTOCOL_SMARTBFT,
}

# Protocols whose report windows are bounded by wall-clock time rather than by
# consensus ticks: window_consensus_count == 0 marks a wall-clock window.
WALL_CLOCK_PROTOCOLS = (
    agent_pb2.PROTOCOL_PBFT,
    agent_pb2.PROTOCOL_HOTSTUFF,
    agent_pb2.PROTOCOL_AUTOBAHN,
    agent_pb2.PROTOCOL_SMARTBFT,
)


def _parse_protocol(value: str) -> int:
    normalized = value.strip().lower().replace("-", "_")
    protocol = PROTOCOL_NAME_TO_ENUM.get(normalized)
    if protocol is None:
        raise argparse.ArgumentTypeError(
            f"Invalid protocol '{value}'. Valid options: {', '.join(PROTOCOL_NAME_TO_ENUM.keys())}"
        )
    return protocol


def _protocols_compatible(incoming_protocol: int, expected_protocol: int) -> bool:
    return incoming_protocol in (agent_pb2.PROTOCOL_UNSPECIFIED, expected_protocol)


def _dataset_timeout_field_names(protocol: int) -> Tuple[str, ...]:
    if protocol in (agent_pb2.PROTOCOL_PBFT, agent_pb2.PROTOCOL_SMARTBFT):
        return ("election_timeout_milliseconds_used",)
    if protocol == agent_pb2.PROTOCOL_SBFT:
        return (
            "election_timeout_milliseconds_used",
            "slow_path_timeout_milliseconds_used",
            "batch_timeout_milliseconds_used",
        )
    if protocol == agent_pb2.PROTOCOL_TENDERMINT:
        return (
            "propose_timeout_milliseconds_used",
            "prevote_timeout_milliseconds_used",
            "precommit_timeout_milliseconds_used",
        )
    if protocol == agent_pb2.PROTOCOL_HOTSTUFF:
        return ("timeout_delay_milliseconds_used",)
    if protocol == agent_pb2.PROTOCOL_AUTOBAHN:
        return (
            "timeout_delay_milliseconds_used",
            "car_timeout_milliseconds_used",
            "fast_path_timeout_milliseconds_used",
        )
    raise ValueError(f"Unsupported protocol: {protocol}")


def _dataset_timeout_values(protocol: int, timeout_used: object) -> Tuple[object, ...]:
    if protocol in (agent_pb2.PROTOCOL_PBFT, agent_pb2.PROTOCOL_SMARTBFT):
        return (getattr(timeout_used, "election_timeout_milliseconds", ""),)
    if protocol == agent_pb2.PROTOCOL_SBFT:
        return (
            getattr(timeout_used, "election_timeout_milliseconds", ""),
            getattr(timeout_used, "slow_path_timeout_milliseconds", ""),
            getattr(timeout_used, "batch_timeout_milliseconds", ""),
        )
    if protocol == agent_pb2.PROTOCOL_TENDERMINT:
        return (
            getattr(timeout_used, "propose_timeout_milliseconds", ""),
            getattr(timeout_used, "prevote_timeout_milliseconds", ""),
            getattr(timeout_used, "precommit_timeout_milliseconds", ""),
        )
    if protocol == agent_pb2.PROTOCOL_HOTSTUFF:
        return (getattr(timeout_used, "timeout_delay_milliseconds", ""),)
    if protocol == agent_pb2.PROTOCOL_AUTOBAHN:
        return (
            getattr(timeout_used, "timeout_delay_milliseconds", ""),
            getattr(timeout_used, "car_timeout_milliseconds", ""),
            getattr(timeout_used, "fast_path_timeout_milliseconds", ""),
        )
    raise ValueError(f"Unsupported protocol: {protocol}")


def _extract_state(report: agent_pb2.ReportLocal, protocol: int) -> object:
    if protocol == agent_pb2.PROTOCOL_PBFT:
        if not report.HasField("pbft_state"):
            raise ValueError("ReportLocal missing pbft_state")
        return report.pbft_state
    if protocol == agent_pb2.PROTOCOL_SBFT:
        if not report.HasField("sbft_state"):
            raise ValueError("ReportLocal missing sbft_state")
        return report.sbft_state
    if protocol == agent_pb2.PROTOCOL_TENDERMINT:
        if not report.HasField("tendermint_state"):
            raise ValueError("ReportLocal missing tendermint_state")
        return report.tendermint_state
    if protocol == agent_pb2.PROTOCOL_HOTSTUFF:
        if not report.HasField("hotstuff_state"):
            raise ValueError("ReportLocal missing hotstuff_state")
        return report.hotstuff_state
    if protocol == agent_pb2.PROTOCOL_AUTOBAHN:
        if not report.HasField("autobahn_state"):
            raise ValueError("ReportLocal missing autobahn_state")
        return report.autobahn_state
    if protocol == agent_pb2.PROTOCOL_SMARTBFT:
        if not report.HasField("smartbft_state"):
            raise ValueError("ReportLocal missing smartbft_state")
        return report.smartbft_state
    raise ValueError(f"Unsupported protocol: {protocol}")


def _set_state(report: agent_pb2.ReportLocal, protocol: int, state: object) -> None:
    if protocol == agent_pb2.PROTOCOL_PBFT:
        report.pbft_state.CopyFrom(state)
        return
    if protocol == agent_pb2.PROTOCOL_SBFT:
        report.sbft_state.CopyFrom(state)
        return
    if protocol == agent_pb2.PROTOCOL_TENDERMINT:
        report.tendermint_state.CopyFrom(state)
        return
    if protocol == agent_pb2.PROTOCOL_HOTSTUFF:
        report.hotstuff_state.CopyFrom(state)
        return
    if protocol == agent_pb2.PROTOCOL_AUTOBAHN:
        report.autobahn_state.CopyFrom(state)
        return
    if protocol == agent_pb2.PROTOCOL_SMARTBFT:
        report.smartbft_state.CopyFrom(state)
        return
    raise ValueError(f"Unsupported protocol: {protocol}")


def _copy_state(state: object) -> object:
    cloned = state.__class__()
    cloned.CopyFrom(state)
    return cloned


def _extract_reward_payload(protocol: int, reward: agent_pb2.Reward) -> Optional[Tuple[int, object, object]]:
    if protocol == agent_pb2.PROTOCOL_PBFT:
        if not reward.HasField("pbft"):
            return None
        payload = reward.pbft
        return payload.episode, payload.report, payload.timeout_used
    if protocol == agent_pb2.PROTOCOL_SBFT:
        if not reward.HasField("sbft"):
            return None
        payload = reward.sbft
        return payload.episode, payload.report, payload.timeout_used
    if protocol == agent_pb2.PROTOCOL_TENDERMINT:
        if not reward.HasField("tendermint"):
            return None
        payload = reward.tendermint
        return payload.episode, payload.report, payload.timeout_used
    if protocol == agent_pb2.PROTOCOL_HOTSTUFF:
        if not reward.HasField("hotstuff"):
            return None
        payload = reward.hotstuff
        return payload.episode, payload.report, payload.timeout_used
    if protocol == agent_pb2.PROTOCOL_AUTOBAHN:
        if not reward.HasField("autobahn"):
            return None
        payload = reward.autobahn
        return payload.episode, payload.report, payload.timeout_used
    if protocol == agent_pb2.PROTOCOL_SMARTBFT:
        if not reward.HasField("smartbft"):
            return None
        payload = reward.smartbft
        return payload.episode, payload.report, payload.timeout_used
    return None


def _report_total_transactions(report: object) -> int:
    return int(getattr(report, "total_transactions", 0))

def _report_throughput_tps(report: object) -> float:
    return float(getattr(report, "throughput_tps", 0.0))


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


@dataclass(frozen=True)
class TimeoutDecision:
    timeout: agent_pb2.Timeout
    start_tick: int
    report_seq: int
    window_consensus_count: int


def _reward_timeouts_agree(timeouts: List[object]) -> bool:
    if not timeouts:
        return False
    first = timeouts[0].SerializeToString(deterministic=True)
    return all(
        timeout.SerializeToString(deterministic=True) == first for timeout in timeouts[1:]
    )


def _parse_required_int(parent: ET.Element, tag: str) -> int:
    node = parent.find(tag)
    if node is None or node.text is None:
        raise ValueError(f"Missing required tag <{tag}>")
    try:
        return int(node.text.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid integer in <{tag}>: {node.text!r}") from exc


def _parse_optional_float(parent: ET.Element, tag: str) -> Optional[float]:
    node = parent.find(tag)
    if node is None or node.text is None:
        return None
    try:
        return float(node.text.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid float in <{tag}>: {node.text!r}") from exc


def _parse_optional_int(parent: ET.Element, tag: str) -> Optional[int]:
    node = parent.find(tag)
    if node is None or node.text is None:
        return None
    try:
        return int(node.text.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid integer in <{tag}>: {node.text!r}") from exc


def _parse_warmup_seconds(root: ET.Element) -> float:
    warmup_ms = _parse_optional_int(root, "warmUpTimeMs")
    if warmup_ms is not None:
        return max(0.0, warmup_ms / 1000.0)

    warmup_s = _parse_optional_float(root, "warmUpTime")
    if warmup_s is None:
        return 0.0
    return max(0.0, warmup_s)


def load_agent_injection_schedule(path: Path) -> Tuple[float, List[AgentInjectionPhase]]:
    root = ET.parse(path).getroot()
    warmup_s = _parse_warmup_seconds(root)
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

    return warmup_s, phases


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
        action_codec: ActionCodec[agent_pb2.Timeout],
        aggregator: BaseAggregator,
        consensus_stub: Optional[agent_pb2_grpc.ConsensusStub],
        share_timeout_s: Optional[float],
        consensus_timeout_s: Optional[float],
        signing_enabled: bool,
        use_default_keys: bool,
        dataset_csv: Optional[Path],
        agent_injection_schedule: Optional[List[AgentInjectionPhase]],
        injection_start_mono: float,
        injection_warmup_s: float,
    ):
        self._protocol = int(protocol)
        self.node_id = node_id
        self._collect_wait_s = collect_wait_s
        self._sharing = sharing
        self._model = model
        self._feature_extractor = feature_extractor
        self._action_codec = action_codec
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
        self._injection_warmup_s = max(0.0, float(injection_warmup_s))
        self._active_injection_phase_index = -1
        self._current_episode = 0
        self._sealed_through_episode = 0
        # Reports are bucketed by episode alone. Bucketing by report window
        # would partition a quorum across nodes whose report lengths differ,
        # so no bucket would ever reach 2f+1; the window is instead reconciled
        # by taking the median across the collected reports.
        self._quorum_episodes: Set[int] = set()
        self._reports_by_episode: Dict[int, Dict[int, agent_pb2.ReportLocal]] = {}
        self._timers_by_episode: Dict[int, threading.Timer] = {}
        self._submitted_episodes: Set[int] = set()
        self._finalized_episodes: Set[int] = set()
        self._received_episodes: Set[int] = set()
        self._pending_episodes: Set[int] = set()
        self._timeouts_by_episode: Dict[int, TimeoutDecision] = {}
        self._state_by_episode: Dict[int, object] = {}
        self._phase_by_episode: Dict[int, int] = {}
        self._pending_rewards_by_episode: Dict[int, Tuple[int, object, object]] = {}
        self._report_feature_field_names = get_report_feature_field_names(self._protocol)
        self._dataset_timeout_field_names = _dataset_timeout_field_names(self._protocol)
        self._dataset_handle = None
        self._dataset_writer = None
        self._setup_dataset_csv(dataset_csv)

    def _setup_dataset_csv(self, dataset_csv: Optional[Path]) -> None:
        if dataset_csv is None:
            return

        dataset_csv.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "episode",
            "phase",
            "report_node_id",
            *(
                f"feature_{field_name}"
                for field_name in self._report_feature_field_names
            ),
            *(
                f"reward_{field_name}"
                for field_name in self._report_feature_field_names
            ),
            *self._dataset_timeout_field_names,
        ]
        write_header = not dataset_csv.exists() or dataset_csv.stat().st_size == 0
        if not write_header:
            with dataset_csv.open("r", newline="") as existing_handle:
                existing_header = next(csv.reader(existing_handle), [])
            if existing_header != header:
                raise ValueError(
                    f"Dataset CSV schema mismatch in {dataset_csv}. "
                    "Remove the existing file before using the new episode-row schema."
                )

        self._dataset_handle = dataset_csv.open("a", newline="")
        self._dataset_writer = csv.writer(self._dataset_handle)
        if write_header:
            self._dataset_writer.writerow(header)
            self._dataset_handle.flush()

    def _is_episode_sealed_locked(self, episode: int) -> bool:
        if episode <= self._sealed_through_episode:
            return True
        if episode in self._submitted_episodes:
            return True
        if episode in self._finalized_episodes:
            return True
        return episode in self._quorum_episodes

    def _derive_window_from_reports(
        self, reports: List[agent_pb2.ReportLocal]
    ) -> Tuple[int, int, int]:
        # Derive a robust consensus window from participant reports.
        report_seq = int(
            statistics.median_low([int(report.report_seq) for report in reports])
        )
        start_tick = int(
            statistics.median_low([int(report.start_tick) for report in reports])
        )
        window_consensus_count = int(
            statistics.median_low(
                [int(report.window_consensus_count) for report in reports]
            )
        )
        is_wall_clock = (
            self._protocol in WALL_CLOCK_PROTOCOLS
            and window_consensus_count == 0
        )
        if report_seq < start_tick or (
            report_seq == start_tick and not is_wall_clock
        ):
            raise ValueError(
                f"Invalid consensus report window: start={start_tick}, seq={report_seq}"
            )
        return start_tick, report_seq, window_consensus_count

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
        elapsed_injection_s = elapsed_s - self._injection_warmup_s
        if elapsed_injection_s < 0.0:
            return -1
        active_index = -1
        for idx, phase in enumerate(self._agent_injection_schedule):
            if elapsed_injection_s >= phase.at_time_s:
                active_index = idx
            else:
                break
        return active_index

    def _dirty_pbft_state(self, state: object) -> object:
        """Dirty a PbftReport or SmartbftReport; both share the same layout."""
        baseline_total_tx = max(int(state.total_transactions), 1)
        baseline_total_cons = max(int(state.total_consensus_instances), 1)
        dirty_report = state.__class__()
        dirty_report.CopyFrom(state)
        dirty_report.total_transactions = self._rng.randint(0, baseline_total_tx * 3)
        dirty_report.total_consensus_instances = self._rng.randint(0, baseline_total_cons * 3)
        dirty_report.avg_consensus_latency_ms = float(
            max(0.0, float(state.avg_consensus_latency_ms) * self._rng.uniform(0.4, 2.0))
        )
        dirty_report.p50_consensus_latency_ms = float(
            max(
                dirty_report.avg_consensus_latency_ms,
                float(state.p50_consensus_latency_ms) * self._rng.uniform(0.4, 2.1),
            )
        )
        dirty_report.p95_consensus_latency_ms = float(
            max(
                dirty_report.p50_consensus_latency_ms,
                float(state.p95_consensus_latency_ms) * self._rng.uniform(0.4, 2.2),
            )
        )
        dirty_report.throughput_tps = float(
            max(0.0, float(state.throughput_tps) * self._rng.uniform(0.4, 1.8))
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
        dirty_report.timeout_ms = max(0, int(round(float(state.timeout_ms) * self._rng.uniform(0.8, 1.2))))
        dirty_report.avg_inter_commit_gap_ms = float(
            max(0.0, float(state.avg_inter_commit_gap_ms) * self._rng.uniform(0.4, 2.0))
        )
        dirty_report.p50_inter_commit_gap_ms = float(
            max(
                0.0,
                float(state.p50_inter_commit_gap_ms) * self._rng.uniform(0.4, 2.1),
            )
        )
        dirty_report.p95_inter_commit_gap_ms = float(
            max(
                dirty_report.p50_inter_commit_gap_ms,
                float(state.p95_inter_commit_gap_ms) * self._rng.uniform(0.4, 2.2),
            )
        )
        dirty_report.view_change_count = self._rng.randint(
            0, max(int(state.view_change_count), 1) * 3
        )
        dirty_report.no_progress_view_change_count = self._rng.randint(
            0, max(int(state.no_progress_view_change_count), 1) * 3
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

    def _dirty_tendermint_state(
        self, state: agent_pb2.TendermintReport
    ) -> agent_pb2.TendermintReport:
        baseline_total_tx = max(int(state.total_transactions), 1)
        baseline_total_cons = max(int(state.total_consensus_instances), 1)
        dirty_report = agent_pb2.TendermintReport()
        dirty_report.CopyFrom(state)
        dirty_report.total_transactions = self._rng.randint(0, baseline_total_tx * 3)
        dirty_report.total_consensus_instances = self._rng.randint(0, baseline_total_cons * 3)
        dirty_report.avg_consensus_latency_ms = float(
            max(0.0, float(state.avg_consensus_latency_ms) * self._rng.uniform(0.4, 2.0))
        )
        dirty_report.p50_consensus_latency_ms = float(
            max(
                dirty_report.avg_consensus_latency_ms,
                float(state.p50_consensus_latency_ms) * self._rng.uniform(0.4, 2.2),
            )
        )
        dirty_report.p90_consensus_latency_ms = float(
            max(
                dirty_report.p50_consensus_latency_ms,
                float(state.p90_consensus_latency_ms) * self._rng.uniform(0.4, 2.4),
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
        dirty_report.p50_batch_size = float(
            max(
                float(dirty_report.avg_batch_size),
                float(state.p50_batch_size) * self._rng.uniform(0.4, 1.8),
            )
        )
        dirty_report.p90_batch_size = float(
            max(
                float(dirty_report.p50_batch_size),
                float(state.p90_batch_size) * self._rng.uniform(0.4, 2.0),
            )
        )
        dirty_report.leader_change_count = self._rng.randint(
            0, max(int(state.leader_change_count), 1) * 3
        )
        dirty_report.regency_change_count = self._rng.randint(
            0, max(int(state.regency_change_count), 1) * 3
        )
        dirty_report.propose_latency_ms = float(
            max(0.0, float(state.propose_latency_ms) * self._rng.uniform(0.4, 2.0))
        )
        dirty_report.prevote_latency_ms = float(
            max(0.0, float(state.prevote_latency_ms) * self._rng.uniform(0.4, 2.0))
        )
        dirty_report.precommit_latency_ms = float(
            max(0.0, float(state.precommit_latency_ms) * self._rng.uniform(0.4, 2.0))
        )
        return dirty_report

    def _dirty_hotstuff_state(
        self, state: agent_pb2.HotstuffReport
    ) -> agent_pb2.HotstuffReport:
        baseline_total_tx = max(int(state.total_transactions), 1)
        baseline_total_cons = max(int(state.total_consensus_instances), 1)
        dirty_report = agent_pb2.HotstuffReport()
        dirty_report.CopyFrom(state)
        dirty_report.total_transactions = self._rng.randint(0, baseline_total_tx * 3)
        dirty_report.total_consensus_instances = self._rng.randint(0, baseline_total_cons * 3)
        dirty_report.avg_consensus_latency_ms = float(
            max(0.0, float(state.avg_consensus_latency_ms) * self._rng.uniform(0.4, 2.0))
        )
        dirty_report.p50_consensus_latency_ms = float(
            max(
                dirty_report.avg_consensus_latency_ms,
                float(state.p50_consensus_latency_ms) * self._rng.uniform(0.4, 2.1),
            )
        )
        dirty_report.p95_consensus_latency_ms = float(
            max(
                dirty_report.p50_consensus_latency_ms,
                float(state.p95_consensus_latency_ms) * self._rng.uniform(0.4, 2.2),
            )
        )
        dirty_report.throughput_tps = float(
            max(0.0, float(state.throughput_tps) * self._rng.uniform(0.4, 1.8))
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
        dirty_report.timeout_ms = max(0, int(round(float(state.timeout_ms) * self._rng.uniform(0.8, 1.2))))
        dirty_report.avg_inter_commit_gap_ms = float(
            max(0.0, float(state.avg_inter_commit_gap_ms) * self._rng.uniform(0.4, 2.0))
        )
        dirty_report.p50_inter_commit_gap_ms = float(
            max(
                0.0,
                float(state.p50_inter_commit_gap_ms) * self._rng.uniform(0.4, 2.1),
            )
        )
        dirty_report.p95_inter_commit_gap_ms = float(
            max(
                dirty_report.p50_inter_commit_gap_ms,
                float(state.p95_inter_commit_gap_ms) * self._rng.uniform(0.4, 2.2),
            )
        )
        dirty_report.view_change_count = self._rng.randint(
            0, max(int(state.view_change_count), 1) * 3
        )
        dirty_report.no_progress_view_change_count = self._rng.randint(
            0, max(int(state.no_progress_view_change_count), 1) * 3
        )
        return dirty_report

    def _dirty_autobahn_state(
        self, state: agent_pb2.AutobahnReport
    ) -> agent_pb2.AutobahnReport:
        baseline_total_tx = max(int(state.total_transactions), 1)
        baseline_total_cons = max(int(state.total_consensus_instances), 1)
        dirty_report = agent_pb2.AutobahnReport()
        dirty_report.CopyFrom(state)
        dirty_report.total_transactions = self._rng.randint(0, baseline_total_tx * 3)
        dirty_report.total_consensus_instances = self._rng.randint(0, baseline_total_cons * 3)
        dirty_report.avg_consensus_latency_ms = float(
            max(0.0, float(state.avg_consensus_latency_ms) * self._rng.uniform(0.4, 2.0))
        )
        dirty_report.p50_consensus_latency_ms = float(
            max(
                dirty_report.avg_consensus_latency_ms,
                float(state.p50_consensus_latency_ms) * self._rng.uniform(0.4, 2.1),
            )
        )
        dirty_report.p95_consensus_latency_ms = float(
            max(
                dirty_report.p50_consensus_latency_ms,
                float(state.p95_consensus_latency_ms) * self._rng.uniform(0.4, 2.2),
            )
        )
        dirty_report.throughput_tps = float(
            max(0.0, float(state.throughput_tps) * self._rng.uniform(0.4, 1.8))
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
        dirty_report.timeout_ms = max(0, int(round(float(state.timeout_ms) * self._rng.uniform(0.8, 1.2))))
        dirty_report.avg_inter_commit_gap_ms = float(
            max(0.0, float(state.avg_inter_commit_gap_ms) * self._rng.uniform(0.4, 2.0))
        )
        dirty_report.p50_inter_commit_gap_ms = float(
            max(
                0.0,
                float(state.p50_inter_commit_gap_ms) * self._rng.uniform(0.4, 2.1),
            )
        )
        dirty_report.p95_inter_commit_gap_ms = float(
            max(
                dirty_report.p50_inter_commit_gap_ms,
                float(state.p95_inter_commit_gap_ms) * self._rng.uniform(0.4, 2.2),
            )
        )
        dirty_report.view_change_count = self._rng.randint(
            0, max(int(state.view_change_count), 1) * 3
        )
        dirty_report.no_progress_view_change_count = self._rng.randint(
            0, max(int(state.no_progress_view_change_count), 1) * 3
        )
        return dirty_report

    def _dirty_state(self, state: object) -> object:
        if self._protocol in (agent_pb2.PROTOCOL_PBFT, agent_pb2.PROTOCOL_SMARTBFT):
            return self._dirty_pbft_state(state)
        if self._protocol == agent_pb2.PROTOCOL_SBFT:
            return self._dirty_sbft_state(state)
        if self._protocol == agent_pb2.PROTOCOL_TENDERMINT:
            return self._dirty_tendermint_state(state)
        if self._protocol == agent_pb2.PROTOCOL_HOTSTUFF:
            return self._dirty_hotstuff_state(state)
        if self._protocol == agent_pb2.PROTOCOL_AUTOBAHN:
            return self._dirty_autobahn_state(state)
        raise ValueError(f"Unsupported protocol: {self._protocol}")

    def _log_completed_episode_row_locked(
        self,
        phase: int,
        report_node_id: int,
        episode: int,
        timeout_used: object,
        reward_report: object,
        feature_report: object,
    ) -> None:
        if self._dataset_writer is None or self._dataset_handle is None:
            return

        feature_values = report_to_feature_values(self._protocol, feature_report)
        reward_values = report_to_feature_values(self._protocol, reward_report)
        timeout_values = _dataset_timeout_values(self._protocol, timeout_used)
        self._dataset_writer.writerow(
            [
                episode,
                phase,
                report_node_id,
                *feature_values,
                *reward_values,
                *timeout_values,
            ]
        )
        self._dataset_handle.flush()

    def SendReport(self, request: agent_pb2.ReportLocal, context) -> empty_pb2.Empty:
        if not _protocols_compatible(request.protocol, self._protocol):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Protocol mismatch in report: "
                f"expected={self._protocol}, got={request.protocol}",
            )
        is_wall_clock = (
            self._protocol in WALL_CLOCK_PROTOCOLS
            and int(request.window_consensus_count) == 0
        )
        if int(request.report_seq) < int(request.start_tick) or (
            int(request.report_seq) == int(request.start_tick)
            and not is_wall_clock
        ):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Invalid report window: report_seq must be greater than start_tick "
                "unless this is a wall-clock window.",
            )
        try:
            incoming_state = _extract_state(request, self._protocol)
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        local_report = agent_pb2.ReportLocal(
            node_id=request.node_id,
            episode=request.episode,
            protocol=self._protocol,
            start_tick=request.start_tick,
            report_seq=request.report_seq,
            window_consensus_count=request.window_consensus_count,
        )
        _set_state(local_report, self._protocol, incoming_state)
        if self._sharing and request.HasField("reward"):
            # In sharing mode the reward travels with the report so that it is
            # aggregated by the same consensus batch that carries the features.
            local_report.reward.CopyFrom(request.reward)

        signed_report: Optional[agent_pb2.ReportLocal] = None
        with self._lock:
            if not self._sharing:
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
            if self._is_episode_sealed_locked(local_report.episode):
                context.abort(
                    grpc.StatusCode.ALREADY_EXISTS,
                    "Episode is already sealed.",
                )
            if local_report.episode > self._current_episode:
                self._current_episode = local_report.episode
            self._received_episodes.add(local_report.episode)

            if not self._sharing:
                print(
                    f"[episode {local_report.episode}] accepted valid local report "
                    f"from node={local_report.node_id} (sharing=off)"
                )
                state = _extract_state(local_report, self._protocol)
                timeout = self._predict_timeout_locked(
                    local_report.episode,
                    state,
                    int(local_report.start_tick),
                    int(local_report.report_seq),
                    int(local_report.window_consensus_count),
                )
                self._timeouts_by_episode[local_report.episode] = TimeoutDecision(
                    timeout=timeout,
                    start_tick=int(local_report.start_tick),
                    report_seq=int(local_report.report_seq),
                    window_consensus_count=int(local_report.window_consensus_count),
                )
            else:
                self._pending_episodes.add(local_report.episode)
                signed_report = self._sign_report(local_report)
                self._store_report_locked(signed_report)

        # Broadcast regardless of whether the report was stored locally. A
        # follow-up report is a no-op here but is exactly what heals a peer that
        # missed this node's first report for the episode.
        if signed_report is not None:
            self._broadcast_signed_report_async(signed_report)

        return empty_pb2.Empty()

    def ShareReport(self, request: agent_pb2.ReportLocal, context) -> empty_pb2.Empty:
        if not self._sharing:
            return empty_pb2.Empty()
        if not _protocols_compatible(request.protocol, self._protocol):
            return empty_pb2.Empty()
        is_wall_clock = (
            self._protocol in WALL_CLOCK_PROTOCOLS
            and int(request.window_consensus_count) == 0
        )
        if int(request.report_seq) < int(request.start_tick) or (
            int(request.report_seq) == int(request.start_tick)
            and not is_wall_clock
        ):
            return empty_pb2.Empty()
        if not self._verify_report(request):
            return empty_pb2.Empty()

        stored = agent_pb2.ReportLocal()
        stored.CopyFrom(request)

        with self._lock:
            if stored.episode < self._current_episode:
                return empty_pb2.Empty()
            if self._is_episode_sealed_locked(stored.episode):
                return empty_pb2.Empty()
            self._store_report_locked(stored)

        return empty_pb2.Empty()

    def DeliverConsensus(
        self, request: agent_pb2.ReportBatch, context
    ) -> empty_pb2.Empty:
        if not request.reports:
            return empty_pb2.Empty()
        if not _protocols_compatible(request.protocol, self._protocol):
            return empty_pb2.Empty()

        print(f"[episode {request.episode}] consensus batch {len(request.reports)} reports")
        for report in request.reports:
            local = _extract_state(report, self._protocol)
            if self._protocol in (agent_pb2.PROTOCOL_PBFT, agent_pb2.PROTOCOL_SMARTBFT):
                print(
                    f"node={report.node_id} "
                    f"total_tx={local.total_transactions} "
                    f"throughput={local.throughput_tps:.6f} "
                    f"avg_latency_ms={local.avg_consensus_latency_ms:.6f} "
                    f"p50_latency_ms={local.p50_consensus_latency_ms:.6f} "
                    f"p95_latency_ms={local.p95_consensus_latency_ms:.6f} "
                    f"timeout_ms={local.timeout_ms} "
                    f"view_changes={local.view_change_count}"
                )
            elif self._protocol == agent_pb2.PROTOCOL_SBFT:
                print(
                    f"node={report.node_id} "
                    f"total_tx={local.total_transactions} "
                    f"throughput={local.throughput_tps:.6f} "
                    f"avg_latency_ms={local.avg_consensus_latency_ms:.6f} "
                    f"p95_latency_ms={local.p95_consensus_latency_ms:.6f} "
                    f"timeout_violation_rate={local.timeout_violation_rate:.6f}"
                )
            elif self._protocol in (
                agent_pb2.PROTOCOL_HOTSTUFF,
                agent_pb2.PROTOCOL_AUTOBAHN,
            ):
                print(
                    f"node={report.node_id} "
                    f"total_tx={local.total_transactions} "
                    f"throughput={local.throughput_tps:.6f} "
                    f"avg_latency_ms={local.avg_consensus_latency_ms:.6f} "
                    f"p50_latency_ms={local.p50_consensus_latency_ms:.6f} "
                    f"p95_latency_ms={local.p95_consensus_latency_ms:.6f} "
                    f"timeout_ms={local.timeout_ms} "
                    f"view_changes={local.view_change_count}"
                )
            else:
                print(
                    f"node={report.node_id} "
                    f"total_tx={local.total_transactions} "
                    f"throughput={local.throughput_tps:.6f} "
                    f"avg_latency_ms={local.avg_consensus_latency_ms:.6f} "
                    f"p50_latency_ms={local.p50_consensus_latency_ms:.6f} "
                    f"p90_latency_ms={local.p90_consensus_latency_ms:.6f} "
                    f"avg_batch_size={local.avg_batch_size:.6f} "
                    f"p50_batch_size={local.p50_batch_size:.6f} "
                    f"p90_batch_size={local.p90_batch_size:.6f} "
                    f"propose_latency_ms={local.propose_latency_ms:.6f} "
                    f"prevote_latency_ms={local.prevote_latency_ms:.6f} "
                    f"precommit_latency_ms={local.precommit_latency_ms:.6f}"
                )

        aggregated_report = self._aggregator.aggregate(
            node_id=self.node_id, report_batch=request
        )
        try:
            start_tick, report_seq, window_consensus_count = (
                self._derive_window_from_reports(list(request.reports))
            )
        except ValueError:
            return empty_pb2.Empty()
        with self._lock:
            if request.episode in self._finalized_episodes:
                return empty_pb2.Empty()
            if request.episode > self._current_episode:
                self._current_episode = request.episode

            timer = self._timers_by_episode.pop(request.episode, None)
            if timer is not None:
                timer.cancel()
            self._reports_by_episode.pop(request.episode, None)
            self._quorum_episodes.discard(int(request.episode))
            self._submitted_episodes.add(request.episode)
            self._finalized_episodes.add(request.episode)
            self._sealed_through_episode = max(
                self._sealed_through_episode, int(request.episode)
            )
            self._pending_episodes.discard(request.episode)
            self._received_episodes.add(request.episode)
            self._prune_stale_episodes_locked()

            # Apply the agreed reward before predicting: both the features and
            # the reward for this batch come from consensus, so every node feeds
            # its model exactly the same experience.
            self._apply_aggregated_reward_locked(request)

            timeout = self._predict_timeout_locked(
                request.episode,
                aggregated_report,
                start_tick,
                report_seq,
                window_consensus_count,
            )
            self._timeouts_by_episode[request.episode] = TimeoutDecision(
                timeout=timeout,
                start_tick=start_tick,
                report_seq=report_seq,
                window_consensus_count=window_consensus_count,
            )

        return empty_pb2.Empty()

    def GetTimeout(
        self, request: agent_pb2.TimeoutRequest, context
    ) -> agent_pb2.TimeoutStatus:
        if not _protocols_compatible(request.protocol, self._protocol):
            return agent_pb2.TimeoutStatus(
                episode=request.episode,
                status=agent_pb2.TimeoutStatus.Status.NOT_RECEIVED,
            )
        with self._lock:
            # Answer with the newest decision at or after the requested episode.
            # When consensus has moved past a lagging node, that is what lets the
            # node fast-forward instead of polling an episode it will never get.
            episode = request.episode
            decision = self._timeouts_by_episode.get(episode)
            if decision is None:
                later = [
                    candidate
                    for candidate in self._timeouts_by_episode
                    if candidate > request.episode
                ]
                if later:
                    episode = max(later)
                    decision = self._timeouts_by_episode[episode]
            if decision is not None:
                if episode != request.episode:
                    print(
                        f"[episode {request.episode}] answering with newer decision "
                        f"for episode {episode}"
                    )
                return agent_pb2.TimeoutStatus(
                    episode=episode,
                    status=agent_pb2.TimeoutStatus.Status.READY,
                    timeout=decision.timeout,
                    start_tick=decision.start_tick,
                    report_seq=decision.report_seq,
                    window_consensus_count=decision.window_consensus_count,
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
            self._quorum_episodes.clear()
            self._submitted_episodes.clear()
            self._finalized_episodes.clear()
            self._received_episodes.clear()
            self._pending_episodes.clear()
            self._timeouts_by_episode.clear()
            self._state_by_episode.clear()
            self._phase_by_episode.clear()
            self._pending_rewards_by_episode.clear()
            self._current_episode = 0
            self._sealed_through_episode = 0
        return empty_pb2.Empty()

    def _store_report_locked(self, report: agent_pb2.ReportLocal) -> bool:
        if self._is_episode_sealed_locked(report.episode):
            return False

        reports_for_episode = self._reports_by_episode.setdefault(report.episode, {})
        existing = reports_for_episode.get(report.node_id)
        if existing is not None:
            # Keep the first report a node sent for an episode. Follow-up reports
            # only exist because that node is still waiting for a recommendation,
            # and the first reports across nodes are the most time-aligned.
            return False

        reports_for_episode[report.node_id] = report
        reports_collected = len(reports_for_episode)

        print(
            f"[episode {report.episode} seq={report.report_seq} "
            f"window_consensus_count={report.window_consensus_count}] "
            f"accepted valid report from node={report.node_id} "
            f"({reports_collected}/{self._threshold})"
        )

        if (
            reports_collected >= self._threshold
            and report.episode not in self._timers_by_episode
            and report.episode not in self._submitted_episodes
        ):
            self._quorum_episodes.add(int(report.episode))
            self._sealed_through_episode = max(
                self._sealed_through_episode, int(report.episode)
            )
            timer = threading.Timer(
                self._collect_wait_s,
                self._finalize_episode,
                args=(int(report.episode),),
            )
            self._timers_by_episode[report.episode] = timer
            timer.daemon = True
            timer.start()
        return True

    def _finalize_episode(self, episode: int) -> None:
        with self._lock:
            if episode in self._finalized_episodes or episode in self._submitted_episodes:
                return
            if episode not in self._quorum_episodes:
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
                    f"(node={self.node_id}, episode={episode}, "
                    f"kept={len(snapshot)})"
                )
            self._submitted_episodes.add(episode)
            self._reports_by_episode.pop(episode, None)
            self._quorum_episodes.discard(episode)
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

        reward_report_copy = reward_report.__class__()
        reward_report_copy.CopyFrom(reward_report)
        timeout_used_copy = timeout_used.__class__()
        timeout_used_copy.CopyFrom(timeout_used)

        self._apply_reward_payload_locked(
            report_node_id=int(request.node_id),
            reward_episode=int(reward_episode),
            reward_report=reward_report_copy,
            timeout_used=timeout_used_copy,
        )

    def _prune_stale_episodes_locked(self) -> None:
        """Drop bookkeeping for episodes that can no longer be finalized.

        An episode that never reaches quorum leaves a partial report bucket
        behind, and an episode that is never ordered leaves a reward that can
        never be matched to features.
        """
        threshold = int(self._sealed_through_episode)
        for episode in [ep for ep in self._reports_by_episode if ep < threshold]:
            self._reports_by_episode.pop(episode, None)
            self._pending_episodes.discard(episode)
        for episode in [ep for ep in self._pending_rewards_by_episode if ep < threshold]:
            self._pending_rewards_by_episode.pop(episode, None)
            print(
                f"[learning] dropped unmatched reward for episode {episode}: "
                "consensus never produced features for it"
            )

    def _apply_aggregated_reward_locked(self, report_batch: agent_pb2.ReportBatch) -> None:
        """Apply one reward, agreed by consensus, to this node's model.

        The reward for episode e-1 is piggybacked on the reports for episode e,
        so the delivered batch carries one reward per contributing node. Taking
        the median over them makes the reward Byzantine-robust and identical on
        every node, instead of each node training on its own local measurement.
        """
        payloads: List[Tuple[int, int, object, object]] = []
        for report in report_batch.reports:
            if not report.HasField("reward"):
                continue
            payload = _extract_reward_payload(self._protocol, report.reward)
            if payload is None:
                continue
            reward_episode, reward_report, timeout_used = payload
            payloads.append(
                (int(report.node_id), int(reward_episode), reward_report, timeout_used)
            )
        if not payloads:
            return

        episode_counts = Counter(episode for _, episode, _, _ in payloads)
        reward_episode, present = episode_counts.most_common(1)[0]
        required = (self._threshold + 1) // 2  # f+1
        if present < required:
            print(
                f"[learning] skipped reward update for episode {reward_episode}: "
                f"only {present} of {len(report_batch.reports)} reports carry it, need {required}"
            )
            return

        group = [payload for payload in payloads if payload[1] == reward_episode]
        timeouts = [payload[3] for payload in group]
        if not _reward_timeouts_agree(timeouts):
            # Aggregating rewards measured under different timeouts would label
            # the training row with an action no node actually ran.
            print(
                f"[learning] skipped reward update for episode {reward_episode}: "
                f"contributing nodes report different timeouts "
                f"({[node_id for node_id, _, _, _ in group]})"
            )
            return

        synthetic = agent_pb2.ReportBatch(
            episode=reward_episode, protocol=self._protocol
        )
        for node_id, _, reward_report, _ in group:
            entry = synthetic.reports.add()
            entry.node_id = node_id
            entry.episode = reward_episode
            entry.protocol = self._protocol
            _set_state(entry, self._protocol, reward_report)

        aggregated_reward = self._aggregator.aggregate(
            node_id=self.node_id, report_batch=synthetic
        )
        self._apply_reward_payload_locked(
            report_node_id=self.node_id,
            reward_episode=reward_episode,
            reward_report=aggregated_reward,
            timeout_used=timeouts[0],
        )

    def _apply_reward_payload_locked(
        self,
        report_node_id: int,
        reward_episode: int,
        reward_report: object,
        timeout_used: object,
    ) -> None:
        feature_report = self._state_by_episode.get(reward_episode)
        if feature_report is None:
            self._pending_rewards_by_episode[reward_episode] = (
                report_node_id,
                reward_report,
                timeout_used,
            )
            print(
                "[learning] deferred reward update "
                f"(episode={reward_episode}): waiting for consensus state"
            )
            return

        try:
            action = self._action_codec.encode(timeout_used)
        except ValueError as exc:
            print(f"[learning] skipped reward for episode {reward_episode}: {exc}")
            self._pending_rewards_by_episode.pop(reward_episode, None)
            return

        total_transactions = _report_total_transactions(reward_report)
        throughput_tps = _report_throughput_tps(reward_report)
        engineered_reward = reward_engineering(
            self._protocol, reward_report, timeout_used
        )
        phase = self._phase_by_episode.get(reward_episode, -1)
        self._log_completed_episode_row_locked(
            phase=phase,
            report_node_id=report_node_id,
            episode=reward_episode,
            timeout_used=timeout_used,
            reward_report=reward_report,
            feature_report=feature_report,
        )
        if self._protocol in (agent_pb2.PROTOCOL_PBFT, agent_pb2.PROTOCOL_SMARTBFT):
            reward_timeout = f"{timeout_used.election_timeout_milliseconds}"
        elif self._protocol == agent_pb2.PROTOCOL_SBFT:
            reward_timeout = (
                f"{timeout_used.election_timeout_milliseconds}/"
                f"{timeout_used.slow_path_timeout_milliseconds}/"
                f"{timeout_used.batch_timeout_milliseconds}"
            )
        elif self._protocol == agent_pb2.PROTOCOL_HOTSTUFF:
            reward_timeout = f"{timeout_used.timeout_delay_milliseconds}"
        elif self._protocol == agent_pb2.PROTOCOL_AUTOBAHN:
            reward_timeout = (
                f"{timeout_used.timeout_delay_milliseconds}/"
                f"{timeout_used.car_timeout_milliseconds}/"
                f"{timeout_used.fast_path_timeout_milliseconds}"
            )
        else:
            reward_timeout = (
                f"{timeout_used.propose_timeout_milliseconds}/"
                f"{timeout_used.prevote_timeout_milliseconds}/"
                f"{timeout_used.precommit_timeout_milliseconds}"
            )

        if phase == -1:
            print(
                "[learning] skipped model update during warmup "
                f"(episode={reward_episode}, timeout={reward_timeout}, "
                f"throughput_tps={throughput_tps:.6f}, "
                f"total_tx={total_transactions}, "
                f"reward={engineered_reward:.2f})"
            )
        else:
            features = self._feature_extractor.featurize(feature_report)
            self._model.update_model(features, action, engineered_reward)
            print(
                "[learning] applied reward update "
                f"(episode={reward_episode}, timeout={reward_timeout}, "
                f"throughput_tps={throughput_tps:.6f}, "
                f"total_tx={total_transactions}, "
                f"reward={engineered_reward:.2f})"
            )

        # Reward for this episode has been consumed.
        self._state_by_episode.pop(reward_episode, None)
        self._phase_by_episode.pop(reward_episode, None)
        self._pending_rewards_by_episode.pop(reward_episode, None)

    def _dataset_phase_for_episode_locked(self) -> int:
        elapsed_s = max(0.0, time.monotonic() - self._injection_start_mono)
        if elapsed_s < self._injection_warmup_s:
            return -1
        if isinstance(self._model, ScheduledTimeoutModel) and self._model.is_warmup_active():
            return -1
        return self._current_injection_phase_index_locked()

    def _predict_timeout_locked(
        self,
        episode: int,
        report: object,
        start_tick: int,
        report_seq: int,
        window_consensus_count: int,
    ) -> agent_pb2.Timeout:
        is_wall_clock = (
            self._protocol in WALL_CLOCK_PROTOCOLS
            and window_consensus_count == 0
        )
        if report_seq < start_tick or (
            report_seq == start_tick and not is_wall_clock
        ):
            raise ValueError(
                f"Invalid report window for timeout prediction: start={start_tick}, seq={report_seq}"
            )
        cached_report = _copy_state(report)
        self._state_by_episode[episode] = cached_report
        self._phase_by_episode[episode] = self._dataset_phase_for_episode_locked()

        features = self._feature_extractor.featurize(report)
        action = self._model.predict_action(features)
        timeout = self._action_codec.decode(action)

        pending_reward = self._pending_rewards_by_episode.get(episode)
        if pending_reward is not None:
            rn_id, reward_report, timeout_used = pending_reward
            self._apply_reward_payload_locked(
                report_node_id=rn_id,
                reward_episode=episode,
                reward_report=reward_report,
                timeout_used=timeout_used,
            )

        return timeout


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
    heuristic_multiplier: float,
    latency_floor_k: float,
    latency_floor_quantile: float,
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

    # Domain constraint on the action space, shared by whichever model is
    # built below: a timeout under the consensus latency the replicas are
    # already achieving cannot be met, so it is not worth a decision. The
    # field is the window mean, avg_consensus_latency_ms, matching what
    # continuous_heuristic reads below, so the constraint and the heuristic
    # are compared on the same measurement.
    latency_index, instances_index = latency_feature_indices(
        get_report_feature_field_names(protocol)
    )
    latency_floor = (
        None
        if latency_floor_k <= 0.0
        else LatencyFloor(
            multiplier=latency_floor_k, quantile=latency_floor_quantile
        )
    )

    if model_type == "continuous_heuristic":
        action_codec = ContinuousTimeoutActionCodec(protocol)
        feature_fields = get_report_feature_field_names(protocol)
        arms_ms = get_timeout_values_ms(protocol)
        model = ContinuousHeuristicTimeoutModel(
            action_codec=action_codec,
            n_features=get_report_feature_dim(protocol),
            latency_index=feature_fields.index("avg_consensus_latency_ms"),
            transactions_index=feature_fields.index("total_transactions"),
            multiplier=heuristic_multiplier,
            clamp_ms=(min(arms_ms), max(arms_ms)),
            initial_timeout_ms=sorted(arms_ms)[len(arms_ms) // 2],
        )
    elif model_type == "continuous":
        action_codec = ContinuousTimeoutActionCodec(protocol)
        discrete_action_count = get_action_space_size(protocol)
        model = ContinuousThompsonSamplingModel(
            action_dim=action_codec.action_dim,
            n_features=get_report_feature_dim(protocol),
            anchor_actions=action_codec.anchor_actions(),
            initial_design_size=discrete_action_count,
            memory_size=discrete_action_count * 32,
            seed=42,
            latency_floor=(
                None
                if latency_floor is None
                else NormalizedLatencyFloor(
                    action_codec=action_codec,
                    latency_index=latency_index,
                    n_features=get_report_feature_dim(protocol),
                    instances_index=instances_index,
                    latency_floor=latency_floor,
                )
            ),
        )
    elif model_type == "random":
        # Seeded like the other models so that, in sharing mode, every node
        # draws the same action from the same agreed features. Reward
        # aggregation is only meaningful when the fleet explores in lockstep.
        model = RandomTimeoutModel(
            action_dim=get_action_space_size(protocol),
            seed=42,
        )
    elif model_type == "linear":
        model = LinearModel(
            action_dim=get_action_space_size(protocol),
            n_features=get_report_feature_dim(protocol),
            lr=0.05,
        )
    elif model_type == "pavlov":
        # Win-stay/lose-shift baseline: no context and no learning, so it takes
        # neither n_features nor a seed. protocol only fixes the arm ordering.
        model = PavlovTimeoutModel(
            action_dim=get_action_space_size(protocol),
            protocol=protocol,
        )
    elif model_type in ("cmab", "thompson"):
        model = ThompsonSamplingModel(
            action_dim=get_action_space_size(protocol),
            n_features=get_report_feature_dim(protocol),
            seed=42,
        )
    elif model_type == "cmab_v2":
        # Earlier CMAB iteration, previously simulator-only. Kept reachable
        # from the agent so a phase-change response can be compared against
        # the current cmab on hardware rather than only on resampled data.
        model = ThompsonSamplingModelV2(
            action_dim=get_action_space_size(protocol),
            n_features=get_report_feature_dim(protocol),
            seed=42,
        )
    elif model_type == "cmab_reset":
        model = CMAB(
            action_dim=get_action_space_size(protocol),
            n_features=get_report_feature_dim(protocol),
            seed=42,
        )
    elif model_type == "cmab_v4":
        model = SurvivalGatedThompsonSamplingModel(
            action_dim=get_action_space_size(protocol),
            n_features=get_report_feature_dim(protocol),
            seed=42,
        )
        print("[cmab_v4] survival-gated: liveness gate + reward ranking")
    elif model_type == "claude_v2":
        # bucket_size 32 rather than the class default of 8. The exploit branch
        # is only contextual if its trees can split, and the inherited
        # min_samples_leaf=6 needs two leaves of 6, so a window of 8 can never
        # split: fitting the run's own rows gives a single leaf on 40 of 40
        # draws at 8, versus 2.0 leaves at 16 and 3.5 at 32. At 8 the model
        # falls back to a bootstrap mean of the window, which is the flat,
        # context-free estimate it was written to replace.
        model = StarvationFreeThompsonSamplingModel(
            action_dim=get_action_space_size(protocol),
            n_features=get_report_feature_dim(protocol),
            bucket_size=32,
            seed=42,
        )
        print("[claude_v2] bucket_size=32 (class default 8)")
    elif model_type == "codex_v1":
        model = DiscountedLinUCB(
            action_dim=get_action_space_size(protocol),
            n_features=get_report_feature_dim(protocol),
            seed=42,
        )
    elif model_type == "scheduled":
        if schedule_file is None:
            raise ValueError("--schedule-file is required when --model-type=scheduled")
        model = ScheduledTimeoutModel.from_csv(
            protocol,
            schedule_file,
            start_monotonic=start_mono,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # continuous takes the floor through its own candidate sampler, above.
    # continuous_heuristic is already a multiple of the same latency, and
    # scheduled is a fixed-baseline replay that must not be overridden.
    #
    # In sharing mode the floor stays identical on every replica for free: it
    # is estimated from the consensus-agreed feature vector, which is the same
    # on all of them, so constraining the action space does not break the
    # lockstep the aggregated reward depends on.
    floor_exempt = ("continuous_heuristic", "scheduled")
    if latency_floor is not None and model_type not in (
        ("continuous",) + floor_exempt
    ):
        model = LatencyFloorPolicy(
            model=model,
            timeouts_ms=get_timeout_values_ms(protocol),
            latency_index=latency_index,
            n_features=get_report_feature_dim(protocol),
            instances_index=instances_index,
            latency_floor=latency_floor,
        )
    if latency_floor is not None:
        applied = "not applied (latency rule already)" if model_type in floor_exempt else "applied"
        print(
            f"[latency_floor] k={latency_floor_k} "
            f"quantile={latency_floor_quantile} {applied} to {model_type}"
        )

    feature_extractor = ProtocolReportFeatureExtractor(protocol)
    if model_type == "scheduled":
        action_codec = ScheduledTimeoutActionCodec(protocol)
    elif model_type not in ("continuous", "continuous_heuristic"):
        action_codec = ProtocolTimeoutActionCodec(protocol)
    if aggregate_function == "median":
        aggregator: BaseAggregator = MedianAggregator(protocol=protocol)
    elif aggregate_function == "weighted_median":
        aggregator = WeightedMedianAggregator(protocol=protocol, weights=aggregate_weight)
    else:
        raise ValueError(f"Unsupported aggregation function: {aggregate_function}")

    injection_warmup_s = 0.0
    agent_injection_schedule = None
    if failure_spec is not None:
        injection_warmup_s, agent_injection_schedule = load_agent_injection_schedule(
            failure_spec
        )
    if agent_injection_schedule:
        print(
            "[injection] loaded schedule "
            f"(phases={len(agent_injection_schedule)}, warmup_s={injection_warmup_s:g}, "
            f"start_unix_ms={injection_start_unix_ms})"
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
        action_codec=action_codec,
        aggregator=aggregator,
        consensus_stub=consensus_stub,
        share_timeout_s=share_timeout_s,
        consensus_timeout_s=consensus_timeout_s,
        signing_enabled=signing_enabled,
        use_default_keys=use_default_keys,
        dataset_csv=dataset_csv,
        agent_injection_schedule=agent_injection_schedule,
        injection_start_mono=start_mono,
        injection_warmup_s=injection_warmup_s,
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
    _install_timestamped_print()

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
        default=100,
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
        help=(
            "Protocol for this agent: pbft, smartbft, sbft, tendermint, hotstuff, or autobahn."
        ),
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=(
            "random",
            "pavlov",
            "linear",
            "cmab",
            "cmab_reset",
            "cmab_v2",
            "cmab_v4",
            "claude_v2",
            "codex_v1",
            "continuous",
            "continuous_heuristic",
            "scheduled",
            "thompson",
        ),
        default="random",
        help=(
            "random: random timeout selection; "
            "pavlov: win-stay/lose-shift heuristic vs the last reward (no "
            "learning); linear: learned linear model; "
            "cmab: bootstrapped ensemble contextual multi-armed bandit; "
            "cmab_reset: blocked CMAB with replay-triggered resets "
            "(requires xgboost); "
            "claude_v2: cmab with the staleness bonus and contextual exploit "
            "that stop an arm being starved after a bad phase, with a 32-deep "
            "per-arm window so those trees can actually split; "
            "codex_v1: discounted LinUCB, one ridge model per arm; "
            "cmab_v2: the earlier CMAB iteration, one bootstrapped tree "
            "ensemble over (context, arm) with per-head fitting; "
            "cmab_v4: survival-gated - factorises reward into "
            "P(alive | context, arm) times performance, gates on the first and "
            "ranks on the second, so a timeout that stalls is ruled out from a "
            "handful of samples instead of averaged into a conditional mean; "
            "continuous: bootstrapped continuous contextual bandit; "
            "continuous_heuristic: k * avg_consensus_latency_ms, clamped to "
            "the arm range; no learning; "
            "thompson: deprecated alias for cmab; "
            "scheduled: timeout from a time-based schedule CSV. "
            "The remaining CMAB iterations (v1, v3) are simulator-only; see "
            "simulator.py --model."
        ),
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
        "--heuristic-multiplier",
        type=float,
        default=DEFAULT_PROPORTIONAL_K,
        help=(
            "Multiplier k for --model-type=continuous_heuristic: "
            "timeout_ms = clamp(k * avg_consensus_latency_ms, arm range). "
            f"Fit it with learning_agent/fit_proportional.py (default: "
            f"{DEFAULT_PROPORTIONAL_K})."
        ),
    )

    parser.add_argument(
        "--reward-design",
        choices=REWARD_DESIGNS,
        default=DEFAULT_REWARD_DESIGN,
        help=(
            "Reward shaping the agent learns from. reward_tps is throughput "
            "alone, concave past a 1000 tps knee. reward_mixed additionally "
            "discounts it by the share of the measurement window one fault "
            "would cost - the applied timeout plus one consensus latency - so "
            "a longer timer has to earn the detection delay it buys. Latency "
            "enters reward_mixed only as an overhead relative to the window, "
            "never as a level, because an absolute latency term ranks the "
            "delay environment instead of the arm (default: "
            f"{DEFAULT_REWARD_DESIGN})."
        ),
    )
    parser.add_argument(
        "--latency-floor-k",
        type=float,
        default=0.0,
        help=(
            "Domain constraint: never apply a timeout below k * the consensus "
            "latency the replicas are already achieving, estimated as a low "
            "quantile over recent committing windows. A timeout under that "
            "latency cannot be met, so it buys view changes and stalls "
            "throughput; excluding it costs the learner nothing it could have "
            "won. 0 disables the constraint (default: 0; try "
            f"{DEFAULT_LATENCY_FLOOR_K})."
        ),
    )
    parser.add_argument(
        "--latency-floor-quantile",
        type=float,
        default=DEFAULT_LATENCY_FLOOR_QUANTILE,
        help=(
            "Quantile of the recent-latency window taken as the round time to "
            "bound against. Lower resists contamination - latency measured "
            "under a badly chosen timeout is inflated by that choice - while "
            "higher is needed when faults are intermittent, because a low "
            "quantile then selects the windows that saw no fault and the floor "
            "stops binding at all. See DEFAULT_LATENCY_FLOOR_QUANTILE in "
            f"latency_floor.py for the measurements (default: "
            f"{DEFAULT_LATENCY_FLOOR_QUANTILE})."
        ),
    )

    parser.add_argument(
        "--schedule-file",
        type=Path,
        default=None,
        help=(
            "Path to schedule CSV. PBFT and SmartBFT format: time_s,election_timeout_ms. "
            "SBFT format: time_s,election_timeout_ms,slow_path_timeout_ms,batch_timeout_ms. "
            "Tendermint format: time_s,propose_timeout_ms; prevote and "
            "precommit are derived as one-third of propose. "
            "HotStuff format: time_s,timeout_ms. "
            "Autobahn format: time_s,timeout_delay_ms,car_timeout_ms,fast_path_timeout_ms. "
            "Negative time_s values are allowed for pre-start offsets; schedule "
            "activations are shifted so the earliest entry begins at time 0. "
            "Required when --model-type=scheduled."
        ),
    )
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV output path for completed episodes. Each row contains "
            "decision-time feature_* metrics, reward-window reward_* metrics, "
            "and the applied timeout. Scalar rewards are derived from "
            "reward_throughput_tps when needed."
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
    if args.model_type == "thompson":
        args.model_type = "cmab"

    # Process-wide, before any report is scored; see reward_utils.
    set_reward_design(args.reward_design)
    print(f"[reward] design={args.reward_design}")

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
        heuristic_multiplier=args.heuristic_multiplier,
        latency_floor_k=args.latency_floor_k,
        latency_floor_quantile=args.latency_floor_quantile,
        dataset_csv=args.dataset_csv,
        failure_spec=args.failure_spec,
        injection_start_unix_ms=args.injection_start_unix_ms,
    )


if __name__ == "__main__":
    main()
