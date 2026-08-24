import math
from typing import Callable, Mapping, Optional, Tuple, Type

import agent_pb2


REWARD_THROUGHPUT_FIELD = "reward_throughput_tps"

# Throughput reward is concave: every transaction per second up to the knee is
# worth full credit, and everything above it is discounted. The agent should
# therefore treat losing 200 tps at 600 tps as a much bigger deal than losing
# 200 tps at 3000 tps, which is what we want when the timeout choice is really
# about staying out of the degraded regime rather than about peak speed.
REWARD_THROUGHPUT_KNEE_TPS = 1000.0

# Marginal value of throughput above the knee, relative to below it. 0.25 means
# 1000 extra tps past the knee is worth as much as 250 tps below it.
REWARD_THROUGHPUT_ABOVE_KNEE_WEIGHT = 0.25


def reward_from_throughput(throughput_tps: float) -> float:
    """Apply the reward shaping used by the online learning agent."""
    value = float(max(0.0, throughput_tps))
    if value <= REWARD_THROUGHPUT_KNEE_TPS:
        return value / REWARD_THROUGHPUT_KNEE_TPS
    excess = (value - REWARD_THROUGHPUT_KNEE_TPS) / REWARD_THROUGHPUT_KNEE_TPS
    return 1.0 + REWARD_THROUGHPUT_ABOVE_KNEE_WEIGHT * excess


def calculate_pbft_reward(
    *,
    total_transactions: int,
    total_consensus_instances: int,
    avg_consensus_latency_ms: float,
    p50_consensus_latency_ms: float,
    p95_consensus_latency_ms: float,
    throughput_tps: float,
    avg_batch_size: float,
    p95_batch_size: float,
    leader_change_count: int,
    regency_change_count: int,
    timeout_ms: int,
    avg_inter_commit_gap_ms: float,
    p50_inter_commit_gap_ms: float,
    p95_inter_commit_gap_ms: float,
    view_change_count: int,
    no_progress_view_change_count: int,
) -> float:
    """Calculate an online PBFT reward from a complete reward report."""
    return reward_from_throughput(throughput_tps)


def calculate_sbft_reward(
    *,
    total_transactions: int,
    total_consensus_instances: int,
    avg_consensus_latency_ms: float,
    p95_consensus_latency_ms: float,
    p99_consensus_latency_ms: float,
    throughput_tps: float,
    timeout_violation_rate: float,
    avg_batch_size: float,
    p95_batch_size: float,
    leader_change_count: int,
    regency_change_count: int,
) -> float:
    """Calculate an online SBFT reward from a complete reward report."""
    return reward_from_throughput(throughput_tps)


def calculate_tendermint_reward(
    *,
    total_transactions: int,
    total_consensus_instances: int,
    avg_consensus_latency_ms: float,
    p50_consensus_latency_ms: float,
    p90_consensus_latency_ms: float,
    throughput_tps: float,
    timeout_violation_rate: float,
    avg_batch_size: float,
    p50_batch_size: float,
    p90_batch_size: float,
    leader_change_count: int,
    regency_change_count: int,
    propose_latency_ms: float,
    prevote_latency_ms: float,
    precommit_latency_ms: float,
) -> float:
    """Calculate an online Tendermint reward from a complete reward report."""
    return (throughput_tps - p90_consensus_latency_ms) / 4000.0


def reward_from_report_row(
    row: Mapping[str, Optional[str]], row_number: int
) -> float:
    raw_value = row.get(REWARD_THROUGHPUT_FIELD)
    if raw_value is None or raw_value.strip() == "":
        raise ValueError(
            f"Missing reward report value '{REWARD_THROUGHPUT_FIELD}' "
            f"at CSV line {row_number}."
        )
    try:
        throughput_tps = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid reward report value for '{REWARD_THROUGHPUT_FIELD}' "
            f"at CSV line {row_number}: {raw_value!r}"
        ) from exc
    return reward_from_throughput(throughput_tps)


_REPORT_TYPES: Mapping[int, Type[object]] = {
    agent_pb2.PROTOCOL_PBFT: agent_pb2.PbftReport,
    agent_pb2.PROTOCOL_SBFT: agent_pb2.SbftReport,
    agent_pb2.PROTOCOL_TENDERMINT: agent_pb2.TendermintReport,
}

_REPORT_REWARD_CALCULATORS: Mapping[int, Callable[..., float]] = {
    agent_pb2.PROTOCOL_PBFT: calculate_pbft_reward,
    agent_pb2.PROTOCOL_SBFT: calculate_sbft_reward,
    agent_pb2.PROTOCOL_TENDERMINT: calculate_tendermint_reward,
}


def get_reward_report_field_names(protocol: int) -> Tuple[str, ...]:
    """Return every field expected by a protocol's reward calculator."""
    report_type = _REPORT_TYPES.get(int(protocol))
    if report_type is None:
        raise ValueError(f"Unsupported protocol: {protocol}")
    return tuple(field.name for field in report_type.DESCRIPTOR.fields)


def calculate_reward_from_report_row(
    protocol: int,
    row: Mapping[str, object],
    row_number: int,
    prefix: str = "reward_",
) -> float:
    """Calculate a reward from all report fields stored in a CSV row."""
    report_type = _REPORT_TYPES.get(int(protocol))
    calculator = _REPORT_REWARD_CALCULATORS.get(int(protocol))
    if report_type is None or calculator is None:
        raise ValueError(f"Unsupported protocol: {protocol}")

    values = {}
    for field in report_type.DESCRIPTOR.fields:
        column = f"{prefix}{field.name}"
        raw_value = row.get(column)
        if raw_value is None or str(raw_value).strip() == "":
            raise ValueError(
                f"Missing reward report value '{column}' at CSV line {row_number}."
            )
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid reward report value for '{column}' at CSV line "
                f"{row_number}: {raw_value!r}"
            ) from exc
        if not math.isfinite(numeric_value):
            raise ValueError(
                f"Non-finite reward report value for '{column}' at CSV line "
                f"{row_number}: {raw_value!r}"
            )
        if field.type in (
            field.TYPE_INT32,
            field.TYPE_INT64,
            field.TYPE_UINT32,
            field.TYPE_UINT64,
            field.TYPE_SINT32,
            field.TYPE_SINT64,
            field.TYPE_FIXED32,
            field.TYPE_FIXED64,
            field.TYPE_SFIXED32,
            field.TYPE_SFIXED64,
        ):
            values[field.name] = int(numeric_value)
        else:
            values[field.name] = numeric_value

    return float(calculator(**values))
