from typing import Mapping, Optional


REWARD_THROUGHPUT_FIELD = "reward_throughput_tps"


def reward_from_throughput(throughput_tps: float) -> float:
    """Apply the reward shaping used by the online learning agent."""
    value = float(max(0.0, throughput_tps))
    if value < 1000.0:
        return (value / 1000.0) ** 2 * 1000.0
    return value


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
