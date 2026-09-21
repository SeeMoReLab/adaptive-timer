#!/usr/bin/env python3
"""Plot SmartBFT reward distributions by timeout and phase."""

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "learning_agent"))
import agent_pb2
from reward_utils import calculate_reward_from_report_row


# A hairline of surface colour between stacked segments, so adjacent bands stay
# separable rather than reading as one block.
SEGMENT_GAP_COLOR = "#ffffff"
SEGMENT_GAP_WIDTH = 1.0

TIMEOUT_COLUMNS = (
    "election_timeout_milliseconds_used",
    "timeout_milliseconds_used",
    "timeout_ms",
)

PHASE_LABELS: Dict[int, str] = {
    0: "No net delay",
    1: "No net delay + 300 ms proposal delay",
    2: "No net delay + 600 ms proposal delay",
    3: "Small net delay A",
    4: "Small net delay B",
    5: "Small net delay A + 300 ms proposal delay",
    6: "Small net delay B + 300 ms proposal delay",
    7: "Large net delay",
    8: "Recovered: no net delay",
}


@dataclass(frozen=True)
class RewardSample:
    phase: int
    timeout_ms: int
    reward: float


@dataclass(frozen=True)
class RewardStats:
    p10: float
    median: float
    p90: float
    count: int


def _agent_sort_key(path: Path) -> Tuple[int, str]:
    match = re.search(r"agent(\d+)", path.stem)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def _parse_float(value: str, field_name: str, path: Path, row_number: int) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}:{row_number}: invalid {field_name} value {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{path}:{row_number}: non-finite {field_name} value {value!r}")
    return parsed


def _parse_int(value: str, field_name: str, path: Path, row_number: int) -> int:
    return int(round(_parse_float(value, field_name, path, row_number)))


def _choose_column(fieldnames: Iterable[str], requested: str, candidates: Sequence[str]) -> str:
    names = set(fieldnames)
    if requested != "auto":
        if requested not in names:
            raise ValueError(f"Column {requested!r} not found")
        return requested
    for column in candidates:
        if column in names:
            return column
    raise ValueError("No matching column found. Tried: " + ", ".join(candidates))


def _parse_timeout_options(value: str) -> List[int]:
    options: List[int] = []
    for raw in value.split(","):
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            timeout = int(round(float(stripped)))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid timeout option {stripped!r}"
            ) from exc
        if timeout <= 0:
            raise argparse.ArgumentTypeError("Timeout options must be positive")
        options.append(timeout)
    if not options:
        raise argparse.ArgumentTypeError("At least one timeout option is required")
    return options


def read_agent_csv(path: Path, timeout_column: str, reward_column: str) -> List[RewardSample]:
    samples: List[RewardSample] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return samples

        timeout_field = _choose_column(reader.fieldnames, timeout_column, TIMEOUT_COLUMNS)
        required = {"phase", timeout_field}
        if reward_column != "reward":
            required.add(reward_column)
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")

        for row_number, row in enumerate(reader, start=2):
            if not any(value and value.strip() for value in row.values()):
                continue
            phase = _parse_int(row.get("phase", ""), "phase", path, row_number)
            if phase < 0:
                continue
            if reward_column == "reward":
                reward = calculate_reward_from_report_row(
                    agent_pb2.PROTOCOL_SMARTBFT, row, row_number=row_number
                )
            else:
                raw_reward = row.get(reward_column, "")
                if raw_reward is None or not raw_reward.strip():
                    continue
                reward = _parse_float(raw_reward, reward_column, path, row_number)
            samples.append(
                RewardSample(
                    phase=phase,
                    timeout_ms=_parse_int(
                        row.get(timeout_field, ""), timeout_field, path, row_number
                    ),
                    reward=reward,
                )
            )
    return samples


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def compute_stats(samples: List[RewardSample]) -> Dict[int, Dict[int, RewardStats]]:
    grouped: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        grouped[sample.phase][sample.timeout_ms].append(sample.reward)

    stats: Dict[int, Dict[int, RewardStats]] = {}
    for phase, by_timeout in grouped.items():
        stats[phase] = {}
        for timeout_ms, rewards in by_timeout.items():
            stats[phase][timeout_ms] = RewardStats(
                p10=quantile(rewards, 0.10),
                median=quantile(rewards, 0.50),
                p90=quantile(rewards, 0.90),
                count=len(rewards),
            )
    return stats


def plot_reward_stats(
    stats: Dict[int, Dict[int, RewardStats]],
    timeout_options: List[int],
    output: Optional[Path],
    title: str,
    y_label: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required to render the plot. Install it with `pip install matplotlib`."
        ) from exc

    phases = sorted(stats)
    if not phases:
        raise SystemExit("No completed episode rows with phase>=0 found.")

    cols = min(3, len(phases))
    rows = int(math.ceil(len(phases) / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(5.2 * cols, 3.8 * rows),
        sharey=False,
        squeeze=False,
    )

    median_color = "#2563eb"
    p90_color = "#f97316"
    p10_color = "#7c3aed"
    x_positions = list(range(len(timeout_options)))
    for index, phase in enumerate(phases):
        ax = axes[index // cols][index % cols]
        medians: List[float] = []
        p90_extensions: List[float] = []
        p10s: List[float] = []
        mid_extensions: List[float] = []
        counts: List[int] = []
        phase_max = 0.0
        for timeout in timeout_options:
            stat = stats.get(phase, {}).get(timeout)
            if stat is None:
                medians.append(math.nan)
                p90_extensions.append(0.0)
                p10s.append(math.nan)
                mid_extensions.append(0.0)
                counts.append(0)
                continue
            median = max(0.0, stat.median)
            extension = max(0.0, stat.p90 - stat.median)
            medians.append(median)
            p90_extensions.append(extension)
            # Clamped like the median above: this chart pins the y-axis at 0.
            p10 = max(0.0, stat.p10)
            p10s.append(p10)
            mid_extensions.append(max(0.0, median - p10))
            counts.append(stat.count)
            phase_max = max(phase_max, median + extension)

        # Three stacked bands: 0..p10, p10..median, median..p90.
        ax.bar(
            x_positions,
            p10s,
            color=p10_color,
            label="p10",
            edgecolor=SEGMENT_GAP_COLOR,
            linewidth=SEGMENT_GAP_WIDTH,
        )
        ax.bar(
            x_positions,
            mid_extensions,
            bottom=p10s,
            color=median_color,
            label="median - p10",
            edgecolor=SEGMENT_GAP_COLOR,
            linewidth=SEGMENT_GAP_WIDTH,
        )
        ax.bar(
            x_positions,
            p90_extensions,
            bottom=medians,
            color=p90_color,
            label="p90 - median",
            edgecolor=SEGMENT_GAP_COLOR,
            linewidth=SEGMENT_GAP_WIDTH,
        )
        for x, median, extension, count in zip(
            x_positions, medians, p90_extensions, counts
        ):
            top = median + extension if math.isfinite(median) else 0.0
            if count == 0:
                ax.text(
                    x,
                    0.0,
                    "n=0",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#666666",
                )
            else:
                ax.text(
                    x,
                    top,
                    f"n={count}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#333333",
                )

        phase_label = PHASE_LABELS.get(phase, f"Phase {phase}")
        ax.set_title(f"Phase {phase}: {phase_label}")
        ax.set_xticks(x_positions, [str(timeout) for timeout in timeout_options])
        ax.set_xlabel("Timeout ms")
        if phase_max > 0:
            ax.set_ylim(0, phase_max * 1.25)
        else:
            ax.set_ylim(0, 1)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    for index in range(len(phases), rows * cols):
        axes[index // cols][index % cols].axis("off")

    for row in range(rows):
        axes[row][0].set_ylabel(y_label)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight", dpi=160)
        print(f"Saved plot to: {output}")
    else:
        plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot p10, median and p90 reward by timeout for each SmartBFT phase. "
            "Only completed episodes with phase>=0 are used."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing agent*.csv files for one SmartBFT run.",
    )
    parser.add_argument(
        "--pattern",
        default="agent*.csv",
        help="Glob pattern relative to --input-dir. Default: agent*.csv.",
    )
    parser.add_argument(
        "--timeouts",
        type=_parse_timeout_options,
        default=_parse_timeout_options("300,800,1800"),
        help="Comma-separated timeout options for the x-axis. Default: 300,800,1800.",
    )
    parser.add_argument(
        "--timeout-column",
        default="auto",
        help=(
            "Timeout column to group by. Default auto tries "
            + ", ".join(TIMEOUT_COLUMNS)
            + "."
        ),
    )
    parser.add_argument(
        "--reward-column",
        default="reward",
        help=(
            "Value to aggregate. Default 'reward' calculates the current online "
            "SmartBFT reward (reward_utils.calculate_smartbft_reward) from the "
            "reward_* report columns; a CSV column may also be supplied, such as "
            "reward_throughput_tps."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional image output path. If omitted, shows an interactive window.",
    )
    parser.add_argument(
        "--title",
        default="SmartBFT Reward by Timeout and Phase",
        help="Plot title.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    csv_paths = sorted(args.input_dir.glob(args.pattern), key=_agent_sort_key)
    if not csv_paths:
        raise SystemExit(f"No CSV files matched: {args.input_dir / args.pattern}")

    samples: List[RewardSample] = []
    for path in csv_paths:
        samples.extend(read_agent_csv(path, args.timeout_column, args.reward_column))

    y_label = "Reward" if args.reward_column == "reward" else args.reward_column
    plot_reward_stats(
        compute_stats(samples), args.timeouts, args.output, args.title, y_label
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
