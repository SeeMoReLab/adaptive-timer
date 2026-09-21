#!/usr/bin/env python3
"""Plot aligned Tendermint throughput, timeout, and failure-phase data."""

import argparse
import bisect
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence

KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
MONITOR_RE = re.compile(
    r"^--\s+Monitor\s+t=(?P<elapsed_s>\d+)s\s+finished_delta=(?P<finished>\d+)\b"
)
BENCHMARK_START_RE = re.compile(r"synced start-unix-ms:\s*(?P<unix_ms>\d+)")
NODE_FILE_RE = re.compile(r"node_(?P<node>\d+)\.log$")
# CometBFT log lines carry a time of day only, e.g. "I[17:13:55.089] ...".
NODE_TIMESTAMP_RE = re.compile(r"^[A-Z]\[(?P<time_of_day>\d{2}:\d{2}:\d{2}\.\d{3})\]")
# The network applier is the one process that prints the node's local wall clock
# next to the shared benchmark start, which is what anchors the node log clock.
APPLIER_ANCHOR_RE = re.compile(
    r"^\[(?P<local>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+"
    r"Waiting\s+(?P<wait_s>[0-9.]+)s\s+for\s+start_unix_ms=(?P<unix_ms>\d+)"
)

# The applied value is logged both on every height ("applying timeouts", carrying
# the effective value as *_after_ms) and on each agent recommendation ("applying
# new timeout", carrying it as *_ms).
TIMER_FIELDS: Dict[str, tuple[str, str]] = {
    "propose": ("propose_ms", "propose_after_ms"),
    "prevote": ("prevote_ms", "prevote_after_ms"),
    "precommit": ("precommit_ms", "precommit_after_ms"),
}
# The same "applying timeouts" line reports what was in effect beforehand, which
# seeds the series for nodes that log nothing before the first throughput sample.
TIMER_BEFORE_FIELDS: Dict[str, str] = {
    "propose": "propose_before_ms",
    "prevote": "prevote_before_ms",
    "precommit": "precommit_before_ms",
}
TIMEOUT_MESSAGES = (
    "adaptive_timer: applying new timeout",
    "adaptive_timer: applying timeouts",
)
PHASE_COLORS = ("#dceaf7", "#fff0c2", "#e6d9f2", "#d8ead2", "#f4d1d1", "#d7e5e5")


@dataclass(frozen=True)
class MonitorSample:
    elapsed_s: float
    throughput_tps: float


@dataclass(frozen=True)
class TimeoutEvent:
    elapsed_s: float
    timeout_ms: float


@dataclass(frozen=True)
class PhaseSpan:
    label: str
    start_s: float
    end_s: Optional[float]


def parse_monitor_series(
    file_path: Path, skip_warmup_lines: int
) -> tuple[datetime, list[MonitorSample]]:
    """Parse the benchmark start and per-interval Monitor throughput samples."""
    benchmark_start = None
    elapsed_s: list[float] = []
    finished: list[float] = []

    with file_path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            start_match = BENCHMARK_START_RE.search(line)
            if start_match:
                benchmark_start = datetime.fromtimestamp(
                    int(start_match.group("unix_ms")) / 1000.0, tz=timezone.utc
                )

            monitor_match = MONITOR_RE.search(line)
            if not monitor_match:
                continue
            sample_elapsed = float(monitor_match.group("elapsed_s"))
            sample_finished = float(monitor_match.group("finished"))
            if elapsed_s and sample_elapsed < elapsed_s[-1]:
                raise ValueError(
                    f"Monitor timestamps go backwards at {file_path}:{line_number}: "
                    f"t={sample_elapsed:g}s follows t={elapsed_s[-1]:g}s"
                )
            if elapsed_s and sample_elapsed == elapsed_s[-1]:
                # The monitor makes a final best-effort flush on shutdown, which
                # lands in the same whole second as the last periodic sample.
                finished[-1] += sample_finished
                continue
            elapsed_s.append(sample_elapsed)
            finished.append(sample_finished)

    if benchmark_start is None:
        raise ValueError(f"Benchmark start timestamp not found in {file_path}")
    if not elapsed_s:
        raise ValueError(f"No '-- Monitor' lines found in {file_path}")
    if skip_warmup_lines >= len(elapsed_s):
        raise ValueError(
            f"Only {len(elapsed_s)} Monitor lines found in {file_path}, "
            f"cannot skip {skip_warmup_lines}"
        )

    # finished_delta counts transactions completed since the previous sample, so
    # the rate needs the real gap rather than an assumed one-second interval.
    samples = []
    for index, sample_elapsed in enumerate(elapsed_s):
        interval_s = sample_elapsed - elapsed_s[index - 1] if index else sample_elapsed
        if interval_s <= 0:
            raise ValueError(
                f"Monitor sample at t={sample_elapsed:g}s in {file_path} has a "
                f"non-positive interval of {interval_s:g}s"
            )
        samples.append(MonitorSample(sample_elapsed, finished[index] / interval_s))
    return benchmark_start, samples[skip_warmup_lines:]


def node_clock_at_start(applier_log: Path, benchmark_start: datetime) -> datetime:
    """Return the node's local clock reading at the shared benchmark start."""
    if applier_log.is_file():
        with applier_log.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                anchor_match = APPLIER_ANCHOR_RE.search(line)
                if anchor_match:
                    logged_at = datetime.strptime(
                        anchor_match.group("local"), "%Y-%m-%d %H:%M:%S"
                    )
                    return logged_at + timedelta(
                        seconds=float(anchor_match.group("wait_s"))
                    )
    print(
        f"Warning: no start_unix_ms anchor found in {applier_log}; assuming the node "
        f"clock matches the UTC benchmark start.",
        file=sys.stderr,
    )
    return benchmark_start.astimezone(timezone.utc).replace(tzinfo=None)


def timeout_values(line: str, timer: str) -> tuple[Optional[float], Optional[float]]:
    """Return the applied and previously active timeouts from a node log line."""
    if not any(message in line for message in TIMEOUT_MESSAGES):
        return None, None
    values = dict(KV_RE.findall(line))
    applied = None
    for field in TIMER_FIELDS[timer]:
        raw_value = values.get(field)
        if raw_value is not None:
            applied = float(raw_value)
            break
    previous = values.get(TIMER_BEFORE_FIELDS[timer])
    return applied, float(previous) if previous is not None else None


def parse_node_log(
    file_path: Path, timer: str, clock_at_start: datetime
) -> tuple[int, list[TimeoutEvent], Optional[float]]:
    """Parse the applied-timeout step series from one node's CometBFT log."""
    filename_match = NODE_FILE_RE.search(file_path.name)
    if not filename_match:
        raise ValueError(f"Cannot determine node ID from log filename: {file_path.name}")
    node_id = int(filename_match.group("node"))

    events: list[TimeoutEvent] = []
    initial_timeout_ms = None
    previous_timeout_ms = None
    with file_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            timestamp_match = NODE_TIMESTAMP_RE.match(line)
            if not timestamp_match:
                continue
            timeout_ms, before_ms = timeout_values(line, timer)
            if initial_timeout_ms is None and before_ms is not None:
                initial_timeout_ms = before_ms
            if timeout_ms is None or timeout_ms == previous_timeout_ms:
                continue
            events.append(
                TimeoutEvent(
                    elapsed_s=node_elapsed_s(
                        timestamp_match.group("time_of_day"), clock_at_start
                    ),
                    timeout_ms=timeout_ms,
                )
            )
            previous_timeout_ms = timeout_ms

    events.sort(key=lambda event: event.elapsed_s)
    return node_id, events, initial_timeout_ms


def node_elapsed_s(time_of_day: str, clock_at_start: datetime) -> float:
    """Resolve a node log time of day to seconds since the benchmark start."""
    parsed = datetime.strptime(time_of_day, "%H:%M:%S.%f").time()
    # The log omits the date, so pick the calendar day that puts the entry
    # closest to the benchmark start. Runs are far shorter than the 12h this
    # allows, so the choice is unambiguous even across midnight.
    candidates = [
        datetime.combine(clock_at_start.date() + timedelta(days=offset), parsed)
        for offset in (-1, 0, 1)
    ]
    closest = min(candidates, key=lambda moment: abs(moment - clock_at_start))
    return (closest - clock_at_start).total_seconds()


def timeout_per_sample(
    events: Sequence[TimeoutEvent],
    sample_times: Sequence[float],
    initial_timeout_ms: Optional[float],
    node_id: int,
) -> list[float]:
    """Sample the most recently applied timeout at every throughput timestamp."""
    timeline = list(events)
    if initial_timeout_ms is not None and (
        not timeline or timeline[0].elapsed_s > sample_times[0]
    ):
        timeline.append(TimeoutEvent(sample_times[0], initial_timeout_ms))
        timeline.sort(key=lambda event: event.elapsed_s)

    event_times = [event.elapsed_s for event in timeline]
    values = []
    for sample_time in sample_times:
        event_index = bisect.bisect_right(event_times, sample_time) - 1
        if event_index < 0:
            raise ValueError(
                f"No timeout value is known for node {node_id} at t={sample_time:g}s. "
                f"Pass --initial-timeout-ms to seed the series."
            )
        values.append(timeline[event_index].timeout_ms)
    return values


def optional_float(parent: ET.Element, tag: str) -> Optional[float]:
    node = parent.find(tag)
    if node is None or node.text is None or not node.text.strip():
        return None
    return float(node.text.strip())


def phase_label(node: ET.Element, phase_index: int) -> str:
    """Describe a failure phase from its own spec when it carries no name."""
    name_node = node.find("name")
    if name_node is not None and name_node.text and name_node.text.strip():
        return name_node.text.strip()

    parts = []
    delay_ms = optional_float(node, "./network/global/delayMs")
    if delay_ms:
        burst_ms = optional_float(node, "./network/global/burstDurationMs") or 0.0
        interval_ms = optional_float(node, "./network/global/burstIntervalMs") or 0.0
        if 0 < burst_ms < interval_ms:
            parts.append(
                f"+{delay_ms:g} ms burst ({burst_ms / 1000:g}s/{interval_ms / 1000:g}s)"
            )
        else:
            parts.append(f"+{delay_ms:g} ms delay")
    proposal_delay_ms = optional_float(node, "./tendermint/proposalDelay/delayMs")
    if proposal_delay_ms:
        parts.append(f"proposal delay {proposal_delay_ms / 1000:g}s")
    return " + ".join(parts) if parts else f"Phase {phase_index + 1}"


def load_phase_spans(file_path: Path) -> list[PhaseSpan]:
    """Load warm-up and fault phase boundaries from failure_spec.xml."""
    root = ET.parse(file_path).getroot()
    warmup_ms = optional_float(root, "warmUpTimeMs")
    warmup_s = max(0.0, warmup_ms / 1000.0) if warmup_ms is not None else max(
        0.0, optional_float(root, "warmUpTime") or 0.0
    )
    phase_nodes = root.findall("./phases/phase")
    if not phase_nodes:
        raise ValueError(f"No failure phases found in {file_path}")

    explicit_start_tags = ("startAtMs", "atTimeMs", "startAt", "atTime")
    has_explicit_starts = any(
        any(node.find(tag) is not None for tag in explicit_start_tags) for node in phase_nodes
    )
    starts = []
    cumulative_s = 0.0
    for original_index, node in enumerate(phase_nodes):
        start_ms = optional_float(node, "startAtMs")
        if start_ms is None:
            start_ms = optional_float(node, "atTimeMs")
        start_s = start_ms / 1000.0 if start_ms is not None else optional_float(node, "startAt")
        if start_s is None:
            start_s = optional_float(node, "atTime")

        if start_s is None:
            legacy_time = optional_float(node, "time")
            if legacy_time is None:
                raise ValueError(
                    f"Failure phase {original_index + 1} has no start time in {file_path}"
                )
            if has_explicit_starts:
                start_s = legacy_time
            else:
                start_s = cumulative_s
                cumulative_s += legacy_time
        if start_s < 0:
            raise ValueError(f"Failure phase {original_index + 1} has a negative start time")
        starts.append((start_s, original_index, phase_label(node, original_index)))

    starts.sort(key=lambda item: (item[0], item[1]))
    spans = []
    if warmup_s > 0:
        spans.append(PhaseSpan("Warm-up", 0.0, warmup_s))
    for phase_index, (start_s, _original_index, label) in enumerate(starts):
        end_s = warmup_s + starts[phase_index + 1][0] if phase_index + 1 < len(starts) else None
        spans.append(PhaseSpan(label, warmup_s + start_s, end_s))
    return spans


def trailing_average(values: list[float], window: int) -> list[float]:
    """Return a trailing moving average with the same length as values."""
    if window <= 1:
        return values

    smoothed = []
    running_sum = 0.0
    for idx, value in enumerate(values):
        running_sum += value
        if idx >= window:
            running_sum -= values[idx - window]
        smoothed.append(running_sum / min(idx + 1, window))
    return smoothed


def add_phase_backgrounds(axes, spans: Sequence[PhaseSpan], x_min: float, x_max: float) -> None:
    """Shade both aligned plots and label visible phases on the top plot."""
    for phase_index, span in enumerate(spans):
        start_s = max(x_min, span.start_s)
        end_s = min(x_max, span.end_s if span.end_s is not None else x_max)
        if end_s <= start_s:
            continue
        color = PHASE_COLORS[phase_index % len(PHASE_COLORS)]
        for axis in axes:
            axis.axvspan(start_s, end_s, color=color, alpha=0.55, linewidth=0, zorder=0)
        axes[0].text(
            (start_s + end_s) / 2,
            0.98,
            span.label,
            ha="center",
            va="top",
            fontsize=9,
            color="#444444",
            transform=axes[0].get_xaxis_transform(),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot aligned Tendermint throughput and per-node applied timeouts."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a run's tendermint_output.txt, or the run directory holding it.",
    )
    parser.add_argument(
        "--node-logs-dir",
        type=Path,
        help="Directory containing node_N.log files. Default: <run directory>/logs.",
    )
    parser.add_argument(
        "--failure-spec",
        type=Path,
        help="Failure specification used for phase shading. Default: <run directory>/failure_spec.xml.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional image output path. If omitted, shows an interactive window.",
    )
    parser.add_argument(
        "--timer",
        choices=sorted(TIMER_FIELDS),
        default="propose",
        help="Which consensus timer to plot. Default: propose.",
    )
    parser.add_argument(
        "--skip-warmup-lines",
        type=int,
        default=1,
        help="Number of initial Monitor lines to omit. Default: 1, dropping the "
        "monitor's initialization sample.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Trailing moving-average window in Monitor samples. Use 1 to disable smoothing. Default: 1.",
    )
    parser.add_argument(
        "--initial-timeout-ms",
        type=float,
        help="Seed value for nodes whose log has no applied timeout before the first "
        "throughput sample. By default the node log must supply one.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = args.input.resolve()
    if input_path.is_dir():
        input_path = input_path / "tendermint_output.txt"
    run_dir = input_path.parent
    node_logs_dir = (args.node_logs_dir or run_dir / "logs").resolve()
    failure_spec = (args.failure_spec or run_dir / "failure_spec.xml").resolve()

    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")
    if not node_logs_dir.is_dir():
        raise SystemExit(f"Node log directory not found: {node_logs_dir}")
    if not failure_spec.is_file():
        raise SystemExit(f"Failure specification not found: {failure_spec}")
    if args.skip_warmup_lines < 0:
        raise SystemExit("--skip-warmup-lines must be non-negative.")
    if args.smooth_window < 1:
        raise SystemExit("--smooth-window must be >= 1.")
    if args.initial_timeout_ms is not None and args.initial_timeout_ms <= 0:
        raise SystemExit("--initial-timeout-ms must be positive.")

    node_log_paths = sorted(node_logs_dir.glob("node_*.log"))
    if not node_log_paths:
        raise SystemExit(f"No node_N.log files found in {node_logs_dir}")

    try:
        benchmark_start, samples = parse_monitor_series(input_path, args.skip_warmup_lines)
        phase_spans = load_phase_spans(failure_spec)
        timeout_series = {}
        sample_times = [sample.elapsed_s for sample in samples]
        for node_log_path in node_log_paths:
            node_match = NODE_FILE_RE.search(node_log_path.name)
            if not node_match:
                continue
            clock_at_start = node_clock_at_start(
                node_logs_dir / f"applier_{node_match.group('node')}.log", benchmark_start
            )
            node_id, events, logged_initial_ms = parse_node_log(
                node_log_path, args.timer, clock_at_start
            )
            timeout_series[node_id] = timeout_per_sample(
                events,
                sample_times,
                args.initial_timeout_ms or logged_initial_ms,
                node_id,
            )
    except (OSError, ET.ParseError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    throughput = [sample.throughput_tps for sample in samples]
    plotted_throughput = trailing_average(throughput, args.smooth_window)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required to plot Tendermint data. Install it with `pip install matplotlib`."
        ) from exc

    fig, (throughput_ax, timeout_ax) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": (2, 1)}
    )
    add_phase_backgrounds(
        (throughput_ax, timeout_ax), phase_spans, sample_times[0], sample_times[-1]
    )

    if args.smooth_window > 1:
        throughput_ax.plot(
            sample_times, throughput, color="tab:blue", alpha=0.25, linewidth=0.8,
            label="Raw throughput", zorder=2,
        )
        throughput_ax.plot(
            sample_times,
            plotted_throughput,
            color="tab:blue",
            linewidth=1.8,
            label=f"{args.smooth_window}-sample trailing average",
            zorder=3,
        )
        # Sit below the phase labels drawn along the top of this axis.
        throughput_ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.94))
    else:
        throughput_ax.plot(sample_times, throughput, color="tab:blue", linewidth=1.2, zorder=2)
    throughput_ax.set_title(
        f"Tendermint Throughput and Applied {args.timer.capitalize()} Timeout ({run_dir.name})"
    )
    throughput_ax.set_ylabel("Committed transactions/s")

    line_styles = ("-", "--", "-.", ":")
    markers = ("o", "s", "^", "D", "v", "P")
    marker_interval = max(1, len(sample_times) // 30)
    for series_index, node_id in enumerate(sorted(timeout_series)):
        timeout_ax.step(
            sample_times,
            timeout_series[node_id],
            where="post",
            linewidth=1.5,
            linestyle=line_styles[series_index % len(line_styles)],
            marker=markers[series_index % len(markers)],
            markevery=marker_interval,
            markersize=3.5,
            label=f"Node {node_id}",
            zorder=2 + series_index,
        )
    timeout_ax.set_ylabel(f"{args.timer.capitalize()} timeout used (ms)")
    timeout_ax.set_xlabel("Elapsed time since benchmark start (s)")
    # Timeouts often sit flat at the top of their range, so leave headroom rather
    # than letting the legend land on the series.
    timeout_ax.margins(y=0.25)
    timeout_ax.legend(loc="upper right", ncol=min(4, len(timeout_series)))

    for axis in (throughput_ax, timeout_ax):
        axis.grid(True, linestyle="--", alpha=0.35, zorder=1)
        axis.set_xlim(sample_times[0], sample_times[-1])
    fig.tight_layout()

    if args.output:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
