#!/usr/bin/env python3
"""Plot aligned SmartBFT throughput, timeout, and failure-phase data."""

import argparse
import bisect
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
LOG_TIMESTAMP_RE = re.compile(
    r"\[[A-Za-z0-9_-]+ (?P<timestamp>\d{4}-\d{2}-\d{2}T[^\]]+)\]"
)
MONITOR_RE = re.compile(
    r"^\[client (?P<timestamp>\d{4}-\d{2}-\d{2}T[^\]]+)\]\s+Monitor\b"
)
BENCHMARK_START_RE = re.compile(r"Waiting until benchmark start unix ms (?P<unix_ms>\d+)")
SERVER_FILE_RE = re.compile(r"server_(?P<node>\d+)\.log$")
TIMEOUT_MESSAGES = (
    "SmartBFT request timeout active:",
    "applied SmartBFT request timeout:",
    "applied SmartBFT timeouts:",
)
LEGACY_TIMESTAMP_BRACKET_SECONDS = 2.0
PHASE_COLORS = ("#dceaf7", "#fff0c2", "#e6d9f2", "#d8ead2", "#f4d1d1", "#d7e5e5")


@dataclass(frozen=True)
class MonitorSample:
    timestamp: datetime
    succ: float
    # Client-observed latency percentiles for the interval, in ms. A window
    # with no successful transactions reports 0; plotted as a gap.
    p50_ms: float
    p95_ms: float


@dataclass(frozen=True)
class TimeoutEvent:
    timestamp: datetime
    timeout_ms: float


@dataclass(frozen=True)
class PhaseSpan:
    label: str
    start_s: float
    end_s: Optional[float]


@dataclass(frozen=True)
class ParsedTimeoutLog:
    node_id: int
    events: list[TimeoutEvent]
    inferred_legacy_events: int
    skipped_legacy_events: int


def parse_timestamp(raw: str) -> datetime:
    """Parse an ISO-8601 timestamp and require a UTC offset."""
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp has no UTC offset: {raw}")
    return parsed


def parse_monitor_series(file_path: Path, skip_warmup_lines: int) -> tuple[datetime, list[MonitorSample]]:
    """Parse the benchmark start and timestamped Monitor throughput samples."""
    benchmark_start = None
    samples = []

    with file_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            start_match = BENCHMARK_START_RE.search(line)
            if start_match:
                benchmark_start = datetime.fromtimestamp(
                    int(start_match.group("unix_ms")) / 1000.0, tz=timezone.utc
                )

            monitor_match = MONITOR_RE.search(line)
            if not monitor_match:
                continue
            values = dict(KV_RE.findall(line))
            if "succ" not in values:
                continue
            try:
                samples.append(
                    MonitorSample(
                        timestamp=parse_timestamp(monitor_match.group("timestamp")),
                        succ=float(values["succ"]),
                        p50_ms=float(values.get("p50", 0.0)),
                        p95_ms=float(values.get("p95", 0.0)),
                    )
                )
            except ValueError as exc:
                raise ValueError(f"Invalid Monitor line {file_path}:{line_number}: {exc}") from exc

    if benchmark_start is None:
        raise ValueError(f"Benchmark start timestamp not found in {file_path}")
    if skip_warmup_lines >= len(samples):
        raise ValueError(
            f"Only {len(samples)} Monitor lines found in {file_path}, cannot skip {skip_warmup_lines}"
        )
    return benchmark_start, samples[skip_warmup_lines:]


def timeout_value(line: str) -> Optional[tuple[int, float]]:
    """Return node ID and the active total request timeout from an application log line."""
    if not any(message in line for message in TIMEOUT_MESSAGES):
        return None
    values = dict(KV_RE.findall(line))
    timeout = values.get("effective_timeout_ms") or values.get("total_timeout_ms")
    if "node" not in values or timeout is None:
        return None
    try:
        return int(values["node"]), float(timeout)
    except ValueError:
        return None


def parse_timeout_log(file_path: Path) -> ParsedTimeoutLog:
    """Parse exact timestamped timeout changes and tightly bracketed legacy entries."""
    filename_match = SERVER_FILE_RE.search(file_path.name)
    if not filename_match:
        raise ValueError(f"Cannot determine server ID from log filename: {file_path.name}")
    expected_node_id = int(filename_match.group("node"))
    events = []
    inferred_legacy_events = 0
    skipped_legacy_events = 0
    previous_timestamp = None
    pending_legacy = []

    def resolve_pending(next_timestamp: datetime) -> None:
        nonlocal inferred_legacy_events, skipped_legacy_events
        for prior_timestamp, _node_id, timeout_ms in pending_legacy:
            if (
                prior_timestamp is not None
                and 0 <= (next_timestamp - prior_timestamp).total_seconds() <= LEGACY_TIMESTAMP_BRACKET_SECONDS
            ):
                midpoint = prior_timestamp + (next_timestamp - prior_timestamp) / 2
                events.append(TimeoutEvent(midpoint, timeout_ms))
                inferred_legacy_events += 1
            else:
                skipped_legacy_events += 1
        pending_legacy.clear()

    with file_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            timestamp_match = LOG_TIMESTAMP_RE.search(line)
            line_timestamp = (
                parse_timestamp(timestamp_match.group("timestamp")) if timestamp_match else None
            )
            if line_timestamp is not None and pending_legacy:
                resolve_pending(line_timestamp)

            parsed_value = timeout_value(line)
            if parsed_value is not None:
                node_id, timeout_ms = parsed_value
                if node_id != expected_node_id:
                    raise ValueError(
                        f"Node ID {node_id} in {file_path} does not match server filename ID {expected_node_id}"
                    )
                if line_timestamp is not None:
                    events.append(TimeoutEvent(line_timestamp, timeout_ms))
                else:
                    pending_legacy.append((previous_timestamp, node_id, timeout_ms))

            if line_timestamp is not None:
                previous_timestamp = line_timestamp

    skipped_legacy_events += len(pending_legacy)
    events.sort(key=lambda event: event.timestamp)
    return ParsedTimeoutLog(
        node_id=expected_node_id,
        events=events,
        inferred_legacy_events=inferred_legacy_events,
        skipped_legacy_events=skipped_legacy_events,
    )


def timeout_per_sample(
    events: Sequence[TimeoutEvent],
    sample_times: Sequence[datetime],
    benchmark_start: datetime,
    initial_timeout_ms: float,
) -> list[float]:
    """Sample the most recently active timeout at every throughput timestamp."""
    timeline = list(events)
    if not timeline or timeline[0].timestamp > sample_times[0]:
        timeline.append(TimeoutEvent(benchmark_start, initial_timeout_ms))
        timeline.sort(key=lambda event: event.timestamp)

    event_times = [event.timestamp for event in timeline]
    values = []
    for sample_time in sample_times:
        event_index = bisect.bisect_right(event_times, sample_time) - 1
        if event_index < 0:
            raise ValueError(
                f"No timeout value is known at throughput timestamp {sample_time.isoformat()}"
            )
        values.append(timeline[event_index].timeout_ms)
    return values


def optional_float(parent: ET.Element, tag: str) -> Optional[float]:
    node = parent.find(tag)
    if node is None or node.text is None or not node.text.strip():
        return None
    return float(node.text.strip())


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
                raise ValueError(f"Failure phase {original_index + 1} has no start time in {file_path}")
            if has_explicit_starts:
                start_s = legacy_time
            else:
                start_s = cumulative_s
                cumulative_s += legacy_time
        if start_s < 0:
            raise ValueError(f"Failure phase {original_index + 1} has a negative start time")

        name_node = node.find("name")
        name = name_node.text.strip() if name_node is not None and name_node.text else ""
        starts.append((start_s, original_index, name))

    starts.sort(key=lambda item: (item[0], item[1]))
    spans = []
    if warmup_s > 0:
        spans.append(PhaseSpan("Warm-up", 0.0, warmup_s))
    for phase_index, (start_s, _original_index, name) in enumerate(starts):
        end_s = warmup_s + starts[phase_index + 1][0] if phase_index + 1 < len(starts) else None
        spans.append(
            PhaseSpan(name or f"Phase {phase_index + 1}", warmup_s + start_s, end_s)
        )
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
        description="Plot aligned SmartBFT throughput and per-server timeout values."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a run's smartbft_output.txt, or the run directory holding it.",
    )
    parser.add_argument(
        "--server-logs-dir",
        type=Path,
        help="Directory containing server_N.log files. Default: <input directory>/logs/servers.",
    )
    parser.add_argument(
        "--failure-spec",
        type=Path,
        help="Failure specification used for phase shading. Default: <input directory>/failure_spec.xml.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional image output path. If omitted, shows an interactive window.",
    )
    parser.add_argument(
        "--skip-warmup-lines",
        type=int,
        default=12,
        help="Number of initial Monitor lines to omit. Default: 12.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Trailing moving-average window in Monitor samples. Use 1 to disable smoothing. Default: 1.",
    )
    parser.add_argument(
        "--latency-cap-ms",
        type=float,
        help="Cap the latency panel at this value in ms: the axis spans 0 to the cap and larger samples are drawn at the cap.",
    )
    parser.add_argument(
        "--initial-timeout-ms",
        type=float,
        default=800.0,
        help="Actual initial SmartBFT total request timeout for legacy logs without an initial active-timeout entry. Default: 800.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = args.input.resolve()
    if input_path.is_dir():
        input_path = input_path / "smartbft_output.txt"
    run_dir = input_path.parent
    server_logs_dir = (args.server_logs_dir or run_dir / "logs" / "servers").resolve()
    failure_spec = (args.failure_spec or run_dir / "failure_spec.xml").resolve()

    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")
    if not server_logs_dir.is_dir():
        raise SystemExit(f"Server log directory not found: {server_logs_dir}")
    if not failure_spec.is_file():
        raise SystemExit(f"Failure specification not found: {failure_spec}")
    if args.skip_warmup_lines < 0:
        raise SystemExit("--skip-warmup-lines must be non-negative.")
    if args.smooth_window < 1:
        raise SystemExit("--smooth-window must be >= 1.")
    if args.initial_timeout_ms <= 0:
        raise SystemExit("--initial-timeout-ms must be positive.")
    if args.latency_cap_ms is not None and args.latency_cap_ms <= 0:
        raise SystemExit("--latency-cap-ms must be positive.")

    try:
        benchmark_start, samples = parse_monitor_series(input_path, args.skip_warmup_lines)
        phase_spans = load_phase_spans(failure_spec)
        parsed_logs = [parse_timeout_log(path) for path in sorted(server_logs_dir.glob("server_*.log"))]
    except (OSError, ET.ParseError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if not parsed_logs:
        raise SystemExit(f"No server_N.log files found in {server_logs_dir}")

    sample_times = [sample.timestamp for sample in samples]
    elapsed_s = [(sample_time - benchmark_start).total_seconds() for sample_time in sample_times]
    throughput = [sample.succ for sample in samples]
    plotted_throughput = trailing_average(throughput, args.smooth_window)

    timeout_series = {}
    for parsed_log in parsed_logs:
        log_path = server_logs_dir / f"server_{parsed_log.node_id}.log"
        if parsed_log.inferred_legacy_events:
            print(
                f"Warning: inferred timestamps for {parsed_log.inferred_legacy_events} legacy timeout entries in {log_path}.",
                file=sys.stderr,
            )
        if parsed_log.skipped_legacy_events:
            print(
                f"Warning: skipped {parsed_log.skipped_legacy_events} legacy timeout entries without a tight timestamp bracket in {log_path}.",
                file=sys.stderr,
            )
        timeout_series[parsed_log.node_id] = timeout_per_sample(
            parsed_log.events, sample_times, benchmark_start, args.initial_timeout_ms
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required to plot SmartBFT data. Install it with `pip install matplotlib`."
        ) from exc

    fig, (throughput_ax, latency_ax, timeout_ax) = plt.subplots(
        3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": (2, 1.2, 1)}
    )
    add_phase_backgrounds(
        (throughput_ax, latency_ax, timeout_ax), phase_spans, elapsed_s[0], elapsed_s[-1]
    )

    if args.smooth_window > 1:
        throughput_ax.plot(
            elapsed_s, throughput, color="tab:blue", alpha=0.25, linewidth=0.8, label="Raw succ", zorder=2
        )
        throughput_ax.plot(
            elapsed_s,
            plotted_throughput,
            color="tab:blue",
            linewidth=1.8,
            label=f"{args.smooth_window}-sample trailing average",
            zorder=3,
        )
        throughput_ax.legend(loc="upper right")
    else:
        throughput_ax.plot(elapsed_s, throughput, color="tab:blue", linewidth=1.2, zorder=2)
    throughput_ax.set_title(f"SmartBFT Throughput and Applied Timeout ({run_dir.name})")
    throughput_ax.set_ylabel("Successful transactions/s")

    # Client-observed latency percentiles; windows with no successes report
    # 0 and are drawn as gaps.
    p50_series = [sample.p50_ms if sample.p50_ms > 0 else float("nan") for sample in samples]
    p95_series = [sample.p95_ms if sample.p95_ms > 0 else float("nan") for sample in samples]
    if args.latency_cap_ms is not None:
        # Samples above the cap are drawn at the cap so spikes stay visible
        # without setting the axis scale.
        p50_series = [min(value, args.latency_cap_ms) for value in p50_series]
        p95_series = [min(value, args.latency_cap_ms) for value in p95_series]
        latency_ax.set_ylim(0, args.latency_cap_ms)
    latency_ax.plot(elapsed_s, p95_series, color="tab:orange", linewidth=1.0, label="p95", zorder=2)
    latency_ax.plot(elapsed_s, p50_series, color="tab:green", linewidth=1.2, label="p50", zorder=3)
    latency_ax.set_ylabel("Client latency (ms)")
    latency_ax.legend(loc="upper right", ncol=2)

    line_styles = ("-", "--", "-.", ":")
    markers = ("o", "s", "^", "D", "v", "P")
    marker_interval = max(1, len(elapsed_s) // 30)
    for series_index, node_id in enumerate(sorted(timeout_series)):
        timeout_ax.step(
            elapsed_s,
            timeout_series[node_id],
            where="post",
            linewidth=1.5,
            linestyle=line_styles[series_index % len(line_styles)],
            marker=markers[series_index % len(markers)],
            markevery=marker_interval,
            markersize=3.5,
            label=f"Server {node_id}",
            zorder=2 + series_index,
        )
    timeout_ax.set_ylabel("Timeout used (ms)")
    timeout_ax.set_xlabel("Elapsed time since benchmark start (s)")
    timeout_ax.legend(loc="upper right", ncol=min(4, len(timeout_series)))

    for axis in (throughput_ax, latency_ax, timeout_ax):
        axis.grid(True, linestyle="--", alpha=0.35, zorder=1)
        axis.set_xlim(elapsed_s[0], elapsed_s[-1])
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
