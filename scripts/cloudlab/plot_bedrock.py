#!/usr/bin/env python3
"""Plot aligned CppBedrock (PBFT/SBFT) throughput, client latency, replica
views, and applied election timeouts for one run directory.

The client Monitor lines share the SmartBFT format, so the throughput and
latency parsing comes from plot_smartbft.py; the replica logs are parsed
here for the per-second "Stats view=..." line (elections show up as view
steps) and the "applied timeouts: election_timeout_ms=..." line.
"""

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_smartbft import (  # noqa: E402
    KV_RE,
    LOG_TIMESTAMP_RE,
    SERVER_FILE_RE,
    TimeoutEvent,
    add_phase_backgrounds,
    load_phase_spans,
    parse_monitor_series,
    parse_timestamp,
    timeout_per_sample,
    trailing_average,
)

APPLIED_TIMEOUT_MARKERS = ("applied timeouts:", "loaded protocol ")


@dataclass(frozen=True)
class ViewSample:
    timestamp: datetime
    view: int


@dataclass(frozen=True)
class ParsedServerLog:
    node_id: int
    timeout_events: list[TimeoutEvent]
    views: list[ViewSample]
    initial_timeout_ms: float | None


def parse_server_log(file_path: Path) -> ParsedServerLog:
    """Parse the election timeout changes and view samples of one replica."""
    filename_match = SERVER_FILE_RE.search(file_path.name)
    if not filename_match:
        raise ValueError(f"Cannot determine server ID from log filename: {file_path.name}")
    node_id = int(filename_match.group("node"))
    timeout_events: list[TimeoutEvent] = []
    views: list[ViewSample] = []
    initial_timeout_ms = None

    with file_path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            timestamp_match = LOG_TIMESTAMP_RE.search(line)
            if not timestamp_match:
                continue
            values = dict(KV_RE.findall(line))
            if " Stats " in line and "view" in values:
                try:
                    views.append(ViewSample(parse_timestamp(timestamp_match.group("timestamp")), int(values["view"])))
                except ValueError as exc:
                    raise ValueError(f"Invalid Stats line {file_path}:{line_number}: {exc}") from exc
                continue
            if any(marker in line for marker in APPLIED_TIMEOUT_MARKERS) and "election_timeout_ms" in values:
                try:
                    timeout_ms = float(values["election_timeout_ms"])
                except ValueError as exc:
                    raise ValueError(f"Invalid timeout line {file_path}:{line_number}: {exc}") from exc
                timestamp = parse_timestamp(timestamp_match.group("timestamp"))
                if "loaded protocol " in line:
                    initial_timeout_ms = timeout_ms
                timeout_events.append(TimeoutEvent(timestamp, timeout_ms))

    timeout_events.sort(key=lambda event: event.timestamp)
    views.sort(key=lambda sample: sample.timestamp)
    return ParsedServerLog(node_id, timeout_events, views, initial_timeout_ms)


def build_parser(protocol: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Plot aligned CppBedrock {protocol.upper()} throughput, latency, views, and timeouts."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=f"Path to a run's {protocol}_output.txt, or the run directory holding it.",
    )
    parser.add_argument("--server-logs-dir", type=Path, help="Directory with server_N.log files. Default: <run>/logs/servers.")
    parser.add_argument("--failure-spec", type=Path, help="Failure spec for phase shading. Default: <run>/failure_spec.xml.")
    parser.add_argument("--output", type=Path, help="Image output path. If omitted, shows an interactive window.")
    parser.add_argument("--skip-warmup-lines", type=int, default=12, help="Initial Monitor lines to omit. Default: 12.")
    parser.add_argument("--smooth-window", type=int, default=1, help="Trailing moving-average window in samples. Default: 1.")
    parser.add_argument("--latency-cap-ms", type=float, help="Cap the latency panel at this many ms.")
    return parser


def main(protocol: str) -> int:
    args = build_parser(protocol).parse_args()
    input_path = args.input.resolve()
    if input_path.is_dir():
        input_path = input_path / f"{protocol}_output.txt"
    run_dir = input_path.parent
    server_logs_dir = (args.server_logs_dir or run_dir / "logs" / "servers").resolve()
    failure_spec = (args.failure_spec or run_dir / "failure_spec.xml").resolve()

    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")
    if not server_logs_dir.is_dir():
        raise SystemExit(f"Server log directory not found: {server_logs_dir} (run with --log)")
    if not failure_spec.is_file():
        raise SystemExit(f"Failure specification not found: {failure_spec}")
    if args.skip_warmup_lines < 0:
        raise SystemExit("--skip-warmup-lines must be non-negative.")
    if args.smooth_window < 1:
        raise SystemExit("--smooth-window must be >= 1.")
    if args.latency_cap_ms is not None and args.latency_cap_ms <= 0:
        raise SystemExit("--latency-cap-ms must be positive.")

    try:
        benchmark_start, samples = parse_monitor_series(input_path, args.skip_warmup_lines)
        phase_spans = load_phase_spans(failure_spec)
        parsed_logs = [parse_server_log(path) for path in sorted(server_logs_dir.glob("server_*.log"))]
    except (OSError, ET.ParseError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if not parsed_logs:
        raise SystemExit(f"No server_N.log files found in {server_logs_dir}")

    sample_times = [sample.timestamp for sample in samples]
    elapsed_s = [(t - benchmark_start).total_seconds() for t in sample_times]
    throughput = [sample.succ for sample in samples]
    plotted_throughput = trailing_average(throughput, args.smooth_window)

    timeout_series = {}
    view_series = {}
    for parsed in parsed_logs:
        if parsed.initial_timeout_ms is None:
            raise SystemExit(f"server_{parsed.node_id}.log has no 'loaded protocol' line with election_timeout_ms")
        timeout_series[parsed.node_id] = timeout_per_sample(
            parsed.timeout_events, sample_times, benchmark_start, parsed.initial_timeout_ms
        )
        view_series[parsed.node_id] = (
            [(v.timestamp - benchmark_start).total_seconds() for v in parsed.views],
            [v.view for v in parsed.views],
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required. Install it with `pip install matplotlib`.") from exc

    fig, (throughput_ax, latency_ax, view_ax, timeout_ax) = plt.subplots(
        4, 1, figsize=(14, 12), sharex=True, gridspec_kw={"height_ratios": (2, 1.2, 1, 1)}
    )
    axes = (throughput_ax, latency_ax, view_ax, timeout_ax)
    add_phase_backgrounds(axes, phase_spans, elapsed_s[0], elapsed_s[-1])

    if args.smooth_window > 1:
        throughput_ax.plot(elapsed_s, throughput, color="tab:blue", alpha=0.25, linewidth=0.8, label="Raw succ", zorder=2)
        throughput_ax.plot(elapsed_s, plotted_throughput, color="tab:blue", linewidth=1.8,
                           label=f"{args.smooth_window}-sample trailing average", zorder=3)
        throughput_ax.legend(loc="upper right")
    else:
        throughput_ax.plot(elapsed_s, throughput, color="tab:blue", linewidth=1.2, zorder=2)
    throughput_ax.set_title(f"CppBedrock {protocol.upper()} Throughput, Views, and Applied Timeout ({run_dir.name})")
    throughput_ax.set_ylabel("Successful requests/s")

    p50_series = [s.p50_ms if s.p50_ms > 0 else float("nan") for s in samples]
    p95_series = [s.p95_ms if s.p95_ms > 0 else float("nan") for s in samples]
    if args.latency_cap_ms is not None:
        p50_series = [min(v, args.latency_cap_ms) for v in p50_series]
        p95_series = [min(v, args.latency_cap_ms) for v in p95_series]
        latency_ax.set_ylim(0, args.latency_cap_ms)
    latency_ax.plot(elapsed_s, p95_series, color="tab:orange", linewidth=1.0, label="p95", zorder=2)
    latency_ax.plot(elapsed_s, p50_series, color="tab:green", linewidth=1.2, label="p50", zorder=3)
    latency_ax.set_ylabel("Client latency (ms)")
    latency_ax.legend(loc="upper right", ncol=2)

    line_styles = ("-", "--", "-.", ":")
    markers = ("o", "s", "^", "D", "v", "P")
    marker_interval = max(1, len(elapsed_s) // 30)
    for series_index, node_id in enumerate(sorted(view_series)):
        xs, ys = view_series[node_id]
        view_ax.step(xs, ys, where="post", linewidth=1.3, linestyle=line_styles[series_index % len(line_styles)],
                     label=f"Server {node_id}", zorder=2 + series_index)
    view_ax.set_ylabel("View")
    view_ax.legend(loc="upper left", ncol=min(4, len(view_series)))

    for series_index, node_id in enumerate(sorted(timeout_series)):
        timeout_ax.step(elapsed_s, timeout_series[node_id], where="post", linewidth=1.5,
                        linestyle=line_styles[series_index % len(line_styles)],
                        marker=markers[series_index % len(markers)], markevery=marker_interval, markersize=3.5,
                        label=f"Server {node_id}", zorder=2 + series_index)
    timeout_ax.set_ylabel("Election timeout (ms)")
    timeout_ax.set_xlabel("Elapsed time since benchmark start (s)")
    timeout_ax.legend(loc="upper right", ncol=min(4, len(timeout_series)))

    for axis in axes:
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
    raise SystemExit(main("pbft"))
