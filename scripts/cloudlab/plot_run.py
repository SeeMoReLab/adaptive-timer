#!/usr/bin/env python3
"""Plot one CloudLab run of any protocol: client throughput and latency, and every
replica's applied timeouts, aligned on the benchmark start and shaded by failure phase.

Every protocol logs the same three lines; only the prefix before them differs:
  <protocol>_output.txt (the client):
    Waiting until benchmark start unix ms <N>
    [client <ISO-8601 ts>] Monitor duration=<s>s ... succ=<n> ... p50=<ms> p95=<ms> ...
  logs/servers/server_N.log (each replica), once at startup and on every change:
    [... <ISO-8601 ts> ...] applied timeouts: <timer>_ms=<v> [<timer>_ms=<v> ...]
Each <timer>_ms key gets its own panel. CppBedrock replicas also print a per-second
"Stats view=<v>" line, drawn as a view panel when present.

Tendermint's latency is the committing block's time minus the transaction's creation
time, not a client reply as for the other protocols.
"""

import argparse
import gzip
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

ISO_TS = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})"
TIMESTAMP_RE = re.compile(ISO_TS)
START_RE = re.compile(r"Waiting until benchmark start unix ms (\d+)")
MONITOR_RE = re.compile(rf"\[client ({ISO_TS})\] Monitor (duration=.*)")
KV_RE = re.compile(r"([A-Za-z0-9_]+)=(\S+)")
TIMEOUT_MARKER = "applied timeouts:"
SERVER_FILE_RE = re.compile(r"server_(\d+)\.log(?:\.gz)?$")  # the harnesses pull replica logs gzipped
PHASE_COLORS = ("#dceaf7", "#fff0c2", "#e6d9f2", "#d8ead2", "#f4d1d1", "#d7e5e5")
LINE_STYLES = ("-", "--", "-.", ":")


@dataclass(frozen=True)
class MonitorSample:
    elapsed_s: float
    tps: float
    # 0 when nothing finished in the interval; plotted as a gap.
    p50_ms: float
    p95_ms: float


@dataclass(frozen=True)
class ReplicaLog:
    node_id: int
    # timer key -> [(elapsed_s, ms)], in log order.
    timeouts: dict[str, list[tuple[float, float]]]
    views: list[tuple[float, int]]


@dataclass(frozen=True)
class PhaseSpan:
    label: str
    start_s: float
    end_s: Optional[float]


def parse_timestamp(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def parse_client_output(path: Path, skip_lines: int) -> tuple[datetime, list[MonitorSample]]:
    """Parse the benchmark start and the Monitor samples, as seconds since that start."""
    start = None
    samples = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            start_match = START_RE.search(line)
            if start_match and start is None:
                start = datetime.fromtimestamp(int(start_match.group(1)) / 1000.0, tz=timezone.utc)
                continue
            monitor_match = MONITOR_RE.search(line)
            if not monitor_match:
                continue
            if start is None:
                raise ValueError(f"{path}:{line_number}: Monitor line before the benchmark start line")
            values = dict(KV_RE.findall(monitor_match.group(2)))
            try:
                duration_s = float(values["duration"].rstrip("s"))
                succ = float(values["succ"])
                p50_ms = float(values["p50"])
                p95_ms = float(values["p95"])
            except KeyError as exc:
                raise ValueError(f"{path}:{line_number}: Monitor line has no {exc} field") from exc
            if duration_s <= 0:
                raise ValueError(f"{path}:{line_number}: Monitor duration {duration_s}s is not positive")
            elapsed_s = (parse_timestamp(monitor_match.group(1)) - start).total_seconds()
            samples.append(MonitorSample(elapsed_s, succ / duration_s, p50_ms, p95_ms))

    if start is None:
        raise ValueError(f"No 'Waiting until benchmark start unix ms' line in {path}")
    if skip_lines >= len(samples):
        raise ValueError(f"Only {len(samples)} Monitor lines in {path}, cannot skip {skip_lines}")
    return start, samples[skip_lines:]


def parse_replica_log(path: Path, start: datetime) -> ReplicaLog:
    """Parse the applied-timeout changes and view samples of one replica."""
    name_match = SERVER_FILE_RE.search(path.name)
    if not name_match:
        raise ValueError(f"Cannot determine the replica ID from {path.name}")
    timeouts: dict[str, list[tuple[float, float]]] = {}
    views = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            is_timeout = TIMEOUT_MARKER in line
            is_stats = " Stats " in line and " view=" in line
            if not is_timeout and not is_stats:
                continue
            timestamp_match = TIMESTAMP_RE.search(line)
            if timestamp_match is None:
                raise ValueError(f"{path}:{line_number}: no ISO-8601 timestamp")
            elapsed_s = (parse_timestamp(timestamp_match.group(0)) - start).total_seconds()
            if is_stats:
                views.append((elapsed_s, int(dict(KV_RE.findall(line))["view"])))
                continue
            values = {
                key: float(value)
                for key, value in KV_RE.findall(line.split(TIMEOUT_MARKER, 1)[1])
                if key.endswith("_ms")
            }
            if not values:
                raise ValueError(f"{path}:{line_number}: '{TIMEOUT_MARKER}' line has no *_ms value")
            if timeouts and set(values) != set(timeouts):
                raise ValueError(
                    f"{path}:{line_number}: timers {sorted(values)} differ from earlier lines' {sorted(timeouts)}"
                )
            for key, value in values.items():
                timeouts.setdefault(key, []).append((elapsed_s, value))
    if not timeouts:
        raise ValueError(f"No '{TIMEOUT_MARKER}' line in {path}")
    return ReplicaLog(int(name_match.group(1)), timeouts, views)


def step_series(
    events: Sequence[tuple[float, float]], x_min: float, x_max: float, what: str
) -> tuple[list[float], list[float]]:
    """Step series over [x_min, x_max] from (time, value) change events."""
    before = [value for time_s, value in events if time_s <= x_min]
    if not before:
        raise ValueError(f"{what}: no value logged before the first plotted sample at t={x_min:g}s")
    xs, ys = [x_min], [before[-1]]
    for time_s, value in events:
        if x_min < time_s <= x_max:
            xs.append(time_s)
            ys.append(value)
    xs.append(x_max)
    ys.append(ys[-1])
    return xs, ys


def optional_float(parent: ET.Element, path: str) -> Optional[float]:
    node = parent.find(path)
    if node is None or node.text is None or not node.text.strip():
        return None
    return float(node.text.strip())


def phase_label(node: ET.Element, index: int) -> str:
    """Name a phase by its index and the faults its spec injects."""
    name = node.findtext("name")
    if name and name.strip():
        return name.strip()

    parts = []
    delay_ms = optional_float(node, "./network/global/delayMs")
    if delay_ms:
        burst_ms = optional_float(node, "./network/global/burstDurationMs") or 0.0
        interval_ms = optional_float(node, "./network/global/burstIntervalMs") or 0.0
        parts.append(f"+{delay_ms:g} ms" + (" burst" if 0 < burst_ms < interval_ms else ""))
    for source in node.findall("./network/sources/source"):
        source_delay_ms = optional_float(source, "delayMs")
        if source_delay_ms:
            parts.append(f"straggler {(source.findtext('src') or '?').strip()} {source_delay_ms:g} ms")
    # Protocol sections: <protocol>/<fault>/replicas/replica with an absolute delay or
    # one relative to the current timeout.
    for section in node:
        if section.tag == "network":
            continue
        for fault in section:
            for replica in fault.findall("./replicas/replica"):
                replica_id = (replica.findtext("id") or "?").strip()
                absolute_ms = optional_float(replica, "delayMs")
                relative_ms = optional_float(replica, "delayRelativeToTimeoutMs")
                if absolute_ms:
                    parts.append(f"{fault.tag} {replica_id} {absolute_ms:g} ms")
                elif relative_ms is not None:
                    parts.append(f"{fault.tag} {replica_id} T{relative_ms:+g} ms")
            fault_delay_ms = optional_float(fault, "delayMs")
            if fault_delay_ms:
                parts.append(f"{fault.tag} {fault_delay_ms:g} ms")
    return "\n".join([f"p{index}"] + parts)


def load_phase_spans(path: Path) -> list[PhaseSpan]:
    """Load the warm-up and fault phase boundaries from failure_spec.xml."""
    root = ET.parse(path).getroot()
    warmup_ms = optional_float(root, "warmUpTimeMs")
    warmup_s = warmup_ms / 1000.0 if warmup_ms is not None else optional_float(root, "warmUpTime") or 0.0
    phase_nodes = root.findall("./phases/phase")
    if not phase_nodes:
        raise ValueError(f"No failure phases in {path}")

    starts = []
    for index, node in enumerate(phase_nodes):
        start_ms = optional_float(node, "atTimeMs")
        start_s = start_ms / 1000.0 if start_ms is not None else optional_float(node, "atTime")
        if start_s is None or start_s < 0:
            raise ValueError(f"Failure phase {index} in {path} has no non-negative atTime")
        starts.append((start_s, index, node))
    starts.sort(key=lambda item: (item[0], item[1]))

    spans = [PhaseSpan("Warm-up", 0.0, warmup_s)] if warmup_s > 0 else []
    for position, (start_s, _index, node) in enumerate(starts):
        end_s = warmup_s + starts[position + 1][0] if position + 1 < len(starts) else None
        spans.append(PhaseSpan(phase_label(node, position), warmup_s + start_s, end_s))
    return spans


def trailing_average(values: list[float], window: int) -> list[float]:
    smoothed = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        smoothed.append(running_sum / min(index + 1, window))
    return smoothed


def add_phase_backgrounds(axes, spans: Sequence[PhaseSpan], x_min: float, x_max: float) -> None:
    """Shade every panel and label the visible phases on the top one."""
    for index, span in enumerate(spans):
        start_s = max(x_min, span.start_s)
        end_s = min(x_max, span.end_s if span.end_s is not None else x_max)
        if end_s <= start_s:
            continue
        for axis in axes:
            axis.axvspan(start_s, end_s, color=PHASE_COLORS[index % len(PHASE_COLORS)],
                         alpha=0.55, linewidth=0, zorder=0)
        axes[0].text((start_s + end_s) / 2, 0.98, span.label, ha="center", va="top", fontsize=8,
                     color="#444444", transform=axes[0].get_xaxis_transform())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True,
                        help="A run directory, or the <protocol>_output.txt inside it.")
    parser.add_argument("--server-logs-dir", type=Path, help="Default: <run>/logs/servers.")
    parser.add_argument("--failure-spec", type=Path, help="Default: <run>/failure_spec.xml.")
    parser.add_argument("--output", type=Path, help="Image path. If omitted, shows an interactive window.")
    parser.add_argument("--skip-warmup-lines", type=int, default=12,
                        help="Initial Monitor lines to omit. Default: 12.")
    parser.add_argument("--smooth-window", type=int, default=1,
                        help="Trailing moving-average window in samples. Default: 1 (none).")
    parser.add_argument("--latency-cap-ms", type=float,
                        help="Cap the latency panel at this many ms; larger samples are drawn at the cap.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = args.input.resolve()
    if input_path.is_dir():
        outputs = sorted(input_path.glob("*_output.txt"))
        if len(outputs) != 1:
            raise SystemExit(f"Expected one *_output.txt in {input_path}, found {[p.name for p in outputs]}")
        input_path = outputs[0]
    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")
    protocol = input_path.name.removesuffix("_output.txt")
    run_dir = input_path.parent
    server_logs_dir = (args.server_logs_dir or run_dir / "logs" / "servers").resolve()
    failure_spec = (args.failure_spec or run_dir / "failure_spec.xml").resolve()
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
        start, samples = parse_client_output(input_path, args.skip_warmup_lines)
        phase_spans = load_phase_spans(failure_spec)
        replica_paths = sorted([p for p in server_logs_dir.glob("server_*.log*") if SERVER_FILE_RE.search(p.name)],
                               key=lambda path: int(SERVER_FILE_RE.search(path.name).group(1)))
        if not replica_paths:
            raise ValueError(f"No server_N.log files in {server_logs_dir}")
        replicas = [parse_replica_log(path, start) for path in replica_paths]
        # Panels follow the order the replica logs the timers in.
        timer_keys = list(replicas[0].timeouts)
        for replica in replicas:
            if set(replica.timeouts) != set(timer_keys):
                raise ValueError(f"server_{replica.node_id}.log logs timers {list(replica.timeouts)}, "
                                 f"server_{replicas[0].node_id}.log logs {timer_keys}")
        x_min, x_max = samples[0].elapsed_s, samples[-1].elapsed_s
        timeout_series = {
            key: {r.node_id: step_series(r.timeouts[key], x_min, x_max, f"server_{r.node_id} {key}")
                  for r in replicas}
            for key in timer_keys
        }
    except (OSError, ET.ParseError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    has_views = any(replica.views for replica in replicas)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required. Install it with `pip install matplotlib`.") from exc

    height_ratios = [2, 1.2] + ([1] if has_views else []) + [1] * len(timer_keys)
    fig, axes = plt.subplots(len(height_ratios), 1, figsize=(14, 2.6 * sum(height_ratios)), sharex=True,
                             gridspec_kw={"height_ratios": height_ratios})
    throughput_ax, latency_ax = axes[0], axes[1]
    add_phase_backgrounds(axes, phase_spans, x_min, x_max)

    elapsed_s = [s.elapsed_s for s in samples]
    throughput = [s.tps for s in samples]
    if args.smooth_window > 1:
        throughput_ax.plot(elapsed_s, throughput, color="tab:blue", alpha=0.25, linewidth=0.8,
                           label="Raw", zorder=2)
        throughput_ax.plot(elapsed_s, trailing_average(throughput, args.smooth_window), color="tab:blue",
                           linewidth=1.8, label=f"{args.smooth_window}-sample trailing average", zorder=3)
        throughput_ax.legend(loc="lower right")
    else:
        throughput_ax.plot(elapsed_s, throughput, color="tab:blue", linewidth=1.2, zorder=2)
    throughput_ax.set_title(f"{protocol} throughput, latency and applied timeouts ({run_dir.name})")
    throughput_ax.set_ylabel("Committed requests/s")

    p50 = [s.p50_ms if s.p50_ms > 0 else float("nan") for s in samples]
    p95 = [s.p95_ms if s.p95_ms > 0 else float("nan") for s in samples]
    if args.latency_cap_ms is not None:
        p50 = [min(v, args.latency_cap_ms) for v in p50]
        p95 = [min(v, args.latency_cap_ms) for v in p95]
        latency_ax.set_ylim(0, args.latency_cap_ms)
    latency_ax.plot(elapsed_s, p95, color="tab:orange", linewidth=1.0, label="p95", zorder=2)
    latency_ax.plot(elapsed_s, p50, color="tab:green", linewidth=1.2, label="p50", zorder=3)
    latency_ax.set_ylabel("Latency (ms)")
    latency_ax.legend(loc="upper right", ncol=2)

    panel = 2
    if has_views:
        view_ax = axes[panel]
        panel += 1
        for index, replica in enumerate(replicas):
            if replica.views:
                view_ax.step([t for t, _ in replica.views], [v for _, v in replica.views], where="post",
                             linewidth=1.3, linestyle=LINE_STYLES[index % len(LINE_STYLES)],
                             label=f"Server {replica.node_id}", zorder=2 + index)
        view_ax.set_ylabel("View")
        view_ax.legend(loc="upper left", ncol=len(replicas))

    for key in timer_keys:
        timeout_ax = axes[panel]
        panel += 1
        for index, (node_id, (xs, ys)) in enumerate(sorted(timeout_series[key].items())):
            timeout_ax.step(xs, ys, where="post", linewidth=1.5, linestyle=LINE_STYLES[index % len(LINE_STYLES)],
                            label=f"Server {node_id}", zorder=2 + index)
        timeout_ax.set_ylabel(key.removesuffix("_ms").replace("_", " ") + " (ms)")
        timeout_ax.margins(y=0.25)
        timeout_ax.legend(loc="upper right", ncol=len(replicas))

    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.35, zorder=1)
        axis.set_xlim(x_min, x_max)
    axes[-1].set_xlabel("Elapsed time since benchmark start (s)")
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
