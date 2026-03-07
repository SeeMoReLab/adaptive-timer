#!/usr/bin/env python3
"""Plot SmallBank monitor metrics from one or more output.txt files."""

import argparse
import math
import re
from pathlib import Path

KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")


def parse_monitor_series(file_path: Path):
    rows = []

    with file_path.open(encoding="utf-8") as handle:
        for line in handle:
            if "Monitor" not in line:
                continue

            values = dict(KV_RE.findall(line))
            required = ("trxs", "tps", "avg_ms", "p95", "p99", "err")
            if not all(key in values for key in required):
                continue

            rows.append(
                {
                    "trxs": int(values["trxs"]),
                    "tps": float(values["tps"]),
                    "avg_ms": float(values["avg_ms"]),
                    "p95": float(values["p95"]),
                    "p99": float(values["p99"]),
                    "err": int(values["err"]),
                }
            )

    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot SmallBank monitor metrics from one or more output.txt files."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="One or more output.txt file paths.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help=(
            "Alternative input mode: read all .txt files in this directory and "
            "keep only files with at least one valid '-- Monitor' line."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional image output path. If omitted, opens an interactive window.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Warm-up samples to trim from the start of each input (rows[warmup:]).",
    )
    parser.add_argument(
        "--only",
        choices=("tps", "p95"),
        help="Plot only one metric: tps or p95. Default plots all metrics.",
    )
    return parser


def _make_unique_labels(paths):
    labels = []
    seen = {}
    for path in paths:
        label = path.stem if path.stem else path.name
        if label in seen:
            seen[label] += 1
            label = f"{label}#{seen[label]}"
        else:
            seen[label] = 1
        labels.append(label)
    return labels


def _trim_warmup(rows, warmup: int):
    if warmup <= 0:
        return rows
    return rows[warmup:]


def _visible_sum(values, left_x: float, right_x: float):
    if not values:
        return 0

    start = max(0, math.ceil(left_x))
    end = min(len(values) - 1, math.floor(right_x))
    if end < start:
        return 0
    return int(sum(values[start : end + 1]))


def _avg_nonzero(values):
    nonzero = [value for value in values if value != 0]
    if not nonzero:
        return None
    return float(sum(nonzero) / len(nonzero))


def _visible_avg(values, left_x: float, right_x: float):
    if not values:
        return None

    start = max(0, math.ceil(left_x))
    end = min(len(values) - 1, math.floor(right_x))
    if end < start:
        return None

    window = values[start : end + 1]
    return _avg_nonzero(window)


def _collect_input_paths(parser: argparse.ArgumentParser, args) -> list[Path]:
    file_inputs = [path.expanduser().resolve() for path in args.inputs]
    dir_input = args.input_dir.expanduser().resolve() if args.input_dir else None

    if dir_input and file_inputs:
        parser.error("Provide either positional inputs or --input-dir, not both.")

    if dir_input:
        if not dir_input.exists():
            parser.error(f"Input directory not found: {dir_input}")
        if not dir_input.is_dir():
            parser.error(f"--input-dir is not a directory: {dir_input}")

        txt_files = sorted(
            path
            for path in dir_input.iterdir()
            if path.is_file() and path.suffix.lower() == ".txt"
        )
        if not txt_files:
            parser.error(f"No .txt files found in input directory: {dir_input}")

        valid_files = [path for path in txt_files if parse_monitor_series(path)]
        if not valid_files:
            parser.error(
                "No .txt files with at least one valid '-- Monitor' line found in: "
                f"{dir_input}"
            )
        return valid_files

    if not file_inputs:
        parser.error("Provide at least one input file or use --input-dir.")

    for path in file_inputs:
        if not path.exists():
            parser.error(f"Input file not found: {path}")

    return file_inputs


def _format_avg(value):
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: matplotlib. Install with:\n"
            "  python3 -m pip install --user matplotlib"
        ) from exc

    input_paths = _collect_input_paths(parser, args)
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    warmup = args.warmup

    labels = _make_unique_labels(input_paths)
    series = []
    for path, label in zip(input_paths, labels):
        rows = parse_monitor_series(path)
        rows = _trim_warmup(rows, warmup)
        if not rows:
            parser.error(
                f"No monitor lines parsed from: {path} "
                f"(after warm-up slice rows[{warmup}:])"
            )
        series.append((label, path, rows))

    if args.only == "tps":
        fig, ax_tps = plt.subplots(1, 1, figsize=(12, 4))
        ax_lat = None
        ax_err = None
    elif args.only == "p95":
        fig, ax_lat = plt.subplots(1, 1, figsize=(12, 4))
        ax_tps = None
        ax_err = None
    else:
        fig, (ax_tps, ax_lat, ax_err) = plt.subplots(
            3, 1, figsize=(12, 8), sharex=False, gridspec_kw={"height_ratios": [2, 2, 1]}
        )

    tps_meta = []
    p95_meta = []
    for label, _, rows in series:
        t = list(range(len(rows)))
        trxs = [r["trxs"] for r in rows]
        tps = [r["tps"] for r in rows]
        p95 = [r["p95"] for r in rows]
        err = [r["err"] for r in rows]
        total_trxs = int(sum(trxs))

        if ax_tps is not None:
            (line_tps,) = ax_tps.plot(
                t,
                tps,
                linewidth=1.2,
                label=f"{label} total={total_trxs}",
            )
            tps_meta.append(
                {
                    "line": line_tps,
                    "label": label,
                    "trxs": trxs,
                    "total": total_trxs,
                }
            )

        if ax_lat is not None:
            avg_p95 = _avg_nonzero(p95)
            (line_p95,) = ax_lat.plot(
                t,
                p95,
                linewidth=1.2,
                label=f"{label} avg={_format_avg(avg_p95)}",
            )
            p95_meta.append(
                {
                    "line": line_p95,
                    "label": label,
                    "p95": p95,
                    "avg": avg_p95,
                }
            )

        if ax_err is not None:
            ax_err.plot(t, err, linewidth=1.1, label=label)

    if ax_tps is not None:
        ax_tps.set_title("Throughput")
        ax_tps.set_ylabel("TPS")
        ax_tps.set_xlabel("Elapsed time (s)")
        ax_tps.grid(True, linestyle="--", alpha=0.4)
        legend_refresh_in_progress = False

        def refresh_tps_legend():
            nonlocal legend_refresh_in_progress
            if legend_refresh_in_progress:
                return
            legend_refresh_in_progress = True
            try:
                left_x, right_x = ax_tps.get_xlim()
                for meta in tps_meta:
                    visible_total = _visible_sum(meta["trxs"], left_x, right_x)
                    meta["line"].set_label(
                        f"{meta['label']} total={meta['total']} visible={visible_total}"
                    )
                ax_tps.legend(loc="upper right")
                fig.canvas.draw_idle()
            finally:
                legend_refresh_in_progress = False

        ax_tps.callbacks.connect("xlim_changed", lambda _ax: refresh_tps_legend())
        refresh_tps_legend()

    if ax_lat is not None:
        ax_lat.set_title("Latency (p95)")
        ax_lat.set_ylabel("p95 (ms)")
        ax_lat.set_xlabel("Elapsed time (s)")
        ax_lat.grid(True, linestyle="--", alpha=0.4)

        p95_legend_refresh_in_progress = False

        def refresh_p95_legend():
            nonlocal p95_legend_refresh_in_progress
            if p95_legend_refresh_in_progress:
                return
            p95_legend_refresh_in_progress = True
            try:
                left_x, right_x = ax_lat.get_xlim()
                for meta in p95_meta:
                    visible_avg = _visible_avg(meta["p95"], left_x, right_x)
                    meta["line"].set_label(
                        f"{meta['label']} avg={_format_avg(meta['avg'])} "
                        f"visible_avg={_format_avg(visible_avg)}"
                    )
                ax_lat.legend(loc="upper right")
                fig.canvas.draw_idle()
            finally:
                p95_legend_refresh_in_progress = False

        ax_lat.callbacks.connect("xlim_changed", lambda _ax: refresh_p95_legend())
        refresh_p95_legend()

    if ax_err is not None:
        ax_err.set_title("Errors")
        ax_err.set_ylabel("err")
        ax_err.set_xlabel("Elapsed time (s)")
        ax_err.grid(True, linestyle="--", alpha=0.4)
        ax_err.legend(loc="upper right")

    fig.suptitle(
        "SmallBank Monitor Comparison\n"
        f"inputs={len(series)}, warmup={warmup}, only={args.only or 'all'}"
    )
    fig.tight_layout()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"Saved plot to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
