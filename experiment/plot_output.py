#!/usr/bin/env python3
"""Plot SmallBank monitor metrics from one or more output.txt files."""

import argparse
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
            required = ("tps", "avg_ms", "p95", "p99", "err")
            if not all(key in values for key in required):
                continue

            rows.append(
                {
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
        nargs="+",
        type=Path,
        help="One or more output.txt file paths.",
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

    input_paths = [path.expanduser().resolve() for path in args.inputs]
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    warmup = args.warmup
    for path in input_paths:
        if not path.exists():
            parser.error(f"Input file not found: {path}")

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

    fig, (ax_tps, ax_lat, ax_err) = plt.subplots(
        3, 1, figsize=(12, 8), sharex=False, gridspec_kw={"height_ratios": [2, 2, 1]}
    )

    for label, _, rows in series:
        t = list(range(len(rows)))
        tps = [r["tps"] for r in rows]
        p95 = [r["p95"] for r in rows]
        err = [r["err"] for r in rows]

        ax_tps.plot(t, tps, linewidth=1.2, label=label)
        ax_lat.plot(t, p95, linewidth=1.2, label=label)
        ax_err.plot(t, err, linewidth=1.1, label=label)

    ax_tps.set_title("Throughput")
    ax_tps.set_ylabel("TPS")
    ax_tps.set_xlabel("Elapsed time (s)")
    ax_tps.grid(True, linestyle="--", alpha=0.4)
    ax_tps.legend(loc="upper right")

    ax_lat.set_title("Latency (p95)")
    ax_lat.set_ylabel("p95 (ms)")
    ax_lat.set_xlabel("Elapsed time (s)")
    ax_lat.grid(True, linestyle="--", alpha=0.4)
    ax_lat.legend(loc="upper right")

    ax_err.set_title("Errors")
    ax_err.set_ylabel("err")
    ax_err.set_xlabel("Elapsed time (s)")
    ax_err.grid(True, linestyle="--", alpha=0.4)
    ax_err.legend(loc="upper right")

    fig.suptitle(
        "SmallBank Monitor Comparison\n"
        f"inputs={len(series)}, warmup={warmup}"
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
