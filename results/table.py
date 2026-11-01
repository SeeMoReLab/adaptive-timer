#!/usr/bin/env python3
"""Per-phase throughput, p50 latency, reward and timeout tables for the runs in
results/<protocol>/<run>/, as Markdown, one row per run sorted by mean reward.

Each phase is scored from the client output as results/judge.py
does: from 20 s after its onset (the agent applies a switch at its next decision) to
the next onset, in 20 s chunks. Per chunk, tps is the mean of the per-second successful
commits, p50 the commit-weighted mean of the per-second p50s (a chunk with no commits
scores 10 s), and the reward is the learning agent's reward_tps_latency of the two.
A phase's tps, p50 and reward are the means over its chunks, so reward = tps / N -
p50 / 1,000 (N: learning_agent/reward_utils.py REWARD_NORMALIZER_TPS) holds in every cell; "mean" is the mean over phases. The timeout is the
median of replica 0's played values in force during the scored window (T/W for SBFT),
placed in time as plot.py does.

Usage: results/table.py --protocol sbft [--runs 'main_*;val_*'] [--ignore 'p0;p5']
(--ignore drops those phases from every table, the mean and the sort order.)
"""
import argparse
import re
import statistics
import sys

sys.dont_write_bytecode = True  # keep results/ free of __pycache__
from plot import client_seconds, phase_onsets, reward_normalizer_tps, reward_tps_latency, select_runs, timeouts

SKIP_S = 20
CHUNK_S = 20
EMPTY_P50_MS = 10000.0


def played(series, lo, hi):
    """Median of the values in force during [lo, hi) of one timeout series."""
    vals = [v for t, v in series if lo < t < hi]
    before = [v for t, v in series if t <= lo]
    if before:
        vals.append(before[-1])
    return statistics.median(vals) if vals else None


def score(run, protocol):
    """[{tps, p50, R, T}] per phase."""
    t0, seconds = client_seconds(run)
    end = max(r[0] for r in seconds) + 1.0
    _, onsets = phase_onsets(run, protocol)
    series = timeouts(run, t0, end - 1.0)
    bounds = onsets + [end]
    out = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        sec = [r for r in seconds if lo + SKIP_S <= r[0] < hi]
        chunks = []
        for k in range(0, len(sec) - CHUNK_S + 1, CHUNK_S):
            ch = sec[k:k + CHUNK_S]
            succ = sum(r[1] for r in ch)
            tps = succ / len(ch)
            p50 = sum(r[1] * r[2] for r in ch) / succ if succ else EMPTY_P50_MS
            chunks.append((tps, p50, reward_tps_latency(tps, p50, reward_normalizer_tps(protocol))))
        if not chunks:
            raise ValueError(f"{run}: phase at {lo:.0f} s has no complete {CHUNK_S} s chunk (run cut short?)")
        t = [played(v, lo + SKIP_S, hi) for v in series.values()]
        out.append(dict(tps=statistics.mean(c[0] for c in chunks), p50=statistics.mean(c[1] for c in chunks),
                        R=statistics.mean(c[2] for c in chunks),
                        T="/".join(f"{x:.0f}" for x in t if x is not None) or "-"))
    return out


def table(title, rows, labels, cell, mean=None):
    """Print a Markdown table padded so its columns line up in a terminal."""
    header = ["run"] + (["mean"] if mean else []) + labels
    body = [[name] + ([mean(ph)] if mean else []) + [cell(p) for p in ph] for name, ph in rows]
    widths = [max(len(r[i]) for r in [header] + body) for i in range(len(header))]

    def line(cells):
        # The run name is left-aligned, numbers right-aligned.
        return "| " + " | ".join(c.ljust(w) if i == 0 else c.rjust(w)
                                 for i, (c, w) in enumerate(zip(cells, widths))) + " |"

    print(f"\n### {title}\n")
    print(line(header))
    print("|" + "|".join(("-" * (w + 1) + ":") if i else ("-" * (w + 2)) for i, w in enumerate(widths)) + "|")
    for r in body:
        print(line(r))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--protocol", required=True,
                    choices=["pbft", "sbft", "smartbft", "tendermint", "hotstuff", "autobahn"])
    ap.add_argument("--runs", default="*",
                    help="';'-separated globs over run directory names, e.g. 'main_*;val_*' (default: all)")
    ap.add_argument("--ignore", default="",
                    help="';'-separated phases to leave out, e.g. 'p0;p5' (default: none)")
    args = ap.parse_args()
    ignore = set()
    for tok in filter(None, (t.strip() for t in args.ignore.split(";"))):
        m = re.fullmatch(r"p(\d+)", tok)
        if not m:
            raise SystemExit(f"--ignore: '{tok}' is not a phase like p0")
        ignore.add(int(m.group(1)))

    rows = [(re.sub(r"_\d{8}_\d{6}$", "", run.name), score(run, args.protocol))
            for run in select_runs(args.protocol, args.runs)]
    n = len(rows[0][1])
    if any(len(ph) != n for _, ph in rows):
        raise SystemExit("the selected runs have different phase counts; table runs of one spec together")
    missing = sorted(i for i in ignore if i >= n)
    if missing:
        raise SystemExit(f"--ignore: {', '.join(f'p{i}' for i in missing)} not in these runs (phases p0-p{n - 1})")
    keep = [i for i in range(n) if i not in ignore]
    if not keep:
        raise SystemExit("--ignore leaves no phase")
    labels = [f"p{i}" for i in keep]
    rows = [(name, [ph[i] for i in keep]) for name, ph in rows]
    rows.sort(key=lambda r: -statistics.mean(p["R"] for p in r[1]))
    print(f"## {args.protocol}: {len(rows)} runs")
    table("Throughput (tps)", rows, labels, lambda p: f"{p['tps']:.0f}",
          lambda ph: f"{statistics.mean(p['tps'] for p in ph):.0f}")
    table("p50 latency (ms)", rows, labels, lambda p: f"{p['p50']:.0f}",
          lambda ph: f"{statistics.mean(p['p50'] for p in ph):.0f}")
    table("Reward", rows, labels, lambda p: f"{p['R']:+.3f}",
          lambda ph: f"{statistics.mean(p['R'] for p in ph):+.3f}")
    table("Timeout played (ms)", rows, labels, lambda p: p["T"])


if __name__ == "__main__":
    main()
