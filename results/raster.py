#!/usr/bin/env python3
"""Timeout raster: every run of a protocol as one horizontal strip over time, so all
configurations fit in one figure.

Top panel: the election timeout replica 0 had in force in each 20 s bin (placed in
time as plot.py does; for SBFT the panel shows T, a second panel W), on a log colour
scale, with a dot where that episode's report counted view changes (dot area grows
with the count). Bottom panel: each 20 s bin's client reward (plot.py buckets) minus
the oracle's in the same bin, so blue is ahead of the oracle and red behind. Rows are
the oracle, the heuristic, then the models sorted by mean R (table.py's judge).
Vertical lines mark the phase onsets, labelled from the protocol's "Phases:" line in
RESULTS.md.

Usage: results/raster.py --protocol pbft [--out results/plots/raster_pbft.png]
"""
import argparse
import csv
import re
import statistics
import sys

sys.dont_write_bytecode = True  # keep results/ free of __pycache__
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from plot import RESULTS, buckets, client_seconds, phase_onsets, report_times, select_runs, timeouts
from table import score

BIN_S = 20.0
GAP_LIM = 0.3  # reward gap colour limit; larger gaps saturate
TITLES = {"pbft": "PBFT", "sbft": "SBFT", "smartbft": "SmartBFT", "tendermint": "Tendermint",
          "hotstuff": "HotStuff", "autobahn": "Autobahn"}


def label(run):
    """'a26_discrete_gp_pre_floor_setB_<stamp>' -> 'discrete_gp pre + floor'."""
    name = re.sub(r"_set[AB]_\d{8}_\d{6}$", "", run.name).split("_", 1)[1]
    if name.startswith("oracle") or name == "heuristic":
        return name
    m = re.fullmatch(r"((?:discrete|continuous)_(?:gp|linucb|ppo))_(pre|untr)(_floor)?", name)
    if not m:
        raise ValueError(f"{run.name}: unrecognised configuration '{name}'")
    return f"{m.group(1)} {m.group(2)}" + (" + floor" if m.group(3) else "")


def phase_names(protocol):
    text = (RESULTS / "RESULTS.md").read_text()
    section = re.search(rf"^## {TITLES[protocol]}\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    m = section and re.search(r"^Phases: (.*)\.$", section.group(1), re.M)
    if not m:
        raise ValueError(f"RESULTS.md: no 'Phases:' line under '## {TITLES[protocol]}'")
    return [re.sub(r"^p\d+ ", "", p) for p in m.group(1).split(", ")]


def view_changes(run, t0, end_s):
    """[(t_s, count)] from replica 0's per-episode report, at the report time."""
    reports = report_times(run, t0, end_s)
    with (run / "agent0.csv").open() as f:
        return [(reports[int(r["episode"])], float(r["reward_view_change_count"]))
                for r in csv.DictReader(f) if int(r["episode"]) in reports]


def in_force(series, t):
    vals = [v for ts, v in series if ts <= t]
    return vals[-1] if vals else np.nan


def run_rows(run, protocol, edges):
    t0, seconds = client_seconds(run)
    end = max(r[0] for r in seconds)
    played = timeouts(run, t0, end)
    reward = {b[0]: b[3] for b in buckets(seconds, BIN_S, protocol)}
    mids = edges[:-1] + BIN_S / 2
    knobs = [[in_force(s, t) for t in mids] for s in played.values()]
    r = [reward.get(lo, np.nan) for lo in edges[:-1]]
    return dict(label=label(run), knobs=knobs, knob_names=list(played), reward=np.array(r),
                vc=view_changes(run, t0, end), mean=statistics.mean(p["R"] for p in score(run, protocol)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--protocol", required=True, choices=list(TITLES))
    ap.add_argument("--out", type=str, help="save to this PNG instead of opening a window")
    args = ap.parse_args()
    if args.out:
        matplotlib.use("Agg")

    runs = select_runs(args.protocol, "*")
    _, onsets = phase_onsets(runs[0], args.protocol)
    end = max(max(r[0] for r in client_seconds(run)[1]) for run in runs)
    edges = np.arange(0.0, end + BIN_S, BIN_S)
    rows = [run_rows(run, args.protocol, edges) for run in runs]
    oracle = [r for r in rows if r["label"].startswith("oracle")]
    if len(oracle) != 1:
        raise SystemExit(f"expected one oracle run under results/{args.protocol}, found {len(oracle)}")
    rest = [r for r in rows if r is not oracle[0]]
    heur = [r for r in rest if r["label"] == "heuristic"]
    models = sorted((r for r in rest if r["label"] != "heuristic"), key=lambda r: -r["mean"])
    rows = oracle + heur + models
    names = phase_names(args.protocol)
    if len(names) != len(onsets):
        raise SystemExit(f"RESULTS.md lists {len(names)} phases, the spec has {len(onsets)} onsets")

    knob_names = rows[0]["knob_names"]
    n_knob = len(knob_names)
    fig, axes = plt.subplots(n_knob + 1, 1, figsize=(15, 4.2 * (n_knob + 1) + 1.0), sharex=True,
                             gridspec_kw=dict(hspace=0.12))
    ylabels = [f"{r['label']}  {r['mean']:+.3f}" for r in rows]
    extent = (edges[0], edges[-1], len(rows) - 0.5, -0.5)

    for k, ax in enumerate(axes[:n_knob]):
        grid = np.array([r["knobs"][k] for r in rows])
        finite = grid[np.isfinite(grid)]
        im = ax.imshow(grid, aspect="auto", interpolation="nearest", extent=extent, cmap="viridis",
                       norm=LogNorm(vmin=finite.min(), vmax=finite.max()))
        cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.025)
        cb.set_label(f"{knob_names[k].replace('_', ' ')} (ms)")
        ticks = sorted(set(np.round(np.quantile(finite, [0, 0.5, 1]), -1)) | set(rows[0]["knobs"][k]) - {np.nan})
        cb.set_ticks([t for t in ticks if np.isfinite(t)])
        cb.set_ticklabels([f"{t:.0f}" for t in ticks if np.isfinite(t)])
        if k == 0:
            for i, r in enumerate(rows):
                pts = [(t, c) for t, c in r["vc"] if c > 0]
                if pts:
                    ax.scatter([p[0] for p in pts], [i] * len(pts), s=[6 + 10 * np.log2(p[1]) for p in pts],
                               c="red", edgecolors="white", linewidths=0.3, zorder=3)
        ax.set_title(f"{knob_names[k].replace('_', ' ')} in force per {BIN_S:g} s bin "
                     f"(red dots: episodes with view changes)", fontsize=10, loc="left", pad=22)

    gap = np.array([r["reward"] - rows[0]["reward"] for r in rows])
    ax = axes[-1]
    im = ax.imshow(gap, aspect="auto", interpolation="nearest", extent=extent, cmap="RdBu",
                   vmin=-GAP_LIM, vmax=GAP_LIM)
    cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.025, extend="both")
    cb.set_label("reward minus oracle")
    ax.set_title(f"client reward per {BIN_S:g} s bin minus the oracle's (blue ahead, red behind)",
                 fontsize=10, loc="left")
    ax.set_xlabel("time since benchmark start (s)")

    for ax in axes:
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(ylabels, fontsize=7.5, family="monospace")
        for x in onsets:
            ax.axvline(x, color="white", lw=1.2)
            ax.axvline(x, color="black", lw=0.5)
        ax.axhline(0.5, color="white", lw=1.5)
        ax.axhline(1.5 if heur else 0.5, color="white", lw=1.5)
    bounds = onsets + [edges[-1]]
    top = axes[0].secondary_xaxis("top")
    top.set_xticks([(lo + hi) / 2 for lo, hi in zip(bounds[:-1], bounds[1:])])
    top.set_xticklabels([f"p{i} {n}" for i, n in enumerate(names)], fontsize=8)
    top.tick_params(length=0)
    fig.suptitle(f"{TITLES[args.protocol]}: {len(rows)} runs, one per configuration", y=0.995)

    if args.out:
        fig.savefig(args.out, dpi=130, bbox_inches="tight")
        print(args.out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
