#!/usr/bin/env python3
"""Plot the timeout decision taken per timer, per agent, across phases.

Reads agent0..N.csv from a dataset directory (the per-replica learning-agent
CSVs produced by run_tendermint.sh --dataset) and renders a grid:
rows = agents, cols = timers (propose / prevote / precommit). Each subplot
shows the chosen timeout (150/400/900 ms) over episodes, with the injection
phase shaded in the background.

Usage:
    python3 plot_timeout_decisions.py [DATA_DIR] [--out PNG]
"""
import argparse
import csv
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Publication styling.
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,          # embed TrueType so text stays editable
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

LINE_COLOR = "#1f3b73"

TIMERS = [
    ("propose_timeout_milliseconds_used", "propose"),
    ("prevote_timeout_milliseconds_used", "prevote"),
    ("precommit_timeout_milliseconds_used", "precommit"),
]
TIMEOUT_LEVELS = [150, 400, 900]

# phase index -> (label, shade color)
PHASE_INFO = {
    -1: ("warm-up", "#dddddd"),
    0: ("base (no fault)", "#c7e9c0"),
    1: ("small propose delay", "#fdd0a2"),
    2: ("large propose delay", "#fdae6b"),
    3: ("small net delay", "#c6dbef"),
    4: ("net + propose delay", "#9ecae1"),
    5: ("large net delay", "#6baed6"),
}


def agent_index(path):
    m = re.search(r"agent(\d+)\.csv$", path)
    return int(m.group(1)) if m else 0


def load_decisions(path):
    """One decision row per episode: (episode, phase, {timer: ms})."""
    by_ep = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                ep = int(r["reward_episode"])
            except (KeyError, ValueError):
                continue
            if ep in by_ep:
                continue  # both matched-flag rows carry the same decision
            try:
                row = {
                    "episode": ep,
                    "phase": int(r["phase"]),
                    "vals": {name: int(float(r[col])) for col, name in TIMERS},
                }
            except (KeyError, ValueError):
                continue
            by_ep[ep] = row
    return [by_ep[k] for k in sorted(by_ep)]


def phase_spans(rows):
    """Contiguous (start_ep, end_ep, phase) runs over the decision sequence."""
    spans = []
    for row in rows:
        ep, ph = row["episode"], row["phase"]
        if spans and spans[-1][2] == ph and ep == spans[-1][1] + 1:
            spans[-1][1] = ep
        else:
            spans.append([ep, ep, ph])
    return spans


def main():
    ap = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_dst")
    ap.add_argument("data_dir", nargs="?", default=default_dir,
                    help="Directory containing agentN.csv (default: ./tmp_dst)")
    ap.add_argument("--out", default=None,
                    help="Output PNG path (combined mode only)")
    ap.add_argument("--split", action="store_true",
                    help="Write one figure per timeout type instead of a combined grid")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "agent*.csv")), key=agent_index)
    if not files:
        raise SystemExit(f"No agent*.csv found in {args.data_dir}")

    agents = [(agent_index(f), load_decisions(f)) for f in files]
    agents = [(i, rows) for i, rows in agents if rows]
    n_agents = len(agents)

    def draw_panel(ax, rows, tname, seen_phases):
        for s0, s1, ph in phase_spans(rows):
            color = PHASE_INFO.get(ph, ("?", "#eeeeee"))[1]
            ax.axvspan(s0 - 0.5, s1 + 0.5, color=color, alpha=0.55, lw=0)
            seen_phases.add(ph)
        eps = [row["episode"] for row in rows]
        ys = [row["vals"][tname] for row in rows]
        ax.step(eps, ys, where="post", color=LINE_COLOR, lw=1.1)
        ax.set_yticks(TIMEOUT_LEVELS)
        ax.set_ylim(60, 990)
        ax.margins(x=0.01)
        ax.grid(axis="y", ls=":", lw=0.6, alpha=0.5)
        ax.tick_params(length=3)

    def phase_legend(seen_phases):
        return [
            Patch(facecolor=PHASE_INFO[p][1], alpha=0.6,
                  label=f"phase {p}: {PHASE_INFO[p][0]}")
            for p in sorted(seen_phases) if p in PHASE_INFO
        ]

    outputs = []

    def save_both(fig, stem):
        paths = []
        for ext in ("pdf", "png"):
            p = f"{stem}.{ext}"
            fig.savefig(p)
            paths.append(p)
        plt.close(fig)
        return paths

    if args.split:
        # One figure per timeout type; all agents stacked as rows.
        for _, tname in TIMERS:
            seen = set()
            fig, axes = plt.subplots(
                n_agents, 1, figsize=(11, 2.2 * n_agents),
                sharex=True, sharey=True, squeeze=False,
            )
            for r, (aid, rows) in enumerate(agents):
                ax = axes[r][0]
                draw_panel(ax, rows, tname, seen)
                ax.set_ylabel(f"agent {aid}\ntimeout (ms)", fontsize=10)
                if r == n_agents - 1:
                    ax.set_xlabel("episode")
            handles = phase_legend(seen)
            fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)),
                       frameon=False, bbox_to_anchor=(0.5, 0.0))
            fig.tight_layout(rect=(0, 0.05, 1, 1.0))
            outputs += save_both(fig, os.path.join(args.data_dir, f"timeout_decisions_{tname}"))
    else:
        # Combined grid: rows = agents, cols = timers.
        seen = set()
        fig, axes = plt.subplots(
            n_agents, len(TIMERS),
            figsize=(5.2 * len(TIMERS), 2.4 * n_agents),
            sharex=True, sharey=True, squeeze=False,
        )
        for r, (aid, rows) in enumerate(agents):
            for c, (_, tname) in enumerate(TIMERS):
                ax = axes[r][c]
                draw_panel(ax, rows, tname, seen)
                if r == 0:
                    ax.set_title(f"{tname} timeout", fontsize=11, fontweight="bold")
                if c == 0:
                    ax.set_ylabel(f"agent {aid}\ntimeout (ms)", fontsize=10)
                if r == n_agents - 1:
                    ax.set_xlabel("episode")
        handles = phase_legend(seen)
        fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)),
                   frameon=False, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.04, 1, 1.0))
        if args.out:
            fig.savefig(args.out)
            plt.close(fig)
            outputs.append(args.out)
        else:
            outputs += save_both(fig, os.path.join(args.data_dir, "timeout_decisions_by_phase"))

    n_dec = sum(len(r) for _, r in agents)
    for out in outputs:
        print(f"wrote {out}")
    print(f"({n_agents} agents, {n_dec} decisions)")


if __name__ == "__main__":
    main()
