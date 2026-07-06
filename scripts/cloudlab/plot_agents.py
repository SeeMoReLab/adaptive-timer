#!/usr/bin/env python3
"""Plot the timeout decision taken per timer, per agent, across phases.

Reads agent0..N.csv from a dataset directory (the per-replica learning-agent
CSVs) and renders a grid: rows = agents, cols = timers. Each subplot shows the
chosen timeout over episodes, with the injection phase shaded in the background.

The CSV schema is auto-detected:
  * tendermint  -> three timers (propose / prevote / precommit), levels 150/400/900
  * SmartBFT    -> one timer   (election),                       levels 100/700/1900

Usage:
    python3 plot_agents.py [DATA_DIR] [--out PNG] [--split]
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

# --- Schemas ------------------------------------------------------------
TENDERMINT_TIMERS = [
    ("propose_timeout_milliseconds_used", "propose"),
    ("prevote_timeout_milliseconds_used", "prevote"),
    ("precommit_timeout_milliseconds_used", "precommit"),
]
TENDERMINT_LEVELS = [150, 400, 900]

SMARTBFT_TIMERS = [("election_timeout_milliseconds_used", "election")]
SMARTBFT_LEVELS = [100, 700, 1900]


# Fault schedules: phase index -> (label, shade color).
# Tendermint (run_tendermint.sh):
TENDERMINT_PHASE_META = {
    -1: ("warm-up", "#dddddd"),
    0: ("base (no fault)", "#c7e9c0"),
    1: ("small propose delay", "#fdd0a2"),
    2: ("large propose delay", "#fdae6b"),
    3: ("small net delay", "#c6dbef"),
    4: ("net + propose delay", "#9ecae1"),
    5: ("large net delay", "#6baed6"),
}

# SmartBFT (config/pbft/failure_spec.xml). Colors group phases by fault regime
# so contiguous same-regime blocks read as one band; the legend dedupes by label.
_SB = {"base": "#c7e9c0", "p3": "#fdd0a2", "p9": "#e6805a",
       "n5": "#c6dbef", "n5p3": "#b3a2d4", "n15": "#6b8fd6"}
SMARTBFT_PHASE_META = {
    0: ("base (no fault)", _SB["base"]),
    **{p: ("leader propose +300ms", _SB["p3"]) for p in range(1, 6)},
    **{p: ("leader propose +900ms", _SB["p9"]) for p in range(6, 11)},
    11: ("net delay 50ms", _SB["n5"]),
    **{p: ("net 50ms + propose 300ms", _SB["n5p3"]) for p in range(12, 17)},
    17: ("net delay 150ms", _SB["n15"]),
    18: ("base (recovery)", _SB["base"]),
}


def detect_schema(path):
    """Return (timers, levels, phase_meta) by inspecting the CSV header."""
    with open(path, newline="") as fh:
        header = set(next(csv.reader(fh)))
    if ("election_timeout_milliseconds_used" in header
            and "propose_timeout_milliseconds_used" not in header):
        return SMARTBFT_TIMERS, SMARTBFT_LEVELS, SMARTBFT_PHASE_META
    return TENDERMINT_TIMERS, TENDERMINT_LEVELS, TENDERMINT_PHASE_META


def phase_color(ph, meta):
    if ph in meta:
        return meta[ph][1]
    return plt.get_cmap("tab20")(ph % 20)


def phase_label(ph, meta):
    if ph in meta:
        return f"phase {ph}: {meta[ph][0]}"
    return f"phase {ph}"


def agent_index(path):
    m = re.search(r"agent(\d+)\.csv$", path)
    return int(m.group(1)) if m else 0


def load_decisions(path, timers):
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
                    "vals": {name: int(float(r[col])) for col, name in timers},
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
    ap.add_argument("--width", type=float, default=None,
                    help="Total figure width in inches (default: scales with timer count)")
    ap.add_argument("--row-height", type=float, default=2.4,
                    help="Height in inches per agent row (default: 2.4)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "agent*.csv")), key=agent_index)
    if not files:
        raise SystemExit(f"No agent*.csv found in {args.data_dir}")

    timers, levels, phase_meta = detect_schema(files[0])
    ylo, yhi = levels[0] * 0.4, levels[-1] * 1.08

    agents = [(agent_index(f), load_decisions(f, timers)) for f in files]
    agents = [(i, rows) for i, rows in agents if rows]
    n_agents = len(agents)
    if n_agents == 0:
        raise SystemExit(
            f"agent*.csv found in {args.data_dir} but no rows matched the "
            f"detected schema (timers={[c for c, _ in timers]}). Wrong dataset?")

    def draw_panel(ax, rows, tname, seen_phases):
        for s0, s1, ph in phase_spans(rows):
            ax.axvspan(s0 - 0.5, s1 + 0.5, color=phase_color(ph, phase_meta), alpha=0.5, lw=0)
            seen_phases.add(ph)
        eps = [row["episode"] for row in rows]
        ys = [row["vals"][tname] for row in rows]
        ax.step(eps, ys, where="post", color=LINE_COLOR, lw=1.1)
        ax.set_yticks(levels)
        ax.set_ylim(ylo, yhi)
        ax.margins(x=0.01)
        ax.grid(axis="y", ls=":", lw=0.6, alpha=0.5)
        ax.tick_params(length=3)

    def phase_legend(seen_phases):
        # Dedupe by regime label so same-regime phases collapse to one entry.
        handles, seen_labels = [], set()
        for p in sorted(seen_phases):
            lab = phase_meta[p][0] if p in phase_meta else f"phase {p}"
            if lab in seen_labels:
                continue
            seen_labels.add(lab)
            handles.append(Patch(facecolor=phase_color(p, phase_meta), alpha=0.6, label=lab))
        return handles

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
        split_w = args.width if args.width else 16.0
        for _, tname in timers:
            seen = set()
            fig, axes = plt.subplots(
                n_agents, 1, figsize=(split_w, args.row_height * n_agents),
                sharex=True, sharey=True, squeeze=False,
            )
            for r, (aid, rows) in enumerate(agents):
                ax = axes[r][0]
                draw_panel(ax, rows, tname, seen)
                ax.set_ylabel(f"agent {aid}\ntimeout (ms)", fontsize=10)
                if r == n_agents - 1:
                    ax.set_xlabel("episode")
            handles = phase_legend(seen)
            ncol = min(6, len(handles)) or 1
            fig.legend(handles=handles, loc="lower center", ncol=ncol,
                       frameon=False, bbox_to_anchor=(0.5, 0.0))
            fig.tight_layout(rect=(0, 0.05, 1, 1.0))
            outputs += save_both(fig, os.path.join(args.data_dir, f"timeout_decisions_{tname}"))
    else:
        # Combined grid: rows = agents, cols = timers.
        seen = set()
        grid_w = args.width if args.width else max(16.0, 6.5 * len(timers))
        fig, axes = plt.subplots(
            n_agents, len(timers),
            figsize=(grid_w, args.row_height * n_agents),
            sharex=True, sharey=True, squeeze=False,
        )
        for r, (aid, rows) in enumerate(agents):
            for c, (_, tname) in enumerate(timers):
                ax = axes[r][c]
                draw_panel(ax, rows, tname, seen)
                if r == 0:
                    ax.set_title(f"{tname} timeout", fontsize=11, fontweight="bold")
                if c == 0:
                    ax.set_ylabel(f"agent {aid}\ntimeout (ms)", fontsize=10)
                if r == n_agents - 1:
                    ax.set_xlabel("episode")
        handles = phase_legend(seen)
        ncol = min(6, len(handles)) or 1
        fig.legend(handles=handles, loc="lower center", ncol=ncol,
                   frameon=False, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.06, 1, 1.0))
        if args.out:
            fig.savefig(args.out)
            plt.close(fig)
            outputs.append(args.out)
        else:
            outputs += save_both(fig, os.path.join(args.data_dir, "timeout_decisions_by_phase"))

    n_dec = sum(len(r) for _, r in agents)
    for out in outputs:
        print(f"wrote {out}")
    print(f"({n_agents} agents, {n_dec} decisions, timers={[c for c, _ in timers]})")


if __name__ == "__main__":
    main()
