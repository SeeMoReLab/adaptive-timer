#!/usr/bin/env python3
"""EDA: correlation between report features (+ chosen timeouts + phase) and reward.

Reads agent0..N.csv from a dataset directory and uses the reward-labeled rows
(matched_episode_features == 1), which carry both the decision-time state
features and the realized reward (throughput). Renders:

  1. feature_reward_correlation.{png,pdf} -- each variable's Pearson & Spearman
     correlation with reward, sorted (the "what drives reward" view).
  2. feature_correlation_heatmap.{png,pdf} -- full Pearson correlation matrix.

Usage:
    python3 plot_feature_reward_corr.py [DATA_DIR]
"""
import argparse
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})

# Decision-time state features (Tendermint report schema).
FEATS = [
    "total_transactions", "total_consensus_instances", "avg_consensus_latency_ms",
    "p95_consensus_latency_ms", "p99_consensus_latency_ms", "throughput_tps",
    "timeout_violation_rate", "avg_batch_size", "p95_batch_size",
    "leader_change_count", "regency_change_count",
    "propose_latency_ms", "prevote_latency_ms", "precommit_latency_ms",
]
# Chosen action (the agent's decision) + regime label.
ACTIONS = [
    "propose_timeout_milliseconds_used",
    "prevote_timeout_milliseconds_used",
    "precommit_timeout_milliseconds_used",
]
REWARD = "reward"  # realized throughput after applying the action


def load(data_dir):
    cols = FEATS + ACTIONS + ["phase"]
    data = {c: [] for c in cols}
    rewards = []
    for f in sorted(glob.glob(os.path.join(data_dir, "agent*.csv"))):
        for r in csv.DictReader(open(f)):
            if r.get("matched_episode_features") != "1":
                continue
            try:
                rw = float(r[REWARD])
                vals = {c: float(r[c]) for c in cols}
            except (KeyError, ValueError):
                continue
            rewards.append(rw)
            for c in cols:
                data[c].append(vals[c])
    return data, np.array(rewards)


def spearman(a, b):
    """Spearman = Pearson on ranks (no scipy dependency)."""
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


def main():
    ap = argparse.ArgumentParser()
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
    ap.add_argument("data_dir", nargs="?", default=default_dir)
    args = ap.parse_args()

    data, reward = load(args.data_dir)
    n = len(reward)
    if n == 0:
        raise SystemExit(f"No reward-labeled rows found in {args.data_dir}")

    names = FEATS + ACTIONS + ["phase"]
    mat = np.array([data[c] for c in names], dtype=float)  # (vars, n)

    # ---- 1. correlation with reward, Pearson + Spearman, sorted by |Spearman|
    pear, spear = [], []
    for row in mat:
        if np.std(row) == 0:
            pear.append(0.0); spear.append(0.0)
        else:
            pear.append(np.corrcoef(row, reward)[0, 1])
            spear.append(spearman(row, reward))
    pear, spear = np.array(pear), np.array(spear)
    order = np.argsort(np.abs(spear))

    def color(name):
        if name in ACTIONS:
            return "#c0392b"      # action columns = the agent's lever
        if name == "phase":
            return "#7f8c8d"
        return "#1f3b73"

    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(order))
    h = 0.4
    ax.barh(y + h / 2, spear[order], height=h, color=[color(names[i]) for i in order],
            label="Spearman (rank)")
    ax.barh(y - h / 2, pear[order], height=h, alpha=0.45,
            color=[color(names[i]) for i in order], label="Pearson")
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("correlation with reward (throughput)")
    ax.set_title(f"Feature ↔ reward correlation  ({args.data_dir}, n={n})", fontweight="bold")
    ax.set_xlim(-1, 1)
    ax.grid(axis="x", ls=":", lw=0.6, alpha=0.5)
    # annotate the action rows explicitly (the "is there signal to learn?" rows)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#1f3b73", label="state feature"),
        Patch(color="#c0392b", label="chosen timeout (action)"),
        Patch(color="#7f8c8d", label="phase (regime)"),
    ], loc="lower right", frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(args.data_dir, f"feature_reward_correlation.{ext}")
        fig.savefig(p); print(f"wrote {p}")
    plt.close(fig)

    # ---- 2. full Pearson correlation heatmap
    full = np.vstack([mat, reward[None, :]])
    full_names = names + ["reward"]
    # drop zero-variance rows so corrcoef doesn't emit NaNs
    keep = [i for i in range(full.shape[0]) if np.std(full[i]) > 0]
    full, full_names = full[keep], [full_names[i] for i in keep]
    C = np.corrcoef(full)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(full_names)))
    ax.set_yticks(range(len(full_names)))
    ax.set_xticklabels(full_names, rotation=90, fontsize=7)
    ax.set_yticklabels(full_names, fontsize=7)
    for i in range(len(full_names)):
        for j in range(len(full_names)):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                    fontsize=5.5, color="black" if abs(C[i, j]) < 0.6 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    ax.set_title(f"Correlation matrix  ({args.data_dir}, n={n})", fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(args.data_dir, f"feature_correlation_heatmap.{ext}")
        fig.savefig(p); print(f"wrote {p}")
    plt.close(fig)

    # ---- console summary
    print(f"\ncorrelation with reward (sorted by |Spearman|), n={n}:")
    for i in order[::-1]:
        tag = "  [ACTION]" if names[i] in ACTIONS else ""
        print(f"  {names[i]:32s} pearson={pear[i]:+.3f}  spearman={spear[i]:+.3f}{tag}")


if __name__ == "__main__":
    main()
