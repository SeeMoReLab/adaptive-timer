"""
plot_agents.py
--------------
Place this file at:  scripts/cloudlab/plot_agents.py

Each agent gets its own subplot (shared x-axis). One figure per directory,
saved inside that directory.

Usage:
    python scripts/cloudlab/plot_agents.py              # auto-mode
    python scripts/cloudlab/plot_agents.py --dirs tmp tmp_16 tmp_32
    python scripts/cloudlab/plot_agents.py --out-dir ./plots
    python scripts/cloudlab/plot_agents.py --out-name "{dir}_throughput.png"

Dependencies:
    pip install pandas matplotlib
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import pandas as pd

# ── publication-quality rcParams ─────────────────────────────────────────────
matplotlib_rc = {
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "DejaVu Serif"],
    "font.size":            11,
    "axes.labelsize":       13,
    "axes.titlesize":       13,
    "legend.fontsize":      10,
    "xtick.labelsize":      11,
    "ytick.labelsize":      11,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.linewidth":       0.7,
    "xtick.major.width":    0.7,
    "ytick.major.width":    0.7,
    "xtick.minor.visible":  False,
    "ytick.minor.visible":  False,
    "figure.dpi":           150,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
}
plt.rcParams.update(matplotlib_rc)

# ── colour palette (one colour per agent, consistent across dirs) ────────────
COLORS = ["#378ADD", "#E27B3A", "#2BA37A", "#C94F7C", "#7B5EA7", "#BA7517"]

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIRS = ["tmp_128", "tmp_16", "tmp_32"]


# ─────────────────────────────────────────────────────────────────────────────

def load_agent(path: str) -> pd.DataFrame | None:
    df = pd.read_csv(path)
    required = {"reward_episode", "reward", "phase"}
    missing = required - set(df.columns)
    if missing:
        print(f"  [warn] {os.path.basename(path)} missing columns: {missing} — skipping")
        return None
    return df.sort_values("reward_episode").reset_index(drop=True)


def phase_change_episodes(df: pd.DataFrame) -> list[dict]:
    changes, prev = [], None
    for _, row in df.iterrows():
        ph = row["phase"]
        if prev is not None and ph != prev:
            changes.append({"episode": row["reward_episode"], "phase": ph})
        prev = ph
    return changes


def plot_one_dir(csv_paths: list[str], dir_name: str, out_path: str) -> bool:
    csv_paths = sorted(csv_paths)
    n = len(csv_paths)
    if n == 0:
        return False

    fig, axes = plt.subplots(
        nrows=n, ncols=1,
        figsize=(13, 3 * n),
        sharex=True,
        squeeze=False,
    )
    fig.suptitle(f"Agent throughput — {dir_name}", y=1.01)

    for i, path in enumerate(csv_paths):
        ax    = axes[i][0]
        label = os.path.splitext(os.path.basename(path))[0]
        color = COLORS[i % len(COLORS)]

        df = load_agent(path)
        if df is None:
            ax.set_visible(False)
            continue

        # ── line ─────────────────────────────────────────────────────────────
        ax.plot(
            df["reward_episode"], df["reward"],
            color=color, linewidth=1.2, alpha=0.9,
        )
        ax.set_facecolor("#fafafa")

        # ── phase-change verticals ───────────────────────────────────────────
        y_top = df["reward"].max()
        first_vline = True
        for change in phase_change_episodes(df):
            ep, ph = change["episode"], change["phase"]
            ax.axvline(
                x=ep, color="#888888", linewidth=1.0,
                linestyle="--", alpha=0.55, zorder=1,
                label="phase change" if first_vline else "_nolegend_",
            )
            first_vline = False
            ax.text(
                ep, y_top * 1.01, f" phase {ph}",
                fontsize=8, color="#888888",
                va="top", rotation=90, clip_on=True,
            )

        # ── legend ───────────────────────────────────────────────────────────
        vline_handles, vline_labels = ax.get_legend_handles_labels()
        agent_handle = Line2D([0], [0], color=color, linewidth=1.5)
        ax.legend(
            [agent_handle] + vline_handles,
            [label]        + vline_labels,
            framealpha=0.85, edgecolor="#dddddd",
            loc="upper right",
        )

        # ── axes style ───────────────────────────────────────────────────────
        ax.set_ylabel("throughput", labelpad=4)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.grid(axis="y", color="#cccccc", linewidth=0.5, linestyle="--", alpha=0.6)
        ax.grid(axis="x", color="#eeeeee", linewidth=0.4)
        ax.spines["left"].set_linewidth(0.7)
        ax.spines["bottom"].set_linewidth(0.7)
        ax.tick_params(colors="#444444")

    axes[-1][0].set_xlabel("reward_episode", labelpad=6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)          # dpi/bbox picked up from rcParams
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-agent subplots for multiple directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dirs", nargs="+", default=DEFAULT_DIRS,
        help=f"Dirs to scan (default: {DEFAULT_DIRS})",
    )
    parser.add_argument(
        "--base", default=SCRIPT_DIR,
        help="Base path prepended to each --dirs entry "
             "(default: directory containing this script)",
    )
    parser.add_argument(
        "--out-name", default="{dir}_plot.png",
        help="Output filename pattern — {dir} replaced by dir name.",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Save all plots here instead of inside each scanned dir.",
    )
    args = parser.parse_args()

    print(f"Base path : {args.base}")
    print(f"Scanning  : {args.dirs}\n")

    total_saved = 0
    for dir_name in args.dirs:
        scan_path = os.path.join(args.base, dir_name)

        if not os.path.isdir(scan_path):
            print(f"[warn] not found: {scan_path} — skipping")
            continue

        csv_paths = sorted(glob.glob(os.path.join(scan_path, "agent*.csv")))
        if not csv_paths:
            print(f"[warn] no agent*.csv in {scan_path} — skipping")
            continue

        print(f"[{dir_name}] {len(csv_paths)} agent(s): "
              f"{[os.path.basename(p) for p in csv_paths]}")

        filename = args.out_name.format(dir=dir_name)
        out_path  = os.path.join(args.out_dir, filename) if args.out_dir \
                    else os.path.join(scan_path, filename)

        saved = plot_one_dir(csv_paths, dir_name=dir_name, out_path=out_path)
        if saved:
            total_saved += 1

    print(f"\nDone. {total_saved}/{len(args.dirs)} plot(s) saved.")


if __name__ == "__main__":
    main()