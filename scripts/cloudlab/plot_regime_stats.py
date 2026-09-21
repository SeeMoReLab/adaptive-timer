"""Faceted bar plot of a metric's mean by (timeout, regime) for a cloudlab run.

Reads completed episode rows from agent*.csv (each = one node), folds the 19
experiment phases into the named regimes from config/smartbft/failure_spec.xml, and
plots one panel per regime: bars = election timeout (light->dark = short->long).
The "best" bar per regime is outlined (highest for reward-like metrics, lowest for
cost metrics such as feature_leader_change_count.

Usage (from anywhere):
    python scripts/cloudlab/plot_regime_stats.py --metric reward
    python scripts/cloudlab/plot_regime_stats.py --metric feature_leader_change_count --lower-is-better
    python scripts/cloudlab/plot_regime_stats.py --metric feature_regency_change_count --lower-is-better \
        --data-dir scripts/cloudlab/tmp
"""
import argparse
import glob
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "learning_agent"))
from reward_utils import reward_tps

# phase index (0..18) -> regime, from config/smartbft/failure_spec.xml
REGIMES = [
    ("Base (clean)",              [0]),
    ("Leader proposal +300ms",    [1, 2, 3, 4, 5]),
    ("Leader proposal +900ms",    [6, 7, 8, 9, 10]),
    ("Network +50ms",             [11]),
    ("Net +50ms & leader +300ms", [12, 13, 14, 15, 16]),
    ("Network +150ms",            [17]),
    ("Recovery (clean)",          [18]),
]


def add_derived_reward(df):
    if 'reward_throughput_tps' not in df.columns:
        raise SystemExit(
            "Column 'reward_throughput_tps' is required to derive reward."
        )
    df = df.copy()
    df['reward'] = df['reward_throughput_tps'].astype(float).map(
        reward_tps
    )
    return df


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        '--metric',
        default='reward',
        help='column to average; reward is derived from reward_throughput_tps',
    )
    p.add_argument('--data-dir', default=os.path.join(here, 'tmp'), help='dir with agent*.csv')
    p.add_argument('--out-dir', default=None, help='output dir (default: --data-dir)')
    p.add_argument('--timeout-col', default='election_timeout_milliseconds_used')
    p.add_argument('--phase-col', default='phase')
    p.add_argument('--lower-is-better', action='store_true',
                   help='outline the LOWEST bar per regime (cost metric); default outlines highest')
    p.add_argument('--ncols', type=int, default=4)
    args = p.parse_args()
    out_dir = args.out_dir or args.data_dir

    files = sorted(glob.glob(os.path.join(args.data_dir, 'agent*.csv')))
    if not files:
        raise SystemExit(f'No agent*.csv in {args.data_dir}')
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    metric = args.metric
    if metric == 'reward':
        df = add_derived_reward(df)
    if metric not in df.columns and f'feature_{metric}' in df.columns:
        metric = f'feature_{metric}'
    for col in (metric, args.timeout_col, args.phase_col):
        if col not in df.columns:
            raise SystemExit(f"Column '{col}' not found. Available: {list(df.columns)}")
    df = df.copy()
    df[args.phase_col] = df[args.phase_col].astype(int)
    df[args.timeout_col] = df[args.timeout_col].astype(int)
    df[metric] = df[metric].astype(float)

    timeouts = sorted(df[args.timeout_col].unique())
    regimes = [(name, ph) for name, ph in REGIMES if df[args.phase_col].isin(ph).any()]

    n = len(timeouts)
    shades = [plt.cm.Blues(0.35 + 0.5 * (i / max(n - 1, 1))) for i in range(n)]
    color = dict(zip(timeouts, shades))
    hilite = '#2ca02c' if args.lower_is_better else '#d62728'
    hilite_label = 'lowest (best)' if args.lower_is_better else 'highest'

    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 12, 'axes.titlesize': 11, 'axes.labelsize': 11,
        'axes.spines.top': False, 'axes.spines.right': False,
    })

    ncols = args.ncols
    nrows = math.ceil(len(regimes) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.6 * nrows), squeeze=False)
    flat = list(axes.flat)

    print(f"{'regime':30s} " + "  ".join(f"{t}ms" for t in timeouts))
    for ax, (name, phases) in zip(flat, regimes):
        sub = df[df[args.phase_col].isin(phases)]
        vals = [sub.loc[sub[args.timeout_col] == t, metric].mean() for t in timeouts]
        ns = [int((sub[args.timeout_col] == t).sum()) for t in timeouts]
        bars = ax.bar([str(t) for t in timeouts], vals, color=[color[t] for t in timeouts], width=0.66, zorder=3)
        if any(np.isfinite(v) for v in vals):
            best = int(np.nanargmin(vals) if args.lower_is_better else np.nanargmax(vals))
            bars[best].set_edgecolor(hilite); bars[best].set_linewidth(2.5)
        for b, v, cnt in zip(bars, vals, ns):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f'{v:,.2f}\n(n={cnt})',
                        ha='center', va='bottom', fontsize=8)
        ax.set_title(name)
        top = np.nanmax(vals)
        ax.set_ylim(0, top * 1.28 if top and top > 0 else 1)
        ax.set_xlabel('election timeout (ms)')
        ax.grid(True, axis='y', linestyle=':', alpha=0.5)
        print(f"{name:30s} " + "  ".join(f"{v:8.2f}" for v in vals))
    for ax in flat:
        if not ax.has_data():
            ax.set_visible(False)
    for r in range(nrows):
        axes[r, 0].set_ylabel(f'mean {metric}')

    handles = [plt.Rectangle((0, 0), 1, 1, color=color[t]) for t in timeouts]
    handles.append(plt.Rectangle((0, 0), 1, 1, fc='white', ec=hilite, lw=2.5))
    fig.legend(handles, [f'{t} ms' for t in timeouts] + [hilite_label], ncol=len(timeouts) + 1,
               frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f'Mean {args.metric} by timeout per regime — matched rows, pooled over agents', y=0.99)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'{args.metric}_by_timeout_regime.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print('\nsaved', out)


if __name__ == '__main__':
    main()
