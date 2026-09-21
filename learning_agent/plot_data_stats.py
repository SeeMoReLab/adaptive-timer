"""Faceted bar plot of a metric's mean by (timeout, phase) from the PBFT dataset.

One panel per phase; bars = election timeout (light->dark = short->long); the
"best" bar per phase is outlined (highest for reward-like metrics, lowest for
cost-like metrics such as feature_leader_change_count). Rows are pooled over
agents; each row is one completed episode.

Usage:
    python plot_data_stats.py --metric reward
    python plot_data_stats.py --metric feature_leader_change_count --lower-is-better
    python plot_data_stats.py --metric reward_throughput_tps --data-dir data/pbft --out-dir plot
"""
import argparse
import glob
import math
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reward_utils import reward_tps

# PhaseMapping-intended timeout per phase (annotation only; edit if your labels differ).
INTENDED = {0: 300, 1: 300, 2: 300, 3: 800, 4: 800, 5: 1800}


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


def load_completed(data_dir, timeout_col, phase_col, metric):
    files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    if not files:
        raise SystemExit(f'No CSV files found in {data_dir}')
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if metric == 'reward':
        df = add_derived_reward(df)
    if metric not in df.columns and f'feature_{metric}' in df.columns:
        metric = f'feature_{metric}'
    for col in (timeout_col, phase_col, metric):
        if col not in df.columns:
            raise SystemExit(f"Column '{col}' not in data. Available: {list(df.columns)}")
    df = df.copy()
    df[timeout_col] = df[timeout_col].astype(int)
    df[phase_col] = df[phase_col].astype(int)
    df[metric] = df[metric].astype(float)
    return df, metric


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        '--metric',
        default='reward',
        help='column to average; reward is derived from reward_throughput_tps',
    )
    p.add_argument('--data-dir', default=os.path.join(os.path.dirname(__file__), 'data', 'pbft'))
    p.add_argument('--out-dir', default=os.path.join(os.path.dirname(__file__), 'plot'))
    p.add_argument('--timeout-col', default='election_timeout_milliseconds_used')
    p.add_argument('--phase-col', default='phase')
    p.add_argument('--lower-is-better', action='store_true',
                   help='outline the LOWEST bar per phase (cost metric); default outlines highest')
    p.add_argument('--phases', type=int, nargs='*', default=None,
                   help='phases to show (default: all present that are >= 0)')
    p.add_argument('--ncols', type=int, default=3)
    args = p.parse_args()

    df, metric = load_completed(
        args.data_dir, args.timeout_col, args.phase_col, args.metric
    )

    timeouts = sorted(df[args.timeout_col].unique())
    phases = args.phases if args.phases is not None else sorted(p for p in df[args.phase_col].unique() if p >= 0)

    tbl = (df[df[args.phase_col].isin(phases)]
           .groupby([args.phase_col, args.timeout_col])[metric].mean()
           .unstack().reindex(index=phases, columns=timeouts))

    # light -> dark blue ramp for the (ordinal) timeout
    n = len(timeouts)
    shades = [plt.cm.Blues(0.35 + 0.5 * (i / max(n - 1, 1))) for i in range(n)]
    color = dict(zip(timeouts, shades))
    hilite = '#2ca02c' if args.lower_is_better else '#d62728'
    hilite_label = 'lowest (best)' if args.lower_is_better else 'highest'

    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 12, 'axes.titlesize': 12, 'axes.labelsize': 12,
        'axes.spines.top': False, 'axes.spines.right': False,
    })

    ncols = args.ncols
    nrows = math.ceil(len(phases) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.5 * nrows), squeeze=False)
    flat = axes.flat

    for ax, ph in zip(flat, phases):
        vals = [tbl.loc[ph, t] for t in timeouts]
        bars = ax.bar([str(t) for t in timeouts], vals, color=[color[t] for t in timeouts],
                      width=0.66, zorder=3)
        finite = [v for v in vals if np.isfinite(v)]
        if finite:
            best = int(np.nanargmin(vals) if args.lower_is_better else np.nanargmax(vals))
            bars[best].set_edgecolor(hilite)
            bars[best].set_linewidth(2.5)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f'{v:,.2f}', ha='center', va='bottom', fontsize=9)
        title = f'Phase {ph}'
        if ph in INTENDED:
            title += f'   intended: {INTENDED[ph]} ms'
        ax.set_title(title)
        top = np.nanmax(vals) if finite else 1
        ax.set_ylim(0, top * 1.20 if top > 0 else 1)
        ax.set_xlabel('election timeout (ms)')
        ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    for ax in list(flat):  # hide any unused panels
        if not ax.has_data():
            ax.set_visible(False)
    for r in range(nrows):
        axes[r, 0].set_ylabel(f'mean {metric}')

    handles = [plt.Rectangle((0, 0), 1, 1, color=color[t]) for t in timeouts]
    handles.append(plt.Rectangle((0, 0), 1, 1, fc='white', ec=hilite, lw=2.5))
    labels = [f'{t} ms' for t in timeouts] + [hilite_label]
    fig.legend(handles, labels, ncol=len(labels), frameon=False,
               loc='lower center', bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f'Mean {args.metric} by (timeout, phase) — matched rows, pooled over agents', y=0.99)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f'{args.metric}_by_timeout_phase.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print('saved', out)
    print(tbl.round(3).to_string())


if __name__ == '__main__':
    main()
