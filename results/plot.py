#!/usr/bin/env python3
"""Per protocol, one figure of the finalized runs in results/<protocol>/<run>/ (those with the
row.json finalize.py wrote), saved as results/plots/<protocol>.png. Four columns of
panels: the oracle and the heuristic (and static, when it runs); the GP configurations; the
LinUCB ones; the PPO ones. Rows: the client's mean latency and the timeout replica 0 had in force;
SmartBFT, whose throughput the timeout moves, gets a throughput row on top. The model columns show
the floor variants (Kevin, 2026-10-21; --all adds the no-floor rows, dashed = floor); within a column the
colour is the configuration (continuous or discrete, pretrained or untrained); SBFT's fast-path wait W is
dotted in the run's colour. A missing configuration
or a run that cannot be read is reported on stderr and the figure is still drawn, panels empty if
need be.
The time axis is the spec's phases only: 0 is the first phase onset (the warm-up before it is
left out) and the axis ends where the last phase does, one phase length after its onset (the
run's planned end; anything after is left out). Throughput and latency come from the client's
per-second Monitor lines in client.log. Each point is the mean over a --window of seconds, the
windows tiled from the first onset so none straddles a phase edge (the phase length, 300 s in the
real specs, must be divisible by the window): tps is the mean of the per-second successful
commits and the latency the commit-weighted mean of the per-second means (p50 until 2026-10-21); a window with no commits takes
its own length as its latency, as the agent's empty-window rule does.
The timeouts are replica 0's played values (agent0.csv *_milliseconds_used; the derived Tendermint
prevote/precommit timeouts are left out), drawn from when they were applied: the agent's report
time for that episode (logs/agents/agent_0.log) plus the reply wait. The agent's episode clock
starts when it connects, not at the warm-up, so episode numbers alone cannot place them. Vertical
lines mark the failure_spec phase onsets. Inside each phase of the throughput and latency panels, one
number per run in the run's colour (italic = floor variant under --all): the mean of that run's
windows in the phase, as drawn.
Usage: results/plot.py [--protocol sbft] [--window 20] [--all]   (default: every protocol, 20 s windows)
"""
import argparse
import csv
import fnmatch
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parent
sys.path.insert(0, str(RESULTS.parent / "learning_agent"))
from reward_utils import reward_normalizer_tps, reward_tps_latency  # noqa: E402

START = re.compile(r"benchmark start unix ms (\d+)")
MON = re.compile(r"\[client (\S+)\] Monitor (duration=.*)")
KV = re.compile(r"(\w+)=(\S+)")
# "[2026-10-08 07:35:00] [episode 77] accepted valid local report ..." (scheduled) or
# "[... ] [episode 1 seq=100 window_consensus_count=0] accepted valid report ..." (sharing;
# the first of the quorum's reports is taken): the agent's wall clock (the nodes run on
# UTC), to the second.
REPORT = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] \[episode (\d+)[^\]]*\] accepted")
# Replicas apply an episode's timeouts when the reply arrives, at the end of the reply
# wait after the report (every harness defaults it to 2 s; the CppBedrock run.env
# records it and is checked). The last value is drawn for one more 20 s episode.
REPLY_WAIT_S = 2.0
EPISODE_S = 20.0
DERIVED_TIMEOUTS = ("prevote_", "precommit_")


def run_env(run):
    env = {}
    for line in (run / "run.env").read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep:
            env[key] = value
    return env


def client_seconds(run):
    """(benchmark start unix s, [(t_s, succ, p50_ms)] per Monitor line, t_s from the start)."""
    t0, rows = None, []
    for line in (run / "client.log").read_text(errors="replace").splitlines():
        m = START.search(line)
        if m and t0 is None:
            t0 = int(m.group(1)) / 1000.0
            continue
        m = MON.search(line)
        if m and t0 is not None:
            v = dict(KV.findall(m.group(2)))
            ts = datetime.fromisoformat(m.group(1).replace("Z", "+00:00")).timestamp()
            rows.append((ts - t0, int(v["succ"]), float(v["avg_ms"])))
    if t0 is None:
        raise ValueError(f"{run}: no 'benchmark start unix ms' line in client.log")
    if not rows:
        raise ValueError(f"{run}: no client Monitor lines in client.log")
    return t0, rows


def buckets(rows, width, protocol, start, end):
    """Per window of `width` seconds tiled from `start` to `end`: (window start in s from `start`,
    tps, p50_ms, reward); a window the client output does not reach is left out."""
    out = []
    lo = start
    while lo + width <= end + 1e-9:
        sec = [r for r in rows if lo <= r[0] < lo + width]
        if sec:
            succ = sum(r[1] for r in sec)
            tps = succ / len(sec)
            p50 = sum(r[1] * r[2] for r in sec) / succ if succ else width * 1000.0
            out.append((lo - start, tps, p50, reward_tps_latency(tps, p50, reward_normalizer_tps(protocol))))
        lo += width
    return out


def phase_onsets(run, protocol):
    """(warm-up s, [phase onsets in s from the benchmark start], end of the last phase): the last
    phase is as long as the one before it."""
    spec = re.sub(r"<!--.*?-->", "", (run / "failure_spec.xml").read_text(), flags=re.S)
    warm = float(re.search(r"<warmUpTime>([\d.]+)</warmUpTime>", spec).group(1))
    # Tendermint's appliers fire 0.5 s after the client's start (cal/cl_judge.py).
    offset = 0.5 if protocol == "tendermint" else 0.0
    onsets = [warm + float(t) + offset for t in re.findall(r"<atTime>([\d.]+)</atTime>", spec)]
    if len(onsets) < 2:
        raise ValueError(f"{run}: failure_spec.xml has {len(onsets)} phase onset(s); two or more are needed")
    return warm, onsets, onsets[-1] + (onsets[-1] - onsets[-2])


def report_times(run, t0, end_s):
    """{episode: report time in s from the benchmark start} from replica 0's agent log."""
    path = run / "logs" / "agents" / "agent_0.log"
    if not path.exists():
        raise ValueError(f"{run}: no {path.relative_to(run)} to place the timeouts in time")
    out = {}
    for line in path.read_text(errors="replace").splitlines():
        m = REPORT.match(line)
        if m:
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
            out.setdefault(int(m.group(2)), ts - t0)
    if not out:
        raise ValueError(f"{path}: no '[episode N] accepted' report lines")
    if not (-60.0 < min(out.values()) and max(out.values()) < end_s + 60.0):
        raise ValueError(f"{path}: report times {min(out.values()):.0f}..{max(out.values()):.0f} s fall "
                         f"outside the client's 0..{end_s:.0f} s (clock or timezone mismatch?)")
    return out


def timeouts(run, t0, end_s):
    """{column: [(t_s, ms)]} from replica 0, each value from the time it was applied."""
    path = run / "agent0.csv"
    if not path.exists():
        return {}
    reports = report_times(run, t0, end_s)
    last = max(reports)
    placed = []
    series = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            ep = int(r["episode"])
            if ep in reports:
                t = reports[ep] + REPLY_WAIT_S
            elif ep > last:
                # The agent log can lose its tail at shutdown (an unflushed buffer: the
                # Tendermint oracle's ends 28 episodes early); wall-clock episodes run
                # every 20 s, so a trailing episode is placed on that grid.
                t = reports[last] + (ep - last) * EPISODE_S + REPLY_WAIT_S
                placed.append(ep)
            else:
                raise ValueError(f"{run}: agent0.csv episode {ep} has no report line in agent_0.log")
            for col, val in r.items():
                if col.endswith("_milliseconds_used") and not col.startswith(DERIVED_TIMEOUTS) and val:
                    series.setdefault(col.removesuffix("_milliseconds_used"), []).append((t, float(val)))
    if placed:
        print(f"warning: {run.name}: agent_0.log ends at episode {last}; episodes "
              f"{placed[0]}-{placed[-1]} placed on the {EPISODE_S:g} s grid after it", file=sys.stderr)
    # Carry each last value to the end of its episode so the final step is drawn.
    return {k: v + [(v[-1][0] + EPISODE_S, v[-1][1])] for k, v in series.items()}


def run_series(run, protocol, width):
    """(windows, phase onsets, end, timeouts) of one run, times in s from the first phase onset;
    the timeouts clipped to the phases."""
    env = run_env(run)
    if float(env.get("learning_reply_wait_ms", REPLY_WAIT_S * 1000.0)) != REPLY_WAIT_S * 1000.0:
        raise ValueError(f"{run}: run.env reply wait {env['learning_reply_wait_ms']} ms, plot.py assumes {REPLY_WAIT_S:g} s")
    t0, seconds = client_seconds(run)
    _, onsets, end = phase_onsets(run, protocol)
    lengths = {round(b - a, 3) for a, b in zip(onsets, onsets[1:])}
    if any(round(l % width, 6) not in (0.0, round(width, 6)) for l in lengths):
        raise ValueError(f"{run}: phase lengths {sorted(lengths)} s are not divisible by the {width:g} s window")
    start = onsets[0]
    played = {}
    for col, pts in timeouts(run, t0, max(r[0] for r in seconds)).items():
        before = [p for p in pts if p[0] <= start]
        clipped = ([(start, before[-1][1])] if before else []) + [p for p in pts if start < p[0] < end]
        if clipped:
            clipped.append((end, clipped[-1][1]))
        played[col] = [(t - start, v) for t, v in clipped]
    return buckets(seconds, width, protocol, start, end), [o - start for o in onsets], end - start, played


def select_runs(protocol, patterns):
    """Run directories under results/<protocol>/ with a client.log whose name matches any
    of the ';'-separated globs in `patterns` (table.py and raster.py)."""
    globs = [g.strip() for g in patterns.split(";") if g.strip()]
    if not globs:
        raise SystemExit(f"--runs {patterns!r} has no patterns")
    base = RESULTS / protocol
    runs = sorted(d for d in base.iterdir() if d.is_dir() and (d / "client.log").exists()
                  and any(fnmatch.fnmatch(d.name, g) for g in globs))
    if not runs:
        raise SystemExit(f"no runs with a client.log under {base} matching {patterns!r}")
    return runs


PROTOCOLS = ("pbft", "sbft", "smartbft", "tendermint", "hotstuff", "autobahn")
# Column title -> the configurations it holds, in legend order.
COLUMNS = (
    ("oracle, heuristic", ["oracle", "heuristic", "static"]),
    ("GP", [f"{m}_gp_{t}{f}" for m in ("continuous", "discrete") for t in ("pre", "untr") for f in ("", "_floor")]),
    ("LinUCB", [f"{m}_linucb_{t}{f}" for m in ("continuous", "discrete") for t in ("pre", "untr") for f in ("", "_floor")]),
    ("PPO", [f"{m}_ppo_{t}{f}" for m in ("continuous", "discrete") for t in ("pre", "untr") for f in ("", "_floor")]),
)
EXPECTED = ("oracle", "heuristic") + tuple(l for _, ls in COLUMNS[1:] for l in ls)
# Dark blue, orange, green, magenta: each at least 3:1 against white so the phase means stay legible
# (tab10's orange and green were not); a column holds at most four configurations.
COLORS = ("#1c5cab", "#c2410c", "#00754a", "#a8197a")
FLOOR_DASHED = False  # set when both variants are drawn (--all): the floor one dashed


def finalized_runs(protocol):
    """{configuration label: run dir} from results/<protocol>/*/row.json; problems go to stderr."""
    out = {}
    base = RESULTS / protocol
    if not base.is_dir():
        print(f"{protocol}: no results/{protocol}/", file=sys.stderr)
        return out
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        if (d / "draw.json").exists() and not (d / "row.json").exists():
            continue  # an earlier draw of a configuration rerun on purpose; the row is drawn
        if not (d / "row.json").exists():
            print(f"{protocol}: {d.name} has no row.json (not finalized); skipped", file=sys.stderr)
            continue
        label = json.loads((d / "row.json").read_text())["label"]
        if label in out:
            print(f"{protocol}: two runs of {label} ({out[label].name}, {d.name}); the first is drawn", file=sys.stderr)
            continue
        out[label] = d
    for label in EXPECTED:
        if label not in out:
            print(f"{protocol}: no {label} run", file=sys.stderr)
    return out


def style_of(label, column_labels):
    """Colour by configuration (floor and no floor share one), dashed for the floor variant."""
    base = label.removesuffix("_floor")
    bases = []
    for l in column_labels:
        b = l.removesuffix("_floor")
        if b not in bases:
            bases.append(b)
    return dict(color=COLORS[bases.index(base) % len(COLORS)],
                ls="--" if label.endswith("_floor") and FLOOR_DASHED else "-", lw=0.9)


def annotate_phases(ax, data, col, edges, k, style, label, fmt):
    """Write the mean of data[*][col] over each phase's windows at the phase's centre, the run's
    k-th line from the top of the panel."""
    for lo, hi in zip(edges, edges[1:]):
        vals = [b[col] for b in data if lo - 1e-6 <= b[0] < hi - 1e-6]
        if vals:
            ax.text((lo + hi) / 2, 0.97 - 0.07 * k, fmt(sum(vals) / len(vals)), transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=6.5, color=style["color"],
                    fontstyle="italic" if label.endswith("_floor") and FLOOR_DASHED else "normal",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=0.5))


def plot_protocol(protocol, width, floor_only=True):
    runs = finalized_runs(protocol)
    if floor_only:  # Kevin 2026-10-21: the model columns show the floor variants only
        runs = {l: r for l, r in runs.items() if l in ("oracle", "heuristic", "static") or l.endswith("_floor")}
    row_names = (["throughput"] if protocol == "smartbft" else []) + ["latency", "timeout"]
    fig, axes = plt.subplots(len(row_names), len(COLUMNS), figsize=(24, 3.6 * len(row_names)),
                             sharex=True, sharey="row", squeeze=False)
    onsets, end = [], None
    drawn = [0] * len(COLUMNS)  # runs drawn so far per column: the row of its phase means
    for c, (title, labels) in enumerate(COLUMNS):
        axes[0][c].set_title(title)
        for label in labels:
            run = runs.get(label)
            if run is None:
                continue
            try:
                data, run_onsets, run_end, played = run_series(run, protocol, width)
            except Exception as exc:  # a run with a broken artifact is reported, the figure still drawn
                print(f"{protocol}: {run.name}: {type(exc).__name__}: {exc}; skipped", file=sys.stderr)
                continue
            if not data:
                print(f"{protocol}: {run.name}: no client window inside the phases; skipped", file=sys.stderr)
                continue
            onsets, end = onsets or run_onsets, end or run_end
            if run_onsets != onsets:
                print(f"{protocol}: {run.name}: phase onsets differ from the first run drawn", file=sys.stderr)
            style = style_of(label, labels)
            # Each window's mean is drawn at the window's start, so a change in the window after a phase onset
            # sits on the onset line (at the centre it sat half a window past it).
            t = [b[0] for b in data]
            for r, name in enumerate(row_names):
                ax = axes[r][c]
                if name == "throughput":
                    ax.plot(t, [b[1] for b in data], label=label, **style)
                    annotate_phases(ax, data, 1, run_onsets + [run_end], drawn[c], style, label,
                                    lambda v: f"{v / 1000:.1f}k" if v >= 1000 else f"{v:.0f}")
                elif name == "latency":
                    ax.plot(t, [b[2] for b in data], label=label, **style)
                    annotate_phases(ax, data, 2, run_onsets + [run_end], drawn[c], style, label, lambda v: f"{v:.0f}")
                else:
                    for j, pts in enumerate(played.values()):
                        ax.step([p[0] for p in pts], [p[1] for p in pts], where="post",
                                **{**style, "ls": style["ls"] if j == 0 else ":"}, label=label if j == 0 else None)
            drawn[c] += 1
        axes[-1][c].set_xlabel("time since the first phase onset (s)")
        handles, names = axes[-1][c].get_legend_handles_labels()
        if handles:
            axes[-1][c].legend(handles, names, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    for r, name in enumerate(row_names):
        axes[r][0].set_ylabel({"throughput": "throughput (tps)", "latency": "mean latency (ms)",
                               "timeout": "timeout (ms)" + (": T solid/dashed, W dotted" if protocol == "sbft" else "")}[name])
        if name == "latency":
            axes[r][0].set_yscale("log")
    for row in axes:
        for ax in row:
            for x in onsets:
                ax.axvline(x, color="grey", ls="--", lw=0.6)
            if end:
                ax.set_xlim(0, end)
                ax.set_xticks(onsets + [end])  # one tick per phase onset (300 s apart in the real specs) and the end
            ax.grid(alpha=0.3)
    fig.suptitle(f"{protocol}: {len(runs)} finalized runs, {width:g} s windows")
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--protocol", choices=PROTOCOLS, help="one protocol (default: all)")
    ap.add_argument("--all", action="store_true", help="draw the no-floor model rows too (dashed = floor variant)")
    ap.add_argument("--window", type=int, default=20,
                    help="seconds averaged per point; must divide the 300 s phase length (default 20)")
    args = ap.parse_args()
    if args.window <= 0 or 300 % args.window:
        ap.error(f"--window {args.window}: must be a positive divisor of 300 (the phase length in seconds)")
    for protocol in ([args.protocol] if args.protocol else PROTOCOLS):
        global FLOOR_DASHED
        FLOOR_DASHED = args.all
        fig = plot_protocol(protocol, args.window, floor_only=not args.all)
        out = RESULTS / "plots" / f"{protocol}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(out.relative_to(RESULTS.parent))


if __name__ == "__main__":
    main()
