#!/usr/bin/env python3
"""Per protocol, one figure of the finalized runs in results/<protocol>/<run>/ (those with the
row.json or draw.json finalize.py wrote): of each configuration, the draw with the highest mean R (Kevin,
2026-10-26; the latest draw until then), since time series do not average across runs; saved as results/plots/<protocol>.png. Four columns of
panels: the oracle and the heuristic (and the static runs, when any ran; Kevin 2026-10-28); the GP configurations; the
LinUCB ones; the PPO ones. Rows: the client's mean latency and the timeout replica 0 had in force;
SmartBFT, whose throughput the timeout moves, gets a throughput row on top. The model columns show
the floor variants (Kevin, 2026-10-21; --all adds the no-floor rows, dash-dot when pretrained, dotted when
untrained). Styles (Kevin, 2026-10-26): the oracle bold grey in every column as the reference, the heuristic
black; continuous models blue, discrete red; pretrained dark and solid, untrained lighter and dashed; one line
weight. SBFT's fast-path wait W
is the thin line in the run's colour and style. A missing configuration
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
starts when it connects, not at the warm-up, so episode numbers alone cannot place them. The
phases alternate between a white and a light grey background. Above each throughput and latency panel, one
row of numbers per legend entry, in the legend's order (the oracle first; a missing run leaves its row empty, so
the rows sit the same in every column): over each phase, the mean of that run's windows in the phase (the
latency axis ends at 2,000 ms, Autobahn's and HotStuff's at 4,000; a line above it gets an arrowhead on the top
edge at its peak; the numbers are not cut); in each phase the lowest latency and the highest throughput (SmartBFT) are bold, the oracle not counted.
Usage: results/plot.py [--protocol sbft] [--window 20] [--all]   (default: every protocol, 20 s windows)
With PBFT (or no --protocol) the motivation figure and the scalability figure (plot_scalability: results/plots/scalability.png, latency as grouped bars by replica count) are drawn too.
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
import matplotlib.patches  # noqa: E402
import matplotlib.ticker  # noqa: E402

RESULTS = Path(__file__).resolve().parent
sys.path.insert(0, str(RESULTS.parent / "learning_agent"))
from reward_utils import reward_normalizer_tps, reward_tps_latency  # noqa: E402
sys.path.insert(0, str(RESULTS))
from scoring import STATIC_LABELS, score  # noqa: E402  (results/scoring.py: the judge's mean R per run, cached)

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
    ("Oracle, heuristic", ["oracle", "heuristic"]),
    ("GP", [f"{m}_gp_{t}{f}" for m in ("continuous", "discrete") for t in ("pre", "untr") for f in ("", "_floor")]),
    ("LinUCB", [f"{m}_linucb_{t}{f}" for m in ("continuous", "discrete") for t in ("pre", "untr") for f in ("", "_floor")]),
    ("PPO", [f"{m}_ppo_{t}{f}" for m in ("continuous", "discrete") for t in ("pre", "untr") for f in ("", "_floor")]),
)
EXPECTED = ("oracle", "heuristic") + tuple(l for _, ls in COLUMNS[1:] for l in ls)
# Kevin 2026-10-26: the oracle bold grey in every column; continuous blue, discrete red (Kevin: not orange); pretrained the
# dark shade and solid, untrained a lighter shade (still about 3:1, so its numbers stay legible) and dashed; one line
# weight. Under --all the no-floor variants are dash-dot (pretrained) and dotted (untrained).
ORACLE_COLOR = "#8C8C8C"
# The static runs (in the first column, after the heuristic, when any ran; Kevin 2026-10-28): one hue per timeout,
# shortest to longest green, orange, purple (ColorBrewer Dark2; Kevin 2026-10-28: more contrast than the purple shades
# of 2026-10-26), none of them the models' blue and red; SBFT's fast-path wait W 10 solid, 150 dashed.
STATIC_SHADES = ("#1B9E77", "#D95F02", "#7570B3")
OTHER_COLORS = {"heuristic": "#000000", "static": "#6B3E99"}
# Pretrained in LaTeX's (xcolor) blue and red, untrained their 60% tints, blue!60 and red!60 (Kevin 2026-10-26).
KIND_COLORS = {"continuous": ("#0000FF", "#6666FF"), "discrete": ("#FF0000", "#FF6666")}  # (pretrained, untrained)
BAND_FONT = 11  # the phase-mean numbers above each panel (6.5, then 8.5, until 2026-10-26)
BAND_STEP = 14  # points between their rows
BAND_LIFT = 9  # points from the panel's top edge to the lowest row, clear of the over-the-top arrowheads
LW = 1.0
# The latency axis ends here (Kevin 2026-10-26); a stall's windows above it leave the panel. Autobahn and HotStuff 4,000
# (Autobahn's oracle peaks at 3,488 ms, HotStuff's 5+150 phase sits near 1,800); every other protocol 2,000.
LATENCY_TOP_MS = {"autobahn": 4000, "hotstuff": 4000}
LATENCY_TOP_DEFAULT_MS = 2000
PHASE_SHADE = "#F7F7F7"  # every other phase's background
PROTOCOL = None  # the protocol being drawn (style_of's static shades)
FLOOR_SHADES = False  # set under --all: the no-floor variants dash-dot / dotted


STATIC_TIE_R = 0.01  # own-phase R within this of the best counts as a tie, broken on the whole-run mean R


def oracle_phase_labels(protocol):
    """Per phase of config/<protocol>/failure_spec.xml, the static label of the rung the oracle
    (learning_agent/config/schedule_<protocol>.csv) plays at the phase's midpoint, e.g. "static_250" or "static_650_150"."""
    rows = [tuple(int(x) for x in l.split(",")) for l in
            (RESULTS.parent / "learning_agent" / "config" / f"schedule_{protocol}.csv").read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    xml = re.sub(r"<!--.*?-->", "", (RESULTS.parent / "config" / protocol / "failure_spec.xml").read_text(), flags=re.S)
    onsets = [int(x) for x in re.findall(r"<atTime>(\d+)", xml)]
    mid = [(a + b) / 2 for a, b in zip(onsets, onsets[1:] + [onsets[-1] + 300])]
    return ["static_" + "_".join(str(v) for v in [r for r in rows if r[0] <= m][-1][1:]) for m in mid]


def best_draw(label, ds, own):
    """The draw a figure shows. Statics (Kevin 2026-10-28): the highest mean R over the phases where the oracle plays
    that rung (own: label -> phase indices), ties within STATIC_TIE_R broken on the whole-run mean R; a static the
    oracle never plays, and every other configuration, the highest whole-run mean R (Kevin, 2026-10-26)."""
    if not own.get(label):
        return max(ds, key=lambda d: score(d)["mean"])
    phase_r = {d: sum(score(d)["phases"][i]["R"] for i in own[label]) / len(own[label]) for d in ds}
    top = max(phase_r.values())
    return max((d for d in ds if phase_r[d] >= top - STATIC_TIE_R), key=lambda d: score(d)["mean"])


def finalized_runs(protocol):
    """{configuration label: run dir}: of each configuration's draws in results/<protocol>/ (row.json and draw.json),
    the one best_draw picks (Kevin, 2026-10-26: the best draw, not the latest; 2026-10-28: a static's best on its own
    phases); problems go to stderr."""
    draws = {}
    base = RESULTS / protocol
    if not base.is_dir():
        print(f"{protocol}: no results/{protocol}/", file=sys.stderr)
        return {}
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        f = d / "row.json" if (d / "row.json").exists() else d / "draw.json"
        if not f.exists():
            print(f"{protocol}: {d.name} has no row.json or draw.json (not finalized); skipped", file=sys.stderr)
            continue
        draws.setdefault(json.loads(f.read_text())["label"], []).append(d)
    labels = oracle_phase_labels(protocol)
    own = {l: [i for i, x in enumerate(labels) if x == l] for l in STATIC_LABELS[protocol]}
    out = {label: best_draw(label, ds, own) for label, ds in draws.items()}
    for label in EXPECTED:
        if label not in out:
            print(f"{protocol}: no {label} run", file=sys.stderr)
    return out


def fig_label(label):
    """A run's name in the figures (Kevin 2026-10-30): capitalized, the models as C-/D- (continuous, discrete), GP, LinUCB
    or PPO, then pre or un (pretrained, untrained); "_floor" dropped, every model row being floored (under --all the
    no-floor variants say so)."""
    if label in ("oracle", "heuristic", "static"):
        return label.capitalize()
    if label.startswith("static_"):  # static_<T> or SBFT's static_<T>_<W>
        return "Static " + "/".join(label.split("_")[1:])
    m = re.fullmatch(r"(continuous|discrete)_(gp|linucb|ppo)_(pre|untr)(_floor)?", label)
    if not m:
        raise ValueError(f"no figure label for {label!r}")
    name = f"{m.group(1)[0].upper()}-{dict(gp='GP', linucb='LinUCB', ppo='PPO')[m.group(2)]} {dict(pre='pre', untr='un')[m.group(3)]}"
    return name if m.group(4) else name + " (no floor)"


def style_of(label):
    """The oracle bold grey, the heuristic black; a model's colour by its kind (continuous, discrete), dark and solid
    when pretrained, lighter and dashed when untrained; under --all its no-floor variant dash-dot or dotted."""
    if label == "oracle":
        return dict(color=ORACLE_COLOR, ls="-", lw=3.5 * LW, zorder=1)  # bold, behind the models (Kevin 2026-10-26)
    if label in OTHER_COLORS:
        return dict(color=OTHER_COLORS[label], ls="-", lw=LW, zorder=2)
    if label.startswith("static_"):
        t = sorted({int(l.split("_")[1]) for l in STATIC_LABELS[PROTOCOL]})
        parts = [int(x) for x in label.split("_")[1:]]
        return dict(color=STATIC_SHADES[t.index(parts[0])], ls="--" if len(parts) > 1 and parts[1] > 10 else "-",
                    lw=LW, zorder=2)
    dark, light = KIND_COLORS[label.split("_", 1)[0]]
    untrained = "_untr" in label
    no_floor = FLOOR_SHADES and not label.endswith("_floor")
    return dict(color=light if untrained else dark, ls=(":" if untrained else "-.") if no_floor else ("--" if untrained else "-"),
                lw=LW, zorder=2)


def over_top(ax, data, top, style):
    """An upward arrowhead on the panel's top edge, in the run's colour, at the highest window of each stretch of
    windows above the axis (Kevin 2026-10-26), so a line that leaves the chart is marked where it does."""
    stretch = []
    for b in data + [(None, None, 0.0)]:
        if b[2] > top:
            stretch.append(b)
        elif stretch:
            peak = max(stretch, key=lambda x: x[2])
            ax.plot([peak[0]], [top], marker="^", markersize=6, ls="none", color=style["color"],
                    markeredgecolor="white", markeredgewidth=0.5, clip_on=False, zorder=6)
            stretch = []


def phase_band(ax, rows, edges, n, fmt, best=None, title=None):
    """Above the panel, an n-row band in the legend's order: for each (k, data, col, colour, ranked, name) in rows, in
    row k, the run's name left of the panel (Kevin 2026-10-30) and the mean of data[*][col] over each phase's windows
    at the phase's centre. title, rotated like a y label left of the names (Kevin 2026-10-30), names the band.
    best "min" or "max" bolds each phase's best shown value among the ranked rows, ties at the shown precision all
    bold (Kevin 2026-10-26: the lowest latency and the highest throughput, the oracle not ranked)."""
    means = {}
    names = []
    for k, data, col, color, ranked, name in rows:
        names.append(ax.annotate(name, (0, 1), xycoords="axes fraction", xytext=(-8, BAND_LIFT + BAND_STEP * (n - 1 - k)),
                                 textcoords="offset points", ha="right", va="bottom", fontsize=BAND_FONT, color=color,
                                 annotation_clip=False))
        for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
            vals = [b[col] for b in data if lo - 1e-6 <= b[0] < hi - 1e-6]
            if vals:
                means[(k, i)] = (sum(vals) / len(vals), color, ranked)
    if title and names:
        # The names are right-aligned on one edge, so the longest one reaches furthest left.
        widest = max(names, key=lambda t: len(t.get_text()))
        ax.annotate(title, (0, 1), xycoords=(widest, "axes fraction"),
                    xytext=(-6, BAND_LIFT + (BAND_STEP * (n - 1) + BAND_FONT) / 2), textcoords="offset points",
                    ha="right", va="center", rotation=90, multialignment="center",
                    fontsize=matplotlib.rcParams["axes.labelsize"], annotation_clip=False)
    top = {}
    if best:
        pick = min if best == "min" else max
        for (k, i), (v, _, ranked) in means.items():
            if ranked:
                top[i] = pick(top.get(i, v), v)
    for (k, i), (v, color, ranked) in means.items():
        bold = ranked and i in top and fmt(v) == fmt(top[i])
        ax.annotate(fmt(v), ((edges[i] + edges[i + 1]) / 2, 1), xycoords=("data", "axes fraction"),
                    xytext=(0, BAND_LIFT + BAND_STEP * (n - 1 - k)), textcoords="offset points", ha="center", va="bottom",
                    fontsize=BAND_FONT, color=color, fontweight="bold" if bold else "normal", annotation_clip=False)


def plot_protocol(protocol, width, floor_only=True):
    runs = finalized_runs(protocol)
    if floor_only:  # Kevin 2026-10-21: the model columns show the floor variants only
        runs = {l: r for l, r in runs.items() if l in ("oracle", "heuristic", "static") or l.endswith("_floor")
                or l.startswith("static_")}
    global PROTOCOL
    PROTOCOL = protocol
    statics = list(STATIC_LABELS[protocol]) if any(l in runs for l in STATIC_LABELS[protocol]) else []
    columns = [("Oracle, heuristic, static" if statics else COLUMNS[0][0], COLUMNS[0][1] + statics)] + list(COLUMNS[1:])
    row_names = (["throughput"] if protocol == "smartbft" else []) + ["latency", "timeout"]
    fig, axes = plt.subplots(len(row_names), len(columns), figsize=(6 * len(columns), 3.6 * len(row_names)),
                             sharex=True, sharey="row", squeeze=False)
    series = {}
    for label, run in runs.items():
        try:
            data, run_onsets, run_end, played = run_series(run, protocol, width)
        except Exception as exc:  # a run with a broken artifact is reported, the figure still drawn
            print(f"{protocol}: {run.name}: {type(exc).__name__}: {exc}; skipped", file=sys.stderr)
            continue
        if not data:
            print(f"{protocol}: {run.name}: no client window inside the phases; skipped", file=sys.stderr)
            continue
        series[label] = (data, run_onsets, run_end, played)
    first = next(iter(series.values()), None)
    onsets, end = (first[1], first[2]) if first else ([], None)
    for label, (_, o, _, _) in series.items():
        if o != onsets:
            print(f"{protocol}: {runs[label].name}: phase onsets differ from the first run drawn", file=sys.stderr)
    # Every column: the oracle first, then its configurations in COLUMNS order; a slot keeps its row in the band
    # whether or not its run is there, so the number rows sit in the same order in every column and figure.
    slots = [["oracle"] + [l for l in labels if l != "oracle" and (not floor_only or l in ("heuristic", "static")
                                                                  or l.endswith("_floor") or l.startswith("static_"))]
             for _, labels in columns]
    tallest = max(len(col) for col in slots)
    for c, (title, _) in enumerate(columns):
        axes[0][c].set_title(title, pad=BAND_LIFT + 3 + BAND_STEP * tallest)  # above the tallest band, the same height in every column
        bands = {}  # row name -> [(k, data, col, colour)], drawn once the column's runs are in
        for k, label in enumerate(slots[c]):
            if label not in series:
                continue
            data, run_onsets, run_end, played = series[label]
            style = style_of(label)
            tag = fig_label(label)
            # Each window's mean is drawn at the window's start, so a change in the window after a phase onset
            # sits on the onset (at the centre it sat half a window past it).
            t = [b[0] for b in data]
            for r, name in enumerate(row_names):
                ax = axes[r][c]
                if name == "throughput":
                    ax.plot(t, [b[1] for b in data], **style)
                    bands.setdefault(name, []).append((k, data, 1, style["color"], label != "oracle", tag))
                elif name == "latency":
                    ax.plot(t, [b[2] for b in data], **style)
                    over_top(ax, data, LATENCY_TOP_MS.get(protocol, LATENCY_TOP_DEFAULT_MS), style)
                    bands.setdefault(name, []).append((k, data, 2, style["color"], label != "oracle", tag))
                else:
                    for j, pts in enumerate(played.values()):
                        ax.step([p[0] for p in pts], [p[1] for p in pts], where="post",
                                **{**style, "lw": style["lw"] if j == 0 else 0.5 * style["lw"]}, label=tag if j == 0 else None)
        for name, rows in bands.items():
            phase_band(axes[row_names.index(name)][c], rows, onsets + [end], len(slots[c]),
                       (lambda v: f"{v / 1000:.1f}k" if v >= 1000 else f"{v:.0f}") if name == "throughput"
                       else (lambda v: f"{v:.0f}"), best="min" if name == "latency" else "max",
                       title=None if c else {"throughput": "Phase mean\nthroughput (tps)", "latency": "Phase mean\nlatency (ms)"}[name])
        axes[-1][c].set_xlabel("Time (s)")  # since the first phase onset
        handles, names = axes[-1][c].get_legend_handles_labels()
        if handles:
            axes[-1][c].legend(handles, names, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=len(handles),
                               handlelength=1.5, handletextpad=0.4, columnspacing=0.8)  # one row (Kevin 2026-10-30), narrow enough for SBFT's seven
    for r, name in enumerate(row_names):
        axes[r][0].set_ylabel({"throughput": "Throughput (tps)", "latency": "Mean latency (ms)",
                               "timeout": "Timeout (ms)" + (": T, W thin" if protocol == "sbft" else "")}[name])
        if name == "latency":
            # Linear from 0 (Kevin 2026-10-26; log until then), shared by the row; the band's phase means are not cut.
            axes[r][0].set_ylim(0, LATENCY_TOP_MS.get(protocol, LATENCY_TOP_DEFAULT_MS))
            axes[r][0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    for row in axes:
        for ax in row:
            for i, (lo, hi) in enumerate(zip(onsets, onsets[1:] + ([end] if end else []))):
                if i % 2:
                    ax.axvspan(lo, hi, color=PHASE_SHADE, lw=0, zorder=0)
            if end:
                ax.set_xlim(0, end)
                ax.set_xticks(onsets + [end])  # one tick per phase onset (300 s apart in the real specs) and the end
            ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# The motivation figure's phase labels (Kevin 2026-10-28), PBFT's spec: p0 base, p1 base + hold D300, p2 base + hold D550,
# p3 5+50, p4 5+50 + hold D500, p5 5+150, p6 base.
# Drawn as "p<i>: <label>", the index a subscript counted from 1 (Kevin 2026-10-29; the specs count from p0); wrapped so no line is wider than its phase.
MOTIVATION_PHASES = ("No fault", "Small\nproposal delay", "Large\nproposal delay", "Small\nnetwork delay",
                     "Small network\ndelay, large\nproposal delay", "Large\nnetwork delay", "No fault")


MOTIVATION_TOP_MS = 1500  # the motivation figure's latency axis (Kevin 2026-10-28; 2,000 like the PBFT figure until then)


def plot_motivation(width):
    """results/plots/motivation.png (Kevin 2026-10-28): PBFT's mean latency over the run for the oracle and the three
    static timeouts (each its best draw, as in the protocol figures), one panel; no phase-mean band and plain labels
    (Kevin 2026-10-28)."""
    global PROTOCOL
    PROTOCOL = "pbft"
    runs = finalized_runs("pbft")
    labels = ["oracle"] + sorted(STATIC_LABELS["pbft"], key=lambda l: int(l.split("_")[1]))
    missing = [l for l in labels if l not in runs]
    if missing:
        raise SystemExit(f"motivation: no finalized PBFT run of {', '.join(missing)}")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    top, onsets, end = MOTIVATION_TOP_MS, None, None
    for label in labels:
        data, o, e, _ = run_series(runs[label], "pbft", width)
        if onsets is None:
            onsets, end = o, e
        elif o != onsets:
            raise SystemExit(f"motivation: {runs[label].name}: phase onsets differ from the oracle's")
        style = style_of(label)
        name = "Oracle" if label == "oracle" else f"Static {label.split('_')[1]} ms"
        ax.plot([b[0] for b in data], [b[2] for b in data], **style, label=name)
        over_top(ax, data, top, style)
    edges = onsets + [end]
    if len(edges) - 1 != len(MOTIVATION_PHASES):
        raise SystemExit(f"motivation: {len(edges) - 1} phases in the runs, {len(MOTIVATION_PHASES)} labels")
    for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
        if i % 2:
            ax.axvspan(lo, hi, color=PHASE_SHADE, lw=0, zorder=0)
        ax.annotate(f"p$_{i + 1}$: {MOTIVATION_PHASES[i]}", ((lo + hi) / 2, 1), xycoords=("data", "axes fraction"), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7, annotation_clip=False)
    ax.set_xlim(0, end)
    ax.set_xticks(edges)
    ax.set_ylim(0, top)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean latency (ms)")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=len(labels))  # one row (Kevin 2026-10-28)
    fig.tight_layout()
    return fig


SCAL_LABELS = ("oracle", "heuristic", "continuous_gp_pre_floor", "discrete_gp_pre_floor", "continuous_linucb_untr_floor",
               "discrete_linucb_untr_floor")


def plot_scalability():
    """results/plots/scalability.png (Kevin 2026-10-30): the sweep's latency as grouped bars, one group per replica
    count (the four-replica main table first), one bar per configuration: the oracle grey, the heuristic black,
    continuous models blue, discrete red, pretrained the dark shade and solid, untrained the lighter shade and hatched (as
    the protocol figures' lighter dashed lines); the mean over the phases and the draws, a thin bar spanning the draws where there are several."""
    global PROTOCOL
    PROTOCOL = "pbft"
    base = RESULTS / "scalability" / "pbft"
    groups = []  # (name, {label: [latency per draw]})
    e0 = {}
    for d in sorted((RESULTS / "pbft").iterdir()) if (RESULTS / "pbft").is_dir() else []:
        f = d / "row.json" if (d / "row.json").exists() else d / "draw.json"
        if f.exists():
            label = json.loads(f.read_text())["label"]
            if label in SCAL_LABELS:
                e0.setdefault(label, []).append(score(d)["lat"])
    if e0:
        groups.append(("4\nmain table", e0))
    for nd in sorted(base.iterdir(), key=lambda d: int(d.name[1:])) if base.is_dir() else []:
        g = {}
        for d in sorted(nd.iterdir()):
            if (d / "row.json").exists():
                label = json.loads((d / "row.json").read_text())["label"]
                if label in SCAL_LABELS:
                    g.setdefault(label, []).append(score(d)["lat"])
        if g:
            groups.append((nd.name[1:], g))
    if not groups:
        raise SystemExit("scalability: no runs")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    k, w = len(SCAL_LABELS), 0.8 / len(SCAL_LABELS)
    handles = []
    for j, label in enumerate(SCAL_LABELS):
        untrained = "_untr" in label
        if label == "oracle":
            color = ORACLE_COLOR
        elif label in OTHER_COLORS:
            color = OTHER_COLORS[label]
        else:  # the kind's colour, dark when pretrained, the lighter shade when untrained (as the line figures)
            color = KIND_COLORS[label.split("_", 1)[0]][1 if untrained else 0]
        hatch = "//" if untrained else ""
        xs, ys, err = [], [], [[], []]
        for i, (_, g) in enumerate(groups):
            if label not in g:
                continue
            v = g[label]
            xs.append(i + (j - (k - 1) / 2) * w); ys.append(sum(v) / len(v))
            err[0].append(ys[-1] - min(v)); err[1].append(max(v) - ys[-1])
        ax.bar(xs, ys, width=w * 0.92, color=color, hatch=hatch, edgecolor="white" if hatch else color, lw=0.5, zorder=2)
        ax.errorbar(xs, ys, yerr=err, fmt="none", ecolor="black", elinewidth=0.6, capsize=1.5, zorder=3)
        handles.append(matplotlib.patches.Patch(facecolor=color, hatch=hatch, edgecolor="white" if hatch else color))
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([name for name, _ in groups])
    ax.set_xlabel("Replicas (two per machine; the main table one per machine)")
    ax.set_ylabel("Latency (ms)")
    ax.grid(True, axis="y", lw=0.3, alpha=0.5, zorder=0)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.legend(handles, [fig_label(l) for l in SCAL_LABELS], fontsize=7, loc="upper left", ncol=3)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--protocol", choices=PROTOCOLS, help="one protocol (default: all)")
    ap.add_argument("--all", action="store_true", help="draw the no-floor model rows too (dash-dot pretrained, dotted untrained)")
    ap.add_argument("--window", type=int, default=20,
                    help="seconds averaged per point; must divide the 300 s phase length (default 20)")
    args = ap.parse_args()
    if args.window <= 0 or 300 % args.window:
        ap.error(f"--window {args.window}: must be a positive divisor of 300 (the phase length in seconds)")
    for protocol in ([args.protocol] if args.protocol else PROTOCOLS):
        global FLOOR_SHADES
        FLOOR_SHADES = args.all
        fig = plot_protocol(protocol, args.window, floor_only=not args.all)
        out = RESULTS / "plots" / f"{protocol}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)  # 110 until 2026-10-26 (Kevin: better resolution)
        fig.savefig(out.with_suffix(".pdf"))  # vector, for LaTeX
        plt.close(fig)
        print(out.relative_to(RESULTS.parent), out.with_suffix(".pdf").relative_to(RESULTS.parent))
    if args.protocol in (None, "pbft"):
        fig = plot_motivation(args.window)
        out = RESULTS / "plots" / "motivation.png"
        fig.savefig(out, dpi=300)
        fig.savefig(out.with_suffix(".pdf"))
        plt.close(fig)
        print(out.relative_to(RESULTS.parent), out.with_suffix(".pdf").relative_to(RESULTS.parent))
        fig = plot_scalability()
        out = RESULTS / "plots" / "scalability.png"
        fig.savefig(out, dpi=300)
        fig.savefig(out.with_suffix(".pdf"))
        plt.close(fig)
        print(out.relative_to(RESULTS.parent), out.with_suffix(".pdf").relative_to(RESULTS.parent))


if __name__ == "__main__":
    main()
