#!/usr/bin/env python3
"""results/scoring.py: per-run phase scores and the aligned per-phase tables RESULTS.md is built from (results_md.py).
score(run) runs the client judge (judge.py) on a run directory and adds, per phase, the throughput and the latency term
of R over the judge's own windows and the median timeout replica 0 played; scores are cached by run name in
results_cache.json (not tracked; rebuilt on first use). Moved from scripts/cloudlab/tmp/cal/r512/results_md.py on
2026-10-24, functions unchanged."""
import csv
import json
import re
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # keep results/ free of __pycache__
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from judge import CHUNK_S, EMPTY_P50_MS, SKIP_S, judge, phases as spec_phases, seconds  # noqa: E402

CACHE = HERE / "results_cache.json"


def display(label):
    if label == "oracle":
        return "oracle"
    if label == "heuristic":
        return "heuristic"
    m = re.match(r"(continuous|discrete)_(gp|linucb|ppo)_(pre|untr)(_floor)?$", label)
    if not m:
        return label
    return f"{m.group(1)}_{m.group(2)} {'pretrained' if m.group(3) == 'pre' else 'untrained'}{' + floor' if m.group(4) else ''}"


def tps_latency(run, out):
    """Per phase, over the judge's own window (onset + SKIP_S to the next onset): the mean throughput of the judged
    seconds and the mean of the chunk latencies the score uses (commit-weighted mean latency per CHUNK_S chunk, a chunk
    with no commits at EMPTY_P50_MS). The judge reports the first; the second is R's latency term."""
    warm, ats = spec_phases(run)
    env = dict(l.split("=", 1) for l in (run / "run.env").read_text().splitlines() if "=" in l)
    length = float(env["experiment_time_s"])
    rows = [r for r in seconds(run) if r[0] < length]
    end = min(length, max(r[0] for r in rows) + 1.0)
    offset = 0.5 if run.parent.name == "tendermint" or "tendermint" in str(run) else 0.0
    starts = [warm + a + offset for a in ats] + [end]
    res = []
    for i in range(len(ats)):
        sec = [r for r in rows if starts[i] + SKIP_S <= r[0] < starts[i + 1]]
        lats = []
        for k in range(0, len(sec) - CHUNK_S + 1, CHUNK_S):
            ch = sec[k:k + CHUNK_S]
            n = sum(r[1] for r in ch)
            lats.append(min(sum(r[1] * r[3] for r in ch) / n, EMPTY_P50_MS) if n else EMPTY_P50_MS)
        res.append((out[i]["tps"], sum(lats) / len(lats) if lats else float("nan")))
    return res


def score(run):
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    if run.name in cache and "lat" in cache[run.name]:
        return cache[run.name]
    _, out = judge(run)
    rows = list(csv.DictReader(open(run / "agent0.csv"))) if (run / "agent0.csv").exists() else []
    tcol = next((c for c in (rows[0].keys() if rows else []) if c.endswith("_used") and "timeout" in c), None)
    ats = [x["at"] for x in out]
    ends = ats[1:] + [10 ** 9]
    tl = tps_latency(run, out)
    phases = []
    for x, e, (tps, lat) in zip(out, ends, tl):
        v = [float(r[tcol]) for r in rows if tcol and x["at"] <= 20 * int(r["episode"]) < e and r.get(tcol)]
        phases.append({"R": x["R"], "z": int(x["stall"]), "T": statistics.median(v) if v else None, "tps": tps, "lat": lat})
    entry = {"mean": sum(x["R"] for x in out) / len(out), "phases": phases,
             "tps": sum(t for t, _ in tl) / len(tl), "lat": sum(l for _, l in tl) / len(tl)}
    cache[run.name] = entry
    CACHE.write_text(json.dumps(cache, indent=0))
    return entry


def cell(ph):
    return f"{ph['R']:+.2f}" + (f" z{ph['z']}" if ph["z"] else "")


def inner_mean(phases, key):
    """Mean of a per-phase value over the phases except the first and the last (the base phases that open and
    close every spec; Kevin, 2026-10-21); the plain mean when there are fewer than three phases."""
    inner = phases[1:-1] if len(phases) >= 3 else phases
    return sum(ph[key] for ph in inner) / len(inner)


def table(rows):
    n = max(len(r[2]) for r in rows)
    heads = [f"p{i}" for i in range(n)]
    w0 = max(len("configuration"), max(len(r[0]) for r in rows))
    ws = [max(len(h), max(len(cell(r[2][i])) for r in rows if i < len(r[2]))) for i, h in enumerate(heads)]
    line = lambda c, m, k, cells: ("| " + c.ljust(w0) + " | " + m.rjust(6) + " | " + k.rjust(7) + " | "
                                   + " | ".join(x.rjust(ws[i]) for i, x in enumerate(cells)) + " |")
    out = [line("configuration", "mean R", "inner R", heads),
           "|" + "-" * (w0 + 2) + "|" + "-" * 8 + "|" + "-" * 9 + "|" + "|".join("-" * (w + 2) for w in ws) + "|"]
    out += [line(r[0], f"{r[1]:+.3f}", f"{inner_mean(r[2], 'R'):+.3f}", [cell(ph) for ph in r[2]]) for r in rows]
    return "\n".join(out) + "\n"


def value_table(rows, key, fmt, total_key):
    """Like table(), one value per phase."""
    if not rows:
        return ""
    n = max(len(r[2]) for r in rows)
    heads = [f"p{i}" for i in range(n)]
    w0 = max(len("configuration"), max(len(r[0]) for r in rows))
    cells = [[fmt(ph[key]) for ph in r[2]] for r in rows]
    means = [fmt(r[4][total_key]) for r in rows]
    inners = [fmt(inner_mean(r[2], key)) for r in rows]
    ws = [max(len(h), max(len(c[i]) for c in cells if i < len(c))) for i, h in enumerate(heads)]
    wm = max(len("mean"), max(len(m) for m in means))
    wi = max(len("inner"), max(len(m) for m in inners))
    line = lambda c, m, k, cs: ("| " + c.ljust(w0) + " | " + m.rjust(wm) + " | " + k.rjust(wi) + " | "
                                + " | ".join(x.rjust(ws[i]) for i, x in enumerate(cs)) + " |")
    out = [line("configuration", "mean", "inner", heads),
           "|" + "-" * (w0 + 2) + "|" + "-" * (wm + 2) + "|" + "-" * (wi + 2) + "|" + "|".join("-" * (w + 2) for w in ws) + "|"]
    out += [line(r[0], m, k, c) for r, m, k, c in zip(rows, means, inners, cells)]
    return "\n".join(out) + "\n"
