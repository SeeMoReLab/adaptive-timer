#!/usr/bin/env python3
"""results/results_md.py [--write]: results/RESULTS.md from the finalized runs in results/<protocol>/<run>/, one per
configuration (26 per protocol: the oracle, the heuristic and the 24 model configurations). Each run carries the
row.json that scripts/cloudlab/tmp/cal/restart/finalize.py wrote when it accepted the run (configuration, host set,
frozen code, prior md5, sim-gate verdict, validate.py flags); a run without one, or two runs of one configuration,
is an error. Earlier draws of a configuration carry draw.json instead (finalize.py's REPEATS; Kevin 2026-10-25: a
second draw of every configuration; runs named <run>_<draw>): per protocol a summary of mean R and inner R over the
draws (min, max, mean, each draw), then every draw as its own row. Scoring is the client judge (results/judge.py) through results/scoring.py. The per-protocol header reads the constants installed in learning_agent/. Without --write prints the text."""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True  # keep results/ free of __pycache__
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "learning_agent"))
from scoring import EMPTY_P50_MS, STATIC_LABELS, display, inner_mean, score, table, value_table  # noqa: E402  (results/scoring.py, judge.py)
import agent_pb2  # noqa: E402
import continuous_bandit as cb  # noqa: E402
from models import get_timeout_values_ms  # noqa: E402
from reward_utils import reward_normalizer_tps  # noqa: E402

PROTOCOLS = ("pbft", "sbft", "smartbft", "tendermint", "hotstuff", "autobahn")
NAMES = {"tendermint": "Tendermint", "autobahn": "Autobahn", "pbft": "PBFT", "sbft": "SBFT", "hotstuff": "HotStuff",
         "smartbft": "SmartBFT"}
MODEL_LABELS = [f"{m}_{t}{f}" for m in ("continuous_gp", "discrete_gp", "continuous_linucb", "discrete_linucb",
                                          "continuous_ppo", "discrete_ppo") for t in ("pre", "untr") for f in ("", "_floor")]
LABELS = ("oracle", "heuristic", *MODEL_LABELS)
PHASES = {  # the real specs (config/<p>/failure_spec.xml), frozen
    "pbft": "p0 base, p1 base + hold D300, p2 base + hold D550, p3 5+50, p4 5+50 + hold D500, p5 5+150, p6 base",
    "sbft": "p0 base, p1 base + hold D300, p2 5+50, p3 5+50 + straggler 30 + hold D550, p4 5+50 + straggler 300 + "
            "hold D350, p5 5+150 + straggler 30, p6 5+150 + straggler 450, p7 base (p3 and p5 straggler 60 until "
            "2026-10-22)",
    "smartbft": "p0 base, p1 base + hold 300, p2 base + hold 600, p3 5+50, p4 5+50 + hold 300, p5 5+150, p6 base",
    "tendermint": "p0 base, p1 base + slow proposer, p2 5+50, p3 5+50 + slow proposer, p4 5+150, p5 base",
    "hotstuff": "p0 base, p1 base + slow leader, p2 5+50, p3 5+50 + slow leader, p4 5+150, p5 base",
    "autobahn": "p0 base, p1 base + slow leader, p2 5+50, p3 5+50 + slow leader, p4 5+150, p5 base",
}
NOTES = {  # per-protocol state outside the rows, to drop when its rows run
    "pbft": "Rows under restart-v4 (2026-10-22). The earlier restart-v3 rows were removed (not kept in this repository): that freeze dated the "
            "request timer from the proposal tick that admits a request rather than from its receipt, so T 250 never "
            "expired at 5+50 and the ladder did not separate (DESIGN.md section 2). Recalibrated on restart-cal8.",
    "sbft": "Rows under restart-v4 (2026-10-22), recalibrated on restart-cal9 with the receipt-dated timer and a 30 ms "
            "straggler in p3 and p5 (60 ms until 2026-10-22, under which 650/150 lost p3 to 650/10). The SBFT simulator "
            "ranks p5 only on a tie (DESIGN.md section 8).",
    "tendermint": "Oracle and discrete arms changed (Kevin, 2026-10-23): the oracle is 150 ms in p2 and p3, the best measured "
                  "p3 on prefix cells of the compressed spec (125 +0.17 with stalls, 150 +0.54, 175 +0.53; 200 +0.51 on the "
                  "full spec), and the discrete arms are 100/150/400 from restart-v5 (100/200/400 before; those discrete "
                  "rows and the 200 ms oracle draws were removed). The oracle row ran under restart-v4 from "
                  "the same schedule (a file outside the code hash). The heuristic and the continuous rows opened at 200 ms, "
                  "the earlier middle rung; the opening covers only the first decision, in p0, where every timeout ties. "
                  "Second draw (restart-v7, Kevin 2026-10-25: kept and averaged with the first): every row opened at 150 ms, "
                  "and the pretrained continuous GP and PPO rows loaded the restart-v5 priors, which differ from the "
                  "restart-v2 priors their first draw loaded (the opening is the simulator's starting action); the "
                  "LinUCB priors are identical.",
}


HEAD = """# Results: learned timeouts against the oracle on 5 machines

Updated {stamp}; FREEZE {freeze} ({hash}), each section naming the frozen code its rows ran under. Each configuration has one or more draws (full-spec runs, four replicas and the client on five CloudLab machines; results/<protocol>/<run>_<draw>/). Per protocol, three summaries give each configuration's R, throughput and latency (whole run and inner) as the mean over its draws and each draw's value, and the tables after them give every draw as its own row. Each run is scored by the client judge: R = tps / N minus the client's latency in seconds (the commit-weighted mean of its per-second p50, a chunk with no commits at 10 s) per 20 s chunk, averaged per phase (the first 20 s after each onset skipped; the warm-up before the first onset and anything after the last phase's 300 s, the run's planned end, are outside every window), N the protocol's offered rate. Mean R is over the spec's phases; inner R (and the inner throughput and latency) leave out the first and the last phase, the base phases that open and close every spec (Kevin, 2026-10-21). z = seconds without a commit in the phase. Rows are sorted by their configuration's mean R over its draws, then by draw; the set (A, B or C) a draw ran on is in brackets (every set pulls rows from one queue per protocol). Pretrained = the simulator prior installed (its sim-gate verdict in brackets); every model learns online from the run's own rewards. Floor = the protocol's latency floor (multiplier x a hold-free signal + offset). Every engine and model opens at the ladder's middle rung. Generated by results/results_md.py from the runs in results/<protocol>/."""


def header(p: str) -> str:
    proto = getattr(agent_pb2, "PROTOCOL_" + p.upper())
    h, f = cb.HEURISTIC_SPECS[proto], cb.floor_spec(proto)
    ladder = "/".join(str(v) for v in get_timeout_values_ms(proto))
    if p == "sbft":
        from models import SBFT_SLOW_PATH_VALUES_MS
        ladder += " x " + "/".join(str(v) for v in SBFT_SLOW_PATH_VALUES_MS)
    rate = f"{int(reward_normalizer_tps(p)):,}" + (" (closed loop, 512 terminals)" if p == "smartbft" else "")
    floor = f"{f.multiplier:g} x {f.signal}" + (f" + {f.offset_ms:g} ms" if f.offset_ms else "")
    return (f"N {rate}; ladder {ladder}, opening {'/'.join(str(v) for v in cb.opening_action_ms(proto))}; heuristic "
            f"k {h.multiplier:g} on {h.signal}; floor {floor}; range {h.bounds_ms[0]}-{h.bounds_ms[1]} ms.\n\n"
            f"Phases: {PHASES[p]}.")


def mean_score(scs: list) -> dict:
    """score() averaged over draws: mean R, throughput and latency per phase and overall, z rounded, T the median."""
    n = len(scs)
    if len({len(sc["phases"]) for sc in scs}) != 1:
        raise SystemExit("draws of one configuration differ in their phase count")
    avg = lambda xs: sum(xs) / len(xs)
    phases = []
    for i in range(len(scs[0]["phases"])):
        ph = [sc["phases"][i] for sc in scs]
        ts = sorted(x["T"] for x in ph if x["T"] is not None)
        phases.append({"R": avg([x["R"] for x in ph]), "z": round(avg([x["z"] for x in ph])),
                       "T": ts[len(ts) // 2] if ts else None, "tps": avg([x["tps"] for x in ph]),
                       "lat": avg([x["lat"] for x in ph])})
    return {"mean": avg([sc["mean"] for sc in scs]), "phases": phases, "tps": avg([sc["tps"] for sc in scs]),
            "lat": avg([sc["lat"] for sc in scs]), "draws": n}


def rows_for(p: str) -> tuple:
    """(one table row per run, flag lines, frozen-code names, {label: (mean over its draws, [(draw, set, score)])}) from
    results/<p>/<run>_<draw>/row.json (the latest draw) and draw.json (earlier draws). Run rows are ordered by their
    configuration's mean R over its draws, then by draw."""
    freezes, seen, draws = set(), {}, {}
    folder = HERE / p
    for d in sorted(folder.iterdir()) if folder.is_dir() else []:
        if not d.is_dir():
            continue
        kind = "row.json" if (d / "row.json").exists() else "draw.json"
        if not (d / kind).exists():
            raise SystemExit(f"{d.relative_to(ROOT)}: no row.json or draw.json (finalize.py did not accept it)")
        row = json.loads((d / kind).read_text())
        if row["protocol"] != p or (row["label"] not in LABELS and row["label"] not in STATIC_LABELS[p]):
            raise SystemExit(f"{d.relative_to(ROOT)}: {kind} names {row['protocol']} {row['label']}")
        if "draw" not in row:
            raise SystemExit(f"{d.relative_to(ROOT)}: {kind} has no draw number (run finalize.py --apply)")
        if kind == "row.json":
            if row["label"] in seen:
                raise SystemExit(f"{p}: two rows of {row['label']}: {seen[row['label']]} and {d.name}")
            seen[row["label"]] = d.name
        draws.setdefault(row["label"], []).append((row["draw"], row, d))
        freezes.add(f"{row['freeze_name']} ({row['code_hash'][:12]})")
    if set(draws) - set(seen):
        raise SystemExit(f"{p}: draws without a row: {sorted(set(draws) - set(seen))}")
    configs, rows = {}, []
    for label, ds in draws.items():
        ds.sort(key=lambda x: x[0])
        if [n for n, _, _ in ds] != list(range(1, len(ds) + 1)):
            raise SystemExit(f"{p} {label}: draws {[n for n, _, _ in ds]}, not 1..{len(ds)}")
        scs = [(n, r["set"][-1], score(d)) for n, r, d in ds]
        configs[label] = (mean_score([sc for _, _, sc in scs]), scs)
        for (n, r, d), (_, s, sc) in zip(ds, scs):
            name = f"{display(label)}, draw {n} [{s}]" + (f" (gate {r['gate']})" if r["gate"] else "")
            rows.append((name, sc["mean"], sc["phases"], label, sc, d.name, r["flags"], configs[label][0]["mean"], n))
    rows.sort(key=lambda r: (-r[7], r[8]))
    flags = [f"{r[0]}: {f}" for r in rows for f in r[6]]  # in table order
    return rows, flags, sorted(freezes), configs


def summary_table(configs: dict, key: str, name: str, fmt, name_of=display) -> str:
    """Per configuration (sorted by its mean R over the draws): the value over the whole run and its inner value
    (score()'s mean/tps/lat and the inner mean of the phases' R/tps/lat), as the mean over the draws and each draw's
    value (Kevin 2026-10-27: no min/max)."""
    per = sorted({k for _, scs in configs.values() for k, _, _ in scs})  # the draw numbers present, as columns
    whole = (lambda sc: sc["mean"]) if key == "R" else (lambda sc: sc[key])
    cells = []
    for label, (_, scs) in sorted(configs.items(), key=lambda kv: -kv[1][0]["mean"]):
        cols = []
        for value in (whole, lambda sc: inner_mean(sc["phases"], key)):
            v = {k: value(sc) for k, _, sc in scs}
            cols += [fmt(sum(v.values()) / len(v)), *[fmt(v[k]) if k in v else "" for k in per]]
        cells.append((name_of(label), *cols))
    head = ("configuration", f"{name} mean", *[f"{name} d{k}" for k in per], "inner mean", *[f"inner d{k}" for k in per])
    return aligned(head, cells)


STUDIES = [  # results/studies/pbft/<folder> (runs/2026-10-23-pbft-studies/spec.md): folder, title, what changes, label prefix
    ("dense", "E1 dense ladder", "discrete ladder 175/250/400/550/850/1200/1800: seven rungs, the oracle's three among them", "e1_"),
    ("fibonacci", "E2 Fibonacci ladder", "discrete ladder 144/233/377/610/987/1597/2584: seven rungs from a rule, no calibration input", "e2_"),
    ("sparse", "E3 too few rungs", "the E3 spec (results/studies/pbft/sparse/failure_spec_e3_r1.xml: p0 base, p1 base + D300, p2 5+50 + D150, p3 5+50 + D500, "
     "p4 5+150 + D350, p5 5+150, p6 base), whose phases need five timeouts (250/350/550/800/1200); the agent has 250/550/1200, "
     "so p2 (350) and p4 (800) are off its ladder. Compared with the 3-rung oracle, the best the agent's ladder allows", "e3_"),
    ("local", "E4 no sharing", "learning mode: each replica decides from its own report instead of the shared median", "e4_"),
    ("nohold", "E5 no-hold calibration", "the simulator fitted on blind cells without proposal holds (view-change overhead 55 ms "
     "against 161), the pretrained priors trained on it and the heuristic's k from it (3.06 against 3.96)", "e5_"),
    ("window", "E6 floor window", "the latency floor over 4 or 16 committing windows (their median) instead of the last one", "e6w"),
    ("poisoning", "E7 data poisoning", "replica 0 (the first f) shares random reports in every phase (results/studies/pbft/poisoning/"
     "failure_spec_e7.xml, the main spec plus <agent><dirtyReport>): each feature of its shared report is its own value scaled "
     "by a random 0.4 to 2.0 (2.2 for the p90/p95 percentiles), its counts drawn from 0 to 3 times their value; the reward it "
     "shares is untouched. The oracle reads no report, so the comparison is each row against E0's same row", "e7_"),
    ("nonuniform", "E8 non-uniform network", "a static extra one-way delay per replica pair on top of every phase "
     "(results/studies/pbft/nonuniform/failure_spec_e8.xml): the replicas sit on a line at 0, 0, 10 and 30 ms, a pair's extra "
     "is the distance between them, so one-way delays are 5 to 35 ms at base, 55 to 85 at 5+50 and 155 to 185 at 5+150. "
     "The rows run in sharing mode and, marked \"no sharing\", in learning mode (each replica from its own report, as E4). "
     "Compared with E8's own oracle (the main oracle schedule on this spec), since the spec differs", "e8_"),
]
STUDY_LABELS = {
    "dense": [l for l in MODEL_LABELS if l.startswith("discrete")],
    "fibonacci": [l for l in MODEL_LABELS if l.startswith("discrete")],
    "sparse": ["oracle5", "oracle3"] + [l for l in MODEL_LABELS if l.startswith("discrete")],
    "local": ["heuristic"] + MODEL_LABELS,
    "nohold": ["heuristic"] + [l for l in MODEL_LABELS if "_pre" in l],
    "window": [f"{w}_{l}" for w in ("4", "16") for l in MODEL_LABELS if l.endswith("_floor")],
    "poisoning": ["heuristic"] + [l for l in MODEL_LABELS if l.endswith("_floor")],
    "nonuniform": ["oracle", "heuristic", "learning_heuristic"] + [l for l in MODEL_LABELS if l.endswith("_floor")] + ["learning_" + l for l in [l for l in MODEL_LABELS if l.endswith("_floor")]],
}


def aligned(head: tuple, cells: list) -> str:
    """A Markdown table padded so its columns line up as plain text (first column left, the rest right)."""
    w = [max(len(str(c[i])) for c in [head, *cells]) for i in range(len(head))]
    fmt = lambda c: "| " + " | ".join(str(x).ljust(w[i]) if i == 0 else str(x).rjust(w[i]) for i, x in enumerate(c)) + " |"
    rule = "|" + "|".join("-" * (w[i] + 1) + ("-" if i == 0 else ":") for i in range(len(head))) + "|"
    return "\n".join([fmt(head), rule, *map(fmt, cells)]) + "\n"


def study_name(folder: str, label: str) -> str:
    if folder == "sparse" and label.startswith("oracle"):
        return "oracle, 5 rungs" if label == "oracle5" else "oracle, 3 rungs (the agent's ladder)"
    if folder == "window":
        w, base = label.split("_", 1)
        return f"{display(base)}, window {w}"
    if folder == "nonuniform" and label.startswith("learning_"):
        return f"{display(label.removeprefix('learning_'))}, no sharing"
    return display(label)


def studies_part(e0: dict) -> str:
    """The PBFT studies: per study its runs (results/studies/pbft/<folder>/<run>_<draw>/row.json, calibration cells left
    out; Kevin 2026-10-27: draws 2 and 3 of every study row) against E0, the PBFT table's row of the same configuration
    (E3: against its own 3-rung oracle), each as the mean over its draws."""
    base = HERE / "studies" / "pbft"
    out = ["\n## PBFT studies\n\nPBFT only (runs/2026-10-23-pbft-studies/spec.md; draws 2 and 3: "
           "runs/2026-10-27-study-draws/spec.md). Each study changes one thing against E0, "
           "the PBFT table above (same engine, priors array-identical), and runs every variant where it "
           "applies. Scored as above; a configuration's values are the mean over its draws (E0's too), and each \"vs\" is the "
           "configuration's value (mean R, throughput, latency) minus E0's for the same configuration, e.g. E1 discrete_gp "
           "pretrained minus the PBFT table's discrete_gp pretrained, not the oracle (E3 and E8: minus their own oracle, "
           "since their specs differ). Throughput and latency are the means over the phases. The summaries give each draw's "
           "value, and the per-phase tables every draw as its own row. Calibration cells are in each study's calibration/ folder.\n"]
    for folder, title, what, prefix in STUDIES:
        draws = {}
        for d in sorted((base / folder).iterdir()) if (base / folder).is_dir() else []:
            if not (d / "row.json").exists():
                continue
            row = json.loads((d / "row.json").read_text())
            if "draw" not in row:
                raise SystemExit(f"{d.relative_to(ROOT)}: row.json has no draw number (collect_draws.py --apply)")
            draws.setdefault(row["label"][len(prefix):], []).append((row["draw"], row, d))
        out.append(f"\n### {title}\n\nChanges: {what}.\n\n")
        if not draws:
            out.append("Pending: every row.\n")
            continue
        configs, rows = {}, []
        for label, ds in draws.items():
            ds.sort(key=lambda x: x[0])
            if [n for n, _, _ in ds] != list(range(1, len(ds) + 1)):
                raise SystemExit(f"{folder} {label}: draws {[n for n, _, _ in ds]}, not 1..{len(ds)}")
            scs = [(n, r["set"][-1], score(d)) for n, r, d in ds]
            configs[label] = (mean_score([sc for _, _, sc in scs]), scs)
            for (n, r, d), (_, st, sc) in zip(ds, scs):
                rows.append((f"{study_name(folder, label)}, draw {n} [{st}]", sc["mean"], sc["phases"], label, sc, d.name,
                             r["flags"], configs[label][0]["mean"], n))
        rows.sort(key=lambda r: (-r[7], r[8]))
        own = {"sparse": "oracle3", "nonuniform": "oracle"}.get(folder)  # a study on its own spec: against its own oracle
        ref = {own: configs[own][0]} if own in configs else ({} if own else e0)
        cells = []
        for label, (m, scs) in sorted(configs.items(), key=lambda kv: -kv[1][0]["mean"]):
            key = own if own else (label.split("_", 1)[1] if folder == "window" else label)
            v = None if label == own else ref.get(key)
            cells.append((study_name(folder, label), str(len(scs)), f"{m['mean']:+.3f}",
                          "" if v is None else f"{m['mean'] - v['mean']:+.3f}",
                          f"{m['tps']:,.0f}", "" if v is None else f"{m['tps'] - v['tps']:+,.0f}",
                          f"{m['lat']:,.0f}", "" if v is None else f"{m['lat'] - v['lat']:+,.0f}"))
        nm = lambda l, f=folder: study_name(f, l)
        out.append(aligned(("configuration", "draws", "mean R", "vs", "tps", "vs", "latency ms", "vs"), cells)
                   + "\nSummary over the draws (R):\n\n" + summary_table(configs, "R", "R", lambda v: f"{v:+.3f}", nm)
                   + "\nThroughput (tx/s, mean over the phases; N 20,000):\n\n"
                   + summary_table(configs, "tps", "tps", lambda v: f"{v:,.0f}", nm)
                   + "\nLatency (ms, mean over the phases):\n\n" + summary_table(configs, "lat", "lat", lambda v: f"{v:,.0f}", nm)
                   + "\nEvery draw, per phase:\n\n" + table(rows))
        out.append("\nThroughput (committed tx/s, as above; N 20,000):\n\n" + value_table(rows, "tps", lambda v: f"{v:,.0f}", "tps"))
        out.append("\nLatency (ms, as above):\n\n" + value_table(rows, "lat", lambda v: f"{v:,.0f}", "lat"))
        pending = [study_name(folder, l) for l in STUDY_LABELS[folder] if l not in configs]
        if pending:
            out.append(f"\nPending ({len(pending)}): {', '.join(pending)}.\n")
    return "".join(out)


SCAL_LABELS = ["oracle", "heuristic", "continuous_gp_pre_floor", "discrete_gp_pre_floor", "discrete_linucb_untr_floor",
               "continuous_linucb_untr_floor"]


def scalability_part(e0: dict) -> str:
    """The PBFT scalability sweep (runs/2026-10-28-pbft-scalability/spec.md; Kevin 2026-10-30): results/scalability/pbft/
    n<N>/<run>_<draw>/row.json, the real spec and the main oracle schedule at N replicas on N/2 machines, two per machine,
    the client on its own machine, on the scalability draft of the engine (changes 1-8 and 10 of the spec; not frozen,
    each run's code hash recorded) with the n=4 priors unchanged. Per size, the six rows against E0, the PBFT table's row
    of the same configuration at four replicas on the frozen engine, each as the mean over its draws; then a
    cross-size summary (mean R per size)."""
    base = HERE / "scalability" / "pbft"
    out = ["\n## PBFT scalability\n\nPBFT only (runs/2026-10-28-pbft-scalability/spec.md). The real spec and the main oracle "
           "schedule at N replicas, two per machine on N/2 machines plus the client's machine (one draw = one run of "
           "2,110 s), on the scalability draft of the engine (changes 1-8 and 10 of the spec: measured recovery budget, "
           "striped client, NewView by digests, one held leader per tick, single-source fetches, and the 32 MiB channel "
           "warm-up at start that pre-grows gRPC 1.51's flow-control windows, without which a never-led leader's first "
           "4.2 MB pre-prepare took 6-8 round trips at 5+50 and the oracle's p4 stormed at every size above 4), the "
           "pretrained rows loading the n=4 priors unchanged, the rate 20,000 throughout. The draft is not frozen: each "
           "row.json records the run's code hash. Scored as above; \"vs\" is the configuration's value minus E0's, the "
           "PBFT table's row of the same configuration at four replicas on the frozen engine (mean over draws on both "
           "sides). Rows are the oracle, the heuristic, and the best pretrained and untrained model of each action "
           "space, floored.\n"]
    sizes = {}
    for nd in sorted(base.iterdir(), key=lambda d: int(d.name[1:])) if base.is_dir() else []:
        if not nd.is_dir() or not nd.name.startswith("n"):
            continue
        draws = {}
        for d in sorted(nd.iterdir()):
            if not (d / "row.json").exists():
                continue
            row = json.loads((d / "row.json").read_text())
            draws.setdefault(row["label"], []).append((row["draw"], row, d))
        sizes[int(nd.name[1:])] = draws
    if not sizes:
        return out[0] + "\nPending: every row.\n"
    hashes, grid = set(), {}
    for n, draws in sizes.items():
        configs, rows = {}, []
        for label, ds in draws.items():
            ds.sort(key=lambda x: x[0])
            # A draw may land before a lower one (2026-10-31 pipeline: 28 replicas draw 3 before draw 2), so only
            # distinct draw numbers are required here; the columns carry the draw numbers.
            if len({k for k, _, _ in ds}) != len(ds):
                raise SystemExit(f"scalability n{n} {label}: draws {[k for k, _, _ in ds]} repeat a draw")
            scs = [(k, r["set"], score(d)) for k, r, d in ds]
            configs[label] = (mean_score([sc for _, _, sc in scs]), scs)
            for (k, r, d), (_, st, sc) in zip(ds, scs):
                hashes.add(r["code_hash"][:12])
                rows.append((f"{display(label)}, draw {k} [{st}]", sc["mean"], sc["phases"], label, sc, d.name,
                             r["flags"], configs[label][0]["mean"], k))
            grid[(label, n)] = configs[label][0]
        rows.sort(key=lambda r: (-r[7], r[8]))
        cells = []
        for label, (m, scs) in sorted(configs.items(), key=lambda kv: -kv[1][0]["mean"]):
            v = e0.get(label)
            cells.append((display(label), str(len(scs)), f"{m['mean']:+.3f}", "" if v is None else f"{m['mean'] - v['mean']:+.3f}",
                          f"{m['tps']:,.0f}", "" if v is None else f"{m['tps'] - v['tps']:+,.0f}",
                          f"{m['lat']:,.0f}", "" if v is None else f"{m['lat'] - v['lat']:+,.0f}"))
        out.append(f"\n### {n} replicas on {n // 2} machines\n\n"
                   + aligned(("configuration", "draws", "mean R", "vs E0", "tps", "vs", "latency ms", "vs"), cells)
                   + "\nSummary over the draws (R):\n\n" + summary_table(configs, "R", "R", lambda v: f"{v:+.3f}")
                   + "\nLatency (ms, mean over the phases):\n\n" + summary_table(configs, "lat", "lat", lambda v: f"{v:,.0f}")
                   + "\nEvery draw, per phase:\n\n" + table(rows))
        out.append("\nThroughput (committed tx/s, as above; N 20,000):\n\n" + value_table(rows, "tps", lambda v: f"{v:,.0f}", "tps"))
        out.append("\nLatency (ms, as above):\n\n" + value_table(rows, "lat", lambda v: f"{v:,.0f}", "lat"))
        pending = [display(l) for l in SCAL_LABELS if l not in configs]
        if pending:
            out.append(f"\nPending ({len(pending)}): {', '.join(pending)}.\n")
    ns = sorted(sizes)

    def across(value, fmt, draws=False) -> str:  # one table: configurations x sizes, the mean over the draws, E0 first
        cells = []
        for label in SCAL_LABELS:
            v = e0.get(label)
            cells.append((display(label), "" if v is None else fmt(value(v)),
                          *[fmt(value(grid[(label, n)])) + (f" ({grid[(label, n)]['draws']})" if draws else "")
                            if (label, n) in grid else "" for n in ns]))
        return aligned(("configuration", "E0 (4)", *[f"{n}" for n in ns]), cells)
    whole = lambda key: (lambda m: m["mean"] if key == "R" else m[key])
    inner = lambda key: (lambda m: inner_mean(m["phases"], key))
    r3, i0 = (lambda v: f"{v:+.3f}"), (lambda v: f"{v:,.0f}")
    out.append("\n### Across sizes\n\nPer size, the mean over the draws; E0 is the four-replica PBFT table on the frozen "
               "engine (its draws). Inner leaves out the first and the last phase, the base phases that open and close the "
               "spec.\n\nR (draw count in brackets):\n\n" + across(whole("R"), r3, draws=True)
               + "\nInner R:\n\n" + across(inner("R"), r3)
               + "\nThroughput (committed tx/s, mean over the phases; N 20,000):\n\n" + across(whole("tps"), i0)
               + "\nInner throughput:\n\n" + across(inner("tps"), i0)
               + "\nLatency (ms, mean over the phases):\n\n" + across(whole("lat"), i0)
               + "\nInner latency:\n\n" + across(inner("lat"), i0)
               + f"\nRows ran under the draft's code hashes {', '.join(sorted(hashes))} (the harness's client-route and "
               "stale-route fixes of 2026-10-29 18:58 changed the hash between the 28-replica oracle and the rows after it; "
               "the engine is the same).\n")
    return "".join(out)


def main(write: bool) -> None:
    freeze = dict(l.split("=", 1) for l in (ROOT / "FREEZE").read_text().splitlines() if "=" in l)
    parts = [HEAD.format(stamp=datetime.now().strftime("%Y-%m-%d %H:%M"), freeze=freeze["freeze_name"],
                         hash=freeze["code_hash"][:12])]
    for p in PROTOCOLS:
        rows, flags, freezes, configs = rows_for(p)
        parts.append(f"\n## {NAMES[p]}\n\n{header(p)}\n\n")
        if p in NOTES:
            parts.append(NOTES[p] + "\n\n")
        if not rows:
            parts.append("Pending: every row.\n")
            continue
        n = int(reward_normalizer_tps(p))
        parts.append(f"Rows ran under frozen code {', '.join(freezes)}.\n\nSummary over the draws (d1, d2, ... are the "
                     "draws by start time; mean = their mean; inner leaves out the first and the last phase; rows by mean R):"
                     "\n\nR:\n\n" + summary_table(configs, "R", "R", lambda v: f"{v:+.3f}")
                     + f"\nThroughput (tx/s, mean over the phases; N {n:,}):\n\n"
                     + summary_table(configs, "tps", "tps", lambda v: f"{v:,.0f}")
                     + "\nLatency (ms, mean over the phases):\n\n" + summary_table(configs, "lat", "lat", lambda v: f"{v:,.0f}")
                     + "\nEvery draw, per phase:\n\n" + table(rows))
        parts.append(f"\nThroughput (committed tx/s at the client, mean over the judged seconds of each phase; N {n:,}):\n\n"
                     + value_table(rows, "tps", lambda v: f"{v:,.0f}", "tps"))
        parts.append(f"\nLatency (ms, mean over the judged 20 s chunks of the client's commit-weighted p50; a chunk "
                     f"with no commits counts {EMPTY_P50_MS:,.0f} ms, as in R):\n\n"
                     + value_table(rows, "lat", lambda v: f"{v:,.0f}", "lat"))
        if flags:
            parts.append("\nFlags (validate.py; engine defects kept by D16, the row stands):\n\n"
                         + "".join(f"- {f}\n" for f in flags))
        have = {r[3] for r in rows}
        pending = [display(l) for l in LABELS if l not in have]
        if pending:
            parts.append(f"\nPending: {', '.join(pending)}.\n")
    e0 = {label: mean for label, (mean, _) in rows_for("pbft")[3].items()}
    parts.append(studies_part(e0))
    parts.append(scalability_part(e0))
    text = "".join(parts)
    if write:
        (HERE / "RESULTS.md").write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main("--write" in sys.argv)
