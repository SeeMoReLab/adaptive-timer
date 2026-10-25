#!/usr/bin/env python3
"""results/results_md.py [--write]: results/RESULTS.md from the finalized runs in results/<protocol>/<run>/, one per
configuration (26 per protocol: the oracle, the heuristic and the 24 model configurations). Each run carries the
row.json that scripts/cloudlab/tmp/cal/restart/finalize.py wrote when it accepted the run (configuration, host set,
frozen code, prior md5, sim-gate verdict, validate.py flags); a run without one, or two runs of one configuration,
is an error (finalize.py moves the extra to archive/<protocol>/). An earlier draw of a configuration rerun on purpose
carries draw.json instead (finalize.py's REPEATS) and is listed, with the row, under "Repeated draws". Scoring is the client judge (results/judge.py)
through results/scoring.py. The per-protocol header reads the constants installed in learning_agent/. Without --write prints the text."""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True  # keep results/ free of __pycache__
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "learning_agent"))
from scoring import EMPTY_P50_MS, display, score, table, value_table  # noqa: E402  (results/scoring.py, judge.py)
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
    "pbft": "Rows under restart-v4 (2026-10-22). The earlier restart-v3 rows are in archive/pbft/: that freeze dated the "
            "request timer from the proposal tick that admits a request rather than from its receipt, so T 250 never "
            "expired at 5+50 and the ladder did not separate (DESIGN.md section 2). Recalibrated on restart-cal8.",
    "sbft": "Rows under restart-v4 (2026-10-22), recalibrated on restart-cal9 with the receipt-dated timer and a 30 ms "
            "straggler in p3 and p5 (60 ms until 2026-10-22, under which 650/150 lost p3 to 650/10). The SBFT simulator "
            "ranks p5 only on a tie (DESIGN.md section 8).",
    "tendermint": "Oracle and discrete arms changed (Kevin, 2026-10-23): the oracle is 150 ms in p2 and p3, the best measured "
                  "p3 on prefix cells of the compressed spec (125 +0.17 with stalls, 150 +0.54, 175 +0.53; 200 +0.51 on the "
                  "full spec), and the discrete arms are 100/150/400 from restart-v5 (100/200/400 before; those discrete "
                  "rows and the 200 ms oracle draws are in archive/tendermint/). The oracle row ran under restart-v4 from "
                  "the same schedule (a file outside the code hash). The heuristic and the continuous rows opened at 200 ms, "
                  "the earlier middle rung; the opening covers only the first decision, in p0, where every timeout ties.",
}


HEAD = """# Results: learned timeouts against the oracle on 5 machines

Updated {stamp}; FREEZE {freeze} ({hash}), each section naming the frozen code its rows ran under. One full-spec run per configuration (four replicas and the client on five CloudLab machines), scored by the client judge: R = tps / N minus the client's latency in seconds (the commit-weighted mean of its per-second p50, a chunk with no commits at 10 s) per 20 s chunk, averaged per phase (the first 20 s after each onset skipped; the warm-up before the first onset and anything after the last phase's 300 s, the run's planned end, are outside every window), N the protocol's offered rate. Mean R is over the spec's phases; inner R (and the inner throughput and latency) leave out the first and the last phase, the base phases that open and close every spec (Kevin, 2026-10-21). z = seconds without a commit in the phase. Rows sorted by mean R; the set (A or B) each row ran on is in brackets (both sets pull rows from one queue per protocol). Pretrained = the simulator prior installed (its sim-gate verdict in brackets); every model learns online from the run's own rewards. Floor = the protocol's latency floor (multiplier x a hold-free signal + offset). Every engine and model opens at the ladder's middle rung. Generated by results/results_md.py from the runs in results/<protocol>/."""


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


def rows_for(p: str) -> tuple:
    """(rows for table(), flag lines, frozen-code names, earlier draws) from results/<p>/*/row.json and draw.json."""
    rows, flags, freezes, seen, earlier = [], [], set(), {}, []
    folder = HERE / p
    for d in sorted(folder.iterdir()) if folder.is_dir() else []:
        if not d.is_dir():
            continue
        if (d / "draw.json").exists() and not (d / "row.json").exists():
            draw = json.loads((d / "draw.json").read_text())
            if draw["protocol"] != p or draw["label"] not in LABELS:
                raise SystemExit(f"{d.relative_to(ROOT)}: draw.json names {draw['protocol']} {draw['label']}")
            earlier.append((draw, d))
            continue
        if not (d / "row.json").exists():
            raise SystemExit(f"{d.relative_to(ROOT)}: no row.json (finalize.py did not accept it)")
        row = json.loads((d / "row.json").read_text())
        if row["protocol"] != p or row["label"] not in LABELS:
            raise SystemExit(f"{d.relative_to(ROOT)}: row.json names {row['protocol']} {row['label']}")
        if row["label"] in seen:
            raise SystemExit(f"{p}: two runs of {row['label']}: {seen[row['label']]} and {d.name}")
        seen[row["label"]] = d.name
        sc = score(d)
        name = f"{display(row['label'])} [{row['set'][-1]}]" + (f" (gate {row['gate']})" if row["gate"] else "")
        rows.append((name, sc["mean"], sc["phases"], row["label"], sc, d.name, row["flags"]))
        freezes.add(f"{row['freeze_name']} ({row['code_hash'][:12]})")
    rows.sort(key=lambda r: -r[1])
    flags = [f"{r[0]}: {f}" for r in rows for f in r[6]]  # in table order
    return rows, flags, sorted(freezes), earlier


def repeated_table(rows: list, earlier: list) -> str:
    """Every draw of each configuration with an earlier draw kept, by start time; the latest is the row."""
    out = []
    for label in sorted({draw["label"] for draw, _ in earlier}, key=LABELS.index):
        entries = [(d.name[-15:], draw["set"][-1], "", score(d), d.name) for draw, d in earlier if draw["label"] == label]
        row = next((r for r in rows if r[3] == label), None)
        if row is not None:
            entries.append((row[5][-15:], re.search(r"\[(\w)\]", row[0]).group(1), ", the row", row[4], row[5]))
        for stamp, s, tag, sc, name in sorted(entries):
            when = f"{stamp[4:6]}-{stamp[6:8]} {stamp[9:11]}:{stamp[11:13]}"
            out.append((f"{display(label)} {when} [{s}]{tag}", sc["mean"], sc["phases"], label, sc, name, []))
    return table(out)


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
]
STUDY_LABELS = {
    "dense": [l for l in MODEL_LABELS if l.startswith("discrete")],
    "fibonacci": [l for l in MODEL_LABELS if l.startswith("discrete")],
    "sparse": ["oracle5", "oracle3"] + [l for l in MODEL_LABELS if l.startswith("discrete")],
    "local": ["heuristic"] + MODEL_LABELS,
    "nohold": ["heuristic"] + [l for l in MODEL_LABELS if "_pre" in l],
    "window": [f"{w}_{l}" for w in ("4", "16") for l in MODEL_LABELS if l.endswith("_floor")],
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
    return display(label)


def studies_part(e0: dict) -> str:
    """The PBFT studies: per study its rows (results/studies/pbft/<folder>/*/row.json, calibration cells left out) against
    E0, the PBFT table's row of the same configuration (E3: against its own 3-rung oracle)."""
    base = HERE / "studies" / "pbft"
    out = ["\n## PBFT studies\n\nPBFT only (runs/2026-10-23-pbft-studies/spec.md). Each study changes one thing against E0, "
           "the PBFT table above (same engine, priors array-identical, one run per row), and runs every variant where it "
           "applies. Scored as above; \"vs\" is the row's mean R minus E0's for the same configuration (E3: minus its 3-rung "
           "oracle). Calibration cells are in each study's calibration/ folder.\n"]
    for folder, title, what, prefix in STUDIES:
        rows, have = [], set()
        for d in sorted((base / folder).iterdir()) if (base / folder).is_dir() else []:
            if not (d / "row.json").exists():
                continue
            row = json.loads((d / "row.json").read_text())
            label = row["label"][len(prefix):]
            if label in have:
                raise SystemExit(f"{folder}: two runs of {row['label']}")
            have.add(label)
            sc = score(d)
            rows.append((f"{study_name(folder, label)} [{row['set'][-1]}]", sc["mean"], sc["phases"], label, sc, d.name, row["flags"]))
        rows.sort(key=lambda r: -r[1])
        out.append(f"\n### {title}\n\nChanges: {what}.\n\n")
        if not rows:
            out.append("Pending: every row.\n")
            continue
        ref = {r[3]: r[1] for r in rows if r[3] == "oracle3"} if folder == "sparse" else e0
        cells = []
        for r in rows:
            key = "oracle3" if folder == "sparse" else (r[3].split("_", 1)[1] if folder == "window" else r[3])
            v = ref.get(key)
            cells.append((r[0], f"{r[1]:+.3f}", "" if v is None or r[3] == "oracle3" else f"{r[1] - v:+.3f}"))
        out.append(aligned(("configuration", "mean R", "vs"), cells) + "\nPer phase:\n\n" + table(rows))
        pending = [study_name(folder, l) for l in STUDY_LABELS[folder] if l not in have]
        if pending:
            out.append(f"\nPending ({len(pending)}): {', '.join(pending)}.\n")
    return "".join(out)


def main(write: bool) -> None:
    freeze = dict(l.split("=", 1) for l in (ROOT / "FREEZE").read_text().splitlines() if "=" in l)
    parts = [HEAD.format(stamp=datetime.now().strftime("%Y-%m-%d %H:%M"), freeze=freeze["freeze_name"],
                         hash=freeze["code_hash"][:12])]
    for p in PROTOCOLS:
        rows, flags, freezes, earlier = rows_for(p)
        parts.append(f"\n## {NAMES[p]}\n\n{header(p)}\n\n")
        if p in NOTES:
            parts.append(NOTES[p] + "\n\n")
        if not rows:
            parts.append("Pending: every row.\n")
            continue
        parts.append(f"Rows ran under frozen code {', '.join(freezes)}.\n\n" + table(rows))
        n = int(reward_normalizer_tps(p))
        parts.append(f"\nThroughput (committed tx/s at the client, mean over the judged seconds of each phase; N {n:,}):\n\n"
                     + value_table(rows, "tps", lambda v: f"{v:,.0f}", "tps"))
        parts.append(f"\nLatency (ms, mean over the judged 20 s chunks of the client's commit-weighted p50; a chunk "
                     f"with no commits counts {EMPTY_P50_MS:,.0f} ms, as in R):\n\n"
                     + value_table(rows, "lat", lambda v: f"{v:,.0f}", "lat"))
        if earlier:
            parts.append("\nRepeated draws (configurations rerun on purpose; the latest draw is the row above, the "
                         "earlier ones stay in results/ with draw.json):\n\n" + repeated_table(rows, earlier))
        if flags:
            parts.append("\nFlags (validate.py; engine defects kept by D16, the row stands):\n\n"
                         + "".join(f"- {f}\n" for f in flags))
        have = {r[3] for r in rows}
        pending = [display(l) for l in LABELS if l not in have]
        if pending:
            parts.append(f"\nPending: {', '.join(pending)}.\n")
    e0 = {r[3]: r[1] for r in rows_for("pbft")[0]}
    parts.append(studies_part(e0))
    text = "".join(parts)
    if write:
        (HERE / "RESULTS.md").write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main("--write" in sys.argv)
