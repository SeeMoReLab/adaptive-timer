#!/usr/bin/env python3
"""Convergence of the learned, floored configurations after each phase onset (Kevin, 2026-10-27).

Reads every draw's per-episode timeout log (agent0.csv, the agreed decisions every replica applied) under
results/<protocol>/, the oracle's draws for the reference timeouts, and the client judge (scoring.score) for
latency. Writes results/CONVERGENCE.md. The first phase (p0) is excluded.

Definitions (applied per draw and phase, over the phase's episodes in the dataset's own phase column):
  settle episode   the first episode k (0 = the first decision after the onset) from which the action stays
                   within 10% of one value (discrete: the same rung) for at least 3 consecutive episodes, all
                   inside the phase; the value is the median of those 3
  converged        a settle episode exists
  settled value    the median action over the settled run: the settle episode and the following episodes
                   while they stay within 10% of that value (discrete: on that rung)
  oracle match     discrete: the settled value is the oracle's rung; continuous: within 20% of it
  CV               std / mean of the timeout (SBFT: election timeout) over the judge's scored episodes of the phase (judge.played)
  switches         changes of value between consecutive settled runs (each a maximal run of 3 or more within
                   10%) inside the phase
The action is the election/propose/delay timeout; on SBFT it is the (election timeout, fast-path wait) pair,
and both coordinates must satisfy the tolerance. Tendermint's prevote and precommit are derived from the
propose timeout, so only that is used.

Usage: python3 results/convergence.py
"""
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from judge import phases as spec_phases  # noqa: E402
from scoring import score  # noqa: E402

PROTOCOLS = ("pbft", "sbft", "smartbft", "tendermint", "hotstuff", "autobahn")
NAMES = {"pbft": "PBFT", "sbft": "SBFT", "smartbft": "SmartBFT", "tendermint": "Tendermint",
         "hotstuff": "HotStuff", "autobahn": "Autobahn"}
CONFIGS = tuple(f"{kind}_{model}_{prior}_floor" for kind in ("discrete", "continuous")
                for model in ("gp", "linucb", "ppo") for prior in ("pre", "untr"))
SETTLE_TOL, ORACLE_TOL, SETTLE_LEN = 0.10, 0.20, 3
EPISODE_S = 20
RUN = re.compile(r"^(?P<proto>[a-z]+)_(?P<label>.+)_set[A-Z]_\d{8}_\d{6}_(?P<draw>\d+)$")


def action_columns(proto):
    if proto == "sbft":
        return ("election_timeout_milliseconds_used", "slow_path_timeout_milliseconds_used")
    if proto == "tendermint":
        return ("propose_timeout_milliseconds_used",)
    if proto in ("hotstuff", "autobahn"):
        return ("timeout_delay_milliseconds_used",)
    return ("election_timeout_milliseconds_used",)


def draws(proto, label):
    out = {}
    for d in sorted((HERE / proto).iterdir()):
        m = RUN.match(d.name)
        if m and m.group("proto") == proto and m.group("label") == label:
            out[int(m.group("draw"))] = d
    return dict(sorted(out.items()))


def episodes(run, proto):
    """[(episode, phase, action tuple)] from replica 0's dataset."""
    cols = action_columns(proto)
    rows = list(csv.DictReader((run / "agent0.csv").open()))
    return [(int(r["episode"]), int(r["phase"]), tuple(float(r[c]) for c in cols)) for r in rows]


def judged(run, proto, phase):
    """Episodes the judge scores in a phase: decision at about 20 k + 10 s inside [onset + 20 s, next onset)."""
    warm, ats = spec_phases(run)
    offset = 0.5 if proto == "tendermint" else 0.0
    lo = warm + ats[phase] + offset + 20
    hi = warm + ats[phase + 1] + offset if phase + 1 < len(ats) else float("inf")
    return lo, hi


def close(a, b, discrete, tol=SETTLE_TOL):
    if discrete:
        return a == b
    return all(abs(x - y) <= tol * y for x, y in zip(a, b))


def median_action(actions):
    return tuple(statistics.median(a[i] for a in actions) for i in range(len(actions[0])))


def settled_runs(actions, discrete):
    """Maximal runs of SETTLE_LEN or more within tolerance: [(start, end_exclusive, value)]."""
    runs, i = [], 0
    while i + SETTLE_LEN <= len(actions):
        ref = median_action(actions[i:i + SETTLE_LEN])
        if all(close(a, ref, discrete) for a in actions[i:i + SETTLE_LEN]):
            j = i + SETTLE_LEN
            while j < len(actions) and close(actions[j], ref, discrete):
                j += 1
            runs.append((i, j, median_action(actions[i:j])))
            i = j
        else:
            i += 1
    return runs


def fmt_action(a):
    return "/".join(f"{x:.0f}" for x in a)


def analyse(proto):
    oracle_runs = draws(proto, "oracle")
    if not oracle_runs:
        raise SystemExit(f"{proto}: no oracle runs")
    # The oracle's action per phase: median over its draws' episodes (the schedule is constant within a phase).
    oracle_eps = [episodes(r, proto) for r in oracle_runs.values()]
    n_phases = max(p for _, p, _ in oracle_eps[0]) + 1
    oracle = {p: median_action([a for eps in oracle_eps for _, ph, a in eps if ph == p]) for p in range(n_phases)}
    oracle_lat = {p: statistics.mean(score(r)["phases"][p]["lat"] for r in oracle_runs.values()) for p in range(n_phases)}
    res = {}
    for label in CONFIGS:
        discrete = label.startswith("discrete")
        runs = draws(proto, label)
        if not runs:
            raise SystemExit(f"{proto}: no runs for {label}")
        per_draw = {}
        for draw, run in runs.items():
            eps = episodes(run, proto)
            lats = score(run)["phases"]
            phs = {}
            for p in range(1, n_phases):
                ph_eps = [(e, a) for e, ph, a in eps if ph == p]
                actions = [a for _, a in ph_eps]
                sruns = settled_runs(actions, discrete)
                lo, hi = judged(run, proto, p)
                jt = [a[0] for e, a in ph_eps if lo <= EPISODE_S * e + 10 < hi]
                cv = statistics.pstdev(jt) / statistics.mean(jt) if len(jt) > 1 else float("nan")
                values = [v for _, _, v in sruns]
                switches = sum(1 for x, y in zip(values, values[1:]) if not close(x, y, discrete))
                first = sruns[0] if sruns else None
                phs[p] = dict(
                    k=first[0] if first else None,
                    value=first[2] if first else None,
                    match=(close(first[2], oracle[p], discrete, ORACLE_TOL) if first else None),
                    # The value it ends the phase on, if every settled run in the phase agrees (for the flags).
                    steady=(values[0] if values and switches == 0 else None),
                    cv=cv, switches=switches, lat=lats[p]["lat"])
            per_draw[draw] = phs
        res[label] = per_draw
    return n_phases, oracle, oracle_lat, res


def flags(n_phases, oracle, oracle_lat, res):
    """Chains of 2+ consecutive phases where one draw holds the same settled value while the oracle's
    action changes, the value being off the oracle's in at least one of them."""
    out = []
    for label, per_draw in res.items():
        discrete = label.startswith("discrete")
        for draw, phs in per_draw.items():
            p = 1
            while p < n_phases:
                v = phs[p]["steady"]
                q = p
                while v is not None and q + 1 < n_phases and phs[q + 1]["steady"] is not None \
                        and close(phs[q + 1]["steady"], v, discrete):
                    q += 1
                chain = range(p, q + 1)
                if v is not None and q > p and len({oracle[c] for c in chain}) > 1 \
                        and any(not close(v, oracle[c], discrete, ORACLE_TOL) for c in chain):
                    out.append((label, draw, p, q, v, [oracle[c] for c in chain],
                                [(phs[c]["lat"], oracle_lat[c]) for c in chain]))
                p = q + 1
    return out


def table(header, rows):
    cols = list(zip(*([header] + rows)))
    widths = [max(len(str(c)) for c in col) for col in cols]
    line = lambda r: "| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |"
    return "\n".join([line(header), "|" + "|".join("-" * (w + 2) for w in widths) + "|"] + [line(r) for r in rows])


def pct(x):
    return f"{100 * x:.0f}%" if x == x else "-"


def main():
    out = [
        "# Convergence after phase onsets",
        "",
        "Generated by `results/convergence.py` (do not edit by hand). For every protocol, every learned configuration with the floor and every draw, from the per-episode timeout log (`agent0.csv`, the agreed decisions every replica applied). The first phase is excluded.",
        "",
        "Definitions. Episodes are 20 s and a phase is 15 episodes; k = 0 is the first decision after the onset (taken about 20 s after it), so k episodes are about 20 (k + 1) s after the onset. The **settle episode** is the first k from which the action stays within 10% of one value (discrete: the same rung) for at least 3 consecutive episodes inside the phase; the phase **converged** if one exists. The **settled value** is the median over that settled run (it extends while the action stays within 10%). **Oracle match**: discrete, the settled value is the oracle's rung; continuous, within 20% of it. **CV** is std / mean of the timeout (on SBFT the election timeout) over the judge's scored episodes of the phase. **Switches** counts changes of value between consecutive settled runs (each 3 or more episodes within 10%) inside the phase. On SBFT the action is the (election timeout, fast-path wait) pair and both coordinates must satisfy the tolerance; on Tendermint it is the propose timeout (prevote and precommit are derived from it). Latency is the client judge's per-phase mean (`scoring.py`), averaged over draws, against the oracle's mean over its draws.",
        "",
        "Configuration names: D/C = discrete/continuous, pre/un = pretrained/untrained, all with the floor.",
        "",
    ]
    summaries, details, all_flags = [], [], []
    for proto in PROTOCOLS:
        n_phases, oracle, oracle_lat, res = analyse(proto)
        name = NAMES[proto]
        srows, drows = [], []
        for label, per_draw in res.items():
            short = ("D" if label.startswith("discrete") else "C") + " " + label.split("_")[1].replace("linucb", "LinUCB").replace("gp", "GP").replace("ppo", "PPO") + " " + ("pre" if "_pre_" in label else "un")
            cells = [phs[p] for phs in per_draw.values() for p in range(1, n_phases)]
            conv = [c for c in cells if c["k"] is not None]
            srows.append([short, f"{len(per_draw)}", pct(len(conv) / len(cells)),
                          f"{statistics.median(c['k'] for c in conv):.0f}" if conv else "-",
                          pct(sum(c["match"] for c in conv) / len(conv)) if conv else "-"])
            for p in range(1, n_phases):
                ph = [per_draw[d][p] for d in sorted(per_draw)]
                cv_ = [c for c in ph if c["k"] is not None]
                k_med = statistics.median(c["k"] for c in cv_) if cv_ else None
                drows.append([
                    short, f"p{p}", f"{len(cv_)}/{len(ph)}",
                    f"{k_med:.0f} ({EPISODE_S * (k_med + 1):.0f} s)" if k_med is not None else "-",
                    ", ".join(fmt_action(c["value"]) if c["value"] else "-" for c in ph),
                    fmt_action(oracle[p]),
                    f"{sum(c['match'] for c in cv_)}/{len(cv_)}" if cv_ else "-",
                    f"{statistics.mean(c['lat'] for c in ph):.0f} / {oracle_lat[p]:.0f}",
                    ", ".join(f"{c['cv']:.2f}" for c in ph),
                    ", ".join(str(c["switches"]) for c in ph),
                ])
        fl = flags(n_phases, oracle, oracle_lat, res)
        all_flags += [(name, *f) for f in fl]
        summaries.append(f"### {name}\n\nOracle per phase (p1-p{n_phases - 1}): " + ", ".join(fmt_action(oracle[p]) for p in range(1, n_phases)) + " ms.\n\n"
                         + table(["configuration", "draws", "phases converged", "median episodes to settle", "converged on oracle value"], srows))
        details.append(f"### {name}\n\n" + table(
            ["configuration", "phase", "converged", "median k to settle", "settled value per draw", "oracle", "oracle match", "latency / oracle (ms)", "CV per draw", "switches per draw"], drows))
    out += ["## Summary per protocol", "", "Columns: share of (draw, phase) cells that converged; median settle episode k over converged cells; share of converged cells whose settled value matches the oracle's.", ""]
    out += ["\n\n".join(summaries), ""]
    out += ["## Settled off the oracle across an oracle change", "",
            "One draw holding the same settled value (every settled run in each phase agreeing) through two or more consecutive phases while the oracle's timeout changes, and off the oracle's value in at least one of them. The last column shows whether holding it cost anything: a held rung can be harmless in a phase (an equally good timeout).", ""]
    if all_flags:
        out.append(table(["protocol", "configuration", "draw", "phases", "held value", "oracle over those phases", "latency / oracle per phase (ms)"],
                         [[n, l, str(d), f"p{p}-p{q}", fmt_action(v), ", ".join(fmt_action(o) for o in os_),
                           ", ".join(f"{a:.0f}/{b:.0f}" for a, b in lats)] for n, l, d, p, q, v, os_, lats in all_flags]))
    else:
        out.append("None.")
    out += ["", "## Per phase", "", "Per (configuration, phase): draws converged; median settle episode k (and seconds after the onset); each draw's settled value (`-` = did not converge); the oracle's value; draws whose settled value matches it; mean latency over draws against the oracle's; CV and switches per draw.", ""]
    out += ["\n\n".join(details), ""]
    (HERE / "CONVERGENCE.md").write_text("\n".join(out))
    print(f"wrote {HERE / 'CONVERGENCE.md'} ({len(all_flags)} flags)")


if __name__ == "__main__":
    main()
