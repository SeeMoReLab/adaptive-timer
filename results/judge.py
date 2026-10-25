#!/usr/bin/env python3
"""Per-phase client judge for any protocol's full-spec run (client output only). Lives in results/ since 2026-10-24
(scripts/cloudlab/tmp/cal/pt/judge.py until then, now a shim that loads this file).

Phases start at warmUpTime + atTime on the client clock, read from the run's own
failure_spec.xml; the client clock is the client's "benchmark start unix ms" line
(Tendermint's onsets land 0.5 s later, as in cal/cl_judge.py). Each phase is scored
from 20 s after its boundary (the agent applies a switch at its next decision) to
the next boundary, in 20 s chunks: R = tps/N - latency s per chunk (N the protocol's offered rate,
learning_agent/reward_utils.py REWARD_NORMALIZER_TPS; tps/10000 until 2026-10-12), the latency the
commit-weighted mean of the client's per-second mean latencies (avg_ms; the per-second p50 until
2026-10-21: Kevin, "shouldn't we use mean?", as the agents' reward does since 2026-10-09; a median
never sees the minority of requests an election or a re-pinned hold delays), a chunk with no
commits scoring 10 s (as cal/ab/phase_judge.py), and no chunk scoring above 10 s (Kevin 2026-10-10:
the few requests that commit at the end of a stall carry 17-29 s latencies, so an
uncapped trickle scored below a full stall). Only seconds that end before the run's
planned length (run.env experiment_time_s) count (Kevin 2026-10-10): the client's
shutdown sample (about 0.83 s, no commits, 'context canceled') and anything logged
past the planned end are not experiment time. Also: mean tps, errors, stalled seconds, and
the median timeout(s) replica 0's agent played in the phase (agent0.csv).

Usage: judge.py RUN_DIR [RUN_DIR ...]
"""
import csv
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True  # keep results/ free of __pycache__
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "learning_agent"))
from reward_utils import reward_normalizer_tps  # noqa: E402

SKIP_S = 20
CHUNK_S = 20
EMPTY_P50_MS = 10000.0
START = re.compile(r"benchmark start unix ms (\d+)")
MON = re.compile(r"\[client (\S+)\] Monitor (duration=.*)")
KV = re.compile(r"(\w+)=(\S+)")


def protocol_of(run):
    for line in (run / "run.env").read_text().splitlines():
        if line.startswith("protocol="):
            return line.split("=", 1)[1]
    return run.parent.name


def phases(run):
    spec = re.sub(r"<!--.*?-->", "", (run / "failure_spec.xml").read_text(), flags=re.S)
    warm = float(re.search(r"<warmUpTime>([\d.]+)</warmUpTime>", spec).group(1))
    return warm, [float(t) for t in re.findall(r"<atTime>([\d.]+)</atTime>", spec)]


def seconds(run):
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
            rows.append((ts - t0, int(v["succ"]), int(v.get("err", 0)), float(v["avg_ms"])))
    if t0 is None:
        raise SystemExit(f"{run}: no 'benchmark start unix ms' line in client.log")
    return rows


def played(run, lo_s, hi_s, t0_offset):
    """Median of each *_milliseconds_used column over replica 0's episodes whose
    decision fell in [lo_s, hi_s) on the client clock (episode k decides at about
    t0_offset + 20 k + 10 s)."""
    path = run / "agent0.csv"
    if not path.exists():
        return "-"
    rows = list(csv.DictReader(path.open()))
    cols = [c for c in (rows[0] if rows else {}) if c.endswith("_milliseconds_used")]
    vals = {c: [] for c in cols}
    for r in rows:
        t = 20.0 * int(r["episode"]) + 10.0
        if lo_s <= t < hi_s:
            for c in cols:
                if r[c] not in ("", None):
                    vals[c].append(float(r[c]))
    out = [f"{statistics.median(v):.0f}" for c, v in vals.items() if v and not c.startswith(("prevote", "precommit"))]
    return "/".join(out) if out else "-"


def judge(run):
    proto = protocol_of(run)
    warm, ats = phases(run)
    offset = 0.5 if proto == "tendermint" else 0.0
    norm = reward_normalizer_tps(proto)
    env = dict(l.split("=", 1) for l in (run / "run.env").read_text().splitlines() if "=" in l)
    if "experiment_time_s" not in env:
        raise SystemExit(f"{run}: no experiment_time_s in run.env (the planned length bounds the judged seconds)")
    length = float(env["experiment_time_s"])
    rows = [r for r in seconds(run) if r[0] < length]  # r[0] is the end of the sample's second
    end = min(length, max(r[0] for r in rows) + 1.0)
    starts = [warm + a + offset for a in ats] + [end]
    out = []
    for i in range(len(ats)):
        lo, hi = starts[i] + SKIP_S, starts[i + 1]
        sec = [r for r in rows if lo <= r[0] < hi]
        rewards = []
        for k in range(0, len(sec) - CHUNK_S + 1, CHUNK_S):
            ch = sec[k:k + CHUNK_S]
            s = sum(r[1] for r in ch)
            q = min(sum(r[1] * r[3] for r in ch) / s, EMPTY_P50_MS) if s else EMPTY_P50_MS
            rewards.append(s / len(ch) / norm - q / 1000.0)
        tps = sum(r[1] for r in sec) / len(sec) if sec else float("nan")
        out.append(dict(
            phase=i, at=ats[i],
            R=sum(rewards) / len(rewards) if rewards else float("nan"),
            tps=tps, err=sum(r[2] for r in sec), stall=sum(1 for r in sec if r[1] == 0),
            T=played(run, lo, hi, 0.0)))
    return proto, out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for arg in sys.argv[1:]:
        run = Path(arg)
        proto, out = judge(run)
        mean = sum(p["R"] for p in out) / len(out)
        print(f"== {run.name} ({proto})  mean R {mean:+.3f}")
        for p in out:
            print(f"  p{p['phase']} t={p['at']:<5g} R={p['R']:+.3f} tps={p['tps']:7.0f} "
                  f"err={p['err']:6d} stall_s={p['stall']:3d} T={p['T']}")
