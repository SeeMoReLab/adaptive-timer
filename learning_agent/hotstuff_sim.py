"""Analytic batched-HotStuff simulator for pretraining the continuous model.

Built from the mechanics of autobahn/baselines/hotstuff (core.rs, the
pacemaker and the proposal-delay fault in adaptive/src/failure.rs) and the
behaviour measured on CloudLab in scripts/cloudlab/tmp/cal/hs (2026-10-02),
not fit to the failure spec. Four replicas, round-robin leaders, quorum 3,
votes to the next leader, a pacemaker timer reset on every round advance, a
100 ms ``min_block_delay`` sleep after each of a leader's own proposals, and a
3-chain commit:

- Round period ``p0 = max(2h + d1, (D + h + d2) / 3)``: two one-way hops
  (block out, votes to the next leader) when the network sets the pace, or the
  leaders' block-delay sleeps when it does (at 5 ms the sleep binds; measured
  42.6 ms there, 112 at 5+50, 314 at 5+150).
- A block commits when the block three rounds later arrives, so the report's
  consensus latency (first seen to commit) is the sum of three consecutive
  rounds, plus h for a replica's own blocks.
- The leader of a round waits ``p + h`` for the next block. No socket sets
  TCP_NODELAY, so a vote sent right after a Timeout waits one round trip: a
  leader that times out in its own round makes the next round take one hop
  more (``p3 = p0 + h``), which makes the next leader time out too. The
  3-hop state is forced for ``T < p0 + h`` (``W2``), sustains itself up to
  ``T < p3 + h`` (``W3``), and in between the run stays in whichever state it
  is in until a perturbation (a link-delay change, a slow leader tripping a
  timer) tips it into the 3-hop state.
- Below ``p3`` the replicas waiting for the next block time out too:
  every replica times out every round, rounds stretch to ``p3 + g h`` and
  part of the load is lost; below ``c h`` nothing commits.
- A slow leader (one replica, one round in four) holds its proposal, either
  until ``X`` ms before its live timeout (``delayRelativeToTimeoutMs -X``) or
  by a fixed ``hold``. A hold that outlasts the previous leader's wait trips
  that leader's timer (and the Nagle hop); one that outlasts the other
  replicas' wait ends the round in a timeout certificate instead, ``T + 2h``.
- The next leader's receipt of a block to its QC (proposal_to_qc_ms, 2026-10-19)
  is the round without the block's own hop: ``base - h``, the votes' hop plus
  processing where the network sets the pace and the wait for sleeping voters
  where the block-delay sleeps do. Under a slow leader whose hold outlasts that
  wait the voters have woken by the time a block arrives, so it is the votes'
  hop plus processing in every round; a shorter hold leaves the sleeping-voter
  value (measured on real_c120, 2026-10-19 hsc_ cells: 34 ms at base, 6.5 under
  the base slow leader at T 300/700, 57-58 at 5+50 with or without it, 156 at
  5+150; 30-35 at base under a 6-9 ms hold, hs2_heuristic p1). A
  proposer's hold precedes its block, so it never enters; a round a timeout
  certificate ends forms no QC and gives none. Below the collapse threshold QCs
  still form between the timeouts, about 2h (325 ms measured at 5+150, T 300).
- The mempool sheds a replica's own intake once ``Q`` transactions are
  queued, so a round of ``d`` ms commits at most ``Q / r`` of them (``r`` the
  residence in rounds): throughput is ``min(offered, Q / (r d))`` time-weighted
  over the cycle's rounds, the shed load lost.
- A round's batch is the load its duration brings in, capped by the mempool,
  times a lognormal spread measured per regime (2026-10-19, restart D11; was
  a synthetic 1.1 x the mean): p95 / mean batch 2.21 at base, where the
  block-delay sleeps set the pace, and 1.20 at 5+50, where the network does
  (hs1_oracle_onset300_setB, 2026-10-16). p95_batch_size is the window's
  nearest-rank p95 of these per-round samples.
- regency_change_count is what baselines/node/src/node.rs reports: a commit
  whose round is more than one past the previous commit's, counted from a
  window's second commit on (adaptive/src/metrics.rs). A round a timeout
  certificate ends loses its block, so the next commit skips it; a collapse
  skips rounds too. The 3-hop and degraded regimes' timeouts skip none (0 in
  all 2,262 archived real p0/p1 episodes, 17-20 view changes per window on
  average), so the field is 0 outside a held block lost to a TC and a
  collapse's end (2026-10-19, restart D11; was always 0).

Episodes follow the wall-clock learning windows in lib_rustbft.sh: 7 s
features, 2 s reply wait (applied at the deadline), 4 s warm-up, 7 s reward.
A window in which nothing committed gets its own length as its latency, as
the agent does for HotStuff (main.py, Kevin 2026-10-02).

Training distribution (Kevin, 2026-10-19, restart D11; sim_common.py, the
recipe every simulator shares):

- constants: the fitted ones (sim_fit.py FREE: processing, the degraded
  regime's shape, the collapse threshold, how readily the bistable band
  tips, the mempool residence) 0.5-2x around the SimConstants defaults as
  before; the unfitted ones (round_sigma and the two batch spreads) with
  around, 0.5-2x around their measured values (round_sigma was uniform
  0.01-0.08);
- offered load: draw_load_tps around the 105,000 tx/s rate (0.5-1.5x; was
  uniform 42,000-168,000);
- base one-way delay: draw_base_delay_ms once per run; a fault phase's
  one-way is the base plus, in LINK_PHASE_SHARE of the phases,
  draw_link_extra_ms (was one log-uniform 2-250 ms per phase, a third of it
  under 10 ms, hence a third without extra delay). No netem jitter is
  modelled (JITTER_RATIO 0; check_spec_files requires the spec's to match);
- phases: a base phase at 0 (base delay, no slow leader; was a full random
  phase), then a fault at each fault_start_times onset;
- slow leader per fault phase, the mix as before: none (35%), a hold
  relative to the live timeout (45%) or a fixed hold (20%). A fixed hold is
  draw_hold_ms (was log-uniform 10-2000). A relative hold is max(0, T - X):
  X is the timeout above which the slow leader costs, the hold's scale, so
  it is drawn from the same draw_hold_ms, 50-1500 ms (was log-uniform
  10-2000), as autobahn_sim does. That covers the real spec's X = 100 at
  base and 400 at 5+50 (delayRelativeToTimeoutMs -100 / -400; about 13% of
  the relative draws within -20% / +25% of each), and the hold then follows
  T across the whole 100-3000 action range as the engine's does, from 0 at
  T = X to 3000 - X;
- initial timeout: draw_initial_timeout_ms (a quarter of the runs on the
  opening action, 300; was a quarter on the parameters.json default 5000);
- no initial backlog: the simulator has no backlog state (the mempool cap
  sheds the excess each round).
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import agent_pb2
from continuous_bandit import CONTEXT_SPECS
from continuous_bandit import opening_action_ms, train_gp_prior
from prior_windows import PriorWindows, check_prior_windows
from reward_utils import DEFAULT_REWARD_DESIGN, reward_normalizer_tps
from sim_common import (
    JITTER_RATIO, around, draw_base_delay_ms, draw_hold_ms, draw_initial_timeout_ms,
    draw_link_extra_ms, draw_load_tps, fault_start_times,
)

N_NODES = 4
MIN_BLOCK_DELAY_MS = 100.0
# config/hotstuff/parameters.json mempool.max_queued_transactions: a replica
# sheds its own intake at this many queued transactions (mempool/src/core.rs
# handle_own_payload; the count includes peers' payloads, so it bounds the
# whole system's backlog). The max_queue_delay age bound (5 s) never binds
# outside a collapse.
MAX_QUEUED_TRANSACTIONS = 20000.0
# config/hotstuff/parameters.json consensus.timeout_delay, the engine's own default;
# every run of the restart overrides it with the middle rung (--initial-timeout, D10b).
DEFAULT_TIMEOUT_MS = 5000.0
FEATURE_S = 7.0
REPLY_S = 2.0
WARMUP_S = 4.0
REWARD_S = 7.0
EPISODE_S = FEATURE_S + REPLY_S + WARMUP_S + REWARD_S
SIM_WINDOWS = PriorWindows.from_seconds(FEATURE_S, REPLY_S, WARMUP_S, REWARD_S)

DEFAULT_PRIOR_PATH = CONTEXT_SPECS[agent_pb2.PROTOCOL_HOTSTUFF].prior_path
# The offered rate, N of reward_tps_latency; training draws the load around it.
RATE_TPS = reward_normalizer_tps("hotstuff")


@dataclass
# 105,000 tx/s refit (2026-10-12, runs/2026-10-12-recal3): 7,500 fresh candidates (seeds 2-6) on the bl512_ blind runs, with the
# mempool queue cap (residence_rounds), tps loss scale N/10 and collapse_hops (1.3, 3.0); blind loss 0.538 (previous defaults 0.818),
# regret 0.004; real-spec check (bl512_real incl. the ladder reruns) loss 0.464 (defaults 0.825), regret 0.004 (defaults 0.001).
# Fitted defaults: blind best-by-loss candidate i=886 (loss 0.437) of s3fit_hotstuff_blind_s8.jsonl, restart S3 blind refit (seeds 7,8,9,10,11 x 1500, best of the seeds; real-spec mean regret 0.000 PASS, 2026-10-20).
class SimConstants:
    """Defaults are the calibration host (cloudlab_1; 105,000 tx/s since the
    2026-10-12 bl512_ runs, 10,000 before); training draws every field from
    sample_constants, the fitted ones around these."""

    # Per-round processing when the network sets the pace (measured 2-4 ms)
    # and in the sleep-bound regime (42.6 ms at 5 ms one-way gives 23).
    proc_net_ms: float = 7.024497
    proc_sleep_ms: float = 37.394695
    # Lognormal spread of a round's duration.
    round_sigma: float = 0.03
    # Lognormal spread of a round's batch around the load its duration brings
    # in, where the block-delay sleeps set the pace and where the network does
    # (p95 / mean 2.21 at base, 1.20 at 5+50: hs1_oracle_onset300_setB p0/p2,
    # 2026-10-16; restart D11, 2026-10-19). Measured, not fitted.
    batch_sigma_sleep: float = 0.58
    batch_sigma_net: float = 0.115
    load_tps: float = 105000.0
    # Degraded regime (T under p3): rounds take p3 + degraded_hops * h and
    # this share of the load commits (measured 4.0-4.2 h and 0.79-0.91 at 5+50).
    degraded_hops: float = 0.524994
    degraded_share: float = 0.779299
    # Nothing commits below collapse_hops * h (2 h degrades, 1.55 h collapses).
    collapse_hops: float = 2.283238
    # Probability that a perturbation tips a run in the bistable band into the
    # 3-hop state (the 5+150 onset did at 3.3 h, the 5+50 one did not at 3.3 h).
    tip_prob: float = 0.2373
    # Rounds a transaction stays queued: a round commits at most
    # MAX_QUEUED_TRANSACTIONS / residence_rounds of them.
    residence_rounds: float = 1.361231


@dataclass
class Phase:
    start_s: float
    one_way_ms: float = 5.0
    # Slow leader: hold until hold_offset_ms before the live timeout, or a
    # fixed hold_ms; neither means no slow leader.
    hold_offset_ms: Optional[float] = None
    hold_ms: Optional[float] = None


def committed_tps(cycle_ms: List[float], offered_tps: float, residence_rounds: float) -> float:
    """Throughput of a cycle of rounds under the mempool cap: each round of d
    ms commits min(offered d, Q / r), the excess shed."""
    if residence_rounds <= 0:
        raise ValueError(f"residence_rounds must be positive, got {residence_rounds}")
    per_round = MAX_QUEUED_TRANSACTIONS / residence_rounds
    tx = sum(min(offered_tps * d / 1000.0, per_round) for d in cycle_ms)
    return 1000.0 * tx / sum(cycle_ms)


def _rank(values: np.ndarray, q: float) -> float:
    """Nearest-rank percentile, sorted[ceil(q n) - 1], as metrics.rs."""
    ordered = np.sort(values)
    return float(ordered[max(0, math.ceil(q * len(ordered)) - 1)])


@dataclass
class _Window:
    start_s: float
    end_s: float
    rounds: List[float] = field(default_factory=list)
    latencies: List[float] = field(default_factory=list)
    batches: List[float] = field(default_factory=list)
    qcs: List[float] = field(default_factory=list)
    regency_changes: int = 0
    timeouts: float = 0.0
    stalled_timeouts: float = 0.0
    tx: float = 0.0

    def report(self, timeout_ms: float, empty_latency_ms: float) -> Dict[str, float]:
        length_s = self.end_s - self.start_s
        committed = len(self.latencies)
        # A round's duration is the gap to the previous commit; the window's
        # first reaches back before it, and gaps reset per window (metrics.rs:
        # n commits give n-1 gaps).
        gaps = np.asarray(self.rounds[1:])
        if committed:
            lat = np.asarray(self.latencies)
            avg, p50, p95 = float(lat.mean()), _rank(lat, 0.50), _rank(lat, 0.95)
        else:
            # The agent's empty-window rule: the window's own length.
            avg = p50 = p95 = empty_latency_ms
        tx = self.tx if committed else 0.0
        batch = tx / committed if committed else 0.0
        # Nearest-rank p95 of the per-round batches (metrics.rs percentile_usize).
        p95_batch = _rank(np.asarray(self.batches), 0.95) if committed else 0.0
        return {
            "total_transactions": tx,
            "total_consensus_instances": float(committed),
            "avg_consensus_latency_ms": avg,
            "p50_consensus_latency_ms": p50,
            "p95_consensus_latency_ms": p95,
            "throughput_tps": tx / length_s,
            "avg_batch_size": batch,
            "p95_batch_size": p95_batch,
            "leader_change_count": float(max(0, committed - 1)),
            "regency_change_count": float(self.regency_changes),
            "timeout_ms": float(timeout_ms),
            "avg_inter_commit_gap_ms": float(gaps.mean()) if len(gaps) else 0.0,
            "p50_inter_commit_gap_ms": _rank(gaps, 0.50) if len(gaps) else 0.0,
            "p95_inter_commit_gap_ms": _rank(gaps, 0.95) if len(gaps) else 0.0,
            "view_change_count": float(round(self.timeouts)),
            "no_progress_view_change_count": float(round(self.stalled_timeouts)),
            # Window mean; 0 when no round formed a QC (the Rust side's empty value).
            "proposal_to_qc_ms": float(np.mean(self.qcs)) if self.qcs else 0.0,
        }


class HotstuffSim:
    def __init__(
        self,
        constants: SimConstants,
        phases: List[Phase],
        seed: int,
        start_offset_s: float = 0.0,
        initial_timeout_ms: float = DEFAULT_TIMEOUT_MS,
    ):
        self.c = constants
        self.phases = sorted(phases, key=lambda ph: ph.start_s)
        self.rng = np.random.default_rng(seed)
        self.episode_start_s = float(start_offset_s)
        self.timeout_ms = float(initial_timeout_ms)
        self.locked = False
        self._last_one_way: Optional[float] = None
        # Position of the slow leader in the 4-round cycle and of this
        # replica's own blocks; the end of the last round and the next one,
        # drawn but not yet ended.
        self._slot = 0
        self._clock_ms = 0.0
        self._next_round: Optional[Tuple[float, float, Optional[float], float, bool]] = None
        # The next commit skips a round (a block lost to a timeout certificate,
        # a collapse): a regency change (node.rs adapt_hotstuff_commits).
        self._gap_pending = False
        self._lost_ms = 0.0  # duration of rounds lost to a timeout certificate since the last commit

    def phase_at(self, t_s: float) -> Tuple[int, Phase]:
        idx = 0
        for i, ph in enumerate(self.phases):
            if ph.start_s <= t_s:
                idx = i
        return idx, self.phases[idx]

    def _hold(self, ph: Phase) -> float:
        if ph.hold_offset_ms is not None:
            return max(0.0, self.timeout_ms - ph.hold_offset_ms)
        return float(ph.hold_ms) if ph.hold_ms is not None else 0.0

    def _second(self, t_s: float, windows: List[_Window]) -> None:
        """Advance one second at ``t_s`` and add its rounds to every window covering it."""
        c, T = self.c, self.timeout_ms
        _, ph = self.phase_at(t_s)
        h = ph.one_way_ms
        sleep_bound = (MIN_BLOCK_DELAY_MS + h + c.proc_sleep_ms) / 3.0 > 2.0 * h + c.proc_net_ms
        p0 = max(2.0 * h + c.proc_net_ms, (MIN_BLOCK_DELAY_MS + h + c.proc_sleep_ms) / 3.0)
        p3 = p0 + h
        w2, w3 = p0 + h + c.proc_net_ms, p3 + h + c.proc_net_ms
        perturbed = self._last_one_way is not None and h > self._last_one_way
        self._last_one_way = h
        covering = [w for w in windows if w.start_s <= t_s < w.end_s]

        if T < c.collapse_hops * h:
            # Nothing commits; every replica keeps timing out.
            n = 1000.0 / (T + 0.5 * h)
            # QCs still form between the timeouts, the votes a Nagle hop late (measured
            # 325 ms at 5+150, T 300: about 2h; hsc_T300 cell, 2026-10-19).
            qc = (2.0 * h + c.proc_net_ms) * float(math.exp(self.rng.normal(0.0, c.round_sigma)))
            for w in covering:
                w.timeouts += n
                w.stalled_timeouts += n
                w.qcs.append(qc)
            self.locked = True
            self._clock_ms = (t_s + 1.0) * 1000.0
            self._next_round = None
            self._gap_pending = True
            return

        hold = self._hold(ph)
        # The voters are awake under a slow leader only once the hold outlasts the slack the
        # round already spends on sleeping voters (set below; 2026-10-19: a 6-9 ms hold left
        # the real field at 30-35, the sleeping-voter value).
        hold_active = False
        if T >= w3:
            self.locked = False
        elif T < w2:
            self.locked = True
        elif perturbed and self.rng.random() < c.tip_prob:
            self.locked = True

        degraded = T < p3 + c.proc_net_ms
        base = p3 + c.degraded_hops * h if degraded else (p3 if self.locked else p0)
        share = c.degraded_share if degraded else 1.0
        # One 4-round cycle; round 3 is the slow leader's (its hold precedes its block).
        cycle = [base, base, base, base]
        # The round a timeout certificate ends (the slow leader's, held too long) forms no QC.
        tc_round = False
        timeouts_per_cycle = (4.0 if degraded else (1.0 if self.locked else 0.0))
        if hold > 0.0 and not degraded:
            # Where the block-delay sleeps set the pace, the hold first uses up
            # the slack the round already spends waiting for sleeping voters
            # (measured at 5 ms: a 70 / 200 / 640 ms hold adds 42 / 165 / 601 ms).
            hold = max(0.0, hold - max(0.0, base - (2.0 * h + c.proc_net_ms)))
            hold_active = hold > 0.0
            if base + hold + c.proc_net_ms >= T:
                # The waiting replicas time out before the held block arrives:
                # a timeout certificate ends the round and the block is lost.
                cycle[3] = T + 2.0 * h + c.proc_net_ms
                tc_round = True
                timeouts_per_cycle += 3.0
                if w2 <= T < w3:
                    self.locked = True
            elif base + h + hold + c.proc_net_ms >= T:
                # The previous leader times out; its next vote waits a round trip.
                cycle[3] = base + hold + h
                timeouts_per_cycle += 1.0
                if w2 <= T < w3 and self.rng.random() < c.tip_prob:
                    self.locked = True
            else:
                cycle[3] = base + hold

        # Load commits by time, capped per round by the mempool; rounds land
        # in the second they end in.
        tps = committed_tps(cycle, share * c.load_tps, c.residence_rounds)
        per_round_cap = MAX_QUEUED_TRANSACTIONS / c.residence_rounds
        batch_sigma = c.batch_sigma_sleep if sleep_bound else c.batch_sigma_net
        for w in covering:
            w.tx += tps
        second_end = (t_s + 1.0) * 1000.0
        if self._next_round is None:
            # Idle until now (start-up, after a collapse); a round already
            # under way keeps its start, however long it runs.
            self._clock_ms = max(self._clock_ms, t_s * 1000.0)
        while True:
            if self._next_round is None:
                k = self._slot % N_NODES
                dur = cycle[k] * float(math.exp(self.rng.normal(0.0, c.round_sigma)))
                following = sum(cycle[(k + j) % N_NODES] for j in range(3))
                latency = following * float(math.exp(self.rng.normal(0.0, c.round_sigma)))
                if k == 0:
                    latency += h  # this replica's own block
                qc_mean = (h + c.proc_net_ms) if hold_active and not degraded else max(c.proc_net_ms, base - h)
                lost = k == 3 and tc_round
                qc = None if lost else qc_mean * float(
                    math.exp(self.rng.normal(0.0, c.round_sigma)))
                # The round's batch: the load its duration brings in, capped by the
                # mempool, with the regime's mean-one lognormal spread.
                batch = min(share * c.load_tps * dur / 1000.0, per_round_cap) * float(
                    math.exp(self.rng.normal(-0.5 * batch_sigma ** 2, batch_sigma)))
                self._next_round = (dur, latency, qc, batch, lost)
            dur, latency, qc, batch, lost = self._next_round
            if self._clock_ms + dur > second_end:
                break
            self._clock_ms += dur
            self._next_round = None
            self._slot += 1
            for w in covering:
                if not lost:
                    # A window counts a skipped round from its second commit on (metrics.rs).
                    if self._gap_pending and w.latencies:
                        w.regency_changes += 1
                    # A round lost to a timeout certificate commits nothing: its duration
                    # joins the gap before the next commit.
                    w.rounds.append(dur + self._lost_ms)
                    w.latencies.append(latency)
                    w.batches.append(batch)
                if qc is not None:
                    w.qcs.append(qc)
                # A replica's share of the cycle's timeouts.
                w.timeouts += timeouts_per_cycle / N_NODES
                if degraded:
                    w.stalled_timeouts += 0.2
            if lost:
                self._lost_ms += dur
                self._gap_pending = True
            else:
                self._lost_ms = 0.0
                self._gap_pending = False

    def run_episode(
        self, choose: Callable[[Dict[str, float]], float]
    ) -> Tuple[Dict[str, float], float, Dict[str, float], int]:
        t0 = self.episode_start_s
        feature = _Window(t0, t0 + FEATURE_S)
        for s in range(int(FEATURE_S)):
            self._second(t0 + s, [feature])
        report = feature.report(self.timeout_ms, FEATURE_S * 1000.0)
        new_timeout = float(choose(report))
        for s in range(int(FEATURE_S), int(FEATURE_S + REPLY_S)):
            self._second(t0 + s, [])
        self.timeout_ms = new_timeout
        reward = _Window(t0 + FEATURE_S + REPLY_S + WARMUP_S, t0 + EPISODE_S)
        for s in range(int(FEATURE_S + REPLY_S), int(EPISODE_S)):
            self._second(t0 + s, [reward])
        # The phase in force at the apply instant, as main.py labels an episode.
        phase_idx, _ = self.phase_at(t0 + FEATURE_S + REPLY_S)
        self.episode_start_s = t0 + EPISODE_S
        return report, new_timeout, reward.report(new_timeout, REWARD_S * 1000.0), phase_idx


def sample_constants(rng: np.random.Generator) -> SimConstants:
    # Every constant 0.5-2x around its default (the fitted sim_fit.py FREE set as
    # before, the measured spreads since restart D11, 2026-10-19), the share and
    # the probability capped at 1 by narrowing the factor; the load around the
    # protocol's rate (sim_common).
    h = SimConstants()
    return SimConstants(
        proc_net_ms=around(rng, h.proc_net_ms),
        proc_sleep_ms=around(rng, h.proc_sleep_ms),
        round_sigma=around(rng, h.round_sigma),
        batch_sigma_sleep=around(rng, h.batch_sigma_sleep),
        batch_sigma_net=around(rng, h.batch_sigma_net),
        load_tps=draw_load_tps(rng, RATE_TPS),
        degraded_hops=around(rng, h.degraded_hops),
        degraded_share=around(rng, h.degraded_share, hi=min(2.0, 1.0 / h.degraded_share)),
        collapse_hops=around(rng, h.collapse_hops),
        tip_prob=around(rng, h.tip_prob, hi=min(2.0, 1.0 / h.tip_prob)),
        residence_rounds=around(rng, h.residence_rounds),
    )


# train --no-proposal-faults: a prior that never saw a delayed or silent
# proposer, so its pretraining covers the network-delay regimes only. The
# "blind pretrained" arm of the ablation - see main.py --prior-path.
NO_PROPOSAL_FAULTS = False


# Share of the fault phases with an extra link delay: the old one-way draw,
# log-uniform 2-250 ms, put a third of the phases under 10 ms (restart D11,
# 2026-10-19, as autobahn_sim).
LINK_PHASE_SHARE = 2.0 / 3.0


def sample_phase(rng: np.random.Generator, start_s: float, base_ms: float) -> Phase:
    """A fault phase at start_s over the run's base one-way delay (module docstring)."""
    one_way = base_ms + (draw_link_extra_ms(rng) if rng.random() < LINK_PHASE_SHARE else 0.0)
    kind = rng.random()
    offset = hold = None
    if kind < 0.35:
        pass
    elif kind < 0.8:
        # Relative hold max(0, T - X): X, the timeout above which the slow leader
        # costs, is drawn as a hold magnitude (module docstring).
        offset = draw_hold_ms(rng)
    else:
        hold = draw_hold_ms(rng)
    if NO_PROPOSAL_FAULTS:
        offset = hold = None
    return Phase(start_s=start_s, one_way_ms=one_way, hold_offset_ms=offset, hold_ms=hold)


def sample_schedule(rng: np.random.Generator, total_s: float) -> List[Phase]:
    base_ms = draw_base_delay_ms(rng)
    phases = [Phase(0.0, one_way_ms=base_ms)]
    phases += [sample_phase(rng, t, base_ms) for t in fault_start_times(rng, total_s)]
    return phases


# config/hotstuff/failure_spec.xml: warm-up 10 s, then (atTime, one-way, hold
# offset) per 300 s phase, None for no slow leader, and the oracle's timeout
# per phase (learning_agent/config/schedule_hotstuff.csv). Used only to
# evaluate policies in simulation, never to train; check_spec_files raises when
# either file disagrees (restart D11, 2026-10-19).
SPEC_PHASES: Tuple[Tuple[float, float, Optional[float]], ...] = (
    (0.0, 5.0, None),
    (300.0, 5.0, 100.0),
    (600.0, 55.0, None),
    (900.0, 55.0, 400.0),
    (1200.0, 155.0, None),
    (1500.0, 5.0, None),
)
SPEC_ORACLE_MS: Tuple[float, ...] = (100.0, 100.0, 300.0, 300.0, 700.0, 100.0)
SPEC_WARMUP_S = 10.0
# Episode e applies at warm-up + 20 (e - 1) s, on the phase onsets, as the engine does.
SPEC_START_S = SPEC_WARMUP_S - FEATURE_S - REPLY_S
SPEC_EPISODES = int((SPEC_WARMUP_S + SPEC_PHASES[-1][0] + 300.0 + 10.0) // EPISODE_S)


_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC_XML_PATH = os.path.join(_REPO_DIR, "config", "hotstuff", "failure_spec.xml")
SPEC_CSV_PATH = os.path.join(_REPO_DIR, "learning_agent", "config", "schedule_hotstuff.csv")
# The spec's slow leader: replica 3, the simulator's slot 3.
SLOW_LEADER_ID = "3"


def _spec_extra_delay(net, where: str) -> float:
    """A <network> section's global extra one-way delay; raises on anything else."""
    if net is None:
        return 0.0
    for tag in ("sources", "edges", "silence"):
        node = net.find(tag)
        if node is not None and (len(node) or (node.text or "").strip()):
            raise ValueError(f"{where}: non-empty <{tag}>; hotstuff_sim models only a global link delay")
    g = net.find("global")
    delay = float(g.findtext("delayMs") or 0.0) if g is not None else 0.0
    if delay == 0.0:
        return 0.0
    dur, intv = g.findtext("burstDurationMs"), g.findtext("burstIntervalMs")
    if dur is None or intv is None or not float(dur) >= float(intv) > 0.0:
        raise ValueError(f"{where}: global delay {delay:g} ms is not continuous (burstDurationMs >= burstIntervalMs > 0)")
    return delay


def _spec_hold_offset(section, where: str) -> Optional[float]:
    """The slow leader's hold offset X (delayRelativeToTimeoutMs -X), None without one."""
    pd = section.find("proposalDelay") if section is not None else None
    if pd is None:
        return None
    reps = [r for r in (pd.find("replicas") if pd.find("replicas") is not None else [])
            if (r.findtext("id") or "").strip()]
    if not reps:
        return None
    if len(reps) > 1 or reps[0].findtext("id").strip() != SLOW_LEADER_ID or reps[0].findtext("delayMs") is not None:
        raise ValueError(f"{where}: SPEC_PHASES models one relative slow leader, replica {SLOW_LEADER_ID}")
    rel = reps[0].findtext("delayRelativeToTimeoutMs")
    if rel is None:
        raise ValueError(f"{where}: replica {SLOW_LEADER_ID} has no delayRelativeToTimeoutMs")
    return -float(rel)


def check_spec_files(xml_path: str = SPEC_XML_PATH, csv_path: str = SPEC_CSV_PATH) -> None:
    """Raise unless SPEC_WARMUP_S / SPEC_PHASES match the failure spec (warm-up, base
    delay plus each phase's continuous extra delay, replica 3's relative hold, jitter
    JITTER_RATIO) and SPEC_ORACLE_MS the oracle CSV (each phase's last row at or
    before its atTime, no switch inside a phase, the warm-up row the first phase's,
    as evaluate_spec plays it)."""
    import xml.etree.ElementTree as ET

    root = ET.parse(xml_path).getroot()
    warm = float(root.findtext("warmUpTime"))
    gnd = root.find("globalNetworkDelay")
    base = float(gnd.findtext("delayMs"))
    jitter = gnd.findtext("jitterRatio")
    if jitter is None or float(jitter) != JITTER_RATIO:
        raise ValueError(f"{xml_path}: jitterRatio {jitter} != {JITTER_RATIO} (the applier requires it)")
    phases = []
    for i, ph in enumerate(root.find("phases")):
        where = f"{xml_path} phase #{i}"
        if ph.find("atTimeMs") is not None:
            raise ValueError(f"{where}: atTimeMs; SPEC_PHASES is in seconds")
        phases.append((float(ph.findtext("atTime")),
                       base + _spec_extra_delay(ph.find("network"), where),
                       _spec_hold_offset(ph.find("hotstuff"), where)))
    if warm != SPEC_WARMUP_S or tuple(phases) != SPEC_PHASES:
        raise ValueError(f"{xml_path} gives warm-up {warm:g} and phases {phases}; "
                         f"hotstuff_sim has {SPEC_WARMUP_S:g} and {list(SPEC_PHASES)}")
    rows = []
    with open(csv_path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                t, ms = line.split(",")
                rows.append((float(t), float(ms)))
    starts = [at for at, _, _ in SPEC_PHASES]
    inside = [t for t, _ in rows if t >= 0.0 and t not in starts]
    if inside:
        raise ValueError(f"{csv_path}: rows at {inside} s switch inside a phase; SPEC_ORACLE_MS cannot")
    oracle = []
    for at in starts:
        before = [ms for t, ms in rows if t <= at]
        if not before:
            raise ValueError(f"{csv_path}: no row covers the phase at {at:g} s")
        oracle.append(before[-1])
    warm_rows = [ms for t, ms in rows if t < 0.0]
    if warm_rows and warm_rows[-1] != SPEC_ORACLE_MS[0]:
        raise ValueError(f"{csv_path}: warm-up row {warm_rows[-1]:g} != the first phase's {SPEC_ORACLE_MS[0]:g}")
    if tuple(oracle) != SPEC_ORACLE_MS:
        raise ValueError(f"{csv_path} gives the oracle {oracle}; hotstuff_sim has {list(SPEC_ORACLE_MS)}")


def spec_schedule() -> List[Phase]:
    check_spec_files()
    phases = [Phase(0.0, one_way_ms=5.0)]
    for at, one_way, offset in SPEC_PHASES:
        phases.append(Phase(SPEC_WARMUP_S + at, one_way_ms=one_way, hold_offset_ms=offset))
    return phases


# --- Agent loop, data generation, prior training, evaluation -----------------


def feature_vector(report: Dict[str, float]) -> np.ndarray:
    from models import HOTSTUFF_REPORT_FEATURE_FIELD_NAMES

    return np.asarray(
        [report[name] for name in HOTSTUFF_REPORT_FEATURE_FIELD_NAMES], dtype=float
    )


_INT_FIELDS = (
    "total_transactions", "total_consensus_instances", "leader_change_count",
    "regency_change_count", "timeout_ms", "view_change_count",
    "no_progress_view_change_count",
)


def episode_reward(report: Dict[str, float]) -> float:
    """Reward of one reward window under the process-wide reward design, the
    same calculator main.reward_engineering uses."""
    from reward_utils import calculate_hotstuff_reward

    return calculate_hotstuff_reward(
        **{
            name: int(round(value)) if name in _INT_FIELDS else float(value)
            for name, value in report.items()
            if name != "proposal_to_qc_ms"  # a model input, not a reward term (as smartbft_sim)
        }
    )


def run_agent(
    sim: HotstuffSim,
    episodes: int,
    predict: Callable[[np.ndarray], float],
    update: Optional[Callable[[np.ndarray, float, float], None]],
    spec_warmup: bool = True,
) -> List[Tuple[int, float, Dict[str, float]]]:
    """Drive ``sim`` like LearningAgentServicer: predict at the feature deadline,
    reward from the reward window, no update for warm-up episodes.

    ``spec_warmup``: the schedule opens with spec_schedule's warm-up phase, so
    spec phase k is sim phase k + 1. False for a sample_schedule run, whose
    phase 0 is a real phase and is updated on."""
    shift = 1 if spec_warmup else 0
    rows = []
    for _ in range(episodes):
        decision_s = sim.episode_start_s + FEATURE_S
        holder: Dict[str, np.ndarray] = {}

        def choose(report: Dict[str, float]) -> float:
            holder["x"] = feature_vector(report)
            return predict(holder["x"])

        _, timeout_ms, reward_report, phase_idx = sim.run_episode(choose)
        phase = phase_idx - shift if decision_s >= SPEC_WARMUP_S else -1
        if update is not None and phase != -1:
            update(holder["x"], timeout_ms, episode_reward(reward_report))
        rows.append((phase, timeout_ms, reward_report))
    return rows


def training_sim(rng: np.random.Generator, seed: int) -> Tuple[HotstuffSim, int]:
    """One randomized training run: (simulator, episodes)."""
    episodes = int(rng.integers(40, 120))
    # The shared initial timeout (sim_common; restart D11, 2026-10-19: was a
    # quarter of the runs on the parameters.json default 5000).
    initial = draw_initial_timeout_ms(rng, agent_pb2.PROTOCOL_HOTSTUFF)
    sim = HotstuffSim(
        sample_constants(rng),
        sample_schedule(rng, episodes * EPISODE_S),
        seed=seed,
        start_offset_s=float(rng.uniform(0.0, 20.0)),
        initial_timeout_ms=initial,
    )
    return sim, episodes


def _generate_one(args: Tuple[int, str, int]) -> Tuple[np.ndarray, np.ndarray]:
    from reward_utils import set_reward_design

    seed, design, hold = args
    set_reward_design(design)
    from continuous_bandit import ContinuousTimeoutActionCodec, report_context
    from models import HOTSTUFF_REPORT_FEATURE_FIELD_NAMES

    spec = CONTEXT_SPECS[agent_pb2.PROTOCOL_HOTSTUFF]
    lower_ms, upper_ms = spec.bounds_ms
    rng = np.random.default_rng(seed)
    codec = ContinuousTimeoutActionCodec(agent_pb2.PROTOCOL_HOTSTUFF, bounds_ms=spec.bounds_ms)
    sim, episodes = training_sim(rng, seed)
    prev = codec.normalize_ms(int(round(min(max(sim.timeout_ms, lower_ms), upper_ms))))
    xs, ys = [], []
    # hold > 1: an action is drawn at the start of a hold and kept for h
    # episodes; the sample is (context at the hold's start, action, mean reward
    # over the hold), the held-action return SimReturnGPModel's prior predicts.
    held: Dict[str, float] = {}
    rewards: List[float] = []
    for _ in range(episodes):

        def choose(report):
            if not held:
                x = feature_vector(report)
                ctx = report_context(spec, HOTSTUFF_REPORT_FEATURE_FIELD_NAMES, x)
                if rng.random() < 0.6:
                    a = float(rng.random())
                else:
                    a = float(np.clip(prev + rng.normal(0.0, 0.08), 0.0, 1.0))
                ms = codec._denormalize_value(a, 0)
                held.update(ctx=ctx, a=codec.normalize_ms(ms), ms=float(ms))
            return held["ms"]

        _, _, reward_report, _ = sim.run_episode(choose)
        rewards.append(episode_reward(reward_report))
        if len(rewards) == hold:
            xs.append(np.append(held["ctx"], held["a"]))
            ys.append(float(np.mean(rewards)))
            prev = held["a"]
            held.clear()
            rewards = []
    return np.asarray(xs), np.asarray(ys)


def generate(n_runs: int, seed: int, workers: int, design: str, hold: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    from multiprocessing import Pool

    if int(hold) < 1:
        raise ValueError(f"hold must be >= 1 episode, got {hold}")
    seeds = [(seed + i, design, int(hold)) for i in range(n_runs)]
    with Pool(workers) as pool:
        parts = pool.map(_generate_one, seeds, chunksize=8)
    parts = [p for p in parts if len(p[0])]
    return np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def make_policy(kind: str, prior_path: str = DEFAULT_PRIOR_PATH):
    """(predict, update) closures: ``oracle:<a,b,c>`` plays the spec phase's rung,
    ``fixed:<ms>`` one timeout, ``candidate`` SimReturnGPModel as main.serve builds it."""
    from continuous_bandit import ContinuousTimeoutActionCodec, SimReturnGPModel
    from models import HOTSTUFF_REPORT_FEATURE_FIELD_NAMES as names
    from reward_utils import get_reward_design

    protocol = agent_pb2.PROTOCOL_HOTSTUFF
    if kind.startswith("fixed:"):
        value = float(kind.split(":", 1)[1])
        return (lambda x: value), None
    if kind == "candidate":
        check_prior_windows(prior_path, SIM_WINDOWS)
        spec = CONTEXT_SPECS[protocol]
        codec = ContinuousTimeoutActionCodec(protocol, bounds_ms=spec.bounds_ms)
        model = SimReturnGPModel(
            action_codec=codec,
            field_names=names,
            prior_path=prior_path,
            reward_design=get_reward_design(),
        )

        def predict(x: np.ndarray) -> float:
            timeout = codec.decode(model.predict_action(x))
            return float(timeout.hotstuff.timeout_delay_milliseconds)

        def update(x: np.ndarray, timeout_ms: float, reward: float) -> None:
            used = agent_pb2.HotstuffTimeout(timeout_delay_milliseconds=int(round(timeout_ms)))
            lower, upper = spec.bounds_ms
            if lower <= used.timeout_delay_milliseconds <= upper:
                model.update_model(x, codec.encode(used), reward)

        return predict, update
    raise ValueError(kind)


def evaluate_spec(kind: str, constants: SimConstants, seed: int, **kwargs) -> np.ndarray:
    """Per-phase (mean reward after the switch episode, mean p50, median timeout,
    mean reward over the whole phase including its switch episode) of one policy
    on the spec."""
    sim = HotstuffSim(constants, spec_schedule(), seed=seed, start_offset_s=SPEC_START_S)
    if kind == "oracle":
        def predict(x: np.ndarray) -> float:
            idx, _ = sim.phase_at(sim.episode_start_s + FEATURE_S + REPLY_S)
            return SPEC_ORACLE_MS[max(0, idx - 1)]
        update = None
    else:
        predict, update = make_policy(kind, **kwargs)
    rows = run_agent(sim, SPEC_EPISODES, predict, update)
    out = np.zeros((len(SPEC_PHASES), 4))
    for ph in range(len(SPEC_PHASES)):
        whole = [r for r in rows if r[0] == ph]
        sel = whole[1:]
        out[ph] = (
            np.mean([episode_reward(r[2]) for r in sel]),
            np.mean([r[2]["p50_consensus_latency_ms"] for r in sel]),
            np.median([r[1] for r in sel]),
            np.mean([episode_reward(r[2]) for r in whole]),
        )
    return out


def pretrain_adapter():
    """This simulator as pretrain.py drives it: training runs as train draws
    them, spec runs as evaluate_spec builds them (SPEC_EPISODES), and the
    oracle evaluate_spec plays: the rung of the phase in force when the
    timeout is applied."""
    from functools import partial

    from pretrain import SimAdapter, timeout_message

    def to_timeout(timeout_ms: float) -> agent_pb2.Timeout:
        return agent_pb2.Timeout(hotstuff=agent_pb2.HotstuffTimeout(
            timeout_delay_milliseconds=int(round(timeout_ms))))

    def from_timeout(timeout: agent_pb2.Timeout) -> float:
        return float(timeout_message(timeout, "hotstuff").timeout_delay_milliseconds)

    def oracle(sim: HotstuffSim) -> Callable[[np.ndarray], float]:
        def predict(x: np.ndarray) -> float:
            idx, _ = sim.phase_at(sim.episode_start_s + FEATURE_S + REPLY_S)
            return SPEC_ORACLE_MS[max(0, idx - 1)]

        return predict

    return SimAdapter(
        protocol=agent_pb2.PROTOCOL_HOTSTUFF,
        simulator="hotstuff_sim",
        training_run=lambda seed: training_sim(np.random.default_rng(seed), seed),
        spec_run=lambda seed: (
            # The engine opens at the middle rung (--initial-timeout in the rows, D10b).
            HotstuffSim(SimConstants(), spec_schedule(), seed=seed, start_offset_s=SPEC_START_S,
                        initial_timeout_ms=float(opening_action_ms(agent_pb2.PROTOCOL_HOTSTUFF)[0])),
            SPEC_EPISODES,
        ),
        run_agent=run_agent,
        run_training=partial(run_agent, spec_warmup=False),
        to_timeout=to_timeout,
        from_timeout=from_timeout,
        row_phase=lambda row: row[0],
        row_reward=lambda row: episode_reward(row[2]),
        spec_phase_count=len(SPEC_PHASES),
        oracle=oracle,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="generate randomized episodes and fit the prior")
    t.add_argument("--code-hash", required=True,
                   help="code_hash of the tree training the prior (scripts/cloudlab/code_state.py), recorded in the prior and compared with FREEZE by the harness")
    t.add_argument("--runs", type=int, default=4000)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    t.add_argument("--out", default=DEFAULT_PRIOR_PATH)
    t.add_argument("--reward-design", default=DEFAULT_REWARD_DESIGN)
    t.add_argument("--no-proposal-faults", action="store_true",
                   help="train on network-delay phases only; the prior never "
                        "sees a delayed or silent proposer")
    # One episode: HotStuff rotates its leader every round, nothing is re-pinned
    # per 60 s (restart D11, 2026-10-19; PBFT/SBFT/SmartBFT hold 3).
    t.add_argument("--hold-episodes", type=int, default=1)
    t.add_argument("--target-floor", type=float, default=-2.0)
    t.add_argument("--hidden", default="128,128,64", help="MLP hidden layer sizes")
    e = sub.add_parser("eval", help="compare policies on the spec schedule in simulation")
    e.add_argument("--seeds", type=int, default=10)
    e.add_argument("--prior", default=DEFAULT_PRIOR_PATH)
    e.add_argument("--rungs", default="100,300,700", help="the fixed timeouts to compare")
    e.add_argument("--randomize", action="store_true", help="random constants per seed")
    e.add_argument("--reward-design", default=DEFAULT_REWARD_DESIGN)
    args = parser.parse_args()
    from reward_utils import set_reward_design

    set_reward_design(args.reward_design)
    if args.cmd == "train":
        global NO_PROPOSAL_FAULTS
        NO_PROPOSAL_FAULTS = bool(args.no_proposal_faults)  # inherited by the forked workers
        x, y = generate(args.runs, args.seed, args.workers, args.reward_design, args.hold_episodes)
        rmse = train_gp_prior(
            agent_pb2.PROTOCOL_HOTSTUFF, x, y, args.out, seed=args.seed, design=args.reward_design,
            hold=args.hold_episodes, target_floor=args.target_floor, windows=SIM_WINDOWS,
            sim_constants=asdict(SimConstants()), code_hash=args.code_hash,
            hidden=tuple(int(v) for v in args.hidden.split(",")))
        print(f"samples={len(x)} hold_episodes={args.hold_episodes} target_floor={args.target_floor} "
              f"reward_std={y.std():.4f} floored_std={np.maximum(y, args.target_floor).std():.4f} "
              f"holdout_rmse={rmse:.4f} -> {args.out}")
        return

    rungs = [float(v) for v in args.rungs.split(",")]
    rng = np.random.default_rng(12345)
    for s in range(args.seeds):
        constants = sample_constants(rng) if args.randomize else SimConstants()
        if args.randomize:
            constants.load_tps = RATE_TPS
        res = {"oracle": evaluate_spec("oracle", constants, s)}
        for r in rungs:
            res[f"fixed:{r:g}"] = evaluate_spec(f"fixed:{r:g}", constants, s)
        res["candidate"] = evaluate_spec("candidate", constants, s, prior_path=args.prior)
        line = " | ".join(
            f"{k} " + " ".join(
                f"p{ph} {v[ph,0]:+.3f}@{v[ph,2]:4.0f} (whole {v[ph,3]:+.3f})" for ph in range(len(SPEC_PHASES))
            )
            for k, v in res.items()
        )
        print(f"seed {s} | {line}")


if __name__ == "__main__":
    main()
