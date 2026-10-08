"""Analytic Autobahn simulator for pretraining the continuous model.

Built from the mechanics of autobahn/primary/src (core.rs, proposer.rs,
leader.rs, the vote and proposal faults in adaptive/src/failure.rs) as mapped
in scripts/cloudlab/tmp/cal/pt/map_autobahn.txt, and calibrated on the cells
measured on CloudLab in scripts/cloudlab/tmp/cal/ab (2026-10-02, cloudlab_1,
10,000 tx/s), not fit to the failure spec. One slot at a time, in order:

- Cars. Every lane makes a car when its 100 ms ``max_header_delay`` timer has
  run and its previous car is certified, so the lanes tick together at
  ``C = max(100, car_hops h + car_proc_ms)`` (header rates 39.6 / 20.4 / 8.7
  per s at 5 / 55 / 155 ms one-way give 101 / 196 / 460 ms per lane). Each car
  carries one payload digest per 100 ms worker batch, about 40 per s at any
  load the harness runs.
- Slots. The leader of (slot, view) is ``(slot + view) % 4``, replica 3 leads
  one slot in four. Prepare(s) goes out once 3 of the 4 lanes have a newly
  certified tip (about 2h after the car is made), Prepare(s-1) has reached the
  leader (the ticket) and slot s-4 has committed (k = 4); it proposes every
  car the leader has seen, certified or not. So three cars in four wait about
  one certification for their slot and the rest a slot more (report p50 and
  p95 about 100 / 176 at base). Slots execute strictly in order.
- Commit path. Votes go to the leader only. With all four votes within the
  fast-path wait F = 50 of the third the leader commits (3 hops); otherwise
  the third vote starts F, then Confirm, votes and Commit (5 hops + F). No
  socket sets TCP_NODELAY, so every hop carries a uniform extra of up to
  ``2 nagle_frac min(h, nagle_cap_ms)``, and every view's round trips a
  common lognormal spread (``view_sigma``). The extra grows with h through
  5+150: replica 3's Prepare-to-Confirm vote gap is 2h + F plus 10 / 120 /
  260 ms at h = 5 / 55 / 155, and a view lasts T + 80 at 5+50 and T + 230
  at 5+150 (2026-10-07 single-host runs). A vote is rarely late by an
  exponential tail.
- Timer. Slot s's timer starts when Prepare(s-1) is processed (or slot s-4
  commits); expiry is a Timeout, and Timeouts form a TC one hop later. The
  next view's leader proposes ``view_change_hops`` capped hops after it.
  Without vote withholding a view whose votes are cast still commits on any
  3 of them, so healthy Autobahn has no floor. With replica 3 withholding its
  votes every quorum needs all three honest replicas, and a TC that reaches
  one of them before the Confirm starves the view: the slot pays the TC plus
  a new view's slow path, and below the path's need the views starve one
  after another (the soft floors near 400 at 5+50 and 1000 at 5+150).
- Replica 3's own views. A leader processes its own vote locally
  (core.rs:1586-1588); only votes sent to another leader are held
  (core.rs:1601-1614). So a view replica 3 leads, one in four, is a healthy
  view: any two honest votes make its quorum, all three its fast path, which
  needs only the Prepare in before the TC (a Commit is accepted in any view).
  Below the slow path's need a slot cycles views until replica 3's turn, so
  a timeout far below the need costs less than one just under it, whose
  failed views each last T + h + half the extra. Measured at 5+150 (GP run
  pt_ab_continuous_gp_h8_20261007_180512): slots end one or two views after
  their first replica-3 view in 72-94% at T = 100-430, client R about -1.4
  at T = 100, -1.8 to -2.2 at 346-627, -0.8 at 915-1135.
- Timeout traffic. A genuine timeout gives the receivers data to piggyback
  their delayed ACKs on, so recent timeouts take ``flush_ms`` off every
  hop's extra. That is what makes a timeout above the link's need cost
  latency (healthy 5+50: 566 ms client p50 at T=200, 627 at 1000; withheld
  5+150: 1772 at 1000, 2079 at 2000) and keeps a low timeout from stalling.
- Slow leader. Replica 3 holds every Prepare it leads (any view) until X ms
  before its live timeout (``T - X``) or by a fixed hold; a held Prepare is
  dropped once a TC moves the slot on. Later slots wait on it through the
  ticket, so a timeout above what the link needs is paid once per four slots.
- Late spurious timeouts. Replica 3's withheld votes arrive 10 s late, complete
  the old slot's fast QC and re-arm a later slot's timer, which fires
  harmlessly unless that slot's late Commit cancels it first: about one view
  change per slot per replica while T is under three slot intervals (map
  section 1.6). They count in the report and nowhere else.

The replica report is synthesized as the Rust side computes it: one sample
per committed header (about four per slot), latency from the header's first
sighting at the reporting replica to its execution, transactions and
throughput counted in payload digests, leader changes between consecutive
committed headers' authors, bursty inter-commit gaps (their p50 is the host's in-burst delivery gap,
``burst_gap_ms``, drawn per run so the prior learns to ignore it). The
client-observed
latency (header latency plus the batch and car waits, the quantity the
calibration cells were judged on) is kept beside it for ``check`` and
``eval``.

Episodes follow the wall-clock learning windows in lib_rustbft.sh: 10 s
features, 2 s reply wait (applied at the deadline), 3 s warm-up, 5 s reward.

Constants the mechanics do not fix (car and slot processing, the TCP extra
per hop and what timeout traffic removes of it, the view-change delay, noise)
were fit to the measured cells and the 2026-10-07 single-host 5+150
points (``check``: the client R ranking within 11 of the 12 (link, fault)
groups, mean absolute R error 0.10) and are drawn per simulated training run
from ranges around those values. The simulator knows
the fault mechanisms, not the spec's values.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

import agent_pb2
from continuous_bandit import CONTEXT_SPECS

N_NODES = 4
# config/autobahn/parameters.json: k, fast_path_timeout, max_header_delay,
# max_batch_delay.
K = 4
FAST_PATH_MS = 50.0
MAX_HEADER_DELAY_MS = 100.0
MAX_BATCH_DELAY_MS = 100.0
# The faulty replica (failure_spec.xml: vote withholding and the slow leader).
ATTACKER = 3
# How long replica 3 holds its votes (failure_spec.xml voteDelay).
VOTE_HOLD_MS = 10000.0
# The judge's request timeout: a 20 s chunk with no commits scores this p50.
CLIENT_STALL_P50_MS = 10000.0
# config/autobahn/parameters.json timeout_delay: in effect until the first
# decision.
DEFAULT_TIMEOUT_MS = 5000.0
FEATURE_S = 10.0
REPLY_S = 2.0
WARMUP_S = 3.0
REWARD_S = 5.0
EPISODE_S = FEATURE_S + REPLY_S + WARMUP_S + REWARD_S

DEFAULT_PRIOR_PATH = CONTEXT_SPECS[agent_pb2.PROTOCOL_AUTOBAHN].prior_path


@dataclass
class SimConstants:
    """Defaults are the calibration host (cloudlab_1, 10,000 tx/s): the car
    period from the measured header rates, the rest a search over the
    ``check`` cells plus the single-host 5+150 withheld points at T = 100
    and 346 (rank agreement, then R level, then the oracle report's p50 and
    p95; scripts/cloudlab/tmp/cal/pt/absim/search3.py), with the TCP extra
    uncapped through 5+150 as the Prepare-to-Confirm vote gaps and view
    lengths measure it. Training draws every field from sample_constants."""

    # Car period max(100, car_hops * h + car_proc_ms); lanes jitter by
    # lane_sigma * C around the common tick, the tick by car_sigma.
    car_hops: float = 2.65
    car_proc_ms: float = 50.0
    car_sigma: float = 0.05
    lane_sigma: float = 0.15
    # A car is certified cert_hops * h plus one hop's TCP extra after it is made.
    cert_hops: float = 1.739
    # From the third lane's newly certified car to the Prepare.
    slot_proc_ms: float = 5.0
    # TCP extra per hop: uniform on [0, 2 nagle_frac min(h, nagle_cap_ms)],
    # nagle_common of it shared by all voters of a round. It grows with h
    # through 5+150: replica 3's Prepare-to-Confirm vote gap is 2h + F plus
    # 10 / 120 / 260 ms at h = 5 / 55 / 155 (both 2026-10-07 single-host runs,
    # 5-host 70 / 271 / 594), and a view lasts T + 80 at 5+50 and T + 230 at
    # 5+150 (T + h plus half the extra), so the cap sits above the links
    # calibrated on.
    nagle_frac: float = 0.422
    nagle_cap_ms: float = 216.7
    nagle_common: float = 0.641
    # Timeout traffic gives the receivers data to piggyback their ACKs on, so
    # the delayed-ACK part of the extra goes: each hop's extra shrinks by
    # flush_ms times the recent timeouts per slot (at most one).
    flush_ms: float = 15.21
    # A new view's Prepare goes out view_change_hops * min(h, nagle_cap_ms)
    # after the replicas start its timer.
    view_change_hops: float = 0.627
    # Lognormal spread of one view's round trips, common to its messages.
    view_sigma: float = 0.294
    # Per message processing.
    proc_ms: float = 0.8
    # A vote is late with this probability, by an exponential of mean
    # tail_hops * h + tail_ms (a minor term: 0 or 0.03 move check little).
    vote_tail_prob: float = 0.01
    tail_hops: float = 4.0
    tail_ms: float = 0.0
    # Client latency over the header's: a batch wait uniform on [0, 100] and a
    # car wait uniform on [0, 2 client_car_frac C], plus client_proc_ms.
    client_car_frac: float = 0.729
    client_proc_ms: float = 0.0
    # Transactions per second; enters only the client view (the report counts
    # digests, which do not depend on the load).
    load_tps: float = 10000.0
    # Share of the timeouts reported as no-progress, against the measured
    # 15 / 97 at base where every slot's late timeout interleaves with commits.
    no_progress_scale: float = 0.3
    # Gap between the headers of one slot as the executor delivers them, the
    # report's p50 inter-commit gap: a CPU constant of the host, 0.033-0.065 ms
    # on the 5-host layout (20261005 oracles) and 0.015-0.030 ms with all
    # replicas on one host (20261007 runs), about 20% apart window to window.
    burst_gap_ms: float = 0.04


@dataclass
class Phase:
    start_s: float
    one_way_ms: float = 5.0
    # Replica 3 withholds its consensus votes (VOTE_HOLD_MS).
    vote_withhold: bool = False
    # Slow leader: replica 3 holds each Prepare it leads until hold_offset_ms
    # before the live timeout, or by a fixed hold_ms; neither means none.
    hold_offset_ms: Optional[float] = None
    hold_ms: Optional[float] = None


def car_period_ms(c: SimConstants, h: float) -> float:
    return max(MAX_HEADER_DELAY_MS, c.car_hops * h + c.car_proc_ms)


@dataclass
class _Window:
    """The headers, slots and timeouts of one report window [start, end) in ms."""

    start_ms: float
    end_ms: float
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    latencies: np.ndarray = field(default_factory=lambda: np.zeros(0))
    digests: np.ndarray = field(default_factory=lambda: np.zeros(0))
    leader_changes: float = 0.0
    slots: int = 0
    timeouts: int = 0

    def report(self, timeout_ms: float, no_progress_scale: float, burst_gap_ms: float) -> Dict[str, float]:
        length_s = (self.end_ms - self.start_ms) / 1000.0
        n = len(self.latencies)
        if n:
            lat = np.sort(self.latencies)
            dig = np.sort(self.digests)
            # metrics.rs percentiles: the ceil-rank rule.
            rank = lambda v, q: float(v[max(0, math.ceil(q * len(v)) - 1)])
            avg, p50, p95 = float(lat.mean()), rank(lat, 0.5), rank(lat, 0.95)
            gaps = np.diff(self.times)
            # Headers of one slot are delivered in a burst, burst_gap_ms apart.
            gaps = np.sort(np.where(gaps < burst_gap_ms, burst_gap_ms, gaps)) if len(gaps) else np.zeros(1)
            gap_avg, gap_p50, gap_p95 = float(gaps.mean()), rank(gaps, 0.5), rank(gaps, 0.95)
            tx, batch_avg, batch_p95 = float(dig.sum()), float(dig.mean()), rank(dig, 0.95)
        else:
            # main.EMPTY_WINDOW_PROTOCOLS: the agent gives a window in which
            # nothing committed its own length as its latency (the Rust side
            # reports 0).
            avg = p50 = p95 = self.end_ms - self.start_ms
            gap_avg = gap_p50 = gap_p95 = 0.0
            tx = batch_avg = batch_p95 = 0.0
        if self.slots:
            no_progress = self.timeouts * self.timeouts / (self.timeouts + self.slots) * no_progress_scale
        else:
            no_progress = float(self.timeouts)
        return {
            "total_transactions": tx,
            "total_consensus_instances": float(n),
            "avg_consensus_latency_ms": avg,
            "p50_consensus_latency_ms": p50,
            "p95_consensus_latency_ms": p95,
            "throughput_tps": tx / length_s,
            "avg_batch_size": batch_avg,
            "p95_batch_size": batch_p95,
            "leader_change_count": float(self.leader_changes),
            "timeout_ms": float(timeout_ms),
            "avg_inter_commit_gap_ms": gap_avg,
            "p50_inter_commit_gap_ms": gap_p50,
            "p95_inter_commit_gap_ms": gap_p95,
            "view_change_count": float(self.timeouts),
            "no_progress_view_change_count": float(round(no_progress)),
        }


class AutobahnSim:
    def __init__(
        self,
        constants: SimConstants,
        phases: List[Phase],
        seed: int,
        start_offset_s: float = 0.0,
        initial_timeout_ms: float = DEFAULT_TIMEOUT_MS,
        track_client: bool = False,
    ):
        self.c = constants
        self.phases = sorted(phases, key=lambda ph: ph.start_s)
        self.rng = np.random.default_rng(seed)
        self._gap_rng = np.random.default_rng([seed, 1])
        self.episode_start_s = float(start_offset_s)
        self.timeout_ms = float(initial_timeout_ms)
        self._ub: List[float] = []
        self._ui = 0
        self._nb: List[float] = []
        self._ni = 0
        # Slot pipeline: the next slot, when the next leader holds slot s-1's
        # first delivered Prepare (its tips, the ticket), the commits of the
        # last K slots, the execution frontier, and the slot being resolved
        # [view, start, timer start, first Prepare, best commit] while its
        # views run.
        self._slot = 0
        self._prev_ticket = 0.0
        self._commits: Deque[float] = deque([-math.inf] * K, maxlen=K)
        self._preps: Deque[Tuple[float, bool]] = deque(maxlen=K - 1)
        self._exec = 0.0
        self._chain: Optional[list] = None
        # Recent genuine timeouts and commits (the TCP flush; late spurious
        # timeouts fire 10 s on, away from any slot's messages, and count not).
        self._recent_to: Deque[float] = deque()
        self._recent_slots: Deque[float] = deque()
        # Cars: one tick per row (creation time per lane, digests per lane),
        # the row index of _cars[0], each lane's next car not yet proposed.
        self._cars: Deque[Tuple[Tuple[float, ...], Tuple[float, ...]]] = deque()
        self._car_base = 0
        self._lane_next = [0, 0, 0, 0]
        self._next_tick_ms = 0.0
        self._last_period = car_period_ms(constants, self.phase_at(0.0)[1].one_way_ms)
        # Records: header (execution time, latency, digests), slot (execution
        # time, leader changes), timeouts at the reporting replica (replica 0).
        self._hdr_t: List[float] = []
        self._hdr_lat: List[float] = []
        self._hdr_dig: List[float] = []
        self._slot_t: List[float] = []
        self._slot_lc: List[float] = []
        self._to_t: List[float] = []
        # Client view (check and eval only): per second, (latency, tx) pairs.
        self.track_client = track_client
        self.client: Dict[int, List[Tuple[float, float]]] = {}

    # --- random numbers -----------------------------------------------------

    def _u(self) -> float:
        if self._ui >= len(self._ub):
            self._ub = self.rng.random(1 << 14).tolist()
            self._ui = 0
        self._ui += 1
        return self._ub[self._ui - 1]

    def _n(self) -> float:
        if self._ni >= len(self._nb):
            self._nb = self.rng.standard_normal(1 << 12).tolist()
            self._ni = 0
        self._ni += 1
        return self._nb[self._ni - 1]

    # --- phases ---------------------------------------------------------------

    def phase_at(self, t_s: float) -> Tuple[int, Phase]:
        idx = 0
        for i, ph in enumerate(self.phases):
            if ph.start_s <= t_s:
                idx = i
        return idx, self.phases[idx]

    def _hold(self, ph: Phase, timeout_ms: float) -> float:
        if ph.hold_offset_ms is not None:
            return max(0.0, timeout_ms - ph.hold_offset_ms)
        return float(ph.hold_ms) if ph.hold_ms is not None else 0.0

    # --- cars -------------------------------------------------------------------

    def _extra(self, h: float, t_ms: float) -> Tuple[float, float]:
        """(width of the uniform TCP extra, the part timeout traffic removes) on
        one hop at one-way delay h and time t_ms."""
        c = self.c
        # Timeouts per committed slot over the last 2 s (a stall counts as one).
        for recent in (self._recent_to, self._recent_slots):
            while recent and recent[0] < t_ms - 2000.0:
                recent.popleft()
        rate = min(1.0, len(self._recent_to) / max(1, len(self._recent_slots)))
        return 2.0 * c.nagle_frac * min(h, c.nagle_cap_ms), c.flush_ms * rate

    def _hop(self, extra: Tuple[float, float], shared_u: float) -> float:
        """One hop's TCP extra; shared_u is the draw common to a round's voters."""
        common = self.c.nagle_common
        e = extra[0] * (common * shared_u + (1.0 - common) * self._u()) - extra[1]
        return e if e > 0.0 else 0.0

    def _car(self, lane: int, idx: int) -> Tuple[float, float, float]:
        """(creation time, certification time, digests) of lane's idx-th car."""
        c = self.c
        while idx - self._car_base >= len(self._cars):
            tick = self._next_tick_ms
            h = self.phase_at(tick / 1000.0)[1].one_way_ms
            period = car_period_ms(c, h)
            times = tuple(tick + abs(self._n()) * c.lane_sigma * period for _ in range(N_NODES))
            extra = self._extra(h, tick)
            certs = tuple(t + c.cert_hops * h + self._hop(extra, self._u()) for t in times)
            # One digest per 100 ms worker batch since the lane's previous car.
            mean = self._last_period / MAX_BATCH_DELAY_MS
            digests = tuple(float(math.floor(mean + self._u())) for _ in range(N_NODES))
            self._cars.append((times, certs, digests))
            self._last_period = period
            self._next_tick_ms = tick + period * math.exp(c.car_sigma * self._n())
        times, certs, digests = self._cars[idx - self._car_base]
        return times[lane], certs[lane], digests[lane]

    def _trim_cars(self) -> None:
        low = min(self._lane_next)
        while self._car_base < low:
            self._cars.popleft()
            self._car_base += 1

    # --- one view ---------------------------------------------------------------

    def _view_commit(self, prep: float, h: float, extra: Tuple[float, float], withhold: bool,
                     tc: float) -> float:
        """Commit time at the reporting replica of a view whose Prepare went out
        at ``prep`` and was delivered before the TC; inf if the TC starves it."""
        c, u = self.c, self._u
        tail_mean = c.tail_hops * h + c.tail_ms
        n = 3 if withhold else N_NODES
        hop = self._hop
        scale = math.exp(c.view_sigma * self._n())

        def round_trip(t0: float) -> List[float]:
            # Prepare (or Confirm) out and the vote back, per voter.
            base = t0 + 2.0 * c.proc_ms
            out_u, back_u = u(), u()
            out = []
            for _ in range(n):
                v = base + scale * (2.0 * h + hop(extra, out_u) + hop(extra, back_u))
                if u() < c.vote_tail_prob:
                    v -= tail_mean * math.log(1.0 - u())
                out.append(v)
            out.sort()
            return out

        votes = round_trip(prep)
        if withhold:
            confirm = votes[-1] + FAST_PATH_MS
        elif votes[3] <= votes[2] + FAST_PATH_MS:
            return votes[3] + h + hop(extra, u()) + c.proc_ms
        else:
            confirm = votes[2] + FAST_PATH_MS
        # A replica holding the TC refuses the Confirm (core.rs:1374-1384); with
        # votes withheld one honest refusal starves the view, otherwise two
        # (also in a view replica 3 leads, where its own vote is not held).
        confirm_u = u()
        refusals = sum(tc + extra[0] * (u() - 0.5) < confirm + h + hop(extra, confirm_u)
                       for _ in range(n))
        if refusals >= (1 if withhold else 2):
            return math.inf
        votes = round_trip(confirm)
        return votes[2] + h + hop(extra, u()) + c.proc_ms

    # --- the slot pipeline --------------------------------------------------------

    def _advance(self, horizon_ms: float) -> None:
        """Resolve slots until the next timer would start at or after ``horizon_ms``
        (the timeout in effect there is not known yet)."""
        c = self.c
        while True:
            st = self._chain
            if st is None:
                arm = max(self._prev_ticket, self._commits[0])
                if arm >= horizon_ms:
                    return
                # 3 of 4 lanes with a newly certified tip (core.rs enough_coverage).
                certs = sorted(self._car(l, self._lane_next[l])[1] for l in range(N_NODES))
                start = max(certs[N_NODES - 2] + c.slot_proc_ms, arm)
                st = self._chain = [1, start, arm, None, math.inf]
            view, start, timer, first_prep, best = st
            if timer >= horizon_ms:
                return
            T = self.timeout_ms
            ph = self.phase_at(timer / 1000.0)[1]
            h = ph.one_way_ms
            extra = self._extra(h, timer)
            leader = (self._slot + view - 1) % N_NODES
            prep = start + (self._hold(ph, T) if leader == ATTACKER else 0.0)
            deadline = timer + T
            tc = deadline + h + 0.5 * extra[0] + c.proc_ms
            # A Prepare that arrives after the TC is refused, a held one dropped.
            if prep + h <= tc:
                if first_prep is None:
                    first_prep = prep
                # Replica 3 counts its own vote at once when it leads.
                withheld = ph.vote_withhold and leader != ATTACKER
                best = min(best, self._view_commit(prep, h, extra, withheld, tc))
            if best > deadline:
                self._to_t.append(deadline)
                self._recent_to.append(deadline)
            if best <= tc:
                self._finish(best, first_prep, ph, T)
                continue
            # The TC moves the slot on; the next view's leader proposes at once.
            self._chain = [view + 1, tc + c.view_change_hops * min(h, c.nagle_cap_ms) + c.proc_ms, tc, first_prep, best]

    def _finish(self, commit: float, first_prep: float, ph: Phase, timeout_ms: float) -> None:
        c, u = self.c, self._u
        h = ph.one_way_ms
        self._recent_slots.append(commit)
        # Late spurious timeouts: an old slot's late Commit re-arms a later
        # slot's timer 10 s on; it fires unless that slot's own late Commit
        # lands within T. The map reads k = 4 slots apart; three apart fits
        # the measured share (31% of slots at 5+50 and T=750, where four slot
        # intervals are 900 ms, none at T=1000).
        if len(self._preps) == self._preps.maxlen:
            old_prep, old_withheld = self._preps[0]
            if old_withheld and first_prep - old_prep > timeout_ms:
                self._to_t.append(old_prep + VOTE_HOLD_MS + timeout_ms)
        self._preps.append((first_prep, ph.vote_withhold))
        self._commits.append(commit)
        self._prev_ticket = first_prep + h
        self._exec = max(self._exec, commit)
        ex = self._exec
        # Every car the leader had seen when it sent the Prepare commits with it.
        lanes = 0
        period = car_period_ms(c, h)
        for lane in range(N_NODES):
            got = False
            while True:
                created, _, digests = self._car(lane, self._lane_next[lane])
                if created + h > first_prep:
                    break
                got = True
                # The reporting replica (0) sees its own cars at once.
                lat = ex - (created if lane == 0 else created + h)
                self._hdr_t.append(ex)
                self._hdr_lat.append(lat)
                self._hdr_dig.append(digests)
                if self.track_client:
                    clat = (ex - created + MAX_BATCH_DELAY_MS * u()
                            + 2.0 * c.client_car_frac * period * u() + c.client_proc_ms)
                    self.client.setdefault(int(ex // 1000.0), []).append(
                        (clat, digests * c.load_tps * MAX_BATCH_DELAY_MS / 1000.0 / N_NODES))
                self._lane_next[lane] += 1
            lanes += got
        self._trim_cars()
        if lanes:
            # Authors change between lanes in a slot and, three times in four,
            # across the slot boundary.
            self._slot_t.append(ex)
            self._slot_lc.append(lanes - 1 + (1.0 if u() < 0.75 else 0.0))
        self._slot += 1
        self._chain = None

    # --- windows and episodes ---------------------------------------------------

    def _window(self, start_ms: float, end_ms: float) -> _Window:
        t = np.asarray(self._hdr_t)
        sel = (t >= start_ms) & (t < end_ms)
        st = np.asarray(self._slot_t)
        ssel = (st >= start_ms) & (st < end_ms)
        to = np.asarray(self._to_t)
        return _Window(
            start_ms, end_ms,
            times=t[sel],
            latencies=np.asarray(self._hdr_lat)[sel],
            digests=np.asarray(self._hdr_dig)[sel],
            leader_changes=float(np.asarray(self._slot_lc)[ssel].sum()),
            slots=int(ssel.sum()),
            timeouts=int(((to >= start_ms) & (to < end_ms)).sum()),
        )

    def _prune(self, before_ms: float) -> None:
        keep = [i for i, t in enumerate(self._hdr_t) if t >= before_ms]
        self._hdr_t = [self._hdr_t[i] for i in keep]
        self._hdr_lat = [self._hdr_lat[i] for i in keep]
        self._hdr_dig = [self._hdr_dig[i] for i in keep]
        keep = [i for i, t in enumerate(self._slot_t) if t >= before_ms]
        self._slot_t = [self._slot_t[i] for i in keep]
        self._slot_lc = [self._slot_lc[i] for i in keep]
        self._to_t = [t for t in self._to_t if t >= before_ms]

    def run_episode(
        self, choose: Callable[[Dict[str, float]], float]
    ) -> Tuple[Dict[str, float], float, Dict[str, float], int]:
        t0 = self.episode_start_s * 1000.0
        self._advance(t0 + FEATURE_S * 1000.0)
        report = self._window(t0, t0 + FEATURE_S * 1000.0).report(
            self.timeout_ms, self.c.no_progress_scale, self._burst_gap())
        new_timeout = float(choose(report))
        self._advance(t0 + (FEATURE_S + REPLY_S) * 1000.0)
        self.timeout_ms = new_timeout
        self._advance(t0 + EPISODE_S * 1000.0)
        reward = self._window(t0 + (FEATURE_S + REPLY_S + WARMUP_S) * 1000.0, t0 + EPISODE_S * 1000.0)
        phase_idx, _ = self.phase_at(self.episode_start_s + FEATURE_S)
        self.episode_start_s += EPISODE_S
        self._prune(t0 + EPISODE_S * 1000.0)
        return (report, new_timeout,
                reward.report(new_timeout, self.c.no_progress_scale, self._burst_gap()), phase_idx)

    def _burst_gap(self) -> float:
        # Its own stream, so the window-to-window spread leaves the run's draws alone.
        return self.c.burst_gap_ms * float(self._gap_rng.uniform(0.8, 1.25))

    def client_score(self, start_s: float, end_s: float) -> Tuple[float, float]:
        """(R, tx-weighted p50 in ms) on the client between start_s and end_s, as
        scripts/cloudlab/tmp/cal/ab/report.py and phase_judge.py score it: per
        second the tx-weighted median latency, R = tps / 10000 - p50 s per 20 s
        chunk (a chunk with no commits scores p50 as 10 s), averaged."""
        if not self.track_client:
            raise ValueError("client_score needs AutobahnSim(track_client=True)")
        rows = []
        for sec in range(int(math.ceil(start_s)), int(end_s)):
            pairs = self.client.get(sec)
            if not pairs:
                rows.append((0.0, 0.0))
                continue
            pairs.sort()
            lat = np.asarray([p[0] for p in pairs])
            w = np.cumsum([p[1] for p in pairs])
            rows.append((float(w[-1]), float(lat[np.searchsorted(w, 0.5 * w[-1])])))
        rewards = []
        for k in range(0, len(rows) - 20 + 1, 20):
            chunk = rows[k:k + 20]
            s = sum(r[0] for r in chunk)
            q = sum(r[0] * r[1] for r in chunk) / s if s else CLIENT_STALL_P50_MS
            rewards.append(s / len(chunk) / 10000.0 - q / 1000.0)
        succ = sum(r[0] for r in rows)
        p50 = sum(r[0] * r[1] for r in rows) / succ if succ else float("nan")
        return (float(np.mean(rewards)) if rewards else float("nan")), p50


def sample_constants(rng: np.random.Generator) -> SimConstants:
    return SimConstants(
        car_hops=float(rng.uniform(2.0, 3.3)),
        car_proc_ms=float(rng.uniform(30.0, 70.0)),
        car_sigma=float(rng.uniform(0.02, 0.1)),
        lane_sigma=float(rng.uniform(0.05, 0.3)),
        cert_hops=float(rng.uniform(1.2, 2.2)),
        slot_proc_ms=float(rng.uniform(2.0, 15.0)),
        nagle_frac=float(rng.uniform(0.3, 0.65)),
        nagle_cap_ms=float(rng.uniform(150.0, 300.0)),
        nagle_common=float(rng.uniform(0.45, 0.85)),
        flush_ms=float(rng.uniform(5.0, 30.0)),
        view_change_hops=float(rng.uniform(0.3, 0.9)),
        view_sigma=float(rng.uniform(0.15, 0.4)),
        proc_ms=float(rng.uniform(0.4, 2.0)),
        vote_tail_prob=float(rng.uniform(0.0, 0.03)),
        tail_hops=float(rng.uniform(2.0, 6.0)),
        tail_ms=float(rng.uniform(0.0, 50.0)),
        load_tps=float(rng.uniform(4000.0, 16000.0)),
        no_progress_scale=float(rng.uniform(0.15, 0.5)),
        burst_gap_ms=float(math.exp(rng.uniform(math.log(0.012), math.log(0.07)))),
    )


# train --no-proposal-faults: a prior that never saw a slow leader (vote
# withholding still on), so its pretraining covers the network-delay regimes
# only. The "blind pretrained" arm of the ablation - see main.py --prior-path.
NO_PROPOSAL_FAULTS = False


def sample_phase(rng: np.random.Generator, start_s: float) -> Phase:
    one_way = float(math.exp(rng.uniform(math.log(2.0), math.log(250.0))))
    withhold = bool(rng.random() < 0.6)
    kind = rng.random()
    offset = hold = None
    if kind < 0.35:
        pass
    elif kind < 0.8:
        offset = float(math.exp(rng.uniform(math.log(10.0), math.log(2000.0))))
    else:
        hold = float(math.exp(rng.uniform(math.log(10.0), math.log(2000.0))))
    if NO_PROPOSAL_FAULTS:
        offset = hold = None
    return Phase(start_s=start_s, one_way_ms=one_way, vote_withhold=withhold,
                 hold_offset_ms=offset, hold_ms=hold)


def sample_schedule(rng: np.random.Generator, total_s: float) -> List[Phase]:
    phases = [sample_phase(rng, 0.0)]
    t = float(rng.uniform(40.0, 400.0))
    while t < total_s:
        phases.append(sample_phase(rng, t))
        t += float(rng.uniform(40.0, 400.0))
    return phases


# config/autobahn/failure_spec.xml: warm-up 10 s, then (atTime, one-way, hold
# offset) per 300 s phase, None for no slow leader, replica 3 withholding its
# votes in every phase, and the oracle's timeout per phase
# (learning_agent/config/schedule_autobahn.csv). Used only to evaluate
# policies in simulation, never to train.
SPEC_PHASES: Tuple[Tuple[float, float, Optional[float]], ...] = (
    (0.0, 5.0, None),
    (300.0, 5.0, 190.0),
    (600.0, 55.0, None),
    (900.0, 55.0, 690.0),
    (1200.0, 155.0, None),
    (1500.0, 5.0, None),
)
SPEC_ORACLE_MS: Tuple[float, ...] = (250.0, 250.0, 750.0, 750.0, 1250.0, 250.0)
SPEC_RATE_TPS = 10000.0
SPEC_WARMUP_S = 10.0
SPEC_EPISODES = int((SPEC_WARMUP_S + SPEC_PHASES[-1][0] + 300.0 + 10.0) // EPISODE_S)
# Measured client R per phase, oracle5_20261005_1626 and oracle1_h2_20261005_1743.
SPEC_ORACLE_CLIENT_R: Tuple[Tuple[float, ...], ...] = (
    (0.785, 0.769, 0.161, 0.153, -0.876, 0.784),
    (0.784, 0.774, 0.171, 0.165, -0.870, 0.788),
)


def spec_schedule() -> List[Phase]:
    phases = [Phase(0.0, one_way_ms=5.0)]
    for at, one_way, offset in SPEC_PHASES:
        phases.append(Phase(SPEC_WARMUP_S + at, one_way_ms=one_way, vote_withhold=True,
                            hold_offset_ms=offset))
    return phases


# --- Agent loop, data generation, prior training, evaluation -----------------


def feature_vector(report: Dict[str, float]) -> np.ndarray:
    from models import AUTOBAHN_REPORT_FEATURE_FIELD_NAMES

    return np.asarray(
        [report[name] for name in AUTOBAHN_REPORT_FEATURE_FIELD_NAMES], dtype=float
    )


_INT_FIELDS = (
    "total_transactions", "total_consensus_instances", "leader_change_count",
    "timeout_ms", "view_change_count", "no_progress_view_change_count",
)


def episode_reward(report: Dict[str, float]) -> float:
    """Reward of one reward window under the process-wide reward design, the
    same calculator main.reward_engineering uses."""
    from reward_utils import calculate_autobahn_reward

    return calculate_autobahn_reward(
        **{
            name: int(round(value)) if name in _INT_FIELDS else float(value)
            for name, value in report.items()
        }
    )


def run_agent(
    sim: AutobahnSim,
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


def training_sim(rng: np.random.Generator, seed: int) -> Tuple[AutobahnSim, int]:
    """One randomized training run: (simulator, episodes)."""
    lower_ms, upper_ms = CONTEXT_SPECS[agent_pb2.PROTOCOL_AUTOBAHN].bounds_ms
    episodes = int(rng.integers(40, 120))
    # A quarter of the runs start on the parameters.json default, as a real
    # run does before its first decision.
    if rng.random() < 0.25:
        initial = DEFAULT_TIMEOUT_MS
    else:
        initial = float(math.exp(rng.uniform(math.log(lower_ms), math.log(upper_ms))))
    sim = AutobahnSim(
        sample_constants(rng),
        sample_schedule(rng, episodes * EPISODE_S),
        seed=seed,
        start_offset_s=float(rng.uniform(0.0, 20.0)),
        initial_timeout_ms=initial,
    )
    return sim, episodes


def spec_sim(seed: int, constants: Optional[SimConstants] = None,
             track_client: bool = False) -> Tuple[AutobahnSim, int]:
    """The spec schedule at SPEC_RATE_TPS: (simulator, episodes)."""
    c = replace(constants if constants is not None else SimConstants(), load_tps=SPEC_RATE_TPS)
    sim = AutobahnSim(c, spec_schedule(), seed=seed, start_offset_s=0.2, track_client=track_client)
    return sim, SPEC_EPISODES


def _generate_one(args: Tuple[int, str, int]) -> Tuple[np.ndarray, np.ndarray]:
    from reward_utils import set_reward_design

    seed, design, hold = args
    set_reward_design(design)
    from continuous_bandit import ContinuousTimeoutActionCodec, report_context
    from models import AUTOBAHN_REPORT_FEATURE_FIELD_NAMES

    spec = CONTEXT_SPECS[agent_pb2.PROTOCOL_AUTOBAHN]
    lower_ms, upper_ms = spec.bounds_ms
    rng = np.random.default_rng(seed)
    codec = ContinuousTimeoutActionCodec(agent_pb2.PROTOCOL_AUTOBAHN, bounds_ms=spec.bounds_ms)
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
                ctx = report_context(spec, AUTOBAHN_REPORT_FEATURE_FIELD_NAMES, x)
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


def train_prior(
    x: np.ndarray,
    y: np.ndarray,
    out_path: str,
    seed: int,
    design: str,
    hold: int = 1,
    target_floor: float = -math.inf,
    hidden: Tuple[int, ...] = (128, 128, 64),
) -> float:
    from sklearn.neural_network import MLPRegressor

    # Rewards below the floor are fit as the floor, so a collapse is a collapse
    # and the fit spends its error on the range that ranks usable timeouts; the
    # model applies the same floor to the online reward.
    y = np.maximum(y, target_floor)
    x_mean, x_std = x.mean(axis=0), x.std(axis=0) + 1e-9
    y_mean, y_std = float(y.mean()), float(y.std() + 1e-9)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    cut = int(0.9 * len(x))
    tr, va = order[:cut], order[cut:]
    mlp = MLPRegressor(
        hidden_layer_sizes=tuple(hidden),
        activation="relu",
        alpha=1e-5,
        batch_size=512,
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=seed,
    )
    mlp.fit((x[tr] - x_mean) / x_std, (y[tr] - y_mean) / y_std)
    pred = mlp.predict((x[va] - x_mean) / x_std) * y_std + y_mean
    rmse = float(np.sqrt(np.mean((pred - y[va]) ** 2)))
    arrays = {"n_layers": np.asarray(len(mlp.coefs_))}
    for i, (w, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        arrays[f"W{i}"] = w
        arrays[f"b{i}"] = b
    arrays.update(
        x_mean=x_mean,
        x_std=x_std,
        y_mean=np.asarray(y_mean),
        y_std=np.asarray(y_std),
        reward_design=np.asarray(design),
        hold_episodes=np.asarray(int(hold)),
        target_floor=np.asarray(float(target_floor)),
    )
    np.savez(out_path, **arrays)
    return rmse


def make_policy(kind: str, prior_path: str = DEFAULT_PRIOR_PATH):
    """(predict, update) closures: ``fixed:<ms>`` one timeout, ``candidate``
    SimReturnGPModel as main.serve builds it."""
    from continuous_bandit import ContinuousTimeoutActionCodec, SimReturnGPModel
    from models import AUTOBAHN_REPORT_FEATURE_FIELD_NAMES as names
    from reward_utils import get_reward_design

    protocol = agent_pb2.PROTOCOL_AUTOBAHN
    if kind.startswith("fixed:"):
        value = float(kind.split(":", 1)[1])
        return (lambda x: value), None
    if kind == "candidate":
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
            return float(timeout.autobahn.timeout_delay_milliseconds)

        def update(x: np.ndarray, timeout_ms: float, reward: float) -> None:
            used = agent_pb2.AutobahnTimeout(timeout_delay_milliseconds=int(round(timeout_ms)))
            lower, upper = spec.bounds_ms
            if lower <= used.timeout_delay_milliseconds <= upper:
                model.update_model(x, codec.encode(used), reward)

        return predict, update
    raise ValueError(f"Unknown policy {kind!r}; expected oracle, fixed:<ms> or candidate")


def evaluate_spec(kind: str, constants: SimConstants, seed: int, **kwargs) -> np.ndarray:
    """Per-phase (mean reward after the switch episode, mean report p50, median
    timeout, mean reward over the whole phase including its switch episode,
    client R as phase_judge.py scores it) of one policy on the spec."""
    sim, episodes = spec_sim(seed, constants, track_client=True)
    if kind == "oracle":
        def predict(x: np.ndarray) -> float:
            idx, _ = sim.phase_at(sim.episode_start_s + FEATURE_S + REPLY_S)
            return SPEC_ORACLE_MS[max(0, idx - 1)]
        update = None
    else:
        predict, update = make_policy(kind, **kwargs)
    rows = run_agent(sim, episodes, predict, update)
    end_s = sim.episode_start_s
    out = np.zeros((len(SPEC_PHASES), 5))
    for ph in range(len(SPEC_PHASES)):
        whole = [r for r in rows if r[0] == ph]
        sel = whole[1:]
        lo = SPEC_WARMUP_S + SPEC_PHASES[ph][0]
        hi = SPEC_WARMUP_S + SPEC_PHASES[ph + 1][0] if ph + 1 < len(SPEC_PHASES) else end_s
        out[ph] = (
            np.mean([episode_reward(r[2]) for r in sel]),
            np.mean([r[2]["p50_consensus_latency_ms"] for r in sel]),
            np.median([r[1] for r in sel]),
            np.mean([episode_reward(r[2]) for r in whole]),
            sim.client_score(lo + 20.0, hi)[0],
        )
    return out


def pretrain_adapter():
    """This simulator as pretrain.py drives it: training runs as train draws
    them, spec runs as spec_sim builds them, and the oracle evaluate_spec
    plays: the rung of the phase in force when the timeout is applied."""
    from functools import partial

    from pretrain import SimAdapter, timeout_message

    def to_timeout(timeout_ms: float) -> agent_pb2.Timeout:
        return agent_pb2.Timeout(autobahn=agent_pb2.AutobahnTimeout(
            timeout_delay_milliseconds=int(round(timeout_ms))))

    def from_timeout(timeout: agent_pb2.Timeout) -> float:
        return float(timeout_message(timeout, "autobahn").timeout_delay_milliseconds)

    def oracle(sim: AutobahnSim) -> Callable[[np.ndarray], float]:
        def predict(x: np.ndarray) -> float:
            idx, _ = sim.phase_at(sim.episode_start_s + FEATURE_S + REPLY_S)
            return SPEC_ORACLE_MS[max(0, idx - 1)]

        return predict

    return SimAdapter(
        protocol=agent_pb2.PROTOCOL_AUTOBAHN,
        simulator="autobahn_sim",
        training_run=lambda seed: training_sim(np.random.default_rng(seed), seed),
        # Client tracking draws from the run's stream, so evaluate_spec's runs
        # are reproduced only with it on.
        spec_run=lambda seed: spec_sim(seed, track_client=True),
        run_agent=run_agent,
        run_training=partial(run_agent, spec_warmup=False),
        to_timeout=to_timeout,
        from_timeout=from_timeout,
        row_phase=lambda row: row[0],
        row_reward=lambda row: episode_reward(row[2]),
        spec_phase_count=len(SPEC_PHASES),
        oracle=oracle,
    )


# --- check: the measured calibration cells -------------------------------------

# scripts/cloudlab/tmp/cal/pt/map_autobahn.txt section 5 (cloudlab_1, 10,000
# tx/s, 180 s per cell, client R and p50 for t > 60 s). Group key (one-way,
# vote withholding, hold offset X); rows (T, R, client p50 or None).
MEASURED_CELLS: Dict[Tuple[float, bool, Optional[float]], Tuple[Tuple[float, float, Optional[float]], ...]] = {
    (5.0, False, None): ((100, 0.844, 157), (1000, 0.846, 154)),
    (55.0, False, None): ((100, 0.415, 584), (150, 0.429, 571), (200, 0.435, 566),
                          (300, 0.409, 591), (400, 0.387, 613), (1000, 0.374, 627)),
    (155.0, False, None): ((300, -0.428, 1425), (500, -0.379, 1377), (700, -0.434, 1433),
                           (900, -0.480, 1483), (2000, -0.599, 1600)),
    (5.0, True, None): ((100, 0.775, 225), (1000, 0.788, 212)),
    (55.0, True, None): ((200, -0.354, 1356), (300, 0.105, 895), (400, 0.260, 744),
                         (500, 0.236, 763), (1000, 0.160, 841)),
    (155.0, True, None): ((500, -2.109, 3117), (600, -1.679, 2677), (700, -1.168, 2168),
                          (750, -1.155, 2146), (800, -1.035, 2044), (1000, -0.773, 1772),
                          (1200, -0.810, 1808), (1250, -0.836, 1836), (1500, -0.928, 1926),
                          (2000, -1.081, 2079)),
    (5.0, True, 190.0): ((250, 0.778, 222), (750, 0.575, 428), (1250, 0.313, 683)),
    (5.0, True, 240.0): ((300, 0.780, 220), (800, 0.569, 429), (1300, 0.323, 673)),
    (55.0, True, 690.0): ((250, -0.134, 1136), (750, 0.160, 839), (1250, -0.071, 1073)),
    # Two draws at 300 and 800 (+0.130/+0.120, +0.165/+0.150), averaged.
    (55.0, True, 740.0): ((250, -0.177, 1183), (300, 0.125, None), (800, 0.158, None),
                          (1300, -0.063, 1065)),
    (155.0, True, 1190.0): ((750, -1.047, 2051), (1250, -0.836, 1837)),
    (155.0, True, 1240.0): ((800, -1.007, None), (1300, -0.888, None)),
}
# Measured R closer than this is a tie: either order reproduces it.
TIE_R = 0.015
CHECK_RUN_S = 180.0
CHECK_STEADY_S = 60.0
# Report means at the oracle, both 20261005 oracle runs (map_autobahn.txt
# section 3): per spec phase, (instances, p50, p95, view changes) per 10 s
# feature window.
MEASURED_ORACLE_REPORT = (
    (396, 101, 176, 97), (396, 108, 177, 95), (204, 596, 809, 13),
    (203, 601, 819, 13.4), (86, 1389, 1897, 29), (395, 99, 168, 98),
)


def _check_job(args) -> Tuple[Tuple, float, int, Dict[str, float], float]:
    key, timeout_ms, seed = args
    from reward_utils import set_reward_design

    set_reward_design("reward_tps_latency")
    t0 = time.time()
    h, withhold, offset = key
    # As the cells ran: 10 s healthy warm-up at base, then the fault; the
    # cell's timeout from the start.
    phases = [Phase(0.0), Phase(10.0, one_way_ms=h, vote_withhold=withhold, hold_offset_ms=offset)]
    sim = AutobahnSim(replace(SimConstants(), load_tps=SPEC_RATE_TPS), phases, seed=seed,
                      start_offset_s=0.0, initial_timeout_ms=float(timeout_ms), track_client=True)
    reps = []
    for _ in range(int(CHECK_RUN_S // EPISODE_S)):
        s0 = sim.episode_start_s
        feature, _, reward, _ = sim.run_episode(lambda r: float(timeout_ms))
        if s0 >= CHECK_STEADY_S:
            reps.append(feature)
    r, p50 = sim.client_score(CHECK_STEADY_S, CHECK_RUN_S)
    res = {
        "R": r,
        "clat": p50,
        "p50": float(np.mean([x["p50_consensus_latency_ms"] for x in reps])),
        "inst": float(np.mean([x["total_consensus_instances"] for x in reps])),
        "vc": float(np.mean([x["view_change_count"] for x in reps])),
        "agent": float(np.mean([episode_reward(x) for x in reps])),
    }
    return key, timeout_ms, seed, res, time.time() - t0


def _oracle_job(seed: int):
    from reward_utils import set_reward_design

    set_reward_design("reward_tps_latency")
    sim, episodes = spec_sim(seed)
    rows = []
    for _ in range(episodes):
        s0 = sim.episode_start_s
        feature, _, _, _ = sim.run_episode(
            lambda r: SPEC_ORACLE_MS[max(0, sim.phase_at(sim.episode_start_s + FEATURE_S + REPLY_S)[0] - 1)])
        rows.append((sim.phase_at(s0)[0] - 1, feature))
    out = []
    for ph in range(len(SPEC_PHASES)):
        sel = [f for p, f in rows if p == ph][2:]
        out.append([float(np.mean([f[k] for f in sel])) for k in (
            "total_consensus_instances", "p50_consensus_latency_ms", "p95_consensus_latency_ms",
            "view_change_count", "no_progress_view_change_count", "avg_batch_size",
            "leader_change_count", "p95_inter_commit_gap_ms")])
    return out


def run_check(seeds: int, workers: int) -> str:
    from multiprocessing import Pool

    jobs = [(key, t, 1000 + s) for key, rows in MEASURED_CELLS.items() for t, _, _ in rows
            for s in range(seeds)]
    with Pool(workers) as pool:
        results = pool.map(_check_job, jobs, chunksize=1)
        oracle = pool.map(_oracle_job, [1000 + s for s in range(seeds)])
    agg: Dict[Tuple, List[Dict[str, float]]] = {}
    secs = []
    for key, t, _, res, dt in results:
        agg.setdefault((key, t), []).append(res)
        secs.append(dt)
    lines = [f"autobahn_sim check seeds={seeds} rate={SPEC_RATE_TPS:.0f} host constants {SimConstants()}",
             f"wall s per simulated run: mean {np.mean(secs):.2f} (max {np.max(secs):.2f}) for "
             f"{CHECK_RUN_S:.0f} s simulated",
             "group (h, withhold, X)   T     sim: R  (sd)  client p50 | measured R  p50 | agent: p50  inst/10s vc/10s reward"]
    matched, pairs_ok, pairs_all, level = 0, 0, 0, []
    for key, rows in MEASURED_CELLS.items():
        sim_r = {}
        for t, r_meas, p50_meas in rows:
            vals = agg[(key, t)]
            m = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
            sd = float(np.std([v["R"] for v in vals]))
            sim_r[t] = m["R"]
            level.append(m["R"] - r_meas)
            ms = f"{p50_meas:5.0f}" if p50_meas is not None else "    -"
            lines.append(f"({key[0]:3.0f}, {'vw' if key[1] else '--'}, {str(key[2]):>6})  {t:5.0f}  "
                         f"{m['R']:+7.3f} ({sd:.3f}) {m['clat']:7.0f} | {r_meas:+7.3f} {ms} | "
                         f"{m['p50']:7.0f} {m['inst']:6.0f} {m['vc']:6.1f} {m['agent']:+7.3f}")
        # Pairwise order: every pair the measurement separates by TIE_R or more.
        bad = []
        n_pairs = 0
        for i, (ta, ra, _) in enumerate(rows):
            for tb, rb, _ in rows[i + 1:]:
                if abs(ra - rb) < TIE_R:
                    continue
                n_pairs += 1
                if (ra > rb) != (sim_r[ta] > sim_r[tb]):
                    bad.append(f"{ta:.0f}/{tb:.0f}")
        pairs_all += n_pairs
        pairs_ok += n_pairs - len(bad)
        matched += not bad
        want = [t for t, _, _ in sorted(rows, key=lambda r: -r[1])]
        got = sorted(sim_r, key=lambda t: -sim_r[t])
        lines.append(f"    measured order {want}; sim {got}; "
                     f"{'OK' if not bad else 'DIFF ' + ','.join(bad)} ({n_pairs - len(bad)}/{n_pairs} pairs)")
    lines.append(f"rankings reproduced in {matched}/{len(MEASURED_CELLS)} groups, "
                 f"{pairs_ok}/{pairs_all} separated pairs; level error mean {np.mean(level):+.3f} "
                 f"mean abs {np.mean(np.abs(level)):.3f} max abs {np.max(np.abs(level)):.3f}")
    lines.append("oracle on the spec, per 10 s feature window (sim mean of seeds, skipping two "
                 "episodes per phase | measured): inst p50 p95 vc | np batch lc p95gap")
    o = np.mean(np.asarray(oracle), axis=0)
    for ph in range(len(SPEC_PHASES)):
        mi, mp, mq, mv = MEASURED_ORACLE_REPORT[ph]
        lines.append(f"  p{ph} T={SPEC_ORACLE_MS[ph]:4.0f}  {o[ph,0]:5.0f} {o[ph,1]:6.0f} {o[ph,2]:6.0f} "
                     f"{o[ph,3]:5.1f} | {mi:5.0f} {mp:6.0f} {mq:6.0f} {mv:5.1f} | {o[ph,4]:5.1f} "
                     f"{o[ph,5]:5.2f} {o[ph,6]:5.0f} {o[ph,7]:6.0f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="generate randomized episodes and fit the prior")
    t.add_argument("--runs", type=int, default=4000)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    t.add_argument("--out", default=DEFAULT_PRIOR_PATH)
    t.add_argument("--reward-design", default="reward_tps_latency")
    t.add_argument("--no-proposal-faults", action="store_true",
                   help="train without a slow leader; the prior never sees a "
                        "delayed proposer (vote withholding stays)")
    t.add_argument("--hold-episodes", type=int, default=1)
    t.add_argument("--target-floor", type=float, default=-2.0)
    t.add_argument("--hidden", default="128,128,64", help="MLP hidden layer sizes")
    e = sub.add_parser("eval", help="compare policies on the spec schedule in simulation")
    e.add_argument("--seeds", type=int, default=10)
    e.add_argument("--prior", default=None,
                   help="prior for the candidate; without one only the oracle and the rungs run")
    e.add_argument("--rungs", default="250,750,1250", help="the fixed timeouts to compare")
    e.add_argument("--randomize", action="store_true", help="random constants per seed")
    e.add_argument("--reward-design", default="reward_tps_latency")
    c = sub.add_parser("check", help="the measured calibration cells at the host constants")
    c.add_argument("--seeds", type=int, default=5)
    c.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()
    from reward_utils import set_reward_design

    set_reward_design(args.reward_design if args.cmd != "check" else "reward_tps_latency")
    if args.cmd == "check":
        print(run_check(args.seeds, args.workers))
        return
    if args.cmd == "train":
        global NO_PROPOSAL_FAULTS
        NO_PROPOSAL_FAULTS = bool(args.no_proposal_faults)  # inherited by the forked workers
        t0 = time.time()
        x, y = generate(args.runs, args.seed, args.workers, args.reward_design, args.hold_episodes)
        gen_s = time.time() - t0
        rmse = train_prior(x, y, args.out, args.seed, args.reward_design, args.hold_episodes,
                           args.target_floor, tuple(int(v) for v in args.hidden.split(",")))
        print(f"samples={len(x)} hold_episodes={args.hold_episodes} target_floor={args.target_floor} "
              f"reward_std={y.std():.4f} floored_std={np.maximum(y, args.target_floor).std():.4f} "
              f"holdout_rmse={rmse:.4f} generate_s={gen_s:.1f} -> {args.out}")
        return

    rungs = [float(v) for v in args.rungs.split(",")]
    rng = np.random.default_rng(12345)
    print("measured oracle client R " + " | ".join(
        " ".join(f"p{ph} {v:+.3f}" for ph, v in enumerate(run)) for run in SPEC_ORACLE_CLIENT_R))
    for s in range(args.seeds):
        constants = sample_constants(rng) if args.randomize else SimConstants()
        t0 = time.time()
        res = {"oracle": evaluate_spec("oracle", constants, s)}
        oracle_s = time.time() - t0
        for r in rungs:
            res[f"fixed:{r:g}"] = evaluate_spec(f"fixed:{r:g}", constants, s)
        if args.prior is not None:
            res["candidate"] = evaluate_spec("candidate", constants, s, prior_path=args.prior)
        print(f"seed {s} ({oracle_s:.2f} s per {SPEC_EPISODES}-episode run)")
        for k, v in res.items():
            print(f"  {k:12s} " + " ".join(
                f"p{ph} {v[ph,0]:+.3f}@{v[ph,2]:4.0f} (client {v[ph,4]:+.3f})" for ph in range(len(SPEC_PHASES))))


if __name__ == "__main__":
    main()
