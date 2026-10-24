"""Analytic CppBedrock PBFT simulator for pretraining the continuous model.

Built from the engine's mechanics (CppBedrock/adaptive_timers.MD, the request
watchdog in PendingRequestTimer.cpp and EntityViewChange.cpp, the proposal
delay controller in FailureSpec.cpp, the report accounting in AgentClient.cpp)
and the harness constants in scripts/cloudlab/lib_bedrock.sh, not fit to run
data. Four replicas, a client broadcasting ``load_tps`` requests/s, one leader
proposing on a fixed cadence; one batch is one consensus instance:

- Every 100 ms (``--proposal-interval``) the leader takes the oldest pending
  requests up to ``batch_cap`` (a fitted effective cap; the harness allows 8192
  requests and 4.5 MiB since 2026-10-11) when fewer than 16 batches are in flight
  (``--max-inflight-batches``); a full pipeline skips the tick. Unexecuted
  requests are capped at 32,768 (``--max-pending-requests``); arrivals beyond
  the cap are rejected and lost.
- A batch commits after three one-way hops (pre-prepare, prepare, commit) plus
  engine processing, and executes in sequence order. Its latency is dated from
  the oldest request it carries (``firstSeenUs``), which by the tick has
  waited about one interval, so the healthy latency is the hop model
  ``100 + 3h``. Lognormal processing noise (and netem jitter per hop, 0 in
  every spec since 2026-10-19) puts p95 about 20 ms above p50.
- Request watchdog (``PendingRequestTimer``): the oldest unexecuted request is
  watched, ``deadline = max(since, floor) + T << backoff``, capped at 30 s
  (``kMaxElectionBackoffMs``); ``since`` is its arrival, or for a request that
  arrived before the floor the moment it became the head (CppBedrock ec31a88:
  an inherited backlog owes one timeout per head, not one deadline for all of
  it). Expiry starts a view change: in-flight batches are cancelled and their
  requests re-proposed by the next leader with their arrival times kept, and
  the view installs after two hops (ViewChange, NewView) plus
  ``vc_overhead_ms`` of evidence exchange, NewView verification and body
  fetches. In the engine an expiry only suspects the leader since 5c0a3cf: the
  replica stays in the view and broadcasts a Suspect, and the view change
  starts once f+1 replicas suspect it. This aggregate timer stands for all four
  replicas starving within one tick, where f+1 suspicions mirror the f+1
  ViewChange join, so the leader's 2f+1 point is unchanged; clustered expiries
  can install up to about one hop later, which the refit of ``vc_overhead_ms``
  absorbs (predicted, not measured; no hop term is modelled for it). Installing sets the floor at that instant, since the incoming
  primary is not blamed for the backlog it inherits; when the floored backlog
  has drained the floor moves once, to that moment (6d7ac54), so what queued
  behind it is timed from there. A view that does not install within the
  backed-off wait starts another view change (``onViewChangeWaitExpired``).
  The backoff doubles per view change started and resets when a batch
  carrying a request that arrived after the floor executes, re-flooring the
  deadlines at the reset when it was raised (``EntityRequests.cpp``, 73b9f08
  after 820d5e5 and 6d7ac54; ported 2026-10-19, restart D11). Lowering T
  floors the deadlines at the change (``applyAgentTimeouts``, commit e1a3c9a),
  so a decrease never condemns pending requests retroactively. A rung under
  its level's tail therefore fires on ordinary latency and storms; one landing
  on a deep backlog cascades until the doubling outlasts the drain. The
  engine's relay of a skipped head (``relaySkippedHead``, 2026-10-20: when a
  batch of later arrivals has executed past the watched request, a backup relays
  it to the leader once per view and times it from the relay; until then the
  expiry re-armed from the last progress) cannot fire here: one shared pool executes in
  arrival order, so no executed batch is newer than the watched head.
- Leader proposal delay D (``<proposalDelay>`` with the ``leader`` token):
  whoever leads when the (phase, interval tick) key changes is pinned for that
  tick and holds each proposal D after allocating its sequence, so the held
  batch occupies a window slot. A rung below the delayed latency escapes to a
  clean leader once per tick; a rung above sleeps through it and pays D on
  every request. Above a knee the hold throttles the pipeline: the cost grows
  to ``excess_factor`` x D, and the throttling part is lognormal per batch, so
  the spread widens with it (measured at base 1.0 x D up to 150 ms and 1.7 x D
  at 200 with p95 172 ms over p50; about 1.0 x D at 5+50). Each re-pin drains
  and refloods the pipeline, and the first ``spike_batches`` proposals after
  it carry an extra ``spike_ms``, which is what lets a rung just above the
  delayed latency escape once per tick.

Episodes follow the wall-clock learning windows in lib_bedrock.sh: 7 s
features, 2 s reply wait (applied at the deadline), 4 s warm-up, 7 s reward.
Reports carry the 16 PbftReport fields the models read, as
``AgentClient::MetricsWindow`` fills them on replica 0: one latency and
batch-size sample per executed batch, nearest-rank percentiles, inter-commit
gaps between consecutive executions in the window, view changes as
started (no-progress when nothing committed since the last), leader and
regency changes as views installed, and the election timeout in effect.

Training runs follow the shared pretraining recipe (sim_common.py; Kevin,
2026-10-19, restart D11). Constants the mechanics do not fix are drawn per
run: the fitted ones (processing, the byte cap, the delayed-leader shape, and
since D11 the view-change overhead and its spread) 0.5-2 x their installed
values, the batching wait 0.5-1 x its cited 1 (an oldest request cannot
predate the interval); the load 0.5-1.5 x N (20,000), the base delay 2-10 ms,
no netem jitter, a start-up backlog in 30% of the runs. Faults start at
40-400 s with 40-400 s gaps and keep the PBFT mix (none, hold, extra delay,
both: 0.2 / 0.35 / 0.35 / 0.1) with the shared magnitudes, an extra link delay
log-uniform 10-250 ms and a leader hold log-uniform 50-1500 ms, re-pinned every
0.5-2 x the spec's 60 s. The initial timeout is the opening rung in a quarter
of the runs, else log-uniform over the action range. The prior learned from it
has to hold across all of them.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

import agent_pb2
from continuous_bandit import CONTEXT_SPECS
from continuous_bandit import opening_action_ms, train_gp_prior
from prior_windows import PriorWindows, check_prior_windows
from reward_utils import DEFAULT_REWARD_DESIGN, reward_normalizer_tps
from sim_common import schedule_rows, scheduled_action
from sim_common import (
    JITTER_RATIO,
    around,
    draw_base_delay_ms,
    draw_hold_ms,
    draw_initial_backlog,
    draw_initial_timeout_ms,
    draw_link_extra_ms,
    draw_load_tps,
    fault_start_times,
)

PROPOSAL_INTERVAL_MS = 100.0
MAX_INFLIGHT_BATCHES = 16
MAX_PENDING_REQUESTS = 32768
BATCH_MAX_REQUESTS = 8192
MAX_ELECTION_BACKOFF_MS = 30000.0
# config.pbft.yaml timers.view_change_ms, the engine's own default; every run of the
# restart overrides it with the middle rung (--initial-timeout, D10b), as spec_run does.
DEFAULT_ELECTION_TIMEOUT_MS = 8000.0
FEATURE_S = 7.0
REPLY_S = 2.0
WARMUP_S = 4.0
REWARD_S = 7.0
EPISODE_S = FEATURE_S + REPLY_S + WARMUP_S + REWARD_S
SIM_WINDOWS = PriorWindows.from_seconds(FEATURE_S, REPLY_S, WARMUP_S, REWARD_S)

DEFAULT_PRIOR_PATH = CONTEXT_SPECS[agent_pb2.PROTOCOL_PBFT].prior_path


@dataclass
# Fitted defaults: blind best-by-loss candidate i=1434 (loss 0.046) of s3fit_pbft_blind_s9.jsonl, restart S3 blind refit on restart-cal9 (seeds 7,8,9,10,11 x 1500; prefix-state ranking check PASS), 2026-10-22.
class SimConstants:
    """Defaults are the calibration setup of adaptive_timers.MD (5 ms base,
    20,000 req/s), used only to pre-screen policies; with them the healthy p50
    comes out 168 / 318 / 618 at base, 5+50, 5+150 against the measured
    168 / 335 / 630. Training draws every field from sample_constants."""

    base_delay_ms: float = 5.0
    # netem jitter as a fraction of each hop; the PBFT spec sets 0.
    jitter_ratio: float = 0.0
    # Engine time per batch across the three hops (admission, digest,
    # signatures, execution) and its lognormal spread; p95 sits ~20 ms over p50.
    proc_ms: float = 21.757308
    proc_sigma: float = 0.161796
    proc_per_tx_us: float = 1.818326
    load_tps: float = 20000.0
    # Fitted effective requests per batch (sim_fit FREE); the harness limits are 8192
    # requests and 4.5 MiB (8192 x 576 B with 512 B requests) since 2026-10-11, 512 KiB before.
    batch_cap: int = 7821
    # Age of a batch's oldest request at the tick, in proposal intervals: 1
    # when the client's lanes spread arrivals over the interval.
    batching_wait: float = 1.0
    # View change beyond its two hops: evidence exchange, NewView
    # verification, body fetches; one maintenance interval by default. Fitted
    # on the blind cells from 2026-10-19 (restart D11, sim_fit.py FREE); these
    # defaults stand until that refit is installed.
    vc_overhead_ms: float = 161.468719
    vc_sigma: float = 0.395047
    knee_ms: float = 163.403758
    excess_factor: float = 1.041471
    excess_sigma: float = 0.789366
    spike_ms: float = 57.055355
    spike_batches: int = 3
    # Requests already queued when the run starts: the ramp into an empty
    # pipeline overshoots pending, and any timeout elects while it drains.
    initial_backlog: int = 0


@dataclass
class Phase:
    start_s: float
    extra_delay_ms: float = 0.0
    proposal_delay_ms: float = 0.0
    # <interval>: the leader window is re-pinned once per tick of it.
    interval_s: float = 60.0


@dataclass
class _Batch:
    t_exec: float
    oldest_ms: float
    size: float
    chunks: List[List[float]]
    # Phase clocks as the engine reports them (AgentClient.cpp): the hold and
    # the pre-prepare transfer precede them, so a held leader inflates the
    # consensus latency and none of the phases. pre: pre-prepare arrival to the
    # prepare quorum (one hop); prep: prepare quorum to the commit quorum (one
    # hop); commit: commit quorum to execution (processing and the queue).
    t_arrival: float = 0.0
    pre_ms: float = 0.0
    prep_ms: float = 0.0


@dataclass
class _Window:
    start_s: float
    end_s: float
    txs: float = 0.0
    instances: int = 0
    lat: List[float] = field(default_factory=list)
    batch: List[float] = field(default_factory=list)
    gap: List[float] = field(default_factory=list)
    pre: List[float] = field(default_factory=list)
    prep: List[float] = field(default_factory=list)
    com: List[float] = field(default_factory=list)
    # Gaps reset at every window (AgentClient.cpp): n commits give n-1 gaps.
    last_exec_ms: Optional[float] = None
    view_changes: int = 0
    no_progress: int = 0
    new_views: int = 0

    def covers(self, t_ms: float) -> bool:
        return self.start_s * 1000.0 <= t_ms < self.end_s * 1000.0

    def report(self, timeout_ms: float) -> Dict[str, float]:
        def avg(v):
            return float(np.mean(v)) if v else 0.0

        def pct(v, q):
            # Nearest rank, sorted[ceil(p n) - 1], as AgentClient.cpp percentile.
            if not v:
                return 0.0
            ordered = sorted(v)
            return float(ordered[max(0, math.ceil(q / 100.0 * len(ordered)) - 1)])

        elapsed = self.end_s - self.start_s
        txs = float(round(self.txs))
        # main._fill_empty_window_latency: the agent gives a window in which
        # nothing committed its own length as its latency (the report gives 0).
        empty_ms = elapsed * 1000.0
        return {
            "total_transactions": txs,
            "total_consensus_instances": float(self.instances),
            "avg_consensus_latency_ms": avg(self.lat) if self.lat else empty_ms,
            "p50_consensus_latency_ms": pct(self.lat, 50) if self.lat else empty_ms,
            "p95_consensus_latency_ms": pct(self.lat, 95) if self.lat else empty_ms,
            "throughput_tps": txs / elapsed if elapsed > 0 else 0.0,
            "avg_batch_size": avg(self.batch),
            "p95_batch_size": pct(self.batch, 95),
            "leader_change_count": float(self.new_views),
            "regency_change_count": float(self.new_views),
            "timeout_ms": float(int(round(timeout_ms))),
            "avg_inter_commit_gap_ms": avg(self.gap),
            "p50_inter_commit_gap_ms": pct(self.gap, 50),
            "p95_inter_commit_gap_ms": pct(self.gap, 95),
            "view_change_count": float(self.view_changes),
            "no_progress_view_change_count": float(self.no_progress),
            # Window means; 0 when nothing committed, as the replicas report.
            "pre_prepare_latency_ms": avg(self.pre),
            "prepare_latency_ms": avg(self.prep),
            "commit_latency_ms": avg(self.com),
        }


class PbftSim:
    """Event-driven simulation (ticks, executions, deadlines, view changes)
    driven one episode at a time."""

    def __init__(
        self,
        constants: SimConstants,
        phases: Sequence[Phase],
        seed: int,
        start_offset_s: float = 0.0,
        initial_timeout_ms: float = DEFAULT_ELECTION_TIMEOUT_MS,
    ):
        self.c = constants
        self.phases = sorted(phases, key=lambda p: p.start_s)
        self.rng = np.random.default_rng(seed)
        self.t_ms = 0.0
        # (time ms, no commit since the previous election) of every election, for the
        # reports' counts over the last episode (_count_view_changes).
        self._vc_log: List[Tuple[float, bool]] = []
        self.view = 0
        self.timeout_ms = float(initial_timeout_ms)
        self.backoff = 0
        self.floor_ms = -math.inf
        # PendingRequestTimer's extendFloor_: set by every floor, cleared when the
        # floor moves once to the drain of its backlog (_select_watch).
        self.extend_floor = False
        # The watched request (the oldest unexecuted one): its arrival and the
        # instant its deadline is dated from; None when nothing is pending.
        self.watch_arrival: Optional[float] = None
        self.watch_since = 0.0
        self.in_view_change = False
        self.t_install_ms = math.inf
        self.t_wait_ms = math.inf
        self.commits_since_vc = 0
        self.next_tick_ms = PROPOSAL_INTERVAL_MS
        # Requests as [oldest arrival, count] chunks, oldest first; batches in
        # flight hold their chunks until execution.
        self.pending: Deque[List[float]] = deque()
        self.pending_n = 0.0
        self.inflight: Deque[_Batch] = deque()
        self.inflight_n = 0.0
        self.last_exec_ms: Optional[float] = None
        self.last_arrival_ms = 0.0
        self.pin_key: Optional[Tuple[int, int]] = None
        self.pinned_view: Optional[int] = None
        self.spike_left = 0
        if constants.initial_backlog > 0:
            self.pending.append([0.0, float(constants.initial_backlog)])
            self.pending_n = float(constants.initial_backlog)
            self._select_watch(0.0)
        self.episode_start_s = float(start_offset_s)
        self._advance_to(start_offset_s * 1000.0, None)

    def phase_at(self, t_s: float) -> Tuple[int, Phase]:
        idx = 0
        for i, p in enumerate(self.phases):
            if t_s >= p.start_s:
                idx = i
        return idx, self.phases[idx]

    def _unexecuted(self) -> float:
        return self.pending_n + self.inflight_n

    def _arrive(self, now_ms: float) -> None:
        dt = now_ms - self.last_arrival_ms
        if dt <= 0.0:
            return
        room = max(0.0, MAX_PENDING_REQUESTS - self._unexecuted())
        admitted = min(self.c.load_tps * dt / 1000.0, room)
        if admitted > 0.0:
            # Arrivals spread over (last, now]; the oldest dates the chunk.
            oldest = self.last_arrival_ms + (1.0 - self.c.batching_wait) * dt
            self.pending.append([oldest, admitted])
            self.pending_n += admitted
            if self.watch_arrival is None:
                self._select_watch(now_ms)  # track(): an empty timer watches the newcomer
        self.last_arrival_ms = now_ms

    def _oldest_arrival(self) -> Optional[float]:
        if self.inflight:
            return self.inflight[0].oldest_ms
        if self.pending:
            return self.pending[0][0]
        return None

    def _backoff_ms(self) -> float:
        base = max(1.0, self.timeout_ms)
        return min(base * 2.0 ** self.backoff, max(base, MAX_ELECTION_BACKOFF_MS))

    def _deadline(self) -> float:
        return max(self.watch_since, self.floor_ms) + self._backoff_ms()

    def _select_watch(self, now_ms: float) -> None:
        """PendingRequestTimer::selectWatchLocked: watch the oldest unexecuted
        request. Every floor is set at an event, where _arrive has just closed
        a chunk, so no chunk straddles a floor and a chunk's oldest stamp says
        on which side of it all its requests arrived."""
        previous_floored = self.watch_arrival is not None and self.watch_arrival < self.floor_ms
        self.watch_arrival = self._oldest_arrival()
        if self.watch_arrival is None:
            return
        # What arrived while the floored backlog drained waited behind it, not
        # behind the primary: when it clears, the floor moves once, to now.
        if self.extend_floor and previous_floored and self.watch_arrival >= self.floor_ms:
            self.floor_ms = now_ms
            self.extend_floor = False
        # An inherited request is timed from when it becomes the head, the rest
        # from arrival.
        if self.watch_arrival < self.floor_ms:
            self.watch_since = max(self.floor_ms, now_ms)
        else:
            self.watch_since = self.watch_arrival

    def _set_floor(self, t_ms: float) -> None:
        """PendingRequestTimer::setFloor: no deadline before t_ms."""
        self.floor_ms = t_ms
        self.extend_floor = True

    def _advance_to(self, end_ms: float, windows: Optional[Sequence[_Window]]) -> None:
        while True:
            t_exec = self.inflight[0].t_exec if self.inflight else math.inf
            t_next = min(self.next_tick_ms, t_exec, self.t_install_ms, self.t_wait_ms)
            if not self.in_view_change:
                if self.watch_arrival is not None:
                    deadline = self._deadline()
                    if deadline < t_next:
                        # A deadline already past fires now. The engine's
                        # skipped-head relay never applies (module docstring).
                        t_fire = max(deadline, self.t_ms)
                        if t_fire > end_ms:
                            break
                        self._arrive(t_fire)
                        self.t_ms = t_fire
                        self._start_view_change(windows)
                        continue
            if t_next > end_ms:
                break
            self._arrive(t_next)
            self.t_ms = t_next
            if t_next == t_exec:
                self._execute(windows)
            elif t_next == self.t_wait_ms:
                self._view_change_wait_expired(windows)
            elif t_next == self.t_install_ms:
                self._install(windows)
            else:
                self._tick()
        self._arrive(end_ms)
        self.t_ms = end_ms

    def _tick(self) -> None:
        t = self.t_ms
        self.next_tick_ms += PROPOSAL_INTERVAL_MS
        idx, ph = self.phase_at(t / 1000.0)
        self._update_pin(idx, ph, t)
        # The tick still admits ingress during a view change; only the leader's
        # proposal waits for the new view.
        if self.in_view_change or not self.pending or len(self.inflight) >= MAX_INFLIGHT_BATCHES:
            return
        self._propose(ph, t)

    def _update_pin(self, idx: int, ph: Phase, t: float) -> None:
        if ph.proposal_delay_ms <= 0.0:
            self.pin_key = None
            self.pinned_view = None
            return
        interval_ms = ph.interval_s * 1000.0
        tick = int((t - ph.start_s * 1000.0) // interval_ms) if interval_ms > 0.0 else 0
        key = (idx, tick)
        if key != self.pin_key:
            # pinLeaderWindow: the leader at the key change is the victim for
            # the whole tick; a leader elected later is clean until the next.
            self.pin_key = key
            self.pinned_view = self.view
            self.spike_left = self.c.spike_batches

    def _propose(self, ph: Phase, t: float) -> None:
        c = self.c
        u = self.rng.uniform
        chunks: List[List[float]] = []
        size = 0.0
        while self.pending and size < c.batch_cap:
            arrival, n = self.pending[0]
            take = min(n, c.batch_cap - size)
            chunks.append([arrival, take])
            size += take
            if take >= n:
                self.pending.popleft()
            else:
                self.pending[0][1] = n - take
        self.pending_n -= size
        hold = 0.0
        if ph.proposal_delay_ms > 0.0 and self.pinned_view == self.view:
            d_ms = ph.proposal_delay_ms
            hold = d_ms
            if d_ms > c.knee_ms:
                hold += d_ms * (c.excess_factor - 1.0) * float(self.rng.lognormal(0.0, c.excess_sigma))
            if self.spike_left > 0:
                hold += c.spike_ms * float(u(0.5, 1.5))
                self.spike_left -= 1
        d = c.base_delay_ms + ph.extra_delay_ms
        jr = c.jitter_ratio
        # The pre-prepare is many packets and lands with its last one, so its
        # hop sits in the upper half of the jitter band; votes are one packet.
        h1 = d * (1.0 + jr * float(u(0.0, 1.0)))
        h2 = d * (1.0 + jr * float(u(-1.0, 1.0)))
        h3 = d * (1.0 + jr * float(u(-1.0, 1.0)))
        hops = h1 + h2 + h3
        proc = c.proc_ms * float(self.rng.lognormal(0.0, c.proc_sigma)) + c.proc_per_tx_us * size / 1000.0
        ready = t + hold + hops + proc
        if self.inflight:
            prev = self.inflight[-1].t_exec
        else:
            prev = self.last_exec_ms if self.last_exec_ms is not None else -math.inf
        self.inflight.append(_Batch(max(ready, prev), chunks[0][0], size, chunks,
                                    t_arrival=t + hold + h1, pre_ms=h2, prep_ms=h3))
        self.inflight_n += size

    def _execute(self, windows: Optional[Sequence[_Window]]) -> None:
        b = self.inflight.popleft()
        self.inflight_n -= b.size
        t = b.t_exec
        if windows is not None:
            for w in windows:
                if w.covers(t):
                    w.txs += b.size
                    w.instances += 1
                    w.lat.append(t - b.oldest_ms)
                    w.batch.append(b.size)
                    w.pre.append(b.pre_ms)
                    w.prep.append(b.prep_ms)
                    w.com.append(max(0.0, t - b.t_arrival - b.pre_ms - b.prep_ms))
                    if w.last_exec_ms is not None:
                        w.gap.append(t - w.last_exec_ms)
                    w.last_exec_ms = t
        # complete(): the batch at the head of the order carries the watched
        # request, so the watch moves on.
        self._select_watch(t)
        # PBFT restores the timeout once a request executes in the new view: any
        # request that arrived after the floor (CppBedrock 73b9f08); what is
        # still pending gets a fresh timeout per head when the backoff was raised.
        # No batch executes during a view change (its instances are cancelled).
        if b.chunks[-1][0] >= self.floor_ms:
            if self.backoff:
                self._set_floor(t)
            self.backoff = 0
        self.commits_since_vc += 1
        self.last_exec_ms = t

    def _record_view_change(self, windows: Optional[Sequence[_Window]]) -> None:
        # Counted at the report over the last episode (the elections of the episode before the window's end, as AgentClient / learning.go count them (2026-10-19, restart D14)).
        self._vc_log.append((self.t_ms, self.commits_since_vc == 0))
        self.commits_since_vc = 0
        self.backoff = min(self.backoff + 1, 30)
        self.view += 1

    def _view_change_duration(self) -> float:
        _, ph = self.phase_at(self.t_ms / 1000.0)
        d = self.c.base_delay_ms + ph.extra_delay_ms
        return 2.0 * d * (1.0 + self.c.jitter_ratio * float(self.rng.uniform(0.0, 1.0))) + (
            self.c.vc_overhead_ms * float(self.rng.lognormal(0.0, self.c.vc_sigma))
        )

    def _start_view_change(self, windows: Optional[Sequence[_Window]]) -> None:
        self._record_view_change(windows)
        # Cancelled instances keep their requests and arrival times for the
        # next leader to re-propose.
        while self.inflight:
            for chunk in reversed(self.inflight.pop().chunks):
                self.pending.appendleft(chunk)
        self.pending_n += self.inflight_n
        self.inflight_n = 0.0
        self.in_view_change = True
        self.t_install_ms = self.t_ms + self._view_change_duration()
        self.t_wait_ms = self.t_ms + self._backoff_ms()

    def _view_change_wait_expired(self, windows: Optional[Sequence[_Window]]) -> None:
        # NewView deadline expired: the next view is tried with a doubled wait.
        self._record_view_change(windows)
        self.t_install_ms = self.t_ms + self._view_change_duration()
        self.t_wait_ms = self.t_ms + self._backoff_ms()

    def _install(self, windows: Optional[Sequence[_Window]]) -> None:
        self.in_view_change = False
        self.t_install_ms = math.inf
        self.t_wait_ms = math.inf
        # The new primary is timed from the moment its view began (reset(), then
        # setFloor()).
        self._select_watch(self.t_ms)
        self._set_floor(self.t_ms)
        for w in windows or ():
            if w.covers(self.t_ms):
                w.new_views += 1

    def run_episode(
        self, choose: Callable[[Dict[str, float]], float]
    ) -> Tuple[Dict[str, float], float, Dict[str, float], int]:
        """Run one episode; ``choose(features) -> timeout_ms`` is called at the
        feature deadline. Returns (features, timeout, reward report, phase index
        at the apply instant)."""
        s = self.episode_start_s
        wa = _Window(s, s + FEATURE_S)
        self._advance_to((s + FEATURE_S) * 1000.0, [wa])
        self._count_view_changes(wa)
        features = wa.report(self.timeout_ms)
        # main.py labels an episode with the phase at its apply instant, so the
        # onset episode carries the phase its reward window measures.
        phase_idx, _ = self.phase_at(s + FEATURE_S + REPLY_S)
        new_timeout = float(choose(features))
        # Applied at the end of the reply wait (applyAgentTimeouts): instances
        # in flight keep going; a lower timeout floors the deadlines at the
        # change instead of putting every older request into the past.
        self._advance_to((s + FEATURE_S + REPLY_S) * 1000.0, None)
        if new_timeout < self.timeout_ms:
            self._set_floor(self.t_ms)
        self.timeout_ms = new_timeout
        wb = _Window(s + FEATURE_S + REPLY_S + WARMUP_S, s + EPISODE_S)
        self._advance_to((s + EPISODE_S) * 1000.0, [wb])
        self._count_view_changes(wb)
        self.episode_start_s = s + EPISODE_S
        return features, new_timeout, wb.report(new_timeout), phase_idx

    def _count_view_changes(self, w: _Window) -> None:
        """The window's view-change counts: the elections in the episode ending at its end."""
        end_ms, from_ms = w.end_s * 1000.0, (w.end_s - EPISODE_S) * 1000.0
        recent = [stalled for t, stalled in self._vc_log if from_ms < t <= end_ms]
        w.view_changes, w.no_progress = len(recent), sum(recent)
        self._vc_log = [(t, st) for t, st in self._vc_log if t > from_ms]


# --- Randomized environments -------------------------------------------------


# --sim-constants (train, eval; pretrain.py --sim-constants): an alternative host
# constants set, a JSON object of SimConstants fields over the installed
# defaults (E5 of runs/2026-10-18-pbft-ladder-variants: a fit without proposal
# delay, the delayed-leader shape neutral). None keeps the installed defaults.
# Set before the workers fork, so they inherit it, as NO_PROPOSAL_FAULTS.
HOST_CONSTANTS: Optional[SimConstants] = None


def set_host_constants(path: str) -> None:
    global HOST_CONSTANTS
    with open(path) as f:
        values = json.load(f)
    if not isinstance(values, dict):
        raise ValueError(f"{path}: expected a JSON object of SimConstants fields")
    try:
        HOST_CONSTANTS = replace(SimConstants(), **values)
    except TypeError as exc:
        raise ValueError(f"{path}: {exc}") from None


def host_constants() -> SimConstants:
    """The installed defaults, or the --sim-constants set (a fresh copy:
    callers mutate it)."""
    return SimConstants() if HOST_CONSTANTS is None else replace(HOST_CONSTANTS)


def sample_constants(rng: np.random.Generator) -> SimConstants:
    """Per-run physics (module docstring; sim_common.py, restart D11). The
    fitted constants (sim_fit.py FREE: processing, the byte cap, the
    delayed-leader shape, the view-change overhead and spread) are drawn
    0.5-2 x their installed values; the excess factor scales its excess over D,
    since the hold never costs less than D, and the byte cap stays within the
    count limit. The batching wait is drawn 0.5-2 x its cited value but at most
    1, and base delay, jitter, load and backlog come from sim_common. A spike
    of 0 (a neutral delayed-leader shape in a --sim-constants set) stays 0."""
    h = host_constants()
    return SimConstants(
        base_delay_ms=draw_base_delay_ms(rng),
        jitter_ratio=JITTER_RATIO,
        proc_ms=around(rng, h.proc_ms),
        proc_sigma=around(rng, h.proc_sigma),
        proc_per_tx_us=around(rng, h.proc_per_tx_us),
        load_tps=draw_load_tps(rng, reward_normalizer_tps("pbft")),
        batch_cap=int(round(around(rng, h.batch_cap, hi=min(2.0, BATCH_MAX_REQUESTS / h.batch_cap)))),
        batching_wait=around(rng, h.batching_wait, hi=min(2.0, 1.0 / h.batching_wait)),
        vc_overhead_ms=around(rng, h.vc_overhead_ms),
        vc_sigma=around(rng, h.vc_sigma),
        knee_ms=around(rng, h.knee_ms),
        excess_factor=1.0 + around(rng, h.excess_factor - 1.0),
        excess_sigma=around(rng, h.excess_sigma),
        spike_ms=around(rng, h.spike_ms) if h.spike_ms > 0 else 0.0,
        spike_batches=max(1, int(round(around(rng, h.spike_batches)))),
        initial_backlog=draw_initial_backlog(rng, MAX_PENDING_REQUESTS),
    )


# train --no-proposal-faults: a prior that never saw a delayed or silent
# proposer, so its pretraining covers the network-delay regimes only. The
# "blind pretrained" arm of the ablation - see main.py --prior-path.
NO_PROPOSAL_FAULTS = False


# The spec's hold re-pin interval (<proposalDelay><interval>), the centre of the
# training draw.
SPEC_REPIN_INTERVAL_S = 60.0


def sample_phase(rng: np.random.Generator, start_s: float) -> Phase:
    """One fault: the PBFT mix, with sim_common's magnitudes (restart D11)."""
    kind = rng.random()
    extra, prop = 0.0, 0.0
    if kind < 0.2:
        pass
    elif kind < 0.55:
        prop = draw_hold_ms(rng)
    elif kind < 0.9:
        extra = draw_link_extra_ms(rng)
    else:
        prop = draw_hold_ms(rng)
        extra = draw_link_extra_ms(rng)
    if NO_PROPOSAL_FAULTS:
        prop = 0.0
    return Phase(
        start_s=start_s,
        extra_delay_ms=extra,
        proposal_delay_ms=prop,
        interval_s=around(rng, SPEC_REPIN_INTERVAL_S),
    )


def sample_schedule(rng: np.random.Generator, total_s: float) -> List[Phase]:
    return [Phase(start_s=0.0)] + [sample_phase(rng, t) for t in fault_start_times(rng, total_s)]


SPEC_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "config", "pbft", "failure_spec.xml"))


def check_spec_schedule(phases: Sequence[Phase], path: str = SPEC_PATH) -> None:
    """Raise unless ``phases`` are the spec file's (restart D11, 2026-10-19): its
    warm-up, base link and jitter as WARMUP_PHASE_S and the SimConstants
    defaults, and per phase the onset (warm-up + atTime; 0 for the base phase),
    the continuous global delay, and the leader hold with its re-pin interval.
    Anything else the spec injects (stragglers, edges, silence, a numbered
    replica, partial bursts) raises."""
    import xml.etree.ElementTree as ET

    def num(node, tag, where):
        text = node.findtext(tag)
        if text is None:
            raise ValueError(f"{path}: {where} has no <{tag}>")
        return float(text)

    root = ET.parse(path).getroot()
    c = SimConstants()
    got = (num(root, "warmUpTime", "the spec"),
           num(root.find("globalNetworkDelay"), "delayMs", "<globalNetworkDelay>"),
           num(root.find("globalNetworkDelay"), "jitterRatio", "<globalNetworkDelay>"))
    if got != (WARMUP_PHASE_S, c.base_delay_ms, c.jitter_ratio):
        raise ValueError(f"{path}: warm-up, base delay, jitter {got} != the simulator's "
                         f"{(WARMUP_PHASE_S, c.base_delay_ms, c.jitter_ratio)}")
    spec = root.find("phases").findall("phase")
    if len(spec) != len(phases):
        raise ValueError(f"{path}: {len(spec)} phases, spec_schedule has {len(phases)}")
    for i, (ph, sim) in enumerate(zip(spec, phases)):
        where = f"phase {i}"
        extra, hold, interval = 0.0, 0.0, None
        net = ph.find("network")
        if net is not None:
            for tag in ("sources", "edges", "silence"):
                if net.find(tag) is not None and len(net.find(tag)):
                    raise ValueError(f"{path}: {where} has <{tag}> entries pbft_sim cannot model")
            gl = net.find("global")
            if gl is not None:
                extra = num(gl, "delayMs", where)
                dur, itv = gl.findtext("burstDurationMs"), gl.findtext("burstIntervalMs")
                if extra > 0 and not (dur is not None and float(dur) == float(itv or 0) > 0):
                    raise ValueError(f"{path}: {where} has partial delay bursts ({dur}/{itv} ms)")
        pd = ph.find("pbft/proposalDelay")
        if pd is not None:
            replicas = list(pd.find("replicas") if pd.find("replicas") is not None else [])
            if len(replicas) > 1 or any((r.findtext("id") or "").strip() != "leader" for r in replicas):
                raise ValueError(f"{path}: {where} delays a replica other than the leader token")
            if replicas:
                hold, interval = num(replicas[0], "delayMs", where), num(pd, "interval", where)
        start = WARMUP_PHASE_S + num(ph, "atTime", where)
        if i == 0:
            # The simulated base phase starts at 0 and so covers the warm-up,
            # which the spec leaves fault-free: its first phase must be too.
            if start != WARMUP_PHASE_S or extra or hold:
                raise ValueError(f"{path}: phase 0 must be a fault-free atTime 0 phase")
            start = 0.0
        want = (start, extra, hold)
        have = (sim.start_s, sim.extra_delay_ms, sim.proposal_delay_ms)
        if have != want or (interval is not None and sim.interval_s != interval):
            raise ValueError(f"{path}: {where} (start, extra, hold, interval) "
                             f"{want + (interval,)} != spec_schedule's {have + (sim.interval_s,)}")


def spec_schedule() -> List[Phase]:
    """config/pbft/failure_spec.xml (seven phases, delays re-pinned every
    60 s; p1 hold 300 since 2026-10-14, p2 hold 550 since 2026-10-13): warm-up 10 s, phases at warm-up + atTime. Used only to evaluate
    policies in simulation, never to train. Checked against the file on every
    build (check_spec_schedule)."""
    w = 10.0
    phases = [
        Phase(0.0),
        Phase(w + 300.0, proposal_delay_ms=300.0),  # 120 -> 200 -> 300, 2026-10-14 (Kevin): five machines tolerated 120 at T 250 and deposed 200 late
        Phase(w + 600.0, proposal_delay_ms=550.0),  # 400 -> 550, 2026-10-13 (Kevin): 550 deposes the held leader
        Phase(w + 900.0, extra_delay_ms=50.0),
        Phase(w + 1200.0, extra_delay_ms=50.0, proposal_delay_ms=500.0),
        Phase(w + 1500.0, extra_delay_ms=150.0),
        Phase(w + 1800.0),
    ]
    check_spec_schedule(phases)
    return phases


# --- Agent loop, data generation, prior training, evaluation -----------------


def feature_vector(report: Dict[str, float]) -> np.ndarray:
    from models import PBFT_REPORT_FEATURE_FIELD_NAMES

    return np.asarray(
        [report[name] for name in PBFT_REPORT_FEATURE_FIELD_NAMES], dtype=float
    )


_INT_FIELDS = (
    "total_transactions", "total_consensus_instances", "leader_change_count",
    "regency_change_count", "timeout_ms", "view_change_count",
    "no_progress_view_change_count",
)
# Report fields the reward calculator does not take (phase window means).
_PHASE_FIELDS = ("pre_prepare_latency_ms", "prepare_latency_ms", "commit_latency_ms")


def episode_reward(report: Dict[str, float]) -> float:
    """Reward of one reward window under the process-wide reward design, the
    same calculator main.reward_engineering uses; the applied timeout is the
    report's own timeout_ms, as in a PbftReport."""
    from reward_utils import calculate_pbft_reward

    return calculate_pbft_reward(
        **{
            name: int(round(value)) if name in _INT_FIELDS else float(value)
            for name, value in report.items() if name not in _PHASE_FIELDS
        }
    )


WARMUP_PHASE_S = 10.0  # failure_spec warmUpTime: decisions before it are phase -1
# Spec runs start the episode clock so that episode e applies at warm-up + 20(e-1), on the phase onsets.
SPEC_START_OFFSET_S = WARMUP_PHASE_S - FEATURE_S - REPLY_S


def run_agent(
    sim: PbftSim,
    episodes: int,
    predict: Callable[[np.ndarray], float],
    update: Optional[Callable[[np.ndarray, float, float], None]],
) -> List[Tuple[int, float, Dict[str, float]]]:
    """Drive ``sim`` like LearningAgentServicer: predict at the feature deadline,
    reward from the reward window, no update for warm-up episodes."""
    rows = []
    for _ in range(episodes):
        decision_s = sim.episode_start_s + FEATURE_S
        holder: Dict[str, np.ndarray] = {}

        def choose(report: Dict[str, float]) -> float:
            holder["x"] = feature_vector(report)
            return predict(holder["x"])

        _, timeout_ms, reward_report, phase_idx = sim.run_episode(choose)
        phase = phase_idx if decision_s >= WARMUP_PHASE_S else -1
        if update is not None and phase != -1:
            update(holder["x"], timeout_ms, episode_reward(reward_report))
        rows.append((phase, timeout_ms, reward_report))
    return rows


def training_sim(rng: np.random.Generator, seed: int) -> Tuple[PbftSim, int]:
    """One randomized training run: (simulator, episodes)."""
    episodes = int(rng.integers(40, 120))
    initial = draw_initial_timeout_ms(rng, agent_pb2.PROTOCOL_PBFT)  # sim_common, restart D11
    sim = PbftSim(
        sample_constants(rng),
        sample_schedule(rng, episodes * EPISODE_S),
        seed=seed,
        start_offset_s=float(rng.uniform(0.0, 20.0)),
        initial_timeout_ms=initial,
    )
    return sim, episodes


def _generate_one(args: Tuple[int, str, int]) -> Tuple[np.ndarray, np.ndarray]:
    """One randomized run. Every ``hold`` episodes an action is drawn and held for
    ``hold`` episodes; a sample is (context at the hold's first decision, action) ->
    the mean reward over the hold (hold 1 = per episode), as sbft_sim's."""
    from reward_utils import set_reward_design

    seed, design, hold = args
    set_reward_design(design)
    from continuous_bandit import ContinuousTimeoutActionCodec, report_context
    from models import PBFT_REPORT_FEATURE_FIELD_NAMES

    spec = CONTEXT_SPECS[agent_pb2.PROTOCOL_PBFT]
    lower_ms, upper_ms = spec.bounds_ms
    rng = np.random.default_rng(seed)
    codec = ContinuousTimeoutActionCodec(agent_pb2.PROTOCOL_PBFT, bounds_ms=spec.bounds_ms)
    sim, episodes = training_sim(rng, seed)
    prev = codec.normalize_ms(int(round(min(max(sim.timeout_ms, lower_ms), upper_ms))))
    xs, ys = [], []
    for _ in range(episodes // hold):
        holder = {}

        def choose(report):
            if "a" in holder:
                return holder["ms"]
            x = feature_vector(report)
            holder["ctx"] = report_context(spec, PBFT_REPORT_FEATURE_FIELD_NAMES, x)
            if rng.random() < 0.6:
                a = float(rng.random())
            else:
                a = float(np.clip(prev + rng.normal(0.0, 0.08), 0.0, 1.0))
            ms = codec._denormalize_value(a, 0)
            holder["a"] = codec.normalize_ms(ms)
            holder["ms"] = float(ms)
            return holder["ms"]

        rewards = []
        for _ in range(hold):
            _, _, reward_report, _ = sim.run_episode(choose)
            rewards.append(episode_reward(reward_report))
        xs.append(np.append(holder["ctx"], holder["a"]))
        ys.append(float(np.mean(rewards)))
        prev = holder["a"]
    return np.asarray(xs), np.asarray(ys)


def generate(n_runs: int, seed: int, workers: int, design: str, hold: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    from multiprocessing import Pool

    if hold < 1:
        raise ValueError("--hold-episodes must be at least 1")
    seeds = [(seed + i, design, hold) for i in range(n_runs)]
    with Pool(workers) as pool:
        parts = pool.map(_generate_one, seeds, chunksize=8)
    return np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def make_policy(kind: str, prior_path: str = DEFAULT_PRIOR_PATH, **kwargs):
    """(predict, update) closures around the real model classes, built as main.serve does."""
    from continuous_bandit import (
        ContinuousTimeoutActionCodec,
        SimPriorGPModel,
    )
    from latency_floor import LatencyFloor, LatencyFloorPolicy, latency_feature_indices
    from models import (
        DiscountedLinUCB,
        ProtocolTimeoutActionCodec,
        PBFT_REPORT_FEATURE_FIELD_NAMES as names,
        get_timeout_values_ms,
    )

    protocol = agent_pb2.PROTOCOL_PBFT
    arms = get_timeout_values_ms(protocol)
    if kind == "heuristic":
        from model_builder import build_model

        built = build_model(protocol, "continuous_heuristic", prior_windows=SIM_WINDOWS)
        model, codec = built.model, built.action_codec
    elif kind == "discrete_linucb":
        codec = ProtocolTimeoutActionCodec(protocol)
        latency_index, instances_index = latency_feature_indices(names)
        model = LatencyFloorPolicy(
            model=DiscountedLinUCB(action_dim=len(arms), n_features=len(names), seed=42),
            timeouts_ms=arms,
            latency_index=latency_index,
            n_features=len(names),
            instances_index=instances_index,
            latency_floor=LatencyFloor(multiplier=1.0, quantile=0.5),
        )
    elif kind == "candidate":
        codec = ContinuousTimeoutActionCodec(protocol, bounds_ms=CONTEXT_SPECS[protocol].bounds_ms)
        from reward_utils import get_reward_design

        check_prior_windows(prior_path, SIM_WINDOWS)
        model = SimPriorGPModel(
            action_codec=codec,
            field_names=names,
            prior_path=prior_path,
            reward_design=get_reward_design(),
            **kwargs,
        )
    else:
        raise ValueError(kind)

    def predict(x: np.ndarray) -> float:
        timeout = codec.decode(model.predict_action(x))
        return float(timeout.pbft.election_timeout_milliseconds)

    def update(x: np.ndarray, timeout_ms: float, reward: float) -> None:
        used = agent_pb2.PbftTimeout(election_timeout_milliseconds=int(round(timeout_ms)))
        model.update_model(x, codec.encode(used), reward)

    return predict, update


STRICT_PHASES = (1, 2, 4)  # runs/2026-09-29-continuous-port/spec.md rule: escape phases


def evaluate_spec(kind: str, constants: SimConstants, seed: int, **kwargs) -> np.ndarray:
    """Per-phase (tps, mean latency, median timeout) of one policy on the spec."""
    predict, update = make_policy(kind, **kwargs)
    sim = PbftSim(constants, spec_schedule(), seed=seed, start_offset_s=SPEC_START_OFFSET_S)
    rows = run_agent(sim, 105, predict, update)
    out = np.zeros((7, 3))
    for ph in range(7):
        sel = [r for r in rows if r[0] == ph]
        out[ph] = (
            np.mean([r[2]["throughput_tps"] for r in sel]),
            np.mean([r[2]["avg_consensus_latency_ms"] for r in sel]),
            np.median([r[1] for r in sel]),
        )
    return out


def pretrain_adapter():
    """This simulator as pretrain.py drives it: training runs as train draws
    them, spec runs as evaluate_spec builds them (host constants, 105
    episodes); the spec phase is run_agent's, and the oracle plays
    learning_agent/config/schedule_pbft.csv as ScheduledTimeoutModel does."""
    from pretrain import SimAdapter, timeout_message

    def to_timeout(timeout_ms: float) -> agent_pb2.Timeout:
        return agent_pb2.Timeout(pbft=agent_pb2.PbftTimeout(
            election_timeout_milliseconds=int(round(timeout_ms))))

    def from_timeout(timeout: agent_pb2.Timeout) -> float:
        return float(timeout_message(timeout, "pbft").election_timeout_milliseconds)

    rows = schedule_rows("pbft")

    def oracle(sim) -> Callable[[np.ndarray], float]:
        # The apply time: the decision at the feature deadline reaches the
        # replicas a reply window later (Kevin, 2026-10-19, restart: the sim gate
        # compares against the oracle on all six).
        def predict(x: np.ndarray) -> float:
            return scheduled_action(rows, sim.episode_start_s + FEATURE_S + REPLY_S, WARMUP_PHASE_S)[0]

        return predict

    return SimAdapter(
        protocol=agent_pb2.PROTOCOL_PBFT,
        simulator="pbft_sim",
        training_run=lambda seed: training_sim(np.random.default_rng(seed), seed),
        spec_run=lambda seed: (
            # The engine opens at the middle rung (--initial-timeout in the rows, D10b).
            PbftSim(host_constants(), spec_schedule(), seed=seed, start_offset_s=SPEC_START_OFFSET_S,
                    initial_timeout_ms=float(opening_action_ms(agent_pb2.PROTOCOL_PBFT)[0])), 105),
        run_agent=run_agent,
        # A training schedule's phase 0 is a real phase, so run_agent already
        # updates every decision from 10 s.
        run_training=run_agent,
        to_timeout=to_timeout,
        from_timeout=from_timeout,
        row_phase=lambda row: row[0],
        row_reward=lambda row: episode_reward(row[2]),
        spec_phase_count=7,
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
    t.add_argument("--target-floor", type=float, default=-2.0,
                   help="fit rewards below this value as this value "
                        "(SimReturnGPModel applies the same floor online, and "
                        "refuses a prior fit to the raw reward); default -2, as "
                        "the pretraining recipe passes; -inf for a raw fit")
    t.add_argument("--hold-episodes", type=int, default=3,
                   help="episodes each training action is held, the target their "
                        "mean reward; 3 = the 60 s leader re-pin (restart D11)")
    t.add_argument("--no-proposal-faults", action="store_true",
                   help="train on network-delay phases only; the prior never "
                        "sees a delayed or silent proposer")
    t.add_argument("--sim-constants", default=None,
                   help="JSON of SimConstants fields replacing the installed defaults")
    e = sub.add_parser("eval", help="compare policies on the spec schedule in simulation")
    e.add_argument("--seeds", type=int, default=10)
    e.add_argument("--prior", default=DEFAULT_PRIOR_PATH)
    e.add_argument("--randomize", action="store_true", help="random constants per seed")
    e.add_argument("--sim-constants", default=None,
                   help="JSON of SimConstants fields replacing the installed defaults")
    e.add_argument("--reward-design", default=DEFAULT_REWARD_DESIGN,
                   help="reward the candidate learns from; the baselines learn from "
                        "the active reward (reward_utils.DEFAULT_REWARD_DESIGN)")
    args = parser.parse_args()
    from reward_utils import set_reward_design

    if args.sim_constants is not None:
        set_host_constants(args.sim_constants)  # before generate() forks its workers
        print(f"[sim] constants from {args.sim_constants}: {HOST_CONSTANTS}")
    if args.cmd == "train":
        global NO_PROPOSAL_FAULTS
        NO_PROPOSAL_FAULTS = bool(args.no_proposal_faults)  # inherited by the forked workers
        x, y = generate(args.runs, args.seed, args.workers, args.reward_design, args.hold_episodes)
        rmse = train_gp_prior(
            agent_pb2.PROTOCOL_PBFT, x, y, args.out, seed=args.seed, design=args.reward_design,
            hold=args.hold_episodes, target_floor=args.target_floor, windows=SIM_WINDOWS,
            sim_constants=asdict(host_constants()), code_hash=args.code_hash)
        print(f"samples={len(x)} hold_episodes={args.hold_episodes} target_floor={args.target_floor} "
              f"reward_std={y.std():.4f} holdout_rmse={rmse:.4f} -> {args.out}")
        return

    rng = np.random.default_rng(12345)
    for s in range(args.seeds):
        constants = sample_constants(rng) if args.randomize else host_constants()
        if args.randomize:
            constants.load_tps = 20000.0
        res = {}
        for k in ("discrete_linucb", "heuristic", "candidate"):
            set_reward_design(args.reward_design if k == "candidate" else DEFAULT_REWARD_DESIGN)
            res[k] = evaluate_spec(k, constants, s, **({"prior_path": args.prior} if k == "candidate" else {}))
        wins = 0
        line = []
        for ph in range(7):
            bt = max(res["discrete_linucb"][ph, 0], res["heuristic"][ph, 0])
            bl = min(res["discrete_linucb"][ph, 1], res["heuristic"][ph, 1])
            ct, cl = res["candidate"][ph, 0], res["candidate"][ph, 1]
            ok = ct >= 0.95 * bt and (cl < bl if ph in STRICT_PHASES else cl <= 1.05 * bl)
            wins += ok
            line.append(
                f"p{ph} {ct:5.0f}/{cl:5.0f}@{res['candidate'][ph,2]:4.0f} "
                f"[{res['heuristic'][ph,0]:5.0f}/{res['heuristic'][ph,1]:5.0f}@{res['heuristic'][ph,2]:4.0f} "
                f"{res['discrete_linucb'][ph,0]:5.0f}/{res['discrete_linucb'][ph,1]:5.0f}@{res['discrete_linucb'][ph,2]:4.0f}]{'+' if ok else '-'}"
            )
        print(f"seed {s} phases_ok={wins}/7 vc={constants.vc_overhead_ms:.0f} | " + " | ".join(line))


if __name__ == "__main__":
    main()
