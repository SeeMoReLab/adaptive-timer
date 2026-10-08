"""Analytic CppBedrock PBFT simulator for pretraining the continuous model.

Built from the engine's mechanics (CppBedrock/adaptive_timers.MD, the request
watchdog in PendingRequestTimer.cpp and EntityViewChange.cpp, the proposal
delay controller in FailureSpec.cpp, the report accounting in AgentClient.cpp)
and the harness constants in scripts/cloudlab/lib_bedrock.sh, not fit to run
data. Four replicas, a client broadcasting ``load_tps`` requests/s, one leader
proposing on a fixed cadence; one batch is one consensus instance:

- Every 100 ms (``--proposal-interval``) the leader takes the oldest pending
  requests up to ``batch_cap`` (8192 by count; the 512 KiB byte budget binds
  earlier) when fewer than 16 batches are in flight
  (``--max-inflight-batches``); a full pipeline skips the tick. Unexecuted
  requests are capped at 32,768 (``--max-pending-requests``); arrivals beyond
  the cap are rejected and lost.
- A batch commits after three one-way hops (pre-prepare, prepare, commit) plus
  engine processing, and executes in sequence order. Its latency is dated from
  the oldest request it carries (``firstSeenUs``), which by the tick has
  waited about one interval, so the healthy latency is the hop model
  ``100 + 3h``. Netem jitter per hop and lognormal processing noise put p95
  about 20 ms above p50.
- Request watchdog (``PendingRequestTimer``): the oldest unexecuted request is
  timed from its arrival, ``deadline = max(arrival, floor) + T << backoff``,
  capped at 30 s (``kMaxElectionBackoffMs``). Expiry starts a view change:
  in-flight batches are cancelled and their requests re-proposed by the next
  leader with their arrival times kept, and the view installs after two hops
  (ViewChange, NewView) plus ``vc_overhead_ms`` of evidence exchange, NewView
  verification and body fetches. Installing floors every inherited deadline at
  that instant, since the incoming primary is not blamed for the backlog it
  inherits; a view that does not install within the backed-off wait starts
  another view change (``onViewChangeWaitExpired``). The backoff doubles per
  view change started and resets only when the watched request completes
  before a deadline that was not floored, or when the queue drains
  (``EntityRequests.cpp``). Lowering T floors the deadlines at the change
  (``applyAgentTimeouts``, commit e1a3c9a), so a decrease never condemns
  pending requests retroactively. A rung under its level's tail therefore
  fires on ordinary latency and storms; one landing on a deep backlog cascades
  until the doubling outlasts the drain.
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

Episodes follow the wall-clock learning windows in lib_bedrock.sh: 10 s
features, 2 s reply wait (applied at the deadline), 3 s warm-up, 5 s reward.
Reports carry the 16 PbftReport fields as ``AgentClient::MetricsWindow``
fills them on replica 0: one latency and batch-size sample per executed
batch, inter-commit gaps between consecutive executions, view changes as
started (no-progress when nothing committed since the last), leader and
regency changes as views installed, and the election timeout in effect.

Constants the mechanics do not fix (link delay, processing costs, the byte
cap, the view-change overhead, the knee, the load, a start-up backlog) are
drawn per simulated run from ranges wider than the calibration host, so the
prior learned from it has to hold across them.
"""

from __future__ import annotations

import argparse
import math
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

import agent_pb2
from continuous_bandit import CONTEXT_SPECS

PROPOSAL_INTERVAL_MS = 100.0
MAX_INFLIGHT_BATCHES = 16
MAX_PENDING_REQUESTS = 32768
BATCH_MAX_REQUESTS = 8192
MAX_ELECTION_BACKOFF_MS = 30000.0
# config.pbft.yaml timers.view_change_ms: the election timeout before the
# first decision.
DEFAULT_ELECTION_TIMEOUT_MS = 8000.0
FEATURE_S = 10.0
REPLY_S = 2.0
WARMUP_S = 3.0
REWARD_S = 5.0
EPISODE_S = FEATURE_S + REPLY_S + WARMUP_S + REWARD_S

DEFAULT_PRIOR_PATH = CONTEXT_SPECS[agent_pb2.PROTOCOL_PBFT].prior_path


@dataclass
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
    proc_ms: float = 45.0
    proc_sigma: float = 0.2
    proc_per_tx_us: float = 4.0
    load_tps: float = 20000.0
    # Requests per batch where the 512 KiB budget binds (5,500 at 40k in
    # adaptive_timers.MD); the count limit is 8192.
    batch_cap: int = 5500
    # Age of a batch's oldest request at the tick, in proposal intervals: 1
    # when the client's lanes spread arrivals over the interval.
    batching_wait: float = 1.0
    # View change beyond its two hops: evidence exchange, NewView
    # verification, body fetches; one maintenance interval by default.
    vc_overhead_ms: float = 250.0
    vc_sigma: float = 0.2
    knee_ms: float = 175.0
    excess_factor: float = 1.2
    excess_sigma: float = 0.35
    spike_ms: float = 160.0
    spike_batches: int = 2
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


@dataclass
class _Window:
    start_s: float
    end_s: float
    txs: float = 0.0
    instances: int = 0
    lat: List[float] = field(default_factory=list)
    batch: List[float] = field(default_factory=list)
    gap: List[float] = field(default_factory=list)
    view_changes: int = 0
    no_progress: int = 0
    new_views: int = 0

    def covers(self, t_ms: float) -> bool:
        return self.start_s * 1000.0 <= t_ms < self.end_s * 1000.0

    def report(self, timeout_ms: float) -> Dict[str, float]:
        def avg(v):
            return float(np.mean(v)) if v else 0.0

        def pct(v, q):
            # Linear interpolation between ranks, as AgentClient.cpp percentile.
            return float(np.percentile(v, q)) if v else 0.0

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
        self.view = 0
        self.timeout_ms = float(initial_timeout_ms)
        self.backoff = 0
        self.floor_ms = -math.inf
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

    def _deadline(self, arrival_ms: float) -> float:
        return max(arrival_ms, self.floor_ms) + self._backoff_ms()

    def _advance_to(self, end_ms: float, windows: Optional[Sequence[_Window]]) -> None:
        while True:
            t_exec = self.inflight[0].t_exec if self.inflight else math.inf
            t_next = min(self.next_tick_ms, t_exec, self.t_install_ms, self.t_wait_ms)
            if not self.in_view_change:
                oldest = self._oldest_arrival()
                if oldest is not None:
                    deadline = self._deadline(oldest)
                    if deadline < t_next:
                        # A deadline already past (the backoff just reset under
                        # a request older than T) fires now.
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
        hops = (
            d * (1.0 + jr * float(u(0.0, 1.0)))
            + d * (1.0 + jr * float(u(-1.0, 1.0)))
            + d * (1.0 + jr * float(u(-1.0, 1.0)))
        )
        proc = c.proc_ms * float(self.rng.lognormal(0.0, c.proc_sigma)) + c.proc_per_tx_us * size / 1000.0
        ready = t + hold + hops + proc
        if self.inflight:
            prev = self.inflight[-1].t_exec
        else:
            prev = self.last_exec_ms if self.last_exec_ms is not None else -math.inf
        self.inflight.append(_Batch(max(ready, prev), chunks[0][0], size, chunks))
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
                    if self.last_exec_ms is not None:
                        w.gap.append(t - self.last_exec_ms)
        # The backoff restores only on evidence the timeout is adequate: the
        # watched entry (this batch carries the oldest) completed before a
        # deadline granted from arrival, not floored. Draining the queue
        # counts either way.
        if self.backoff:
            floored = self.floor_ms > b.oldest_ms
            if (not floored and t < self._deadline(b.oldest_ms)) or self._unexecuted() < 0.5:
                self.backoff = 0
        self.commits_since_vc += 1
        self.last_exec_ms = t

    def _record_view_change(self, windows: Optional[Sequence[_Window]]) -> None:
        for w in windows or ():
            if w.covers(self.t_ms):
                w.view_changes += 1
                if self.commits_since_vc == 0:
                    w.no_progress += 1
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
        # The new primary is timed from the moment its view began.
        self.floor_ms = self.t_ms
        for w in windows or ():
            if w.covers(self.t_ms):
                w.new_views += 1

    def run_episode(
        self, choose: Callable[[Dict[str, float]], float]
    ) -> Tuple[Dict[str, float], float, Dict[str, float], int]:
        """Run one episode; ``choose(features) -> timeout_ms`` is called at the
        feature deadline. Returns (features, timeout, reward report, phase index
        at decision time)."""
        s = self.episode_start_s
        wa = _Window(s, s + FEATURE_S)
        self._advance_to((s + FEATURE_S) * 1000.0, [wa])
        features = wa.report(self.timeout_ms)
        phase_idx, _ = self.phase_at(s + FEATURE_S)
        new_timeout = float(choose(features))
        # Applied at the end of the reply wait (applyAgentTimeouts): instances
        # in flight keep going; a lower timeout floors the deadlines at the
        # change instead of putting every older request into the past.
        self._advance_to((s + FEATURE_S + REPLY_S) * 1000.0, None)
        if new_timeout < self.timeout_ms:
            self.floor_ms = self.t_ms
        self.timeout_ms = new_timeout
        wb = _Window(s + FEATURE_S + REPLY_S + WARMUP_S, s + EPISODE_S)
        self._advance_to((s + EPISODE_S) * 1000.0, [wb])
        self.episode_start_s = s + EPISODE_S
        return features, new_timeout, wb.report(new_timeout), phase_idx


# --- Randomized environments -------------------------------------------------


def sample_constants(rng: np.random.Generator) -> SimConstants:
    return SimConstants(
        base_delay_ms=float(rng.uniform(2.0, 10.0)),
        jitter_ratio=float(rng.uniform(0.0, 0.1)),
        proc_ms=float(rng.uniform(0.0, 70.0)),
        proc_sigma=float(rng.uniform(0.05, 0.3)),
        proc_per_tx_us=float(rng.uniform(0.0, 10.0)),
        load_tps=float(rng.uniform(8000.0, 26000.0)),
        batch_cap=int(rng.integers(4000, BATCH_MAX_REQUESTS + 1)),
        batching_wait=float(rng.uniform(0.6, 1.0)),
        vc_overhead_ms=float(math.exp(rng.uniform(math.log(50.0), math.log(1500.0)))),
        vc_sigma=float(rng.uniform(0.0, 0.5)),
        knee_ms=float(rng.uniform(120.0, 250.0)),
        excess_factor=float(rng.uniform(1.0, 1.8)),
        excess_sigma=float(rng.uniform(0.0, 0.8)),
        spike_ms=float(rng.uniform(50.0, 450.0)),
        spike_batches=int(rng.integers(1, 5)),
        initial_backlog=int(rng.integers(0, MAX_PENDING_REQUESTS + 1)) if rng.random() < 0.5 else 0,
    )


# train --no-proposal-faults: a prior that never saw a delayed or silent
# proposer, so its pretraining covers the network-delay regimes only. The
# "blind pretrained" arm of the ablation - see main.py --prior-path.
NO_PROPOSAL_FAULTS = False


def sample_phase(rng: np.random.Generator, start_s: float) -> Phase:
    kind = rng.random()
    extra, prop = 0.0, 0.0
    if kind < 0.2:
        pass
    elif kind < 0.55:
        prop = float(math.exp(rng.uniform(math.log(50.0), math.log(2000.0))))
    elif kind < 0.9:
        extra = float(rng.uniform(10.0, 300.0))
    else:
        prop = float(math.exp(rng.uniform(math.log(50.0), math.log(2000.0))))
        extra = float(rng.uniform(10.0, 300.0))
    if NO_PROPOSAL_FAULTS:
        prop = 0.0
    return Phase(
        start_s=start_s,
        extra_delay_ms=extra,
        proposal_delay_ms=prop,
        interval_s=float(rng.uniform(30.0, 120.0)),
    )


def sample_schedule(rng: np.random.Generator, total_s: float) -> List[Phase]:
    phases = [Phase(start_s=0.0)]
    t = float(rng.uniform(40.0, 400.0))
    while t < total_s:
        phases.append(sample_phase(rng, t))
        t += float(rng.uniform(40.0, 400.0))
    return phases


def spec_schedule() -> List[Phase]:
    """config/pbft/failure_spec.xml (seven phases, delays re-pinned every
    60 s): warm-up 10 s, phases at warm-up + atTime. Used only to evaluate
    policies in simulation, never to train."""
    w = 10.0
    return [
        Phase(0.0),
        Phase(w + 300.0, proposal_delay_ms=120.0),
        Phase(w + 600.0, proposal_delay_ms=400.0),
        Phase(w + 900.0, extra_delay_ms=50.0),
        Phase(w + 1200.0, extra_delay_ms=50.0, proposal_delay_ms=500.0),
        Phase(w + 1500.0, extra_delay_ms=150.0),
        Phase(w + 1800.0),
    ]


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


def episode_reward(report: Dict[str, float]) -> float:
    """Reward of one reward window under the process-wide reward design, the
    same calculator main.reward_engineering uses; the applied timeout is the
    report's own timeout_ms, as in a PbftReport."""
    from reward_utils import calculate_pbft_reward

    return calculate_pbft_reward(
        **{
            name: int(round(value)) if name in _INT_FIELDS else float(value)
            for name, value in report.items()
        }
    )


WARMUP_PHASE_S = 10.0  # failure_spec warmUpTime: decisions before it are phase -1


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
    lower_ms, upper_ms = CONTEXT_SPECS[agent_pb2.PROTOCOL_PBFT].bounds_ms
    episodes = int(rng.integers(40, 120))
    # A quarter of the runs start on the CppBedrock default, as a real run
    # does before its first decision.
    if rng.random() < 0.25:
        initial = DEFAULT_ELECTION_TIMEOUT_MS
    else:
        initial = float(math.exp(rng.uniform(math.log(lower_ms), math.log(upper_ms))))
    sim = PbftSim(
        sample_constants(rng),
        sample_schedule(rng, episodes * EPISODE_S),
        seed=seed,
        start_offset_s=float(rng.uniform(0.0, 20.0)),
        initial_timeout_ms=initial,
    )
    return sim, episodes


def _generate_one(args: Tuple[int, str]) -> Tuple[np.ndarray, np.ndarray]:
    from reward_utils import set_reward_design

    seed, design = args
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
    for _ in range(episodes):
        holder = {}

        def choose(report):
            nonlocal prev
            x = feature_vector(report)
            holder["ctx"] = report_context(spec, PBFT_REPORT_FEATURE_FIELD_NAMES, x)
            if rng.random() < 0.6:
                a = float(rng.random())
            else:
                a = float(np.clip(prev + rng.normal(0.0, 0.08), 0.0, 1.0))
            ms = codec._denormalize_value(a, 0)
            holder["a"] = codec.normalize_ms(ms)
            return float(ms)

        _, _, reward_report, _ = sim.run_episode(choose)
        xs.append(np.append(holder["ctx"], holder["a"]))
        ys.append(episode_reward(reward_report))
        prev = holder["a"]
    return np.asarray(xs), np.asarray(ys)


def generate(n_runs: int, seed: int, workers: int, design: str) -> Tuple[np.ndarray, np.ndarray]:
    from multiprocessing import Pool

    seeds = [(seed + i, design) for i in range(n_runs)]
    with Pool(workers) as pool:
        parts = pool.map(_generate_one, seeds, chunksize=8)
    return np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def train_prior(
    x: np.ndarray,
    y: np.ndarray,
    out_path: str,
    seed: int,
    design: str,
    target_floor: float = -math.inf,
) -> float:
    from sklearn.neural_network import MLPRegressor

    # Rewards below the floor are fit as the floor. SimReturnGPModel applies
    # the same floor to the online reward, so the residual compares like with
    # like, and the fit stops spending its error on collapses instead of on
    # the structure that ranks the usable timeouts. A raw fit (-inf) is only
    # usable by SimPriorGPModel.
    y = np.maximum(y, target_floor)
    x_mean, x_std = x.mean(axis=0), x.std(axis=0) + 1e-9
    y_mean, y_std = float(y.mean()), float(y.std() + 1e-9)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    cut = int(0.9 * len(x))
    tr, va = order[:cut], order[cut:]
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 128, 64),
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
        target_floor=np.asarray(float(target_floor)),
    )
    np.savez(out_path, **arrays)
    return rmse


def make_policy(kind: str, prior_path: str = DEFAULT_PRIOR_PATH, **kwargs):
    """(predict, update) closures around the real model classes, built as main.serve does."""
    from continuous_bandit import (
        ContinuousHeuristicTimeoutModel,
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
        codec = ContinuousTimeoutActionCodec(protocol)
        model = ContinuousHeuristicTimeoutModel(
            action_codec=codec,
            n_features=len(names),
            latency_index=names.index("avg_consensus_latency_ms"),
            instances_index=names.index("total_consensus_instances"),
            multiplier=1.0,
            clamp_ms=(min(arms), max(arms)),
            initial_timeout_ms=sorted(arms)[len(arms) // 2],
        )
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
    # The first batch executes about 0.2 s after the client starts, so the
    # episode clock (AgentClient firstSampleTime_) leads the spec clock by
    # that much and every phase switch lands just before a decision.
    sim = PbftSim(constants, spec_schedule(), seed=seed, start_offset_s=0.2)
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
    episodes); the spec phase is run_agent's, and there is no oracle."""
    from pretrain import SimAdapter, timeout_message

    def to_timeout(timeout_ms: float) -> agent_pb2.Timeout:
        return agent_pb2.Timeout(pbft=agent_pb2.PbftTimeout(
            election_timeout_milliseconds=int(round(timeout_ms))))

    def from_timeout(timeout: agent_pb2.Timeout) -> float:
        return float(timeout_message(timeout, "pbft").election_timeout_milliseconds)

    return SimAdapter(
        protocol=agent_pb2.PROTOCOL_PBFT,
        simulator="pbft_sim",
        training_run=lambda seed: training_sim(np.random.default_rng(seed), seed),
        spec_run=lambda seed: (
            PbftSim(SimConstants(), spec_schedule(), seed=seed, start_offset_s=0.2), 105),
        run_agent=run_agent,
        # A training schedule's phase 0 is a real phase, so run_agent already
        # updates every decision from 10 s.
        run_training=run_agent,
        to_timeout=to_timeout,
        from_timeout=from_timeout,
        row_phase=lambda row: row[0],
        row_reward=lambda row: episode_reward(row[2]),
        spec_phase_count=7,
        oracle=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="generate randomized episodes and fit the prior")
    t.add_argument("--runs", type=int, default=4000)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    t.add_argument("--out", default=DEFAULT_PRIOR_PATH)
    t.add_argument("--reward-design", default="reward_mixed")
    t.add_argument("--target-floor", type=float, default=-math.inf,
                   help="fit rewards below this value as this value "
                        "(SimReturnGPModel applies the same floor online, and "
                        "refuses a prior fit to the raw reward); default: no floor")
    t.add_argument("--no-proposal-faults", action="store_true",
                   help="train on network-delay phases only; the prior never "
                        "sees a delayed or silent proposer")
    e = sub.add_parser("eval", help="compare policies on the spec schedule in simulation")
    e.add_argument("--seeds", type=int, default=10)
    e.add_argument("--prior", default=DEFAULT_PRIOR_PATH)
    e.add_argument("--randomize", action="store_true", help="random constants per seed")
    e.add_argument("--reward-design", default="reward_mixed",
                   help="reward the candidate learns from; the baselines keep reward_mixed")
    args = parser.parse_args()
    from reward_utils import set_reward_design

    if args.cmd == "train":
        global NO_PROPOSAL_FAULTS
        NO_PROPOSAL_FAULTS = bool(args.no_proposal_faults)  # inherited by the forked workers
        x, y = generate(args.runs, args.seed, args.workers, args.reward_design)
        rmse = train_prior(x, y, args.out, args.seed, args.reward_design, args.target_floor)
        print(f"samples={len(x)} reward_std={y.std():.4f} holdout_rmse={rmse:.4f} -> {args.out}")
        return

    rng = np.random.default_rng(12345)
    for s in range(args.seeds):
        constants = sample_constants(rng) if args.randomize else SimConstants()
        if args.randomize:
            constants.load_tps = 20000.0
        res = {}
        for k in ("discrete_linucb", "heuristic", "candidate"):
            set_reward_design(args.reward_design if k == "candidate" else "reward_mixed")
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
