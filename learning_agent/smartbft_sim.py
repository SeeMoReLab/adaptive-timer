"""Analytic SmartBFT (smallbank) simulator for pretraining the continuous model.

Built from the library's mechanics and the smallbank harness constants, not
fit to run data. One consensus instance is simulated at a time, all four
nodes in step:

- The client is closed loop: ``terminals`` requests are outstanding at all
  times (config/smartbft/smallbank.xml, 512, rate 0). Each is broadcast to
  every replica, and a terminal resubmits ``client_turnaround_ms`` after its
  batch is decided. Throughput is therefore terminals / request latency, and
  a phase that stretches the latency collapses the throughput on its own.
- The leader proposes one batch at a time (SmartBFT is sequential): cut when
  256 requests are pending or 50 ms after it became free, whichever is first
  (``BatchBuilder.NextBatch``). A decision takes the pre-prepare hop plus the
  proposal transfer, the prepare hop, the commit hop and processing:
  ``3h + transfer + work`` with ``h`` the one-way link delay.
- The agent's timeout ``T`` is split into a forward timer ``T/2`` and a
  complain timer ``T/2`` per pending request (``splitRequestTimeout``), so a
  request complains ``T`` after its timer started. Timers start at arrival and
  restart for every pending request on each agent apply (``RestartTimers``)
  and on each new-view install, so a decrease dates deadlines from the change,
  never from arrival. Broadcast clients give every follower the same pool, so
  the followers' complaints land together: two ViewChange votes (quorum - 1
  with ``SpeedUpViewChange`` off, which is f + 1 at n = 4) make every node
  join, deposing the leader.
- A view change is ViewChange (one hop to join), ViewData to the new leader
  (one hop), NewView (one hop) and processing. If NewView is not installed
  within ``T x backoff`` the view changer syncs and starts again with the
  backoff one higher (``checkIfTimeout``); backoff resets on install. The
  new leader is ``view mod 4``. A batch whose complaint fires during its
  commit hop still lands (the followers' commits are already out), but the
  view changes all the same.
- Proposal delay ``D`` (``proposalSendDelay``): the pinned leader holds each
  pre-prepare ``D`` ms on its send line. The pin is re-resolved to the current
  leader every ``interval`` (60 s) of the phase, so a deposed leader's
  successor runs clean until the next tick.
- Storm mechanics follow from the above: after an install every pending
  request has deadline ``install + T``, the new leader must drain the whole
  pool (terminals / 256 batches) before then, and a view that cannot is
  deposed again after delivering what it got through. A view whose first
  decision misses ``T`` delivers nothing.

Episodes follow the wall-clock learning windows of run_smartbft.sh: 7 s
features, 2 s reply wait (apply at its end), 4 s warm-up, 7 s reward. Reports
carry the 16 SmartbftReport fields with ``learning_metrics.go``'s
definitions: latency from the replica's own submit mark to the decision, over
delivered requests; throughput over the window; regency and leader changes
between consecutive decisions in the window; ``view_change_count`` one per
``start_view_change`` event at this node (retries included);
``no_progress_view_change_count`` always 0 (smallbank never records it);
``timeout_ms`` the applied T: the one in force when the window opened,
updated by each decision.

Constants the mechanics do not fix are drawn per simulated run
(sample_constants): base one-way delay 1-12 ms, jitter 0-30% of each hop and
256-1024 terminals from ranges wider than the real setup; the fitted
constants (client turnaround, per-decision processing and its per-request
part, proposal transfer per request and per request per ms of link delay,
lognormal spread, view-change processing, new leader's first proposal)
log-uniformly 0.5-2x around the SimConstants defaults. Phases
(sample_phase): proposal delay log-uniform 50 ms-2 s re-pinned every
30-120 s, extra link delay 10-300 ms, 40-400 s apart. The spec phases (300/600 ms, 50/150 ms) sit inside those ranges and
are used only to evaluate policies.
"""

from __future__ import annotations

import argparse
import bisect
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from prior_windows import PriorWindows, check_prior_windows

N_NODES = 4
BATCH_MAX = 256  # RequestBatchMaxCount
BATCH_INTERVAL_MS = 50.0  # RequestBatchMaxInterval
FEATURE_S = 7.0
REPLY_S = 2.0
WARMUP_S = 4.0
REWARD_S = 7.0
EPISODE_S = FEATURE_S + REPLY_S + WARMUP_S + REWARD_S
SIM_WINDOWS = PriorWindows.from_seconds(FEATURE_S, REPLY_S, WARMUP_S, REWARD_S)


@dataclass
# Fitted defaults: blind best-by-loss candidate i=251 (loss 10.399) of fit7_smartbft_blind.jsonl, 7/2/4/7 calibration (blindall, 2026-10-09).
class SimConstants:
    """Defaults are an evaluation point eyeballed from the base phase (about
    11,000 tps at 35 ms), used to pre-screen policies; training draws the
    fitted fields 0.5-2x around them and the rest from wide ranges
    (sample_constants)."""

    base_delay_ms: float = 5.0
    # netem jitter as a fraction of each hop's delay (apply_network_failures
    # DEFAULT_NETWORK_DELAY_JITTER_RATIO, 0.2 when the spec sets none).
    jitter_ratio: float = 0.2
    terminals: int = 512
    # Reply hop plus the client's own turnaround before the next request of a
    # terminal reaches the replicas.
    client_turnaround_ms: float = 9.956832
    # Leader batch assembly, verification and execution per decision.
    proc_ms: float = 3.362654
    proc_per_tx_ms: float = 0.011038
    # The 44 KB proposal takes longer to arrive than a vote, and more so on a
    # slow link (window-limited, so it scales with the RTT).
    transfer_per_tx_ms: float = 0.006563
    transfer_per_tx_per_delay_ms: float = 0.000864
    # Per-decision lognormal spread of processing and transfer.
    decision_sigma: float = 0.310334
    # ViewData collection, NewView, install; a sync when a node is behind.
    vc_proc_ms: float = 124.293797
    # The new leader's first proposal after install.
    new_view_first_ms: float = 58.417096


@dataclass
class Phase:
    start_s: float
    extra_delay_ms: float = 0.0
    proposal_delay_ms: float = 0.0
    interval_s: float = 60.0  # proposalDelay <interval>: leader re-pinned per tick


def _pct(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile, as percentileDurationMS / percentileInt."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))])


def _wpct(pairs: Sequence[Tuple[float, int]], q: float) -> float:
    """Nearest-rank percentile of (value, count) pairs."""
    if not pairs:
        return 0.0
    ordered = sorted(pairs)
    target = q * sum(w for _, w in ordered)
    cum = 0
    for v, w in ordered:
        cum += w
        if cum >= target:
            return float(v)
    return float(ordered[-1][0])


@dataclass
class _Window:
    start_s: float
    end_s: float
    txs: int = 0
    decisions: int = 0
    lat: List[Tuple[float, int]] = field(default_factory=list)
    batch: List[int] = field(default_factory=list)
    gaps: List[float] = field(default_factory=list)
    last_decision_ms: Optional[float] = None
    last_view: Optional[int] = None
    leader_changes: int = 0
    regency_changes: int = 0
    view_changes: int = 0
    timeout_ms: float = 0.0

    def file(self, event: Tuple) -> None:
        t_ms, kind, payload = event
        if kind == "vc":
            self.view_changes += 1
            return
        n, lat, view, timeout_ms = payload
        self.decisions += 1
        self.txs += n
        self.batch.append(n)
        self.lat.extend(lat)
        if self.last_decision_ms is not None:
            if t_ms > self.last_decision_ms:
                self.gaps.append(t_ms - self.last_decision_ms)
            if view != self.last_view:
                self.regency_changes += 1
                if (view - self.last_view) % N_NODES != 0:
                    self.leader_changes += 1
        self.last_decision_ms = t_ms
        self.last_view = view
        self.timeout_ms = timeout_ms

    def report(self) -> Dict[str, float]:
        elapsed = self.end_s - self.start_s
        total_lat = sum(v * w for v, w in self.lat)
        n_lat = sum(w for _, w in self.lat)
        # main._fill_empty_window_latency: the agent gives a window in which
        # nothing committed its own length as its latency (the report gives 0).
        empty_ms = elapsed * 1000.0
        return {
            "total_transactions": float(self.txs),
            "total_consensus_instances": float(self.decisions),
            "avg_consensus_latency_ms": total_lat / n_lat if n_lat else empty_ms,
            "p50_consensus_latency_ms": _wpct(self.lat, 0.50) if n_lat else empty_ms,
            "p95_consensus_latency_ms": _wpct(self.lat, 0.95) if n_lat else empty_ms,
            "throughput_tps": self.txs / elapsed if elapsed > 0 else 0.0,
            "avg_batch_size": self.txs / self.decisions if self.decisions else 0.0,
            "p95_batch_size": _pct(self.batch, 0.95),
            "leader_change_count": float(self.leader_changes),
            "regency_change_count": float(self.regency_changes),
            "timeout_ms": float(self.timeout_ms),
            "avg_inter_commit_gap_ms": float(np.mean(self.gaps)) if self.gaps else 0.0,
            "p50_inter_commit_gap_ms": _pct(self.gaps, 0.50),
            "p95_inter_commit_gap_ms": _pct(self.gaps, 0.95),
            "view_change_count": float(self.view_changes),
            "no_progress_view_change_count": 0.0,
        }


class SmartbftSim:
    """Continuous-time simulation driven one episode at a time."""

    def __init__(
        self,
        constants: SimConstants,
        phases: Sequence[Phase],
        seed: int,
        start_offset_s: float = 0.0,
        initial_timeout_ms: float = 5000.0,
    ):
        self.c = constants
        self.phases = sorted(phases, key=lambda p: p.start_s)
        self.rng = np.random.default_rng(seed)
        self.t_ms = 0.0
        self.view = 0
        self.timeout_ms = float(initial_timeout_ms)
        self.timer_restart_ms = 0.0
        self._pending_apply: Optional[Tuple[float, float]] = None
        self._pin_key: Optional[Tuple[int, int]] = None
        self._pinned_leader = 0
        # Pending requests as (arrival_ms, count) cohorts in arrival order; the
        # closed loop keeps every terminal here, some with a future arrival.
        self.pool: List[Tuple[float, int]] = []
        spread = float(self.rng.uniform(5.0, 50.0))
        for _ in range(8):
            bisect.insort(self.pool, (float(self.rng.uniform(0.0, spread)), constants.terminals // 8))
        rest = constants.terminals - 8 * (constants.terminals // 8)
        if rest:
            bisect.insort(self.pool, (spread, rest))
        self._carry: List[Tuple] = []
        self.episode_start_s = float(start_offset_s)
        self._advance_to(start_offset_s * 1000.0, None)

    @property
    def leader(self) -> int:
        return self.view % N_NODES

    def phase_at(self, t_s: float) -> Tuple[int, Phase]:
        idx = 0
        for i, p in enumerate(self.phases):
            if t_s >= p.start_s:
                idx = i
        return idx, self.phases[idx]

    def _apply_if_due(self, now_ms: float) -> None:
        """applyBaseRequestTimeout: new T, every pending timer restarted."""
        if self._pending_apply is not None and self._pending_apply[0] <= now_ms:
            at_ms, new_timeout = self._pending_apply
            self.timeout_ms = new_timeout
            self.timer_restart_ms = max(self.timer_restart_ms, at_ms)
            self._pending_apply = None

    def _proposal_delay(self, t_ms: float, ph_idx: int, ph: Phase) -> float:
        if ph.proposal_delay_ms <= 0.0:
            return 0.0
        tick = int((t_ms / 1000.0 - ph.start_s) // ph.interval_s)
        if self._pin_key != (ph_idx, tick):
            self._pin_key = (ph_idx, tick)
            self._pinned_leader = self.leader
        return ph.proposal_delay_ms if self._pinned_leader == self.leader else 0.0

    def _cut_batch(self, t_free: float) -> Tuple[float, List[Tuple[float, int]]]:
        """When the leader's batch is cut and which cohorts (arrival, count) it
        takes: full at 256, else whatever arrived by the 50 ms tick."""
        w = t_free
        while True:
            cum = 0
            t_full = None
            for arrival, count in self.pool:
                cum += count
                if cum >= BATCH_MAX:
                    t_full = max(arrival, w)
                    break
            if t_full is not None and t_full <= w + BATCH_INTERVAL_MS:
                t_cut = t_full
                break
            if self.pool[0][0] <= w + BATCH_INTERVAL_MS:
                t_cut = w + BATCH_INTERVAL_MS
                break
            w += BATCH_INTERVAL_MS
        batch: List[Tuple[float, int]] = []
        room = BATCH_MAX
        for arrival, count in self.pool:
            if arrival > t_cut or room == 0:
                break
            take = min(room, count)
            batch.append((arrival, take))
            room -= take
        return t_cut, batch

    def _deliver(self, batch: List[Tuple[float, int]], t_decide: float) -> int:
        n = 0
        for arrival, take in batch:
            head_arrival, head_count = self.pool[0]
            if head_count == take:
                self.pool.pop(0)
            else:
                self.pool[0] = (head_arrival, head_count - take)
            n += take
        c = self.c
        back = t_decide + c.client_turnaround_ms * (1.0 + 0.5 * float(self.rng.uniform(-1.0, 1.0)))
        bisect.insort(self.pool, (back, n))
        return n

    def _step(self) -> List[Tuple]:
        """Simulate one proposal from self.t_ms: a decision, or the view change
        its complaint starts. Returns timed events for the windows."""
        c = self.c
        u = self.rng.uniform
        ph_idx, ph = self.phase_at(self.t_ms / 1000.0)
        d = c.base_delay_ms + ph.extra_delay_ms
        jr = c.jitter_ratio

        def hop() -> float:
            return d * (1.0 + jr * float(u(-1.0, 1.0)))

        t_cut, batch = self._cut_batch(self.t_ms)
        self._apply_if_due(t_cut)
        n = sum(take for _, take in batch)
        held = self._proposal_delay(t_cut, ph_idx, ph)
        transfer = n * (c.transfer_per_tx_ms + c.transfer_per_tx_per_delay_ms * d)
        work = c.proc_ms + c.proc_per_tx_ms * n
        spread = float(self.rng.lognormal(0.0, c.decision_sigma))
        # The proposal is many packets, complete when the last lands, so its
        # hop sits in the upper half of the jitter band.
        t_decide = (
            t_cut + held + d * (1.0 + jr * float(u(0.0, 1.0))) + transfer * spread
            + hop() + hop() + work * spread
        )
        # The oldest pending request complains first; every follower holds it.
        deadline = max(batch[0][0], self.timer_restart_ms) + self.timeout_ms + d * jr * float(u(0.0, 1.0))
        if self._pending_apply is not None and t_cut < self._pending_apply[0] < min(deadline, t_decide):
            self._apply_if_due(self._pending_apply[0])
            deadline = self.timer_restart_ms + self.timeout_ms + d * jr * float(u(0.0, 1.0))
        events: List[Tuple] = []
        if t_decide <= deadline + hop():
            # A complaint during the commit hop comes after the followers sent
            # their commits, so the leader still decides and the batch lands.
            n = self._deliver(batch, t_decide)
            lat = [(t_decide - arrival, take) for arrival, take in batch]
            events.append((t_decide, "decision", (n, lat, self.view, self.timeout_ms)))
            if t_decide <= deadline:
                self.t_ms = t_decide
                return events
            # The ViewChange votes are out regardless: the view still changes.

        t_start = max(deadline, self.t_ms)
        backoff = 1
        while True:
            self.view += 1
            events.append((t_start, "vc", None))
            t_install = t_start + hop() + hop() + hop() + c.vc_proc_ms * float(self.rng.lognormal(0.0, c.decision_sigma))
            if t_install <= t_start + self.timeout_ms * backoff:
                break
            # view_change_timeout: sync, then start again with a longer wait.
            t_start += self.timeout_ms * backoff
            backoff += 1
        self._apply_if_due(t_install)
        self.timer_restart_ms = t_install
        self.t_ms = t_install + c.new_view_first_ms * float(self.rng.lognormal(0.0, c.decision_sigma))
        return events

    def _advance_to(self, end_ms: float, windows: Optional[Sequence[_Window]]) -> None:
        events = self._carry
        self._carry = []
        while self.t_ms < end_ms:
            events.extend(self._step())
        for ev in events:
            if ev[0] >= end_ms:
                self._carry.append(ev)
            elif windows is not None:
                t_s = ev[0] / 1000.0
                for w in windows:
                    if w.start_s <= t_s < w.end_s:
                        w.file(ev)

    def run_episode(
        self, choose: Callable[[Dict[str, float]], float]
    ) -> Tuple[Dict[str, float], float, Dict[str, float], int]:
        """Run one episode; ``choose(features) -> timeout_ms`` is called at the
        feature deadline. Returns (features, timeout, reward report, phase index
        at decision time)."""
        s = self.episode_start_s
        # Each window opens with the applied T, as learning.go sets metrics.timeout.
        wa = _Window(s, s + FEATURE_S, timeout_ms=self.timeout_ms)
        self._advance_to((s + FEATURE_S) * 1000.0, [wa])
        features = wa.report()
        # The phase in force at the apply instant, as main.py labels an episode,
        # so the onset episode carries the phase its reward window measures.
        phase_idx, _ = self.phase_at(s + FEATURE_S + REPLY_S)
        new_timeout = float(choose(features))
        # Applied at the end of the reply wait, restarting every pending timer.
        self._pending_apply = ((s + FEATURE_S + REPLY_S) * 1000.0, new_timeout)
        wb = _Window(s + FEATURE_S + REPLY_S + WARMUP_S, s + EPISODE_S, timeout_ms=new_timeout)
        self._advance_to((s + EPISODE_S) * 1000.0, [wb])
        # An apply that landed between a new view's install and its first
        # proposal is still pending here; nothing was proposed since.
        self._apply_if_due(self.t_ms)
        self.episode_start_s = s + EPISODE_S
        return features, new_timeout, wb.report(), phase_idx


# --- Randomized environments -------------------------------------------------


def _around(rng: np.random.Generator, host: float, lo: float = 0.5, hi: float = 2.0) -> float:
    """A host constant scaled by a log-uniform factor in [lo, hi]."""
    if host <= 0:
        raise ValueError(f"cannot draw around a non-positive default {host}")
    return float(host * math.exp(rng.uniform(math.log(lo), math.log(hi))))


def sample_constants(rng: np.random.Generator) -> SimConstants:
    # The fitted constants (blindall/smartbft/fit/sim_fit.py FREE) 0.5-2x
    # around the installed defaults; base delay, jitter and terminals stay wide.
    h = SimConstants()
    return SimConstants(
        base_delay_ms=float(rng.uniform(1.0, 12.0)),
        jitter_ratio=float(rng.uniform(0.0, 0.3)),
        terminals=int(rng.integers(256, 1025)),
        client_turnaround_ms=_around(rng, h.client_turnaround_ms),
        proc_ms=_around(rng, h.proc_ms),
        proc_per_tx_ms=_around(rng, h.proc_per_tx_ms),
        transfer_per_tx_ms=_around(rng, h.transfer_per_tx_ms),
        transfer_per_tx_per_delay_ms=_around(rng, h.transfer_per_tx_per_delay_ms),
        decision_sigma=_around(rng, h.decision_sigma),
        vc_proc_ms=_around(rng, h.vc_proc_ms),
        new_view_first_ms=_around(rng, h.new_view_first_ms),
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
    """config/smartbft/failure_spec.xml (seven phases, +50/+150 ms continuous):
    warm-up 10 s, phases at warm-up + atTime. Used only to evaluate policies in
    simulation, never to train."""
    w = 10.0
    return [
        Phase(0.0),
        Phase(w + 300.0, proposal_delay_ms=300.0),
        Phase(w + 600.0, proposal_delay_ms=600.0),
        Phase(w + 900.0, extra_delay_ms=50.0),
        Phase(w + 1200.0, extra_delay_ms=50.0, proposal_delay_ms=300.0),
        Phase(w + 1500.0, extra_delay_ms=150.0),
        Phase(w + 1800.0),
    ]


# --- Agent loop, data generation, prior training, evaluation -----------------


def feature_vector(report: Dict[str, float]) -> np.ndarray:
    from models import PBFT_REPORT_FEATURE_FIELD_NAMES

    return np.asarray(
        [report[name] for name in PBFT_REPORT_FEATURE_FIELD_NAMES], dtype=float
    )


def episode_reward(report: Dict[str, float]) -> float:
    """Reward of one reward window under the process-wide reward design, the
    same calculator main.reward_engineering uses; timeout_ms comes from the
    report, as it does there."""
    from reward_utils import calculate_smartbft_reward

    counts = (
        "total_transactions", "total_consensus_instances", "leader_change_count",
        "regency_change_count", "timeout_ms", "view_change_count",
        "no_progress_view_change_count",
    )
    return calculate_smartbft_reward(
        **{k: (int(v) if k in counts else float(v)) for k, v in report.items()}
    )


WARMUP_PHASE_S = 10.0  # failure_spec warmUpTime: decisions before it are phase -1
# Episode e applies at warm-up + 20(e-1), on the phase onsets, as on the engine.
SPEC_START_OFFSET_S = WARMUP_PHASE_S - FEATURE_S - REPLY_S


def run_agent(
    sim: SmartbftSim,
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


def training_sim(rng: np.random.Generator, seed: int) -> Tuple[SmartbftSim, int]:
    """One randomized training run: (simulator, episodes)."""
    from continuous_bandit import context_spec

    import agent_pb2

    lo, hi = context_spec(agent_pb2.PROTOCOL_SMARTBFT).bounds_ms
    episodes = int(rng.integers(40, 120))
    sim = SmartbftSim(
        sample_constants(rng),
        sample_schedule(rng, episodes * EPISODE_S),
        seed=seed,
        start_offset_s=float(rng.uniform(0.0, 20.0)),
        initial_timeout_ms=float(math.exp(rng.uniform(math.log(lo), math.log(hi)))),
    )
    return sim, episodes


def _generate_one(args: Tuple[int, str]) -> Tuple[np.ndarray, np.ndarray]:
    from reward_utils import set_reward_design

    seed, design = args
    set_reward_design(design)
    from continuous_bandit import ContinuousTimeoutActionCodec, context_spec, report_context
    from models import PBFT_REPORT_FEATURE_FIELD_NAMES

    import agent_pb2

    rng = np.random.default_rng(seed)
    spec = context_spec(agent_pb2.PROTOCOL_SMARTBFT)
    codec = ContinuousTimeoutActionCodec(agent_pb2.PROTOCOL_SMARTBFT, bounds_ms=spec.bounds_ms)
    sim, episodes = training_sim(rng, seed)
    prev = codec.normalize_ms(int(round(sim.timeout_ms)))
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
    arrays.update(SIM_WINDOWS.arrays())
    np.savez(out_path, **arrays)
    return rmse


import agent_pb2  # noqa: E402
from continuous_bandit import context_spec  # noqa: E402

DEFAULT_PRIOR_PATH = context_spec(agent_pb2.PROTOCOL_SMARTBFT).prior_path


def make_policy(kind: str, prior_path: str = DEFAULT_PRIOR_PATH, **kwargs):
    """(predict, update) closures around the real model classes, built as main.serve does."""
    from continuous_bandit import (
        ContinuousTimeoutActionCodec,
        SimPriorGPModel,
    )
    from latency_floor import LatencyFloor, LatencyFloorPolicy, latency_feature_indices
    from models import (
        DiscountedLinUCB,
        PBFT_REPORT_FEATURE_FIELD_NAMES as names,
        ProtocolTimeoutActionCodec,
        get_timeout_values_ms,
    )

    protocol = agent_pb2.PROTOCOL_SMARTBFT
    if kind == "heuristic":
        from model_builder import build_model

        built = build_model(protocol, "continuous_heuristic")
        model, codec = built.model, built.action_codec
    elif kind == "discrete_linucb":
        codec = ProtocolTimeoutActionCodec(protocol)
        latency_index, instances_index = latency_feature_indices(names)
        model = LatencyFloorPolicy(
            model=DiscountedLinUCB(action_dim=3, n_features=len(names), seed=42),
            timeouts_ms=get_timeout_values_ms(protocol),
            latency_index=latency_index,
            n_features=len(names),
            instances_index=instances_index,
            latency_floor=LatencyFloor(multiplier=1.0, quantile=0.5),
        )
    elif kind == "candidate":
        codec = ContinuousTimeoutActionCodec(protocol, bounds_ms=context_spec(protocol).bounds_ms)
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
        return float(timeout.smartbft.election_timeout_milliseconds)

    def update(x: np.ndarray, timeout_ms: float, reward: float) -> None:
        used = agent_pb2.Timeout(
            smartbft=agent_pb2.SmartbftTimeout(
                election_timeout_milliseconds=int(round(timeout_ms))
            )
        )
        model.update_model(x, codec.encode(used.smartbft), reward)

    return predict, update


N_SPEC_PHASES = 7
STRICT_LATENCY_PHASES = (1, 2, 4)  # runs/2026-09-29-continuous-port/spec.md, provisional


def evaluate_spec(kind: str, constants: SimConstants, seed: int, **kwargs) -> np.ndarray:
    """Per-phase (tps, mean latency, median timeout) of one policy on the spec."""
    predict, update = make_policy(kind, **kwargs)
    sim = SmartbftSim(constants, spec_schedule(), seed=seed, start_offset_s=SPEC_START_OFFSET_S)
    rows = run_agent(sim, 105, predict, update)
    out = np.zeros((N_SPEC_PHASES, 3))
    for ph in range(N_SPEC_PHASES):
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
        return agent_pb2.Timeout(smartbft=agent_pb2.SmartbftTimeout(
            election_timeout_milliseconds=int(round(timeout_ms))))

    def from_timeout(timeout: agent_pb2.Timeout) -> float:
        return float(timeout_message(timeout, "smartbft").election_timeout_milliseconds)

    return SimAdapter(
        protocol=agent_pb2.PROTOCOL_SMARTBFT,
        simulator="smartbft_sim",
        training_run=lambda seed: training_sim(np.random.default_rng(seed), seed),
        spec_run=lambda seed: (
            SmartbftSim(SimConstants(), spec_schedule(), seed=seed,
                        start_offset_s=SPEC_START_OFFSET_S), 105),
        run_agent=run_agent,
        # A training schedule's phase 0 is a real phase, so run_agent already
        # updates every decision from 10 s.
        run_training=run_agent,
        to_timeout=to_timeout,
        from_timeout=from_timeout,
        row_phase=lambda row: row[0],
        row_reward=lambda row: episode_reward(row[2]),
        spec_phase_count=N_SPEC_PHASES,
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
            constants.terminals = 512
        res = {}
        for k in ("discrete_linucb", "heuristic", "candidate"):
            set_reward_design(args.reward_design if k == "candidate" else "reward_mixed")
            res[k] = evaluate_spec(k, constants, s, **({"prior_path": args.prior} if k == "candidate" else {}))
        wins = 0
        line = []
        for ph in range(N_SPEC_PHASES):
            # Rule 3 of the port spec: strict latency where T decides it.
            bt = max(res["discrete_linucb"][ph, 0], res["heuristic"][ph, 0])
            bl = min(res["discrete_linucb"][ph, 1], res["heuristic"][ph, 1])
            ct, cl = res["candidate"][ph, 0], res["candidate"][ph, 1]
            ok = ct >= 0.95 * bt and (cl < bl if ph in STRICT_LATENCY_PHASES else cl <= 1.05 * bl)
            wins += ok
            line.append(
                f"p{ph} {ct:5.0f}/{cl:5.0f}@{res['candidate'][ph,2]:4.0f} "
                f"[{res['heuristic'][ph,0]:5.0f}/{res['heuristic'][ph,1]:5.0f}@{res['heuristic'][ph,2]:4.0f} "
                f"{res['discrete_linucb'][ph,0]:5.0f}/{res['discrete_linucb'][ph,1]:5.0f}@{res['discrete_linucb'][ph,2]:4.0f}]{'+' if ok else '-'}"
            )
        print(f"seed {s} phases_ok={wins}/{N_SPEC_PHASES} terminals={constants.terminals} | " + " | ".join(line))


if __name__ == "__main__":
    main()
