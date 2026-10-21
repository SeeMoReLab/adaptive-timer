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
  leader every ``interval`` (60 s on the spec) of the phase, so a deposed leader's
  successor runs clean until the next tick.
- Storm mechanics follow from the above: after an install every pending
  request has deadline ``install + T``, the new leader must drain the whole
  pool (terminals / 256 batches) before then, and a view that cannot is
  deposed again after delivering what it got through. A view whose first
  decision misses ``T`` delivers nothing.

Episodes follow the wall-clock learning windows of run_smartbft.sh: 7 s
features, 2 s reply wait (apply at its end), 4 s warm-up, 7 s reward. Reports
carry the 17 SmartbftReport fields with ``learning_metrics.go``'s
definitions: latency from the replica's own submit mark to the decision, over
delivered requests; ``proposal_to_decision_ms`` the mean per decision of
proposal arrival at a follower -> decision (prepare hop, commit hop and
processing, never the hold, which precedes the arrival; 0 with no decision); throughput over the window; regency and leader changes
between consecutive decisions in the window; ``view_change_count`` one per
``start_view_change`` event at this node (retries included);
``no_progress_view_change_count`` always 0 (smallbank never records it);
``timeout_ms`` the applied T: the one in force when the window opened,
updated by each decision.

Training runs follow the shared pretraining recipe (sim_common; Kevin,
2026-10-19, restart D11). Per run (sample_constants): base one-way delay
2-10 ms, no netem jitter (JITTER_RATIO: every spec sets jitterRatio 0; drawn
0-30% until then), and the closed loop's offered load, its 512 terminals,
scaled by the shared load factor 0.5-1.5 (256-1024 until then). The eight
fitted constants (client turnaround, per-decision processing and its
per-request part, proposal transfer per request and per request per ms of
link delay, lognormal spread, view-change processing, new leader's first
proposal; sim_fit FREE) are drawn 0.5-2x around the SimConstants defaults,
and so is the unfitted turnaround spread (+-50%, fixed until then); the
initial 5-50 ms arrival spread is a start-up transient, not drawn around.
The closed loop keeps every
terminal pending, so there is no initial backlog to draw. Phases
(sample_phase) keep this protocol's mix (20% clean, 35% proposal delay, 35%
extra link delay, 10% both) with the shared magnitudes: a leader hold
log-uniform 50-1500 ms (50 ms-2 s until then), re-pinned to the current
leader every 30-120 s, 0.5-2x around the spec's 60 s (uniform 30-120 s
until then), an extra link
delay log-uniform 10-250 ms (uniform 10-300 until then); the first fault at
40-400 s and the next ones 40-400 s apart. A run's first T is
draw_initial_timeout_ms (log-uniform over the action range until then). The
spec phases are built from config/smartbft/failure_spec.xml (spec_schedule)
and used only to evaluate policies.
"""

from __future__ import annotations

import argparse
import bisect
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from prior_windows import PriorWindows, check_prior_windows
from sim_common import schedule_rows, scheduled_action
from sim_common import (
    JITTER_RATIO,
    around,
    draw_base_delay_ms,
    draw_hold_ms,
    draw_initial_timeout_ms,
    draw_link_extra_ms,
    draw_load_tps,
    fault_start_times,
)

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
# 512 B refit (2026-10-12, runs/2026-10-12-recal3): 7,500 fresh candidates (seeds 2-6) on the bl512_ blind runs (closed loop, 512 terminals,
# batch 256); blind loss 0.329 (previous defaults 0.637); real-spec check (bl512_real) loss 0.149 (defaults 0.375), regret 0.004.
# Fitted defaults: blind best-by-loss candidate i=109 (loss 0.115) of s3fit_smartbft_blind_s7.jsonl, restart S3 blind refit (seeds 7,8,9,10,11 x 1500, best of the seeds; real-spec mean regret 0.003 PASS, 2026-10-20).
class SimConstants:
    """Defaults are an evaluation point eyeballed from the base phase (about
    11,000 tps at 35 ms), used to pre-screen policies; training draws the
    fitted fields 0.5-2x around them and the rest by the shared recipe
    (sample_constants)."""

    base_delay_ms: float = 5.0
    # netem jitter as a fraction of each hop's delay: 0 on every spec (Kevin,
    # 2026-10-19, restart D11; the applier's 0.2 default until then).
    jitter_ratio: float = JITTER_RATIO
    terminals: int = 512
    # Reply hop plus the client's own turnaround before the next request of a
    # terminal reaches the replicas.
    client_turnaround_ms: float = 11.181441
    # The turnaround's spread, +- this fraction (not fitted; drawn around in
    # training, Kevin, 2026-10-19, restart D11).
    turnaround_spread: float = 0.5
    # Leader batch assembly, verification and execution per decision.
    proc_ms: float = 3.158509
    proc_per_tx_ms: float = 0.003524
    # The 44 KB proposal takes longer to arrive than a vote, and more so on a
    # slow link (window-limited, so it scales with the RTT).
    transfer_per_tx_ms: float = 0.027503
    transfer_per_tx_per_delay_ms: float = 0.000152
    # Per-decision lognormal spread of processing and transfer.
    decision_sigma: float = 0.080992
    # ViewData collection, NewView, install; a sync when a node is behind.
    vc_proc_ms: float = 103.315215
    # The new leader's first proposal after install.
    new_view_first_ms: float = 28.188503


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
    p2d: List[float] = field(default_factory=list)

    def file(self, event: Tuple) -> None:
        t_ms, kind, payload = event
        if kind == "vc":
            return  # counted at the report over the last episode (the elections of the episode before the window's end, as learning.go counts them (2026-10-19, restart D14))
        n, lat, view, timeout_ms, p2d = payload
        self.decisions += 1
        self.p2d.append(p2d)
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
            "proposal_to_decision_ms": float(np.mean(self.p2d)) if self.p2d else 0.0,
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
        # Times (ms) of every election, for the reports' counts over the last episode.
        self._vc_times: List[float] = []
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
        back = t_decide + c.client_turnaround_ms * (1.0 + c.turnaround_spread * float(self.rng.uniform(-1.0, 1.0)))
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
        # hop sits in the upper half of the jitter band. The followers verify
        # it on arrival (proposal_to_decision_ms runs from there).
        t_arrive = t_cut + held + d * (1.0 + jr * float(u(0.0, 1.0))) + transfer * spread
        t_decide = t_arrive + hop() + hop() + work * spread
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
            events.append((t_decide, "decision", (n, lat, self.view, self.timeout_ms, t_decide - t_arrive)))
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
                continue
            if ev[1] == "vc":
                self._vc_times.append(ev[0])
            if windows is not None:
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
        self._count_view_changes(wa)
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
        self._count_view_changes(wb)
        return features, new_timeout, wb.report(), phase_idx

    def _count_view_changes(self, w: "_Window") -> None:
        """The window's count: the elections in the episode ending at its end."""
        end_ms, from_ms = w.end_s * 1000.0, (w.end_s - EPISODE_S) * 1000.0
        w.view_changes = sum(1 for t in self._vc_times if from_ms < t <= end_ms)
        self._vc_times = [t for t in self._vc_times if t > from_ms]


# --- Randomized environments -------------------------------------------------


def sample_constants(rng: np.random.Generator) -> SimConstants:
    # The fitted constants (blindall/smartbft/fit/sim_fit.py FREE) 0.5-2x
    # around the installed defaults; base delay, jitter and the load by the
    # shared recipe, the closed loop's load being its terminals (Kevin,
    # 2026-10-19, restart D11).
    h = SimConstants()
    return SimConstants(
        base_delay_ms=draw_base_delay_ms(rng),
        jitter_ratio=JITTER_RATIO,
        terminals=int(round(draw_load_tps(rng, float(h.terminals)))),
        client_turnaround_ms=around(rng, h.client_turnaround_ms),
        proc_ms=around(rng, h.proc_ms),
        proc_per_tx_ms=around(rng, h.proc_per_tx_ms),
        transfer_per_tx_ms=around(rng, h.transfer_per_tx_ms),
        transfer_per_tx_per_delay_ms=around(rng, h.transfer_per_tx_per_delay_ms),
        decision_sigma=around(rng, h.decision_sigma),
        vc_proc_ms=around(rng, h.vc_proc_ms),
        new_view_first_ms=around(rng, h.new_view_first_ms),
        # Unfitted: around its value, capped so a turnaround stays >= 0.
        turnaround_spread=around(rng, h.turnaround_spread, hi=min(2.0, 1.0 / h.turnaround_spread)),
    )


# train --no-proposal-faults: a prior that never saw a delayed or silent
# proposer, so its pretraining covers the network-delay regimes only. The
# "blind pretrained" arm of the ablation - see main.py --prior-path.
NO_PROPOSAL_FAULTS = False


# The spec's hold re-pin interval (<proposalDelay><interval>), the centre of the
# training draw.
SPEC_REPIN_INTERVAL_S = 60.0


def sample_phase(rng: np.random.Generator, start_s: float) -> Phase:
    # This protocol's mix, the shared magnitudes (sim_common); the hold's
    # re-pin interval 0.5-2x around the spec's 60 s, as pbft_sim (Kevin,
    # 2026-10-19, restart D11; uniform 30-120 s until then).
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


SPEC_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "config", "smartbft", "failure_spec.xml"
)


def _spec_children(node, allowed: set, path: str, where: str) -> None:
    extra = {child.tag for child in node} - allowed
    if extra:
        raise ValueError(f"{path} {where}: {sorted(extra)} have no smartbft_sim mapping")


def spec_schedule(path: str = SPEC_PATH) -> List[Phase]:
    """The phases of config/smartbft/failure_spec.xml (Kevin, 2026-10-19,
    restart D11: built from the file, literals until then): the base phase at
    0, each later phase at warm-up + atTime with its continuous extra link
    delay and its leader's proposal hold and re-pin interval. Raises on
    anything the simulator cannot model and on a warm-up, base delay, jitter
    or phase count other than the ones evaluate_spec runs with
    (WARMUP_PHASE_S, SimConstants(), N_SPEC_PHASES). Used only to evaluate
    policies in simulation, never to train."""
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    _spec_children(root, {"schemaVersion", "warmUpTime", "globalNetworkDelay", "phases"}, path, "failureSpec")
    warm = float(root.findtext("warmUpTime"))
    base = float(root.findtext("globalNetworkDelay/delayMs"))
    jitter = float(root.findtext("globalNetworkDelay/jitterRatio"))
    host = SimConstants()
    if (warm, base, jitter) != (WARMUP_PHASE_S, host.base_delay_ms, host.jitter_ratio):
        raise ValueError(
            f"{path}: warm-up {warm:g} s, base {base:g} ms, jitter {jitter:g}; the simulator "
            f"evaluates at {WARMUP_PHASE_S:g} s, {host.base_delay_ms:g} ms, {host.jitter_ratio:g}"
        )
    phases: List[Phase] = []
    for i, ph in enumerate(root.findall("phases/phase")):
        where = f"phase {i}"
        _spec_children(ph, {"atTime", "network", "pbft", "agent"}, path, where)
        for k in ("dirtyReport", "ignoreReport"):
            if ph.findall(f"agent/{k}/replicas/*"):
                raise ValueError(f"{path} {where}: agent {k} has no smartbft_sim mapping")
        net = ph.find("network")
        extra = 0.0
        if net is not None:
            _spec_children(net, {"directedEdges", "global", "sources", "edges"}, path, f"{where}/network")
            if net.findall("sources/*") or net.findall("edges/*"):
                raise ValueError(f"{path} {where}: per-source or per-edge delay has no smartbft_sim mapping")
            delay = float(net.findtext("global/delayMs") or 0.0)
            dur_text = net.findtext("global/burstDurationMs")
            interval_text = net.findtext("global/burstIntervalMs")
            if delay > 0.0 and net.findtext("global/probability") is not None:
                raise ValueError(f"{path} {where}: a probabilistic delay has no smartbft_sim mapping")
            if delay > 0.0 and dur_text is None and interval_text is None:
                # No burst fields: the applier's legacy continuous delay.
                extra = delay
            elif delay > 0.0:
                if dur_text is None or interval_text is None:
                    raise ValueError(f"{path} {where}: burstDurationMs and burstIntervalMs go together")
                dur, interval = float(dur_text), float(interval_text)
                if dur > 0.0:
                    if not 0.0 < interval <= dur:
                        raise ValueError(f"{path} {where}: bursty delay has no smartbft_sim mapping")
                    extra = delay
        hold, repin_s = 0.0, Phase(0.0).interval_s
        if ph.find("pbft") is not None:
            _spec_children(ph.find("pbft"), {"proposalDelay"}, path, f"{where}/pbft")
            replicas = ph.findall("pbft/proposalDelay/replicas/replica")
            if len(replicas) > 1 or any((r.findtext("id") or "").strip() != "leader" for r in replicas):
                raise ValueError(f"{path} {where}: the simulator holds only the leader's proposals")
            if replicas:
                hold = float(replicas[0].findtext("delayMs"))
                repin_s = float(ph.findtext("pbft/proposalDelay/interval") or math.inf)
        at_s = float(ph.findtext("atTime"))
        if i == 0 and (at_s != 0.0 or extra or hold):
            raise ValueError(f"{path}: the first phase must be a fault-free base phase at atTime 0")
        phases.append(Phase(0.0 if i == 0 else warm + at_s, extra_delay_ms=extra,
                            proposal_delay_ms=hold, interval_s=repin_s))
    if len(phases) != N_SPEC_PHASES:
        raise ValueError(f"{path}: {len(phases)} phases, evaluate_spec reports {N_SPEC_PHASES}")
    return phases


# --- Agent loop, data generation, prior training, evaluation -----------------


def feature_vector(report: Dict[str, float]) -> np.ndarray:
    from models import SMARTBFT_REPORT_FEATURE_FIELD_NAMES

    return np.asarray(
        [report[name] for name in SMARTBFT_REPORT_FEATURE_FIELD_NAMES], dtype=float
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
        **{k: (int(v) if k in counts else float(v)) for k, v in report.items() if k != "proposal_to_decision_ms"}
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
    import agent_pb2

    episodes = int(rng.integers(40, 120))
    sim = SmartbftSim(
        sample_constants(rng),
        sample_schedule(rng, episodes * EPISODE_S),
        seed=seed,
        start_offset_s=float(rng.uniform(0.0, 20.0)),
        initial_timeout_ms=draw_initial_timeout_ms(rng, agent_pb2.PROTOCOL_SMARTBFT),
    )
    return sim, episodes


def _generate_one(args: Tuple[int, str, int]) -> Tuple[np.ndarray, np.ndarray]:
    from reward_utils import set_reward_design

    seed, design, hold = args
    set_reward_design(design)
    from continuous_bandit import ContinuousTimeoutActionCodec, context_spec, report_context
    from models import SMARTBFT_REPORT_FEATURE_FIELD_NAMES

    import agent_pb2

    rng = np.random.default_rng(seed)
    spec = context_spec(agent_pb2.PROTOCOL_SMARTBFT)
    codec = ContinuousTimeoutActionCodec(agent_pb2.PROTOCOL_SMARTBFT, bounds_ms=spec.bounds_ms)
    sim, episodes = training_sim(rng, seed)
    prev = codec.normalize_ms(int(round(sim.timeout_ms)))
    xs, ys = [], []
    # hold h: an action is drawn at the start of a hold and kept for h episodes;
    # the sample is (context at the hold's start, action, mean reward over the
    # hold), the held-action return SimReturnGPModel's prior predicts (as
    # tendermint_sim; Kevin, 2026-10-19, restart D11: h = 3 here, the leader
    # re-pinned every 60 s). h = 1 is a fresh draw every episode.
    held: Dict[str, float] = {}
    rewards: List[float] = []
    for _ in range(episodes):

        def choose(report):
            x = feature_vector(report)
            ctx = report_context(spec, SMARTBFT_REPORT_FEATURE_FIELD_NAMES, x)
            if not held:
                if rng.random() < 0.6:
                    a = float(rng.random())
                else:
                    a = float(np.clip(prev + rng.normal(0.0, 0.08), 0.0, 1.0))
                ms = codec._denormalize_value(a, 0)
                held["ctx"] = ctx
                held["a"] = codec.normalize_ms(ms)
                held["ms"] = float(ms)
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


def generate(n_runs: int, seed: int, workers: int, design: str, hold: int) -> Tuple[np.ndarray, np.ndarray]:
    from multiprocessing import Pool

    if int(hold) < 1:
        raise ValueError(f"hold must be >= 1 episode, got {hold}")
    seeds = [(seed + i, design, int(hold)) for i in range(n_runs)]
    with Pool(workers) as pool:
        parts = pool.map(_generate_one, seeds, chunksize=8)
    return np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


import agent_pb2  # noqa: E402
from continuous_bandit import context_spec  # noqa: E402
from continuous_bandit import opening_action_ms, train_gp_prior  # noqa: E402

from reward_utils import DEFAULT_REWARD_DESIGN  # noqa: E402

DEFAULT_PRIOR_PATH = context_spec(agent_pb2.PROTOCOL_SMARTBFT).prior_path


def make_policy(kind: str, prior_path: str = DEFAULT_PRIOR_PATH, **kwargs):
    """(predict, update) closures around the real model classes, built as main.serve does."""
    from continuous_bandit import (
        ContinuousTimeoutActionCodec,
        SimReturnGPModel,
    )
    from latency_floor import LatencyFloor, LatencyFloorPolicy, latency_feature_indices
    from models import (
        DiscountedLinUCB,
        SMARTBFT_REPORT_FEATURE_FIELD_NAMES as names,
        ProtocolTimeoutActionCodec,
        get_timeout_values_ms,
    )

    protocol = agent_pb2.PROTOCOL_SMARTBFT
    if kind == "heuristic":
        from model_builder import build_model

        built = build_model(protocol, "continuous_heuristic", prior_windows=SIM_WINDOWS)
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
        # The live agent's GP (model_builder; SimPriorGPModel is unwired), which
        # reads the prior's hold and floor (Kevin, 2026-10-19, restart D11).
        model = SimReturnGPModel(
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
    episodes); the spec phase is run_agent's, and the oracle plays
    learning_agent/config/schedule_smartbft.csv as ScheduledTimeoutModel does."""
    from pretrain import SimAdapter, timeout_message

    def to_timeout(timeout_ms: float) -> agent_pb2.Timeout:
        return agent_pb2.Timeout(smartbft=agent_pb2.SmartbftTimeout(
            election_timeout_milliseconds=int(round(timeout_ms))))

    def from_timeout(timeout: agent_pb2.Timeout) -> float:
        return float(timeout_message(timeout, "smartbft").election_timeout_milliseconds)

    rows = schedule_rows("smartbft")

    def oracle(sim) -> Callable[[np.ndarray], float]:
        # The apply time: the decision at the feature deadline reaches the
        # replicas a reply window later (Kevin, 2026-10-19, restart: the sim gate
        # compares against the oracle on all six).
        def predict(x: np.ndarray) -> float:
            return scheduled_action(rows, sim.episode_start_s + FEATURE_S + REPLY_S, WARMUP_PHASE_S)[0]

        return predict

    return SimAdapter(
        protocol=agent_pb2.PROTOCOL_SMARTBFT,
        simulator="smartbft_sim",
        training_run=lambda seed: training_sim(np.random.default_rng(seed), seed),
        spec_run=lambda seed: (
            # The engine opens at the middle rung (--initial-timeout in the rows, D10b).
            SmartbftSim(SimConstants(), spec_schedule(), seed=seed, start_offset_s=SPEC_START_OFFSET_S,
                        initial_timeout_ms=float(opening_action_ms(agent_pb2.PROTOCOL_SMARTBFT)[0])), 105),
        run_agent=run_agent,
        # A training schedule's phase 0 is a real phase, so run_agent already
        # updates every decision from 10 s.
        run_training=run_agent,
        to_timeout=to_timeout,
        from_timeout=from_timeout,
        row_phase=lambda row: row[0],
        row_reward=lambda row: episode_reward(row[2]),
        spec_phase_count=N_SPEC_PHASES,
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
    t.add_argument("--hold-episodes", type=int, default=3,
                   help="hold each drawn action this many episodes and fit the mean reward over "
                        "the hold; default 3, the leader's 60 s re-pin (restart D11)")
    t.add_argument("--target-floor", type=float, default=-2.0,
                   help="fit rewards below this value as this value "
                        "(SimReturnGPModel applies the same floor online, and "
                        "refuses a prior fit to the raw reward); default -2, as "
                        "the pretraining recipe passes")
    t.add_argument("--no-proposal-faults", action="store_true",
                   help="train on network-delay phases only; the prior never "
                        "sees a delayed or silent proposer")
    e = sub.add_parser("eval", help="compare policies on the spec schedule in simulation")
    e.add_argument("--seeds", type=int, default=10)
    e.add_argument("--prior", default=DEFAULT_PRIOR_PATH)
    e.add_argument("--randomize", action="store_true", help="random constants per seed")
    e.add_argument("--reward-design", default=DEFAULT_REWARD_DESIGN,
                   help=f"reward the candidate learns from; the baselines keep {DEFAULT_REWARD_DESIGN}")
    args = parser.parse_args()
    from reward_utils import set_reward_design

    if args.cmd == "train":
        global NO_PROPOSAL_FAULTS
        NO_PROPOSAL_FAULTS = bool(args.no_proposal_faults)  # inherited by the forked workers
        x, y = generate(args.runs, args.seed, args.workers, args.reward_design, args.hold_episodes)
        rmse = train_gp_prior(
            agent_pb2.PROTOCOL_SMARTBFT, x, y, args.out, seed=args.seed, design=args.reward_design,
            hold=args.hold_episodes, target_floor=args.target_floor, windows=SIM_WINDOWS,
            sim_constants=asdict(SimConstants()), code_hash=args.code_hash)
        print(f"samples={len(x)} hold_episodes={args.hold_episodes} target_floor={args.target_floor} "
              f"reward_std={y.std():.4f} holdout_rmse={rmse:.4f} -> {args.out}")
        return

    rng = np.random.default_rng(12345)
    for s in range(args.seeds):
        constants = sample_constants(rng) if args.randomize else SimConstants()
        if args.randomize:
            constants.terminals = 512
        res = {}
        for k in ("discrete_linucb", "heuristic", "candidate"):
            set_reward_design(args.reward_design if k == "candidate" else DEFAULT_REWARD_DESIGN)
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
