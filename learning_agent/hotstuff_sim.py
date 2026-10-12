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

Episodes follow the wall-clock learning windows in lib_rustbft.sh: 7 s
features, 2 s reply wait (applied at the deadline), 4 s warm-up, 7 s reward.
A window in which nothing committed gets its own length as its latency, as
the agent does for HotStuff (main.py, Kevin 2026-10-02).

Constants the mechanics do not fix are drawn per simulated run: the fitted
ones (processing, the degraded regime's shape, the collapse threshold, how
readily the bistable band tips) 0.5-2x around the SimConstants defaults, the
noise and the load from ranges wider than the calibration host.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import agent_pb2
from continuous_bandit import CONTEXT_SPECS
from prior_windows import PriorWindows, check_prior_windows

N_NODES = 4
MIN_BLOCK_DELAY_MS = 100.0
# config/hotstuff/parameters.json consensus.timeout_delay: in effect until the
# first decision.
DEFAULT_TIMEOUT_MS = 5000.0
FEATURE_S = 7.0
REPLY_S = 2.0
WARMUP_S = 4.0
REWARD_S = 7.0
EPISODE_S = FEATURE_S + REPLY_S + WARMUP_S + REWARD_S
SIM_WINDOWS = PriorWindows.from_seconds(FEATURE_S, REPLY_S, WARMUP_S, REWARD_S)

DEFAULT_PRIOR_PATH = CONTEXT_SPECS[agent_pb2.PROTOCOL_HOTSTUFF].prior_path


@dataclass
# Fitted defaults: blind best-by-loss candidate i=1013 (loss 1.610) of fit7_hotstuff_blind.jsonl, 7/2/4/7 calibration (blindall, 2026-10-09).
class SimConstants:
    """Defaults are the calibration host (cloudlab_1, 10,000 tx/s); training
    draws every field from sample_constants, the fitted ones around these."""

    # Per-round processing when the network sets the pace (measured 2-4 ms)
    # and in the sleep-bound regime (42.6 ms at 5 ms one-way gives 23).
    proc_net_ms: float = 2.84228
    proc_sleep_ms: float = 28.040112
    # Lognormal spread of a round's duration.
    round_sigma: float = 0.03
    load_tps: float = 10000.0
    # Degraded regime (T under p3): rounds take p3 + degraded_hops * h and
    # this share of the load commits (measured 4.0-4.2 h and 0.79-0.91 at 5+50).
    degraded_hops: float = 1.240074
    degraded_share: float = 0.60067
    # Nothing commits below collapse_hops * h (2 h degrades, 1.55 h collapses).
    collapse_hops: float = 2.061011
    # Probability that a perturbation tips a run in the bistable band into the
    # 3-hop state (the 5+150 onset did at 3.3 h, the 5+50 one did not at 3.3 h).
    tip_prob: float = 0.931279


@dataclass
class Phase:
    start_s: float
    one_way_ms: float = 5.0
    # Slow leader: hold until hold_offset_ms before the live timeout, or a
    # fixed hold_ms; neither means no slow leader.
    hold_offset_ms: Optional[float] = None
    hold_ms: Optional[float] = None


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
        return {
            "total_transactions": tx,
            "total_consensus_instances": float(committed),
            "avg_consensus_latency_ms": avg,
            "p50_consensus_latency_ms": p50,
            "p95_consensus_latency_ms": p95,
            "throughput_tps": tx / length_s,
            "avg_batch_size": batch,
            "p95_batch_size": 1.1 * batch,
            "leader_change_count": float(max(0, committed - 1)),
            "regency_change_count": 0.0,
            "timeout_ms": float(timeout_ms),
            "avg_inter_commit_gap_ms": float(gaps.mean()) if len(gaps) else 0.0,
            "p50_inter_commit_gap_ms": _rank(gaps, 0.50) if len(gaps) else 0.0,
            "p95_inter_commit_gap_ms": _rank(gaps, 0.95) if len(gaps) else 0.0,
            "view_change_count": float(round(self.timeouts)),
            "no_progress_view_change_count": float(round(self.stalled_timeouts)),
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
        self._next_round: Optional[Tuple[float, float]] = None

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
        p0 = max(2.0 * h + c.proc_net_ms, (MIN_BLOCK_DELAY_MS + h + c.proc_sleep_ms) / 3.0)
        p3 = p0 + h
        w2, w3 = p0 + h + c.proc_net_ms, p3 + h + c.proc_net_ms
        perturbed = self._last_one_way is not None and h > self._last_one_way
        self._last_one_way = h
        covering = [w for w in windows if w.start_s <= t_s < w.end_s]
        load_ms = c.load_tps / 1000.0

        if T < c.collapse_hops * h:
            # Nothing commits; every replica keeps timing out.
            n = 1000.0 / (T + 0.5 * h)
            for w in covering:
                w.timeouts += n
                w.stalled_timeouts += n
            self.locked = True
            self._clock_ms = (t_s + 1.0) * 1000.0
            self._next_round = None
            return

        hold = self._hold(ph)
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
        timeouts_per_cycle = (4.0 if degraded else (1.0 if self.locked else 0.0))
        if hold > 0.0 and not degraded:
            # Where the block-delay sleeps set the pace, the hold first uses up
            # the slack the round already spends waiting for sleeping voters
            # (measured at 5 ms: a 70 / 200 / 640 ms hold adds 42 / 165 / 601 ms).
            hold = max(0.0, hold - max(0.0, base - (2.0 * h + c.proc_net_ms)))
            if base + hold + c.proc_net_ms >= T:
                # The waiting replicas time out before the held block arrives:
                # a timeout certificate ends the round and the block is lost.
                cycle[3] = T + 2.0 * h + c.proc_net_ms
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

        # Load commits by time; rounds land in the second they end in.
        for w in covering:
            w.tx += share * load_ms * 1000.0
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
                self._next_round = (dur, latency)
            dur, latency = self._next_round
            if self._clock_ms + dur > second_end:
                break
            self._clock_ms += dur
            self._next_round = None
            self._slot += 1
            for w in covering:
                w.rounds.append(dur)
                w.latencies.append(latency)
                # A replica's share of the cycle's timeouts.
                w.timeouts += timeouts_per_cycle / N_NODES
                if degraded:
                    w.stalled_timeouts += 0.2

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


def _around(rng: np.random.Generator, host: float, lo: float = 0.5, hi: float = 2.0) -> float:
    """A host constant scaled by a log-uniform factor in [lo, hi]."""
    if host <= 0:
        raise ValueError(f"cannot draw around a non-positive default {host}")
    return float(host * math.exp(rng.uniform(math.log(lo), math.log(hi))))


def sample_constants(rng: np.random.Generator) -> SimConstants:
    # The fitted constants (sim_fit.py FREE) 0.5-2x around the defaults; the
    # share and the probability are capped at 1 by narrowing the factor.
    h = SimConstants()
    return SimConstants(
        proc_net_ms=_around(rng, h.proc_net_ms),
        proc_sleep_ms=_around(rng, h.proc_sleep_ms),
        round_sigma=float(rng.uniform(0.01, 0.08)),
        load_tps=float(rng.uniform(4000.0, 16000.0)),
        degraded_hops=_around(rng, h.degraded_hops),
        degraded_share=_around(rng, h.degraded_share, hi=min(2.0, 1.0 / h.degraded_share)),
        collapse_hops=_around(rng, h.collapse_hops),
        tip_prob=_around(rng, h.tip_prob, hi=min(2.0, 1.0 / h.tip_prob)),
    )


# train --no-proposal-faults: a prior that never saw a delayed or silent
# proposer, so its pretraining covers the network-delay regimes only. The
# "blind pretrained" arm of the ablation - see main.py --prior-path.
NO_PROPOSAL_FAULTS = False


def sample_phase(rng: np.random.Generator, start_s: float) -> Phase:
    one_way = float(math.exp(rng.uniform(math.log(2.0), math.log(250.0))))
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
    return Phase(start_s=start_s, one_way_ms=one_way, hold_offset_ms=offset, hold_ms=hold)


def sample_schedule(rng: np.random.Generator, total_s: float) -> List[Phase]:
    phases = [sample_phase(rng, 0.0)]
    t = float(rng.uniform(40.0, 400.0))
    while t < total_s:
        phases.append(sample_phase(rng, t))
        t += float(rng.uniform(40.0, 400.0))
    return phases


# config/hotstuff/failure_spec.xml: warm-up 10 s, then (atTime, one-way, hold
# offset) per 300 s phase, None for no slow leader, and the oracle's timeout
# per phase (learning_agent/config/schedule_hotstuff.csv). Used only to
# evaluate policies in simulation, never to train.
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


def spec_schedule() -> List[Phase]:
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
    lower_ms, upper_ms = CONTEXT_SPECS[agent_pb2.PROTOCOL_HOTSTUFF].bounds_ms
    episodes = int(rng.integers(40, 120))
    # A quarter of the runs start on the parameters.json default, as a real
    # run does before its first decision.
    if rng.random() < 0.25:
        initial = DEFAULT_TIMEOUT_MS
    else:
        initial = float(math.exp(rng.uniform(math.log(lower_ms), math.log(upper_ms))))
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
    arrays.update(SIM_WINDOWS.arrays())
    np.savez(out_path, **arrays)
    return rmse


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
            HotstuffSim(SimConstants(), spec_schedule(), seed=seed, start_offset_s=SPEC_START_S),
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
    t.add_argument("--runs", type=int, default=4000)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    t.add_argument("--out", default=DEFAULT_PRIOR_PATH)
    t.add_argument("--reward-design", default="reward_tps_latency")
    t.add_argument("--no-proposal-faults", action="store_true",
                   help="train on network-delay phases only; the prior never "
                        "sees a delayed or silent proposer")
    t.add_argument("--hold-episodes", type=int, default=1)
    t.add_argument("--target-floor", type=float, default=-2.0)
    t.add_argument("--hidden", default="128,128,64", help="MLP hidden layer sizes")
    e = sub.add_parser("eval", help="compare policies on the spec schedule in simulation")
    e.add_argument("--seeds", type=int, default=10)
    e.add_argument("--prior", default=DEFAULT_PRIOR_PATH)
    e.add_argument("--rungs", default="100,300,700", help="the fixed timeouts to compare")
    e.add_argument("--randomize", action="store_true", help="random constants per seed")
    e.add_argument("--reward-design", default="reward_tps_latency")
    args = parser.parse_args()
    from reward_utils import set_reward_design

    set_reward_design(args.reward_design)
    if args.cmd == "train":
        global NO_PROPOSAL_FAULTS
        NO_PROPOSAL_FAULTS = bool(args.no_proposal_faults)  # inherited by the forked workers
        x, y = generate(args.runs, args.seed, args.workers, args.reward_design, args.hold_episodes)
        rmse = train_prior(x, y, args.out, args.seed, args.reward_design, args.hold_episodes,
                           args.target_floor, tuple(int(v) for v in args.hidden.split(",")))
        print(f"samples={len(x)} hold_episodes={args.hold_episodes} target_floor={args.target_floor} "
              f"reward_std={y.std():.4f} floored_std={np.maximum(y, args.target_floor).std():.4f} "
              f"holdout_rmse={rmse:.4f} -> {args.out}")
        return

    rungs = [float(v) for v in args.rungs.split(",")]
    rng = np.random.default_rng(12345)
    for s in range(args.seeds):
        constants = sample_constants(rng) if args.randomize else SimConstants()
        if args.randomize:
            constants.load_tps = 10000.0
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
