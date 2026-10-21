"""Pretrain and evaluate the agent's models in the protocol simulators.

Every model is built by model_builder.build_model exactly as main.serve builds
it, and driven the way LearningAgentServicer drives it: predict from the
feature report, decode through the model's codec, apply, then encode the
applied timeout and update from the reward window (an encode that raises
ValueError skips the update, as main does). The simulators are reached
through a SimAdapter, which each <protocol>_sim.pretrain_adapter() returns.

    python pretrain.py <protocol> fit  --model-type discrete_linucb|continuous_linucb ...
    python pretrain.py <protocol> ppo  --model-type discrete_ppo|continuous_ppo ...
    python pretrain.py <protocol> eval --model-type <any but scheduled> ...

fit collects (features, encoded action, reward) under a random behaviour
policy over the model's own action space and saves the ridge fit as the
LinUCB prior mean. ppo runs one PPO model on-policy through randomized
training runs and saves its weights. eval runs a model online on the spec,
beside the oracle (where the simulator has one), the fixed rungs and, with
--compare-untrained, the same model without its prior. The GP prior (one
file for continuous_gp and discrete_gp) is trained by each simulator's own
train command, not here. Every prior records the simulator's windows
(SimAdapter.windows, prior_windows), and eval refuses a prior trained under
other windows.

Training runs are the randomized runs each simulator's GP generator draws;
an episode is updated on when its decision is at or after the simulator's
10 s warm-up, on every simulator alike. Simulations belong on CloudLab
(scripts/cloudlab/tmp/cal/pt/remote_sim.sh), not on a laptop.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import multiprocessing
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import agent_pb2
from latency_floor import LatencyFloorPolicy
from model_builder import BuiltModel, build_model, save_model_prior
from prior_windows import PriorWindows
from continuous_bandit import (
    make_latency_floor,
)
from models import TensorModel, action_index_to_timeout, get_action_space_size, set_pbft_timeout_ladder
from reward_utils import DEFAULT_REWARD_DESIGN, REWARD_DESIGNS, set_reward_design

PROTOCOL_SIMULATORS: Dict[str, str] = {
    "pbft": "pbft_sim",
    "sbft": "sbft_sim",
    "smartbft": "smartbft_sim",
    "tendermint": "tendermint_sim",
    "hotstuff": "hotstuff_sim",
    "autobahn": "autobahn_sim",
}

FIT_MODEL_TYPES = ("discrete_linucb", "continuous_linucb")
PPO_MODEL_TYPES = ("discrete_ppo", "continuous_ppo")
EVAL_MODEL_TYPES = (
    "discrete_random",
    "discrete_pavlov",
    "discrete_linear",
    "discrete_tree_ts",
    "discrete_tree_blocks",
    "discrete_linucb",
    "discrete_ppo",
    "discrete_gp",
    "continuous_gp",
    "continuous_linucb",
    "continuous_heuristic",
    "continuous_ppo",
)

# Behaviour policy of fit, as the simulators' GP generators draw theirs: a
# fresh uniform draw with this probability, otherwise the previous action
# (discrete) or a local walk of this width in log-position (continuous).
EXPLORE_PROB = 0.6
WALK_SIGMA = 0.08
# Seed-sequence key of the behaviour policy's stream, apart from the
# simulator's own draws on the run seed.
BEHAVIOUR_STREAM = 0x70726574

Predict = Callable[[np.ndarray], Any]
Update = Callable[[np.ndarray, Any, float], None]


@dataclass(frozen=True)
class SimAdapter:
    """One simulator, as the driver needs it.

    A sim action is what the simulator's run_agent passes around: the timeout
    in ms as a float, or (T, W) in ms on SBFT. Rows are run_agent's rows.
    """

    protocol: int
    simulator: str
    # seed -> (sim, episodes), built exactly as the simulator's GP generator
    # builds a randomized run.
    training_run: Callable[[int], Tuple[Any, int]]
    # seed -> (sim, episodes), built exactly as its spec evaluation builds one.
    spec_run: Callable[[int], Tuple[Any, int]]
    # The simulator's run_agent(sim, episodes, predict, update) on a spec run.
    run_agent: Callable[[Any, int, Predict, Optional[Update]], List[tuple]]
    # The same on a training run: updates every decision at or after 10 s.
    run_training: Callable[[Any, int, Predict, Optional[Update]], List[tuple]]
    to_timeout: Callable[[Any], agent_pb2.Timeout]
    from_timeout: Callable[[agent_pb2.Timeout], Any]
    # Spec phase of a spec-run row (0-based), -1 for a warm-up row.
    row_phase: Callable[[tuple], int]
    # The agent's reward for the row, from the simulator's episode_reward.
    row_reward: Callable[[tuple], float]
    spec_phase_count: int
    # sim -> predict callable playing the spec's oracle schedule, or None.
    oracle: Optional[Callable[[Any], Predict]]

    @property
    def windows(self) -> PriorWindows:
        """The simulator's windows, which its priors record and are checked against."""
        module = importlib.import_module(self.simulator)
        return PriorWindows.from_seconds(
            module.FEATURE_S, module.REPLY_S, module.WARMUP_S, module.REWARD_S
        )


def timeout_message(timeout: agent_pb2.Timeout, field: str) -> Any:
    """The ``field`` message of ``timeout``; any other protocol's is an error."""
    which = timeout.WhichOneof("value")
    if which != field:
        raise ValueError(f"expected a {field} timeout, got {which}")
    return getattr(timeout, field)


def load_adapter(protocol_name: str, peer_spread: float = 0.0) -> SimAdapter:
    if protocol_name not in PROTOCOL_SIMULATORS:
        raise ValueError(
            f"unknown protocol {protocol_name!r}; expected one of "
            f"{', '.join(PROTOCOL_SIMULATORS)}"
        )
    module = importlib.import_module(PROTOCOL_SIMULATORS[protocol_name])
    if protocol_name == "tendermint":
        return module.pretrain_adapter(peer_spread=peer_spread)
    if peer_spread != 0.0:
        raise ValueError(f"--peer-spread is Tendermint's; {protocol_name} has no peer factors")
    return module.pretrain_adapter()


def _unwrap(model: TensorModel) -> TensorModel:
    """The learner under a discrete model's latency-floor wrapper."""
    return model.model if isinstance(model, LatencyFloorPolicy) else model


def _inner(timeout: agent_pb2.Timeout) -> Any:
    # main encodes the per-protocol message the report carries, not the wrapper.
    return getattr(timeout, timeout.WhichOneof("value"))


class ModelDriver:
    """predict/update for run_agent around a BuiltModel, as the agent drives it."""

    def __init__(self, built: BuiltModel, adapter: SimAdapter) -> None:
        self.built = built
        self.adapter = adapter
        self.updates = 0
        self.skipped = 0
        self.reward_sum = 0.0

    def predict(self, x: np.ndarray) -> Any:
        action = self.built.model.predict_action(x)
        return self.adapter.from_timeout(self.built.action_codec.decode(action))

    def update(self, x: np.ndarray, applied: Any, reward: float) -> None:
        try:
            action = self.built.action_codec.encode(_inner(self.adapter.to_timeout(applied)))
        except ValueError:
            # main._apply_reward_payload_locked skips the reward the same way.
            self.skipped += 1
            return
        self.built.model.update_model(x, action, reward)
        self.updates += 1
        self.reward_sum += float(reward)


@dataclass(frozen=True)
class ModelSettings:
    """build_model's settings beside the protocol and model type."""

    heuristic_multiplier: Optional[float] = None
    heuristic_window: Optional[int] = None
    heuristic_quantile: Optional[float] = None
    # The protocol's floor scale, as main's --latency-floor-multiplier (0 disables it).
    floor_multiplier: float = 0.0
    linucb_discount: Optional[float] = None
    linucb_exploration: Optional[float] = None
    ppo_exploration: Optional[float] = None

    def build(self, protocol: int, model_type: str, prior_path: Optional[str],
              windows: PriorWindows) -> BuiltModel:
        # A LatencyFloor keeps its own history, so each model gets a new one; the same
        # floor as the live agent (continuous_bandit.make_latency_floor).
        floor = (
            None
            if self.floor_multiplier <= 0.0
            else make_latency_floor(protocol, self.floor_multiplier)
        )
        return build_model(
            protocol,
            model_type,
            prior_path=prior_path,
            prior_windows=windows,
            latency_floor=floor,
            heuristic_multiplier=self.heuristic_multiplier,
            heuristic_window=self.heuristic_window,
            heuristic_quantile=self.heuristic_quantile,
            linucb_discount=self.linucb_discount,
            linucb_exploration=self.linucb_exploration,
            ppo_exploration=self.ppo_exploration,
        )


def _override_provenance(args: argparse.Namespace) -> Dict[str, str]:
    """The simulator constants the runs were drawn around (as the GP priors record them,
    continuous_bandit.train_gp_prior), and the PBFT overrides only when given."""
    module = importlib.import_module(PROTOCOL_SIMULATORS[args.protocol])
    constants = module.host_constants() if args.protocol == "pbft" else module.SimConstants()
    out = {"sim_constants": json.dumps(asdict(constants), sort_keys=True, default=float)}
    out.update({k: v for k, v in (("timeout_ladder", args.timeout_ladder),
                                  ("sim_constants_path", args.sim_constants)) if v is not None})
    return out


def _worker_init() -> None:
    # One BLAS thread per process: the GP's linear algebra otherwise starts one
    # thread per core in every worker (main.serve limits it the same way).
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=1, user_api="blas")


def _map(fn: Callable, jobs: Sequence, workers: int, chunksize: int = 1) -> List:
    """map over ``jobs``, in forked workers when ``workers`` > 1. Fork, so the
    workers inherit module state set before the pool starts."""
    if workers <= 1:
        _worker_init()
        return [fn(job) for job in jobs]
    with multiprocessing.get_context("fork").Pool(workers, initializer=_worker_init) as pool:
        return pool.map(fn, jobs, chunksize=chunksize)


# --- fit: LinUCB prior mean -----------------------------------------------------


class _Behaviour:
    """fit's behaviour policy over the model's own action space."""

    def __init__(self, built: BuiltModel, adapter: SimAdapter, model_type: str,
                 rng: np.random.Generator) -> None:
        self._built = built
        self._adapter = adapter
        self._rng = rng
        self._discrete = model_type.startswith("discrete_")
        self._arms = get_action_space_size(adapter.protocol)
        model = _unwrap(built.model)
        # Per action dimension (lower, upper) ms: the timeout, and on SBFT the wait.
        self._ranges: List[Tuple[int, int]] = []
        if not self._discrete:
            self._ranges.append((model._lower_ms, model._upper_ms))
            if getattr(model, "_two_d", False):
                self._ranges.append((model._wait_lower_ms, model._wait_upper_ms))
        self._prev: Optional[Any] = None

    def __call__(self, x: np.ndarray) -> Any:
        codec = self._built.action_codec
        explore = self._prev is None or self._rng.random() < EXPLORE_PROB
        if self._discrete:
            arm = int(self._rng.integers(self._arms)) if explore else int(self._prev)
            self._prev = arm
            return self._adapter.from_timeout(codec.decode(np.asarray(float(arm))))
        # A log-space position per dimension: a fresh draw, or a step from the last.
        if explore:
            positions = self._rng.random(len(self._ranges))
        else:
            positions = np.clip(
                self._prev + self._rng.normal(0.0, WALK_SIGMA, len(self._ranges)), 0.0, 1.0
            )
        self._prev = positions
        normalized = []
        for d, (lower_ms, upper_ms) in enumerate(self._ranges):
            log_lower = math.log(float(lower_ms))
            value_ms = int(round(math.exp(log_lower + positions[d] * (math.log(float(upper_ms)) - log_lower))))
            normalized.append(codec.normalize_ms(min(upper_ms, max(lower_ms, value_ms)), d))
        return self._adapter.from_timeout(codec.decode(np.asarray(normalized)))


def _fit_job(args: tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    protocol_name, peer_spread, model_type, settings, design, seed = args
    set_reward_design(design)
    started = time.perf_counter()
    adapter = load_adapter(protocol_name, peer_spread)
    built = settings.build(adapter.protocol, model_type, None, adapter.windows)
    behaviour = _Behaviour(built, adapter, model_type,
                           np.random.default_rng([seed, BEHAVIOUR_STREAM]))
    xs: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    skipped = 0

    def record(x: np.ndarray, applied: Any, reward: float) -> None:
        nonlocal skipped
        try:
            action = built.action_codec.encode(_inner(adapter.to_timeout(applied)))
        except ValueError:
            skipped += 1
            return
        xs.append(np.asarray(x, dtype=float))
        # A discrete model's arm index, or every dimension of a continuous action
        # (the timeout, and on SBFT the wait).
        values = np.asarray(action, dtype=float).reshape(-1)
        actions.append(values[:1] if model_type.startswith("discrete_") else values)
        rewards.append(float(reward))

    sim, episodes = adapter.training_run(seed)
    adapter.run_training(sim, episodes, behaviour, record)
    width = len(xs[0]) if xs else 0
    action_width = len(actions[0]) if actions else 1
    return (
        np.asarray(xs, dtype=float).reshape(len(xs), width),
        np.asarray(actions, dtype=float).reshape(len(actions), action_width),
        np.asarray(rewards, dtype=float),
        skipped,
        time.perf_counter() - started,
    )


def cmd_fit(args: argparse.Namespace, settings: ModelSettings) -> None:
    adapter = load_adapter(args.protocol, args.peer_spread)
    jobs = [
        (args.protocol, args.peer_spread, args.model_type, settings, args.reward_design,
         args.seed + i)
        for i in range(args.runs)
    ]
    started = time.perf_counter()
    parts = _map(_fit_job, jobs, args.workers, chunksize=4)
    wall = time.perf_counter() - started
    parts = [p for p in parts if len(p[2])]
    if not parts:
        raise ValueError(f"{args.runs} runs produced no samples")
    x = np.vstack([p[0] for p in parts])
    a = np.vstack([p[1] for p in parts])
    if a.shape[1] == 1:
        a = a.reshape(-1)
    r = np.concatenate([p[2] for p in parts])
    skipped = sum(p[3] for p in parts)
    run_s = [p[4] for p in parts]

    built = settings.build(adapter.protocol, args.model_type, None, adapter.windows)
    model = _unwrap(built.model)
    theta = model.fit_prior_theta(x, a, r)
    save_model_prior(
        args.out, args.model_type, adapter.protocol, model, {"theta": theta},
        windows=adapter.windows, simulator=adapter.simulator, runs=args.runs, seed=args.seed,
        samples=len(r), skipped=skipped, peer_spread=args.peer_spread, code_hash=args.code_hash,
        **_override_provenance(args),
    )
    # The checkpoint must load into the model main builds for it.
    settings.build(adapter.protocol, args.model_type, args.out, adapter.windows)
    print(
        f"{adapter.simulator} fit {args.model_type}: runs={args.runs} samples={len(r)} "
        f"skipped={skipped} reward mean={r.mean():+.4f} std={r.std():.4f} "
        f"theta {theta.shape} |theta|max={np.abs(theta).max():.4f}"
    )
    print(
        f"timing: wall {wall:.1f}s with {args.workers} workers, {args.runs / wall:.2f} runs/s, "
        f"{len(r) / wall:.0f} samples/s; per run {np.mean(run_s):.3f}s mean "
        f"{np.max(run_s):.3f}s max single-process -> {args.out}"
    )


# --- eval: online on the spec -------------------------------------------------------


def _spec_rows(adapter: SimAdapter, seed: int, policy: str, settings: ModelSettings,
               model_type: Optional[str], prior_path: Optional[str],
               built: Optional[BuiltModel] = None) -> Tuple[List[tuple], int]:
    """run_agent rows of one policy on spec_run(seed), and its skipped updates.
    ``policy`` is "oracle", "rung:<arm index>" or "model"."""
    sim, episodes = adapter.spec_run(seed)
    if policy == "oracle":
        if adapter.oracle is None:
            raise ValueError(f"{adapter.simulator} has no oracle")
        return adapter.run_agent(sim, episodes, adapter.oracle(sim), None), 0
    if policy.startswith("rung:"):
        timeout = action_index_to_timeout(adapter.protocol, int(policy.split(":", 1)[1]))
        applied = adapter.from_timeout(timeout)
        return adapter.run_agent(sim, episodes, lambda x: applied, None), 0
    if policy != "model":
        raise ValueError(f"unknown policy {policy!r}")
    if built is None:
        built = settings.build(adapter.protocol, model_type, prior_path, adapter.windows)
    driver = ModelDriver(built, adapter)
    rows = adapter.run_agent(sim, episodes, driver.predict, driver.update)
    return rows, driver.skipped


def _median_action(actions: Sequence[Any]) -> Any:
    values = np.asarray(actions, dtype=float)
    return np.median(values, axis=0)


def phase_summary(adapter: SimAdapter, rows: List[tuple]) -> Dict[str, np.ndarray]:
    """Per spec phase: mean reward after the phase's first (switch) episode,
    mean over the whole phase, median action after the switch; and the run
    mean over every non-warm-up row."""
    n = adapter.spec_phase_count
    by_phase: List[List[tuple]] = [[] for _ in range(n)]
    for row in rows:
        phase = adapter.row_phase(row)
        if phase < 0:
            continue
        if phase >= n:
            raise ValueError(f"row phase {phase} outside the spec's {n} phases")
        by_phase[phase].append(row)
    for phase, sel in enumerate(by_phase):
        if len(sel) < 2:
            raise ValueError(f"spec phase {phase} has {len(sel)} episodes; need 2 or more")
    rewards = [[adapter.row_reward(r) for r in sel] for sel in by_phase]
    return {
        "after_switch": np.asarray([np.mean(r[1:]) for r in rewards]),
        "whole": np.asarray([np.mean(r) for r in rewards]),
        "action": np.asarray([_median_action([row[1] for row in sel[1:]]) for sel in by_phase]),
        "run": np.asarray(np.mean([v for r in rewards for v in r])),
    }


def _eval_job(args: tuple) -> Tuple[Dict[str, np.ndarray], int, float]:
    protocol_name, design, seed, policy, settings, model_type, prior_path = args
    set_reward_design(design)
    started = time.perf_counter()
    adapter = load_adapter(protocol_name)
    rows, skipped = _spec_rows(adapter, seed, policy, settings, model_type, prior_path)
    return phase_summary(adapter, rows), skipped, time.perf_counter() - started


def _action_label(action: np.ndarray) -> str:
    values = np.atleast_1d(action)
    return ",".join(f"{v:.0f}" for v in values)


def _summary_line(adapter: SimAdapter, label: str, summary: Dict[str, np.ndarray]) -> str:
    phases = " ".join(
        f"p{p} {summary['after_switch'][p]:+.3f}/{summary['whole'][p]:+.3f}"
        f"@{_action_label(summary['action'][p])}"
        for p in range(adapter.spec_phase_count)
    )
    return f"{label:<34} {phases} | run {float(summary['run']):+.3f}"


def _mean_summary(summaries: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return {key: np.mean([s[key] for s in summaries], axis=0) for key in summaries[0]}


def cmd_eval(args: argparse.Namespace, settings: ModelSettings) -> None:
    adapter = load_adapter(args.protocol)
    if args.compare_untrained and args.prior_path is None:
        raise ValueError("--compare-untrained needs --prior-path; without one the model "
                         "is untrained")
    # Built once here so a bad prior or setting fails before any simulation.
    for line in settings.build(adapter.protocol, args.model_type, args.prior_path,
                               adapter.windows).description:
        print(line)

    policies: List[Tuple[str, str, Optional[str]]] = []
    if adapter.oracle is not None:
        policies.append(("oracle", "oracle", None))
    for arm in range(get_action_space_size(adapter.protocol)):
        applied = adapter.from_timeout(action_index_to_timeout(adapter.protocol, arm))
        label = f"rung {_action_label(np.asarray(applied, dtype=float))}"
        policies.append((label, f"rung:{arm}", None))
    model_label = args.model_type
    if args.compare_untrained:
        policies.append((f"{args.model_type} untrained", "model", None))
        model_label = f"{args.model_type} pretrained"
    policies.append((model_label, "model", args.prior_path))

    jobs = [
        (args.protocol, args.reward_design, seed, policy, settings,
         args.model_type, prior if policy == "model" else None)
        for _, policy, prior in policies
        for seed in range(args.seeds)
    ]
    started = time.perf_counter()
    results = _map(_eval_job, jobs, args.workers)
    wall = time.perf_counter() - started

    print(
        f"{adapter.simulator} eval {args.model_type} prior={args.prior_path} seeds={args.seeds} "
        f"reward={args.reward_design}: per phase reward after switch/whole @median action, run mean"
    )
    means: Dict[str, Dict[str, np.ndarray]] = {}
    for index, (label, policy, _) in enumerate(policies):
        chunk = results[index * args.seeds:(index + 1) * args.seeds]
        if args.per_seed:
            for seed, (summary, skipped, _) in enumerate(chunk):
                suffix = f" skipped={skipped}" if skipped else ""
                print(_summary_line(adapter, f"{label} s{seed}", summary) + suffix)
        means[label] = _mean_summary([c[0] for c in chunk])
        skipped = sum(c[1] for c in chunk)
        suffix = f" skipped={skipped}" if skipped else ""
        print(_summary_line(adapter, f"{label} mean", means[label]) + suffix)
    if args.json_out:
        record = {
            "protocol": args.protocol, "model_type": args.model_type, "prior_path": args.prior_path,
            "seeds": args.seeds, "reward_design": args.reward_design,
            "floor_multiplier": settings.floor_multiplier, "policies": {},
        }
        for index, (label, _, _) in enumerate(policies):
            chunk = results[index * args.seeds:(index + 1) * args.seeds]
            record["policies"][label] = [
                {key: np.asarray(value).tolist() for key, value in c[0].items()} | {"skipped": c[1]}
                for c in chunk
            ]
        with open(args.json_out, "w") as handle:
            json.dump(record, handle, indent=1)
    if args.compare_untrained:
        pre, un = means[model_label], means[f"{args.model_type} untrained"]
        diff = " ".join(
            f"p{p} {pre['after_switch'][p] - un['after_switch'][p]:+.3f}/"
            f"{pre['whole'][p] - un['whole'][p]:+.3f}"
            for p in range(adapter.spec_phase_count)
        )
        print(f"{'pretrained - untrained':<34} {diff} | run {float(pre['run'] - un['run']):+.3f}")
    run_s = [c[2] for c in results]
    print(f"timing: wall {wall:.1f}s, per spec run {np.mean(run_s):.3f}s mean single-process")


# --- ppo: on-policy pretraining ------------------------------------------------------

# The model under pretraining, read by the forked workers of a progress eval.
_SNAPSHOT: Optional[BuiltModel] = None


def _snapshot_job(args: tuple) -> Dict[str, np.ndarray]:
    protocol_name, design, seed = args
    set_reward_design(design)
    if _SNAPSHOT is None:
        raise RuntimeError("no model snapshot to evaluate")
    adapter = load_adapter(protocol_name)
    rows, _ = _spec_rows(adapter, seed, "model", ModelSettings(), None, None,
                         built=copy.deepcopy(_SNAPSHOT))
    return phase_summary(adapter, rows)


def cmd_ppo(args: argparse.Namespace, settings: ModelSettings) -> None:
    global _SNAPSHOT
    adapter = load_adapter(args.protocol, args.peer_spread)
    if (args.eval_every > 0) != (args.eval_seeds > 0):
        raise ValueError("--eval-every and --eval-seeds go together")
    _worker_init()
    built = settings.build(adapter.protocol, args.model_type, None, adapter.windows)
    for line in built.description:
        print(line)
    driver = ModelDriver(built, adapter)
    started = time.perf_counter()
    episodes_total = 0
    for i in range(args.runs):
        sim, episodes = adapter.training_run(args.seed + i)
        updates, reward_sum = driver.updates, driver.reward_sum
        adapter.run_training(sim, episodes, driver.predict, driver.update)
        episodes_total += episodes
        done = i + 1
        run_updates = driver.updates - updates
        line = (
            f"run {done}/{args.runs} seed={args.seed + i} episodes={episodes} "
            f"mean reward {(driver.reward_sum - reward_sum) / max(1, run_updates):+.4f}"
        )
        if args.eval_every > 0 and (done % args.eval_every == 0 or done == args.runs):
            _SNAPSHOT = copy.deepcopy(built)
            jobs = [(args.protocol, args.reward_design, seed)
                    for seed in range(args.eval_seeds)]
            summary = _mean_summary(_map(_snapshot_job, jobs, args.workers))
            phases = " ".join(f"p{p} {summary['after_switch'][p]:+.3f}"
                              for p in range(adapter.spec_phase_count))
            line += f" | spec ({args.eval_seeds} seeds) {phases} run {float(summary['run']):+.3f}"
            _SNAPSHOT = None
        print(line, flush=True)
    wall = time.perf_counter() - started
    model = _unwrap(built.model)
    save_model_prior(
        args.out, args.model_type, adapter.protocol, model, model.state_arrays(),
        windows=adapter.windows, simulator=adapter.simulator, runs=args.runs, seed=args.seed,
        updates=driver.updates, skipped=driver.skipped, peer_spread=args.peer_spread,
        latency_floor_multiplier=settings.floor_multiplier, code_hash=args.code_hash,
        **_override_provenance(args),
    )
    settings.build(adapter.protocol, args.model_type, args.out, adapter.windows)
    print(
        f"{adapter.simulator} ppo {args.model_type}: runs={args.runs} episodes={episodes_total} "
        f"updates={driver.updates} skipped={driver.skipped} "
        f"mean reward {driver.reward_sum / max(1, driver.updates):+.4f}"
    )
    print(f"timing: wall {wall:.1f}s, {args.runs / wall:.2f} runs/s, "
          f"{episodes_total / wall:.0f} episodes/s (progress evals included) -> {args.out}")


# --- CLI ---------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("protocol", choices=tuple(PROTOCOL_SIMULATORS))
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser, model_types: Sequence[str]) -> None:
        p.add_argument("--model-type", required=True, choices=tuple(model_types))
        p.add_argument("--reward-design", choices=REWARD_DESIGNS, default=DEFAULT_REWARD_DESIGN,
                       help=f"as main.py's (default: {DEFAULT_REWARD_DESIGN}); a prior trained "
                            "under another design is refused")
        p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                       help="forked worker processes (1: in-process)")
        p.add_argument("--timeout-ladder", default=None,
                       help="PBFT only: comma-separated rungs in ms replacing the 250,550,1200 "
                            "ladder, as main.py's (runs/2026-10-18-pbft-ladder-variants)")
        p.add_argument("--linucb-discount", type=float, default=None,
                       help="LinUCB models: discount factor (default: the class's, 0.95 for both "
                            "models); recorded in the prior's metadata")
        p.add_argument("--linucb-exploration", type=float, default=None,
                       help="LinUCB models: random-exploration share (default: the class's, 0)")
        p.add_argument("--ppo-exploration", type=float, default=None,
                       help="PPO models: uniform share mixed into the policy (default: the class's, 0)")
        p.add_argument("--sim-constants", default=None,
                       help="PBFT only: JSON of pbft_sim SimConstants fields replacing the "
                            "installed defaults, as pbft_sim train --sim-constants")

    def training(p: argparse.ArgumentParser) -> None:
        p.add_argument("--runs", type=int, required=True)
        p.add_argument("--seed", type=int, default=1, help="run i is seed + i")
        p.add_argument("--out", required=True)
        p.add_argument("--code-hash", required=True,
                       help="code_hash of the tree training the prior (scripts/cloudlab/code_state.py), "
                            "recorded in its provenance and compared with FREEZE by the harness")
        p.add_argument("--peer-spread", type=float, default=0.0,
                       help="Tendermint only: training runs draw peer timeout factors with "
                            "sigma in [0, this], as tendermint_sim train --peer-spread "
                            "(default 0: lockstep)")

    def floor(p: argparse.ArgumentParser) -> None:
        p.add_argument("--latency-floor-multiplier", type=float, default=0.0,
                       help="as main.py's (a scale on the protocol's FLOOR_SPECS floor); 0 (the "
                            "default) disables the floor. PPO applies the floor after sampling "
                            "and trains on the floored action, so pretraining PPO with a floor "
                            "biases its policy toward the floored timeouts")

    f = sub.add_parser("fit", help="fit a LinUCB prior mean on random-behaviour samples")
    common(f, FIT_MODEL_TYPES)
    training(f)

    p = sub.add_parser("ppo", help="pretrain a PPO model on-policy through training runs")
    common(p, PPO_MODEL_TYPES)
    floor(p)
    training(p)
    p.add_argument("--eval-every", type=int, default=0,
                   help="every this many runs, evaluate a copy online on the spec")
    p.add_argument("--eval-seeds", type=int, default=0)

    e = sub.add_parser("eval", help="run a model online on the spec beside the oracle and rungs")
    common(e, EVAL_MODEL_TYPES)
    floor(e)
    e.add_argument("--prior-path", default=None)
    e.add_argument("--seeds", type=int, required=True)
    e.add_argument("--compare-untrained", action="store_true",
                   help="also run the model type without its prior on the same seeds")
    e.add_argument("--per-seed", action="store_true", help="print every seed, not only means")
    e.add_argument("--json-out", default=None,
                   help="also write every policy's per-seed summaries (run mean, per-phase reward "
                        "after the switch and whole, median action) to this JSON file, for the gate")
    e.add_argument("--heuristic-multiplier", type=float, default=None)
    e.add_argument("--heuristic-window", type=int, default=None)
    e.add_argument("--heuristic-quantile", type=float, default=None)

    args = parser.parse_args()
    for name in ("runs", "seeds"):
        if getattr(args, name, 1) < 1:
            parser.error(f"--{name} must be >= 1")
    set_reward_design(args.reward_design)
    # PBFT-only overrides, process-wide before any model is built or worker forked.
    if (args.timeout_ladder is not None or args.sim_constants is not None) and args.protocol != "pbft":
        parser.error("--timeout-ladder and --sim-constants are PBFT's")
    if args.timeout_ladder is not None:
        try:
            set_pbft_timeout_ladder([int(v) for v in args.timeout_ladder.split(",")])
        except ValueError as exc:
            parser.error(f"--timeout-ladder {args.timeout_ladder!r}: {exc}")
    if args.sim_constants is not None:
        importlib.import_module(PROTOCOL_SIMULATORS["pbft"]).set_host_constants(args.sim_constants)
    settings = ModelSettings(
        heuristic_multiplier=getattr(args, "heuristic_multiplier", None),
        heuristic_window=getattr(args, "heuristic_window", None),
        heuristic_quantile=getattr(args, "heuristic_quantile", None),
        floor_multiplier=getattr(args, "latency_floor_multiplier", 0.0),
        linucb_discount=args.linucb_discount,
        linucb_exploration=args.linucb_exploration,
        ppo_exploration=args.ppo_exploration,
    )
    {"fit": cmd_fit, "ppo": cmd_ppo, "eval": cmd_eval}[args.cmd](args, settings)


if __name__ == "__main__":
    main()
