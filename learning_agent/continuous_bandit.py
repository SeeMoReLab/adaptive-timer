from collections import deque
from dataclasses import dataclass
import math
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import qmc
from sklearn.tree import DecisionTreeRegressor

import agent_pb2
from latency_floor import LatencyFloor
from models import (
    ActionCodec,
    TensorModel,
    get_timeout_values_ms,
    tendermint_timeouts_from_propose,
)


class ContinuousTimeoutActionCodec(ActionCodec[agent_pb2.Timeout]):
    """Maps normalized continuous actions to protocol timeout messages."""

    def __init__(
        self,
        protocol: int,
        bounds_ms: Tuple[int, int] = (100, 10000),
        wait_bounds_ms: Optional[Tuple[int, int]] = None,
    ):
        self._protocol = int(protocol)
        # timeout_values = get_timeout_values_ms(self._protocol)
        # lower_ms = min(timeout_values)
        # upper_ms = max(timeout_values)
        lower_ms, upper_ms = int(bounds_ms[0]), int(bounds_ms[1])
        if lower_ms <= 0 or upper_ms <= lower_ms:
            raise ValueError(
                f"Invalid continuous timeout bounds: {lower_ms}..{upper_ms}ms"
            )

        if self._protocol in (
            agent_pb2.PROTOCOL_PBFT,
            agent_pb2.PROTOCOL_SMARTBFT,
            agent_pb2.PROTOCOL_TENDERMINT,
            agent_pb2.PROTOCOL_HOTSTUFF,
        ):
            if wait_bounds_ms is not None:
                raise ValueError(
                    "wait_bounds_ms is SBFT's fast-path wait; "
                    f"{agent_pb2.Protocol.Name(self._protocol)} has one timeout dimension"
                )
            self._action_dim = 1
            self._bounds_ms: Tuple[Tuple[int, int], ...] = ((lower_ms, upper_ms),)
        elif self._protocol == agent_pb2.PROTOCOL_SBFT:
            # Dimension 0 the election timeout T, dimension 1 the fast-path wait W.
            if wait_bounds_ms is None:
                raise ValueError("SBFT continuous actions need wait_bounds_ms")
            wait_lower_ms, wait_upper_ms = int(wait_bounds_ms[0]), int(wait_bounds_ms[1])
            if wait_lower_ms <= 0 or wait_upper_ms <= wait_lower_ms:
                raise ValueError(
                    f"Invalid continuous wait bounds: {wait_lower_ms}..{wait_upper_ms}ms"
                )
            self._action_dim = 2
            self._bounds_ms = ((lower_ms, upper_ms), (wait_lower_ms, wait_upper_ms))
        else:
            raise ValueError(
                "Continuous timeout actions currently support only PBFT, "
                "SmartBFT, SBFT, Tendermint and HotStuff"
            )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def protocol(self) -> int:
        return self._protocol

    @property
    def bounds_ms(self) -> Tuple[Tuple[int, int], ...]:
        return self._bounds_ms

    def normalize_ms(self, timeout_ms: int, dimension: int = 0) -> float:
        """Public form of the log-space mapping that decode() inverts."""
        return self._normalize_value(int(timeout_ms), dimension)

    def _normalize_value(self, timeout_ms: int, dimension: int) -> float:
        lower_ms, upper_ms = self._bounds_ms[dimension]
        if timeout_ms < lower_ms or timeout_ms > upper_ms:
            raise ValueError(
                f"Timeout {timeout_ms}ms is outside continuous bounds "
                f"{lower_ms}..{upper_ms}ms"
            )
        numerator = math.log(float(timeout_ms)) - math.log(float(lower_ms))
        denominator = math.log(float(upper_ms)) - math.log(float(lower_ms))
        return numerator / denominator

    def _denormalize_value(self, normalized: float, dimension: int) -> int:
        lower_ms, upper_ms = self._bounds_ms[dimension]
        bounded = min(1.0, max(0.0, float(normalized)))
        log_value = math.log(float(lower_ms)) + bounded * (
            math.log(float(upper_ms)) - math.log(float(lower_ms))
        )
        return min(upper_ms, max(lower_ms, int(round(math.exp(log_value)))))

    def decode(self, action: np.ndarray) -> agent_pb2.Timeout:
        values = np.asarray(action, dtype=float).reshape(-1)
        if values.size != self._action_dim:
            raise ValueError(
                f"Expected {self._action_dim} continuous action values, "
                f"got {values.size}"
            )
        timeout_values = [
            self._denormalize_value(float(values[index].item()), index)
            for index in range(self._action_dim)
        ]

        if self._protocol == agent_pb2.PROTOCOL_PBFT:
            return agent_pb2.Timeout(
                pbft=agent_pb2.PbftTimeout(
                    election_timeout_milliseconds=timeout_values[0]
                )
            )
        if self._protocol == agent_pb2.PROTOCOL_SMARTBFT:
            return agent_pb2.Timeout(
                smartbft=agent_pb2.SmartbftTimeout(
                    election_timeout_milliseconds=timeout_values[0]
                )
            )
        if self._protocol == agent_pb2.PROTOCOL_SBFT:
            return agent_pb2.Timeout(
                sbft=agent_pb2.SbftTimeout(
                    election_timeout_milliseconds=timeout_values[0],
                    slow_path_timeout_milliseconds=timeout_values[1],
                )
            )
        if self._protocol == agent_pb2.PROTOCOL_HOTSTUFF:
            return agent_pb2.Timeout(
                hotstuff=agent_pb2.HotstuffTimeout(
                    timeout_delay_milliseconds=timeout_values[0]
                )
            )
        propose_timeout_ms, prevote_timeout_ms, precommit_timeout_ms = (
            tendermint_timeouts_from_propose(timeout_values[0])
        )
        return agent_pb2.Timeout(
            tendermint=agent_pb2.TendermintTimeout(
                propose_timeout_milliseconds=propose_timeout_ms,
                prevote_timeout_milliseconds=prevote_timeout_ms,
                precommit_timeout_milliseconds=precommit_timeout_ms,
            )
        )

    def encode(self, action: object) -> np.ndarray:
        if isinstance(action, agent_pb2.Timeout):
            if self._protocol == agent_pb2.PROTOCOL_PBFT:
                action = action.pbft
            elif self._protocol == agent_pb2.PROTOCOL_SMARTBFT:
                action = action.smartbft
            elif self._protocol == agent_pb2.PROTOCOL_SBFT:
                action = action.sbft
            elif self._protocol == agent_pb2.PROTOCOL_HOTSTUFF:
                action = action.hotstuff
            else:
                action = action.tendermint

        if self._protocol in (
            agent_pb2.PROTOCOL_PBFT,
            agent_pb2.PROTOCOL_SMARTBFT,
        ):
            timeout_values = [
                int(getattr(action, "election_timeout_milliseconds"))
            ]
        elif self._protocol == agent_pb2.PROTOCOL_SBFT:
            timeout_values = [
                int(getattr(action, "election_timeout_milliseconds")),
                int(getattr(action, "slow_path_timeout_milliseconds")),
            ]
        elif self._protocol == agent_pb2.PROTOCOL_HOTSTUFF:
            timeout_values = [int(getattr(action, "timeout_delay_milliseconds"))]
        else:
            propose_timeout_ms = int(
                getattr(action, "propose_timeout_milliseconds")
            )
            expected = tendermint_timeouts_from_propose(propose_timeout_ms)
            actual = (
                propose_timeout_ms,
                int(getattr(action, "prevote_timeout_milliseconds")),
                int(getattr(action, "precommit_timeout_milliseconds")),
            )
            if actual != expected:
                raise ValueError(
                    "Tendermint prevote and precommit timeouts must be "
                    f"one-third of propose timeout: expected {expected}, got {actual}"
                )
            timeout_values = [propose_timeout_ms]

        normalized = [
            self._normalize_value(timeout_ms, index)
            for index, timeout_ms in enumerate(timeout_values)
        ]
        return np.asarray(normalized, dtype=float)

    def anchor_actions(self) -> np.ndarray:
        timeout_values = get_timeout_values_ms(self._protocol)
        timeout_tuples = [(value,) for value in timeout_values]
        normalized = [
            [
                self._normalize_value(int(timeout_ms), index)
                for index, timeout_ms in enumerate(timeout_tuple)
            ]
            for timeout_tuple in timeout_tuples
        ]
        return np.asarray(normalized, dtype=float)


class NormalizedLatencyFloor:
    """:class:`LatencyFloor` expressed in the codec's normalized action space.

    The continuous models search in [0, 1] over log-spaced milliseconds, so the
    constraint has to be mapped through the same transform before it can bound
    a candidate set.
    """

    def __init__(
        self,
        action_codec: ContinuousTimeoutActionCodec,
        latency_index: int,
        n_features: int,
        instances_index: Optional[int] = None,
        latency_floor: Optional[LatencyFloor] = None,
    ) -> None:
        if not 0 <= int(latency_index) < int(n_features):
            raise ValueError(
                f"latency_index must be in [0, {n_features}), got {latency_index}"
            )
        if instances_index is not None and not 0 <= int(instances_index) < int(
            n_features
        ):
            raise ValueError(
                f"instances_index must be in [0, {n_features}), got {instances_index}"
            )
        self._action_codec = action_codec
        self._latency_index = int(latency_index)
        self._instances_index = (
            None if instances_index is None else int(instances_index)
        )
        self._latency_floor = (
            LatencyFloor() if latency_floor is None else latency_floor
        )

    @property
    def latency_floor(self) -> LatencyFloor:
        return self._latency_floor

    @property
    def floor_ms(self) -> float:
        return self._latency_floor.floor_ms

    def observe_features(self, row: np.ndarray) -> None:
        alive = True
        if self._instances_index is not None:
            alive = float(row[self._instances_index]) > 0.0
        self._latency_floor.observe(float(row[self._latency_index]), alive=alive)

    def lower_bound(self) -> float:
        """Smallest normalized action the constraint permits, in [0, 1)."""
        floor_ms = self._latency_floor.floor_ms
        if floor_ms <= 0.0:
            return 0.0
        lower_ms, upper_ms = self._action_codec.bounds_ms[0]
        if floor_ms <= lower_ms:
            return 0.0
        # A floor above the whole range would leave nothing to play; keep the
        # longest timeout available, as feasible_timeout_mask does.
        if floor_ms >= upper_ms:
            return 1.0
        return self._action_codec.normalize_ms(int(math.ceil(floor_ms)))

    def reset(self) -> None:
        self._latency_floor.reset()


class ContinuousThompsonSamplingModel(TensorModel):
    """Bootstrapped-tree contextual bandit over a continuous action domain."""

    def __init__(
        self,
        action_dim: int,
        n_features: int,
        anchor_actions: np.ndarray,
        initial_design_size: int,
        memory_size: int,
        n_heads: int = 100,
        candidate_count: int = 1024,
        exploration_probability: float = 0.5,
        seed: Optional[int] = None,
        latency_floor: Optional["NormalizedLatencyFloor"] = None,
    ):
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")
        if n_features <= 0:
            raise ValueError(f"n_features must be > 0, got {n_features}")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be > 0, got {n_heads}")
        if candidate_count <= 0:
            raise ValueError(
                f"candidate_count must be > 0, got {candidate_count}"
            )
        if initial_design_size <= 0:
            raise ValueError(
                f"initial_design_size must be > 0, got {initial_design_size}"
            )
        if memory_size <= 0:
            raise ValueError(f"memory_size must be > 0, got {memory_size}")
        if not 0.0 <= exploration_probability <= 1.0:
            raise ValueError(
                "exploration_probability must be between 0 and 1, got "
                f"{exploration_probability}"
            )

        anchors = np.asarray(anchor_actions, dtype=float)
        if anchors.ndim != 2 or anchors.shape[1] != action_dim:
            raise ValueError(
                "anchor_actions must have shape "
                f"(N, {action_dim}), got {tuple(anchors.shape)}"
            )

        self._action_dim = int(action_dim)
        self._obs_dim = int(n_features)
        self._n_heads = int(n_heads)
        self._candidate_count = int(candidate_count)
        self._initial_design_size = int(initial_design_size)
        self._memory_size = int(memory_size)
        self._exploration_probability = float(exploration_probability)
        self._anchor_actions = np.asarray(anchors, dtype=float).astype(float)
        self._latency_floor = latency_floor
        self._seed = (
            int(time.time_ns() & 0x7FFFFFFF) if seed is None else int(seed)
        )

        self.reset_model()

    def _new_tree(self, head_index: int) -> DecisionTreeRegressor:
        return DecisionTreeRegressor(
            criterion="squared_error",
            max_depth=6,
            min_samples_leaf=6,
            min_impurity_decrease=0.01,
            random_state=self._seed + head_index,
        )

    def _lower_bound(self) -> float:
        """Normalized floor under which the action space is not searched."""
        if self._latency_floor is None:
            return 0.0
        return float(self._latency_floor.lower_bound())

    def _candidate_actions(self, lower_bound: float = 0.0) -> np.ndarray:
        sampled = self._candidate_engine.random(self._candidate_count)
        candidate_parts = [sampled, self._anchor_actions]
        if self.memory["actions"]:
            candidate_parts.append(np.asarray(self.memory["actions"], dtype=float))
        candidates = np.vstack(candidate_parts)
        candidates = np.unique(np.clip(candidates, 0.0, 1.0), axis=0)
        if lower_bound <= 0.0:
            return candidates
        feasible = candidates[np.all(candidates >= lower_bound, axis=1)]
        if len(feasible) == 0:
            # Everything the sampler offered is below the floor; the bound
            # itself is still a legal action, so play that rather than nothing.
            return np.full((1, self._action_dim), lower_bound, dtype=float)
        return feasible

    def _predict_one(self, features: np.ndarray) -> np.ndarray:
        if self._latency_floor is not None:
            self._latency_floor.observe_features(features)
        lower_bound = self._lower_bound()

        if self._design_count < self._initial_design_size:
            self._design_count += 1
            design = self._design_engine.random(1)[0].astype(float)
            # Rescale rather than clip, so the initial design still spreads
            # over the feasible range instead of piling up on the bound.
            return lower_bound + (1.0 - lower_bound) * design

        candidates = self._candidate_actions(lower_bound)
        if not self.is_fitted:
            return candidates[int(self._rng.integers(0, len(candidates)))]

        inputs = np.hstack(
            (np.repeat(features.reshape(1, -1), len(candidates), axis=0), candidates)
        )
        if self._rng.random() < self._exploration_probability:
            head_index = int(self._rng.integers(0, self._n_heads))
            predictions = self.models[head_index].predict(inputs)
        else:
            predictions = np.mean(
                np.vstack([model.predict(inputs) for model in self.models]), axis=0
            )

        maximum = float(np.max(predictions))
        best_indices = np.flatnonzero(np.isclose(predictions, maximum))
        selected = int(self._rng.choice(best_indices))
        return candidates[selected]

    def predict_action(self, features: np.ndarray) -> np.ndarray:
        features_np = np.asarray(features, dtype=float)
        single = features_np.ndim == 1
        if single:
            features_np = features_np.reshape(1, -1)
        if features_np.shape[1] != self._obs_dim:
            raise ValueError(
                f"Expected {self._obs_dim} features, got {features_np.shape[1]}"
            )

        actions = np.vstack([self._predict_one(row) for row in features_np])
        result = np.asarray(actions, dtype=float)
        return result[0] if single else result

    def _action_rows(self, actions: np.ndarray, batch_size: int) -> np.ndarray:
        actions_np = np.asarray(actions, dtype=float)
        if actions_np.ndim == 0:
            actions_np = actions_np.reshape(1, 1)
        elif actions_np.ndim == 1:
            if batch_size == 1 and actions_np.size == self._action_dim:
                actions_np = actions_np.reshape(1, self._action_dim)
            elif self._action_dim == 1 and actions_np.size == batch_size:
                actions_np = actions_np.reshape(batch_size, 1)
            else:
                raise ValueError(
                    "Continuous action batch shape mismatch: "
                    f"actions={actions_np.shape}, batch_size={batch_size}"
                )
        if actions_np.shape != (batch_size, self._action_dim):
            raise ValueError(
                "Continuous action batch shape mismatch: "
                f"expected {(batch_size, self._action_dim)}, got {actions_np.shape}"
            )
        if np.any(actions_np < 0.0) or np.any(actions_np > 1.0):
            raise ValueError("Continuous actions must be normalized to [0, 1]")
        return actions_np

    def update_model(
        self,
        features: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | float,
        backward: bool = True,
    ) -> float:
        features_np = np.asarray(features, dtype=float)
        if features_np.ndim == 1:
            features_np = features_np.reshape(1, -1)
        if features_np.shape[1] != self._obs_dim:
            raise ValueError(
                f"Expected {self._obs_dim} features, got {features_np.shape[1]}"
            )

        actions_np = self._action_rows(actions, features_np.shape[0])
        rewards_np = np.asarray(rewards, dtype=float).reshape(-1)
        if rewards_np.size == 1 and features_np.shape[0] > 1:
            rewards_np = np.repeat(rewards_np, features_np.shape[0])
        if rewards_np.size != features_np.shape[0]:
            raise ValueError(
                "rewards batch size mismatch: "
                f"expected {features_np.shape[0]}, got {rewards_np.size}"
            )

        for row, action, reward in zip(features_np, actions_np, rewards_np):
            self.memory["features"].append(row.copy())
            self.memory["actions"].append(action.copy())
            self.memory["rewards"].append(float(reward))

        if backward:
            context = np.asarray(self.memory["features"], dtype=float)
            action_values = np.asarray(self.memory["actions"], dtype=float)
            inputs = np.hstack((context, action_values))
            targets = np.asarray(self.memory["rewards"], dtype=float)
            for model in self.models:
                sample_indices = self._rng.choice(
                    len(inputs), size=len(inputs), replace=True
                )
                model.fit(inputs[sample_indices], targets[sample_indices])
            self.is_fitted = True

        return 0.0

    def reset_model(self) -> None:
        if self._latency_floor is not None:
            self._latency_floor.reset()
        self._rng = np.random.default_rng(self._seed)
        # Owen-scrambled Sobol, as torch.quasirandom.SobolEngine provided
        # before. Both use Joe-Kuo direction numbers, so the sequences are
        # equivalent in distribution but not point-for-point identical: a run
        # seeded the same as one from the torch version will explore the same
        # space in a different order.
        self._design_engine = qmc.Sobol(
            d=self._action_dim, scramble=True, seed=self._seed
        )
        self._candidate_engine = qmc.Sobol(
            d=self._action_dim, scramble=True, seed=self._seed + 1
        )
        self._design_count = 0
        self.models = [self._new_tree(index) for index in range(self._n_heads)]
        self.is_fitted = False
        self.memory: Dict[str, deque[Any]] = {
            "features": deque(maxlen=self._memory_size),
            "actions": deque(maxlen=self._memory_size),
            "rewards": deque(maxlen=self._memory_size),
        }



DEFAULT_PROPORTIONAL_K = 1.0
class ContinuousHeuristicTimeoutModel(TensorModel):
    """Proportional rule over the reported consensus latency.
        timeout_ms = clamp(k * avg_consensus_latency_ms, lower_ms, upper_ms)

    The mean rather than the p95: the p95 of a window is dominated by the
    rounds the current timer already mishandled, so it reports the arm's own
    thrashing back to the rule that chose it. The mean moves with the delay
    environment and is the field the latency floor reads, so the rule and the
    constraint are driven by one measurement.
    """

    def __init__(
        self,
        action_codec: ContinuousTimeoutActionCodec,
        n_features: int,
        latency_index: int,
        transactions_index: int,
        multiplier: float = DEFAULT_PROPORTIONAL_K,
        clamp_ms: Optional[Tuple[int, int]] = None,
        initial_timeout_ms: Optional[int] = None,
    ):
        if n_features <= 0:
            raise ValueError(f"n_features must be > 0, got {n_features}")
        for name, index in (
            ("latency_index", latency_index),
            ("transactions_index", transactions_index),
        ):
            if not 0 <= index < n_features:
                raise ValueError(
                    f"{name} must be in [0, {n_features}), got {index}"
                )
        if not math.isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError(f"multiplier must be > 0, got {multiplier}")

        codec_lower_ms, codec_upper_ms = action_codec.bounds_ms[0]
        if clamp_ms is None:
            clamp_ms = (codec_lower_ms, codec_upper_ms)
        lower_ms, upper_ms = (int(clamp_ms[0]), int(clamp_ms[1]))
        if upper_ms <= lower_ms:
            raise ValueError(f"Invalid clamp bounds: {lower_ms}..{upper_ms}ms")
        if lower_ms < codec_lower_ms or upper_ms > codec_upper_ms:
            raise ValueError(
                f"Clamp bounds {lower_ms}..{upper_ms}ms fall outside the codec "
                f"bounds {codec_lower_ms}..{codec_upper_ms}ms"
            )

        if initial_timeout_ms is None:
            initial_timeout_ms = int(round(math.sqrt(lower_ms * upper_ms)))
        initial_timeout_ms = min(upper_ms, max(lower_ms, int(initial_timeout_ms)))

        self._action_codec = action_codec
        self._obs_dim = int(n_features)
        self._latency_index = int(latency_index)
        self._transactions_index = int(transactions_index)
        self._multiplier = float(multiplier)
        self._lower_ms = lower_ms
        self._upper_ms = upper_ms
        self._initial_timeout_ms = initial_timeout_ms

        self.reset_model()

    @property
    def timeout_ms(self) -> int:
        return self._held_timeout_ms

    def _timeout_for(self, features: np.ndarray) -> int:
        latency_ms = float(features[self._latency_index])
        transactions = float(features[self._transactions_index])
        if transactions <= 0.0 or not math.isfinite(latency_ms) or latency_ms <= 0.0:
            return self._held_timeout_ms

        timeout_ms = int(round(self._multiplier * latency_ms))
        return min(self._upper_ms, max(self._lower_ms, timeout_ms))

    def predict_action(self, features: np.ndarray) -> np.ndarray:
        features_np = np.asarray(features, dtype=float)
        single = features_np.ndim == 1
        if single:
            features_np = features_np.reshape(1, -1)
        if features_np.shape[1] != self._obs_dim:
            raise ValueError(
                f"Expected {self._obs_dim} features, got {features_np.shape[1]}"
            )

        actions = []
        for row in features_np:
            self._held_timeout_ms = self._timeout_for(row)
            actions.append([self._action_codec.normalize_ms(self._held_timeout_ms)])

        result = np.asarray(actions, dtype=float)
        return result[0] if single else result

    def update_model(
        self,
        features: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | float,
        backward: bool = True,
    ) -> float:
        return 0.0

    def reset_model(self) -> None:
        self._held_timeout_ms = self._initial_timeout_ms


# --- Simulator-pretrained Gaussian-process bandit -----------------------------

_PRIOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_models")

# Action range of --model-type=continuous on Tendermint, and where its prior ships.
CONTINUOUS_TENDERMINT_BOUNDS_MS: Tuple[int, int] = (100, 3000)
DEFAULT_TENDERMINT_PRIOR_PATH = os.path.join(_PRIOR_DIR, "continuous_tendermint_prior.npz")


@dataclass(frozen=True)
class ContextSpec:
    """What --model-type=continuous reads and plays on one protocol.

    The context vector is log1p of ``log_fields`` followed by ``raw_fields``
    verbatim, in that order; ``length_scales`` has one GP length scale per
    context dimension (inf drops the field from the kernel; the prior still
    reads it). ``bounds_ms`` is the codec's action range (SBFT's election
    timeout), ``wait_bounds_ms`` SBFT's fast-path wait range (None on the
    one-dimensional protocols), ``prior_path`` where the simulator-trained
    prior ships (input_dim = context_dim + action_dim, trained on the agent's
    reward design) and ``simulator`` the script that trains it.
    """

    log_fields: Tuple[str, ...]
    raw_fields: Tuple[str, ...]
    length_scales: Tuple[float, ...]
    bounds_ms: Tuple[int, int]
    prior_path: str
    simulator: str
    wait_bounds_ms: Optional[Tuple[int, int]] = None

    def __post_init__(self) -> None:
        if len(self.length_scales) != len(self.log_fields) + len(self.raw_fields):
            raise ValueError(
                f"{len(self.length_scales)} length scales for "
                f"{len(self.log_fields) + len(self.raw_fields)} context fields"
            )

    @property
    def context_dim(self) -> int:
        return len(self.length_scales)


# Log-scaled report fields the model reads, in TENDERMINT_REPORT_FEATURE_FIELD_NAMES
# terms, plus the violation rate.
_TENDERMINT_CONTEXT_LOG_FIELDS: Tuple[str, ...] = (
    "avg_consensus_latency_ms",
    "p50_consensus_latency_ms",
    "p90_consensus_latency_ms",
    "propose_latency_ms",
    "prevote_latency_ms",
    "precommit_latency_ms",
    "throughput_tps",
    "avg_batch_size",
)
# GP length scales per context dimension: 0.3 in log units is a factor of 1.35.
# The GP tells contexts apart only by p50 latency, throughput and violation
# rate. The other fields swing from one 8 s window to the next within a phase
# (p3 prevote latency 77-143 ms), which made every window a new context and
# kept the GP from pooling a phase's evidence. inf drops a field from the
# kernel; the prior still reads every field.
_TENDERMINT_CONTEXT_LENGTH_SCALES: Tuple[float, ...] = (
    math.inf,  # avg_consensus_latency_ms
    0.3,  # p50_consensus_latency_ms
    math.inf, math.inf, math.inf, math.inf,  # p90, propose, prevote, precommit
    0.2,  # throughput_tps
    math.inf,  # avg_batch_size
    0.15,  # timeout_violation_rate
)

# PBFT and SmartBFT share the 16-field report (PBFT_REPORT_FEATURE_FIELD_NAMES).
# view_change_count stands in for Tendermint's violation rate. timeout_ms, the
# applied election timeout, is left out on purpose: the model's own last pick
# in the context drove a two-cycle (H15).
_PBFT_CONTEXT_LOG_FIELDS: Tuple[str, ...] = (
    "avg_consensus_latency_ms",
    "p50_consensus_latency_ms",
    "p95_consensus_latency_ms",
    "throughput_tps",
    "avg_batch_size",
    "p50_inter_commit_gap_ms",
    "view_change_count",
)
# Same reading as the Tendermint scales; 0.7 on the log1p count is a factor
# of 2 in view changes (quiet windows report 0, a PBFT storm 3-5 per 10 s
# window, a SmartBFT storm about 18).
_PBFT_CONTEXT_LENGTH_SCALES: Tuple[float, ...] = (
    math.inf,  # avg_consensus_latency_ms
    0.3,  # p50_consensus_latency_ms
    math.inf,  # p95_consensus_latency_ms
    0.2,  # throughput_tps
    math.inf, math.inf,  # avg_batch_size, p50_inter_commit_gap_ms
    0.7,  # view_change_count
)

# SBFT's report is PBFT's plus the phase latencies and path counts
# (SBFT_REPORT_FEATURE_FIELD_NAMES); the fast and slow path counts join
# view_change_count in the kernel. pre_prepare_latency_ms (pre-prepare to
# certificate) carries the model's own W on a miss, so the prior reads it but
# the kernel does not. The (T, W) applied while the context was measured is not
# part of it: the prior's 3-episode return already handles a T that escapes a
# delayed leader, and with the applied pair as an input the policy cycled
# through its own last pick (runs/2026-10-01-sbft-continuous, 22:55: three of
# five fits good without it, one or two with it). The leader's n-th prepare
# gap and its seen/unseen counts tell a miss the wait could catch (a gap under
# the slow path) from one it cannot (unseen by execution, as with a crashed
# replica); the prior reads them, the kernel does not.
_SBFT_CONTEXT_LOG_FIELDS: Tuple[str, ...] = (
    "avg_consensus_latency_ms",
    "p50_consensus_latency_ms",
    "p95_consensus_latency_ms",
    "throughput_tps",
    "avg_batch_size",
    "p50_inter_commit_gap_ms",
    "view_change_count",
    "pre_prepare_latency_ms",
    "prepare_latency_ms",
    "fast_path_count",
    "slow_path_count",
    "nth_prepare_gap_ms",
    "nth_prepare_seen_count",
    "nth_prepare_unseen_count",
)
_SBFT_CONTEXT_LENGTH_SCALES: Tuple[float, ...] = (
    math.inf,  # avg_consensus_latency_ms
    0.3,  # p50_consensus_latency_ms
    math.inf,  # p95_consensus_latency_ms
    0.2,  # throughput_tps
    math.inf, math.inf,  # avg_batch_size, p50_inter_commit_gap_ms
    0.7,  # view_change_count
    math.inf, math.inf,  # pre_prepare_latency_ms, prepare_latency_ms
    0.7,  # fast_path_count
    0.7,  # slow_path_count
    math.inf, math.inf, math.inf,  # nth_prepare_gap_ms, _seen_count, _unseen_count
)

CONTEXT_SPECS: Dict[int, ContextSpec] = {
    agent_pb2.PROTOCOL_TENDERMINT: ContextSpec(
        log_fields=_TENDERMINT_CONTEXT_LOG_FIELDS,
        raw_fields=("timeout_violation_rate",),
        length_scales=_TENDERMINT_CONTEXT_LENGTH_SCALES,
        bounds_ms=CONTINUOUS_TENDERMINT_BOUNDS_MS,
        prior_path=DEFAULT_TENDERMINT_PRIOR_PATH,
        simulator="tendermint_sim.py",
    ),
    # Top rung 1200; base p50 94-168 across hosts, so 100 sits under the floor.
    agent_pb2.PROTOCOL_PBFT: ContextSpec(
        log_fields=_PBFT_CONTEXT_LOG_FIELDS,
        raw_fields=(),
        length_scales=_PBFT_CONTEXT_LENGTH_SCALES,
        bounds_ms=(100, 3000),
        prior_path=os.path.join(_PRIOR_DIR, "continuous_pbft_prior.npz"),
        simulator="pbft_sim.py",
    ),
    # 1800 is the only rung that survives 5+150 and 800 still storms there.
    agent_pb2.PROTOCOL_SMARTBFT: ContextSpec(
        log_fields=_PBFT_CONTEXT_LOG_FIELDS,
        raw_fields=(),
        length_scales=_PBFT_CONTEXT_LENGTH_SCALES,
        bounds_ms=(100, 4000),
        prior_path=os.path.join(_PRIOR_DIR, "continuous_smartbft_prior.npz"),
        simulator="smartbft_sim.py",
    ),
    # Two-dimensional action: election timeout T in bounds_ms, fast-path wait
    # W in wait_bounds_ms.
    agent_pb2.PROTOCOL_SBFT: ContextSpec(
        log_fields=_SBFT_CONTEXT_LOG_FIELDS,
        raw_fields=(),
        length_scales=_SBFT_CONTEXT_LENGTH_SCALES,
        bounds_ms=(100, 3000),
        prior_path=os.path.join(_PRIOR_DIR, "continuous_sbft_prior.npz"),
        simulator="sbft_sim.py",
        wait_bounds_ms=(10, 500),
    ),
    # HotStuff's report is PBFT's field for field (models.py). A healthy round
    # is 2h, so the 100 ms lower bound sits at the block-delay floor of a 5 ms
    # link; 2000 is three times what 5+150 needs.
    agent_pb2.PROTOCOL_HOTSTUFF: ContextSpec(
        log_fields=_PBFT_CONTEXT_LOG_FIELDS,
        raw_fields=(),
        length_scales=_PBFT_CONTEXT_LENGTH_SCALES,
        bounds_ms=(100, 2000),
        prior_path=os.path.join(_PRIOR_DIR, "continuous_hotstuff_prior.npz"),
        simulator="hotstuff_sim.py",
    ),
}


def context_spec(protocol: int) -> ContextSpec:
    spec = CONTEXT_SPECS.get(int(protocol))
    if spec is None:
        raise ValueError(
            "--model-type=continuous has no context spec for "
            f"{agent_pb2.Protocol.Name(int(protocol))}; it runs on PBFT, SmartBFT, "
            "SBFT, Tendermint and HotStuff"
        )
    return spec


def report_context(
    spec: ContextSpec, field_names: Tuple[str, ...], features: np.ndarray
) -> np.ndarray:
    """Context vector for one report: log1p of spec.log_fields, then spec.raw_fields.

    The model's own previous timeout is deliberately not part of it: with it,
    each replica's last pick fed back into its next one and drove a two-cycle
    between low and high timeouts in p3.
    """
    row = np.asarray(features, dtype=float).reshape(-1)
    values = [
        math.log1p(max(0.0, float(row[field_names.index(name)])))
        for name in spec.log_fields
    ]
    values.extend(float(row[field_names.index(name)]) for name in spec.raw_fields)
    return np.asarray(values, dtype=float)


def tendermint_context(field_names: Tuple[str, ...], features: np.ndarray) -> np.ndarray:
    return report_context(CONTEXT_SPECS[agent_pb2.PROTOCOL_TENDERMINT], field_names, features)


class NumpyMLP:
    """ReLU MLP regressor evaluated in numpy, loaded from an .npz of weights."""

    def __init__(self, path: str):
        data = np.load(path)
        count = int(data["n_layers"])
        self._weights = [data[f"W{i}"] for i in range(count)]
        self._biases = [data[f"b{i}"] for i in range(count)]
        self._x_mean = data["x_mean"]
        self._x_std = data["x_std"]
        self._y_mean = float(data["y_mean"])
        self._y_std = float(data["y_std"])
        self.reward_design = str(data["reward_design"])
        # A prior fit on log(reward) (smartbft_sim.train_prior) says so; its outputs are
        # exponentiated back to reward units so the GP residual stays in those units.
        self.target_log = bool(int(data["target_log"])) if "target_log" in data else False
        # Episodes the action was held for in training (tendermint_sim train
        # --hold-episodes); the target was the mean reward over the hold. 1 = one
        # episode's reward, the SimPriorGPModel prior.
        self.hold_episodes = int(data["hold_episodes"]) if "hold_episodes" in data else 1
        # Rewards below this were fit as this value (tendermint_sim train
        # --target-floor); -inf when the prior was fit to the raw reward.
        self.target_floor = float(data["target_floor"]) if "target_floor" in data else -math.inf

    @property
    def input_dim(self) -> int:
        return int(self._x_mean.size)

    @property
    def y_std(self) -> float:
        """Standard deviation of the training target, the reward's natural scale."""
        return self._y_std

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        h = (np.asarray(inputs, dtype=float) - self._x_mean) / self._x_std
        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            h = h @ w + b
            if i < len(self._weights) - 1:
                h = np.maximum(h, 0.0)
        out = h.reshape(-1) * self._y_std + self._y_mean
        return np.exp(out) if self.target_log else out


class SimPriorGPModel(TensorModel):
    """Continuous contextual bandit: simulator prior plus online GP residual.

    The reward of timeout ``a`` in context ``z`` is modelled as
    ``prior(z, a) + f(z, a)``. ``prior`` is an MLP trained offline on the
    protocol's simulator (``ContextSpec.simulator``; randomized environments,
    never the failure spec);
    ``f`` is a Gaussian process fitted online to what the prior got wrong on
    this deployment. ``f`` has a level shared by every timeout in a context
    (``level_std``) plus a part local in the timeout, so a context-wide miss
    of the prior (as in p3, where it is optimistic at every timeout) lowers
    untried timeouts too instead of only the one played. Each decision plays,
    over a log-spaced grid of timeouts,
    the argmax of the lower confidence bound ``prior + mean - confidence * sd``
    of the posterior: a timeout nobody has tried in this context is discounted
    by its uncertainty, so after a failure the model moves toward timeouts it
    has evidence for rather than toward an untried one the prior is optimistic
    about, and a context it has not seen at all falls back on the prior.

    The action is the normalized log timeout of ``action_codec``; on SBFT it
    is the pair (election timeout, fast-path wait), searched over the product
    grid ``grid_size`` x ``wait_grid_size`` with a product RBF kernel over the
    two. The context, scales, and prior come from
    ``CONTEXT_SPECS[action_codec.protocol]``.
    There is no floor, clamp, or rule other than the codec's range.
    """

    def __init__(
        self,
        action_codec: ContinuousTimeoutActionCodec,
        field_names: Tuple[str, ...],
        prior_path: str,
        reward_design: str,
        grid_size: int = 64,
        signal_std: float = 0.08,
        noise_std: float = 0.04,
        action_length_scale: float = 0.1,
        memory_size: int = 256,
        confidence: float = 1.0,
        level_std: float = 0.2,
        wait_grid_size: int = 16,
    ):
        action_dim = int(action_codec.action_dim)
        if action_dim not in (1, 2):
            raise ValueError(
                f"SimPriorGPModel supports one or two timeout dimensions, got {action_dim}"
            )
        for name, value in (
            ("grid_size", grid_size),
            ("wait_grid_size", wait_grid_size),
            ("memory_size", memory_size),
        ):
            if int(value) <= 1:
                raise ValueError(f"{name} must be > 1, got {value}")
        for name, value in (
            ("signal_std", signal_std),
            ("noise_std", noise_std),
            ("action_length_scale", action_length_scale),
            ("confidence", confidence),
            ("level_std", level_std),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be > 0, got {value}")

        self._codec = action_codec
        self._action_dim = action_dim
        self._spec = context_spec(action_codec.protocol)
        self._field_names = tuple(field_names)
        self._obs_dim = len(self._field_names)
        for name in self._spec.log_fields + self._spec.raw_fields:
            if name not in self._field_names:
                raise ValueError(
                    f"Context field {name} is not in the report fields given"
                )
        if not os.path.isfile(prior_path):
            raise FileNotFoundError(
                f"Prior {prior_path} does not exist; train it with "
                f"{self._spec.simulator} train --reward-design {reward_design}"
            )
        self._prior = NumpyMLP(prior_path)
        if self._prior.reward_design != reward_design:
            raise ValueError(
                f"Prior {prior_path} was trained on {self._prior.reward_design}, "
                f"but the agent runs {reward_design}; retrain it with "
                f"{self._spec.simulator} train --reward-design {reward_design}"
            )
        context_dim = self._spec.context_dim
        if self._prior.input_dim != context_dim + action_dim:
            raise ValueError(
                f"Prior {prior_path} takes {self._prior.input_dim} inputs, "
                f"expected {context_dim + action_dim} (context + action)"
            )
        if action_dim == 1:
            self._grid = np.linspace(0.0, 1.0, int(grid_size))
        else:
            # One (T, W) row per pair, T major.
            timeouts = np.linspace(0.0, 1.0, int(grid_size))
            waits = np.linspace(0.0, 1.0, int(wait_grid_size))
            self._grid = np.column_stack(
                (np.repeat(timeouts, waits.size), np.tile(waits, timeouts.size))
            )
            self._grid_axes = (timeouts, waits)
        self._signal_var = float(signal_std) ** 2
        self._noise_var = float(noise_std) ** 2
        self._scales = np.asarray(
            self._spec.length_scales + (float(action_length_scale),) * action_dim,
            dtype=float,
        )
        self._memory_size = int(memory_size)
        self._confidence = float(confidence)
        # Shared by every timeout in a context: how far the prior's level can
        # be off there as a whole (the Tendermint prior's simulator holdout
        # RMSE, 0.204; another protocol's prior passes its own).
        self._level_var = float(level_std) ** 2
        self.reset_model()

    def _kernel(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = self._action_dim
        diff = (a[:, None, :] - b[None, :, :]) / self._scales
        sq = diff * diff
        context = np.exp(-0.5 * np.sum(sq[:, :, :-d], axis=2))
        # Product RBF over the action dimensions; at d = 1 the sum is the one term.
        action = np.exp(-0.5 * np.sum(sq[:, :, -d:], axis=2))
        return context * (self._level_var + self._signal_var * action)

    def _kernel_grid(self, a: np.ndarray, context: np.ndarray) -> np.ndarray:
        """``_kernel(a, _inputs(context, _grid))`` for d = 2, with the context factor
        computed once per row of ``a`` instead of once per (row, grid point), since
        every grid row carries the same context. Equal to ``_kernel`` up to rounding."""
        d = self._action_dim
        dc = (a[:, :-d] - context.reshape(1, -1)) / self._scales[:-d]
        context_factor = np.exp(-0.5 * np.sum(dc * dc, axis=1))
        # The grid is the product of its T and W axes (T major) and the action kernel a
        # product of per-axis RBFs, so it factors into a (rows, T) and a (rows, W) table.
        timeouts, waits = self._grid_axes
        dt = (a[:, -2, None] - timeouts[None, :]) / self._scales[-2]
        dw = (a[:, -1, None] - waits[None, :]) / self._scales[-1]
        action = (
            np.exp(-0.5 * dt * dt)[:, :, None] * np.exp(-0.5 * dw * dw)[:, None, :]
        ).reshape(len(a), -1)
        return context_factor[:, None] * (self._level_var + self._signal_var * action)

    def _inputs(self, context: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return np.hstack(
            (
                np.repeat(context.reshape(1, -1), len(actions), axis=0),
                actions.reshape(-1, self._action_dim),
            )
        )

    def _choose(self, context: np.ndarray) -> np.ndarray | float:
        """The chosen grid row: a float at d = 1, a (T, W) pair at d = 2."""
        grid_inputs = self._inputs(context, self._grid)
        prior = self._prior.predict(grid_inputs)
        if not self._residuals:
            return self._grid[int(np.argmax(prior))]

        x = np.asarray(self._inputs_seen, dtype=float)
        y = np.asarray(self._residuals, dtype=float)
        k_xx = self._kernel(x, x) + self._noise_var * np.eye(len(x))
        k_xg = (
            self._kernel(x, grid_inputs)
            if self._action_dim == 1
            else self._kernel_grid(x, context)
        )
        chol = np.linalg.cholesky(k_xx)
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
        v = (
            np.linalg.solve(chol, k_xg)
            if self._action_dim == 1
            else solve_triangular(chol, k_xg, lower=True)
        )
        mean = k_xg.T @ alpha
        prior_var = self._level_var + self._signal_var
        sd = np.sqrt(np.clip(prior_var - np.sum(v * v, axis=0), 0.0, None))
        return self._grid[int(np.argmax(prior + mean - self._confidence * sd))]

    def predict_action(self, features: np.ndarray) -> np.ndarray:
        row = np.asarray(features, dtype=float).reshape(-1)
        if row.size != self._obs_dim:
            raise ValueError(f"Expected {self._obs_dim} features, got {row.size}")
        action = self._choose(report_context(self._spec, self._field_names, row))
        # A copy, never a view of the grid.
        return np.array(action, dtype=float).reshape(-1)

    def update_model(
        self,
        features: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | float,
        backward: bool = True,
    ) -> float:
        row = np.asarray(features, dtype=float).reshape(-1)
        if row.size != self._obs_dim:
            raise ValueError(f"Expected {self._obs_dim} features, got {row.size}")
        action = np.asarray(actions, dtype=float).reshape(-1)
        if action.size != self._action_dim:
            raise ValueError(
                f"Expected {self._action_dim} action values, got {action.size}"
            )
        reward = float(np.asarray(rewards, dtype=float).reshape(-1)[0])
        inputs = np.append(report_context(self._spec, self._field_names, row), action)
        residual = reward - float(self._prior.predict(inputs.reshape(1, -1))[0])
        self._inputs_seen.append(inputs)
        self._residuals.append(residual)
        return residual

    def reset_model(self) -> None:
        self._inputs_seen: deque[np.ndarray] = deque(maxlen=self._memory_size)
        self._residuals: deque[float] = deque(maxlen=self._memory_size)


def return_prior_path(protocol: int) -> str:
    """Prior file of SimReturnGPModel for ``protocol``: the protocol's prior path
    with ``_return`` before the extension (continuous_tendermint_return_prior.npz)."""
    base, ext = os.path.splitext(context_spec(protocol).prior_path)
    if not base.endswith("_prior"):
        raise ValueError(f"Unexpected prior path {context_spec(protocol).prior_path}")
    return f"{base[:-len('_prior')]}_return_prior{ext}"


class SimReturnGPModel(SimPriorGPModel):
    """SimPriorGPModel for rewards made of throughput and latency only
    (reward_tps_latency): a floored, scale-free fit.

    Such a reward scores a collapse at -5 and lower, and a prior fit to the raw
    reward spends its error on those tails and loses the 0.1-sized structure
    that ranks the usable timeouts (the p3 context of the tps_latency H18 draw
    read flat from 450 to 3000 ms). The prior here is fit with a target floor
    (tendermint_sim train --target-floor): rewards below the floor count as
    the floor, a collapse is a collapse. The model applies the same floor to
    the online reward, so the residual compares like with like; above the
    floor the reward is untouched, so the ranking of usable timeouts is the
    reward's own.

    The GP residual, its kernel variances and the confidence bound are in units
    of the prior's training-target standard deviation, so the same constants
    serve a reward of any scale; H18's constants (signal 0.08, noise 0.04,
    level 0.2 on a reward of spread 0.6) are 0.13, 0.066 and 0.33 in these
    units, and level_std is wider here because this prior fits worse.

    The prior may also be fit to the held-action return (--hold-episodes h >
    1): the mean reward of holding the action h episodes from the context. On
    Tendermint that leans high at base under this reward, so its prior is per
    episode (hold 1). On SBFT the cost of a high T under a proposal delay lands
    at the delay's next re-pin, up to one interval after the T that will pay it
    was chosen, so a per-episode reward cannot see it. With h > 1 the online
    residual is taken against the matching target: the mean of the h rewards
    from that decision on, recorded h - 1 episodes later. At h = 1 the update
    is the per-episode one, unchanged. A term linear in the action was
    also tried in the residual (to extrapolate a slope to untried timeouts);
    replayed on real histories it changed nothing where the reward carried no
    slope and its extrapolations from two or three noisy points were wild, so
    the residual is H18's: a context-shared level plus a local action term.
    """

    def __init__(
        self,
        action_codec: ContinuousTimeoutActionCodec,
        field_names: Tuple[str, ...],
        prior_path: str,
        reward_design: str,
        grid_size: int = 64,
        signal_std: float = 0.13,
        noise_std: float = 0.066,
        action_length_scale: float = 0.1,
        memory_size: int = 256,
        confidence: float = 1.0,
        level_std: float = 0.5,
        wait_grid_size: int = 16,
    ):
        super().__init__(
            action_codec,
            field_names,
            prior_path,
            reward_design,
            grid_size=grid_size,
            signal_std=signal_std,
            noise_std=noise_std,
            action_length_scale=action_length_scale,
            memory_size=memory_size,
            confidence=confidence,
            level_std=level_std,
            wait_grid_size=wait_grid_size,
        )
        if not math.isfinite(self._prior.target_floor):
            raise ValueError(
                f"Prior {prior_path} was fit to the raw reward; SimReturnGPModel needs a "
                f"floored fit: {self._spec.simulator} train --target-floor -2 "
                f"--reward-design {reward_design}"
            )
        self._floor = float(self._prior.target_floor)
        self._y_scale = float(self._prior.y_std)
        if not math.isfinite(self._y_scale) or self._y_scale <= 0.0:
            raise ValueError(f"Prior {prior_path} has a non-positive y_std")
        self._hold = int(self._prior.hold_episodes)
        if self._hold < 1:
            raise ValueError(f"Prior {prior_path} has hold_episodes {self._hold}")
        # Decisions whose h-episode return is not complete yet, (inputs, raw reward),
        # are kept in self._pending (reset_model).

    @property
    def reward_floor(self) -> float:
        """Online rewards below this are taken as this, as in the prior's fit."""
        return self._floor

    def reset_model(self) -> None:
        super().reset_model()
        self._pending = deque()

    def _prior_scaled(self, inputs: np.ndarray) -> np.ndarray:
        return self._prior.predict(inputs) / self._y_scale

    def _choose(self, context: np.ndarray) -> np.ndarray | float:
        grid_inputs = self._inputs(context, self._grid)
        prior = self._prior_scaled(grid_inputs)
        if not self._residuals:
            return self._grid[int(np.argmax(prior))]
        x = np.asarray(self._inputs_seen, dtype=float)
        y = np.asarray(self._residuals, dtype=float)
        k_xx = self._kernel(x, x) + self._noise_var * np.eye(len(x))
        k_xg = (
            self._kernel(x, grid_inputs)
            if self._action_dim == 1
            else self._kernel_grid(x, context)
        )
        chol = np.linalg.cholesky(k_xx)
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
        v = (
            np.linalg.solve(chol, k_xg)
            if self._action_dim == 1
            else solve_triangular(chol, k_xg, lower=True)
        )
        mean = k_xg.T @ alpha
        prior_var = self._level_var + self._signal_var
        sd = np.sqrt(np.clip(prior_var - np.sum(v * v, axis=0), 0.0, None))
        return self._grid[int(np.argmax(prior + mean - self._confidence * sd))]

    def update_model(
        self,
        features: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | float,
        backward: bool = True,
    ) -> float:
        row = np.asarray(features, dtype=float).reshape(-1)
        if row.size != self._obs_dim:
            raise ValueError(f"Expected {self._obs_dim} features, got {row.size}")
        action = np.asarray(actions, dtype=float).reshape(-1)
        if action.size != self._action_dim:
            raise ValueError(
                f"Expected {self._action_dim} action values, got {action.size}"
            )
        inputs = np.append(report_context(self._spec, self._field_names, row), action)
        if self._hold == 1:
            reward = max(self._floor, float(np.asarray(rewards, dtype=float).reshape(-1)[0]))
            residual = reward / self._y_scale - float(self._prior_scaled(inputs.reshape(1, -1))[0])
            self._inputs_seen.append(inputs)
            self._residuals.append(residual)
            return residual
        # h > 1: the oldest pending decision gets the mean of its own reward and the
        # h - 1 that followed it, floored as the prior's targets were.
        self._pending.append((inputs, float(np.asarray(rewards, dtype=float).reshape(-1)[0])))
        if len(self._pending) < self._hold:
            return float("nan")
        target = max(self._floor, float(np.mean([r for _, r in self._pending])))
        oldest, _ = self._pending.popleft()
        residual = target / self._y_scale - float(self._prior_scaled(oldest.reshape(1, -1))[0])
        self._inputs_seen.append(oldest)
        self._residuals.append(residual)
        return residual
