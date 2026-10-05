from collections import deque
import math
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.special import betaln, digamma, polygamma
from scipy.stats import qmc
from sklearn.tree import DecisionTreeRegressor

import agent_pb2
from latency_floor import LatencyFloor
from models import (
    ActionCodec,
    PPOBase,
    TensorModel,
    get_timeout_values_ms,
    tendermint_timeouts_from_propose,
)


class ContinuousTimeoutActionCodec(ActionCodec[agent_pb2.Timeout]):
    """Maps normalized continuous actions to protocol timeout messages."""

    def __init__(self, protocol: int):
        self._protocol = int(protocol)
        # timeout_values = get_timeout_values_ms(self._protocol)
        # lower_ms = min(timeout_values)
        # upper_ms = max(timeout_values)
        lower_ms = 100
        upper_ms = 10000
        if lower_ms <= 0 or upper_ms <= lower_ms:
            raise ValueError(
                f"Invalid continuous timeout bounds: {lower_ms}..{upper_ms}ms"
            )

        if self._protocol in (
            agent_pb2.PROTOCOL_PBFT,
            agent_pb2.PROTOCOL_SMARTBFT,
            agent_pb2.PROTOCOL_TENDERMINT,
        ):
            self._action_dim = 1
        else:
            raise ValueError(
                "Continuous timeout actions currently support only PBFT, "
                "SmartBFT and Tendermint"
            )

        self._bounds_ms: Tuple[Tuple[int, int], ...] = tuple(
            (lower_ms, upper_ms) for _ in range(self._action_dim)
        )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def bounds_ms(self) -> Tuple[Tuple[int, int], ...]:
        return self._bounds_ms

    @property
    def protocol(self) -> int:
        return self._protocol

    def normalize_ms(self, timeout_ms: int, dimension: int = 0) -> float:
        """Public form of the log-space mapping that decode() inverts."""
        return self._normalize_value(int(timeout_ms), dimension)

    def denormalize_ms(self, normalized: float, dimension: int = 0) -> int:
        """Inverse of :meth:`normalize_ms`, without building a Timeout.

        A model that reasons in milliseconds - one whose action space is
        defined relative to a measured latency rather than to the codec's own
        bounds - needs to read back the timeout behind an action it is handed,
        which is otherwise only reachable through decode() and a protocol
        switch on the message it returns.
        """
        return self._denormalize_value(float(normalized), dimension)

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
            else:
                action = action.tendermint

        if self._protocol in (
            agent_pb2.PROTOCOL_PBFT,
            agent_pb2.PROTOCOL_SMARTBFT,
        ):
            timeout_values = [
                int(getattr(action, "election_timeout_milliseconds"))
            ]
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


# The floor already carries its own multiplier - LatencyFloor's k, set by
# --latency-floor-k - so this rule adds nothing on top of it by default: the
# timeout it plays is the floor value itself.
DEFAULT_FLOOR_MULTIPLIER = 1.0


class LatencyFloorHeuristic(TensorModel):
    """Plays the latency floor as the timeout.

        timeout_ms = clamp(multiplier * floor_ms, clamp_ms)

    ``clamp_ms`` defaults to the codec's own bounds, but the agent passes the
    discrete arm range, as it does for ContinuousHeuristicTimeoutModel: the
    codec's lower bound is 100ms, well under the shortest arm any of these
    protocols ships, so a floor measured in a healthy phase would otherwise
    pin the timeout at a value no discrete baseline can play and the
    comparison would be against a different action space.

    Not a bandit: nothing is explored, nothing is learned, and the same
    measurement produces the same timeout every time. Where
    :class:`~latency_floor.LatencyFloorPolicy` uses the floor to rule arms out
    and leaves a learner to choose among what is left, this takes the floor as
    the answer, so the constraint and the decision are one number.

    Its difference from :class:`ContinuousHeuristicTimeoutModel` is the
    estimator, not the shape of the rule. That model multiplies the current
    window's mean latency, so a single degraded window moves the timeout;
    this one multiplies a low quantile over the last several committing
    windows, which is the same estimate the floor constraint bounds with -
    slower to follow a regime change, and much harder for one bad window to
    carry.
    """

    def __init__(
        self,
        action_codec: ContinuousTimeoutActionCodec,
        n_features: int,
        latency_floor: NormalizedLatencyFloor,
        multiplier: float = DEFAULT_FLOOR_MULTIPLIER,
        clamp_ms: Optional[Tuple[int, int]] = None,
        initial_timeout_ms: Optional[int] = None,
    ):
        if n_features <= 0:
            raise ValueError(f"n_features must be > 0, got {n_features}")
        if latency_floor is None:
            raise ValueError(
                "LatencyFloorHeuristic requires a NormalizedLatencyFloor: the "
                "floor is the rule, not an optional constraint on one"
            )
        if action_codec.action_dim != 1:
            raise ValueError(
                "LatencyFloorHeuristic scales a single timeout, so it needs a "
                f"1-dimensional codec, got {action_codec.action_dim}"
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
            # Nothing has been measured yet, so there is nothing to scale. The
            # middle discrete arm is the same cold start
            # ContinuousHeuristicTimeoutModel takes, and it lasts until the
            # first window that commits something.
            arms_ms = sorted(get_timeout_values_ms(action_codec.protocol))
            initial_timeout_ms = arms_ms[len(arms_ms) // 2]
        initial_timeout_ms = min(upper_ms, max(lower_ms, int(initial_timeout_ms)))

        self._action_codec = action_codec
        self._obs_dim = int(n_features)
        self._latency_floor = latency_floor
        self._multiplier = float(multiplier)
        self._lower_ms = lower_ms
        self._upper_ms = upper_ms
        self._initial_timeout_ms = initial_timeout_ms

        self.reset_model()

    @property
    def multiplier(self) -> float:
        return self._multiplier

    @property
    def floor_ms(self) -> float:
        return float(self._latency_floor.floor_ms)

    @property
    def timeout_ms(self) -> int:
        """Milliseconds behind the most recent decision."""
        return self._held_timeout_ms

    def _timeout_for(self, features: np.ndarray) -> int:
        self._latency_floor.observe_features(features)
        floor_ms = float(self._latency_floor.floor_ms)
        if floor_ms <= 0.0:
            # No committing window has been seen yet - a cold start, or a
            # stalled one. Hold the last timeout rather than inventing a floor.
            return self._held_timeout_ms
        timeout_ms = int(round(self._multiplier * floor_ms))
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
        """No-op: the rule has no parameters to move.

        The reward still reaches the agent's CSV, so a run of this model is
        directly comparable with a learned one on the same episodes.
        """
        return 0.0

    def reset_model(self) -> None:
        self._latency_floor.reset()
        self._held_timeout_ms = self._initial_timeout_ms



# Candidate timeouts scored at each decision. Spread over the arm range in the
# codec's log space, 65 points put neighbours within about 2.5% of each other -
# far finer than a single 5s window can resolve, so the grid is not what limits
# the choice.
DEFAULT_CANDIDATE_COUNT = 65


def _refined_ladder_ms(arms_ms: Sequence[int]) -> Tuple[int, ...]:
    """The arm ladder with the geometric midpoint of each pair inserted.

    Geometric rather than arithmetic because the action space is log-spaced:
    the midpoint of 250 and 550 is the timeout equidistant from both in the
    coordinate the model actually searches.
    """
    arms = sorted(int(value) for value in arms_ms)
    refined = []
    for lower, upper in zip(arms, arms[1:]):
        refined.append(lower)
        refined.append(int(round(math.sqrt(float(lower) * float(upper)))))
    refined.append(arms[-1])
    return tuple(sorted(set(refined)))


class ContinuousLinUCB(TensorModel):
    """Discounted LinUCB over a continuous timeout.

    ``DiscountedLinUCB`` (``codex_v1``) fits one ridge model per arm, which is
    the same thing as one shared model over ``one_hot(arm)`` outer-product the
    context. This is that model with the one-hot replaced by a smooth basis
    over the log-timeout::

        z(x, a) = phi(a) (x) x          (Kronecker product)

    Everything else is carried over unchanged: the signed-log row-normalized
    context, the ridge prior, the discount on every statistic, the UCB
    half-width from the discounted covariance sandwich, the random exploration
    floor, and playing each basis centre once before anything is fitted, which
    is what codex_v1's ``+inf`` score for an unpulled arm amounts to. Place the
    centres at the arms and shrink their width to zero and the two models are
    the same model; the parameter count is already identical, ``len(centres) *
    (n_features + 1)``.

    What the smooth basis buys is that neighbouring timeouts share evidence.
    On the ladder an arm is only informed by its own pulls, so an arm avoided
    for a phase keeps a stale estimate and comes back on the confidence bonus
    alone. Here a reward earned at 600ms moves the estimate at 550ms and 700ms
    with it, and the exploration bonus is a bonus for a *region* of the action
    space that has not been tried lately.

    **The basis sums to one at every timeout, and this is not cosmetic.** The
    prediction is then ``sum_j w_j(a) (theta_j . x)``, a weighted average of
    the per-centre reward models, so it cannot exceed the best centre and the
    fitted optimum is always at one. Normalizing the bumps to unit L2 norm
    instead - which looks more like the one-hot it replaces, and keeps the
    confidence bonus from favouring the ends - is a trap: the prediction
    becomes an inner product that rewards spreading weight over several
    positive coefficients, so a blend between two decent centres scores above
    either of them, and since the coefficients are built from the action being
    played, whatever the model plays becomes its own argmax. Measured on
    local/pbft_c6525100g_utah_2/continuous_codex_mixed, that locked the agent
    at 370ms - a timeout no baseline plays - for 104 of 107 episodes and cost
    4600 tps against the same model's heuristic.

    **The centres are the arms plus the midpoint between each pair.** With
    centres only at the arms, an average over them is maximised at an arm and
    the continuous action space buys nothing; with the ladder refined one
    level the model can settle between two arms when that is where the reward
    is (measured on a synthetic reward paying only in 700-900ms: 90% of
    decisions inside it with the midpoints, 0% without). Each centre costs
    ``n_features + 1`` coefficients, which is what limits how fine the ladder
    can get: five centres is 85 against codex_v1's 51.

    The bumps are half a centre-spacing wide, so neighbours overlap at about
    14%: wide enough to share evidence, narrow enough that adjacent centres
    still say different things.

    **The one departure from codex_v1: rewards are centred.** The ridge prior
    is zero, and the discount pulls an unplayed timeout's estimate toward it,
    so zero is what an untried timeout is assumed to be worth. That is only
    sane if the rewards are centred on zero, and they are not: reward_tps is
    1.0 at its 1000 tps knee and this cluster runs at 40000, so a reward is 2
    to 10 and a forgotten timeout reads as worse than the worst thing ever
    measured. Measured on the first run of
    local/pbft_c6525100g_utah_2/continuous_codex_mixed: by the last episode
    the timeout being played scored 5.35 against 0.11 for the longest one,
    while the confidence bonus meant to bring it back tops out at 1.24 - so
    the agent held one timeout for 102 of 107 episodes and never answered a
    phase change. Subtracting a discounted mean of recent rewards makes the
    prior mean "average", which is what optimism in the face of uncertainty
    assumes: a timeout not tried lately is worth finding out about, not
    written off. codex_v1 has the same pathology and is deliberately left
    alone, so the ladder it plays is the one it has always played; pass
    ``center_rewards=False`` to reproduce it here.

    Centring is also why the discount is 0.90 rather than codex_v1's 0.95.
    The effective memory is ``1 / (1 - gamma)`` observations, and a phase of
    this spec is about 15 episodes at a 5s window, so 0.95 remembers 20 - it
    cannot forget a phase inside that phase. 0.95 was nonetheless right for
    codex_v1, and measurably so: with an uncentred reward, forgetting decays
    an estimate toward "worthless", so faster forgetting buries the arm it was
    supposed to free. Once the prior means "average", the two pull the same
    way. On a seven-phase benchmark whose paying timeout moves under the agent
    (3 seeds, 15 episodes per phase, rewards on this hardware's 2..10 scale),
    total regret: uncentred 434 at 0.95 and 456 at 0.85 - the ordering the
    hardware showed for codex_v1 - against centred 320 at 0.95 and 184 at
    0.90.

    **Neither of those two fixes answers how the agent got short in the first
    place, and nothing in this class does.** The initial design plays every
    centre once, and on that run all five decisions landed inside phase 0,
    where the client's offered rate caps throughput: the five timeouts read
    39114..40049 tps, a 2.4% spread, so reward_mixed's
    ``window / (window + timeout + latency)`` discount was the only signal
    left in the sweep - 9.85, 9.81, 9.27, 8.94, 8.34 at 250, 371, 550, 812 and
    1200ms. The sweep fits the reward's own penalty gradient rather than the
    protocol, and the way back up is then the exploration floor alone: 0.05
    over a log-spaced grid whose span above 500ms is 55.8% of it is 2.8% per
    decision, or 0.42 expected probes above 500ms in a 15-episode phase.
    Phases 1, 3, 4 and 6 of that run drew none; the 497ms probe that landed in
    phase 5 scored 2.66 against 1.95 for the twelve episodes the held timeout
    played there and was one sample against forty. So the fault phases are
    never sampled at the length that pays - phase 4 committed 9641 tps here
    against 13234 for the same model at 550ms under reward_tps and 15139 for
    the oracle. Interleaving the design sweep across phases, and drawing the
    exploration floor over the centres rather than over the candidate grid,
    are the two things that would change that; neither is done here.

    The action space is the arm range, not the codec's 100..10000ms bounds, for
    the reason :class:`LatencyFloorHeuristic` gives: a timeout no discrete
    baseline can play makes the comparison against those baselines a comparison
    of two different action spaces. A latency floor, when one is configured,
    trims the bottom of that range exactly as ``feasible_timeout_mask`` trims
    the ladder - the model is not otherwise aware of it, and with no floor
    configured it is a bandit over the range and nothing else.
    """

    def __init__(
        self,
        action_codec: ContinuousTimeoutActionCodec,
        n_features: int,
        basis_centers_ms: Optional[Sequence[int]] = None,
        clamp_ms: Optional[Tuple[int, int]] = None,
        candidate_count: int = DEFAULT_CANDIDATE_COUNT,
        confidence_scale: float = 1.0,
        discount_factor: float = 0.90,
        exploration_floor: float = 0.05,
        ridge: float = 1.0,
        reward_scale: float = 1.0,
        center_rewards: bool = True,
        latency_floor: Optional[NormalizedLatencyFloor] = None,
        seed: Optional[int] = None,
    ) -> None:
        if n_features <= 0:
            raise ValueError(f"n_features must be > 0, got {n_features}")
        if action_codec.action_dim != 1:
            raise ValueError(
                "ContinuousLinUCB places one timeout on a line, so it needs a "
                f"1-dimensional codec, got {action_codec.action_dim}"
            )
        if candidate_count < 2:
            raise ValueError(
                f"candidate_count must be >= 2, got {candidate_count}"
            )
        if confidence_scale < 0.0:
            raise ValueError(
                f"confidence_scale must be >= 0, got {confidence_scale}"
            )
        if not 0.0 < discount_factor <= 1.0:
            raise ValueError(
                f"discount_factor must be in (0, 1], got {discount_factor}"
            )
        if not 0.0 <= exploration_floor <= 1.0:
            raise ValueError(
                f"exploration_floor must be in [0, 1], got {exploration_floor}"
            )
        if ridge <= 0.0:
            raise ValueError(f"ridge must be > 0, got {ridge}")
        if reward_scale <= 0.0:
            raise ValueError(f"reward_scale must be > 0, got {reward_scale}")

        arms_ms = tuple(
            sorted(int(value) for value in get_timeout_values_ms(action_codec.protocol))
        )
        codec_lower_ms, codec_upper_ms = action_codec.bounds_ms[0]
        if clamp_ms is None:
            clamp_ms = (min(arms_ms), max(arms_ms))
        lower_ms, upper_ms = (int(clamp_ms[0]), int(clamp_ms[1]))
        if upper_ms <= lower_ms:
            raise ValueError(f"Invalid clamp bounds: {lower_ms}..{upper_ms}ms")
        if lower_ms < codec_lower_ms or upper_ms > codec_upper_ms:
            raise ValueError(
                f"Clamp bounds {lower_ms}..{upper_ms}ms fall outside the codec "
                f"bounds {codec_lower_ms}..{codec_upper_ms}ms"
            )

        centers_ms = tuple(
            sorted(
                int(value)
                for value in (basis_centers_ms or _refined_ladder_ms(arms_ms))
            )
        )
        if not centers_ms:
            raise ValueError("basis_centers_ms must not be empty")
        if centers_ms[0] < lower_ms or centers_ms[-1] > upper_ms:
            raise ValueError(
                f"Basis centres {centers_ms} fall outside the action range "
                f"{lower_ms}..{upper_ms}ms"
            )

        self._action_codec = action_codec
        self._obs_dim = int(n_features)
        self._lower_ms = lower_ms
        self._upper_ms = upper_ms
        self._centers_ms = centers_ms
        self._candidate_count = int(candidate_count)
        self._confidence_scale = float(confidence_scale)
        self._discount_factor = float(discount_factor)
        self._exploration_floor = float(exploration_floor)
        self._ridge = float(ridge)
        self._reward_scale = float(reward_scale)
        self._center_rewards = bool(center_rewards)
        self._latency_floor = latency_floor
        self._seed = seed

        self._context_dim = self._obs_dim + 1
        self._centers = np.asarray(
            [self._position_of(value) for value in centers_ms], dtype=np.float64
        )
        self._n_basis = len(self._centers)
        # Half the mean centre spacing; see the class docstring for why not
        # a full one.
        self._basis_width = (
            1.0
            if self._n_basis < 2
            else float(
                0.5
                * (self._centers[-1] - self._centers[0])
                / (self._n_basis - 1)
            )
        )
        if self._basis_width <= 0.0:
            self._basis_width = 1.0
        self._feature_dim = self._n_basis * self._context_dim
        self._positions = np.linspace(
            0.0, 1.0, self._candidate_count, dtype=np.float64
        )

        self.reset_model()

    @property
    def timeout_ms(self) -> int:
        """Milliseconds behind the most recent decision."""
        return self._held_timeout_ms

    @property
    def floor_ms(self) -> float:
        return 0.0 if self._latency_floor is None else self._latency_floor.floor_ms

    @property
    def observation_count(self) -> int:
        return int(self._observation_count)

    @property
    def baseline_reward(self) -> float:
        """The reward level an untried timeout is credited with."""
        return self._baseline()

    def _position_of(self, timeout_ms: float) -> float:
        """Where a timeout sits in the action range, in [0, 1] log-space."""
        value = float(timeout_ms)
        if not math.isfinite(value) or value <= 0.0:
            return 0.0
        position = (math.log(value) - math.log(float(self._lower_ms))) / (
            math.log(float(self._upper_ms)) - math.log(float(self._lower_ms))
        )
        return float(min(1.0, max(0.0, position)))

    def _timeout_of(self, position: float) -> int:
        bounded = min(1.0, max(0.0, float(position)))
        log_value = math.log(float(self._lower_ms)) + bounded * (
            math.log(float(self._upper_ms)) - math.log(float(self._lower_ms))
        )
        return min(
            self._upper_ms, max(self._lower_ms, int(round(math.exp(log_value))))
        )

    def _lower_position(self) -> float:
        """Smallest position the latency floor permits, in [0, 1].

        A floor above the whole range leaves the longest timeout, which is what
        feasible_timeout_mask leaves on the discrete ladder.
        """
        if self._latency_floor is None:
            return 0.0
        floor_ms = float(self._latency_floor.floor_ms)
        if not math.isfinite(floor_ms) or floor_ms <= self._lower_ms:
            return 0.0
        if floor_ms >= self._upper_ms:
            return 1.0
        return self._position_of(math.ceil(floor_ms))

    def _candidates(self, lower_position: float) -> np.ndarray:
        if lower_position <= 0.0:
            return self._positions
        feasible = self._positions[self._positions >= lower_position]
        if len(feasible) == 0:
            return np.asarray([lower_position], dtype=np.float64)
        # The bound itself is a legal timeout and is the shortest one left, so
        # it is offered even when the grid steps over it.
        return np.concatenate(([lower_position], feasible))

    def _context(self, row: np.ndarray) -> np.ndarray:
        values = np.asarray(row, dtype=np.float64).reshape(-1)
        if values.size != self._obs_dim:
            raise ValueError(
                f"Expected {self._obs_dim} features, got {values.size}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("features must contain only finite values")
        transformed = np.sign(values) * np.log1p(np.abs(values))
        norm = float(np.linalg.norm(transformed))
        transformed = transformed / max(norm, 1.0)
        return np.concatenate(([1.0], transformed))

    def _basis(self, positions: np.ndarray) -> np.ndarray:
        values = np.asarray(positions, dtype=np.float64).reshape(-1, 1)
        distances = (values - self._centers.reshape(1, -1)) / self._basis_width
        weights = np.exp(-0.5 * distances * distances)
        totals = np.sum(weights, axis=1, keepdims=True)
        return weights / np.maximum(totals, 1e-12)

    def _joint(self, context: np.ndarray, positions: np.ndarray) -> np.ndarray:
        basis = self._basis(positions)
        return basis[:, :, None] * context[None, None, :]

    def _scores(self, context: np.ndarray, positions: np.ndarray) -> np.ndarray:
        inputs = self._joint(context, positions).reshape(len(positions), -1)
        try:
            theta = np.linalg.solve(self._precision, self._reward_vector)
            directions = np.linalg.solve(self._precision, inputs.T).T
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "continuous_codex precision matrix is singular"
            ) from exc
        predicted = inputs @ theta
        variances = np.einsum(
            "ij,jk,ik->i", directions, self._covariance, directions
        )
        return predicted + self._confidence_scale * np.sqrt(
            np.maximum(variances, 0.0)
        )

    def _select_position(
        self, context: np.ndarray, lower_position: float
    ) -> float:
        if self._design_index < self._n_basis:
            # One decision per basis centre before anything is fitted: the
            # continuous form of codex_v1 scoring an unpulled arm at +inf, and
            # deterministic, so replicas sharing a feature vector still draw
            # the same action.
            position = float(self._centers[self._design_index])
            self._design_index += 1
            return max(lower_position, position)

        candidates = self._candidates(lower_position)
        if self._exploration_floor > 0.0 and self._rng.random() < (
            self._exploration_floor
        ):
            return float(candidates[int(self._rng.integers(len(candidates)))])

        scores = self._scores(context, candidates)
        best = float(np.max(scores))
        tied = np.flatnonzero(scores >= best - 1e-12)
        return float(candidates[int(self._rng.choice(tied))])

    def predict_action(self, features: np.ndarray) -> np.ndarray:
        rows = np.asarray(features, dtype=float)
        single = rows.ndim == 1
        if single:
            rows = rows.reshape(1, -1)
        if rows.shape[1] != self._obs_dim:
            raise ValueError(
                f"Expected {self._obs_dim} features, got {rows.shape[1]}"
            )

        actions = []
        for row in rows:
            if self._latency_floor is not None:
                self._latency_floor.observe_features(row)
            position = self._select_position(
                self._context(row), self._lower_position()
            )
            self._held_timeout_ms = self._timeout_of(position)
            actions.append([self._action_codec.normalize_ms(self._held_timeout_ms)])

        result = np.asarray(actions, dtype=float)
        return result[0] if single else result

    def _baseline(self) -> float:
        """What an unmeasured timeout is assumed to be worth.

        Zero - the ridge prior itself - when centring is off, which is what
        codex_v1 does.
        """
        if not self._center_rewards or self._baseline_weight <= 0.0:
            return 0.0
        return self._baseline_total / self._baseline_weight

    def _observe_reward(self, reward: float) -> None:
        """Fold one reward into the baseline, discounted like everything else."""
        self._baseline_total = (
            self._discount_factor * self._baseline_total + reward
        )
        self._baseline_weight = self._discount_factor * self._baseline_weight + 1.0

    def _discount(self) -> None:
        """One observation's decay toward the ridge prior.

        Precision at gamma and covariance at gamma squared, as in
        DiscountedLinUCB: that is what keeps the confidence half-width a
        variance rather than a count of pulls.
        """
        identity = np.eye(self._feature_dim, dtype=np.float64)
        prior_weight = (1.0 - self._discount_factor) * self._ridge
        squared_discount = self._discount_factor * self._discount_factor
        covariance_prior_weight = (1.0 - squared_discount) * self._ridge
        self._precision *= self._discount_factor
        if prior_weight > 0.0:
            self._precision += prior_weight * identity
        self._covariance *= squared_discount
        if covariance_prior_weight > 0.0:
            self._covariance += covariance_prior_weight * identity
        self._reward_vector *= self._discount_factor

    def update_model(
        self,
        features: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | float,
        backward: bool = True,
    ) -> float:
        if not backward:
            return 0.0

        rows = np.asarray(features, dtype=float)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.shape[1] != self._obs_dim:
            raise ValueError(
                f"Expected {self._obs_dim} features, got {rows.shape[1]}"
            )

        action_values = np.asarray(actions, dtype=np.float64).reshape(-1)
        reward_values = np.asarray(rewards, dtype=np.float64).reshape(-1)
        batch_size = len(rows)
        if action_values.size == 1 and batch_size > 1:
            action_values = np.repeat(action_values, batch_size)
        if reward_values.size == 1 and batch_size > 1:
            reward_values = np.repeat(reward_values, batch_size)
        if action_values.size != batch_size:
            raise ValueError(
                f"actions batch size mismatch: expected {batch_size}, "
                f"got {action_values.size}"
            )
        if reward_values.size != batch_size:
            raise ValueError(
                f"rewards batch size mismatch: expected {batch_size}, "
                f"got {reward_values.size}"
            )
        if not np.all(np.isfinite(action_values)):
            raise ValueError("actions must contain only finite values")
        if np.any(action_values < 0.0) or np.any(action_values > 1.0):
            raise ValueError("Continuous actions must be normalized to [0, 1]")
        if not np.all(np.isfinite(reward_values)):
            raise ValueError("rewards must contain only finite values")

        for row, action, reward in zip(rows, action_values, reward_values):
            # The agent trains on the timeout the replicas actually ran with,
            # which is not always the one this model asked for; a timeout
            # outside the action range lands on the nearest end of it.
            timeout_ms = self._action_codec.denormalize_ms(float(action))
            inputs = self._joint(
                self._context(row), np.asarray([self._position_of(timeout_ms)])
            ).reshape(-1)
            centered = float(reward) - self._baseline()
            self._discount()
            self._precision += np.outer(inputs, inputs)
            self._covariance += np.outer(inputs, inputs)
            self._reward_vector += (centered / self._reward_scale) * inputs
            self._observe_reward(float(reward))
            self._observation_count += 1

        return 0.0

    def reset_model(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        if self._latency_floor is not None:
            self._latency_floor.reset()
        identity = np.eye(self._feature_dim, dtype=np.float64)
        self._precision = self._ridge * identity
        self._covariance = self._precision.copy()
        self._reward_vector = np.zeros(self._feature_dim, dtype=np.float64)
        self._observation_count = 0
        self._design_index = 0
        self._baseline_total = 0.0
        self._baseline_weight = 0.0
        self._held_timeout_ms = self._lower_ms


class ContinuousPPO(PPOBase):
    """PPO with a Beta policy over the continuous timeout.

    :class:`DiscretePPO`'s softmax is replaced by a Beta density over the
    position of the timeout in the arm range, measured in log space exactly as
    :class:`ContinuousLinUCB` measures it. The actor emits two numbers,
    ``alpha = 1 + softplus(z0)`` and ``beta = 1 + softplus(z1)``; holding both
    at or above one keeps the density unimodal and finite at the ends, so the
    policy can concentrate anywhere in the range but cannot put an atom on a
    boundary and get stuck there.

    **Beta rather than a clipped Gaussian.** The action is bounded on both
    sides and the obvious alternative - sample a Gaussian, clip to the range -
    piles the clipped mass onto the two end points and reports a density that
    does not integrate to what was actually sampled, which shows up as a biased
    importance ratio precisely at the ends of the ladder where this spec's
    interesting arms live. The Beta is supported on exactly the interval being
    sampled, so no correction is needed.

    **It starts almost uniform, and that is the point.** The output bias is set
    so that ``alpha = beta ~ 1.13`` before any learning: a very broad bump over
    the whole arm range. ``ContinuousLinUCB`` instead opens with a design sweep
    of one decision per basis centre, and its own docstring records what that
    cost - all five landed inside phase 0, where the offered load caps
    throughput and the five timeouts differ by 2.4%, so the sweep fitted the
    reward's penalty gradient rather than the protocol and the model never
    recovered. A broad policy keeps sampling the whole range as the phases go
    by, so the long timeouts get tried inside the phases that pay for them.

    **The action space is the arm range, not the codec's 100..10000ms.** Same
    reason :class:`ContinuousLinUCB` and :class:`LatencyFloorHeuristic` give: a
    timeout no discrete baseline can reach turns a comparison against those
    baselines into a comparison of two different action spaces. A latency
    floor, when configured, raises the bottom of the range; the sampled
    position is clamped up to it, which puts a small atom on the floor itself,
    and that is treated as part of the environment in the same way the discrete
    models treat ``feasible_timeout_mask``.
    """

    def __init__(
        self,
        action_codec: ContinuousTimeoutActionCodec,
        n_features: int,
        clamp_ms: Optional[Tuple[int, int]] = None,
        hidden_sizes: Sequence[int] = (),
        latency_floor: Optional[NormalizedLatencyFloor] = None,
        initial_concentration_bias: float = -2.0,
        **kwargs: Any,
    ) -> None:
        if action_codec.action_dim != 1:
            raise ValueError(
                "ContinuousPPO places one timeout on a line, so it needs a "
                f"1-dimensional codec, got {action_codec.action_dim}"
            )
        self._action_codec = action_codec
        self._latency_floor = latency_floor
        self._initial_concentration_bias = float(initial_concentration_bias)

        codec_lower_ms, codec_upper_ms = action_codec.bounds_ms[0]
        if clamp_ms is None:
            arms_ms = tuple(
                int(value) for value in get_timeout_values_ms(action_codec.protocol)
            )
            clamp_ms = (min(arms_ms), max(arms_ms))
        lower_ms, upper_ms = int(clamp_ms[0]), int(clamp_ms[1])
        if lower_ms < codec_lower_ms or upper_ms > codec_upper_ms:
            raise ValueError(
                f"clamp {lower_ms}..{upper_ms}ms is outside the codec bounds "
                f"{codec_lower_ms}..{codec_upper_ms}ms"
            )
        if upper_ms <= lower_ms:
            raise ValueError(
                f"clamp upper bound must exceed lower, got {lower_ms}..{upper_ms}"
            )
        self._lower_ms = lower_ms
        self._upper_ms = upper_ms
        self._log_lower = math.log(float(lower_ms))
        self._log_span = math.log(float(upper_ms)) - self._log_lower

        super().__init__(
            n_features=n_features,
            head_dim=2,
            hidden_sizes=hidden_sizes,
            **kwargs,
        )

    @property
    def _output_bias(self) -> float:
        return self._initial_concentration_bias

    # -- diagnostics --------------------------------------------------------

    @property
    def timeout_ms(self) -> int:
        """Milliseconds behind the most recent decision."""
        return self._held_timeout_ms

    @property
    def floor_ms(self) -> float:
        return 0.0 if self._latency_floor is None else self._latency_floor.floor_ms

    # -- action range -------------------------------------------------------

    def _position_of(self, timeout_ms: float) -> float:
        value = float(timeout_ms)
        if not math.isfinite(value) or value <= 0.0:
            return 0.0
        position = (math.log(value) - self._log_lower) / self._log_span
        return float(min(1.0, max(0.0, position)))

    def _timeout_of(self, position: float) -> int:
        bounded = min(1.0, max(0.0, float(position)))
        value = math.exp(self._log_lower + bounded * self._log_span)
        return min(self._upper_ms, max(self._lower_ms, int(round(value))))

    def _lower_position(self) -> float:
        """Smallest position the latency floor permits, in [0, 1]."""
        if self._latency_floor is None:
            return 0.0
        floor_ms = float(self._latency_floor.floor_ms)
        if not math.isfinite(floor_ms) or floor_ms <= self._lower_ms:
            return 0.0
        if floor_ms >= self._upper_ms:
            return 1.0
        return self._position_of(math.ceil(floor_ms))

    # -- Beta head ----------------------------------------------------------

    def _concentrations(self, head: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(alpha, beta) = 1 + softplus(head), computed without overflow."""
        values = np.asarray(head, dtype=np.float64)
        softplus = np.logaddexp(0.0, values)
        concentrations = 1.0 + softplus
        return concentrations[:, 0], concentrations[:, 1]

    def _head_terms(
        self, head: np.ndarray, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        alpha, beta = self._concentrations(head)
        positions = np.clip(np.asarray(actions, dtype=np.float64), 1e-6, 1.0 - 1e-6)
        log_u = np.log(positions)
        log_1mu = np.log1p(-positions)

        log_density = (
            (alpha - 1.0) * log_u + (beta - 1.0) * log_1mu - betaln(alpha, beta)
        )
        density = np.exp(np.clip(log_density, -700.0, 700.0))
        floor = self._exploration_floor
        mixture = (1.0 - floor) * density + floor
        log_probs = np.log(np.maximum(mixture, 1e-300))

        # d log f / d alpha and d log f / d beta for the Beta density.
        digamma_sum = digamma(alpha + beta)
        d_log_density_alpha = log_u - (digamma(alpha) - digamma_sum)
        d_log_density_beta = log_1mu - (digamma(beta) - digamma_sum)
        # The mixture only rescales them: d log q = (1-e) f d log f / q.
        share = (1.0 - floor) * density / np.maximum(mixture, 1e-300)
        d_log_alpha = share * d_log_density_alpha
        d_log_beta = share * d_log_density_beta

        # Entropy of the Beta itself. The uniform component only ever adds
        # entropy, so bonusing the Beta's own is the conservative choice and,
        # unlike the mixture's, it is closed form.
        entropy = (
            betaln(alpha, beta)
            - (alpha - 1.0) * digamma(alpha)
            - (beta - 1.0) * digamma(beta)
            + (alpha + beta - 2.0) * digamma_sum
        )
        trigamma_sum = polygamma(1, alpha + beta)
        d_entropy_alpha = -(alpha - 1.0) * polygamma(1, alpha) + (
            alpha + beta - 2.0
        ) * trigamma_sum
        d_entropy_beta = -(beta - 1.0) * polygamma(1, beta) + (
            alpha + beta - 2.0
        ) * trigamma_sum

        # Chain through alpha = 1 + softplus(z): d alpha / d z = sigmoid(z).
        slope = 1.0 / (1.0 + np.exp(-np.asarray(head, dtype=np.float64)))
        d_log_probs = np.stack(
            (d_log_alpha * slope[:, 0], d_log_beta * slope[:, 1]), axis=1
        )
        d_entropy = np.stack(
            (d_entropy_alpha * slope[:, 0], d_entropy_beta * slope[:, 1]), axis=1
        )
        return log_probs, entropy, d_log_probs, d_entropy

    # -- TensorModel --------------------------------------------------------

    def predict_action(self, features: np.ndarray) -> np.ndarray:
        rows = np.asarray(features, dtype=float)
        single = rows.ndim == 1
        if single:
            rows = rows.reshape(1, -1)
        if rows.shape[1] != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} features, got {rows.shape[1]}"
            )

        actions = []
        for row in rows:
            if self._latency_floor is not None:
                self._latency_floor.observe_features(row)
            head = self._actor_head(self._contexts(row.reshape(1, -1)))
            alpha, beta = self._concentrations(head)
            if (
                self._exploration_floor > 0.0
                and self._rng.random() < self._exploration_floor
            ):
                position = float(self._rng.random())
            else:
                position = float(self._rng.beta(float(alpha[0]), float(beta[0])))
            position = max(position, self._lower_position())
            self._held_timeout_ms = self._timeout_of(position)
            actions.append(
                [self._action_codec.normalize_ms(self._held_timeout_ms)]
            )

        result = np.asarray(actions, dtype=float)
        return result[0] if single else result

    def _store_action(self, action: float) -> float:
        """Codec-normalized action -> position in the arm range.

        The agent reports the timeout the replicas ran with, normalized against
        the codec's own 100..10000ms bounds. The policy lives on the arm range,
        so the millisecond value is recovered and re-expressed there; a timeout
        outside the arm range lands on the nearest end of it, which is what
        clamping already did on the way out.
        """
        value = float(action)
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Continuous actions must be normalized to [0, 1], got {value}"
            )
        return self._position_of(self._action_codec.denormalize_ms(value))

    def reset_model(self) -> None:
        super().reset_model()
        if self._latency_floor is not None:
            self._latency_floor.reset()
        self._held_timeout_ms = self._lower_ms
