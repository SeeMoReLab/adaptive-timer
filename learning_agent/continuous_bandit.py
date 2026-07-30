from collections import deque
from itertools import product
import math
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.tree import DecisionTreeRegressor
import torch

import agent_pb2
from models import ActionCodec, TensorModel, get_timeout_values_ms


class ContinuousTimeoutActionCodec(ActionCodec[agent_pb2.Timeout]):
    """Maps normalized continuous actions to protocol timeout messages."""

    def __init__(self, protocol: int):
        self._protocol = int(protocol)
        timeout_values = get_timeout_values_ms(self._protocol)
        lower_ms = min(timeout_values)
        upper_ms = max(timeout_values)
        if lower_ms <= 0 or upper_ms <= lower_ms:
            raise ValueError(
                f"Invalid continuous timeout bounds: {lower_ms}..{upper_ms}ms"
            )

        if self._protocol == agent_pb2.PROTOCOL_PBFT:
            self._action_dim = 1
        elif self._protocol == agent_pb2.PROTOCOL_TENDERMINT:
            self._action_dim = 3
        else:
            raise ValueError(
                "Continuous timeout actions currently support only PBFT and Tendermint"
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

    def decode(self, action: torch.Tensor) -> agent_pb2.Timeout:
        values = action.detach().cpu().to(torch.float64).reshape(-1)
        if values.numel() != self._action_dim:
            raise ValueError(
                f"Expected {self._action_dim} continuous action values, "
                f"got {values.numel()}"
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
        return agent_pb2.Timeout(
            tendermint=agent_pb2.TendermintTimeout(
                propose_timeout_milliseconds=timeout_values[0],
                prevote_timeout_milliseconds=timeout_values[1],
                precommit_timeout_milliseconds=timeout_values[2],
            )
        )

    def encode(self, action: object) -> torch.Tensor:
        if isinstance(action, agent_pb2.Timeout):
            if self._protocol == agent_pb2.PROTOCOL_PBFT:
                action = action.pbft
            else:
                action = action.tendermint

        if self._protocol == agent_pb2.PROTOCOL_PBFT:
            timeout_values = [
                int(getattr(action, "election_timeout_milliseconds"))
            ]
        else:
            timeout_values = [
                int(getattr(action, "propose_timeout_milliseconds")),
                int(getattr(action, "prevote_timeout_milliseconds")),
                int(getattr(action, "precommit_timeout_milliseconds")),
            ]

        normalized = [
            self._normalize_value(timeout_ms, index)
            for index, timeout_ms in enumerate(timeout_values)
        ]
        return torch.tensor(normalized, dtype=torch.float32)

    def anchor_actions(self) -> torch.Tensor:
        timeout_values = get_timeout_values_ms(self._protocol)
        timeout_tuples = (
            [(value,) for value in timeout_values]
            if self._action_dim == 1
            else list(product(timeout_values, repeat=self._action_dim))
        )
        normalized = [
            [
                self._normalize_value(int(timeout_ms), index)
                for index, timeout_ms in enumerate(timeout_tuple)
            ]
            for timeout_tuple in timeout_tuples
        ]
        return torch.tensor(normalized, dtype=torch.float32)


class ContinuousThompsonSamplingModel(TensorModel):
    """Bootstrapped-tree contextual bandit over a continuous action domain."""

    def __init__(
        self,
        action_dim: int,
        n_features: int,
        anchor_actions: torch.Tensor,
        initial_design_size: int,
        memory_size: int,
        n_heads: int = 100,
        candidate_count: int = 1024,
        exploration_probability: float = 0.5,
        seed: Optional[int] = None,
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

        anchors = torch.as_tensor(anchor_actions, dtype=torch.float32)
        if anchors.dim() != 2 or anchors.shape[1] != action_dim:
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
        self._anchor_actions = anchors.cpu().numpy().astype(float)
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

    def _candidate_actions(self) -> np.ndarray:
        sampled = self._candidate_engine.draw(self._candidate_count).cpu().numpy()
        candidate_parts = [sampled, self._anchor_actions]
        if self.memory["actions"]:
            candidate_parts.append(np.asarray(self.memory["actions"], dtype=float))
        candidates = np.vstack(candidate_parts)
        return np.unique(np.clip(candidates, 0.0, 1.0), axis=0)

    def _predict_one(self, features: np.ndarray) -> np.ndarray:
        if self._design_count < self._initial_design_size:
            self._design_count += 1
            return self._design_engine.draw(1).cpu().numpy()[0].astype(float)

        candidates = self._candidate_actions()
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

    def predict_action(self, features: torch.Tensor) -> torch.Tensor:
        device = features.device
        features_np = features.detach().cpu().numpy().astype(float)
        single = features_np.ndim == 1
        if single:
            features_np = features_np.reshape(1, -1)
        if features_np.shape[1] != self._obs_dim:
            raise ValueError(
                f"Expected {self._obs_dim} features, got {features_np.shape[1]}"
            )

        actions = np.vstack([self._predict_one(row) for row in features_np])
        result = torch.tensor(actions, device=device, dtype=torch.float32)
        return result[0] if single else result

    def _action_rows(self, actions: torch.Tensor, batch_size: int) -> np.ndarray:
        actions_np = torch.as_tensor(actions).detach().cpu().numpy().astype(float)
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
        features: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor | float,
        backward: bool = True,
    ) -> float:
        features_np = features.detach().cpu().numpy().astype(float)
        if features_np.ndim == 1:
            features_np = features_np.reshape(1, -1)
        if features_np.shape[1] != self._obs_dim:
            raise ValueError(
                f"Expected {self._obs_dim} features, got {features_np.shape[1]}"
            )

        actions_np = self._action_rows(actions, features_np.shape[0])
        rewards_np = torch.as_tensor(rewards, dtype=torch.float32).reshape(-1).numpy()
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
        self._rng = np.random.default_rng(self._seed)
        self._design_engine = torch.quasirandom.SobolEngine(
            dimension=self._action_dim,
            scramble=True,
            seed=self._seed,
        )
        self._candidate_engine = torch.quasirandom.SobolEngine(
            dimension=self._action_dim,
            scramble=True,
            seed=self._seed + 1,
        )
        self._design_count = 0
        self.models = [self._new_tree(index) for index in range(self._n_heads)]
        self.is_fitted = False
        self.memory: Dict[str, deque[Any]] = {
            "features": deque(maxlen=self._memory_size),
            "actions": deque(maxlen=self._memory_size),
            "rewards": deque(maxlen=self._memory_size),
        }
