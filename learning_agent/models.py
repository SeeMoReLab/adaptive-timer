from abc import ABC, abstractmethod
import csv
from enum import IntEnum
import time
from pathlib import Path
from typing import Dict, Generic, List, Optional, Tuple, TypeVar, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import  DecisionTreeRegressor
from collections import deque
import numpy as np

import agent_pb2
from reward_utils import reward_from_throughput

S = TypeVar("S")  # generic state type
A = TypeVar("A")  # generic action type


class PbftTimeoutOption(IntEnum):
    TIMEOUT_300_MS = 300
    TIMEOUT_800_MS = 800
    TIMEOUT_1800_MS = 1800


PBFT_TIMEOUT_OPTIONS: Tuple[PbftTimeoutOption, ...] = (
    PbftTimeoutOption.TIMEOUT_300_MS,
    PbftTimeoutOption.TIMEOUT_800_MS,
    PbftTimeoutOption.TIMEOUT_1800_MS,
)
PBFT_TIMEOUT_VALUES_MS: Tuple[int, ...] = tuple(
    int(option) for option in PBFT_TIMEOUT_OPTIONS
)
PBFT_TIMEOUT_MS_TO_INDEX: Dict[int, int] = {
    value_ms: idx for idx, value_ms in enumerate(PBFT_TIMEOUT_VALUES_MS)
}
PBFT_TIMEOUT_INDEX_TO_MS: Dict[int, int] = {
    idx: value_ms for idx, value_ms in enumerate(PBFT_TIMEOUT_VALUES_MS)
}


class TimeoutOption(IntEnum):
    TIMEOUT_150_MS = 150
    TIMEOUT_400_MS = 400
    TIMEOUT_900_MS = 900

TIMEOUT_OPTIONS: Tuple[TimeoutOption, ...] = (
    TimeoutOption.TIMEOUT_150_MS,
    TimeoutOption.TIMEOUT_400_MS,
    TimeoutOption.TIMEOUT_900_MS,
)
TIMEOUT_VALUES_MS: Tuple[int, ...] = tuple(int(option) for option in TIMEOUT_OPTIONS)
TIMEOUT_MS_TO_INDEX: Dict[int, int] = {
    value_ms: idx for idx, value_ms in enumerate(TIMEOUT_VALUES_MS)
}
TIMEOUT_INDEX_TO_MS: Dict[int, int] = {
    idx : value_ms for idx, value_ms in enumerate(TIMEOUT_VALUES_MS)
}


class TendermintProposeTimeoutOption(IntEnum):
    TIMEOUT_150_MS = 150
    TIMEOUT_300_MS = 300
    TIMEOUT_600_MS = 600


TENDERMINT_PROPOSE_TIMEOUT_OPTIONS: Tuple[TendermintProposeTimeoutOption, ...] = (
    TendermintProposeTimeoutOption.TIMEOUT_150_MS,
    TendermintProposeTimeoutOption.TIMEOUT_300_MS,
    TendermintProposeTimeoutOption.TIMEOUT_600_MS,
)
TENDERMINT_PROPOSE_TIMEOUT_VALUES_MS: Tuple[int, ...] = tuple(
    int(option) for option in TENDERMINT_PROPOSE_TIMEOUT_OPTIONS
)
TENDERMINT_PROPOSE_TIMEOUT_MS_TO_INDEX: Dict[int, int] = {
    value_ms: idx
    for idx, value_ms in enumerate(TENDERMINT_PROPOSE_TIMEOUT_VALUES_MS)
}


class PhaseMapping(IntEnum):
    PHASE_0 = PbftTimeoutOption.TIMEOUT_300_MS
    PHASE_1 = PbftTimeoutOption.TIMEOUT_300_MS
    PHASE_2 = PbftTimeoutOption.TIMEOUT_300_MS
    PHASE_3 = PbftTimeoutOption.TIMEOUT_800_MS
    PHASE_4 = PbftTimeoutOption.TIMEOUT_800_MS
    PHASE_5 = PbftTimeoutOption.TIMEOUT_1800_MS

PHASE_OPTIONS: Tuple[PhaseMapping, ...] = (
    PhaseMapping.PHASE_0,
    PhaseMapping.PHASE_1,
    PhaseMapping.PHASE_2,
    PhaseMapping.PHASE_3,
    PhaseMapping.PHASE_4,
    PhaseMapping.PHASE_5,
)

PHASE_VALUES_MS: Tuple[int, ...] = tuple(int(option) for option in PHASE_OPTIONS)
PHASE_INDEX_TO_TIMEOUT_INDEX: Dict[int, int] = {
    idx: PBFT_TIMEOUT_MS_TO_INDEX[value_ms]
    for idx, value_ms in enumerate(PHASE_VALUES_MS)
}

PBFT_REPORT_FEATURE_FIELD_NAMES: Tuple[str, ...] = (
    "total_transactions",
    "total_consensus_instances",
    "avg_consensus_latency_ms",
    "p50_consensus_latency_ms",
    "p95_consensus_latency_ms",
    "throughput_tps",
    "avg_batch_size",
    "p95_batch_size",
    "leader_change_count",
    "regency_change_count",
    "timeout_ms",
    "avg_inter_commit_gap_ms",
    "p50_inter_commit_gap_ms",
    "p95_inter_commit_gap_ms",
    "view_change_count",
    "no_progress_view_change_count",
)
SBFT_REPORT_FEATURE_FIELD_NAMES: Tuple[str, ...] = (
    "total_transactions",
    "total_consensus_instances",
    "avg_consensus_latency_ms",
    "p95_consensus_latency_ms",
    "p99_consensus_latency_ms",
    "throughput_tps",
    "timeout_violation_rate",
    "avg_batch_size",
    "p95_batch_size",
    "leader_change_count",
    "regency_change_count",
)
TENDERMINT_REPORT_FEATURE_FIELD_NAMES: Tuple[str, ...] = (
    "total_transactions",
    "total_consensus_instances",
    "avg_consensus_latency_ms",
    "p50_consensus_latency_ms",
    "p90_consensus_latency_ms",
    "throughput_tps",
    "timeout_violation_rate",
    "avg_batch_size",
    "p50_batch_size",
    "p90_batch_size",
    "leader_change_count",
    "regency_change_count",
    "propose_latency_ms",
    "prevote_latency_ms",
    "precommit_latency_ms",
)


def _protocol_name(protocol: int) -> str:
    try:
        return agent_pb2.Protocol.Name(protocol)
    except ValueError:
        return f"UNKNOWN({protocol})"


def _validate_protocol(protocol: int) -> int:
    if protocol not in (
        agent_pb2.PROTOCOL_PBFT,
        agent_pb2.PROTOCOL_SBFT,
        agent_pb2.PROTOCOL_TENDERMINT,
    ):
        raise ValueError(f"Unsupported protocol: {_protocol_name(protocol)}")
    return int(protocol)


def get_report_feature_field_names(protocol: int) -> Tuple[str, ...]:
    protocol = _validate_protocol(protocol)
    if protocol == agent_pb2.PROTOCOL_PBFT:
        return PBFT_REPORT_FEATURE_FIELD_NAMES
    if protocol == agent_pb2.PROTOCOL_SBFT:
        return SBFT_REPORT_FEATURE_FIELD_NAMES
    return TENDERMINT_REPORT_FEATURE_FIELD_NAMES


def get_report_feature_dim(protocol: int) -> int:
    return len(get_report_feature_field_names(protocol))


def get_timeout_values_ms(protocol: int) -> Tuple[int, ...]:
    protocol = _validate_protocol(protocol)
    if protocol == agent_pb2.PROTOCOL_PBFT:
        return PBFT_TIMEOUT_VALUES_MS
    if protocol == agent_pb2.PROTOCOL_TENDERMINT:
        return TENDERMINT_PROPOSE_TIMEOUT_VALUES_MS
    return TIMEOUT_VALUES_MS


def timeout_ms_to_index(timeout_ms: int) -> int:
    try:
        return TIMEOUT_MS_TO_INDEX[timeout_ms]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported timeout {timeout_ms}ms. Valid options: {TIMEOUT_VALUES_MS}"
        ) from exc


def timeout_index_to_ms(index: int) -> int:
    if index < 0:
        return TIMEOUT_VALUES_MS[0]
    if index >= len(TIMEOUT_VALUES_MS):
        return TIMEOUT_VALUES_MS[-1]
    return TIMEOUT_VALUES_MS[index]


def pbft_timeout_ms_to_index(timeout_ms: int) -> int:
    try:
        return PBFT_TIMEOUT_MS_TO_INDEX[timeout_ms]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported PBFT timeout {timeout_ms}ms. "
            f"Valid options: {PBFT_TIMEOUT_VALUES_MS}"
        ) from exc


def pbft_timeout_index_to_ms(index: int) -> int:
    if index < 0:
        return PBFT_TIMEOUT_VALUES_MS[0]
    if index >= len(PBFT_TIMEOUT_VALUES_MS):
        return PBFT_TIMEOUT_VALUES_MS[-1]
    return PBFT_TIMEOUT_VALUES_MS[index]


def timeout_index_to_ms_for_protocol(protocol: int, index: int) -> int:
    protocol = _validate_protocol(protocol)
    if protocol == agent_pb2.PROTOCOL_PBFT:
        return pbft_timeout_index_to_ms(index)
    if protocol == agent_pb2.PROTOCOL_TENDERMINT:
        return tendermint_propose_timeout_index_to_ms(index)
    return timeout_index_to_ms(index)


def _sbft_triple_to_index(
    election_timeout_ms: int, slow_path_timeout_ms: int, batch_timeout_ms: int
) -> int:
    election_idx = timeout_ms_to_index(election_timeout_ms)
    slow_idx = timeout_ms_to_index(slow_path_timeout_ms)
    batch_idx = timeout_ms_to_index(batch_timeout_ms)
    base = len(TIMEOUT_VALUES_MS)
    return (election_idx * base * base) + (slow_idx * base) + batch_idx


def _sbft_index_to_triple(index: int) -> Tuple[int, int, int]:
    base = len(TIMEOUT_VALUES_MS)
    total = base * base * base
    bounded = max(0, min(index, total - 1))
    election_idx = bounded // (base * base)
    rem = bounded % (base * base)
    slow_idx = rem // base
    batch_idx = rem % base
    return (
        timeout_index_to_ms(election_idx),
        timeout_index_to_ms(slow_idx),
        timeout_index_to_ms(batch_idx),
    )


def tendermint_timeouts_from_propose(
    propose_timeout_ms: int,
) -> Tuple[int, int, int]:
    propose_timeout_ms = int(propose_timeout_ms)
    if propose_timeout_ms <= 0:
        raise ValueError(
            f"Tendermint propose timeout must be positive, got {propose_timeout_ms}"
        )
    phase_timeout_ms = max(1, (propose_timeout_ms + 1) // 3)
    return propose_timeout_ms, phase_timeout_ms, phase_timeout_ms


def tendermint_propose_timeout_ms_to_index(propose_timeout_ms: int) -> int:
    try:
        return TENDERMINT_PROPOSE_TIMEOUT_MS_TO_INDEX[int(propose_timeout_ms)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported Tendermint propose timeout {propose_timeout_ms}ms. "
            f"Valid options: {TENDERMINT_PROPOSE_TIMEOUT_VALUES_MS}"
        ) from exc


def tendermint_propose_timeout_index_to_ms(index: int) -> int:
    if index < 0:
        return TENDERMINT_PROPOSE_TIMEOUT_VALUES_MS[0]
    if index >= len(TENDERMINT_PROPOSE_TIMEOUT_VALUES_MS):
        return TENDERMINT_PROPOSE_TIMEOUT_VALUES_MS[-1]
    return TENDERMINT_PROPOSE_TIMEOUT_VALUES_MS[index]


def _tendermint_timeouts_to_index(
    propose_timeout_ms: int, prevote_timeout_ms: int, precommit_timeout_ms: int
) -> int:
    expected = tendermint_timeouts_from_propose(propose_timeout_ms)
    actual = (
        int(propose_timeout_ms),
        int(prevote_timeout_ms),
        int(precommit_timeout_ms),
    )
    if actual != expected:
        raise ValueError(
            "Tendermint prevote and precommit timeouts must be one-third "
            f"of propose timeout: expected {expected}, got {actual}"
        )
    return tendermint_propose_timeout_ms_to_index(propose_timeout_ms)


def _tendermint_index_to_triple(index: int) -> Tuple[int, int, int]:
    return tendermint_timeouts_from_propose(
        tendermint_propose_timeout_index_to_ms(index)
    )


def get_action_space_size(protocol: int) -> int:
    protocol = _validate_protocol(protocol)
    if protocol == agent_pb2.PROTOCOL_PBFT:
        return len(PBFT_TIMEOUT_VALUES_MS)
    if protocol == agent_pb2.PROTOCOL_SBFT:
        return len(TIMEOUT_VALUES_MS) * len(TIMEOUT_VALUES_MS) * len(TIMEOUT_VALUES_MS)
    if protocol == agent_pb2.PROTOCOL_TENDERMINT:
        return len(TENDERMINT_PROPOSE_TIMEOUT_VALUES_MS)
    raise ValueError(f"Unsupported protocol: {_protocol_name(protocol)}")


def report_to_feature_values(protocol: int, state: object) -> List[float]:
    protocol = _validate_protocol(protocol)
    fields = get_report_feature_field_names(protocol)
    return [float(getattr(state, field_name)) for field_name in fields]


def timeout_to_action_index(protocol: int, timeout_used: object) -> int:
    protocol = _validate_protocol(protocol)
    if protocol == agent_pb2.PROTOCOL_PBFT:
        timeout_msg = timeout_used
        return pbft_timeout_ms_to_index(int(timeout_msg.election_timeout_milliseconds))
    if protocol == agent_pb2.PROTOCOL_SBFT:
        timeout_msg = timeout_used
        return _sbft_triple_to_index(
            int(timeout_msg.election_timeout_milliseconds),
            int(timeout_msg.slow_path_timeout_milliseconds),
            int(timeout_msg.batch_timeout_milliseconds),
        )
    if protocol == agent_pb2.PROTOCOL_TENDERMINT:
        timeout_msg = timeout_used
        return _tendermint_timeouts_to_index(
            int(timeout_msg.propose_timeout_milliseconds),
            int(timeout_msg.prevote_timeout_milliseconds),
            int(timeout_msg.precommit_timeout_milliseconds),
        )
    raise ValueError(f"Unsupported protocol: {_protocol_name(protocol)}")


def action_index_to_timeout(protocol: int, index: int) -> agent_pb2.Timeout:
    protocol = _validate_protocol(protocol)
    if protocol == agent_pb2.PROTOCOL_PBFT:
        timeout_ms = pbft_timeout_index_to_ms(index)
        return agent_pb2.Timeout(
            pbft=agent_pb2.PbftTimeout(election_timeout_milliseconds=timeout_ms)
        )
    if protocol == agent_pb2.PROTOCOL_SBFT:
        election_timeout_ms, slow_path_timeout_ms, batch_timeout_ms = _sbft_index_to_triple(
            index
        )
        return agent_pb2.Timeout(
            sbft=agent_pb2.SbftTimeout(
                election_timeout_milliseconds=election_timeout_ms,
                slow_path_timeout_milliseconds=slow_path_timeout_ms,
                batch_timeout_milliseconds=batch_timeout_ms,
            )
        )
    if protocol == agent_pb2.PROTOCOL_TENDERMINT:
        propose_timeout_ms, prevote_timeout_ms, precommit_timeout_ms = (
            _tendermint_index_to_triple(index)
        )
        return agent_pb2.Timeout(
            tendermint=agent_pb2.TendermintTimeout(
                propose_timeout_milliseconds=propose_timeout_ms,
                prevote_timeout_milliseconds=prevote_timeout_ms,
                precommit_timeout_milliseconds=precommit_timeout_ms,
            )
        )
    raise ValueError(f"Unsupported protocol: {_protocol_name(protocol)}")


def describe_action(protocol: int, index: int) -> str:
    protocol = _validate_protocol(protocol)
    if protocol == agent_pb2.PROTOCOL_PBFT:
        return f"election_timeout_ms={pbft_timeout_index_to_ms(index)}"
    if protocol == agent_pb2.PROTOCOL_SBFT:
        election_timeout_ms, slow_path_timeout_ms, batch_timeout_ms = _sbft_index_to_triple(
            index
        )
        return (
            f"election_timeout_ms={election_timeout_ms}, "
            f"slow_path_timeout_ms={slow_path_timeout_ms}, "
            f"batch_timeout_ms={batch_timeout_ms}"
        )
    propose_timeout_ms, prevote_timeout_ms, precommit_timeout_ms = _tendermint_index_to_triple(
        index
    )
    return (
        f"propose_timeout_ms={propose_timeout_ms}, "
        f"prevote_timeout_ms={prevote_timeout_ms}, "
        f"precommit_timeout_ms={precommit_timeout_ms}"
    )

class FeatureExtractor(Generic[S], ABC):
    """Converts an input state into a tensor for model consumption."""

    @abstractmethod
    def featurize(self, state: S) -> np.ndarray:
        pass


class ActionCodec(Generic[A], ABC):
    """Converts between model actions and domain-specific actions."""

    @abstractmethod
    def decode(self, action: np.ndarray) -> A:
        pass

    @abstractmethod
    def encode(self, action: object) -> np.ndarray:
        pass


class TensorModel(ABC):
    """Model that consumes and produces tensors."""

    @abstractmethod
    def predict_action(self, features: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def update_model(
        self, features: np.ndarray, action: np.ndarray, reward: float
    ) -> float:
        pass

    @abstractmethod
    def reset_model(self) -> None:
        pass


class RewardTransformer(ABC):
    """Converts throughput result into float reward."""

    @abstractmethod
    def transform(self, throughput: float) -> float:
        pass

class ProtocolReportFeatureExtractor(FeatureExtractor[object]):
    def __init__(self, protocol: int):
        self._protocol = _validate_protocol(protocol)

    def featurize(self, state: object) -> np.ndarray:
        features = report_to_feature_values(self._protocol, state)
        return np.asarray(features, dtype=float)


class ProtocolTimeoutActionCodec(ActionCodec[agent_pb2.Timeout]):
    """Converts discrete arm indices to and from protocol timeout actions."""

    def __init__(self, protocol: int):
        self._protocol = _validate_protocol(protocol)

    def decode(self, action: np.ndarray) -> agent_pb2.Timeout:
        action = np.asarray(action, dtype=float)
        if action.size == 1:
            option_index = int(round(float(action.reshape(-1)[0])))
        else:
            option_index = int(np.argmax(action))
        return action_index_to_timeout(self._protocol, option_index)

    def encode(self, action: object) -> np.ndarray:
        option_index = timeout_to_action_index(self._protocol, action)
        return np.asarray(float(option_index), dtype=float)

class QuadraticRewardScaler(RewardTransformer):
    """Quadratic shaping below 1000, linear above. Prior work"""

    @staticmethod
    def transform(throughput: float) -> float:
        return reward_from_throughput(throughput)

class LinearModel(TensorModel):
    """Linear policy over protocol-specific discrete timeout options.

    Reward-weighted multinomial logistic regression: each update pushes the
    policy toward the arm that was actually taken, weighted by the reward that
    arm earned. The gradient of that objective is analytic, so nothing here
    needs a differentiation framework:

        p       = softmax(X W^T)
        dL/dz   = s * (p - onehot(taken)) / batch      s = clamp(reward, 0)/1000
        dL/dW   = dL/dz^T X

    followed by an AdamW step. Decoupled weight decay is the ``W *= 1 - lr*wd``
    that precedes the moment update - AdamW's ordering, rather than the
    L2-in-the-gradient of classic Adam.

    Replaces an earlier autograd implementation that used ``torch.optim.AdamW``
    over a ``requires_grad`` weight matrix and ``F.cross_entropy``. The two were
    checked against each other over 400 updates before the torch version was
    removed: weights agreed to 3.7e-07 and argmax agreed on 100% of 2000
    contexts. Two differences were deliberate and are kept here:

    * Arithmetic is float64 throughout. The torch version accumulated in
      float32.
    * :meth:`reset_model` clears the optimizer moments as well as the weights.
      The torch version zeroed the weights but left AdamW's state behind, so a
      reset model's first steps were still scaled by the previous run's moments.
    """

    def __init__(
        self,
        action_dim: int,
        n_features: int,
        lr: float = 0.01,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")
        if n_features <= 0:
            raise ValueError(f"n_features must be > 0, got {n_features}")
        self._action_dim = int(action_dim)
        self._n_features = int(n_features)
        self._lr = float(lr)
        self._weight_decay = float(weight_decay)
        self._betas = (float(betas[0]), float(betas[1]))
        self._eps = float(eps)
        self.reset_model()

    def predict_action(self, features: np.ndarray) -> np.ndarray:
        """Logits for every arm.

        A 1-D context returns a 1-D ``(action_dim,)`` vector, a 2-D batch
        returns ``(batch, action_dim)``.
        """
        features_np = np.asarray(features, dtype=float)
        if features_np.shape[-1] != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} features, got {features_np.shape[-1]}"
            )
        logits = features_np @ self.weights.T
        return np.asarray(logits, dtype=float)

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
        if features_np.shape[1] != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} features, got {features_np.shape[1]}"
            )

        actions_np = np.asarray(actions, dtype=float).reshape(-1)
        rewards_np = np.asarray(rewards, dtype=float).reshape(-1)

        batch_size = features_np.shape[0]
        if actions_np.size == 1 and batch_size > 1:
            actions_np = np.repeat(actions_np, batch_size)
        if rewards_np.size == 1 and batch_size > 1:
            rewards_np = np.repeat(rewards_np, batch_size)
        if actions_np.size != batch_size:
            raise ValueError(
                "actions batch size mismatch: "
                f"expected {batch_size}, got {actions_np.size}"
            )
        if rewards_np.size != batch_size:
            raise ValueError(
                "rewards batch size mismatch: "
                f"expected {batch_size}, got {rewards_np.size}"
            )

        # Out-of-range arms are clamped rather than rejected, preserving the
        # behavior of the autograd version this replaced.
        targets = np.clip(
            np.round(actions_np).astype(int), 0, self._action_dim - 1
        )
        scales = np.clip(rewards_np, 0.0, None) / 1000.0

        logits = features_np @ self.weights.T
        # log-softmax via the shift trick, the same way cross_entropy does it.
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        rows = np.arange(batch_size)
        loss = float((-log_probs[rows, targets] * scales).mean())

        if backward and loss > 0.0:
            d_logits = np.exp(log_probs)
            d_logits[rows, targets] -= 1.0
            d_logits *= scales[:, None]
            d_logits /= batch_size
            self._adamw_step(d_logits.T @ features_np)

        return loss

    def _adamw_step(self, gradient: np.ndarray) -> None:
        beta1, beta2 = self._betas
        self._step += 1
        # Decoupled decay, applied to the weights before the moment update.
        self.weights *= 1.0 - self._lr * self._weight_decay
        self._moment1 = beta1 * self._moment1 + (1.0 - beta1) * gradient
        self._moment2 = beta2 * self._moment2 + (1.0 - beta2) * gradient * gradient
        bias_correction1 = 1.0 - beta1**self._step
        bias_correction2 = 1.0 - beta2**self._step
        denominator = np.sqrt(self._moment2) / np.sqrt(bias_correction2) + self._eps
        self.weights -= (self._lr / bias_correction1) * (self._moment1 / denominator)

    def reset_model(self) -> None:
        self.weights = np.zeros(
            (self._action_dim, self._n_features), dtype=np.float64
        )
        self._moment1 = np.zeros_like(self.weights)
        self._moment2 = np.zeros_like(self.weights)
        self._step = 0

    def save(self) -> np.ndarray:
        return self.weights.copy()

    def load(self, weights: np.ndarray) -> None:
        incoming = np.asarray(weights, dtype=np.float64)
        if incoming.shape != self.weights.shape:
            raise ValueError(
                f"Expected weights of shape {self.weights.shape}, got {incoming.shape}"
            )
        self.weights = incoming.copy()


class RandomTimeoutModel(TensorModel):
    """Selects a random timeout action index each prediction."""

    def __init__(self, action_dim: int, seed: Optional[int] = None):
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")
        self._action_dim = int(action_dim)
        if seed is None:
            seed = int(time.time_ns() & 0x7FFFFFFF)
        self._initial_seed = int(seed)
        self._rng = np.random.default_rng(self._initial_seed)

    def predict_action(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        if features.ndim == 1:
            option_index = int(self._rng.integers(0, self._action_dim))
            return np.asarray(float(option_index), dtype=float)

        batch_size = int(features.shape[0])
        return self._rng.integers(
            0, self._action_dim, size=batch_size
        ).astype(float)

    def update_model(
        self, features: np.ndarray, action: np.ndarray, reward: float
    ) -> float:
        del features, action, reward
        return 0.0

    def reset_model(self) -> None:
        self._rng = np.random.default_rng(self._initial_seed)

    def save(self) -> Dict[str, Any]:
        pass

    def load(self, checkpoint: Dict[str, Any]) -> None:
        pass

class ScheduledTimeoutModel(TensorModel):
    """Returns protocol-specific timeout option indices from a time-based schedule."""

    def __init__(
        self,
        protocol: int,
        schedule: List[Tuple[float, int]],
        start_monotonic: Optional[float] = None,
    ):
        self._protocol = _validate_protocol(protocol)
        if not schedule:
            raise ValueError("Schedule must contain at least one entry.")
        self._schedule = sorted(
            [(float(time_s), int(option_index)) for time_s, option_index in schedule],
            key=lambda entry: entry[0],
        )
        self._effective_schedule: List[Tuple[float, int]] = []
        self._schedule_time_shift_s = 0.0
        self._rebuild_effective_schedule()
        self._start_time = (
            time.monotonic() if start_monotonic is None else float(start_monotonic)
        )
        self._active_row = -1

    @classmethod
    def from_csv(
        cls,
        protocol: int,
        path: Path,
        start_monotonic: Optional[float] = None,
    ) -> "ScheduledTimeoutModel":
        protocol = _validate_protocol(protocol)
        schedule: List[Tuple[float, int]] = []
        with path.open("r", newline="") as handle:
            reader = csv.reader(handle)
            for lineno, row in enumerate(reader, start=1):
                if not row:
                    continue
                first = row[0].strip()
                if not first or first.startswith("#"):
                    continue
                try:
                    time_s = float(first)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid schedule time at line {lineno}: {row}"
                    ) from exc

                if protocol == agent_pb2.PROTOCOL_PBFT:
                    if len(row) < 2:
                        raise ValueError(
                            f"Invalid PBFT schedule at line {lineno}: expected 2 columns time_s,election_timeout_ms."
                        )
                    timeout_ms = int(row[1].strip())
                    option_index = pbft_timeout_ms_to_index(timeout_ms)
                elif protocol == agent_pb2.PROTOCOL_SBFT:
                    if len(row) < 4:
                        raise ValueError(
                            "Invalid SBFT schedule at line "
                            f"{lineno}: expected 4 columns "
                            "time_s,election_timeout_ms,slow_path_timeout_ms,batch_timeout_ms."
                        )
                    election_timeout_ms = int(row[1].strip())
                    slow_path_timeout_ms = int(row[2].strip())
                    batch_timeout_ms = int(row[3].strip())
                    option_index = _sbft_triple_to_index(
                        election_timeout_ms,
                        slow_path_timeout_ms,
                        batch_timeout_ms,
                    )
                else:
                    if len(row) < 2:
                        raise ValueError(
                            "Invalid Tendermint schedule at line "
                            f"{lineno}: expected 2 columns "
                            "time_s,propose_timeout_ms."
                        )
                    propose_timeout_ms = int(row[1].strip())
                    option_index = tendermint_propose_timeout_ms_to_index(
                        propose_timeout_ms
                    )
                schedule.append((time_s, option_index))
        if not schedule:
            raise ValueError(f"No schedule entries found in {path}.")
        return cls(protocol, schedule, start_monotonic=start_monotonic)

    def _rebuild_effective_schedule(self) -> None:
        if not self._schedule:
            raise ValueError("Schedule must contain at least one entry.")
        first_switch_time_s = float(self._schedule[0][0])
        self._schedule_time_shift_s = -min(0.0, first_switch_time_s)
        self._effective_schedule = [
            (switch_time_s + self._schedule_time_shift_s, option_index)
            for switch_time_s, option_index in self._schedule
        ]

    def set_start_monotonic(self, start_monotonic: float) -> None:
        self._start_time = float(start_monotonic)
        self._active_row = -1

    def _current_option_index(self) -> int:
        elapsed_s = time.monotonic() - self._start_time
        active_row = 0
        option_index = self._effective_schedule[0][1]
        for row_idx, (switch_time_s, schedule_option_index) in enumerate(
            self._effective_schedule
        ):
            if elapsed_s >= switch_time_s:
                active_row = row_idx
                option_index = schedule_option_index
            else:
                break
        if active_row != self._active_row:
            switch_time_s, active_option_index = self._effective_schedule[active_row]
            source_switch_time_s, _ = self._schedule[active_row]
            print(
                "[scheduled-model] active schedule: "
                f"time_s>={switch_time_s:g}, "
                f"source_time_s={source_switch_time_s:g}, "
                f"{describe_action(self._protocol, active_option_index)}"
            )
            self._active_row = active_row
        return option_index

    def is_warmup_active(self) -> bool:
        if self._schedule_time_shift_s <= 0.0:
            return False
        elapsed_s = time.monotonic() - self._start_time
        return 0.0 <= elapsed_s < self._schedule_time_shift_s

    def predict_action(self, features: np.ndarray) -> np.ndarray:
        del features
        option_index = self._current_option_index()
        return np.asarray(float(option_index), dtype=float)

    def update_model(
        self, features: np.ndarray, action: np.ndarray, reward: float
    ) -> float:
        del features, action, reward
        return 0.0

    def reset_model(self) -> None:
        self._start_time = time.monotonic()
        self._active_row = -1

    def save(self) -> Dict[str, Any]:
        pass

    def load(self, checkpoint: Dict[str, Any]) -> None:
        pass

class ThompsonSamplingModelV1(TensorModel): 
    """Previous single-tree bootstrapped CMAB implementation."""

    def __init__(
        self, 
        action_dim: int, 
        n_features: int, 
    ):
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")
        
        self._act_dim = int(action_dim)
        self._obs_dim = int(n_features)
        self._max_bucket_size = 256

        self.reset_model()

    def predict_action(self, features: np.ndarray) -> np.ndarray:
        features_np = np.asarray(features, dtype=float)
        if features_np.ndim == 1:
            features_np = features_np.reshape(1, -1)
            
        batch_size = features_np.shape[0]
        predictions = []
    
        for act in range(self._act_dim):
            if not self.is_fitted:
                pred = np.full(batch_size, 1e9)
            else:
                action_col = np.full((batch_size, 1), act)
                state_action_input = np.hstack((features_np, action_col))
                pred = self.model.predict(state_action_input)
            
            predictions.append(pred)
        
        predictions_stacked = np.column_stack(predictions)
        
        # Exploration
        for i in range(batch_size):
            row_preds = predictions_stacked[i]
            tied_indices = np.flatnonzero(np.isclose(row_preds, np.max(row_preds)))
            
            if len(tied_indices) > 1:
                best_index = np.random.choice(tied_indices, replace=True)
                predictions_stacked[i, best_index] += 1e-5

        q_values = np.asarray(predictions_stacked, dtype=float)
        return q_values

    def update_model(
        self, features: np.ndarray, actions: np.ndarray, rewards: np.ndarray | float, backward: bool = True
    ) -> float:
        # An evaluation batch must not enter the replay memory: evaluate_model
        # calls this with backward=False twice per episode, so ingesting here
        # would record every episode three times over.
        if not backward:
            return 0.0
        features_np = np.asarray(features, dtype=float)
        if features_np.ndim == 1:
            features_np = features_np.reshape(1, -1)
            
        if isinstance(actions, np.ndarray):
            actions_np = np.asarray(actions, dtype=int).ravel()
        else:
            actions_np = np.array([int(actions)])
            
        if isinstance(rewards, np.ndarray):
            rewards_np = np.asarray(rewards, dtype=float).ravel()
        else:
            rewards_np = np.full(len(actions_np), float(rewards))

        for i in range(len(actions_np)):
            self.memory["features"].append(features_np[i])
            self.memory["actions"].append(actions_np[i])
            self.memory["rewards"].append(rewards_np[i])
        
        if backward and len(self.memory["rewards"]) > 0:
            X_features = np.array(self.memory["features"])
            X_actions = np.array(self.memory["actions"]).reshape(-1, 1)
            
            # X [features, action]
            X = np.hstack((X_features, X_actions))
            y = np.array(self.memory["rewards"])
            
            # Bootstrapping data
            idx = np.random.choice(len(X), size=len(X), replace=True)
            
            self.model.fit(X[idx], y[idx])
            self.is_fitted = True

        return 0.0

    def reset_model(self) -> None:
        self.model = DecisionTreeRegressor(
            criterion='squared_error',
            max_depth=6,              
            min_samples_leaf=2,    
            min_impurity_decrease=0.01,    
            random_state=42
        )
        self.is_fitted = False
        self._init_buckets()

    def _init_buckets(self):
        self.memory = {
            "features": deque(maxlen=self._max_bucket_size), 
            "actions": deque(maxlen=self._max_bucket_size),
            "rewards": deque(maxlen=self._max_bucket_size)
        }


class ThompsonSamplingModelV2(TensorModel):
    def __init__(
        self,
        action_dim: int,
        n_features: int,
        n_heads: int = 100,
        seed: Optional[int] = None,
    ):
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be > 0, got {n_heads}")

        self._act_dim = int(action_dim)
        self._obs_dim = int(n_features)
        self._n_heads = int(n_heads)
        self._max_bucket_size = 32
        self._seed = seed

        self.reset_model()

    def predict_action(self, features: np.ndarray) -> np.ndarray:
        features_np = np.asarray(features, dtype=float)
        if features_np.ndim == 1:
            features_np = features_np.reshape(1, -1)
            
        batch_size = features_np.shape[0]
        predictions_stacked = np.zeros((batch_size, self._act_dim), dtype=float)
        sampled_heads = self._rng.integers(0, self._n_heads, size=batch_size)

        for i, head_index in enumerate(sampled_heads):
            if not self.is_fitted[head_index]:
                predictions_stacked[i, :] = 0.0
                continue

            state_rows = np.repeat(features_np[i : i + 1], self._act_dim, axis=0)
            action_col = np.arange(self._act_dim, dtype=float).reshape(-1, 1)
            state_action_input = np.hstack((state_rows, action_col))
            predictions_stacked[i, :] = self.models[head_index].predict(
                state_action_input
            )

        # Force basic action coverage before trusting the reward model. This is
        # explicit exploration; posterior-like exploration comes from the sampled head.
        action_counts = np.bincount(
            np.array(self.memory["actions"], dtype=int),
            minlength=self._act_dim,
        )
        untried_actions = np.flatnonzero(action_counts == 0)
        if len(untried_actions) > 0:
            predictions_stacked[:, untried_actions] = 1e9

        for i in range(batch_size):
            row_preds = predictions_stacked[i]
            tied_indices = np.flatnonzero(np.isclose(row_preds, np.max(row_preds)))

            if len(tied_indices) > 1:
                best_index = self._rng.choice(tied_indices)
                predictions_stacked[i, best_index] += 1e-5

        q_values = np.asarray(predictions_stacked, dtype=float)
        return q_values

    def update_model(
        self, features: np.ndarray, actions: np.ndarray, rewards: np.ndarray | float, backward: bool = True
    ) -> float:
        # An evaluation batch must not enter the replay memory: evaluate_model
        # calls this with backward=False twice per episode, so ingesting here
        # would record every episode three times over.
        if not backward:
            return 0.0
        features_np = np.asarray(features, dtype=float)
        if features_np.ndim == 1:
            features_np = features_np.reshape(1, -1)
            
        if isinstance(actions, np.ndarray):
            actions_np = np.asarray(actions, dtype=int).ravel()
        else:
            actions_np = np.array([int(actions)])
            
        if isinstance(rewards, np.ndarray):
            rewards_np = np.asarray(rewards, dtype=float).ravel()
        else:
            rewards_np = np.full(len(actions_np), float(rewards))

        if len(actions_np) == 1 and features_np.shape[0] > 1:
            actions_np = np.repeat(actions_np, features_np.shape[0])
        if len(rewards_np) == 1 and features_np.shape[0] > 1:
            rewards_np = np.repeat(rewards_np, features_np.shape[0])
        if len(actions_np) != features_np.shape[0]:
            raise ValueError(
                "actions batch size mismatch: "
                f"expected {features_np.shape[0]}, got {len(actions_np)}"
            )
        if len(rewards_np) != features_np.shape[0]:
            raise ValueError(
                "rewards batch size mismatch: "
                f"expected {features_np.shape[0]}, got {len(rewards_np)}"
            )

        for i in range(len(actions_np)):
            self.memory["features"].append(features_np[i])
            self.memory["actions"].append(actions_np[i])
            self.memory["rewards"].append(rewards_np[i])
        
        if backward and len(self.memory["rewards"]) > 0:
            X_features = np.array(self.memory["features"])
            X_actions = np.array(self.memory["actions"]).reshape(-1, 1)
            X = np.hstack((X_features, X_actions))
            y = np.array(self.memory["rewards"])

            # GIRO: https://arxiv.org/pdf/1811.05154
            # Optimistic pseudo-reward is the max reward in the sliding window
            # X = np.tile(X, (3, 1))
            # y = np.concatenate(
            #     [np.zeros_like(y), y, np.full_like(y, np.max(y))]
            # )

            for head_index, model in enumerate(self.models):
                idx = self._rng.choice(len(X), size=len(X), replace=True)
                model.fit(X[idx], y[idx])
                self.is_fitted[head_index] = True

        return 0.0

    def reset_model(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self.models = [
            DecisionTreeRegressor(
                criterion='squared_error',
                max_depth=6,
                min_samples_leaf=6,
                min_impurity_decrease=0.01,
                random_state=head_index,
            )
            for head_index in range(self._n_heads)
        ]
        self.is_fitted = [False] * self._n_heads
        self._init_buckets()

    def _init_buckets(self):
        self.memory = {
            "features": deque(maxlen=self._max_bucket_size), 
            "actions": deque(maxlen=self._max_bucket_size),
            "rewards": deque(maxlen=self._max_bucket_size)
        }

class ThompsonSamplingModel(TensorModel):
    """Per-arm bootstrapped-tree CMAB (contextual GIRO structure).

    Each arm keeps its own sliding window of (features, reward) pairs and its
    own ensemble of trees, each head fit on a bootstrap resample of that arm's
    window. Prediction draws one head per arm independently and the caller
    pulls the argmax across arms.
    """

    def __init__(
        self,
        action_dim: int,
        n_features: int,
        n_heads: int = 100,
        seed: Optional[int] = None,
    ):
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be > 0, got {n_heads}")

        self._act_dim = int(action_dim)
        self._obs_dim = int(n_features)
        self._n_heads = int(n_heads)
        self._max_bucket_size = 32
        self._seed = seed

        self.reset_model()

    def predict_action(self, features: np.ndarray, e = 0.1) -> np.ndarray:
        """e-TS: with probability e take the Thompson draw (one bootstrap head
        per arm); with probability 1-e exploit the arm with the highest
        empirical mean over its reward sliding window."""
        features_np = np.asarray(features, dtype=float)
        if features_np.ndim == 1:
            features_np = features_np.reshape(1, -1)

        batch_size = features_np.shape[0]
        predictions_stacked = np.zeros((batch_size, self._act_dim), dtype=float)
        # One independent head (bootstrap draw) per arm, per row.
        sampled_heads = self._rng.integers(
            0, self._n_heads, size=(batch_size, self._act_dim)
        )
        explore = self._rng.random(batch_size) < e

        for i in range(batch_size):
            row = features_np[i : i + 1]
            for arm in range(self._act_dim):
                if not self.is_fitted[arm]:
                    # Arm without a fitted window yet: force a pull.
                    predictions_stacked[i, arm] = 1e9
                    continue
                if explore[i]:
                    head = self.models[arm][sampled_heads[i, arm]]
                    predictions_stacked[i, arm] = head.predict(row)[0]
                else:
                    predictions_stacked[i, arm] = float(
                        np.mean(self.memory[arm]["rewards"])
                    )

        for i in range(batch_size):
            row_preds = predictions_stacked[i]
            tied_indices = np.flatnonzero(np.isclose(row_preds, np.max(row_preds)))

            if len(tied_indices) > 1:
                best_index = self._rng.choice(tied_indices)
                predictions_stacked[i, best_index] += 1e-5

        q_values = np.asarray(predictions_stacked, dtype=float)
        return q_values

    def update_model(
        self, features: np.ndarray, actions: np.ndarray, rewards: np.ndarray | float, backward: bool = True
    ) -> float:
        # An evaluation batch must not enter the replay memory: evaluate_model
        # calls this with backward=False twice per episode, so ingesting here
        # would record every episode three times over.
        if not backward:
            return 0.0
        features_np = np.asarray(features, dtype=float)
        if features_np.ndim == 1:
            features_np = features_np.reshape(1, -1)
            
        if isinstance(actions, np.ndarray):
            actions_np = np.asarray(actions, dtype=int).ravel()
        else:
            actions_np = np.array([int(actions)])
            
        if isinstance(rewards, np.ndarray):
            rewards_np = np.asarray(rewards, dtype=float).ravel()
        else:
            rewards_np = np.full(len(actions_np), float(rewards))

        if len(actions_np) == 1 and features_np.shape[0] > 1:
            actions_np = np.repeat(actions_np, features_np.shape[0])
        if len(rewards_np) == 1 and features_np.shape[0] > 1:
            rewards_np = np.repeat(rewards_np, features_np.shape[0])
        if len(actions_np) != features_np.shape[0]:
            raise ValueError(
                "actions batch size mismatch: "
                f"expected {features_np.shape[0]}, got {len(actions_np)}"
            )
        if len(rewards_np) != features_np.shape[0]:
            raise ValueError(
                "rewards batch size mismatch: "
                f"expected {features_np.shape[0]}, got {len(rewards_np)}"
            )

        for i in range(len(actions_np)):
            arm = int(actions_np[i])
            if not 0 <= arm < self._act_dim:
                raise ValueError(f"action out of range: {arm}")
            self.memory[arm]["features"].append(features_np[i])
            self.memory[arm]["rewards"].append(rewards_np[i])

        if backward:
            window_max = max(
                (max(m["rewards"]) for m in self.memory if m["rewards"]),
                default=0.0,
            )
                                    
            for arm in sorted({int(a) for a in actions_np}):
                X = np.array(self.memory[arm]["features"])
                y = np.array(self.memory[arm]["rewards"])

                            
                for model in self.models[arm]:
                    idx = self._rng.choice(len(X), size=len(X), replace=True)
                    model.fit(X[idx], y[idx])
                self.is_fitted[arm] = True

        return 0.0

    def reset_model(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self.models = [
            [
                DecisionTreeRegressor(
                    criterion='squared_error',
                    max_depth=6,
                    min_samples_leaf=6,
                    min_impurity_decrease=0.01,
                    random_state=arm_index * self._n_heads + head_index,
                )
                for head_index in range(self._n_heads)
            ]
            for arm_index in range(self._act_dim)
        ]
        self.is_fitted = [False] * self._act_dim
        self._init_buckets()

    def _init_buckets(self):
        self.memory = [
            {
                "features": deque(maxlen=self._max_bucket_size),
                "rewards": deque(maxlen=self._max_bucket_size),
            }
            for _ in range(self._act_dim)
        ]

class MultiHeadThompsonSamplingModel(ThompsonSamplingModel):
    """Backward-compatible name for the bootstrapped ensemble CMAB model."""

    def __init__(
        self, 
        action_dim: int, 
        n_features: int,
        n_heads: int = 10,
    ):
        super().__init__(
            action_dim=action_dim,
            n_features=n_features,
            n_heads=n_heads,
        )

class ThompsonSamplingModelV3(TensorModel):
    """Per-arm bootstrapped-tree CMAB with a posterior-driven decision rule.

    Same structure as :class:`ThompsonSamplingModel` (one sliding window and
    one bootstrap ensemble per arm, caller takes the argmax of the returned
    scores) with three changes, each restoring a property that Thompson
    sampling depends on:

    * Every decision is a posterior draw. One bootstrap head is sampled per
      arm and the arm's score is that head's prediction *at the current
      context*. Exploration comes from how much the heads disagree, so it is
      directed at the arms that are actually uncertain and it anneals per arm
      as that arm's window fills in. There is no epsilon and no separate
      greedy branch scoring arms by a context-free window mean.
    * The bootstrap cannot collapse to zero spread. Each arm's history is
      augmented with GIRO-style pseudo-rewards (Kveton et al.,
      https://arxiv.org/pdf/1811.05154) pinned to the best and worst reward
      observed so far, at ``pseudo_reward_coefficient`` weight per real
      sample, so the heads still disagree when an arm's observed rewards are
      near-constant. Plain bootstrap TS can otherwise stop exploring
      entirely.
    * Uncertainty grows back for arms that are not being pulled. A draw is
      shrunk toward an uninformative draw over the observed reward range by
      ``staleness_discount ** steps_since_last_update``, so a stale estimate
      reverts toward the prior instead of freezing an arm out permanently.
      Sliding windows alone only forget data for arms that keep being pulled,
      which is the wrong behavior when the best arm changes mid-run.

    Aging is counted in ingested samples rather than wall-clock time so that
    online and simulated runs with the same seed behave identically.

    ``staleness_discount`` should be read relative to the number of arms: an
    arm is revisited on the order of every ``action_dim`` steps at best, so
    larger action spaces want a value closer to 1.
    """

    def __init__(
        self,
        action_dim: int,
        n_features: int,
        n_heads: int = 30,
        window_size: int = 32,
        pseudo_reward_coefficient: float = 1.0 / 3.0,
        staleness_discount: float = 0.99,
        seed: Optional[int] = None,
    ):
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")
        if n_features <= 0:
            raise ValueError(f"n_features must be > 0, got {n_features}")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be > 0, got {n_heads}")
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}")
        if pseudo_reward_coefficient < 0.0:
            raise ValueError(
                "pseudo_reward_coefficient must be >= 0, got "
                f"{pseudo_reward_coefficient}"
            )
        if not 0.0 < staleness_discount <= 1.0:
            raise ValueError(
                f"staleness_discount must be in (0, 1], got {staleness_discount}"
            )

        self._act_dim = int(action_dim)
        self._obs_dim = int(n_features)
        self._n_heads = int(n_heads)
        self._max_bucket_size = int(window_size)
        self._pseudo_coefficient = float(pseudo_reward_coefficient)
        self._staleness_discount = float(staleness_discount)
        self._seed = int(time.time_ns() & 0x7FFFFFFF) if seed is None else int(seed)

        self.reset_model()

    def predict_action(
        self, features: np.ndarray, greedy: bool = False
    ) -> np.ndarray:
        """Score every arm with one independent posterior draw.

        Returns a ``(batch, action_dim)`` score tensor; the caller pulls the
        argmax. Arms with no observation yet all score ``+inf``, so the argmax
        pulls them in index order, giving a deterministic round-robin initial
        design.

        ``greedy=True`` scores arms by the posterior mean (the ensemble
        average at the current context) with no exploration. It is for offline
        evaluation and reporting, not for acting.
        """
        features_np = np.asarray(features, dtype=float)
        if features_np.ndim == 1:
            features_np = features_np.reshape(1, -1)
        if features_np.shape[1] != self._obs_dim:
            raise ValueError(
                f"Expected {self._obs_dim} features, got {features_np.shape[1]}"
            )

        batch_size = features_np.shape[0]
        scores = np.zeros((batch_size, self._act_dim), dtype=float)
        # One independent head per arm: the per-arm ensembles are fit on
        # disjoint data, so their posteriors are independent.
        sampled_heads = self._rng.integers(
            0, self._n_heads, size=(batch_size, self._act_dim)
        )
        reward_low, reward_high = self._observed_reward_range()

        for i in range(batch_size):
            row = features_np[i : i + 1]
            for arm in range(self._act_dim):
                if not self.is_fitted[arm]:
                    scores[i, arm] = np.inf
                    continue
                if greedy:
                    scores[i, arm] = float(
                        np.mean([head.predict(row)[0] for head in self.models[arm]])
                    )
                    continue
                head = self.models[arm][sampled_heads[i, arm]]
                scores[i, arm] = self._shrink_stale_draw(
                    arm, float(head.predict(row)[0]), reward_low, reward_high
                )

        self._break_ties(scores)
        return np.asarray(scores, dtype=float)

    def update_model(
        self,
        features: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | float,
        backward: bool = True,
    ) -> float:
        # backward=False means "do not learn from this batch". Unlike
        # ThompsonSamplingModel this also skips ingestion, so evaluation
        # batches cannot leak into the windows the acting policy reads.
        if not backward:
            return 0.0

        features_np = np.asarray(features, dtype=float)
        if features_np.ndim == 1:
            features_np = features_np.reshape(1, -1)
        if features_np.shape[1] != self._obs_dim:
            raise ValueError(
                f"Expected {self._obs_dim} features, got {features_np.shape[1]}"
            )

        if isinstance(actions, np.ndarray):
            actions_np = np.asarray(actions, dtype=int).ravel()
        else:
            actions_np = np.array([int(actions)])

        if isinstance(rewards, np.ndarray):
            rewards_np = np.asarray(rewards, dtype=float).ravel()
        else:
            rewards_np = np.full(len(actions_np), float(rewards))

        batch_size = features_np.shape[0]
        if len(actions_np) == 1 and batch_size > 1:
            actions_np = np.repeat(actions_np, batch_size)
        if len(rewards_np) == 1 and batch_size > 1:
            rewards_np = np.repeat(rewards_np, batch_size)
        if len(actions_np) != batch_size:
            raise ValueError(
                f"actions batch size mismatch: expected {batch_size}, "
                f"got {len(actions_np)}"
            )
        if len(rewards_np) != batch_size:
            raise ValueError(
                f"rewards batch size mismatch: expected {batch_size}, "
                f"got {len(rewards_np)}"
            )

        for i in range(batch_size):
            arm = int(actions_np[i])
            if not 0 <= arm < self._act_dim:
                raise ValueError(f"action out of range: {arm}")
            self._step += 1
            self.memory[arm]["features"].append(features_np[i])
            self.memory[arm]["rewards"].append(float(rewards_np[i]))
            self._last_update_step[arm] = self._step

        for arm in sorted({int(action) for action in actions_np}):
            self._fit_arm(arm)

        return 0.0

    def reset_model(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self.models = [
            [
                DecisionTreeRegressor(
                    criterion="squared_error",
                    max_depth=6,
                    min_samples_leaf=6,
                    # No min_impurity_decrease: an absolute impurity floor ties
                    # tree depth to the reward scale, which is arbitrary here
                    # (shaped throughput spans ~0 to 10^3 and beyond).
                    random_state=self._seed
                    + arm_index * self._n_heads
                    + head_index,
                )
                for head_index in range(self._n_heads)
            ]
            for arm_index in range(self._act_dim)
        ]
        self.is_fitted = [False] * self._act_dim
        self._last_update_step = [0] * self._act_dim
        self._step = 0
        self._init_buckets()

    def _init_buckets(self) -> None:
        self.memory = [
            {
                "features": deque(maxlen=self._max_bucket_size),
                "rewards": deque(maxlen=self._max_bucket_size),
            }
            for _ in range(self._act_dim)
        ]

    def _break_ties(self, scores: np.ndarray) -> None:
        """Randomize the winner among exactly equal finite scores, in place.

        Exact equality is the right test: equal scores here come from
        identical computations (constant trees, or the same prior draw), not
        from floating point drift, and the relative tolerance of np.isclose
        would fold genuinely different scores together at large reward
        magnitudes. Infinite scores are left alone so that untried arms keep
        their round-robin order.

        The winner is raised by one representable step rather than a fixed
        epsilon: any increment smaller than the float spacing at that magnitude
        would be rounded away and the tie would silently resolve to the lowest
        arm index. This used to be a float32 step, because scores were narrowed
        to float32 on the way into a torch tensor; scores are float64 now, so
        the step scales with the actual output precision.
        """
        for i in range(scores.shape[0]):
            row = scores[i]
            best = float(np.max(row))
            if not np.isfinite(best):
                continue
            tied = np.flatnonzero(row == best)
            if len(tied) <= 1:
                continue
            row[int(self._rng.choice(tied))] = np.nextafter(best, np.inf)

    def _observed_reward_range(self) -> Tuple[float, float]:
        """Best and worst reward seen across all arms' windows.

        This is the support of the uninformative prior used for pseudo-rewards
        and for staleness shrinkage. When every observed reward is identical
        the range is degenerate and both mechanisms become no-ops, which is
        harmless: all arms then score identically and the tie break picks
        uniformly at random.
        """
        rewards = [
            reward for arm_memory in self.memory for reward in arm_memory["rewards"]
        ]
        if not rewards:
            return 0.0, 0.0
        return float(min(rewards)), float(max(rewards))

    def _shrink_stale_draw(
        self, arm: int, draw: float, reward_low: float, reward_high: float
    ) -> float:
        """Mix a posterior draw with an uninformative one as the arm's last
        observation ages, so an unmeasured arm reverts toward the prior."""
        if self._staleness_discount >= 1.0:
            return draw
        age = self._step - self._last_update_step[arm]
        if age <= 0:
            return draw
        weight = self._staleness_discount**age
        prior_draw = float(self._rng.uniform(reward_low, reward_high))
        return weight * draw + (1.0 - weight) * prior_draw

    def _fit_arm(self, arm: int) -> None:
        """Refit one arm's ensemble on bootstrap resamples of its window,
        augmented with pseudo-rewards at the observed reward extremes."""
        context = np.array(self.memory[arm]["features"], dtype=float)
        observed = np.array(self.memory[arm]["rewards"], dtype=float)
        reward_low, reward_high = self._observed_reward_range()
        sample_count = len(observed)

        # Pseudo-observations reuse the real contexts, so they inflate the
        # spread of the fit without distorting the context distribution.
        inputs = np.vstack((context, context, context))
        targets = np.concatenate(
            (
                observed,
                np.full(sample_count, reward_high),
                np.full(sample_count, reward_low),
            )
        )
        weights = np.concatenate(
            (
                np.ones(sample_count),
                np.full(sample_count, self._pseudo_coefficient),
                np.full(sample_count, self._pseudo_coefficient),
            )
        )
        probabilities = weights / weights.sum()
        draw_size = max(1, int(round(float(weights.sum()))))

        for head in self.models[arm]:
            indices = self._rng.choice(
                len(inputs), size=draw_size, replace=True, p=probabilities
            )
            head.fit(inputs[indices], targets[indices])
        self.is_fitted[arm] = True


def _break_ties_in_place(scores: np.ndarray, rng: np.random.Generator) -> None:
    """Randomize the winner among exactly equal finite scores, in place.

    Same rule as :meth:`ThompsonSamplingModelV3._break_ties`: exact equality
    rather than ``np.isclose`` (equal scores here come from identical
    computations, not float drift), infinite scores left alone so untried arms
    keep their round-robin order, and the winner raised by one representable
    step so the bump cannot be rounded away.
    """
    for i in range(scores.shape[0]):
        row = scores[i]
        best = float(np.max(row))
        if not np.isfinite(best):
            continue
        tied = np.flatnonzero(row == best)
        if len(tied) <= 1:
            continue
        row[int(rng.choice(tied))] = np.nextafter(best, np.inf)


class _BootstrappedForestModel(TensorModel):
    """Shared machinery for the BFTBrain random-forest CMAB baselines.

    Ported from the reference implementation in BFTBrain
    (``code/learning/learning_agent.py``), where the three variants differ only
    in how they partition experience across forests: one forest per
    ``(previous arm, arm)`` transition (:class:`QuadraticRF`), one per arm
    (:class:`MultiRF`), or one shared forest over ``[context ‖ one-hot(arm)]``
    (:class:`SingleRF`).

    All three explore the same way: a forest is refit on a *bootstrap resample*
    of its replay window before it is used to score arms, so the scores are a
    randomized draw rather than a point estimate. This is the same posterior
    approximation :class:`ThompsonSamplingModel` uses, differing in that the
    resample happens at fit time on a single forest instead of being drawn from
    a persistent ensemble of heads. Arms with no data at all score ``+inf`` and
    are pulled in index order, giving a deterministic round-robin over the
    initial design.

    Three deliberate departures from the reference implementation:

    * Rewards are ingested as given. BFTBrain applies its ``reward_engineering``
      shaping inside the model; here that belongs to
      :class:`QuadraticRewardScaler` upstream, so applying it again would square
      the shaping.
    * Contexts and rewards are stored as one atomic pair per ``update_model``
      call. BFTBrain appends them from two separate callbacks that run an
      episode apart, which leaves its ``X`` list ahead of its ``y`` list and
      misaligns the two by that gap when the replay window is applied.
    * ``backward=False`` skips ingestion entirely rather than only skipping the
      refit, matching :class:`ThompsonSamplingModelV3`, so evaluation batches
      cannot leak into the windows the acting policy reads.

    The reference agent also warm-starts its buffers from checkpoint CSVs at
    construction; these models start empty and are warmed by ``update_model``.

    Subclasses must call :meth:`reset_model` at the end of their ``__init__``.
    """

    def __init__(
        self,
        action_dim: int,
        n_features: int,
        max_depth: Optional[int] = 5,
        n_estimators: int = 100,
        replay_buffer_size: Optional[int] = None,
        log_reward: bool = False,
        seed: Optional[int] = None,
    ):
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")
        if n_features <= 0:
            raise ValueError(f"n_features must be > 0, got {n_features}")
        if max_depth is not None and max_depth <= 0:
            raise ValueError(f"max_depth must be > 0 or None, got {max_depth}")
        if n_estimators <= 0:
            raise ValueError(f"n_estimators must be > 0, got {n_estimators}")
        if replay_buffer_size is not None and replay_buffer_size <= 0:
            raise ValueError(
                f"replay_buffer_size must be > 0 or None, got {replay_buffer_size}"
            )

        self._act_dim = int(action_dim)
        self._obs_dim = int(n_features)
        self._max_depth = None if max_depth is None else int(max_depth)
        self._n_estimators = int(n_estimators)
        self._replay_buffer_size = (
            None if replay_buffer_size is None else int(replay_buffer_size)
        )
        self._log_reward = bool(log_reward)
        self._seed = int(time.time_ns() & 0x7FFFFFFF) if seed is None else int(seed)

    def _new_forest(self, index: int) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            random_state=self._seed + index,
        )

    def _new_bucket(self) -> Dict[str, List[Any]]:
        return {"features": [], "rewards": []}

    def _as_batch(self, features: np.ndarray) -> np.ndarray:
        features_np = np.asarray(features, dtype=float)
        if features_np.ndim == 1:
            features_np = features_np.reshape(1, -1)
        if features_np.shape[1] != self._obs_dim:
            raise ValueError(
                f"Expected {self._obs_dim} features, got {features_np.shape[1]}"
            )
        return features_np

    def _normalize_batch(
        self,
        features: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        features_np = self._as_batch(features)

        if isinstance(actions, np.ndarray):
            actions_np = np.asarray(actions, dtype=int).ravel()
        else:
            actions_np = np.array([int(actions)])

        if isinstance(rewards, np.ndarray):
            rewards_np = np.asarray(rewards, dtype=float).ravel()
        else:
            rewards_np = np.full(len(actions_np), float(rewards))

        batch_size = features_np.shape[0]
        if len(actions_np) == 1 and batch_size > 1:
            actions_np = np.repeat(actions_np, batch_size)
        if len(rewards_np) == 1 and batch_size > 1:
            rewards_np = np.repeat(rewards_np, batch_size)
        if len(actions_np) != batch_size:
            raise ValueError(
                f"actions batch size mismatch: expected {batch_size}, "
                f"got {len(actions_np)}"
            )
        if len(rewards_np) != batch_size:
            raise ValueError(
                f"rewards batch size mismatch: expected {batch_size}, "
                f"got {len(rewards_np)}"
            )
        for arm in actions_np:
            if not 0 <= int(arm) < self._act_dim:
                raise ValueError(f"action out of range: {int(arm)}")

        return features_np, actions_np, rewards_np

    def _transform_reward(self, reward: float) -> float:
        """Optionally fit on signed log reward instead of raw reward.

        A squared-error tree fit on raw reward spends its splits where the
        absolute variance is, which here is the high-throughput phases. On this
        dataset an arm's reward spans roughly 590 to 11,000 across phases, so a
        590-to-809 distinction - which is the entire phase 7 decision - is
        worth about a hundredth as much loss as a distinction between two
        phases whose answer the model already has right. The decision only ever
        needs the argmax within a phase, which is a ratio question, so a
        monotone log makes every phase contribute in proportion while leaving
        the within-phase ordering untouched. Signed, so non-positive rewards
        stay ordered.
        """
        if not self._log_reward:
            return reward
        return float(np.sign(reward) * np.log1p(abs(reward)))

    def _replay_length(self, sample_count: int) -> int:
        if self._replay_buffer_size is None:
            return sample_count
        return min(self._replay_buffer_size, sample_count)

    def _fit_forest(self, forest: RandomForestRegressor, bucket: Dict[str, List[Any]]) -> None:
        """Refit one forest on a bootstrap resample of the tail of its window.

        The resample is the exploration mechanism: two consecutive fits on the
        same data give different trees, so the scores this forest produces are
        a draw rather than a fixed estimate.
        """
        count = self._replay_length(len(bucket["rewards"]))
        inputs = np.array(bucket["features"], dtype=float)[-count:]
        targets = np.array(bucket["rewards"], dtype=float)[-count:]
        indices = self._rng.choice(count, size=count, replace=True)
        forest.fit(inputs[indices], targets[indices])

    def _resolve_prev_action(self, prev_action: Optional[int]) -> int:
        if prev_action is None:
            return self._prev_action
        prev_action = int(prev_action)
        if not 0 <= prev_action < self._act_dim:
            raise ValueError(f"prev_action out of range: {prev_action}")
        return prev_action

    def reset_model(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self._prev_action = self._initial_prev_action


class QuadraticRF(_BootstrappedForestModel):
    """One forest per ``(previous arm, arm)`` transition.

    The reference model conditions on which protocol was running when the
    decision was made, on the theory that the reward of switching to an arm
    depends on what it is switching *from*. That gives ``action_dim ** 2``
    forests and buckets, so it needs on the order of ``action_dim`` times more
    data than :class:`MultiRF` to fill in.

    The previous arm is not part of the :class:`TensorModel` signature, so it is
    tracked internally: ``update_model`` records the order arms are ingested in,
    and the arm ingested immediately before a sample is that sample's
    predecessor. Callers that know the arm actually running - and whose reward
    attribution lags their decisions, as BFTBrain's does by two episodes - should
    pass ``prev_action`` to :meth:`predict_action` explicitly rather than rely on
    the tracker, which only advances on ingestion.

    Untried transitions score ``+inf`` to force exploration, but a transition
    stays on hold at ``-inf`` from the moment it is chosen until its reward
    arrives. Without that, every decision in the window before the first reward
    lands would re-pick the same untried bucket.

    That hold expires after ``hold_steps`` ingested samples rather than only on
    the matching reward, which the reference does not need to do: there, every
    decision the model makes is executed, so its reward always comes back and
    always clears the hold. A caller that replays a fixed dataset - the offline
    simulator - ingests whatever the trace recorded, not what the model chose,
    so unconditional holds would accumulate until every untried bucket was held
    at ``-inf`` and forced exploration stopped entirely. The default of 2
    matches the reference's two-episode reward lag.
    """

    def __init__(
        self,
        action_dim: int,
        n_features: int,
        max_depth: Optional[int] = 5,
        n_estimators: int = 100,
        replay_buffer_size: Optional[int] = None,
        log_reward: bool = False,
        initial_prev_action: int = 0,
        hold_steps: int = 2,
        seed: Optional[int] = None,
    ):
        super().__init__(
            action_dim=action_dim,
            n_features=n_features,
            max_depth=max_depth,
            n_estimators=n_estimators,
            replay_buffer_size=replay_buffer_size,
            log_reward=log_reward,
            seed=seed,
        )
        if not 0 <= int(initial_prev_action) < self._act_dim:
            raise ValueError(
                f"initial_prev_action out of range: {initial_prev_action}"
            )
        if hold_steps < 0:
            raise ValueError(f"hold_steps must be >= 0, got {hold_steps}")
        self._initial_prev_action = int(initial_prev_action)
        self._hold_steps = int(hold_steps)
        self.reset_model()

    def predict_action(
        self,
        features: np.ndarray,
        prev_action: Optional[int] = None,
        greedy: bool = False,
    ) -> np.ndarray:
        """Score every arm as a successor to ``prev_action``.

        Returns a ``(batch, action_dim)`` score tensor; the caller pulls the
        argmax. Mutating: every bucket whose window changed since the last
        decision is refit first (a stale fit is what would otherwise be read),
        and the bucket the argmax lands in is put on hold. ``greedy=True``
        suppresses both the exploration sentinels and the on-hold bookkeeping,
        scoring arms with the current fits only - for offline evaluation, not
        for acting.
        """
        features_np = self._as_batch(features)
        previous = self._resolve_prev_action(prev_action)

        # Refit every bucket whose window changed since the last decision. The
        # reference implementation refits a single bucket because it ingests one
        # sample per episode; a batched update can dirty several at once, and a
        # bucket left unfit here would be read below as if it had a model.
        for bucket_prev, bucket_arm in sorted(self._dirty_buckets):
            self._fit_forest(
                self.models[bucket_prev][bucket_arm],
                self.memory[bucket_prev][bucket_arm],
            )
            self.is_fitted[bucket_prev][bucket_arm] = True
        self._dirty_buckets.clear()

        batch_size = features_np.shape[0]
        scores = np.zeros((batch_size, self._act_dim), dtype=float)
        for i in range(batch_size):
            row = features_np[i : i + 1]
            for arm in range(self._act_dim):
                if not self.is_fitted[previous][arm]:
                    if greedy or self._is_on_hold(previous, arm):
                        scores[i, arm] = -np.inf
                    else:
                        scores[i, arm] = np.inf
                    continue
                scores[i, arm] = float(self.models[previous][arm].predict(row)[0])

        _break_ties_in_place(scores, self._rng)

        if not greedy:
            for i in range(batch_size):
                self._on_hold_buckets[
                    (previous, int(np.argmax(scores[i])))
                ] = self._step

        return np.asarray(scores, dtype=float)

    def _is_on_hold(self, prev_action: int, arm: int) -> bool:
        held_at = self._on_hold_buckets.get((prev_action, arm))
        if held_at is None:
            return False
        if self._step - held_at < self._hold_steps:
            return True
        del self._on_hold_buckets[(prev_action, arm)]
        return False

    def update_model(
        self,
        features: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | float,
        backward: bool = True,
        prev_actions: Optional[np.ndarray] = None,
    ) -> float:
        if not backward:
            return 0.0

        features_np, actions_np, rewards_np = self._normalize_batch(
            features, actions, rewards
        )

        if prev_actions is None:
            explicit_prev = None
        else:
            explicit_prev = (
                np.asarray(prev_actions, dtype=int).ravel()
                if isinstance(prev_actions, np.ndarray)
                else np.array(prev_actions, dtype=int).flatten()
            )
            if len(explicit_prev) != len(actions_np):
                raise ValueError(
                    f"prev_actions batch size mismatch: expected {len(actions_np)}, "
                    f"got {len(explicit_prev)}"
                )

        for i in range(len(actions_np)):
            arm = int(actions_np[i])
            previous = self._resolve_prev_action(
                None if explicit_prev is None else explicit_prev[i]
            )
            self.memory[previous][arm]["features"].append(features_np[i])
            self.memory[previous][arm]["rewards"].append(self._transform_reward(float(rewards_np[i])))
            self._dirty_buckets.add((previous, arm))
            self._on_hold_buckets.pop((previous, arm), None)
            self._step += 1
            # This sample's arm is the next sample's predecessor.
            self._prev_action = arm

        return 0.0

    def reset_model(self) -> None:
        super().reset_model()
        self.models = [
            [
                self._new_forest(prev_index * self._act_dim + arm_index)
                for arm_index in range(self._act_dim)
            ]
            for prev_index in range(self._act_dim)
        ]
        self.is_fitted = [
            [False] * self._act_dim for _ in range(self._act_dim)
        ]
        self._dirty_buckets: set = set()
        # bucket -> the ingestion step at which it was put on hold
        self._on_hold_buckets: Dict[Tuple[int, int], int] = {}
        self._step = 0
        self._init_buckets()

    def _init_buckets(self) -> None:
        self.memory = [
            [self._new_bucket() for _ in range(self._act_dim)]
            for _ in range(self._act_dim)
        ]


class MultiRF(_BootstrappedForestModel):
    """One forest per arm, each fit on that arm's own experience.

    The middle point of the three: arms do not share a model, so a forest never
    has to learn the arm boundary from a one-hot column the way
    :class:`SingleRF` does, but there is no transition conditioning and so no
    ``action_dim``-fold data cost the way there is in :class:`QuadraticRF`.

    ``one_hot_prev_action`` reproduces the reference model's ``--multi-onehot``
    mode, appending a one-hot of the previous arm to the context so a per-arm
    forest can still condition on the transition. One deviation: BFTBrain drops
    its last two feature columns when it appends the one-hot, because those
    columns encode protocol capabilities that the one-hot makes redundant. The
    timeout report features here carry no such columns, so nothing is dropped
    and the model input is ``n_features + action_dim`` wide.

    Untried arms score ``+inf``. There is no on-hold mechanism, matching the
    reference: with only ``action_dim`` buckets they fill quickly enough that
    re-picking an untried arm before its reward lands is not worth guarding.
    """

    def __init__(
        self,
        action_dim: int,
        n_features: int,
        max_depth: Optional[int] = 5,
        n_estimators: int = 100,
        replay_buffer_size: Optional[int] = None,
        log_reward: bool = False,
        one_hot_prev_action: bool = False,
        initial_prev_action: int = 0,
        seed: Optional[int] = None,
    ):
        super().__init__(
            action_dim=action_dim,
            n_features=n_features,
            max_depth=max_depth,
            n_estimators=n_estimators,
            replay_buffer_size=replay_buffer_size,
            log_reward=log_reward,
            seed=seed,
        )
        if not 0 <= int(initial_prev_action) < self._act_dim:
            raise ValueError(
                f"initial_prev_action out of range: {initial_prev_action}"
            )
        self._one_hot_prev_action = bool(one_hot_prev_action)
        self._initial_prev_action = int(initial_prev_action)
        self.reset_model()

    def _model_input(self, row: np.ndarray, prev_action: int) -> np.ndarray:
        if not self._one_hot_prev_action:
            return row
        one_hot = np.zeros(self._act_dim, dtype=float)
        one_hot[prev_action] = 1.0
        return np.concatenate((row, one_hot))

    def predict_action(
        self,
        features: np.ndarray,
        prev_action: Optional[int] = None,
        greedy: bool = False,
    ) -> np.ndarray:
        """Score every arm with its own forest.

        ``prev_action`` is only read when ``one_hot_prev_action`` is set; see
        :class:`QuadraticRF` for how the internal tracker resolves it.
        ``greedy=True`` scores with the current fits and never forces an untried
        arm - for offline evaluation, not for acting.
        """
        features_np = self._as_batch(features)
        previous = self._resolve_prev_action(prev_action)

        # See QuadraticRF.predict_action: a batched update can dirty more than
        # the one bucket the reference implementation assumes.
        for arm in sorted(self._dirty_buckets):
            self._fit_forest(self.models[arm], self.memory[arm])
            self.is_fitted[arm] = True
        self._dirty_buckets.clear()

        batch_size = features_np.shape[0]
        scores = np.zeros((batch_size, self._act_dim), dtype=float)
        for i in range(batch_size):
            row = self._model_input(features_np[i], previous).reshape(1, -1)
            for arm in range(self._act_dim):
                if not self.is_fitted[arm]:
                    scores[i, arm] = -np.inf if greedy else np.inf
                    continue
                scores[i, arm] = float(self.models[arm].predict(row)[0])

        _break_ties_in_place(scores, self._rng)
        return np.asarray(scores, dtype=float)

    def update_model(
        self,
        features: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | float,
        backward: bool = True,
        prev_actions: Optional[np.ndarray] = None,
    ) -> float:
        if not backward:
            return 0.0

        features_np, actions_np, rewards_np = self._normalize_batch(
            features, actions, rewards
        )

        if prev_actions is None:
            explicit_prev = None
        else:
            explicit_prev = (
                np.asarray(prev_actions, dtype=int).ravel()
                if isinstance(prev_actions, np.ndarray)
                else np.array(prev_actions, dtype=int).flatten()
            )
            if len(explicit_prev) != len(actions_np):
                raise ValueError(
                    f"prev_actions batch size mismatch: expected {len(actions_np)}, "
                    f"got {len(explicit_prev)}"
                )

        for i in range(len(actions_np)):
            arm = int(actions_np[i])
            previous = self._resolve_prev_action(
                None if explicit_prev is None else explicit_prev[i]
            )
            self.memory[arm]["features"].append(
                self._model_input(features_np[i], previous)
            )
            self.memory[arm]["rewards"].append(self._transform_reward(float(rewards_np[i])))
            self._dirty_buckets.add(arm)
            self._prev_action = arm

        return 0.0

    def reset_model(self) -> None:
        super().reset_model()
        self.models = [self._new_forest(arm) for arm in range(self._act_dim)]
        self.is_fitted = [False] * self._act_dim
        self._dirty_buckets: set = set()
        self._init_buckets()

    def _init_buckets(self) -> None:
        self.memory = [self._new_bucket() for _ in range(self._act_dim)]


class SingleRF(_BootstrappedForestModel):
    """One forest over ``[context ‖ one-hot(arm)]`` shared by every arm.

    Scoring enumerates the arms: the context is tiled ``action_dim`` times, each
    row completed with a different one-hot, and the whole block predicted in one
    call. Every sample trains the single model, so an arm benefits from what the
    others measured - which is the point, and also the risk, since the forest can
    split on the context and ignore the one-hot entirely, collapsing the arms
    together.

    The reference model refits on every decision rather than only when a window
    changes, since there is one window and it changes on every update. That is
    kept: the per-decision bootstrap resample is what makes the scores a draw.

    Before any data arrives the arms are scored uniformly at random, matching
    the reference's random cold-start pick.
    """

    def __init__(
        self,
        action_dim: int,
        n_features: int,
        max_depth: Optional[int] = 5,
        n_estimators: int = 100,
        replay_buffer_size: Optional[int] = None,
        log_reward: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__(
            action_dim=action_dim,
            n_features=n_features,
            max_depth=max_depth,
            n_estimators=n_estimators,
            replay_buffer_size=replay_buffer_size,
            log_reward=log_reward,
            seed=seed,
        )
        self._initial_prev_action = 0
        self.reset_model()

    def predict_action(
        self, features: np.ndarray, greedy: bool = False
    ) -> np.ndarray:
        """Score every arm from the shared forest.

        Mutating: the forest is refit on a fresh bootstrap resample first, so
        two calls on the same context give different scores. ``greedy=True``
        skips the refit and scores from the current fit - for offline
        evaluation, not for acting.
        """
        features_np = self._as_batch(features)
        has_data = len(self.memory["rewards"]) > 0

        if has_data and not greedy:
            self._fit_forest(self.model, self.memory)

        batch_size = features_np.shape[0]
        scores = np.zeros((batch_size, self._act_dim), dtype=float)
        for i in range(batch_size):
            if not self.is_fitted:
                # No data yet: uniform random scores, so the argmax is a
                # uniform random arm.
                scores[i] = self._rng.random(self._act_dim)
                continue
            block = np.hstack(
                (
                    np.tile(features_np[i], (self._act_dim, 1)),
                    np.eye(self._act_dim),
                )
            )
            scores[i] = self.model.predict(block)

        _break_ties_in_place(scores, self._rng)
        return np.asarray(scores, dtype=float)

    def update_model(
        self,
        features: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | float,
        backward: bool = True,
    ) -> float:
        if not backward:
            return 0.0

        features_np, actions_np, rewards_np = self._normalize_batch(
            features, actions, rewards
        )

        for i in range(len(actions_np)):
            one_hot = np.zeros(self._act_dim, dtype=float)
            one_hot[int(actions_np[i])] = 1.0
            self.memory["features"].append(
                np.concatenate((features_np[i], one_hot))
            )
            self.memory["rewards"].append(self._transform_reward(float(rewards_np[i])))
            self._prev_action = int(actions_np[i])

        # Keeping the fit current on ingestion, rather than only on the next
        # decision, is what makes a greedy read well defined at any point. The
        # acting path in predict_action still draws its own bootstrap refit on
        # top of this one.
        self._fit_forest(self.model, self.memory)
        self.is_fitted = True

        return 0.0

    def reset_model(self) -> None:
        super().reset_model()
        self.model = self._new_forest(0)
        self.is_fitted = False
        self._init_buckets()

    def _init_buckets(self) -> None:
        self.memory = self._new_bucket()


class NeighborhoodRobustBandit(TensorModel):
    """claude_v1 - contextual bandit that commits locally and breaks ties by
    what holds in the surrounding region.

    Built against two observed failure modes on the smartbft_wallclock phase
    sweep:

    * :class:`ThompsonSamplingModelV1` never settles. It refits one tree on a
      bootstrap of its buffer on every decision, and that jitter has no term
      that decays, so more experience does not make it more decisive.
    * :class:`ThompsonSamplingModel` is slow to leave a timeout when the
      workload changes - notably going from phase 2 back to phase 1, where it
      holds 800ms well past the point 300ms became correct. The cause is that
      80% of its decisions score arms by a context-free mean over a 32-sample
      window, so right after a change the window is still describing the phase
      it just left.

    Both are addressed by the same structure rather than by two mechanisms.

    **Value estimate.** Every observation is kept as a ``(standardized context,
    arm, reward)`` triple, forever. An arm's value at a context is estimated
    from the samples within ``radius`` of it. Because the estimate is keyed on
    position in context space rather than recency, leaving a regime does not
    degrade what is known about it, and returning does not require relearning.

    **Commitment.** The posterior width is ``sqrt(variance / (n_local +
    prior_weight))`` where ``n_local`` counts samples inside the radius, so it
    keeps shrinking as evidence accumulates and the model does settle. This is
    what :class:`ThompsonSamplingModelV1` lacks: its uncertainty is a fixed
    property of the bootstrap, not a function of how much it knows.

    **Adaptation.** Commitment is local, so it costs nothing when the workload
    moves. A context in a region with no samples inside the radius scores
    ``+inf`` and is explored immediately - there is no window to flush and no
    change detector to trigger. The model is decisive where it has looked and
    curious where it has not, at the same time.

    **Robust tie-break.** The posterior mean is a shrinkage blend

        mu = (n_local * mu_local + prior_weight * mu_wide) / (n_local + prior_weight)

    where ``mu_wide`` is the arm's mean over the ``wide_neighbors`` nearest
    samples - a much broader region. The wide estimate is an empirical prior:
    it decides nothing once local evidence is strong, and it decides everything
    when local evidence cannot.

    That second case is the phase-2 problem. There, 300ms and 800ms differ by
    about 5% against reward noise roughly seven times the gap, so no amount of
    local sampling separates them - the arms are genuinely tied on measured
    reward. The surrounding region is dominated by phase 1, where 300ms wins by
    88%, so the prior lands the tie on 300ms. Since 300ms is also correct for
    phase 1, the phase 2 -> 1 transition then costs nothing: there is no switch
    left to be slow about. The model never has to tell phases 1 and 2 apart,
    which matters because they are the least separable pair in the dataset -
    0.22 apart in standardized units, against within-phase spreads of 0.52 and
    0.72.

    **Mechanisms.**

    * ``prior_cold_start`` - an arm with no samples inside the radius, but
      samples elsewhere, is scored from the wide neighborhood at prior
      confidence rather than ``+inf``. Entering a regime with no opinion at all
      costs a full round of blind probing every time the workload moves.
    * ``robust`` - the local estimate is a median with a MAD spread. A
      neighborhood straddling two regimes carries a few samples from the wrong
      one whose rewards can be an order of magnitude off.
    * ``exploration_probability`` - fraction of decisions that are posterior
      draws rather than the posterior mean. It turns out to matter very little
      here (0.816 to 0.827 acceptance across 0.1 to 0.6), which is itself the
      finding: what limits this model is the variance of the estimate, not how
      often it explores.

    ``pool_batch`` and ``coordinate`` are **off by default and should stay
    off.** Both act across the rows of a batch, and a batch only ever has more
    than one row inside the simulator, where the loader packs an episode's four
    ``report_node_id`` rows together. In production each node runs its own
    agent and ``predict_action`` receives a single context (main.py: featurize
    returns one row); sharing mode aggregates reports into one report first.
    Enabling them improves simulator numbers by exploiting a property of the
    harness that production does not have. With them off, this model returns
    the same decision for a context whether it is passed alone or alongside
    others.

    **Measured on the 0..8..0 sweep, 40 episodes/phase, 3 seeds, scored against
    the acceptable-timeout sets:**

    ==========================  =========  ======
    metric                      this       cmab
    ==========================  =========  ======
    acceptance, all episodes    0.827      0.895
    acceptance, steady state    0.865      0.876
    commitment                  0.769      0.957
    episodes to settle          17.6       8.5
    episodes to settle, 2 -> 1  0.7        40.0
    ==========================  =========  ======

    **This model is worse than cmab on every aggregate metric.** The one thing
    it does better is the case it was built for: when phase 1 returns from
    phase 2, cmab scores 0.00 for the entire 40-episode visit - it never
    recovers 300ms at all - while this model is correct within one episode,
    because it chose 300ms throughout phase 2 and had nothing to switch.

    The reason it loses elsewhere is estimate variance. cmab averages 32 recent
    samples of an arm and ignores context entirely, which is very stable and
    commits hard; this averages the few nearest contexts, which tracks the
    workload but is noisier, so it neither commits as firmly nor settles as
    fast. Widening the local neighborhood does not fix it - acceptance falls
    from 0.821 at 8 neighbors to 0.782 at 48, because a wider neighborhood
    reaches into adjacent phases with different optima. Closing this gap needs
    a better similarity than Euclidean distance over standardized features -
    a learned metric, or tree-derived proximity - not more tuning.
        """

    def __init__(
        self,
        action_dim: int,
        n_features: int,
        radius: float = 1.5,
        local_neighbors: int = 8,
        wide_neighbors: int = 64,
        prior_weight: float = 16.0,
        exploration_probability: float = 0.1,
        robust: bool = True,
        pool_batch: bool = False,
        prior_cold_start: bool = True,
        min_variance_fraction: float = 0.005,
        memory_size: int = 20_000,
        coordinate: bool = False,
        seed: Optional[int] = None,
    ):
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")
        if n_features <= 0:
            raise ValueError(f"n_features must be > 0, got {n_features}")
        if radius <= 0.0:
            raise ValueError(f"radius must be > 0, got {radius}")
        if local_neighbors <= 0:
            raise ValueError(f"local_neighbors must be > 0, got {local_neighbors}")
        if wide_neighbors <= 0:
            raise ValueError(f"wide_neighbors must be > 0, got {wide_neighbors}")
        if prior_weight < 0.0:
            raise ValueError(f"prior_weight must be >= 0, got {prior_weight}")
        if not 0.0 <= exploration_probability <= 1.0:
            raise ValueError(
                "exploration_probability must be in [0, 1], got "
                f"{exploration_probability}"
            )
        if not 0.0 <= min_variance_fraction <= 1.0:
            raise ValueError(
                f"min_variance_fraction must be in [0, 1], got {min_variance_fraction}"
            )
        if memory_size <= 0:
            raise ValueError(f"memory_size must be > 0, got {memory_size}")

        self._act_dim = int(action_dim)
        self._obs_dim = int(n_features)
        self._radius = float(radius)
        self._local_neighbors = int(local_neighbors)
        self._wide_neighbors = int(wide_neighbors)
        self._prior_weight = float(prior_weight)
        self._exploration_probability = float(exploration_probability)
        self._robust = bool(robust)
        self._pool_batch = bool(pool_batch)
        self._prior_cold_start = bool(prior_cold_start)
        self._min_variance_fraction = float(min_variance_fraction)
        self._memory_size = int(memory_size)
        self._coordinate = bool(coordinate)
        self._seed = int(time.time_ns() & 0x7FFFFFFF) if seed is None else int(seed)
        self.reset_model()

    # -- standardization ----------------------------------------------------

    def _observe_features(self, rows: np.ndarray) -> None:
        """Welford, so features are comparable without a pre-pass over data."""
        for row in rows:
            self._feature_count += 1
            delta = row - self._feature_mean
            self._feature_mean += delta / self._feature_count
            self._feature_m2 += delta * (row - self._feature_mean)

    def _standardize(self, rows: np.ndarray) -> np.ndarray:
        if self._feature_count < 2:
            return rows - self._feature_mean
        scale = np.sqrt(np.maximum(self._feature_m2 / (self._feature_count - 1), 0.0))
        scale[scale <= 0.0] = 1.0
        return (rows - self._feature_mean) / scale

    def _global_reward_variance(self) -> float:
        if self._reward_count < 2:
            return 1.0
        return max(self._reward_m2 / (self._reward_count - 1), 1e-12)

    # -- decisions ----------------------------------------------------------

    def predict_action(self, features: np.ndarray, greedy: bool = False) -> np.ndarray:
        rows = self._as_batch(features)
        batch_size = rows.shape[0]
        scores = np.zeros((batch_size, self._act_dim), dtype=float)

        if not self._contexts:
            scores[:] = np.inf
            self._deal_unexplored(scores, rows)
            return scores

        stored = self._standardize(np.asarray(self._contexts, dtype=float))
        arms = np.asarray(self._arms, dtype=int)
        rewards = np.asarray(self._rewards, dtype=float)
        queries = self._standardize(rows)
        scale = np.sqrt(self._obs_dim)

        # One coin per row, so a batch naturally splits between exploiting and
        # probing rather than doing the same thing on all four nodes.
        explore = self._rng.random(batch_size) < self._exploration_probability

        # Per (row, arm): posterior mean, its variance, and its weight in
        # samples. Collected first so the batch can be pooled before drawing.
        means = np.zeros((batch_size, self._act_dim), dtype=float)
        variances = np.zeros((batch_size, self._act_dim), dtype=float)
        weights = np.zeros((batch_size, self._act_dim), dtype=float)
        unknown = np.zeros((batch_size, self._act_dim), dtype=bool)

        for i, query in enumerate(queries):
            # RMS per feature, so radius means the same thing at any width.
            distances = np.linalg.norm(stored - query, axis=1) / scale
            for arm in range(self._act_dim):
                mask = arms == arm
                if not mask.any():
                    unknown[i, arm] = True
                    continue
                arm_distances = distances[mask]
                arm_rewards = rewards[mask]

                n_local = int((arm_distances <= self._radius).sum())
                if n_local == 0 and not self._prior_cold_start:
                    unknown[i, arm] = True
                    continue

                # The mean comes from a tight neighborhood and the confidence
                # from a broad one, because the two want different widths here.
                # Between-phase centroid distance (0.22-0.44) is smaller than
                # within-phase pairwise spread (~0.7-1.0), so any radius wide
                # enough to gather same-phase samples also gathers neighboring
                # phases. Taking the mean from the few nearest keeps it honest;
                # counting the whole region for confidence lets the posterior
                # actually narrow with experience, which a fixed neighbor count
                # can never do - its width floors at sqrt(var / k).
                tight = min(self._local_neighbors, arm_rewards.size)
                closest = np.argpartition(arm_distances, tight - 1)[:tight]
                local_rewards = arm_rewards[closest]
                if self._robust and tight > 2:
                    # A neighborhood straddling two regimes carries a few
                    # samples from the wrong one, whose rewards can be an order
                    # of magnitude off. The median ignores them, the mean does
                    # not - and phase 7 sits next to phases whose 800ms reward
                    # is several times its own.
                    mu_local = float(np.median(local_rewards))
                    deviation = np.median(np.abs(local_rewards - mu_local))
                    variance = float((1.4826 * deviation) ** 2)
                else:
                    mu_local = float(local_rewards.mean())
                    variance = (
                        float(local_rewards.var(ddof=1))
                        if tight > 1
                        else self._global_reward_variance()
                    )

                # The wide neighborhood is the empirical prior.
                k = min(self._wide_neighbors, arm_rewards.size)
                widest = np.argpartition(arm_distances, k - 1)[:k]
                wide_rewards = arm_rewards[widest]
                mu_wide = float(wide_rewards.mean())
                wide_variance = (
                    float(wide_rewards.var(ddof=1))
                    if k > 1
                    else self._global_reward_variance()
                )

                weight = n_local + self._prior_weight
                if n_local == 0:
                    # Nothing measured inside the radius, but this arm has been
                    # measured elsewhere. Entering a new regime with no opinion
                    # at all costs a full round of blind probing every time the
                    # workload moves; the wide neighborhood is a weak guess and
                    # a weak guess beats none. Confidence stays at prior_weight
                    # alone, so it is still explored - just not from scratch.
                    means[i, arm] = mu_wide
                    variances[i, arm] = wide_variance
                    weights[i, arm] = self._prior_weight
                    continue
                means[i, arm] = (
                    n_local * mu_local + self._prior_weight * mu_wide
                ) / weight
                variances[i, arm] = variance
                weights[i, arm] = weight

        if self._pool_batch and batch_size > 1:
            # Every row of a batch is one node's view of the same workload, so
            # their estimates are repeated measurements of one quantity. Pooling
            # them before drawing removes the row-to-row disagreement that comes
            # from feature noise rather than from real uncertainty - which is
            # what a context-free window mean gets for free, and what a purely
            # per-row context model gives up. Exploration is unaffected: the
            # draw still happens per row, off the pooled estimate.
            for arm in range(self._act_dim):
                known = ~unknown[:, arm]
                if known.sum() < 2:
                    continue
                total = weights[known, arm].sum()
                pooled_mean = float((means[known, arm] * weights[known, arm]).sum() / total)
                # Confidence is averaged, not summed: the rows overlap heavily
                # in the samples they drew on, so adding their weights would
                # claim far more independent evidence than exists.
                pooled_weight = float(weights[known, arm].mean())
                pooled_variance = float(variances[known, arm].mean())
                means[known, arm] = pooled_mean
                weights[known, arm] = pooled_weight
                variances[known, arm] = pooled_variance

        floor = self._min_variance_fraction * self._global_reward_variance()
        for i in range(batch_size):
            for arm in range(self._act_dim):
                if unknown[i, arm]:
                    scores[i, arm] = np.inf
                elif greedy or not explore[i]:
                    scores[i, arm] = means[i, arm]
                else:
                    scores[i, arm] = float(
                        self._rng.normal(
                            means[i, arm],
                            np.sqrt(max(variances[i, arm], floor) / weights[i, arm]),
                        )
                    )

        self._break_ties(scores)
        self._deal_unexplored(scores, rows)
        return scores

    def _deal_unexplored(self, scores: np.ndarray, rows: np.ndarray) -> None:
        """Spread a cold start across the batch instead of stacking it.

        Rows are separate nodes and the generator gives each its own timeout,
        so when several arms are unexplored the batch can measure them in
        parallel. Left alone they all tie at ``+inf`` and the argmax hands
        every row the lowest index, turning one episode of work into one per
        arm.

        This assumes nodes may legitimately run different timeouts at once,
        which holds for the offline sweep but not for main.py's sharing mode,
        where the fleet must act in lockstep for an aggregated reward to refer
        to a single action. Pass ``coordinate=False`` there.
        """
        if not self._coordinate or scores.shape[0] < 2:
            return
        unexplored = [
            arm for arm in range(self._act_dim) if np.isposinf(scores[:, arm]).all()
        ]
        if len(unexplored) < 2:
            return
        self._rng.shuffle(unexplored)
        for position, arm in enumerate(unexplored):
            if position >= scores.shape[0]:
                break
            scores[position] = -np.inf
            scores[position, arm] = np.inf

    def _break_ties(self, scores: np.ndarray) -> None:
        """Randomize the winner among exactly equal finite scores, in place.

        Infinite scores are left alone so unexplored arms keep a deterministic
        round-robin order rather than being shuffled by two mechanisms at once.
        """
        for i in range(scores.shape[0]):
            row = scores[i]
            best = float(np.max(row))
            if not np.isfinite(best):
                continue
            tied = np.flatnonzero(row == best)
            if len(tied) > 1:
                row[int(self._rng.choice(tied))] = np.nextafter(best, np.inf)

    # -- learning -----------------------------------------------------------

    def update_model(
        self,
        features: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | float,
        backward: bool = True,
    ) -> float:
        if not backward:
            return 0.0
        rows = self._as_batch(features)
        arms = np.asarray(actions, dtype=int).ravel()
        values = np.asarray(rewards, dtype=float).ravel()
        batch_size = rows.shape[0]
        if arms.size == 1 and batch_size > 1:
            arms = np.repeat(arms, batch_size)
        if values.size == 1 and batch_size > 1:
            values = np.repeat(values, batch_size)
        if arms.size != batch_size:
            raise ValueError(
                f"actions batch size mismatch: expected {batch_size}, got {arms.size}"
            )
        if values.size != batch_size:
            raise ValueError(
                f"rewards batch size mismatch: expected {batch_size}, got {values.size}"
            )

        self._observe_features(rows)
        for row, arm, value in zip(rows, arms, values):
            arm = int(arm)
            if not 0 <= arm < self._act_dim:
                raise ValueError(f"action out of range: {arm}")
            self._contexts.append(row.copy())
            self._arms.append(arm)
            self._rewards.append(float(value))
            self._reward_count += 1
            delta = float(value) - self._reward_mean
            self._reward_mean += delta / self._reward_count
            self._reward_m2 += delta * (float(value) - self._reward_mean)
        return 0.0

    def _as_batch(self, features: np.ndarray) -> np.ndarray:
        rows = np.asarray(features, dtype=float)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.shape[1] != self._obs_dim:
            raise ValueError(
                f"Expected {self._obs_dim} features, got {rows.shape[1]}"
            )
        return rows

    def reset_model(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self._feature_count = 0
        self._feature_mean = np.zeros(self._obs_dim, dtype=float)
        self._feature_m2 = np.zeros(self._obs_dim, dtype=float)
        self._reward_count = 0
        self._reward_mean = 0.0
        self._reward_m2 = 0.0
        self._contexts: deque = deque(maxlen=self._memory_size)
        self._arms: deque = deque(maxlen=self._memory_size)
        self._rewards: deque = deque(maxlen=self._memory_size)
