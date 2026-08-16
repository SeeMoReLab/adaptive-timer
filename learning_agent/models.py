from abc import ABC, abstractmethod
import copy
import csv
from dataclasses import dataclass, field
from enum import IntEnum
import time
from pathlib import Path
from typing import Dict, Generic, List, Optional, Tuple, TypeVar, Any
from sklearn.tree import  DecisionTreeRegressor
from xgboost import XGBRegressor
from collections import deque
import torch
import torch.nn.functional as F
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
    PHASE_5 = PbftTimeoutOption.TIMEOUT_800_MS
    PHASE_6 = PbftTimeoutOption.TIMEOUT_800_MS
    PHASE_7 = PbftTimeoutOption.TIMEOUT_1800_MS
    PHASE_8 = PbftTimeoutOption.TIMEOUT_300_MS

PHASE_OPTIONS: Tuple[PhaseMapping, ...] = (
    PhaseMapping.PHASE_0,
    PhaseMapping.PHASE_1,
    PhaseMapping.PHASE_2,
    PhaseMapping.PHASE_3,
    PhaseMapping.PHASE_4,
    PhaseMapping.PHASE_5,
    PhaseMapping.PHASE_6,
    PhaseMapping.PHASE_7,
    PhaseMapping.PHASE_8,
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
    def featurize(self, state: S) -> torch.Tensor:
        pass


class ActionCodec(Generic[A], ABC):
    """Converts between model actions and domain-specific actions."""

    @abstractmethod
    def decode(self, action: torch.Tensor) -> A:
        pass

    @abstractmethod
    def encode(self, action: object) -> torch.Tensor:
        pass


class TensorModel(ABC):
    """Model that consumes and produces tensors."""

    @abstractmethod
    def predict_action(self, features: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def update_model(
        self, features: torch.Tensor, action: torch.Tensor, reward: float
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

    def featurize(self, state: object) -> torch.Tensor:
        features = report_to_feature_values(self._protocol, state)
        return torch.tensor(features, dtype=torch.float32)

class ProtocolTimeoutActionCodec(ActionCodec[agent_pb2.Timeout]):
    """Converts discrete arm indices to and from protocol timeout actions."""

    def __init__(self, protocol: int):
        self._protocol = _validate_protocol(protocol)

    def decode(self, action: torch.Tensor) -> agent_pb2.Timeout:
        if action.numel() == 1:
            option_index = int(round(float(action.item())))
        else:
            option_index = int(torch.argmax(action).item())
        return action_index_to_timeout(self._protocol, option_index)

    def encode(self, action: object) -> torch.Tensor:
        option_index = timeout_to_action_index(self._protocol, action)
        return torch.tensor(float(option_index), dtype=torch.float32)

class QuadraticRewardScaler(RewardTransformer):
    """Quadratic shaping below 1000, linear above. Prior work"""

    @staticmethod
    def transform(throughput: float) -> float:
        return reward_from_throughput(throughput)

class LinearTorchModel(TensorModel):
    """Linear classifier over protocol-specific discrete timeout options."""

    def __init__(self, action_dim: int, n_features: int, lr: float = 0.01):
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")
        self._action_dim = int(action_dim)
        self.weights = torch.zeros((self._action_dim, n_features), requires_grad=True)
        self.optimizer = torch.optim.AdamW([self.weights], lr=lr,  weight_decay=0.01)

    def predict_action(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 1:
            return torch.mv(self.weights, features)
        return torch.matmul(features, self.weights.t())

    def update_model(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor | float,
        backward: bool = True,
    ) -> float:
        with torch.set_grad_enabled(backward):
            if features.dim() == 1:
                features = features.unsqueeze(0)

            actions_tensor = torch.as_tensor(actions, dtype=torch.float32).view(-1)
            rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32).view(-1)

            batch_size = features.shape[0]
            if actions_tensor.numel() == 1 and batch_size > 1:
                actions_tensor = actions_tensor.repeat(batch_size)
            if rewards_tensor.numel() == 1 and batch_size > 1:
                rewards_tensor = rewards_tensor.repeat(batch_size)
            if actions_tensor.numel() != batch_size:
                raise ValueError(
                    "actions batch size mismatch: "
                    f"expected {batch_size}, got {actions_tensor.numel()}"
                )
            if rewards_tensor.numel() != batch_size:
                raise ValueError(
                    "rewards batch size mismatch: "
                    f"expected {batch_size}, got {rewards_tensor.numel()}"
                )

            target_index = torch.round(actions_tensor).long().view(-1)
            targets = torch.clamp(target_index, 0, self._action_dim - 1)
            logits = torch.matmul(features, self.weights.t())
            reward_scales = torch.clamp(rewards_tensor, min=0.0) / 1000.0
            loss_vector = F.cross_entropy(logits, targets, reduction="none")
            loss = (loss_vector * reward_scales).mean()

            if backward and float(loss.item()) > 0.0:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            return float(loss.item())

    def reset_model(self) -> None:
        torch.nn.init.zeros_(self.weights)

    def save(self) -> torch.Tensor:
        return self.weights.clone().detach()

    def load(self, weights: torch.Tensor) -> None:
        with torch.no_grad():
            self.weights.copy_(weights)

class RandomTimeoutModel(TensorModel):
    """Selects a random timeout action index each prediction."""

    def __init__(self, action_dim: int, seed: Optional[int] = None):
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")
        self._action_dim = int(action_dim)
        if seed is None:
            seed = int(time.time_ns() & 0x7FFFFFFF)
        self._initial_seed = int(seed)
        self._rng = torch.Generator()
        self._rng.manual_seed(self._initial_seed)

    def predict_action(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 1:
            option_index = int(
                torch.randint(0, self._action_dim, (1,), generator=self._rng).item()
            )
            return torch.tensor(float(option_index), dtype=torch.float32)

        batch_size = int(features.shape[0])
        return torch.randint(
            0, self._action_dim, (batch_size,), generator=self._rng, dtype=torch.int64
        ).to(torch.float32)

    def update_model(
        self, features: torch.Tensor, action: torch.Tensor, reward: float
    ) -> float:
        del features, action, reward
        return 0.0

    def reset_model(self) -> None:
        self._rng.manual_seed(self._initial_seed)

    def save(self) -> Dict[str, Any]:
        pass

    def load(self, checkpoint: Dict[str, Any]) -> None:
        pass

class PavlovTimeoutModel(TensorModel):
    """Win-stay/lose-shift (Pavlov) timeout heuristic against the last reward.

    A non-learning baseline: no context, no per-arm statistics, no threshold, no
    calibration, O(1) state. Arms are ordered by ascending timeout, so for pbft
    arm 1 = 300 ms, arm 2 = 800 ms, arm 3 = 1800 ms. The run opens on arm 1 and
    the tracked reward starts at 0, so the first episode is always a win. Then,
    every episode:

        observed >= tracked -> keep the current arm
        observed <  tracked -> step to the next arm (1 -> 2 -> 3 -> 1)
        tracked = observed

    Measured cost of the strict "compare against t-1" reference: per-episode
    noise on real hardware is as large as the between-arm difference, so the
    comparison is close to a coin flip and the rule churns. Over the fixed-arm
    CloudLab runs (local/tps_rule.py, scripts/cloudlab/{300,800,1800}, 134
    episodes, 30% switch cost) it switches on 56 of 134 episodes and yields
    5444 tps, against 5935 for the best fixed arm and 8741 for a per-phase
    oracle -- i.e. as specified it does not beat leaving the timeout alone.
    local/tps_rule.py holds the variants that do, and re-measures either.
    """

    def __init__(
        self,
        action_dim: int,
        protocol: Optional[int] = None,
        start_index: Optional[int] = None,
    ):
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")

        self._action_dim = int(action_dim)
        # Walk the arms in ascending timeout order, not raw index order: for
        # sbft/tendermint the index encodes a triple, so the two differ.
        if protocol is None:
            self._order = list(range(self._action_dim))
        else:
            protocol = _validate_protocol(protocol)
            self._order = sorted(
                range(self._action_dim),
                key=lambda idx: timeout_index_to_ms_for_protocol(protocol, idx),
            )
        self._position = {idx: pos for pos, idx in enumerate(self._order)}

        if start_index is None:
            # Arm 1, the shortest timeout.
            self._start_index = self._order[0]
        else:
            self._start_index = int(start_index)
            if self._start_index not in self._position:
                raise ValueError(
                    f"start_index must be in [0, {self._action_dim}), "
                    f"got {start_index}"
                )

        self.reset_model()

    def _to_index(self, action: torch.Tensor) -> int:
        if action.numel() == 1:
            return int(round(float(action.item())))
        return int(torch.argmax(action).item())

    def predict_action(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 1:
            return torch.tensor(float(self._index), dtype=torch.float32)
        batch_size = int(features.shape[0])
        return torch.full((batch_size,), float(self._index), dtype=torch.float32)

    def update_model(
        self, features: torch.Tensor, action: torch.Tensor, reward: float
    ) -> float:
        del features
        arm = self._to_index(action)
        if arm in self._position:
            # Follow the timeout that actually ran: in shared or consensus mode
            # the applied arm need not be the one this replica predicted.
            self._index = arm

        value = float(reward)
        if value < self._tracked:
            self._index = self._order[
                (self._position[self._index] + 1) % len(self._order)
            ]
        self._tracked = value
        return 0.0

    def reset_model(self) -> None:
        self._tracked = 0.0
        self._index = self._start_index

    @property
    def tracked_reward(self) -> float:
        return self._tracked

    def save(self) -> Dict[str, Any]:
        return {"tracked": self._tracked, "index": self._index}

    def load(self, checkpoint: Dict[str, Any]) -> None:
        self._tracked = float(checkpoint.get("tracked", 0.0))
        index = int(checkpoint.get("index", self._start_index))
        self._index = index if index in self._position else self._start_index

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

    def predict_action(self, features: torch.Tensor) -> torch.Tensor:
        del features
        option_index = self._current_option_index()
        return torch.tensor(float(option_index), dtype=torch.float32)

    def update_model(
        self, features: torch.Tensor, action: torch.Tensor, reward: float
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

class ThompsonSamplingModelOld(TensorModel): 
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

    def predict_action(self, features: torch.Tensor) -> torch.Tensor:
        features_np = features.cpu().numpy()
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

        q_values = torch.tensor(predictions_stacked, device=features.device, dtype=torch.float32)
        return q_values

    def update_model(
        self, features: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor | float, backward: bool = True
    ) -> float:
        features_np = features.cpu().numpy()
        if features_np.ndim == 1:
            features_np = features_np.reshape(1, -1)
            
        if isinstance(actions, torch.Tensor):
            actions_np = actions.cpu().numpy().astype(int).flatten()
        else:
            actions_np = np.array([int(actions)])
            
        if isinstance(rewards, torch.Tensor):
            rewards_np = rewards.cpu().numpy().flatten()
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

class ThompsonSamplingModelv01(TensorModel):
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

    def predict_action(self, features: torch.Tensor, e = 0.1) -> torch.Tensor:
        """e-TS with forced coverage.

        - Any action absent from the shared sliding window is force-pulled
          (ties between several absent actions break uniformly at random).
        - Otherwise the ensemble is summarised per arm into a mean and a
          variance over the fitted bootstrap heads. With probability ``e``
          take the Thompson draw ``N(mean, var)`` (a Gaussian fit to the
          bootstrap posterior instead of one randomly picked head); with
          probability ``1-e`` exploit greedily on the mean."""
        features_np = features.cpu().numpy()
        if features_np.ndim == 1:
            features_np = features_np.reshape(1, -1)

        batch_size = features_np.shape[0]
        predictions_stacked = np.zeros((batch_size, self._act_dim), dtype=float)
        explore = self._rng.random(batch_size) < e

        present_arms = {int(a) for a in self.memory["actions"]}
        missing_arms = [a for a in range(self._act_dim) if a not in present_arms]
        fitted_heads = [h for h in range(self._n_heads) if self.is_fitted[h]]

        for i in range(batch_size):
            state_rows = np.repeat(features_np[i : i + 1], self._act_dim, axis=0)
            action_col = np.arange(self._act_dim, dtype=float).reshape(-1, 1)
            state_action_input = np.hstack((state_rows, action_col))

            if fitted_heads:
                head_preds = np.stack(
                    [self.models[h].predict(state_action_input) for h in fitted_heads]
                )
                mean = np.mean(head_preds, axis=0)
                if explore[i]:
                    std = (
                        np.std(head_preds, axis=0, ddof=1)
                        if len(fitted_heads) > 1
                        else np.zeros_like(mean)
                    )
                    predictions_stacked[i, :] = self._rng.normal(mean, std)
                else:
                    predictions_stacked[i, :] = mean

            # Coverage overrides either branch: absent arms must be re-pulled.
            for arm in missing_arms:
                predictions_stacked[i, arm] = 1e9

        q_values = torch.tensor(predictions_stacked, device=features.device, dtype=torch.float32)
        return q_values

    def update_model(
        self, features: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor | float, backward: bool = True
    ) -> float:
        features_np = features.cpu().numpy()
        if features_np.ndim == 1:
            features_np = features_np.reshape(1, -1)
            
        if isinstance(actions, torch.Tensor):
            actions_np = actions.cpu().numpy().astype(int).flatten()
        else:
            actions_np = np.array([int(actions)])
            
        if isinstance(rewards, torch.Tensor):
            rewards_np = rewards.cpu().numpy().flatten()
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
                max_depth=3,
                min_samples_leaf=1,
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
    """Per-arm bootstrapped-tree CMAB

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

    def predict_action(self, features: torch.Tensor, e = 0.5) -> torch.Tensor:
        """e-TS with forced coverage: any arm absent from the recent-pull
        window (2x bucket size) is force-pulled so stale arms cannot starve.
        Otherwise with probability e take the Thompson draw (one bootstrap
        head per arm); with probability 1-e exploit the arm with the highest
        median over its ensemble of bootstrap-head predictions (each head
        outputs a mean, so this is a robust median-of-means)."""
        features_np = features.cpu().numpy()
        if features_np.ndim == 1:
            features_np = features_np.reshape(1, -1)

        batch_size = features_np.shape[0]
        predictions_stacked = np.zeros((batch_size, self._act_dim), dtype=float)
        # One independent head (bootstrap draw) per arm, per row.
        sampled_heads = self._rng.integers(
            0, self._n_heads, size=(batch_size, self._act_dim)
        )
        explore = self._rng.random(batch_size) < e
        recent_arms = {int(a) for a in self._recent_actions}

        for i in range(batch_size):
            row = features_np[i : i + 1]
            for arm in range(self._act_dim):
                if not self.is_fitted[arm] or arm not in recent_arms:
                    # Arm without a fitted window yet, or with no pull in the
                    # recent-action window: force a pull (ties between forced
                    # arms break uniformly at random below).
                    predictions_stacked[i, arm] = 1e9
                    continue
                if explore[i]:
                    head = self.models[arm][sampled_heads[i, arm]]
                    predictions_stacked[i, arm] = head.predict(row)[0]
                else:
                    head_preds = [
                        self.models[arm][h].predict(row)[0]
                        for h in range(self._n_heads)
                    ]
                    predictions_stacked[i, arm] = float(np.mean(head_preds))

        for i in range(batch_size):
            row_preds = predictions_stacked[i]
            tied_indices = np.flatnonzero(np.isclose(row_preds, np.max(row_preds)))

            if len(tied_indices) > 1:
                best_index = self._rng.choice(tied_indices)
                predictions_stacked[i, best_index] += 1e-5

        q_values = torch.tensor(predictions_stacked, device=features.device, dtype=torch.float32)
        return q_values

    def update_model(
        self, features: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor | float, backward: bool = True
    ) -> float:
        features_np = features.cpu().numpy()
        if features_np.ndim == 1:
            features_np = features_np.reshape(1, -1)
            
        if isinstance(actions, torch.Tensor):
            actions_np = actions.cpu().numpy().astype(int).flatten()
        else:
            actions_np = np.array([int(actions)])
            
        if isinstance(rewards, torch.Tensor):
            rewards_np = rewards.cpu().numpy().flatten()
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
            self._recent_actions.append(arm)

        if backward:
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
                    min_impurity_decrease=0.0,
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
    
        self._recent_actions = deque(maxlen= self._max_bucket_size)

@dataclass
class _BlockPolicy:
    """Q(i,m): the policy distribution frozen at the start of block m.

    ``model`` is the tree that was in force there (None for block 0, before
    anything is fitted) and ``epsilon`` is that block's exploration rate, so
    the pair reproduces the exact sampling distribution Q^{nu_m}(.|x) used at
    the time. ``length`` is block m's length, which doubles as the length of a
    replay phase carrying index m."""

    model: Optional[XGBRegressor]
    epsilon: float
    length: int


@dataclass
class _Replay:
    """One in-flight replay phase: (m, [t, t + 2^m L - 1]) from Algorithm 1."""

    index: int
    rounds_left: int
    rows: List[int] = field(default_factory=list)


class CMAB(TensorModel):
    def __init__(
        self,
        action_dim: int,
        n_features: int,
        epsilon: float = 0.01,
        block_length: int = 8,
        max_block_length: int = 256,
        nu: float = 0.15,
        max_depth: int = 12,
        min_samples_leaf: int = 1,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        enable_replay: bool = True,
        replay_rate: float = 1.0,
        replay_var_ratio: float = 4.0,
        max_snapshots: int = 8,
        decay_epsilon: bool = True,
        seed: Optional[int] = None,
    ):
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")
        if not 0.0 < epsilon <= 1.0:
            raise ValueError(f"epsilon must be in (0, 1], got {epsilon}")
        if block_length <= 0:
            raise ValueError(f"block_length must be > 0, got {block_length}")
        if replay_rate < 0.0:
            raise ValueError(f"replay_rate must be >= 0, got {replay_rate}")
        if min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be >= 1, got {min_samples_leaf}")
        if n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {n_estimators}")
        if learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")

        self._act_dim = int(action_dim)
        self._obs_dim = int(n_features)
        self._epsilon = float(epsilon)
        self._init_block_length = int(block_length)
        self._max_block_length = int(max_block_length)
        self._nu = float(nu)
        self._max_depth = int(max_depth)
        self._min_samples_leaf = int(min_samples_leaf)
        self._n_estimators = int(n_estimators)
        self._learning_rate = float(learning_rate)
        self._enable_replay = bool(enable_replay)
        self._replay_rate = float(replay_rate)
        self._replay_var_ratio = float(replay_var_ratio)
        self._max_snapshots = max(1, int(max_snapshots))
        self._decay_epsilon = bool(decay_epsilon)
        self._seed = seed

        self.reset_model()

    def _block_epsilon(self, block_index: int) -> float:
        if not self._decay_epsilon:
            return self._epsilon
        return float(min(1.0, self._epsilon * (2.0 ** (-0.5 * block_index))))

    def _maybe_start_replay(self) -> None:
        if not self._enable_replay:
            return
        j = self._block_index
        if j <= 0:
            return
        available = [
            m
            for m in range(min(j, len(self._block_policies)))
            if self._block_policies[m].model is not None
        ]
        if not available:
            return

        weights = np.array([2.0 ** (-0.5 * m) for m in available], dtype=float)
        p_rep = (
            self._replay_rate
            * (2.0 ** (-0.5 * j))
            * float(weights.sum())
            / float(self._init_block_length)
        )
        if self._rng.random() >= min(1.0, p_rep):
            return

        m = int(self._rng.choice(available, p=weights / weights.sum()))
        length = int(self._block_policies[m].length)
        self._replays.append(
            _Replay(index=m, rounds_left=length)
        )
        self.replay_starts += 1

    def _active_replay_index(self) -> Optional[int]:
        """M_t from line 13, then m ~ Uniform(M_t) from line 17."""
        indices = sorted({r.index for r in self._replays if r.rounds_left > 0})
        if not indices:
            return None
        return int(self._rng.choice(indices))

    def predict_action(self, features: torch.Tensor, e: Optional[float] = None) -> torch.Tensor:
        override_epsilon = None if e is None else float(e)
        features_np = features.cpu().numpy()
        if features_np.ndim == 1:
            features_np = features_np.reshape(1, -1)

        batch_size = features_np.shape[0]
        predictions_stacked = np.zeros((batch_size, self._act_dim), dtype=float)

        pulled_arms = {int(a) for a in self.memory["actions"]}
        missing_arms = [a for a in range(self._act_dim) if a not in pulled_arms]

        self._pending = []
        for i in range(batch_size):
            # Step 1. Randomly start a replay phase.
            self._maybe_start_replay()

            # Step 2. Sample an action, replaying Q(i,m) when M_t is non-empty.
            replay_m = self._active_replay_index()
            if replay_m is None:
                tree = self.model if self.is_fitted else None
                epsilon = (
                    self._block_epsilon(self._block_index)
                    if override_epsilon is None
                    else override_epsilon
                )
            else:
                policy = self._block_policies[replay_m]
                tree = policy.model
                epsilon = policy.epsilon
            if tree is not None:
                predictions_stacked[i, :] = tree.predict(features_np[i : i + 1])[0]

            if missing_arms:
                arm = int(self._rng.choice(missing_arms))
                missing_arms.remove(arm)
                prob = 1.0 / self._act_dim
            else:
                row_preds = predictions_stacked[i]
                tied = np.flatnonzero(np.isclose(row_preds, np.max(row_preds)))
                greedy_arm = int(self._rng.choice(tied))
                probs = np.full(self._act_dim, epsilon / self._act_dim)
                probs[greedy_arm] += 1.0 - epsilon
                arm = int(self._rng.choice(self._act_dim, p=probs))
                prob = float(probs[arm])

            predictions_stacked[i, arm] = predictions_stacked[i].max() + 1.0
            open_replays = tuple(r for r in self._replays if r.rounds_left > 0)
            self._pending.append((arm, prob, open_replays))
            for replay in open_replays:
                replay.rounds_left -= 1
            self._last_prob_by_arm[arm] = prob

        q_values = torch.tensor(predictions_stacked, device=features.device, dtype=torch.float32)
        return q_values

    def update_model(
        self, features: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor | float, backward: bool = True
    ) -> float:
        features_np = features.cpu().numpy()
        if features_np.ndim == 1:
            features_np = features_np.reshape(1, -1)

        if isinstance(actions, torch.Tensor):
            actions_np = actions.cpu().numpy().astype(int).flatten()
        else:
            actions_np = np.array([int(actions)])

        if isinstance(rewards, torch.Tensor):
            rewards_np = rewards.cpu().numpy().flatten()
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

        pending = (
            self._pending
            if len(self._pending) == features_np.shape[0]
            else None
        )
        self._pending = []

        inserted = 0
        for i in range(len(actions_np)):
            arm = int(actions_np[i])
            if not 0 <= arm < self._act_dim:
                raise ValueError(f"action out of range: {arm}")

            raw_reward = max(0.0, float(rewards_np[i]))
            if raw_reward > self.global_r_max:
                ratio = self.global_r_max / raw_reward
                for target in self.memory["rewards"]:
                    target *= ratio
                self.global_r_max = raw_reward
            scaled_reward = raw_reward / self.global_r_max

            if pending is not None and pending[i][0] == arm:
                prob = pending[i][1]
            else:
                prob = self._last_prob_by_arm.get(arm, 1.0 / self._act_dim)

            ips_target = np.zeros(self._act_dim, dtype=float)
            ips_target[arm] = scaled_reward / prob
            self.memory["features"].append(features_np[i])
            self.memory["actions"].append(arm)
            self.memory["probs"].append(prob)
            self.memory["rewards"].append(ips_target)
            row = len(self.memory["rewards"]) - 1
            self.current_block.append(row)
            if pending is not None and pending[i][0] == arm:
                for replay in pending[i][2]:
                    replay.rows.append(row)
            else:
                for replay in self._replays:
                    replay.rows.append(row)
            inserted += 1

        if backward and inserted and len(self.memory["rewards"]) > 0:
            X = np.array(self.memory["features"])
            Y = np.array(self.memory["rewards"])
            self.model.fit(X, Y)
            self.is_fitted = True
            if self._run_expired_replays(X, Y):
                return 0.0
            self._check_shift(X, Y)

        return 0.0


    def _run_expired_replays(self, X: np.ndarray, Y: np.ndarray) -> bool:
        expired = [r for r in self._replays if r.rounds_left <= 0]
        if not expired:
            return False
        self._replays = [r for r in self._replays if r.rounds_left > 0]

        prior = list(range(self._block_start_row))
        for replay in expired:
            if not replay.rows or not prior:
                continue
            self.replay_tests += 1
            if self._end_of_replay_test(replay, prior, X, Y):
                self.replay_restarts += 1
                self._hard_reset()
                return True
        return False

    def _end_of_replay_test(
        self, replay: _Replay, prior: List[int], X: np.ndarray, Y: np.ndarray
    ) -> bool:

        rows_a = np.asarray(replay.rows, dtype=int)
        rows_b = np.asarray(prior, dtype=int)
        policy = self._block_policies[replay.index]

        threshold = self._nu * float(
            np.sqrt(self._init_block_length / max(1, policy.length))
        )

        candidates = self._candidate_policies(X)
        regret_a = self._empirical_regret(rows_a, Y, candidates)
        regret_b = self._empirical_regret(rows_b, Y, candidates)
        for name, r_a in regret_a.items():
            r_b = regret_b.get(name)
            if r_b is None:
                continue
            if r_a - 4.0 * r_b >= threshold:       
                self.replay_fail_reasons["regret_worse"] += 1
                return True
            if r_b - 4.0 * r_a >= threshold:      
                self.replay_fail_reasons["regret_better"] += 1
                return True

        for actions in candidates.values():         
            v_a = self._propensity_variance(rows_a, actions, policy, X)
            v_b = self._propensity_variance(rows_b, actions, policy, X)
            if v_a - self._replay_var_ratio * v_b >= 0.0:
                self.replay_fail_reasons["variance"] += 1
                return True
        return False

    def _candidate_policies(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        n_rows = X.shape[0]
        candidates: Dict[str, np.ndarray] = {
            f"const{a}": np.full(n_rows, a, dtype=int) for a in range(self._act_dim)
        }
        for m, policy in enumerate(self._block_policies):
            if policy.model is not None:
                candidates[f"block{m}"] = np.argmax(policy.model.predict(X), axis=1)
        if self.is_fitted:
            candidates["current"] = np.argmax(self.model.predict(X), axis=1)
        return candidates

    def _empirical_regret(
        self, rows: np.ndarray, Y: np.ndarray, candidates: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        values: Dict[str, float] = {}
        for name, actions in candidates.items():
            value = self._snips(rows, actions[rows], Y)
            if value is not None:
                values[name] = value
        if not values:
            return {}
        best = max(values.values())
        return {name: best - value for name, value in values.items()}

    def _snips(
        self, rows: np.ndarray, policy_actions: np.ndarray, Y: np.ndarray
    ) -> Optional[float]:
        actions = np.asarray(self.memory["actions"], dtype=int)[rows]
        probs = np.asarray(self.memory["probs"], dtype=float)[rows]
        matched = actions == policy_actions
        if not matched.any():
            return None
        weight_sum = float((matched / probs).sum())
        if weight_sum <= 0.0:
            return None
        return float(Y[rows, policy_actions].sum() / weight_sum)

    def _propensity_variance(
        self,
        rows: np.ndarray,
        policy_actions: np.ndarray,
        policy: _BlockPolicy,
        X: np.ndarray,
    ) -> float:
        if policy.model is None:
            return float(self._act_dim)
        greedy = np.argmax(policy.model.predict(X[rows]), axis=1)
        floor = policy.epsilon / self._act_dim
        probs = np.where(policy_actions[rows] == greedy, 1.0 - policy.epsilon + floor, floor)
        return float(np.mean(1.0 / probs))

    def _check_shift(self, X: np.ndarray, Y: np.ndarray) -> None:
        pulled_arms = {int(a) for a in self.memory["actions"]}
        if len(pulled_arms) < self._act_dim:
            return
        if len(self.current_block) < self.block_length:
            return

        if self.prev_block:
            v_old = self._policy_value(self.prev_block, X, Y)
            v_new = self._policy_value(self.current_block, X, Y)
            if v_old is None or v_new is None:
                self._soft_reset()
                return
            if abs(v_old - v_new) > self._nu:
                self._hard_reset()
                return
            self.block_length = min(self.block_length * 2, self._max_block_length)
        self._soft_reset()

    def _policy_value(
        self, block: List[int], X: np.ndarray, Y: np.ndarray
    ) -> Optional[float]:
        rows = np.asarray(block, dtype=int)
        greedy = np.argmax(self.model.predict(X[rows]), axis=1)
        actions = np.asarray(self.memory["actions"])[rows]
        probs = np.asarray(self.memory["probs"])[rows]
        matched = actions == greedy
        if not matched.any():
            return None
        weight_sum = float((matched / probs).sum())
        return float(Y[rows, greedy].sum() / weight_sum)

    def reset_model(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self.global_r_max = 1.0
        self._last_prob_by_arm = {}
        self.replay_starts = 0
        self.replay_tests = 0
        self.replay_restarts = 0
        self.replay_fail_reasons = {"regret_worse": 0, "regret_better": 0, "variance": 0}
        self._hard_reset()

    def _hard_reset(self):
        self.model = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=self._n_estimators,
            learning_rate=self._learning_rate,
            max_depth=self._max_depth,
            min_child_weight=self._min_samples_leaf,
            random_state=42,
            n_jobs=1,
            verbosity=0,
        )
        self.is_fitted = False
        self.memory = {
                "features": list(),
                "actions": list(),
                "probs": list(),
                "rewards": list(),
            }
        self.block_length = self._init_block_length
        self.current_block = list()
        self.prev_block = list()
        self._pending = list()
        self._block_index = 0
        self._block_start_row = 0
        self._replays = list()
        self._block_policies = [
            _BlockPolicy(model=None, epsilon=self._block_epsilon(0), length=self.block_length)
        ]

    def _soft_reset(self):
        self.prev_block = self.current_block
        self.current_block = list()
        self._block_index += 1
        self._block_start_row = len(self.memory["rewards"])
        self._replays = list()
        self._block_policies.append(
            _BlockPolicy(
                model=copy.deepcopy(self.model) if self.is_fitted else None,
                epsilon=self._block_epsilon(self._block_index),
                length=self.block_length,
            )
        )

        for stale in self._block_policies[: -self._max_snapshots]:
            stale.model = None



class MultiHeadThompsonSamplingModel(ThompsonSamplingModel):
    """Backward-compatible name for the bootstrapped ensemble CMAB model (v03)."""

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