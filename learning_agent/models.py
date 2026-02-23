from abc import ABC, abstractmethod
import csv
from enum import IntEnum
import time
from pathlib import Path
from typing import Dict, Generic, List, Optional, Tuple, TypeVar

import torch
import torch.nn.functional as F

import agent_pb2

S = TypeVar("S")  # generic state type
A = TypeVar("A")  # generic action type


class TimeoutOption(IntEnum):
    TIMEOUT_300_MS = 300
    TIMEOUT_600_MS = 600
    TIMEOUT_1800_MS = 1800

TIMEOUT_OPTIONS: Tuple[TimeoutOption, ...] = (
    TimeoutOption.TIMEOUT_300_MS,
    TimeoutOption.TIMEOUT_600_MS,
    TimeoutOption.TIMEOUT_1800_MS,
)
TIMEOUT_VALUES_MS: Tuple[int, ...] = tuple(int(option) for option in TIMEOUT_OPTIONS)
TIMEOUT_MS_TO_INDEX: Dict[int, int] = {
    value_ms: idx for idx, value_ms in enumerate(TIMEOUT_VALUES_MS)
}

class PhaseMapping(IntEnum):
    PHASE_0 = TimeoutOption.TIMEOUT_300_MS
    PHASE_1 = TimeoutOption.TIMEOUT_300_MS
    PHASE_2 = TimeoutOption.TIMEOUT_300_MS
    PHASE_3 = TimeoutOption.TIMEOUT_300_MS
    PHASE_4 = TimeoutOption.TIMEOUT_600_MS
    PHASE_5 = TimeoutOption.TIMEOUT_600_MS
    PHASE_6 = TimeoutOption.TIMEOUT_600_MS
    PHASE_7 = TimeoutOption.TIMEOUT_1800_MS
    PHASE_8 = TimeoutOption.TIMEOUT_1800_MS
    PHASE_9 = TimeoutOption.TIMEOUT_1800_MS

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
    PhaseMapping.PHASE_9,
)

PHASE_VALUES_MS: Tuple[int, ...] = tuple(int(option) for option in PHASE_OPTIONS)
PHASE_INDEX_TO_TIMEOUT_INDEX: Dict[int, int] = {
    idx: TIMEOUT_MS_TO_INDEX[value_ms] for idx, value_ms in enumerate(PHASE_VALUES_MS)
}

PBFT_REPORT_FEATURE_FIELD_NAMES: Tuple[str, ...] = (
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
    "phase_propose_avg_delay_ms",
    "phase_propose_p95_delay_ms",
    "phase_write_avg_delay_ms",
    "phase_write_p95_delay_ms",
    "phase_accept_avg_delay_ms",
    "phase_accept_p95_delay_ms",
    "phase_post_decision_avg_delay_ms",
    "phase_post_decision_p95_delay_ms",
)
TWO_PC_OVER_PBFT_REPORT_FEATURE_FIELD_NAMES: Tuple[str, ...] = (
    "total_transactions",
    "single_shard_transactions",
    "cross_shard_transactions",
    "prepare_ok_count",
    "prepare_fail_count",
    "commit_ok_count",
    "abort_count",
    "in_doubt_count",
    "prepare_timeout_count",
    "commit_timeout_unknown_outcome_count",
    "abort_best_effort_timeout_count",
    "lock_contention_count",
    "prepare_timeout_rate",
    "abort_rate",
    "commit_rate",
    "lock_contention_rate",
    "avg_end_to_end_latency_ms",
    "p95_end_to_end_latency_ms",
    "p99_end_to_end_latency_ms",
    "avg_prepare_latency_ms",
    "p95_prepare_latency_ms",
    "avg_commit_latency_ms",
    "p95_commit_latency_ms",
    "avg_lock_wait_ms",
    "p95_lock_wait_ms",
    "avg_lock_hold_ms",
    "p95_lock_hold_ms",
    "reason_prepare_lock_busy_count",
    "reason_prepare_validation_fail_count",
    "reason_prepare_timeout_count",
    "reason_prepare_other_count",
    "reason_commit_timeout_unknown_outcome_count",
    "reason_abort_best_effort_timeout_count",
    "reason_other_count",
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


def _protocol_name(protocol: int) -> str:
    try:
        return agent_pb2.Protocol.Name(protocol)
    except ValueError:
        return f"UNKNOWN({protocol})"


def _validate_protocol(protocol: int) -> int:
    if protocol not in (
        agent_pb2.PROTOCOL_PBFT,
        agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT,
        agent_pb2.PROTOCOL_SBFT,
    ):
        raise ValueError(f"Unsupported protocol: {_protocol_name(protocol)}")
    return int(protocol)


def get_report_feature_field_names(protocol: int) -> Tuple[str, ...]:
    protocol = _validate_protocol(protocol)
    if protocol == agent_pb2.PROTOCOL_PBFT:
        return PBFT_REPORT_FEATURE_FIELD_NAMES
    if protocol == agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT:
        return TWO_PC_OVER_PBFT_REPORT_FEATURE_FIELD_NAMES
    return SBFT_REPORT_FEATURE_FIELD_NAMES


def get_report_feature_dim(protocol: int) -> int:
    return len(get_report_feature_field_names(protocol))


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


def _two_pc_pair_to_index(election_timeout_ms: int, prepare_timeout_ms: int) -> int:
    election_idx = timeout_ms_to_index(election_timeout_ms)
    prepare_idx = timeout_ms_to_index(prepare_timeout_ms)
    return election_idx * len(TIMEOUT_VALUES_MS) + prepare_idx


def _two_pc_index_to_pair(index: int) -> Tuple[int, int]:
    total = len(TIMEOUT_VALUES_MS) * len(TIMEOUT_VALUES_MS)
    bounded = max(0, min(index, total - 1))
    election_idx = bounded // len(TIMEOUT_VALUES_MS)
    prepare_idx = bounded % len(TIMEOUT_VALUES_MS)
    return timeout_index_to_ms(election_idx), timeout_index_to_ms(prepare_idx)


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


def get_action_space_size(protocol: int) -> int:
    protocol = _validate_protocol(protocol)
    if protocol == agent_pb2.PROTOCOL_PBFT:
        return len(TIMEOUT_VALUES_MS)
    if protocol == agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT:
        return len(TIMEOUT_VALUES_MS) * len(TIMEOUT_VALUES_MS)
    return len(TIMEOUT_VALUES_MS) * len(TIMEOUT_VALUES_MS) * len(TIMEOUT_VALUES_MS)


def report_to_feature_values(protocol: int, state: object) -> List[float]:
    protocol = _validate_protocol(protocol)
    fields = get_report_feature_field_names(protocol)
    return [float(getattr(state, field_name)) for field_name in fields]


def timeout_to_action_index(protocol: int, timeout_used: object) -> int:
    protocol = _validate_protocol(protocol)
    if protocol == agent_pb2.PROTOCOL_PBFT:
        timeout_msg = timeout_used
        return timeout_ms_to_index(int(timeout_msg.election_timeout_milliseconds))
    if protocol == agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT:
        timeout_msg = timeout_used
        return _two_pc_pair_to_index(
            int(timeout_msg.election_timeout_milliseconds),
            int(timeout_msg.prepare_timeout_milliseconds),
        )

    timeout_msg = timeout_used
    return _sbft_triple_to_index(
        int(timeout_msg.election_timeout_milliseconds),
        int(timeout_msg.slow_path_timeout_milliseconds),
        int(timeout_msg.batch_timeout_milliseconds),
    )


def action_index_to_timeout(protocol: int, index: int) -> agent_pb2.Timeout:
    protocol = _validate_protocol(protocol)
    if protocol == agent_pb2.PROTOCOL_PBFT:
        timeout_ms = timeout_index_to_ms(index)
        return agent_pb2.Timeout(
            pbft=agent_pb2.PbftTimeout(election_timeout_milliseconds=timeout_ms)
        )
    if protocol == agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT:
        election_timeout_ms, prepare_timeout_ms = _two_pc_index_to_pair(index)
        return agent_pb2.Timeout(
            two_pc_over_pbft=agent_pb2.TwoPcOverPbftTimeout(
                election_timeout_milliseconds=election_timeout_ms,
                prepare_timeout_milliseconds=prepare_timeout_ms,
            )
        )

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


def describe_action(protocol: int, index: int) -> str:
    protocol = _validate_protocol(protocol)
    if protocol == agent_pb2.PROTOCOL_PBFT:
        return f"election_timeout_ms={timeout_index_to_ms(index)}"
    if protocol == agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT:
        election_timeout_ms, prepare_timeout_ms = _two_pc_index_to_pair(index)
        return (
            f"election_timeout_ms={election_timeout_ms}, "
            f"prepare_timeout_ms={prepare_timeout_ms}"
        )
    election_timeout_ms, slow_path_timeout_ms, batch_timeout_ms = _sbft_index_to_triple(
        index
    )
    return (
        f"election_timeout_ms={election_timeout_ms}, "
        f"slow_path_timeout_ms={slow_path_timeout_ms}, "
        f"batch_timeout_ms={batch_timeout_ms}"
    )

class FeatureExtractor(Generic[S], ABC):
    """Converts an input state into a tensor for model consumption."""

    @abstractmethod
    def featurize(self, state: S) -> torch.Tensor:
        pass


class ActionDecoder(Generic[A], ABC):
    """Converts a model output tensor into a domain-specific action."""

    @abstractmethod
    def decode(self, action: torch.Tensor) -> A:
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

    @abstractmethod
    def save(self) -> None:
        pass

    @abstractmethod
    def load(self) -> None:
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


class ProtocolTimeoutActionDecoder(ActionDecoder[agent_pb2.Timeout]):
    """Decodes model output into a protocol-specific timeout action."""

    def __init__(self, protocol: int):
        self._protocol = _validate_protocol(protocol)

    def decode(self, action: torch.Tensor) -> agent_pb2.Timeout:
        if action.numel() == 1:
            option_index = int(round(float(action.item())))
        else:
            option_index = int(torch.argmax(action).item())
        return action_index_to_timeout(self._protocol, option_index)

class QuadraticRewardScaler(RewardTransformer):
    """Quadratic shaping below 1000, linear above. Prior work"""

    @staticmethod
    def transform(throughput: float) -> float:
        value = float(max(0, throughput))
        if value < 1000.0:
            return (value / 1000.0) ** 2 * 1000.0
        return value
            

class LinearTorchModel(TensorModel):
    """Linear classifier over protocol-specific discrete timeout options."""

    def __init__(self, action_dim: int, n_features: int, lr: float = 0.01):
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")
        self._action_dim = int(action_dim)
        self.weights = torch.zeros((self._action_dim, n_features), requires_grad=True)
        self.optimizer = torch.optim.SGD([self.weights], lr=lr)

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
        self._schedule = sorted(schedule, key=lambda entry: entry[0])
        if self._schedule[0][0] < 0:
            raise ValueError("Schedule times must be >= 0 seconds.")
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
                if time_s < 0:
                    raise ValueError(
                        f"Invalid schedule at line {lineno}: time_s must be >= 0."
                    )

                if protocol == agent_pb2.PROTOCOL_PBFT:
                    if len(row) < 2:
                        raise ValueError(
                            f"Invalid PBFT schedule at line {lineno}: expected 2 columns time_s,election_timeout_ms."
                        )
                    timeout_ms = int(row[1].strip())
                    option_index = timeout_ms_to_index(timeout_ms)
                elif protocol == agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT:
                    if len(row) < 3:
                        raise ValueError(
                            "Invalid 2PC-over-PBFT schedule at line "
                            f"{lineno}: expected 3 columns "
                            "time_s,election_timeout_ms,prepare_timeout_ms."
                        )
                    election_timeout_ms = int(row[1].strip())
                    prepare_timeout_ms = int(row[2].strip())
                    option_index = _two_pc_pair_to_index(
                        election_timeout_ms, prepare_timeout_ms
                    )
                else:
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
                schedule.append((time_s, option_index))
        if not schedule:
            raise ValueError(f"No schedule entries found in {path}.")
        return cls(protocol, schedule, start_monotonic=start_monotonic)

    def set_start_monotonic(self, start_monotonic: float) -> None:
        self._start_time = float(start_monotonic)
        self._active_row = -1

    def _current_option_index(self) -> int:
        elapsed_s = time.monotonic() - self._start_time
        active_row = 0
        option_index = self._schedule[0][1]
        for row_idx, (switch_time_s, schedule_option_index) in enumerate(self._schedule):
            if elapsed_s >= switch_time_s:
                active_row = row_idx
                option_index = schedule_option_index
            else:
                break
        if active_row != self._active_row:
            switch_time_s, active_option_index = self._schedule[active_row]
            print(
                "[scheduled-model] active schedule: "
                f"time_s>={switch_time_s:g}, "
                f"{describe_action(self._protocol, active_option_index)}"
            )
            self._active_row = active_row
        return option_index

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


class DQNTorchModel(TensorModel):
    """DQN classifier estimating Q-values for discrete options."""

    def __init__(self, action_dim: int, n_features: int, lr: float = 0.001):
        super().__init__()
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")
        
        self._action_dim = int(action_dim)
        
        self.q_network = torch.nn.Linear(n_features, self._action_dim)
        torch.nn.init.zeros_(self.q_network.weight)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

    def predict_action(self, features: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
        if features.dim() == 1:
            features = features.unsqueeze(0)

        if torch.rand(1).item() < epsilon:
            batch_size = features.shape[0]
            return torch.randint(0, self._action_dim, (batch_size,))

        with torch.no_grad():
            q_values = self.q_network(features)
            return q_values

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

            batch_size = features.shape[0]
            actions_tensor = torch.as_tensor(actions, dtype=torch.float32).view(-1)
            rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32).view(-1)

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

            target_index = torch.round(actions_tensor).long()
            valid_actions = torch.clamp(target_index, 0, self._action_dim - 1).view(-1, 1)

            q_values = self.q_network(features).gather(1, valid_actions)
            loss = F.smooth_l1_loss(q_values, rewards_tensor.view(-1, 1).detach())

            if backward:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            return float(loss.item())
    
    def reset_model(self) -> None:
        self.q_network.reset_parameters()

    def save(self) -> torch.Tensor:
        return self.q_network.weight.clone().detach()

    def load(self, weights: torch.Tensor) -> None:
        with torch.no_grad():
            self.q_network.weight.copy_(weights)