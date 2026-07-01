from abc import ABC, abstractmethod
import csv
from enum import IntEnum
import time
from pathlib import Path
from typing import Dict, Generic, List, Optional, Tuple, TypeVar, Any
from sklearn.ensemble import RandomForestRegressor
try:
    from xgboost import XGBRegressor
except ImportError:  # xgboost is optional; only required when regressor_type="xgb"
    XGBRegressor = None
import torch
import torch.nn.functional as F
import numpy as np

import agent_pb2

S = TypeVar("S")  # generic state type
A = TypeVar("A")  # generic action type


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

class PhaseMapping(IntEnum):
    PHASE_0 = TimeoutOption.TIMEOUT_150_MS
    PHASE_1 = TimeoutOption.TIMEOUT_150_MS
    PHASE_2 = TimeoutOption.TIMEOUT_150_MS
    PHASE_3 = TimeoutOption.TIMEOUT_400_MS
    PHASE_4 = TimeoutOption.TIMEOUT_400_MS
    PHASE_5 = TimeoutOption.TIMEOUT_900_MS

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
TENDERMINT_REPORT_FEATURE_FIELD_NAMES: Tuple[str, ...] = (
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
        agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT,
        agent_pb2.PROTOCOL_SBFT,
        agent_pb2.PROTOCOL_TENDERMINT,
    ):
        raise ValueError(f"Unsupported protocol: {_protocol_name(protocol)}")
    return int(protocol)


def get_report_feature_field_names(protocol: int) -> Tuple[str, ...]:
    protocol = _validate_protocol(protocol)
    if protocol == agent_pb2.PROTOCOL_PBFT:
        return PBFT_REPORT_FEATURE_FIELD_NAMES
    if protocol == agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT:
        return TWO_PC_OVER_PBFT_REPORT_FEATURE_FIELD_NAMES
    if protocol == agent_pb2.PROTOCOL_SBFT:
        return SBFT_REPORT_FEATURE_FIELD_NAMES
    return TENDERMINT_REPORT_FEATURE_FIELD_NAMES


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


def _tendermint_triple_to_index(
    propose_timeout_ms: int, prevote_timeout_ms: int, precommit_timeout_ms: int
) -> int:
    propose_idx = timeout_ms_to_index(propose_timeout_ms)
    prevote_idx = timeout_ms_to_index(prevote_timeout_ms)
    precommit_idx = timeout_ms_to_index(precommit_timeout_ms)
    base = len(TIMEOUT_VALUES_MS)
    return (propose_idx * base * base) + (prevote_idx * base) + precommit_idx


def _tendermint_index_to_triple(index: int) -> Tuple[int, int, int]:
    base = len(TIMEOUT_VALUES_MS)
    total = base * base * base
    bounded = max(0, min(index, total - 1))
    propose_idx = bounded // (base * base)
    rem = bounded % (base * base)
    prevote_idx = rem // base
    precommit_idx = rem % base
    return (
        timeout_index_to_ms(propose_idx),
        timeout_index_to_ms(prevote_idx),
        timeout_index_to_ms(precommit_idx),
    )


def get_action_space_size(protocol: int) -> int:
    protocol = _validate_protocol(protocol)
    if protocol == agent_pb2.PROTOCOL_PBFT:
        return len(TIMEOUT_VALUES_MS)
    if protocol == agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT:
        return len(TIMEOUT_VALUES_MS) * len(TIMEOUT_VALUES_MS)
    if protocol in (agent_pb2.PROTOCOL_SBFT, agent_pb2.PROTOCOL_TENDERMINT):
        return len(TIMEOUT_VALUES_MS) * len(TIMEOUT_VALUES_MS) * len(TIMEOUT_VALUES_MS)
    raise ValueError(f"Unsupported protocol: {_protocol_name(protocol)}")


def get_num_timers(protocol: int) -> int:
    """Number of independently-tuned timers (timeout dimensions) per protocol.

    This is the per-timer factor count behind the joint action space and the
    number of heads used by the factored CMAB (MultiHeadCMABModel).
    """
    protocol = _validate_protocol(protocol)
    if protocol == agent_pb2.PROTOCOL_PBFT:
        return 1
    if protocol == agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT:
        return 2
    if protocol in (agent_pb2.PROTOCOL_SBFT, agent_pb2.PROTOCOL_TENDERMINT):
        return 3
    raise ValueError(f"Unsupported protocol: {_protocol_name(protocol)}")


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
    if protocol == agent_pb2.PROTOCOL_SBFT:
        timeout_msg = timeout_used
        return _sbft_triple_to_index(
            int(timeout_msg.election_timeout_milliseconds),
            int(timeout_msg.slow_path_timeout_milliseconds),
            int(timeout_msg.batch_timeout_milliseconds),
        )
    if protocol == agent_pb2.PROTOCOL_TENDERMINT:
        timeout_msg = timeout_used
        return _tendermint_triple_to_index(
            int(timeout_msg.propose_timeout_milliseconds),
            int(timeout_msg.prevote_timeout_milliseconds),
            int(timeout_msg.precommit_timeout_milliseconds),
        )
    raise ValueError(f"Unsupported protocol: {_protocol_name(protocol)}")


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
        return f"election_timeout_ms={timeout_index_to_ms(index)}"
    if protocol == agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT:
        election_timeout_ms, prepare_timeout_ms = _two_pc_index_to_pair(index)
        return (
            f"election_timeout_ms={election_timeout_ms}, "
            f"prepare_timeout_ms={prepare_timeout_ms}"
        )
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
                    if len(row) < 4:
                        raise ValueError(
                            "Invalid Tendermint schedule at line "
                            f"{lineno}: expected 4 columns "
                            "time_s,propose_timeout_ms,prevote_timeout_ms,precommit_timeout_ms."
                        )
                    propose_timeout_ms = int(row[1].strip())
                    prevote_timeout_ms = int(row[2].strip())
                    precommit_timeout_ms = int(row[3].strip())
                    option_index = _tendermint_triple_to_index(
                        propose_timeout_ms,
                        prevote_timeout_ms,
                        precommit_timeout_ms,
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

def _normalize_regressor_type(regressor_type: str) -> str:
    """Normalize the base-regressor selector for the CMAB ensemble."""
    rt = str(regressor_type).strip().lower()
    if rt in ("xgb", "xgboost"):
        return "xgb"
    if rt in ("decision_tree", "decisiontree", "tree", "dt"):
        return "decision_tree"
    raise ValueError(
        f"Unsupported regressor_type: {regressor_type!r}. "
        "Use 'xgb' or 'decision_tree'."
    )


class CMABModel(TensorModel):
    """Contextual multi-armed bandit via bootstrap Thompson sampling (BFTBrain-style).

    A single bagged forest over [state, action] estimates reward. Each update
    refits the forest on a fresh bootstrap resample of *all* accumulated data --
    that bootstrap-refit is the Thompson posterior sample (Eckles & Kaptein 2014;
    BFTBrain, NSDI'25). Prediction reads out the forest *mean* (low variance) and
    takes the argmax action, forcing a one-shot explore of any never-tried action.
    Because the replay buffer is unbounded, successive bootstraps concentrate as
    data grows, so exploration anneals automatically.
    """

    def __init__(
        self,
        action_dim: int,
        n_features: int,
        n_estimators: int = 100,
        regressor_type: str = "dt",
    ):
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")
        if n_estimators <= 0:
            raise ValueError(f"n_estimators must be > 0, got {n_estimators}")

        self._act_dim = int(action_dim)
        self._obs_dim = int(n_features)
        self._n_estimators = int(n_estimators)
        self._regressor_type = _normalize_regressor_type(regressor_type)

        self.reset_model()

    def _make_model(self):
        """Build the reward regressor whose *mean* is the readout (sklearn.ensemble)."""
        if self._regressor_type == "xgb":
            if XGBRegressor is None:
                raise ImportError(
                    "regressor_type='xgb' requires the xgboost package "
                    "(pip install xgboost), or use regressor_type='decision_tree'."
                )
            return XGBRegressor(
                n_estimators=self._n_estimators,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=0,
                n_jobs=1,
                verbosity=0,
            )
        # Bagged trees: RandomForest.predict averages its trees -> posterior-mean readout.
        return RandomForestRegressor(
            n_estimators=self._n_estimators,
            max_depth=5,
            n_jobs=1,
            random_state=0,
        )

    def _mean_predict(self, state_action_input: np.ndarray) -> np.ndarray:
        """Reward estimate per row = MEAN across the n_estimators (the readout)."""
        if hasattr(self.model, "estimators_"):
            # RandomForest: explicitly average the n_estimators trees' predictions.
            return np.mean(
                [tree.predict(state_action_input) for tree in self.model.estimators_],
                axis=0,
            )
        # XGBoost / other ensembles: predict() already returns the ensemble mean.
        return self.model.predict(state_action_input)

    def predict_action(self, features: torch.Tensor) -> torch.Tensor:
        """Average the n_estimators, then select the highest-value action.

        Exploration is *not* here -- it comes from the bootstrap-refit in
        ``update_model``. Returns a peaked vector so the decoder's argmax (and the
        per-head argmax in MultiHeadCMABModel) recovers the chosen action.
        """
        features_np = features.cpu().numpy()
        if features_np.ndim == 1:
            features_np = features_np.reshape(1, -1)

        batch_size = features_np.shape[0]
        q_values_np = np.zeros((batch_size, self._act_dim))
        actions_col = np.arange(self._act_dim).reshape(-1, 1)

        for i in range(batch_size):
            unvisited = np.flatnonzero(self._action_counts == 0)
            if unvisited.size > 0:
                choice = int(np.random.choice(unvisited))       # explore any never-tried action
            elif not self.is_fitted:
                choice = int(np.random.randint(self._act_dim))
            else:
                state_rows = np.repeat(features_np[i : i + 1], self._act_dim, axis=0)
                state_action_input = np.hstack((state_rows, actions_col))
                preds = self._mean_predict(state_action_input)  # mean across n_estimators, per action
                tied = np.flatnonzero(np.isclose(preds, np.max(preds)))
                choice = int(np.random.choice(tied))            # select highest (argmax) + random tie-break
            q_values_np[i, choice] = 1.0

        return torch.tensor(q_values_np, device=features.device, dtype=torch.float32)

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
            self.memory["actions"].append(int(actions_np[i]))
            self.memory["rewards"].append(float(rewards_np[i]))
            if 0 <= int(actions_np[i]) < self._act_dim:
                self._action_counts[int(actions_np[i])] += 1

        if backward and len(self.memory["rewards"]) > 0:
            X_features = np.array(self.memory["features"])
            X_actions = np.array(self.memory["actions"]).reshape(-1, 1)
            # X = [state, action]
            X = np.hstack((X_features, X_actions))
            y = np.array(self.memory["rewards"])

            # Bootstrap resample of ALL accumulated data = one Thompson posterior
            # sample; refit the forest on it. Successive bootstraps concentrate as
            # N grows, so the (re-sampled) mean prediction stops moving -> annealing.
            idx = np.random.choice(len(X), size=len(X), replace=True)
            self.model.fit(X[idx], y[idx])
            self.is_fitted = True

        return 0.0

    def reset_model(self) -> None:
        self.model = self._make_model()
        self.is_fitted = False
        # Unbounded replay buffer: the bootstrap can only concentrate (anneal) if N grows.
        self.memory = {"features": [], "actions": [], "rewards": []}
        self._action_counts = np.zeros(self._act_dim, dtype=np.int64)

class MultiHeadCMABModel(TensorModel):
    """Per-timer factored CMAB (Thompson sampling) for protocols with multiple timers."""

    def __init__(
        self,
        action_dim: int,
        n_features: int,
        n_heads: int,
        n_estimators: int = 100,
        regressor_type: str = "dt",
    ):
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be > 0, got {n_heads}")
        if n_estimators <= 0:
            raise ValueError(f"n_estimators must be > 0, got {n_estimators}")

        # action_dim is the *joint* size (k**m); recover the per-timer base k.
        n_options = int(round(action_dim ** (1.0 / n_heads)))
        if n_options ** n_heads != action_dim:
            raise ValueError(
                f"action_dim ({action_dim}) must equal n_options**n_heads; "
                f"got n_heads={n_heads}, inferred n_options={n_options}"
            )

        self._act_dim = int(action_dim)
        self._obs_dim = int(n_features)
        self._n_heads = int(n_heads)
        self._n_options = n_options
        self._n_estimators = int(n_estimators)
        self._regressor_type = _normalize_regressor_type(regressor_type)

        self._weights = [
            self._n_options ** (self._n_heads - 1 - k) for k in range(self._n_heads)
        ]

        self.reset_model()

    def predict_action(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = 1 if features.dim() == 1 else features.shape[0]

        # Each head draws its own fresh Thompson sample over its timer's options.
        joint = torch.zeros(batch_size, dtype=torch.long, device=features.device)
        for k, head in enumerate(self.heads):
            head_q = head.predict_action(features)          # (batch, n_options)
            choice = torch.argmax(head_q, dim=1)            # (batch,)
            joint = joint + choice.to(joint.device) * self._weights[k]

        # Emit a peaked vector over the joint space so the existing decoder's
        # argmax recovers the composed joint index unchanged.
        q_values = torch.zeros(batch_size, self._act_dim, device=features.device)
        q_values[torch.arange(batch_size, device=features.device), joint] = 1.0
        return q_values

    def update_model(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor | float,
        backward: bool = True,
    ) -> float:
        if isinstance(actions, torch.Tensor):
            actions_np = actions.cpu().numpy().astype(int).flatten()
        else:
            actions_np = np.array([int(actions)])

        # Factor the joint action into per-timer choices and route the *same*
        # state and global reward to each head with its own timer's value.
        for k, head in enumerate(self.heads):
            head_actions = (actions_np // self._weights[k]) % self._n_options
            head.update_model(
                features,
                torch.tensor(head_actions.astype(np.float32)),
                rewards,
                backward=backward,
            )

        return 0.0

    def reset_model(self) -> None:
        self.heads = [
            CMABModel(
                action_dim=self._n_options,
                n_features=self._obs_dim,
                n_estimators=self._n_estimators,
                regressor_type=self._regressor_type,
            )
            for _ in range(self._n_heads)
        ]