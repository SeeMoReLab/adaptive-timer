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

REPORT_FEATURE_FIELD_NAMES: Tuple[str, ...] = (
    "processed_transactions",
    "avg_message_delay",
    "max_message_delay",
    "min_message_delay",
    "std_message_delay",
    "p95_message_delay",
    "p99_message_delay",
    "throughput_tps",
    "timeout_violation_rate",
    "no_op_rate",
    "avg_batch_size",
    "p95_batch_size",
    "leader_change_count",
    "regency_change_count",
    "avg_propose_delay",
    "avg_write_delay",
    "avg_accept_delay",
    "avg_post_decision_delay",
)
REPORT_FEATURE_DIM = len(REPORT_FEATURE_FIELD_NAMES)


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


def report_to_feature_values(state: agent_pb2.Report) -> List[float]:
    return [
        float(state.processed_transactions),
        float(state.avg_message_delay),
        float(state.max_message_delay),
        float(state.min_message_delay),
        float(state.std_message_delay),
        float(state.p95_message_delay),
        float(state.p99_message_delay),
        float(state.throughput_tps),
        float(state.timeout_violation_rate),
        float(state.no_op_rate),
        float(state.avg_batch_size),
        float(state.p95_batch_size),
        float(state.leader_change_count),
        float(state.regency_change_count),
        float(state.avg_propose_delay),
        float(state.avg_write_delay),
        float(state.avg_accept_delay),
        float(state.avg_post_decision_delay),
    ]


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
    ) -> None:
        pass

    @abstractmethod
    def reset_model(self) -> None:
        pass


class ReportFeatureExtractor(FeatureExtractor[agent_pb2.Report]):
    def featurize(self, state: agent_pb2.Report) -> torch.Tensor:
        features = report_to_feature_values(state)
        return torch.tensor(features, dtype=torch.float32)


class TimeoutActionDecoder(ActionDecoder[agent_pb2.Timeout]):
    """Decodes model output into one of the fixed timeout options."""

    def decode(self, action: torch.Tensor) -> agent_pb2.Timeout:
        if action.numel() == 1:
            option_index = int(round(float(action.item())))
        else:
            option_index = int(torch.argmax(action).item())
        timeout_ms = timeout_index_to_ms(option_index)
        return agent_pb2.Timeout(timeout_milliseconds=timeout_ms)


class LinearTorchModel(TensorModel):
    """Linear classifier over the discrete timeout options."""

    def __init__(self, n_features: int, lr: float = 0.01):
        self.weights = torch.zeros(
            (len(TIMEOUT_VALUES_MS), n_features), requires_grad=True
        )
        self.optimizer = torch.optim.SGD([self.weights], lr=lr)

    def predict_action(self, features: torch.Tensor) -> torch.Tensor:
        return torch.mv(self.weights, features)

    def update_model(
        self, features: torch.Tensor, action: torch.Tensor, reward: float
    ) -> None:
        target_index = int(round(float(action.item())))
        target_index = max(0, min(target_index, len(TIMEOUT_VALUES_MS) - 1))
        logits = torch.mv(self.weights, features).unsqueeze(0)
        target = torch.tensor([target_index], dtype=torch.long)
        # Policy-gradient style weighting using throughput-shaped reward.
        reward_scale = max(float(reward), 0.0) / 1000.0
        if reward_scale <= 0.0:
            return
        loss = F.cross_entropy(logits, target) * reward_scale
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def reset_model(self) -> None:
        torch.nn.init.zeros_(self.weights)


class ScheduledTimeoutModel(TensorModel):
    """Returns timeout option indices from a time-based schedule."""

    def __init__(
        self, schedule: List[Tuple[float, int]], start_monotonic: Optional[float] = None
    ):
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
        cls, path: Path, start_monotonic: Optional[float] = None
    ) -> "ScheduledTimeoutModel":
        schedule: List[Tuple[float, int]] = []
        with path.open("r", newline="") as handle:
            reader = csv.reader(handle)
            for lineno, row in enumerate(reader, start=1):
                if not row:
                    continue
                first = row[0].strip()
                if not first or first.startswith("#"):
                    continue
                if len(row) < 2:
                    raise ValueError(
                        f"Invalid schedule at line {lineno}: expected 2 columns time_s,timeout_ms."
                    )
                try:
                    time_s = float(first)
                    timeout_ms = int(row[1].strip())
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid schedule values at line {lineno}: {row}"
                    ) from exc
                if time_s < 0:
                    raise ValueError(
                        f"Invalid schedule at line {lineno}: time_s must be >= 0."
                    )
                option_index = timeout_ms_to_index(timeout_ms)
                schedule.append((time_s, option_index))
        if not schedule:
            raise ValueError(f"No schedule entries found in {path}.")
        return cls(schedule, start_monotonic=start_monotonic)

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
            timeout_ms = timeout_index_to_ms(active_option_index)
            print(
                "[scheduled-model] active schedule: "
                f"time_s>={switch_time_s:g}, timeout_ms={timeout_ms}"
            )
            self._active_row = active_row
        return option_index

    def predict_action(self, features: torch.Tensor) -> torch.Tensor:
        del features
        option_index = self._current_option_index()
        return torch.tensor(float(option_index), dtype=torch.float32)

    def update_model(
        self, features: torch.Tensor, action: torch.Tensor, reward: float
    ) -> None:
        del features, action, reward

    def reset_model(self) -> None:
        self._start_time = time.monotonic()
        self._active_row = -1
