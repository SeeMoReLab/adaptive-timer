from abc import ABC, abstractmethod
import csv
import time
from pathlib import Path
from typing import Generic, List, Tuple, TypeVar

import torch

import agent_pb2

S = TypeVar("S")  # generic state type
A = TypeVar("A")  # generic action type

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
        features = [
            float(state.processed_transactions),
            state.avg_message_delay,
            state.max_message_delay,
            state.min_message_delay,
            state.std_message_delay,
        ]
        return torch.tensor(features, dtype=torch.float32)


class TimeoutActionDecoder(ActionDecoder[agent_pb2.Timeout]):
    def __init__(self, min_timeout_ms: int = 0):
        self.min_timeout_ms = min_timeout_ms

    def decode(self, action: torch.Tensor) -> agent_pb2.Timeout:
        value = float(action.item())
        timeout_ms = int(round(value))
        if timeout_ms < self.min_timeout_ms:
            timeout_ms = self.min_timeout_ms
        return agent_pb2.Timeout(timeout_milliseconds=timeout_ms)


class LinearTorchModel(TensorModel):
    def __init__(self, n_features: int, lr: float = 0.01):
        self.weights = torch.zeros(n_features, requires_grad=True)
        self.optimizer = torch.optim.SGD([self.weights], lr=lr)

    def predict_action(self, features: torch.Tensor) -> torch.Tensor:
        return torch.dot(self.weights, features)

    def update_model(
        self, features: torch.Tensor, action: torch.Tensor, reward: float
    ) -> None:
        y = torch.tensor(reward, dtype=torch.float32)
        pred = torch.dot(self.weights, features)
        loss = (pred - y).pow(2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def reset_model(self) -> None:
        torch.nn.init.zeros_(self.weights)


class ScheduledTimeoutModel(TensorModel):
    """Returns timeout values from a time-based schedule."""

    def __init__(self, schedule: List[Tuple[float, int]]):
        if not schedule:
            raise ValueError("Schedule must contain at least one entry.")
        self._schedule = sorted(schedule, key=lambda entry: entry[0])
        if self._schedule[0][0] < 0:
            raise ValueError("Schedule times must be >= 0 seconds.")
        self._start_time = time.monotonic()

    @classmethod
    def from_csv(cls, path: Path) -> "ScheduledTimeoutModel":
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
                if timeout_ms < 0:
                    raise ValueError(
                        f"Invalid schedule at line {lineno}: timeout_ms must be >= 0."
                    )
                schedule.append((time_s, timeout_ms))
        if not schedule:
            raise ValueError(f"No schedule entries found in {path}.")
        return cls(schedule)

    def _current_timeout_ms(self) -> int:
        elapsed_s = time.monotonic() - self._start_time
        timeout_ms = self._schedule[0][1]
        for switch_time_s, value_ms in self._schedule:
            if elapsed_s >= switch_time_s:
                timeout_ms = value_ms
            else:
                break
        return timeout_ms

    def predict_action(self, features: torch.Tensor) -> torch.Tensor:
        del features
        return torch.tensor(float(self._current_timeout_ms()), dtype=torch.float32)

    def update_model(
        self, features: torch.Tensor, action: torch.Tensor, reward: float
    ) -> None:
        del features, action, reward

    def reset_model(self) -> None:
        self._start_time = time.monotonic()
