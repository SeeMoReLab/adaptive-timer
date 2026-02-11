from abc import ABC, abstractmethod
from typing import Generic, TypeVar

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
