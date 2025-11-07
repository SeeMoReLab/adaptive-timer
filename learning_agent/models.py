from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch

S = TypeVar("S")  # generic state type
A = TypeVar("A")  # generic action type

class BaseModel(Generic[S, A], ABC):
    """Generic base model with state type S and action prediction type A."""

    @abstractmethod
    def predict_action(self, state: S) -> A:
        """Return prediction for a state."""
        pass

    @abstractmethod
    def update_model(self, state: S, action: A, reward: float) -> None:
        """Update the model based on state, action, and reward."""
        pass

    @abstractmethod
    def reset_model(self) -> None:
        """Reset the internal model."""
        pass


class LinearTorchModel(BaseModel[torch.Tensor, torch.Tensor]):
    def __init__(self, n_features: int, lr: float = 0.01):
        self.weights = torch.zeros(n_features, requires_grad=True)
        self.optimizer = torch.optim.SGD([self.weights], lr=lr)

    def predict_action(self, state: torch.Tensor) -> torch.Tensor:
        value = torch.dot(self.weights, state)
        action = int(value.item() > 0)
        return {"action": action, "value": value.item()}

    def update_model(self, state: torch.Tensor, action: torch.Tensor, reward: float) -> None:
        y = torch.tensor(reward, dtype=torch.float32)
        pred = torch.dot(self.weights, state)
        loss = (pred - y).pow(2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def reset_model(self) -> None:
        torch.nn.init.zeros_(self.weights)
