from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch

import agent_pb2

S = TypeVar("State")
A = TypeVar("Action")
IS = TypeVar("InternalState")
IA = TypeVar("InternalAction")

class BaseConverter(Generic[S, A, IS, IA], ABC):
    """Abstract converter between generic and internal types."""

    @abstractmethod
    def encode_state(self, state: S) -> IS:
        pass

    @abstractmethod
    def decode_action(self, action: IA) -> A:
        pass

class GRPCTorchConverter(BaseConverter[agent_pb2.State, agent_pb2.Action, torch.Tensor, torch.Tensor]):
    """Converts gRPC proto from/to torch.Tensor."""

    def encode_state(self, state: agent_pb2.State) -> torch.Tensor:
        """
        Convert a State protobuf message to a torch tensor.
        Order of fields must match the expected model input.
        """

        features = [
            float(state.processed_transactions),
            state.avg_message_delay,
            state.max_message_delay,
            state.min_message_delay,
            state.std_message_delay
        ]

        return torch.tensor(features, dtype=torch.float32)

    def decode_action(self, action: torch.Tensor) -> agent_pb2.Action:
        """
        Convert a State protobuf message to a torch tensor.
        Order of fields must match the expected model input.
        """

        return agent_pb2.Action(
            timeout_milliseconds=1000,
        )
