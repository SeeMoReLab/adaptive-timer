from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np
import agent_pb2

import logging

class BaseAggregator(ABC):
    """Handles the conversion between Protobufs and Numpy for aggregation."""
    
    def aggregate(self, states: Dict[str, agent_pb2.State]) -> agent_pb2.State:
        data = np.array([
            [
                float(s.processed_transactions),
                s.avg_message_delay,
                s.max_message_delay,
                s.min_message_delay,
                s.std_message_delay
            ] for k, s in states.items()
        ], dtype=np.float32)

        aggregated_values = self._calculate(data, indices = list(states.keys()))

        return agent_pb2.State(
            processed_transactions=int(aggregated_values[0]),
            avg_message_delay=aggregated_values[1].item(),
            max_message_delay=aggregated_values[2].item(),
            min_message_delay=aggregated_values[3].item(),
            std_message_delay=aggregated_values[4].item()
        )

    @abstractmethod
    def _calculate(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Core mathematical logic implemented by subclasses."""
        pass

class MedianAggregator(BaseAggregator):
    def _calculate(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return np.median(data, axis=0)

class WeightedMedianAggregator(BaseAggregator):
    def __init__(self, weights: List[float]):
        self.weights = np.array(weights, dtype=np.float32)

    def _calculate(self, data: np.ndarray, indices: List[int]) -> np.ndarray:
        curr_weight = self.weights[indices]
        if len(data) != len(curr_weight):
            raise ValueError(f"Data length ({len(data)}) must match weights ({len(self.weights)})")
            
        result = np.empty(data.shape[1], dtype=data.dtype)
        for i in range(data.shape[1]):
            column = data[:, i]
            sorter = np.argsort(column)
            
            cumsum = np.cumsum(curr_weight[sorter])
            cutoff = curr_weight.sum() / 2.0
            idx = np.searchsorted(cumsum, cutoff)
            result[i] = column[sorter][idx]
        return result