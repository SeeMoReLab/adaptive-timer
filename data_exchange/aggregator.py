from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np
import agent_pb2

import logging


class BaseAggregate(ABC):    
    def aggregate(self, node_id:int, states: Dict[int, agent_pb2.State]) -> agent_pb2.State:
        data = {k: np.array([
                    float(s.processed_transactions),
                    s.avg_message_delay,
                    s.max_message_delay,
                    s.min_message_delay,
                    s.std_message_delay
                ]) for k, s in states.items()
        }

        aggregated_values = self._calculate(data, node_id=node_id)

        return agent_pb2.State(
            processed_transactions=int(aggregated_values[0]),
            avg_message_delay=aggregated_values[1].item(),
            max_message_delay=aggregated_values[2].item(),
            min_message_delay=aggregated_values[3].item(),
            std_message_delay=aggregated_values[4].item()
        )

    @abstractmethod
    def _calculate(self, data: Dict[int, np.array], **kwargs) -> np.ndarray:
        pass

class MedianAggregate(BaseAggregate):
    def _calculate(self, data: Dict[int, np.array], **kwargs) -> np.ndarray:

        return np.median(np.column_stack(list(data.values())), axis=1)

class WeightedMedianAggregate(BaseAggregate):
    def __init__(self, weights: int):
        self.weights = weights

    def _calculate(self, data: Dict[int, agent_pb2.State], node_id: int) -> np.ndarray:
        if node_id not in data.keys():
            raise ValueError(f"[Node:] {node_id} not in states")
        
        others = [arr for k, arr in data.items() if k != node_id]
        weighted_node = ([data[node_id]] * self.weights) + others
    
        return np.median(np.column_stack(weighted_node), axis=1)
    
class Aggregator:
    def __init__(self, function: str = "median", **kwargs) -> None:
        self.function = function
        self.weight = kwargs.get('weight', 1)
        self.__refresh_strategy()

    def set_function(self, function: str):
        self.function = function
        self.__refresh_strategy()
    
    def set_weight(self, weight: int):
        self.weight = weight
        self.__refresh_strategy()

    def consensus_state(self, node_id:int, states: Dict[int, agent_pb2.State]) -> agent_pb2.State:
        return self.aggregator.aggregate(node_id, states)

    def __refresh_strategy(self):
        if self.function == "median":
            self.aggregator = MedianAggregate()
        elif self.function == "weighted_median":
            self.aggregator = WeightedMedianAggregate(weights=self.weight)
        else:
            raise ValueError(f"Unknown aggregator function: {self.function}")