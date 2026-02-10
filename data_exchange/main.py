import argparse
from concurrent import futures
import threading
import time
import uuid
from pathlib import Path
from typing import List, Tuple, Dict
import logging


import grpc
from google.protobuf import empty_pb2

import agent_pb2
import agent_pb2_grpc
from aggregator import MedianAggregator, WeightedMedianAggregator
import random


class LearningAgentServicer(agent_pb2_grpc.LearningAgentServicer):
    """Implementation of the LearningAgent gRPC service."""

    def __init__(self, agent, aggregator, quorum_size:int):
        """
        Args:
            agent: an instance of OnlineAgent.
        """
        self.agent = agent
        self.aggregator = aggregator 
        self._quorum_size = quorum_size
        self._condition = threading.Condition()
        self._states: Dict[int, agent_pb2.State] = {}
        self._consensus_states: agent_pb2.State = None


    def Predict(self, request: agent_pb2.State, context) -> agent_pb2.Prediction:
        """
        Triggered after certain number of transactions or sequence number.
        State here represents only the local state.
        """

        # TODO: run consensus on state here
        pid = str(uuid.uuid4())
        metadata = dict(context.invocation_metadata())
        sender_id = int(metadata.get('replica-id'))
        with self._condition:
            logging.info(f"Received state for from Replica {sender_id}: {request}")
            self.__consensus_state(sender_id, request)
            
       
        logging.info(f"Replica {sender_id} Aggregated state: {self._consensus_states}\n ")
        # pred_id, action = self.agent.predict(consensus_state)

        return agent_pb2.Prediction(
            prediction_id = pid,
            action = agent_pb2.Action(timeout_milliseconds=15),
        )

    def Learn(self, request: agent_pb2.Feedback, context) -> empty_pb2.Empty:
        """
        Triggered after certain number of transactions or sequence number.
        Feedback/reward here represents only the local feedback/reward.
        """
        # TODO: run consensus on reward here
        print("Received feedback for learning:", request)

        self.agent.learn(request.prediction_id, request.reward)
        return empty_pb2.Empty()

    def Reset(self, request: empty_pb2.Empty, context) -> empty_pb2.Empty:
        self._states.clear()
        return empty_pb2.Empty()
    
    def __consensus_state(self, sender_id: int, request: agent_pb2.State) -> agent_pb2.State:
        if self._consensus_states is not None:
            return self._consensus_states
        
        self._states[sender_id] = request

        if len(self._states) >= self._quorum_size and self._consensus_states is None:
            logging.info(f"Replica {sender_id} reached quorum\n ")
            self._consensus_states = self.aggregator.aggregate(self._states)
            self._states.clear()
            self._condition.notify_all()

        while self._consensus_states is None:
            logging.info(f"Replica {sender_id} waiting")
            self._condition.wait()

        return self._consensus_states



def serve(host: str = "0.0.0.0", port: int = 50050, malicious_count: int = 3, function: str = "median"):
    """Start the gRPC server."""

    N = 3*malicious_count+1
    QUORUM = 2*malicious_count+1

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    if function == "median":
        agg = MedianAggregator()
    elif function == "weighted_median":
        agg = WeightedMedianAggregator(weights=[random.random() for _ in range(N)])

    agent_pb2_grpc.add_LearningAgentServicer_to_server(LearningAgentServicer(None, aggregator=agg, quorum_size=QUORUM), server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logging.info(f"LearningAgent server running at {host}:{port} using {function} aggregator")
    server.wait_for_termination()




def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
    random.seed(42)

    parser = argparse.ArgumentParser(description="LearningAgent server")
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=50050,
        help="Port to bind the server (default: 50050)",
    )

    parser.add_argument(
        "--malicious-nodes",
        "-m",
        type=int,
        default=1,
        help="Number of malicious nodes (default: 1)",
    )

    parser.add_argument(
        "--function",
        "-f",
        type=str,
        default="median",
        help="Selected function for Aggregate (default: median)",
    )

    args = parser.parse_args()
    serve(port=args.port, malicious_count=args.malicious_nodes, function = args.function)

if __name__ == "__main__":
    main()
