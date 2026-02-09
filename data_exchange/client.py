import grpc
from google.protobuf import empty_pb2
import argparse
from concurrent import futures
import threading
import numpy as np
import logging

import agent_pb2
import agent_pb2_grpc

class LearningAgentClient:
    """Client for interacting with the remote LearningAgent service."""

    def __init__(self, id:str = 0, address: str = "0.0.0.0:50050"):
        self.channel = grpc.insecure_channel(address)
        self.stub = agent_pb2_grpc.LearningAgentStub(self.channel)
        self._metadata = (('replica-id', id),)

    def predict(self, state: agent_pb2.State) -> agent_pb2.Prediction:
        """Request a prediction (action) for the given state."""
        return self.stub.Predict(state, metadata=self._metadata)

    def learn(self, prediction_id: str, reward: float) -> None:
        """Send feedback (reward) for a previous prediction."""
        feedback = agent_pb2.Feedback(prediction_id=prediction_id, reward=reward)
        self.stub.Learn(feedback)

    def reset(self) -> None:
        """Reset the remote learning agent."""
        self.stub.Reset(empty_pb2.Empty())

    def close(self) -> None:
        """Close the gRPC channel."""
        self.channel.close()

def run_client(client_id: int, port: int, is_malicious: bool, id:str)->None:
    client = LearningAgentClient(address=f"0.0.0.0:{port}", id=id)
    avg_delay = np.random.uniform(0.01, 0.2)

    if is_malicious:
        avg_delay *= 1000

    state = agent_pb2.State(
        processed_transactions=10000,
        avg_message_delay=avg_delay,
        max_message_delay=avg_delay + np.random.uniform(0.01, 0.05),
        min_message_delay=avg_delay - np.random.uniform(0.01, 0.05),
        std_message_delay=np.random.uniform(0.001, 0.02),
    )
    logging.info(f"Client {client_id} requesting prediction...")

    predict = client.predict(state)
    logging.info(f"Client {client_id} received prediction id: {predict.prediction_id} action: {predict.action}")

    client.reset()
    client.close()

if __name__ == "__main__":
    # Experiment example
    np.random.seed(42)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')

    parser = argparse.ArgumentParser(description="Aggregation Clients Experiment")
    parser.add_argument(
        "--port",
        "-u",
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
    args = parser.parse_args()

    roles = [True if idx < args.malicious_nodes else False for idx in range(3*args.malicious_nodes+1)]
    np.random.shuffle(roles)

    logging.info(f"Initiating {len(roles)} clients")
    threads = []
    for i in range(len(roles)):
        t = threading.Thread(target=run_client, args=(i, args.port, roles[i], str(i)))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()

