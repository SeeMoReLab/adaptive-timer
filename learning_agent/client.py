import grpc
from google.protobuf import empty_pb2

import agent_pb2
import agent_pb2_grpc

class LearningAgentClient:
    """Client for interacting with the remote LearningAgent service."""

    def __init__(self, address: str = "0.0.0.0:50051"):
        self.channel = grpc.insecure_channel(address)
        self.stub = agent_pb2_grpc.LearningAgentStub(self.channel)

    def predict(self, state: agent_pb2.State) -> agent_pb2.Prediction:
        """Request a prediction (action) for the given state."""
        return self.stub.Predict(state)

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


if __name__ == "__main__":
    client = LearningAgentClient()

    # Example usage
    state = agent_pb2.State(
        processed_transactions=10000,
        avg_message_delay=0.05,
        max_message_delay=0.1,
        min_message_delay=0.02,
        std_message_delay=0.015,
    )

    print("Requesting prediction...")
    prediction = client.predict(state)
    print(f"Prediction ID: {prediction.prediction_id}")
    print(f"Suggested timeout: {prediction.action.timeout_milliseconds} ms")

    # Send feedback (dummy example)
    client.learn(prediction.prediction_id, reward=0.8)

    # Optionally reset the agent
    client.reset()

    client.close()
