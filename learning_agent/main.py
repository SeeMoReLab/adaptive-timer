import grpc
from concurrent import futures
from google.protobuf import empty_pb2

import agent_pb2
import agent_pb2_grpc
from online import OnlineAgent
from converter import GRPCTorchConverter
from models import LinearTorchModel

class LearningAgentServicer(agent_pb2_grpc.LearningAgentServicer):
    """Implementation of the LearningAgent gRPC service."""

    def __init__(self, agent: OnlineAgent):
        """
        Args:
            agent: an instance of OnlineAgent.
        """
        self.agent = agent

    def Predict(self, request: agent_pb2.State, context) -> agent_pb2.Prediction:
        """
        Triggered after certain number of transactions or sequence number.
        State here represents only the local state.
        """

        # TODO: run consensus on state here

        pred_id, action = self.agent.predict(request)

        return agent_pb2.Prediction(
            prediction_id = pred_id,
            action = action,
        )

    def Learn(self, request: agent_pb2.Feedback, context) -> empty_pb2.Empty:
        """
        Triggered after certain number of transactions or sequence number.
        Feedback/reward here represents only the local feedback/reward.
        """

        # TODO: run consensus on reward here

        self.agent.learn(request.prediction_id, request.reward)
        return empty_pb2.Empty()

    def Reset(self, request: empty_pb2.Empty, context) -> empty_pb2.Empty:
        self.agent.reset()
        return empty_pb2.Empty()


def serve(host: str = "0.0.0.0", port: int = 50051):
    """Start the gRPC server."""

    # TODO: automate n_features input
    model = LinearTorchModel(n_features=5, lr=0.05)
    converter = GRPCTorchConverter()
    agent = OnlineAgent(converter=converter, model=model)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    agent_pb2_grpc.add_LearningAgentServicer_to_server(LearningAgentServicer(agent), server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    print(f"LearningAgent server running at {host}:{port}")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
