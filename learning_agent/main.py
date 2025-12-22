import argparse
import threading
import time
import uuid
from pathlib import Path
from typing import List, Tuple

import grpc
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
        print("Received state for prediction:", request)

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
        print("Received feedback for learning:", request)

        self.agent.learn(request.prediction_id, request.reward)
        return empty_pb2.Empty()

    def Reset(self, request: empty_pb2.Empty, context) -> empty_pb2.Empty:
        self.agent.reset()
        return empty_pb2.Empty()


class ScheduledLearningAgentServicer(agent_pb2_grpc.LearningAgentServicer):
    """
    LearningAgentServicer that returns a constant timeout for configurable periods
    defined in a schedule file of `time_s,timeout_ms` pairs.
    """

    def __init__(self, schedule_path: str):
        self.schedule: List[Tuple[int, int]] = self._load_schedule(schedule_path)
        if not self.schedule:
            raise ValueError(f"Timeout schedule file is empty: {schedule_path}")

        self._lock = threading.Lock()
        self._start_time = time.monotonic()
        self._current_index = 0

    def Predict(self, request: agent_pb2.State, context) -> agent_pb2.Prediction:
        timeout_ms = self._current_timeout_ms()
        prediction_id = str(uuid.uuid4())

        return agent_pb2.Prediction(
            prediction_id=prediction_id,
            action=agent_pb2.Action(timeout_milliseconds=timeout_ms),
        )

    def Learn(self, request: agent_pb2.Feedback, context) -> empty_pb2.Empty:
        print("Scheduled agent received feedback (ignored):", request)
        return empty_pb2.Empty()

    def Reset(self, request: empty_pb2.Empty, context) -> empty_pb2.Empty:
        with self._lock:
            self._start_time = time.monotonic()
            self._current_index = 0
        return empty_pb2.Empty()

    def _current_timeout_ms(self) -> int:
        with self._lock:
            elapsed_s = time.monotonic() - self._start_time
            while (
                self._current_index + 1 < len(self.schedule)
                and elapsed_s >= self.schedule[self._current_index + 1][0]
            ):
                self._current_index += 1

            return self.schedule[self._current_index][1]

    @staticmethod
    def _load_schedule(schedule_path: str) -> List[Tuple[int, int]]:
        schedule: List[Tuple[int, int]] = []

        with open(schedule_path, "r") as schedule_file:
            for lineno, raw_line in enumerate(schedule_file, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                if "," in line:
                    parts = [part.strip() for part in line.split(",")]
                else:
                    parts = line.split()

                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid schedule entry on line {lineno}: expected two columns"
                    )

                try:
                    time_s, timeout_ms = (int(parts[0]), int(parts[1]))
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid integers on line {lineno}: {line}"
                    ) from exc

                if time_s < 0 or timeout_ms <= 0:
                    raise ValueError(
                        f"Non-positive values on line {lineno}: time_s={time_s}, timeout_ms={timeout_ms}"
                    )

                schedule.append((time_s, timeout_ms))

        schedule.sort(key=lambda entry: entry[0])
        return schedule


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


def serve_with_schedule(schedule_path: str, host: str = "0.0.0.0", port: int = 50051):
    """Start the gRPC server using a schedule-driven LearningAgentServicer."""

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    agent_pb2_grpc.add_LearningAgentServicer_to_server(
        ScheduledLearningAgentServicer(schedule_path), server
    )
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    print(
        f"Scheduled LearningAgent server running at {host}:{port} using {schedule_path}"
    )
    server.wait_for_termination()


def main():
    parser = argparse.ArgumentParser(description="LearningAgent server")
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="Port to bind the server (default: 50051)",
    )
    parser.add_argument(
        "--schedule-file",
        type=str,
        help="Path to a schedule file with `time_s,timeout_ms` pairs. If present, uses the scheduled agent.",
    )
    args = parser.parse_args()

    if args.schedule_file and Path(args.schedule_file).is_file():
        serve_with_schedule(args.schedule_file, port=args.port)
    else:
        serve(port=args.port)


if __name__ == "__main__":
    main()
