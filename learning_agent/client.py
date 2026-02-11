import argparse
import random
from pathlib import Path
from typing import Dict

import grpc
from google.protobuf import empty_pb2

import agent_pb2
import agent_pb2_grpc


class LearningAgentClient:
    """Client for interacting with the LearningAgent service."""

    def __init__(self, address: str):
        self.channel = grpc.insecure_channel(address)
        self.stub = agent_pb2_grpc.LearningAgentStub(self.channel)

    def send_report(
        self,
        node_id: int,
        episode: int,
        processed_transactions: int,
        avg_message_delay: float,
        max_message_delay: float,
        min_message_delay: float,
        std_message_delay: float,
    ) -> None:
        report = agent_pb2.Report(
            processed_transactions=processed_transactions,
            avg_message_delay=avg_message_delay,
            max_message_delay=max_message_delay,
            min_message_delay=min_message_delay,
            std_message_delay=std_message_delay,
        )
        local_report = agent_pb2.ReportLocal(
            node_id=node_id,
            episode=episode,
            report=report,
        )
        self.stub.SendReport(local_report)

    def get_timeout(self, episode: int) -> agent_pb2.TimeoutStatus:
        return self.stub.GetTimeout(agent_pb2.TimeoutRequest(episode=episode))

    def share_report(self, report: agent_pb2.ReportLocal) -> None:
        self.stub.ShareReport(report)

    def reset(self) -> None:
        self.stub.Reset(empty_pb2.Empty())

    def close(self) -> None:
        self.channel.close()


def _load_hosts_config(path: Path) -> Dict[int, str]:
    entries: Dict[int, str] = {}
    with path.open("r") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(
                    f"Invalid hosts config at line {lineno}: expected at least 5 columns"
                )
            try:
                node_id = int(parts[0])
                host = parts[1]
                agent_port = int(parts[4])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid hosts config at line {lineno}: {line}"
                ) from exc
            entries[node_id] = f"{host}:{agent_port}"
    if not entries:
        raise ValueError(f"No hosts found in config: {path}")
    return entries


def _random_report(rng: random.Random) -> agent_pb2.Report:
    processed_transactions = rng.randint(500, 20000)
    min_delay = rng.uniform(0.001, 0.05)
    max_delay = rng.uniform(min_delay, min_delay + rng.uniform(0.01, 0.2))
    avg_delay = rng.uniform(min_delay, max_delay)
    std_delay = rng.uniform(0.0005, max(0.001, (max_delay - min_delay) / 2))
    return agent_pb2.Report(
        processed_transactions=processed_transactions,
        avg_message_delay=avg_delay,
        max_message_delay=max_delay,
        min_message_delay=min_delay,
        std_message_delay=std_delay,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LearningAgent client")
    parser.add_argument("--hosts-config", type=Path, default=Path("config/hosts.config"))
    parser.add_argument("--node-id", type=int, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--get",
        action="store_true",
        help="Only query timeout status for the episode; do not send a report.",
    )
    args = parser.parse_args()

    hosts = _load_hosts_config(args.hosts_config)
    address = hosts.get(args.node_id)
    if address is None:
        raise ValueError(f"Node id {args.node_id} not found in {args.hosts_config}")

    client = LearningAgentClient(address=address)
    if args.get:
        status = client.get_timeout(args.episode)
        if status.status == agent_pb2.TimeoutStatus.Status.READY and status.HasField("timeout"):
            print(
                "Timeout ready:",
                status.timeout.timeout_milliseconds,
                "ms",
            )
        else:
            print("Timeout status:", agent_pb2.TimeoutStatus.Status.Name(status.status))
    else:
        rng = random.Random(args.seed)
        report = _random_report(rng)
        client.send_report(
            node_id=args.node_id,
            episode=args.episode,
            processed_transactions=report.processed_transactions,
            avg_message_delay=report.avg_message_delay,
            max_message_delay=report.max_message_delay,
            min_message_delay=report.min_message_delay,
            std_message_delay=report.std_message_delay,
        )
        print("Report sent.")
    client.close()


if __name__ == "__main__":
    main()
