import argparse
import random
from pathlib import Path
from typing import Dict, Optional

import grpc
from google.protobuf import empty_pb2

import agent_pb2
import agent_pb2_grpc

TIMEOUT_OPTIONS_MS = (300, 600, 1800)


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
        reward: Optional[agent_pb2.Reward] = None,
        *,
        p95_message_delay: float = 0.0,
        p99_message_delay: float = 0.0,
        throughput_tps: float = 0.0,
        timeout_violation_rate: float = 0.0,
        no_op_rate: float = 0.0,
        avg_batch_size: float = 0.0,
        p95_batch_size: float = 0.0,
        leader_change_count: int = 0,
        regency_change_count: int = 0,
        avg_propose_delay: float = 0.0,
        avg_write_delay: float = 0.0,
        avg_accept_delay: float = 0.0,
        avg_post_decision_delay: float = 0.0,
    ) -> None:
        report = agent_pb2.Report(
            processed_transactions=processed_transactions,
            avg_message_delay=avg_message_delay,
            max_message_delay=max_message_delay,
            min_message_delay=min_message_delay,
            std_message_delay=std_message_delay,
            p95_message_delay=p95_message_delay,
            p99_message_delay=p99_message_delay,
            throughput_tps=throughput_tps,
            timeout_violation_rate=timeout_violation_rate,
            no_op_rate=no_op_rate,
            avg_batch_size=avg_batch_size,
            p95_batch_size=p95_batch_size,
            leader_change_count=leader_change_count,
            regency_change_count=regency_change_count,
            avg_propose_delay=avg_propose_delay,
            avg_write_delay=avg_write_delay,
            avg_accept_delay=avg_accept_delay,
            avg_post_decision_delay=avg_post_decision_delay,
        )
        local_report = agent_pb2.ReportLocal(
            node_id=node_id,
            episode=episode,
            state=report,
        )
        if reward is not None:
            local_report.reward.CopyFrom(reward)
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
    p95_delay = rng.uniform(avg_delay, max(avg_delay, max_delay * 1.2))
    p99_delay = rng.uniform(p95_delay, max(p95_delay, max_delay * 1.5))
    throughput_tps = processed_transactions / rng.uniform(0.5, 5.0)
    timeout_violation_rate = rng.uniform(0.0, 0.2)
    no_op_rate = rng.uniform(0.0, 0.1)
    avg_batch_size = rng.uniform(1.0, 200.0)
    p95_batch_size = rng.uniform(avg_batch_size, max(avg_batch_size, avg_batch_size * 1.5))
    leader_change_count = rng.randint(0, 8)
    regency_change_count = rng.randint(0, 8)
    avg_propose_delay = rng.uniform(0.0, avg_delay)
    avg_write_delay = rng.uniform(0.0, avg_delay)
    avg_accept_delay = rng.uniform(0.0, avg_delay)
    avg_post_decision_delay = rng.uniform(0.0, avg_delay)
    return agent_pb2.Report(
        processed_transactions=processed_transactions,
        avg_message_delay=avg_delay,
        max_message_delay=max_delay,
        min_message_delay=min_delay,
        std_message_delay=std_delay,
        p95_message_delay=p95_delay,
        p99_message_delay=p99_delay,
        throughput_tps=throughput_tps,
        timeout_violation_rate=timeout_violation_rate,
        no_op_rate=no_op_rate,
        avg_batch_size=avg_batch_size,
        p95_batch_size=p95_batch_size,
        leader_change_count=leader_change_count,
        regency_change_count=regency_change_count,
        avg_propose_delay=avg_propose_delay,
        avg_write_delay=avg_write_delay,
        avg_accept_delay=avg_accept_delay,
        avg_post_decision_delay=avg_post_decision_delay,
    )


def _print_sent_payload(
    node_id: int,
    episode: int,
    report: agent_pb2.Report,
    reward: Optional[agent_pb2.Reward],
) -> None:
    print(
        "Sent ReportLocal:",
        f"node_id={node_id}",
        f"episode={episode}",
        f"state.processed_transactions={report.processed_transactions}",
        f"state.avg_message_delay={report.avg_message_delay:.6f}",
        f"state.max_message_delay={report.max_message_delay:.6f}",
        f"state.min_message_delay={report.min_message_delay:.6f}",
        f"state.std_message_delay={report.std_message_delay:.6f}",
        f"state.throughput_tps={report.throughput_tps:.2f}",
        f"state.timeout_violation_rate={report.timeout_violation_rate:.4f}",
        f"state.no_op_rate={report.no_op_rate:.4f}",
    )
    if reward is None:
        return
    print(
        "Sent Reward:",
        f"reward.episode={reward.episode}",
        f"reward.timeout_milliseconds_used={reward.timeout_milliseconds_used}",
        f"reward.report.processed_transactions={reward.report.processed_transactions}",
        f"reward.report.avg_message_delay={reward.report.avg_message_delay:.6f}",
        f"reward.report.max_message_delay={reward.report.max_message_delay:.6f}",
        f"reward.report.min_message_delay={reward.report.min_message_delay:.6f}",
        f"reward.report.std_message_delay={reward.report.std_message_delay:.6f}",
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
    parser.add_argument(
        "--reward",
        action="store_true",
        help=(
            "Attach a randomized reward payload using a random timeout from "
            f"{TIMEOUT_OPTIONS_MS}."
        ),
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
        reward = None
        if args.reward:
            if args.episode <= 0:
                raise ValueError("--reward requires --episode >= 1")
            reward_report = _random_report(rng)
            reward = agent_pb2.Reward(
                episode=args.episode - 1,
                report=reward_report,
                timeout_milliseconds_used=rng.choice(TIMEOUT_OPTIONS_MS),
            )
        client.send_report(
            node_id=args.node_id,
            episode=args.episode,
            processed_transactions=report.processed_transactions,
            avg_message_delay=report.avg_message_delay,
            max_message_delay=report.max_message_delay,
            min_message_delay=report.min_message_delay,
            std_message_delay=report.std_message_delay,
            reward=reward,
            p95_message_delay=report.p95_message_delay,
            p99_message_delay=report.p99_message_delay,
            throughput_tps=report.throughput_tps,
            timeout_violation_rate=report.timeout_violation_rate,
            no_op_rate=report.no_op_rate,
            avg_batch_size=report.avg_batch_size,
            p95_batch_size=report.p95_batch_size,
            leader_change_count=report.leader_change_count,
            regency_change_count=report.regency_change_count,
            avg_propose_delay=report.avg_propose_delay,
            avg_write_delay=report.avg_write_delay,
            avg_accept_delay=report.avg_accept_delay,
            avg_post_decision_delay=report.avg_post_decision_delay,
        )
        _print_sent_payload(
            node_id=args.node_id,
            episode=args.episode,
            report=report,
            reward=reward,
        )
    client.close()


if __name__ == "__main__":
    main()
