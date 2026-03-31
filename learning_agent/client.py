import argparse
import random
from pathlib import Path
from typing import Dict, Optional

import grpc
from google.protobuf import empty_pb2

import agent_pb2
import agent_pb2_grpc

TIMEOUT_OPTIONS_MS = (150, 400, 900)
PROTOCOL_NAME_TO_ENUM = {
    "pbft": agent_pb2.PROTOCOL_PBFT,
    "two_pc_over_pbft": agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT,
    "sbft": agent_pb2.PROTOCOL_SBFT,
}


def _parse_protocol(value: str) -> int:
    normalized = value.strip().lower().replace("-", "_")
    if normalized == "2pc":
        normalized = "two_pc_over_pbft"
    protocol = PROTOCOL_NAME_TO_ENUM.get(normalized)
    if protocol is None:
        raise argparse.ArgumentTypeError(
            f"Unsupported protocol '{value}'. Valid: {', '.join(PROTOCOL_NAME_TO_ENUM.keys())}"
        )
    return protocol


class LearningAgentClient:
    """Client for interacting with the LearningAgent service."""

    def __init__(self, address: str):
        self.channel = grpc.insecure_channel(address)
        self.stub = agent_pb2_grpc.LearningAgentStub(self.channel)

    def send_report(
        self,
        node_id: int,
        episode: int,
        protocol: int,
        report: object,
        start_tick: int,
        report_seq: int,
        reward: Optional[agent_pb2.Reward] = None,
    ) -> None:
        local_report = agent_pb2.ReportLocal(
            node_id=node_id,
            episode=episode,
            protocol=protocol,
            start_tick=start_tick,
            report_seq=report_seq,
        )
        if protocol == agent_pb2.PROTOCOL_PBFT:
            local_report.pbft_state.CopyFrom(report)
        elif protocol == agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT:
            local_report.two_pc_over_pbft_state.CopyFrom(report)
        elif protocol == agent_pb2.PROTOCOL_SBFT:
            local_report.sbft_state.CopyFrom(report)
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")

        if reward is not None:
            local_report.reward.CopyFrom(reward)
        self.stub.SendReport(local_report)

    def get_timeout(self, episode: int, protocol: int) -> agent_pb2.TimeoutStatus:
        return self.stub.GetTimeout(
            agent_pb2.TimeoutRequest(episode=episode, protocol=protocol)
        )

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


def _random_pbft_report(rng: random.Random) -> agent_pb2.PbftReport:
    total_transactions = rng.randint(500, 20000)
    avg_batch_size = rng.uniform(1.0, 200.0)
    total_consensus_instances = max(
        1, int(round(total_transactions / max(avg_batch_size, 1.0)))
    )
    avg_consensus_latency_ms = rng.uniform(1.0, 200.0)
    p95_consensus_latency_ms = rng.uniform(
        avg_consensus_latency_ms, max(avg_consensus_latency_ms, avg_consensus_latency_ms * 2.0)
    )
    p99_consensus_latency_ms = rng.uniform(
        p95_consensus_latency_ms, max(p95_consensus_latency_ms, p95_consensus_latency_ms * 1.5)
    )
    throughput_tps = total_transactions / rng.uniform(0.5, 5.0)
    timeout_violation_rate = rng.uniform(0.0, 0.2)
    p95_batch_size = rng.uniform(avg_batch_size, max(avg_batch_size, avg_batch_size * 1.5))
    leader_change_count = rng.randint(0, 8)
    regency_change_count = rng.randint(0, 8)
    phase_propose_avg_delay_ms = rng.uniform(0.0, avg_consensus_latency_ms)
    phase_propose_p95_delay_ms = rng.uniform(
        phase_propose_avg_delay_ms, max(phase_propose_avg_delay_ms, phase_propose_avg_delay_ms * 2.0)
    )
    phase_write_avg_delay_ms = rng.uniform(0.0, avg_consensus_latency_ms)
    phase_write_p95_delay_ms = rng.uniform(
        phase_write_avg_delay_ms, max(phase_write_avg_delay_ms, phase_write_avg_delay_ms * 2.0)
    )
    phase_accept_avg_delay_ms = rng.uniform(0.0, avg_consensus_latency_ms)
    phase_accept_p95_delay_ms = rng.uniform(
        phase_accept_avg_delay_ms, max(phase_accept_avg_delay_ms, phase_accept_avg_delay_ms * 2.0)
    )
    phase_post_decision_avg_delay_ms = rng.uniform(0.0, avg_consensus_latency_ms)
    phase_post_decision_p95_delay_ms = rng.uniform(
        phase_post_decision_avg_delay_ms,
        max(phase_post_decision_avg_delay_ms, phase_post_decision_avg_delay_ms * 2.0),
    )
    return agent_pb2.PbftReport(
        total_transactions=total_transactions,
        total_consensus_instances=total_consensus_instances,
        avg_consensus_latency_ms=avg_consensus_latency_ms,
        p95_consensus_latency_ms=p95_consensus_latency_ms,
        p99_consensus_latency_ms=p99_consensus_latency_ms,
        throughput_tps=throughput_tps,
        timeout_violation_rate=timeout_violation_rate,
        avg_batch_size=avg_batch_size,
        p95_batch_size=p95_batch_size,
        leader_change_count=leader_change_count,
        regency_change_count=regency_change_count,
        phase_propose_avg_delay_ms=phase_propose_avg_delay_ms,
        phase_propose_p95_delay_ms=phase_propose_p95_delay_ms,
        phase_write_avg_delay_ms=phase_write_avg_delay_ms,
        phase_write_p95_delay_ms=phase_write_p95_delay_ms,
        phase_accept_avg_delay_ms=phase_accept_avg_delay_ms,
        phase_accept_p95_delay_ms=phase_accept_p95_delay_ms,
        phase_post_decision_avg_delay_ms=phase_post_decision_avg_delay_ms,
        phase_post_decision_p95_delay_ms=phase_post_decision_p95_delay_ms,
    )


def _random_two_pc_report(rng: random.Random) -> agent_pb2.TwoPcOverPbftReport:
    total_transactions = rng.randint(100, 12000)
    single_shard_transactions = rng.randint(0, total_transactions)
    cross_shard_transactions = total_transactions - single_shard_transactions

    prepare_timeout_count = rng.randint(0, max(1, total_transactions // 4))
    prepare_fail_count = rng.randint(prepare_timeout_count, max(prepare_timeout_count, total_transactions // 3))
    prepare_ok_count = max(0, total_transactions - prepare_fail_count)
    abort_count = rng.randint(0, prepare_fail_count)
    commit_ok_count = max(0, total_transactions - abort_count)
    commit_timeout_unknown_outcome_count = rng.randint(0, max(1, total_transactions // 10))
    abort_best_effort_timeout_count = rng.randint(0, max(1, abort_count))
    in_doubt_count = rng.randint(0, max(1, commit_timeout_unknown_outcome_count))
    lock_contention_count = rng.randint(0, total_transactions)

    denom = max(float(total_transactions), 1.0)
    prepare_timeout_rate = float(prepare_timeout_count / denom)
    abort_rate = float(abort_count / denom)
    commit_rate = float(commit_ok_count / denom)
    lock_contention_rate = float(lock_contention_count / denom)

    avg_end_to_end_latency_ms = rng.uniform(1.0, 300.0)
    p95_end_to_end_latency_ms = rng.uniform(
        avg_end_to_end_latency_ms,
        max(avg_end_to_end_latency_ms, avg_end_to_end_latency_ms * 2.0),
    )
    p99_end_to_end_latency_ms = rng.uniform(
        p95_end_to_end_latency_ms,
        max(p95_end_to_end_latency_ms, p95_end_to_end_latency_ms * 1.5),
    )
    avg_prepare_latency_ms = rng.uniform(0.1, avg_end_to_end_latency_ms)
    p95_prepare_latency_ms = rng.uniform(
        avg_prepare_latency_ms, max(avg_prepare_latency_ms, avg_prepare_latency_ms * 2.0)
    )
    avg_commit_latency_ms = rng.uniform(0.1, avg_end_to_end_latency_ms)
    p95_commit_latency_ms = rng.uniform(
        avg_commit_latency_ms, max(avg_commit_latency_ms, avg_commit_latency_ms * 2.0)
    )
    avg_lock_wait_ms = rng.uniform(0.0, avg_end_to_end_latency_ms)
    p95_lock_wait_ms = rng.uniform(avg_lock_wait_ms, max(avg_lock_wait_ms, avg_lock_wait_ms * 2.0))
    avg_lock_hold_ms = rng.uniform(0.0, avg_end_to_end_latency_ms)
    p95_lock_hold_ms = rng.uniform(avg_lock_hold_ms, max(avg_lock_hold_ms, avg_lock_hold_ms * 2.0))

    reason_prepare_lock_busy_count = rng.randint(0, prepare_fail_count)
    remaining_prepare_fail = max(0, prepare_fail_count - reason_prepare_lock_busy_count)
    reason_prepare_validation_fail_count = rng.randint(0, remaining_prepare_fail)
    remaining_prepare_fail -= reason_prepare_validation_fail_count
    reason_prepare_timeout_count = min(prepare_timeout_count, remaining_prepare_fail)
    remaining_prepare_fail -= reason_prepare_timeout_count
    reason_prepare_other_count = max(0, remaining_prepare_fail)

    reason_commit_timeout_unknown_outcome_count = commit_timeout_unknown_outcome_count
    reason_abort_best_effort_timeout_count = min(
        abort_best_effort_timeout_count, abort_count
    )
    reason_other_count = rng.randint(0, max(1, total_transactions // 10))

    return agent_pb2.TwoPcOverPbftReport(
        pbft=_random_pbft_report(rng),
        total_transactions=total_transactions,
        single_shard_transactions=single_shard_transactions,
        cross_shard_transactions=cross_shard_transactions,
        prepare_ok_count=prepare_ok_count,
        prepare_fail_count=prepare_fail_count,
        commit_ok_count=commit_ok_count,
        abort_count=abort_count,
        in_doubt_count=in_doubt_count,
        prepare_timeout_count=prepare_timeout_count,
        commit_timeout_unknown_outcome_count=commit_timeout_unknown_outcome_count,
        abort_best_effort_timeout_count=abort_best_effort_timeout_count,
        lock_contention_count=lock_contention_count,
        prepare_timeout_rate=prepare_timeout_rate,
        abort_rate=abort_rate,
        commit_rate=commit_rate,
        lock_contention_rate=lock_contention_rate,
        avg_end_to_end_latency_ms=avg_end_to_end_latency_ms,
        p95_end_to_end_latency_ms=p95_end_to_end_latency_ms,
        p99_end_to_end_latency_ms=p99_end_to_end_latency_ms,
        avg_prepare_latency_ms=avg_prepare_latency_ms,
        p95_prepare_latency_ms=p95_prepare_latency_ms,
        avg_commit_latency_ms=avg_commit_latency_ms,
        p95_commit_latency_ms=p95_commit_latency_ms,
        avg_lock_wait_ms=avg_lock_wait_ms,
        p95_lock_wait_ms=p95_lock_wait_ms,
        avg_lock_hold_ms=avg_lock_hold_ms,
        p95_lock_hold_ms=p95_lock_hold_ms,
        reason_prepare_lock_busy_count=reason_prepare_lock_busy_count,
        reason_prepare_validation_fail_count=reason_prepare_validation_fail_count,
        reason_prepare_timeout_count=reason_prepare_timeout_count,
        reason_prepare_other_count=reason_prepare_other_count,
        reason_commit_timeout_unknown_outcome_count=reason_commit_timeout_unknown_outcome_count,
        reason_abort_best_effort_timeout_count=reason_abort_best_effort_timeout_count,
        reason_other_count=reason_other_count,
    )


def _random_sbft_report(rng: random.Random) -> agent_pb2.SbftReport:
    total_transactions = rng.randint(500, 20000)
    avg_batch_size = rng.uniform(1.0, 200.0)
    total_consensus_instances = max(
        1, int(round(total_transactions / max(avg_batch_size, 1.0)))
    )
    avg_consensus_latency_ms = rng.uniform(1.0, 250.0)
    p95_consensus_latency_ms = rng.uniform(
        avg_consensus_latency_ms,
        max(avg_consensus_latency_ms, avg_consensus_latency_ms * 2.0),
    )
    p99_consensus_latency_ms = rng.uniform(
        p95_consensus_latency_ms,
        max(p95_consensus_latency_ms, p95_consensus_latency_ms * 1.5),
    )
    throughput_tps = total_transactions / rng.uniform(0.5, 5.0)
    timeout_violation_rate = rng.uniform(0.0, 0.2)
    p95_batch_size = rng.uniform(
        avg_batch_size, max(avg_batch_size, avg_batch_size * 1.5)
    )
    leader_change_count = rng.randint(0, 8)
    regency_change_count = rng.randint(0, 8)
    return agent_pb2.SbftReport(
        total_transactions=total_transactions,
        total_consensus_instances=total_consensus_instances,
        avg_consensus_latency_ms=avg_consensus_latency_ms,
        p95_consensus_latency_ms=p95_consensus_latency_ms,
        p99_consensus_latency_ms=p99_consensus_latency_ms,
        throughput_tps=throughput_tps,
        timeout_violation_rate=timeout_violation_rate,
        avg_batch_size=avg_batch_size,
        p95_batch_size=p95_batch_size,
        leader_change_count=leader_change_count,
        regency_change_count=regency_change_count,
    )


def _random_report(protocol: int, rng: random.Random) -> object:
    if protocol == agent_pb2.PROTOCOL_PBFT:
        return _random_pbft_report(rng)
    if protocol == agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT:
        return _random_two_pc_report(rng)
    if protocol == agent_pb2.PROTOCOL_SBFT:
        return _random_sbft_report(rng)
    raise ValueError(f"Unsupported protocol: {protocol}")


def _random_timeout_used(protocol: int, rng: random.Random) -> object:
    election_timeout_ms = rng.choice(TIMEOUT_OPTIONS_MS)
    if protocol == agent_pb2.PROTOCOL_PBFT:
        return agent_pb2.PbftTimeout(
            election_timeout_milliseconds=election_timeout_ms
        )
    if protocol == agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT:
        prepare_timeout_ms = rng.choice(TIMEOUT_OPTIONS_MS)
        return agent_pb2.TwoPcOverPbftTimeout(
            election_timeout_milliseconds=election_timeout_ms,
            prepare_timeout_milliseconds=prepare_timeout_ms,
        )
    if protocol == agent_pb2.PROTOCOL_SBFT:
        slow_path_timeout_ms = rng.choice(TIMEOUT_OPTIONS_MS)
        batch_timeout_ms = rng.choice(TIMEOUT_OPTIONS_MS)
        return agent_pb2.SbftTimeout(
            election_timeout_milliseconds=election_timeout_ms,
            slow_path_timeout_milliseconds=slow_path_timeout_ms,
            batch_timeout_milliseconds=batch_timeout_ms,
        )
    raise ValueError(f"Unsupported protocol: {protocol}")


def _print_sent_payload(
    node_id: int,
    episode: int,
    protocol: int,
    report: object,
    reward: Optional[agent_pb2.Reward],
) -> None:
    if protocol == agent_pb2.PROTOCOL_PBFT:
        print(
            "Sent ReportLocal:",
            f"node_id={node_id}",
            f"episode={episode}",
            f"protocol={agent_pb2.Protocol.Name(protocol)}",
            f"total_transactions={int(report.total_transactions)}",
            f"throughput_tps={float(report.throughput_tps):.2f}",
        )
    elif protocol == agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT:
        print(
            "Sent ReportLocal:",
            f"node_id={node_id}",
            f"episode={episode}",
            f"protocol={agent_pb2.Protocol.Name(protocol)}",
            f"total_transactions={int(report.total_transactions)}",
            f"prepare_timeout_rate={float(report.prepare_timeout_rate):.3f}",
            f"commit_rate={float(report.commit_rate):.3f}",
        )
    else:
        print(
            "Sent ReportLocal:",
            f"node_id={node_id}",
            f"episode={episode}",
            f"protocol={agent_pb2.Protocol.Name(protocol)}",
            f"total_transactions={int(report.total_transactions)}",
            f"throughput_tps={float(report.throughput_tps):.2f}",
            f"timeout_violation_rate={float(report.timeout_violation_rate):.3f}",
        )
    if reward is None:
        return

    if reward.HasField("pbft"):
        timeout_ms = reward.pbft.timeout_used.election_timeout_milliseconds
        reward_episode = reward.pbft.episode
        timeout_desc = f"reward.election_timeout_ms={timeout_ms}"
    elif reward.HasField("two_pc_over_pbft"):
        election_timeout_ms = (
            reward.two_pc_over_pbft.timeout_used.election_timeout_milliseconds
        )
        prepare_timeout_ms = (
            reward.two_pc_over_pbft.timeout_used.prepare_timeout_milliseconds
        )
        reward_episode = reward.two_pc_over_pbft.episode
        timeout_desc = (
            f"reward.election_timeout_ms={election_timeout_ms} "
            f"reward.prepare_timeout_ms={prepare_timeout_ms}"
        )
    elif reward.HasField("sbft"):
        election_timeout_ms = reward.sbft.timeout_used.election_timeout_milliseconds
        slow_path_timeout_ms = reward.sbft.timeout_used.slow_path_timeout_milliseconds
        batch_timeout_ms = reward.sbft.timeout_used.batch_timeout_milliseconds
        reward_episode = reward.sbft.episode
        timeout_desc = (
            f"reward.election_timeout_ms={election_timeout_ms} "
            f"reward.slow_path_timeout_ms={slow_path_timeout_ms} "
            f"reward.batch_timeout_ms={batch_timeout_ms}"
        )
    else:
        reward_episode = -1
        timeout_desc = "reward.timeout=unknown"

    print(
        "Sent Reward:",
        f"reward.episode={reward_episode}",
        timeout_desc,
    )


def _print_timeout_status(status: agent_pb2.TimeoutStatus) -> None:
    if status.status != agent_pb2.TimeoutStatus.Status.READY or not status.HasField("timeout"):
        print("Timeout status:", agent_pb2.TimeoutStatus.Status.Name(status.status))
        return
    window_desc = (
        f"start_tick={status.start_tick} "
        f"report_seq={status.report_seq}"
    )
    if status.timeout.HasField("pbft"):
        print(
            "Timeout ready:",
            f"election_timeout_ms={status.timeout.pbft.election_timeout_milliseconds}",
            window_desc,
        )
        return
    if status.timeout.HasField("two_pc_over_pbft"):
        print(
            "Timeout ready:",
            "election_timeout_ms="
            f"{status.timeout.two_pc_over_pbft.election_timeout_milliseconds}",
            "prepare_timeout_ms="
            f"{status.timeout.two_pc_over_pbft.prepare_timeout_milliseconds}",
            window_desc,
        )
        return
    if status.timeout.HasField("sbft"):
        print(
            "Timeout ready:",
            "election_timeout_ms="
            f"{status.timeout.sbft.election_timeout_milliseconds}",
            "slow_path_timeout_ms="
            f"{status.timeout.sbft.slow_path_timeout_milliseconds}",
            "batch_timeout_ms="
            f"{status.timeout.sbft.batch_timeout_milliseconds}",
            window_desc,
        )
        return
    print("Timeout status:", agent_pb2.TimeoutStatus.Status.Name(status.status))


def main() -> None:
    parser = argparse.ArgumentParser(description="LearningAgent client")
    parser.add_argument("--hosts-config", type=Path, default=Path("config/hosts.config"))
    parser.add_argument("--node-id", type=int, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument(
        "--protocol",
        type=_parse_protocol,
        default=agent_pb2.PROTOCOL_PBFT,
        help="Protocol: pbft, two_pc_over_pbft (alias: 2pc), or sbft.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--start-tick",
        type=int,
        default=0,
        help="Report window start consensus tick.",
    )
    parser.add_argument(
        "--report-seq",
        type=int,
        default=10,
        help="Report window end consensus tick (exclusive). Must be > --start-tick.",
    )
    parser.add_argument(
        "--get",
        action="store_true",
        help="Only query timeout status for the episode; do not send a report.",
    )
    parser.add_argument(
        "--reward",
        action="store_true",
        help=(
            "Attach a randomized reward payload using random timeout values from "
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
        _print_timeout_status(client.get_timeout(args.episode, args.protocol))
    else:
        if args.report_seq <= args.start_tick:
            raise ValueError("--report-seq must be greater than --start-tick")
        rng = random.Random(args.seed)
        report = _random_report(args.protocol, rng)
        reward = None
        if args.reward:
            if args.episode <= 0:
                raise ValueError("--reward requires --episode >= 1")
            timeout_used = _random_timeout_used(args.protocol, rng)
            if args.protocol == agent_pb2.PROTOCOL_PBFT:
                reward = agent_pb2.Reward(
                    pbft=agent_pb2.PbftReward(
                        episode=args.episode - 1,
                        report=_random_pbft_report(rng),
                        timeout_used=timeout_used,
                    )
                )
            elif args.protocol == agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT:
                reward = agent_pb2.Reward(
                    two_pc_over_pbft=agent_pb2.TwoPcOverPbftReward(
                        episode=args.episode - 1,
                        report=_random_two_pc_report(rng),
                        timeout_used=timeout_used,
                    )
                )
            else:
                reward = agent_pb2.Reward(
                    sbft=agent_pb2.SbftReward(
                        episode=args.episode - 1,
                        report=_random_sbft_report(rng),
                        timeout_used=timeout_used,
                    )
                )
        client.send_report(
            node_id=args.node_id,
            episode=args.episode,
            protocol=args.protocol,
            report=report,
            start_tick=args.start_tick,
            report_seq=args.report_seq,
            reward=reward,
        )
        _print_sent_payload(
            node_id=args.node_id,
            episode=args.episode,
            protocol=args.protocol,
            report=report,
            reward=reward,
        )
    client.close()


if __name__ == "__main__":
    main()
