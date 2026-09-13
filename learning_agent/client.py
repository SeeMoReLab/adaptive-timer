import argparse
import random
from pathlib import Path
from typing import Dict, Optional

import grpc
from google.protobuf import empty_pb2

import agent_pb2
import agent_pb2_grpc

PBFT_TIMEOUT_OPTIONS_MS = (300, 800, 1800)
SBFT_TIMEOUT_OPTIONS_MS = (150, 400, 900)
TENDERMINT_PROPOSE_TIMEOUT_OPTIONS_MS = (150, 450, 750)
HOTSTUFF_TIMEOUT_DELAY_OPTIONS_MS = (800, 2000, 5000)
AUTOBAHN_TIMEOUT_DELAY_OPTIONS_MS = (800, 2000, 5000)
AUTOBAHN_CAR_TIMEOUT_OPTIONS_MS = (500, 1500, 5000)
AUTOBAHN_FAST_PATH_TIMEOUT_OPTIONS_MS = (500, 1500, 5000)
PROTOCOL_NAME_TO_ENUM = {
    "pbft": agent_pb2.PROTOCOL_PBFT,
    "sbft": agent_pb2.PROTOCOL_SBFT,
    "tendermint": agent_pb2.PROTOCOL_TENDERMINT,
    "hotstuff": agent_pb2.PROTOCOL_HOTSTUFF,
    "autobahn": agent_pb2.PROTOCOL_AUTOBAHN,
    "smartbft": agent_pb2.PROTOCOL_SMARTBFT,
}


def _parse_protocol(value: str) -> int:
    normalized = value.strip().lower().replace("-", "_")
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
            window_consensus_count=(
                report.total_consensus_instances
                if protocol in (agent_pb2.PROTOCOL_PBFT, agent_pb2.PROTOCOL_SMARTBFT)
                else 0
            ),
        )
        if protocol == agent_pb2.PROTOCOL_PBFT:
            local_report.pbft_state.CopyFrom(report)
        elif protocol == agent_pb2.PROTOCOL_SBFT:
            local_report.sbft_state.CopyFrom(report)
        elif protocol == agent_pb2.PROTOCOL_TENDERMINT:
            local_report.tendermint_state.CopyFrom(report)
        elif protocol == agent_pb2.PROTOCOL_HOTSTUFF:
            local_report.hotstuff_state.CopyFrom(report)
        elif protocol == agent_pb2.PROTOCOL_AUTOBAHN:
            local_report.autobahn_state.CopyFrom(report)
        elif protocol == agent_pb2.PROTOCOL_SMARTBFT:
            local_report.smartbft_state.CopyFrom(report)
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


def _random_pbft_report(
    rng: random.Random, report_cls: type = agent_pb2.PbftReport
) -> object:
    """Random PbftReport, or a SmartbftReport when report_cls says so."""
    total_transactions = rng.randint(500, 20000)
    avg_batch_size = rng.uniform(1.0, 200.0)
    total_consensus_instances = max(
        1, int(round(total_transactions / max(avg_batch_size, 1.0)))
    )
    avg_consensus_latency_ms = rng.uniform(1.0, 200.0)
    p95_consensus_latency_ms = rng.uniform(
        avg_consensus_latency_ms, max(avg_consensus_latency_ms, avg_consensus_latency_ms * 2.0)
    )
    p50_consensus_latency_ms = rng.uniform(avg_consensus_latency_ms, p95_consensus_latency_ms)
    throughput_tps = total_transactions / rng.uniform(0.5, 5.0)
    p95_batch_size = rng.uniform(avg_batch_size, max(avg_batch_size, avg_batch_size * 1.5))
    leader_change_count = rng.randint(0, 8)
    regency_change_count = rng.randint(0, 8)
    timeout_ms = rng.choice([300, 800, 1800])
    avg_inter_commit_gap_ms = rng.uniform(1.0, 250.0)
    p50_inter_commit_gap_ms = rng.uniform(1.0, avg_inter_commit_gap_ms)
    p95_inter_commit_gap_ms = rng.uniform(
        avg_inter_commit_gap_ms, max(avg_inter_commit_gap_ms, avg_inter_commit_gap_ms * 2.0)
    )
    view_change_count = rng.randint(0, 8)
    no_progress_view_change_count = rng.randint(0, view_change_count)
    return report_cls(
        total_transactions=total_transactions,
        total_consensus_instances=total_consensus_instances,
        avg_consensus_latency_ms=avg_consensus_latency_ms,
        p50_consensus_latency_ms=p50_consensus_latency_ms,
        p95_consensus_latency_ms=p95_consensus_latency_ms,
        throughput_tps=throughput_tps,
        avg_batch_size=avg_batch_size,
        p95_batch_size=p95_batch_size,
        leader_change_count=leader_change_count,
        regency_change_count=regency_change_count,
        timeout_ms=timeout_ms,
        avg_inter_commit_gap_ms=avg_inter_commit_gap_ms,
        p50_inter_commit_gap_ms=p50_inter_commit_gap_ms,
        p95_inter_commit_gap_ms=p95_inter_commit_gap_ms,
        view_change_count=view_change_count,
        no_progress_view_change_count=no_progress_view_change_count,
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


def _random_hotstuff_report(rng: random.Random) -> agent_pb2.HotstuffReport:
    total_transactions = rng.randint(500, 20000)
    avg_batch_size = rng.uniform(1.0, 200.0)
    total_consensus_instances = max(
        1, int(round(total_transactions / max(avg_batch_size, 1.0)))
    )
    avg_consensus_latency_ms = rng.uniform(1.0, 200.0)
    p95_consensus_latency_ms = rng.uniform(
        avg_consensus_latency_ms, max(avg_consensus_latency_ms, avg_consensus_latency_ms * 2.0)
    )
    p50_consensus_latency_ms = rng.uniform(avg_consensus_latency_ms, p95_consensus_latency_ms)
    throughput_tps = total_transactions / rng.uniform(0.5, 5.0)
    p95_batch_size = rng.uniform(avg_batch_size, max(avg_batch_size, avg_batch_size * 1.5))
    leader_change_count = rng.randint(0, 8)
    regency_change_count = rng.randint(0, 8)
    timeout_ms = rng.choice(HOTSTUFF_TIMEOUT_DELAY_OPTIONS_MS)
    avg_inter_commit_gap_ms = rng.uniform(1.0, 250.0)
    p50_inter_commit_gap_ms = rng.uniform(1.0, avg_inter_commit_gap_ms)
    p95_inter_commit_gap_ms = rng.uniform(
        avg_inter_commit_gap_ms, max(avg_inter_commit_gap_ms, avg_inter_commit_gap_ms * 2.0)
    )
    view_change_count = rng.randint(0, 8)
    no_progress_view_change_count = rng.randint(0, view_change_count)
    return agent_pb2.HotstuffReport(
        total_transactions=total_transactions,
        total_consensus_instances=total_consensus_instances,
        avg_consensus_latency_ms=avg_consensus_latency_ms,
        p50_consensus_latency_ms=p50_consensus_latency_ms,
        p95_consensus_latency_ms=p95_consensus_latency_ms,
        throughput_tps=throughput_tps,
        avg_batch_size=avg_batch_size,
        p95_batch_size=p95_batch_size,
        leader_change_count=leader_change_count,
        regency_change_count=regency_change_count,
        timeout_ms=timeout_ms,
        avg_inter_commit_gap_ms=avg_inter_commit_gap_ms,
        p50_inter_commit_gap_ms=p50_inter_commit_gap_ms,
        p95_inter_commit_gap_ms=p95_inter_commit_gap_ms,
        view_change_count=view_change_count,
        no_progress_view_change_count=no_progress_view_change_count,
    )


def _random_autobahn_report(rng: random.Random) -> agent_pb2.AutobahnReport:
    total_transactions = rng.randint(500, 20000)
    avg_batch_size = rng.uniform(1.0, 200.0)
    total_consensus_instances = max(
        1, int(round(total_transactions / max(avg_batch_size, 1.0)))
    )
    avg_consensus_latency_ms = rng.uniform(1.0, 200.0)
    p95_consensus_latency_ms = rng.uniform(
        avg_consensus_latency_ms, max(avg_consensus_latency_ms, avg_consensus_latency_ms * 2.0)
    )
    p50_consensus_latency_ms = rng.uniform(avg_consensus_latency_ms, p95_consensus_latency_ms)
    throughput_tps = total_transactions / rng.uniform(0.5, 5.0)
    p95_batch_size = rng.uniform(avg_batch_size, max(avg_batch_size, avg_batch_size * 1.5))
    leader_change_count = rng.randint(0, 8)
    regency_change_count = rng.randint(0, 8)
    timeout_ms = rng.choice(AUTOBAHN_TIMEOUT_DELAY_OPTIONS_MS)
    avg_inter_commit_gap_ms = rng.uniform(1.0, 250.0)
    p50_inter_commit_gap_ms = rng.uniform(1.0, avg_inter_commit_gap_ms)
    p95_inter_commit_gap_ms = rng.uniform(
        avg_inter_commit_gap_ms, max(avg_inter_commit_gap_ms, avg_inter_commit_gap_ms * 2.0)
    )
    view_change_count = rng.randint(0, 8)
    no_progress_view_change_count = rng.randint(0, view_change_count)
    return agent_pb2.AutobahnReport(
        total_transactions=total_transactions,
        total_consensus_instances=total_consensus_instances,
        avg_consensus_latency_ms=avg_consensus_latency_ms,
        p50_consensus_latency_ms=p50_consensus_latency_ms,
        p95_consensus_latency_ms=p95_consensus_latency_ms,
        throughput_tps=throughput_tps,
        avg_batch_size=avg_batch_size,
        p95_batch_size=p95_batch_size,
        leader_change_count=leader_change_count,
        regency_change_count=regency_change_count,
        timeout_ms=timeout_ms,
        avg_inter_commit_gap_ms=avg_inter_commit_gap_ms,
        p50_inter_commit_gap_ms=p50_inter_commit_gap_ms,
        p95_inter_commit_gap_ms=p95_inter_commit_gap_ms,
        view_change_count=view_change_count,
        no_progress_view_change_count=no_progress_view_change_count,
    )


def _random_report(protocol: int, rng: random.Random) -> object:
    if protocol == agent_pb2.PROTOCOL_PBFT:
        return _random_pbft_report(rng)
    if protocol == agent_pb2.PROTOCOL_SMARTBFT:
        return _random_pbft_report(rng, agent_pb2.SmartbftReport)
    if protocol == agent_pb2.PROTOCOL_SBFT:
        return _random_sbft_report(rng)
    if protocol == agent_pb2.PROTOCOL_TENDERMINT:
        return _random_tendermint_report(rng)
    if protocol == agent_pb2.PROTOCOL_HOTSTUFF:
        return _random_hotstuff_report(rng)
    if protocol == agent_pb2.PROTOCOL_AUTOBAHN:
        return _random_autobahn_report(rng)
    raise ValueError(f"Unsupported protocol: {protocol}")


def _random_timeout_used(protocol: int, rng: random.Random) -> object:
    if protocol == agent_pb2.PROTOCOL_PBFT:
        return agent_pb2.PbftTimeout(
            election_timeout_milliseconds=rng.choice(PBFT_TIMEOUT_OPTIONS_MS)
        )
    if protocol == agent_pb2.PROTOCOL_SMARTBFT:
        return agent_pb2.SmartbftTimeout(
            election_timeout_milliseconds=rng.choice(PBFT_TIMEOUT_OPTIONS_MS)
        )
    if protocol == agent_pb2.PROTOCOL_SBFT:
        election_timeout_ms = rng.choice(SBFT_TIMEOUT_OPTIONS_MS)
        slow_path_timeout_ms = rng.choice(SBFT_TIMEOUT_OPTIONS_MS)
        batch_timeout_ms = rng.choice(SBFT_TIMEOUT_OPTIONS_MS)
        return agent_pb2.SbftTimeout(
            election_timeout_milliseconds=election_timeout_ms,
            slow_path_timeout_milliseconds=slow_path_timeout_ms,
            batch_timeout_milliseconds=batch_timeout_ms,
        )
    if protocol == agent_pb2.PROTOCOL_TENDERMINT:
        propose_timeout_ms = rng.choice(TENDERMINT_PROPOSE_TIMEOUT_OPTIONS_MS)
        phase_timeout_ms = propose_timeout_ms // 3
        return agent_pb2.TendermintTimeout(
            propose_timeout_milliseconds=propose_timeout_ms,
            prevote_timeout_milliseconds=phase_timeout_ms,
            precommit_timeout_milliseconds=phase_timeout_ms,
        )
    if protocol == agent_pb2.PROTOCOL_HOTSTUFF:
        return agent_pb2.HotstuffTimeout(
            timeout_delay_milliseconds=rng.choice(HOTSTUFF_TIMEOUT_DELAY_OPTIONS_MS)
        )
    if protocol == agent_pb2.PROTOCOL_AUTOBAHN:
        timeout_delay_ms = rng.choice(AUTOBAHN_TIMEOUT_DELAY_OPTIONS_MS)
        car_timeout_ms = rng.choice(AUTOBAHN_CAR_TIMEOUT_OPTIONS_MS)
        fast_path_timeout_ms = rng.choice(AUTOBAHN_FAST_PATH_TIMEOUT_OPTIONS_MS)
        return agent_pb2.AutobahnTimeout(
            timeout_delay_milliseconds=timeout_delay_ms,
            car_timeout_milliseconds=car_timeout_ms,
            fast_path_timeout_milliseconds=fast_path_timeout_ms,
        )
    raise ValueError(f"Unsupported protocol: {protocol}")


def _print_sent_payload(
    node_id: int,
    episode: int,
    protocol: int,
    report: object,
    reward: Optional[agent_pb2.Reward],
) -> None:
    if protocol in (agent_pb2.PROTOCOL_PBFT, agent_pb2.PROTOCOL_SMARTBFT):
        print(
            "Sent ReportLocal:",
            f"node_id={node_id}",
            f"episode={episode}",
            f"protocol={agent_pb2.Protocol.Name(protocol)}",
            f"total_transactions={int(report.total_transactions)}",
            f"throughput_tps={float(report.throughput_tps):.2f}",
        )
    elif protocol == agent_pb2.PROTOCOL_SBFT:
        print(
            "Sent ReportLocal:",
            f"node_id={node_id}",
            f"episode={episode}",
            f"protocol={agent_pb2.Protocol.Name(protocol)}",
            f"total_transactions={int(report.total_transactions)}",
            f"throughput_tps={float(report.throughput_tps):.2f}",
            f"timeout_violation_rate={float(report.timeout_violation_rate):.3f}",
        )
    elif protocol in (agent_pb2.PROTOCOL_HOTSTUFF, agent_pb2.PROTOCOL_AUTOBAHN):
        print(
            "Sent ReportLocal:",
            f"node_id={node_id}",
            f"episode={episode}",
            f"protocol={agent_pb2.Protocol.Name(protocol)}",
            f"total_transactions={int(report.total_transactions)}",
            f"throughput_tps={float(report.throughput_tps):.2f}",
            f"view_change_count={int(report.view_change_count)}",
        )
    else:
        print(
            "Sent ReportLocal:",
            f"node_id={node_id}",
            f"episode={episode}",
            f"protocol={agent_pb2.Protocol.Name(protocol)}",
            f"total_transactions={int(report.total_transactions)}",
            f"throughput_tps={float(report.throughput_tps):.2f}",
            f"propose_latency_ms={float(report.propose_latency_ms):.2f}",
            f"prevote_latency_ms={float(report.prevote_latency_ms):.2f}",
            f"precommit_latency_ms={float(report.precommit_latency_ms):.2f}",
        )
    if reward is None:
        return

    if reward.HasField("pbft"):
        timeout_ms = reward.pbft.timeout_used.election_timeout_milliseconds
        reward_episode = reward.pbft.episode
        timeout_desc = f"reward.election_timeout_ms={timeout_ms}"
    elif reward.HasField("smartbft"):
        timeout_ms = reward.smartbft.timeout_used.election_timeout_milliseconds
        reward_episode = reward.smartbft.episode
        timeout_desc = f"reward.election_timeout_ms={timeout_ms}"
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
    elif reward.HasField("tendermint"):
        propose_timeout_ms = reward.tendermint.timeout_used.propose_timeout_milliseconds
        prevote_timeout_ms = reward.tendermint.timeout_used.prevote_timeout_milliseconds
        precommit_timeout_ms = reward.tendermint.timeout_used.precommit_timeout_milliseconds
        reward_episode = reward.tendermint.episode
        timeout_desc = (
            f"reward.propose_timeout_ms={propose_timeout_ms} "
            f"reward.prevote_timeout_ms={prevote_timeout_ms} "
            f"reward.precommit_timeout_ms={precommit_timeout_ms}"
        )
    elif reward.HasField("hotstuff"):
        timeout_delay_ms = reward.hotstuff.timeout_used.timeout_delay_milliseconds
        reward_episode = reward.hotstuff.episode
        timeout_desc = f"reward.timeout_delay_ms={timeout_delay_ms}"
    elif reward.HasField("autobahn"):
        timeout_delay_ms = reward.autobahn.timeout_used.timeout_delay_milliseconds
        car_timeout_ms = reward.autobahn.timeout_used.car_timeout_milliseconds
        fast_path_timeout_ms = reward.autobahn.timeout_used.fast_path_timeout_milliseconds
        reward_episode = reward.autobahn.episode
        timeout_desc = (
            f"reward.timeout_delay_ms={timeout_delay_ms} "
            f"reward.car_timeout_ms={car_timeout_ms} "
            f"reward.fast_path_timeout_ms={fast_path_timeout_ms}"
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
        f"report_seq={status.report_seq} "
        f"window_consensus_count={status.window_consensus_count}"
    )
    if status.timeout.HasField("pbft"):
        print(
            "Timeout ready:",
            f"election_timeout_ms={status.timeout.pbft.election_timeout_milliseconds}",
            window_desc,
        )
        return
    if status.timeout.HasField("smartbft"):
        print(
            "Timeout ready:",
            f"election_timeout_ms={status.timeout.smartbft.election_timeout_milliseconds}",
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
    if status.timeout.HasField("tendermint"):
        print(
            "Timeout ready:",
            "propose_timeout_ms="
            f"{status.timeout.tendermint.propose_timeout_milliseconds}",
            "prevote_timeout_ms="
            f"{status.timeout.tendermint.prevote_timeout_milliseconds}",
            "precommit_timeout_ms="
            f"{status.timeout.tendermint.precommit_timeout_milliseconds}",
            window_desc,
        )
        return
    if status.timeout.HasField("hotstuff"):
        print(
            "Timeout ready:",
            "timeout_delay_ms="
            f"{status.timeout.hotstuff.timeout_delay_milliseconds}",
            window_desc,
        )
        return
    if status.timeout.HasField("autobahn"):
        print(
            "Timeout ready:",
            "timeout_delay_ms="
            f"{status.timeout.autobahn.timeout_delay_milliseconds}",
            "car_timeout_ms="
            f"{status.timeout.autobahn.car_timeout_milliseconds}",
            "fast_path_timeout_ms="
            f"{status.timeout.autobahn.fast_path_timeout_milliseconds}",
            window_desc,
        )
        return
    print("Timeout status:", agent_pb2.TimeoutStatus.Status.Name(status.status))


def _random_tendermint_report(rng: random.Random) -> agent_pb2.TendermintReport:
    total_transactions = rng.randint(500, 20000)
    avg_batch_size = rng.uniform(1.0, 200.0)
    total_consensus_instances = max(
        1, int(round(total_transactions / max(avg_batch_size, 1.0)))
    )
    avg_consensus_latency_ms = rng.uniform(1.0, 250.0)
    p50_consensus_latency_ms = rng.uniform(
        avg_consensus_latency_ms,
        max(avg_consensus_latency_ms, avg_consensus_latency_ms * 1.5),
    )
    p90_consensus_latency_ms = rng.uniform(
        p50_consensus_latency_ms,
        max(p50_consensus_latency_ms, p50_consensus_latency_ms * 1.5),
    )
    throughput_tps = total_transactions / rng.uniform(0.5, 5.0)
    timeout_violation_rate = rng.uniform(0.0, 0.2)
    p50_batch_size = rng.uniform(
        avg_batch_size, max(avg_batch_size, avg_batch_size * 1.5)
    )
    p90_batch_size = rng.uniform(
        p50_batch_size, max(p50_batch_size, p50_batch_size * 1.5)
    )
    leader_change_count = rng.randint(0, 8)
    regency_change_count = rng.randint(0, 8)
    propose_latency_ms = rng.uniform(0.0, avg_consensus_latency_ms)
    prevote_latency_ms = rng.uniform(0.0, avg_consensus_latency_ms)
    precommit_latency_ms = rng.uniform(0.0, avg_consensus_latency_ms)
    return agent_pb2.TendermintReport(
        total_transactions=total_transactions,
        total_consensus_instances=total_consensus_instances,
        avg_consensus_latency_ms=avg_consensus_latency_ms,
        p50_consensus_latency_ms=p50_consensus_latency_ms,
        p90_consensus_latency_ms=p90_consensus_latency_ms,
        throughput_tps=throughput_tps,
        timeout_violation_rate=timeout_violation_rate,
        avg_batch_size=avg_batch_size,
        p50_batch_size=p50_batch_size,
        p90_batch_size=p90_batch_size,
        leader_change_count=leader_change_count,
        regency_change_count=regency_change_count,
        propose_latency_ms=propose_latency_ms,
        prevote_latency_ms=prevote_latency_ms,
        precommit_latency_ms=precommit_latency_ms,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LearningAgent client")
    parser.add_argument("--hosts-config", type=Path, default=Path("config/hosts.config"))
    parser.add_argument("--node-id", type=int, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument(
        "--protocol",
        type=_parse_protocol,
        default=agent_pb2.PROTOCOL_PBFT,
        help="Protocol: pbft, smartbft, sbft, tendermint, hotstuff, or autobahn.",
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
            "Attach a randomized reward payload using protocol-specific timeout "
            "options."
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
            elif args.protocol == agent_pb2.PROTOCOL_SMARTBFT:
                reward = agent_pb2.Reward(
                    smartbft=agent_pb2.SmartbftReward(
                        episode=args.episode - 1,
                        report=_random_pbft_report(rng, agent_pb2.SmartbftReport),
                        timeout_used=timeout_used,
                    )
                )
            elif args.protocol == agent_pb2.PROTOCOL_SBFT:
                reward = agent_pb2.Reward(
                    sbft=agent_pb2.SbftReward(
                        episode=args.episode - 1,
                        report=_random_sbft_report(rng),
                        timeout_used=timeout_used,
                    )
                )
            elif args.protocol == agent_pb2.PROTOCOL_HOTSTUFF:
                reward = agent_pb2.Reward(
                    hotstuff=agent_pb2.HotstuffReward(
                        episode=args.episode - 1,
                        report=_random_hotstuff_report(rng),
                        timeout_used=timeout_used,
                    )
                )
            elif args.protocol == agent_pb2.PROTOCOL_AUTOBAHN:
                reward = agent_pb2.Reward(
                    autobahn=agent_pb2.AutobahnReward(
                        episode=args.episode - 1,
                        report=_random_autobahn_report(rng),
                        timeout_used=timeout_used,
                    )
                )
            else:
                reward = agent_pb2.Reward(
                    tendermint=agent_pb2.TendermintReward(
                        episode=args.episode - 1,
                        report=_random_tendermint_report(rng),
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
