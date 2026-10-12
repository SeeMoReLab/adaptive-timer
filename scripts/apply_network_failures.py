#!/usr/bin/env python3
"""Apply phase-scheduled network faults using tc netem.

This applier intentionally handles only the <network> section of a
failure_spec.xml and ignores the protocol and agent fault sections. Two fault
kinds are supported per phase:

- burst delays on the global / per-source / per-edge rules (delayMs plus the
  burst schedule), applied as netem delay with jitter at <globalNetworkDelay>/<jitterRatio> (default 20 percent);
- <silence>, a list of replica ids that go dark for the phase. Every applier
  cuts a silenced replica off like a crash: iptables REJECT (icmp-port-unreachable)
  in both directions so new connects fail instantly, plus `ss -K` to destroy the
  established connections (black-holed packets would leave them hanging),
  so the replica neither sends nor receives: the fail-silent crash fault of
  the BFT model, without any process lifecycle management.
"""

from __future__ import annotations

import argparse
import heapq
import hashlib
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple

# Default netem jitter as a fraction of the delay, when the spec does not set
# <globalNetworkDelay>/<jitterRatio>. netem jitter reorders packets, which TCP
# treats as loss, so high-throughput protocols should set it to 0 explicitly.
DEFAULT_NETWORK_DELAY_JITTER_RATIO = 0.20


@dataclass(frozen=True)
class BurstDelay:
    delay_ms: float
    burst_duration_ms: float
    burst_interval_ms: float
    baseline_delay_ms: float = 0.0
    burst_start_offset_ms: Optional[float] = None


# iptables chain (filter table) that holds the <silence> rules. It is jumped
# to from INPUT and OUTPUT, flushed on every phase change and deleted at
# teardown, so the rest of the namespace's firewall is never touched.
SILENCE_CHAIN = "ADAPTIVE_SILENCE"


@dataclass
class PhaseNetworkSpec:
    at_time_s: float
    directed_edges: bool
    global_fault: BurstDelay
    source_faults: Dict[int, BurstDelay]
    edge_faults: Dict[Tuple[int, int], BurstDelay]
    # Replicas that are fail-silent for this phase: no traffic in or out.
    silenced: FrozenSet[int]


@dataclass(frozen=True)
class HostEntry:
    replica_id: int
    ip: str


@dataclass(frozen=True)
class IpRule:
    dst_ip: str
    dst_replicas: Tuple[int, ...]
    fault: BurstDelay
    class_id: int


@dataclass(frozen=True)
class BurstEvent:
    at_s: float
    rule: IpRule
    active: bool


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{ts}] {msg}", flush=True)


def _parse_bool(text: str | None, default: bool) -> bool:
    if text is None:
        return default
    value = text.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {text!r}")


def _parse_required_float(parent: ET.Element, tag: str) -> float:
    node = parent.find(tag)
    if node is None or node.text is None:
        raise ValueError(f"Missing required tag <{tag}>")
    try:
        return float(node.text.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid float in <{tag}>: {node.text!r}") from exc


def _parse_required_int(parent: ET.Element, tag: str) -> int:
    node = parent.find(tag)
    if node is None or node.text is None:
        raise ValueError(f"Missing required tag <{tag}>")
    try:
        return int(node.text.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid integer in <{tag}>: {node.text!r}") from exc


def _parse_optional_float(parent: ET.Element, tag: str) -> float | None:
    node = parent.find(tag)
    if node is None or node.text is None:
        return None
    try:
        return float(node.text.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid float in <{tag}>: {node.text!r}") from exc


def _parse_optional_int(parent: ET.Element, tag: str) -> int | None:
    node = parent.find(tag)
    if node is None or node.text is None:
        return None
    try:
        return int(node.text.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid integer in <{tag}>: {node.text!r}") from exc


def _parse_burst_delay(node: ET.Element) -> BurstDelay:
    delay_ms = _parse_required_float(node, "delayMs")
    if delay_ms < 0:
        raise ValueError("delayMs must be >= 0")

    baseline_delay_ms = _parse_optional_float(node, "baselineDelayMs") or 0.0
    if baseline_delay_ms < 0:
        raise ValueError("baselineDelayMs must be >= 0")

    burst_start_offset_ms = _parse_optional_float(node, "burstStartOffsetMs")
    if burst_start_offset_ms is not None and burst_start_offset_ms < 0:
        raise ValueError("burstStartOffsetMs must be >= 0")

    burst_duration_ms = _parse_optional_float(node, "burstDurationMs")
    burst_interval_ms = _parse_optional_float(node, "burstIntervalMs")
    if burst_duration_ms is None and burst_interval_ms is None:
        probability_pct = _parse_optional_float(node, "probability")
        if probability_pct is not None:
            if probability_pct < 0 or probability_pct > 100:
                raise ValueError("probability must be between 0 and 100")
            burst_interval_ms = 10_000.0
            burst_duration_ms = burst_interval_ms * probability_pct / 100.0
        elif delay_ms > 0:
            # Legacy deterministic delay specs had no burst/probability fields.
            burst_duration_ms = 10_000.0
            burst_interval_ms = 10_000.0
        else:
            burst_duration_ms = 0.0
            burst_interval_ms = 0.0
    elif burst_duration_ms is None or burst_interval_ms is None:
        raise ValueError("burstDurationMs and burstIntervalMs must be specified together")

    legacy_loss_pct = _parse_optional_float(node, "lossPct")
    if legacy_loss_pct is not None and legacy_loss_pct != 0:
        raise ValueError(
            "Legacy <lossPct> is no longer supported for network phases; "
            "use burstDurationMs/burstIntervalMs to express delay bursts."
        )

    if burst_duration_ms < 0:
        raise ValueError("burstDurationMs must be >= 0")
    if burst_interval_ms < 0:
        raise ValueError("burstIntervalMs must be >= 0")
    if burst_duration_ms > 0 and burst_interval_ms <= 0:
        raise ValueError("burstIntervalMs must be > 0 when burstDurationMs is > 0")
    if burst_start_offset_ms is not None:
        latest_start_ms = burst_interval_ms - burst_duration_ms
        if burst_start_offset_ms > latest_start_ms:
            raise ValueError(
                "burstStartOffsetMs must leave room for the burst inside its "
                f"interval: {burst_start_offset_ms:g} > {latest_start_ms:g} "
                f"(burstIntervalMs {burst_interval_ms:g} - burstDurationMs "
                f"{burst_duration_ms:g})"
            )

    return BurstDelay(
        delay_ms=delay_ms,
        burst_duration_ms=burst_duration_ms,
        burst_interval_ms=burst_interval_ms,
        baseline_delay_ms=baseline_delay_ms,
        burst_start_offset_ms=burst_start_offset_ms,
    )


def _parse_global_network_delay_ms(root: ET.Element) -> float:
    delay_node = root.find("./globalNetworkDelay/delayMs")
    if delay_node is None or delay_node.text is None:
        return 0.0
    try:
        delay_ms = float(delay_node.text.strip())
    except ValueError as exc:
        raise ValueError(
            f"Invalid float in <globalNetworkDelay>/<delayMs>: {delay_node.text!r}"
        ) from exc
    if delay_ms < 0:
        raise ValueError("globalNetworkDelay delayMs must be >= 0")
    return delay_ms


def _parse_jitter_ratio(root: ET.Element) -> float:
    node = root.find("./globalNetworkDelay/jitterRatio")
    if node is None or node.text is None:
        return DEFAULT_NETWORK_DELAY_JITTER_RATIO
    try:
        ratio = float(node.text.strip())
    except ValueError as exc:
        raise ValueError(
            f"Invalid float in <globalNetworkDelay>/<jitterRatio>: {node.text!r}"
        ) from exc
    if ratio < 0 or ratio >= 1:
        raise ValueError("globalNetworkDelay jitterRatio must be in [0, 1)")
    return ratio


def _parse_warmup_seconds(root: ET.Element) -> float:
    warmup_ms = _parse_optional_int(root, "warmUpTimeMs")
    if warmup_ms is not None:
        return max(0.0, warmup_ms / 1000.0)

    warmup_s = _parse_optional_float(root, "warmUpTime")
    if warmup_s is None:
        return 0.0
    return max(0.0, warmup_s)


def _parse_phases(spec_path: Path) -> Tuple[float, float, float, List[PhaseNetworkSpec]]:
    root = ET.parse(spec_path).getroot()
    warmup_s = _parse_warmup_seconds(root)
    base_delay_ms = _parse_global_network_delay_ms(root)
    jitter_ratio = _parse_jitter_ratio(root)
    phase_nodes = root.findall("./phases/phase")
    if not phase_nodes:
        raise ValueError("No <phase> entries found under <phases>")

    has_at_time = any(p.find("atTime") is not None for p in phase_nodes)
    phases: List[PhaseNetworkSpec] = []

    # Backward compatibility: if atTime is absent, treat <time> as duration.
    cumulative_start = 0.0
    for index, phase_node in enumerate(phase_nodes):
        if has_at_time:
            if phase_node.find("atTime") is not None:
                at_time_s = float(_parse_required_int(phase_node, "atTime"))
            else:
                at_time_s = float(_parse_required_int(phase_node, "time"))
        else:
            at_time_s = cumulative_start
            duration_s = float(_parse_required_int(phase_node, "time"))
            if duration_s < 0:
                raise ValueError("Legacy phase duration <time> must be >= 0")
            cumulative_start += duration_s

        if at_time_s < 0:
            raise ValueError("Phase trigger time must be >= 0")

        network = phase_node.find("network")
        if network is None:
            # No network section in this phase means clear impairments.
            phases.append(
                PhaseNetworkSpec(
                    at_time_s=at_time_s,
                    directed_edges=True,
                    global_fault=BurstDelay(
                        delay_ms=0.0,
                        burst_duration_ms=0.0,
                        burst_interval_ms=0.0,
                    ),
                    source_faults={},
                    edge_faults={},
                    silenced=frozenset(),
                )
            )
            continue

        directed_edges = _parse_bool(network.findtext("directedEdges"), default=True)

        global_node = network.find("global")
        global_fault = (
            _parse_burst_delay(global_node)
            if global_node is not None
            else BurstDelay(delay_ms=0.0, burst_duration_ms=0.0, burst_interval_ms=0.0)
        )

        source_faults: Dict[int, BurstDelay] = {}
        for source_node in network.findall("./sources/source"):
            src = _parse_required_int(source_node, "src")
            source_faults[src] = _parse_burst_delay(source_node)

        edge_faults: Dict[Tuple[int, int], BurstDelay] = {}
        for edge_node in network.findall("./edges/edge"):
            src = _parse_required_int(edge_node, "src")
            dst = _parse_required_int(edge_node, "dst")
            edge_faults[(src, dst)] = _parse_burst_delay(edge_node)

        silenced = _parse_silenced(network)

        phases.append(
            PhaseNetworkSpec(
                at_time_s=at_time_s,
                directed_edges=directed_edges,
                global_fault=global_fault,
                source_faults=source_faults,
                edge_faults=edge_faults,
                silenced=silenced,
            )
        )

        if has_at_time and index > 0 and at_time_s < phases[index - 1].at_time_s:
            raise ValueError("Phases must be non-decreasing by atTime")

    if not has_at_time:
        _log("No <atTime> found; treating <time> as phase duration (legacy behavior).")

    return warmup_s, base_delay_ms, jitter_ratio, phases


def _load_hosts(hosts_config: Path) -> Dict[int, HostEntry]:
    entries: Dict[int, HostEntry] = {}
    with hosts_config.open("r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(
                    f"Invalid hosts config at line {lineno}: expected at least 2 columns"
                )
            try:
                replica_id = int(parts[0])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid replica id at line {lineno}: {parts[0]!r}"
                ) from exc
            entries[replica_id] = HostEntry(replica_id=replica_id, ip=parts[1])
    if not entries:
        raise ValueError(f"No entries found in hosts config: {hosts_config}")
    return entries


def _prepend_netns(cmd: List[str], netns: str | None) -> List[str]:
    if not netns:
        return cmd
    return ["ip", "netns", "exec", netns] + cmd


def _run_command(
    cmd: List[str],
    *,
    use_sudo: bool,
    dry_run: bool,
    check: bool = True,
    netns: str | None = None,
) -> subprocess.CompletedProcess[str]:
    full_cmd = _prepend_netns(cmd, netns)
    if use_sudo and os.geteuid() != 0:
        full_cmd = ["sudo", "-n"] + _prepend_netns(cmd, netns)

    if dry_run:
        _log("DRY RUN: " + " ".join(full_cmd))
        return subprocess.CompletedProcess(full_cmd, 0, "", "")

    return subprocess.run(
        full_cmd,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _resolve_interface(
    preferred: str | None,
    sample_dest_ip: str | None,
    *,
    use_sudo: bool,
    netns: str | None,
) -> str:
    if preferred:
        return preferred

    if sample_dest_ip:
        try:
            output = _run_command(
                ["ip", "route", "get", sample_dest_ip],
                use_sudo=use_sudo,
                dry_run=False,
                check=True,
                netns=netns,
            ).stdout
            match = re.search(r"\bdev\s+(\S+)", output)
            if match:
                return match.group(1)
        except subprocess.CalledProcessError:
            pass

    output = _run_command(
        ["ip", "route", "show", "default"],
        use_sudo=use_sudo,
        dry_run=False,
        check=True,
        netns=netns,
    ).stdout
    for line in output.splitlines():
        match = re.search(r"\bdev\s+(\S+)", line)
        if match:
            return match.group(1)
    raise RuntimeError("Unable to determine network interface. Use --interface.")


def _clear_tc(
    interface: str,
    *,
    tc_bin: str,
    use_sudo: bool,
    dry_run: bool,
    netns: str | None,
) -> None:
    _run_command(
        [tc_bin, "qdisc", "del", "dev", interface, "root"],
        use_sudo=use_sudo,
        dry_run=dry_run,
        check=False,
        netns=netns,
    )


def _silence_peer_ips(
    phase: PhaseNetworkSpec, src_replica_id: int, hosts: Dict[int, HostEntry]
) -> List[str]:
    """Replica IPs whose established TCP connections this replica must drop.

    A silenced replica drops every peer; a healthy replica drops every
    silenced peer. Both sides act, so no packet has to cross the wire.
    """
    ips: List[str] = []
    for replica_id, entry in sorted(hosts.items()):
        if replica_id == src_replica_id:
            continue
        if src_replica_id in phase.silenced or replica_id in phase.silenced:
            if entry.ip not in ips:
                ips.append(entry.ip)
    return ips


def _ensure_silence_chain(
    *, iptables_bin: str, use_sudo: bool, dry_run: bool, netns: str | None
) -> None:
    common = dict(use_sudo=use_sudo, dry_run=dry_run, netns=netns)
    _run_command([iptables_bin, "-N", SILENCE_CHAIN], check=False, **common)
    _run_command([iptables_bin, "-F", SILENCE_CHAIN], **common)
    for hook in ("INPUT", "OUTPUT"):
        present = _run_command(
            [iptables_bin, "-C", hook, "-j", SILENCE_CHAIN], check=False, **common
        )
        if dry_run or present.returncode != 0:
            _run_command([iptables_bin, "-I", hook, "-j", SILENCE_CHAIN], **common)


def _apply_silence(
    *,
    phase: PhaseNetworkSpec,
    src_replica_id: int,
    hosts: Dict[int, HostEntry],
    iptables_bin: str,
    ss_bin: str,
    use_sudo: bool,
    dry_run: bool,
    netns: str | None,
) -> None:
    """Install the phase's <silence> rules (flushing the previous phase's)."""
    common = dict(use_sudo=use_sudo, dry_run=dry_run, netns=netns)
    _ensure_silence_chain(iptables_bin=iptables_bin, **common)
    peer_ips = _silence_peer_ips(phase, src_replica_id, hosts)
    if not peer_ips:
        return
    self_ip = hosts[src_replica_id].ip
    silenced_here = src_replica_id in phase.silenced
    _log(
        f"Silence: replica {src_replica_id} "
        + ("cut off from everything but itself" if silenced_here else f"cut off from {peer_ips}")
        + f" (silenced={sorted(phase.silenced)})"
    )
    # Measured on the CloudLab kernel (6.8): REJECT --reject-with tcp-reset in
    # OUTPUT does NOT fail a local connect() (it times out like a drop), while
    # icmp-port-unreachable fails it instantly with ECONNREFUSED from either
    # side. No catch-all DROP: it would swallow the ICMP error itself. The
    # protocols are TCP-only.
    reject = ["-p", "tcp", "-j", "REJECT", "--reject-with", "icmp-port-unreachable"]
    if silenced_here:
        # Everything except loopback (learning agent) and the replica's own
        # address (primary <-> worker) is rejected, so the client's lanes to
        # this replica stall as well: a crashed replica ingests nothing. The
        # client's established connections keep retransmitting and resume by
        # themselves once the rules are lifted.
        for direction in ("-d", "-s"):
            _run_command(
                [iptables_bin, "-A", SILENCE_CHAIN, direction, "127.0.0.0/8", "-j", "RETURN"],
                **common,
            )
        for direction in ("-d", "-s"):
            _run_command(
                [iptables_bin, "-A", SILENCE_CHAIN, "!", direction, self_ip] + reject,
                **common,
            )
    else:
        for ip in peer_ips:
            for direction in ("-d", "-s"):
                _run_command(
                    [iptables_bin, "-A", SILENCE_CHAIN, direction, ip] + reject,
                    **common,
                )
    # Established connections are not affected by the rules (TCP treats the
    # ICMP error as soft and keeps retransmitting), so destroy the replica to
    # replica ones: the local side gets ECONNABORTED and the peer ECONNRESET
    # on its next write, as a crashed process's kernel would have produced.
    for ip in peer_ips:
        _run_command([ss_bin, "-K", "dst", ip], check=False, **common)


def _clear_silence(
    *, iptables_bin: str, use_sudo: bool, dry_run: bool, netns: str | None
) -> None:
    common = dict(use_sudo=use_sudo, dry_run=dry_run, netns=netns, check=False)
    for hook in ("INPUT", "OUTPUT"):
        _run_command([iptables_bin, "-D", hook, "-j", SILENCE_CHAIN], **common)
    _run_command([iptables_bin, "-F", SILENCE_CHAIN], **common)
    _run_command([iptables_bin, "-X", SILENCE_CHAIN], **common)


def _parse_silenced(network: ET.Element) -> FrozenSet[int]:
    """The <silence> block: replica ids that are fail-silent for the phase."""
    silence = network.find("silence")
    if silence is None:
        return frozenset()
    ids = set()
    for replica_node in silence.findall("replica"):
        if replica_node.text is None or not replica_node.text.strip():
            raise ValueError("<silence><replica> must contain a replica id")
        try:
            ids.add(int(replica_node.text.strip()))
        except ValueError as exc:
            raise ValueError(
                f"Invalid replica id in <silence>: {replica_node.text!r}"
            ) from exc
    return frozenset(ids)


def _choose_fault(phase: PhaseNetworkSpec, src: int, dst: int) -> BurstDelay:
    # <silence> is enforced with iptables (see _apply_silence), not netem, so
    # the delay rules below stay in place for a silenced replica's edges.
    fault = phase.global_fault
    if src in phase.source_faults:
        fault = phase.source_faults[src]
    if (src, dst) in phase.edge_faults:
        return phase.edge_faults[(src, dst)]
    if not phase.directed_edges and (dst, src) in phase.edge_faults:
        return phase.edge_faults[(dst, src)]
    return fault


def _build_ip_rules(
    phase: PhaseNetworkSpec, src_replica_id: int, hosts: Dict[int, HostEntry]
) -> List[IpRule]:
    by_ip: Dict[str, Tuple[BurstDelay, List[int]]] = {}
    for dst_replica_id, entry in sorted(hosts.items()):
        if dst_replica_id == src_replica_id:
            continue
        fault = _choose_fault(phase, src_replica_id, dst_replica_id)
        existing = by_ip.get(entry.ip)
        if existing is None:
            by_ip[entry.ip] = (fault, [dst_replica_id])
            continue
        existing_fault, dst_replicas = existing
        if existing_fault != fault:
            raise ValueError(
                "Cannot express different per-replica faults for replicas sharing "
                f"the same destination IP ({entry.ip}) without namespaces/port rules."
            )
        dst_replicas.append(dst_replica_id)

    rules: List[IpRule] = []
    for rule_idx, (dst_ip, (fault, dst_replicas)) in enumerate(sorted(by_ip.items()), start=1):
        rules.append(
            IpRule(
                dst_ip=dst_ip,
                dst_replicas=tuple(sorted(dst_replicas)),
                fault=fault,
                class_id=100 + rule_idx,
            )
        )
    return rules


def _log_phase_rule_plan(
    *,
    phase_index: int,
    src_replica_id: int,
    rules: List[IpRule],
    base_delay_ms: float,
) -> None:
    for rule in rules:
        fault = rule.fault
        _log(
            f"Phase #{phase_index} rule: src={src_replica_id} "
            f"dst_replicas={list(rule.dst_replicas)} dst_ip={rule.dst_ip} "
            f"base_delay_ms={base_delay_ms:g} "
            f"baseline_delay_ms={fault.baseline_delay_ms:g} "
            f"quiet_total_delay_ms={_quiet_delay_ms(base_delay_ms, fault):g} "
            f"burst_delay_ms={fault.delay_ms:g} "
            f"burst_duration_ms={fault.burst_duration_ms:g} "
            f"burst_interval_ms={fault.burst_interval_ms:g} "
            f"burst_start_offset_ms="
            f"{'random' if fault.burst_start_offset_ms is None else format(fault.burst_start_offset_ms, 'g')} "
            f"burst_total_delay_ms={base_delay_ms + fault.delay_ms:g}"
        )


def _log_tc_state(
    *,
    interface: str,
    tc_bin: str,
    use_sudo: bool,
    dry_run: bool,
    netns: str | None,
) -> None:
    if dry_run:
        return

    probes = [
        ("qdisc", [tc_bin, "qdisc", "show", "dev", interface]),
        ("class", [tc_bin, "class", "show", "dev", interface]),
        ("filter", [tc_bin, "filter", "show", "dev", interface, "parent", "1:"]),
    ]
    for label, cmd in probes:
        result = _run_command(
            cmd,
            use_sudo=use_sudo,
            dry_run=False,
            check=False,
            netns=netns,
        )
        if result.returncode != 0:
            _log(
                f"tc {label} probe failed (rc={result.returncode}) for {interface}: "
                f"{result.stderr.strip()}"
            )
            continue
        output = result.stdout.strip()
        if output:
            _log(f"tc {label} on {interface}:\n{output}")
        else:
            _log(f"tc {label} on {interface}: <empty>")


def _burst_enabled(fault: BurstDelay) -> bool:
    return (
        fault.delay_ms > 0
        and fault.burst_duration_ms > 0
        and fault.burst_interval_ms > 0
    )


def _burst_continuous(fault: BurstDelay) -> bool:
    return _burst_enabled(fault) and fault.burst_duration_ms >= fault.burst_interval_ms


def _stable_fraction(*parts: object) -> float:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value / float(2**64 - 1)


def _quiet_delay_ms(base_delay_ms: float, fault: BurstDelay) -> float:
    """Delay held between bursts: the experiment base plus the phase baseline."""
    return base_delay_ms + fault.baseline_delay_ms


def _initial_delay_ms(base_delay_ms: float, fault: BurstDelay) -> float:
    if _burst_continuous(fault):
        return base_delay_ms + fault.delay_ms
    return _quiet_delay_ms(base_delay_ms, fault)


def _netem_qdisc_cmd(
    tc_bin: str,
    interface: str,
    class_id: int,
    delay_ms: float,
    *,
    verb: str,
    netem_limit: int,
    jitter_ratio: float,
) -> List[str]:
    command = [
        tc_bin,
        "qdisc",
        verb,
        "dev",
        interface,
        "parent",
        f"1:{class_id}",
        "handle",
        f"{class_id}:",
        "netem",
        "limit",
        str(netem_limit),
        "delay",
        f"{delay_ms:g}ms",
    ]
    if delay_ms > 0 and jitter_ratio > 0:
        jitter_ms = delay_ms * jitter_ratio
        command.append(f"{jitter_ms:g}ms")
    return command


def _set_rule_delay(
    *,
    rule: IpRule,
    delay_ms: float,
    interface: str,
    tc_bin: str,
    netem_limit: int,
    jitter_ratio: float,
    use_sudo: bool,
    dry_run: bool,
    netns: str | None,
) -> None:
    _run_command(
        _netem_qdisc_cmd(
            tc_bin,
            interface,
            rule.class_id,
            delay_ms,
            verb="change",
            netem_limit=netem_limit,
            jitter_ratio=jitter_ratio,
        ),
        use_sudo=use_sudo,
        dry_run=dry_run,
        netns=netns,
    )


def _configure_phase_tc(
    *,
    rules: List[IpRule],
    base_delay_ms: float,
    initialized: bool,
    interface: str,
    tc_bin: str,
    link_rate: str,
    netem_limit: int,
    jitter_ratio: float,
    use_sudo: bool,
    dry_run: bool,
    netns: str | None,
) -> bool:
    if not initialized:
        _clear_tc(
            interface,
            tc_bin=tc_bin,
            use_sudo=use_sudo,
            dry_run=dry_run,
            netns=netns,
        )

        _run_command(
            [
                tc_bin,
                "qdisc",
                "replace",
                "dev",
                interface,
                "root",
                "handle",
                "1:",
                "htb",
                "default",
                "999",
            ],
            use_sudo=use_sudo,
            dry_run=dry_run,
            netns=netns,
        )
    else:
        _log(f"Updating tc rules on {interface} without deleting the root qdisc.")

    _run_command(
        [
            tc_bin,
            "class",
            "replace",
            "dev",
            interface,
            "parent",
            "1:",
            "classid",
            "1:999",
            "htb",
            "rate",
            link_rate,
            "ceil",
            link_rate,
        ],
        use_sudo=use_sudo,
        dry_run=dry_run,
        netns=netns,
    )

    for rule in rules:
        _run_command(
            [
                tc_bin,
                "class",
                "replace",
                "dev",
                interface,
                "parent",
                "1:",
                "classid",
                f"1:{rule.class_id}",
                "htb",
                "rate",
                link_rate,
                "ceil",
                link_rate,
            ],
            use_sudo=use_sudo,
            dry_run=dry_run,
            netns=netns,
        )
        if not initialized:
            _run_command(
                _netem_qdisc_cmd(
                    tc_bin,
                    interface,
                    rule.class_id,
                    _initial_delay_ms(base_delay_ms, rule.fault),
                    verb="replace",
                    netem_limit=netem_limit,
                    jitter_ratio=jitter_ratio,
                ),
                use_sudo=use_sudo,
                dry_run=dry_run,
                netns=netns,
            )
        else:
            result = _run_command(
                _netem_qdisc_cmd(
                    tc_bin,
                    interface,
                    rule.class_id,
                    _initial_delay_ms(base_delay_ms, rule.fault),
                    verb="change",
                    netem_limit=netem_limit,
                    jitter_ratio=jitter_ratio,
                ),
                use_sudo=use_sudo,
                dry_run=dry_run,
                check=False,
                netns=netns,
            )
            if result.returncode != 0:
                _log(
                    f"netem change failed for class 1:{rule.class_id} "
                    f"({result.stderr.strip()}); replacing child qdisc."
                )
                _run_command(
                    _netem_qdisc_cmd(
                        tc_bin,
                        interface,
                        rule.class_id,
                        _initial_delay_ms(base_delay_ms, rule.fault),
                        verb="replace",
                        netem_limit=netem_limit,
                        jitter_ratio=jitter_ratio,
                    ),
                    use_sudo=use_sudo,
                    dry_run=dry_run,
                    netns=netns,
                )

        _run_command(
            [
                tc_bin,
                "filter",
                "replace",
                "dev",
                interface,
                "protocol",
                "ip",
                "parent",
                "1:",
                "prio",
                "10",
                "u32",
                "match",
                "ip",
                "dst",
                f"{rule.dst_ip}/32",
                "flowid",
                f"1:{rule.class_id}",
            ],
            use_sudo=use_sudo,
            dry_run=dry_run,
            netns=netns,
        )

    return True


def _build_burst_events(
    *,
    phase_index: int,
    src_replica_id: int,
    rules: List[IpRule],
    phase_duration_s: float,
    burst_seed: int,
) -> List[BurstEvent]:
    events: List[BurstEvent] = []
    if phase_duration_s <= 0:
        return events

    for rule in rules:
        fault = rule.fault
        if not _burst_enabled(fault) or _burst_continuous(fault):
            continue

        interval_index = 0
        interval_s = fault.burst_interval_ms / 1000.0
        while (interval_index + 1) * interval_s <= phase_duration_s + 1e-9:
            start_event, end_event = _burst_event_pair(
                phase_index=phase_index,
                src_replica_id=src_replica_id,
                rule=rule,
                burst_seed=burst_seed,
                interval_index=interval_index,
            )
            events.append(start_event)
            events.append(end_event)
            interval_index += 1

    return sorted(
        events,
        key=lambda event: (event.at_s, 0 if event.active else 1, event.rule.dst_ip),
    )


def _burst_event_pair(
    *,
    phase_index: int,
    src_replica_id: int,
    rule: IpRule,
    burst_seed: int,
    interval_index: int,
) -> Tuple[BurstEvent, BurstEvent]:
    fault = rule.fault
    interval_s = fault.burst_interval_ms / 1000.0
    duration_s = fault.burst_duration_ms / 1000.0
    latest_start_offset_s = interval_s - duration_s
    if fault.burst_start_offset_ms is not None:
        # Every edge bursts at the same point in every interval, so the whole
        # network changes state together and the phase can be aligned to
        # something outside it.
        offset_s = fault.burst_start_offset_ms / 1000.0
    else:
        fraction = _stable_fraction(
            burst_seed,
            phase_index,
            src_replica_id,
            ",".join(str(replica) for replica in rule.dst_replicas),
            rule.dst_ip,
            interval_index,
        )
        offset_s = fraction * latest_start_offset_s
    start_s = interval_index * interval_s + offset_s
    end_s = start_s + duration_s
    return (
        BurstEvent(at_s=start_s, rule=rule, active=True),
        BurstEvent(at_s=end_s, rule=rule, active=False),
    )


def _apply_burst_event(
    *,
    phase_index: int,
    src_replica_id: int,
    event: BurstEvent,
    base_delay_ms: float,
    interface: str,
    tc_bin: str,
    netem_limit: int,
    jitter_ratio: float,
    use_sudo: bool,
    dry_run: bool,
    netns: str | None,
) -> None:
    delay_ms = (
        base_delay_ms + event.rule.fault.delay_ms
        if event.active
        else _quiet_delay_ms(base_delay_ms, event.rule.fault)
    )
    _log(
        f"Phase #{phase_index} burst {'start' if event.active else 'end'}: "
        f"src={src_replica_id} dst_replicas={list(event.rule.dst_replicas)} "
        f"dst_ip={event.rule.dst_ip} delay_ms={delay_ms:g}"
    )
    _set_rule_delay(
        rule=event.rule,
        delay_ms=delay_ms,
        interface=interface,
        tc_bin=tc_bin,
        netem_limit=netem_limit,
        jitter_ratio=jitter_ratio,
        use_sudo=use_sudo,
        dry_run=dry_run,
        netns=netns,
    )


def _push_open_ended_burst_pair(
    heap: List[Tuple[float, int, str, int, BurstEvent]],
    *,
    phase_index: int,
    src_replica_id: int,
    rule: IpRule,
    burst_seed: int,
    interval_index: int,
) -> None:
    start_event, end_event = _burst_event_pair(
        phase_index=phase_index,
        src_replica_id=src_replica_id,
        rule=rule,
        burst_seed=burst_seed,
        interval_index=interval_index,
    )
    heapq.heappush(
        heap,
        (start_event.at_s, 0, start_event.rule.dst_ip, interval_index, start_event),
    )
    heapq.heappush(
        heap,
        (end_event.at_s, 1, end_event.rule.dst_ip, interval_index, end_event),
    )


def _run_open_ended_bursts(
    *,
    phase_index: int,
    phase_start_mono: float,
    base_delay_ms: float,
    src_replica_id: int,
    rules: List[IpRule],
    interface: str,
    tc_bin: str,
    netem_limit: int,
    jitter_ratio: float,
    burst_seed: int,
    use_sudo: bool,
    dry_run: bool,
    netns: str | None,
) -> None:
    heap: List[Tuple[float, int, str, int, BurstEvent]] = []
    elapsed_s = max(0.0, time.monotonic() - phase_start_mono)
    for rule in rules:
        if not _burst_enabled(rule.fault) or _burst_continuous(rule.fault):
            continue
        interval_s = rule.fault.burst_interval_ms / 1000.0
        interval_index = int(elapsed_s // interval_s)
        while True:
            start_event, end_event = _burst_event_pair(
                phase_index=phase_index,
                src_replica_id=src_replica_id,
                rule=rule,
                burst_seed=burst_seed,
                interval_index=interval_index,
            )
            if end_event.at_s > elapsed_s:
                break
            interval_index += 1

        if start_event.at_s <= elapsed_s:
            start_event = BurstEvent(at_s=elapsed_s, rule=rule, active=True)

        heapq.heappush(
            heap,
            (start_event.at_s, 0, start_event.rule.dst_ip, interval_index, start_event),
        )
        heapq.heappush(
            heap,
            (end_event.at_s, 1, end_event.rule.dst_ip, interval_index, end_event),
        )

    if not heap:
        return

    _log(f"Phase #{phase_index} has no next phase; scheduling bursts until stopped.")
    while True:
        _, _, _, interval_index, event = heapq.heappop(heap)
        _sleep_until(phase_start_mono + event.at_s)
        _apply_burst_event(
            phase_index=phase_index,
            src_replica_id=src_replica_id,
            event=event,
            base_delay_ms=base_delay_ms,
            interface=interface,
            tc_bin=tc_bin,
            netem_limit=netem_limit,
            jitter_ratio=jitter_ratio,
            use_sudo=use_sudo,
            dry_run=dry_run,
            netns=netns,
        )
        if not event.active:
            _push_open_ended_burst_pair(
                heap,
                phase_index=phase_index,
                src_replica_id=src_replica_id,
                rule=event.rule,
                burst_seed=burst_seed,
                interval_index=interval_index + 1,
            )


def _sleep_until(target_mono: float) -> None:
    sleep_s = target_mono - time.monotonic()
    if sleep_s > 0:
        time.sleep(sleep_s)


def _run_phase(
    *,
    phase_index: int,
    phase: PhaseNetworkSpec,
    phase_start_mono: float,
    phase_end_mono: float | None,
    base_delay_ms: float,
    tc_initialized: bool,
    src_replica_id: int,
    hosts: Dict[int, HostEntry],
    interface: str,
    tc_bin: str,
    link_rate: str,
    netem_limit: int,
    jitter_ratio: float,
    iptables_bin: str,
    ss_bin: str,
    burst_seed: int,
    use_sudo: bool,
    dry_run: bool,
    netns: str | None,
) -> bool:
    rules = _build_ip_rules(phase, src_replica_id, hosts)
    phase_duration_s = (
        None if phase_end_mono is None else max(0.0, phase_end_mono - phase_start_mono)
    )
    duration_text = "open-ended" if phase_duration_s is None else f"{phase_duration_s:.3f}s"
    _log(
        f"Applying phase #{phase_index} at t={phase.at_time_s:.3f}s "
        f"on {interface}; base_delay_ms={base_delay_ms:g}; "
        f"rules={len(rules)}; duration={duration_text}"
    )
    _log_phase_rule_plan(
        phase_index=phase_index,
        src_replica_id=src_replica_id,
        rules=rules,
        base_delay_ms=base_delay_ms,
    )

    tc_initialized = _configure_phase_tc(
        rules=rules,
        base_delay_ms=base_delay_ms,
        initialized=tc_initialized,
        interface=interface,
        tc_bin=tc_bin,
        link_rate=link_rate,
        netem_limit=netem_limit,
        jitter_ratio=jitter_ratio,
        use_sudo=use_sudo,
        dry_run=dry_run,
        netns=netns,
    )

    _log_tc_state(
        interface=interface,
        tc_bin=tc_bin,
        use_sudo=use_sudo,
        dry_run=dry_run,
        netns=netns,
    )

    _apply_silence(
        phase=phase,
        src_replica_id=src_replica_id,
        hosts=hosts,
        iptables_bin=iptables_bin,
        ss_bin=ss_bin,
        use_sudo=use_sudo,
        dry_run=dry_run,
        netns=netns,
    )

    if phase_duration_s is None:
        _run_open_ended_bursts(
            phase_index=phase_index,
            phase_start_mono=phase_start_mono,
            base_delay_ms=base_delay_ms,
            src_replica_id=src_replica_id,
            rules=rules,
            interface=interface,
            tc_bin=tc_bin,
            netem_limit=netem_limit,
            jitter_ratio=jitter_ratio,
            burst_seed=burst_seed,
            use_sudo=use_sudo,
            dry_run=dry_run,
            netns=netns,
        )
        return tc_initialized

    events = _build_burst_events(
        phase_index=phase_index,
        src_replica_id=src_replica_id,
        rules=rules,
        phase_duration_s=phase_duration_s,
        burst_seed=burst_seed,
    )
    for event in events:
        target_mono = phase_start_mono + event.at_s
        if phase_end_mono is not None and target_mono >= phase_end_mono:
            break
        _sleep_until(target_mono)
        _apply_burst_event(
            phase_index=phase_index,
            src_replica_id=src_replica_id,
            event=event,
            base_delay_ms=base_delay_ms,
            interface=interface,
            tc_bin=tc_bin,
            netem_limit=netem_limit,
            jitter_ratio=jitter_ratio,
            use_sudo=use_sudo,
            dry_run=dry_run,
            netns=netns,
        )

    return tc_initialized


def _compute_start_mono(start_unix_ms: int | None) -> float:
    if start_unix_ms is None:
        return time.monotonic()

    target_epoch_s = start_unix_ms / 1000.0
    now_epoch_s = time.time()
    if target_epoch_s > now_epoch_s:
        sleep_s = target_epoch_s - now_epoch_s
        _log(f"Waiting {sleep_s:.3f}s for start_unix_ms={start_unix_ms}.")
        time.sleep(sleep_s)

    # Anchor a monotonic timeline to the requested epoch start.
    elapsed_since_start_s = max(0.0, time.time() - target_epoch_s)
    return time.monotonic() - elapsed_since_start_s


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply burst-delay phases from failure_spec.xml using tc/netem."
    )
    parser.add_argument(
        "--failure-spec",
        type=Path,
        default=Path("config/smartbft/failure_spec.xml"),
        help="Path to failure spec XML.",
    )
    parser.add_argument(
        "--hosts-config",
        type=Path,
        required=True,
        help="Path to hosts.config with replica_id and IP mappings.",
    )
    parser.add_argument(
        "--replica-id",
        type=int,
        required=True,
        help="Global replica ID of this local applier/source.",
    )
    parser.add_argument(
        "--interface",
        type=str,
        default=None,
        help="Network interface device (auto-detected if omitted).",
    )
    parser.add_argument(
        "--netns",
        type=str,
        default=None,
        help="Optional network namespace name to run tc/ip commands in.",
    )
    parser.add_argument(
        "--start-unix-ms",
        type=int,
        default=None,
        help="Common experiment start time (epoch ms). If omitted, start now.",
    )
    parser.add_argument(
        "--teardown-at-end",
        action="store_true",
        help="Clear tc qdisc at the end instead of leaving the final phase active.",
    )
    parser.add_argument(
        "--tc-bin",
        type=str,
        default="tc",
        help="Path/binary name for tc.",
    )
    parser.add_argument(
        "--iptables-bin",
        type=str,
        default="iptables",
        help="Path/binary name for iptables (used by the <silence> fault).",
    )
    parser.add_argument(
        "--ss-bin",
        type=str,
        default="ss",
        help="Path/binary name for ss (its -K destroys the silenced replica's TCP connections).",
    )
    parser.add_argument(
        "--link-rate",
        type=str,
        default="10000mbit",
        help="HTB class rate used for created classes (default: 10000mbit).",
    )
    parser.add_argument(
        "--netem-limit",
        type=int,
        default=100000,
        help="Packet limit for each netem qdisc (default: 100000).",
    )
    parser.add_argument(
        "--burst-seed",
        type=int,
        default=1,
        help="Seed for deterministic per-edge burst placement.",
    )
    parser.add_argument(
        "--no-sudo",
        action="store_true",
        help="Do not prefix tc commands with sudo when not root.",
    )
    parser.add_argument(
        "--pre-apply-base",
        action="store_true",
        help=(
            "Install the base delay (globalNetworkDelay, with its jitter) on every peer edge "
            "at launch, before waiting for the first phase, and then update the rules in place. "
            "Start the applier before the replicas so their TCP connections are set up at the "
            "real base RTT: Linux keeps the minimum RTT seen for tcp_min_rtt_wlen (300 s), and "
            "RACK's reordering window is a quarter of it, so a connection set up on the bare "
            "veth (RTT ~0.006 ms) reads jittered long delays as loss for its first 300 s."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print tc commands without executing them.",
    )
    args = parser.parse_args()
    if args.netem_limit <= 0:
        raise ValueError("--netem-limit must be greater than zero")

    warmup_s, base_delay_ms, jitter_ratio, phases = _parse_phases(args.failure_spec)
    hosts = _load_hosts(args.hosts_config)
    if args.replica_id not in hosts:
        raise ValueError(
            f"Replica id {args.replica_id} not found in hosts config {args.hosts_config}"
        )

    sample_dest_ip = next(
        (entry.ip for rid, entry in sorted(hosts.items()) if rid != args.replica_id),
        None,
    )
    use_sudo = not args.no_sudo
    interface = _resolve_interface(
        args.interface,
        sample_dest_ip,
        use_sudo=use_sudo,
        netns=args.netns,
    )
    tc_initialized = False
    if args.pre_apply_base:
        base_rules = [
            IpRule(dst_ip=r.dst_ip, dst_replicas=r.dst_replicas,
                   fault=BurstDelay(delay_ms=0.0, burst_duration_ms=0.0, burst_interval_ms=0.0),
                   class_id=r.class_id)
            for r in _build_ip_rules(phases[0], args.replica_id, hosts)
        ]
        tc_initialized = _configure_phase_tc(
            rules=base_rules,
            base_delay_ms=base_delay_ms,
            initialized=False,
            interface=interface,
            tc_bin=args.tc_bin,
            link_rate=args.link_rate,
            netem_limit=args.netem_limit,
            jitter_ratio=jitter_ratio,
            use_sudo=use_sudo,
            dry_run=args.dry_run,
            netns=args.netns,
        )
        _log(f"Pre-applied base delay {base_delay_ms:g} ms on {interface} to {len(base_rules)} peer edge(s) before the first phase.")
    start_mono = _compute_start_mono(args.start_unix_ms)

    _log(
        "Starting network applier "
        f"(replica_id={args.replica_id}, netns={args.netns or 'host'}, "
        f"interface={interface}, warmup_s={warmup_s:.3f}, "
        f"base_delay_ms={base_delay_ms:g}, jitter_ratio={jitter_ratio:g}, phases={len(phases)})"
    )

    for idx, phase in enumerate(phases):
        target_mono = start_mono + warmup_s + phase.at_time_s
        _sleep_until(target_mono)
        next_phase_mono = None
        if idx + 1 < len(phases):
            next_phase_mono = start_mono + warmup_s + phases[idx + 1].at_time_s
        tc_initialized = _run_phase(
            phase_index=idx,
            phase=phase,
            phase_start_mono=target_mono,
            phase_end_mono=next_phase_mono,
            base_delay_ms=base_delay_ms,
            tc_initialized=tc_initialized,
            src_replica_id=args.replica_id,
            hosts=hosts,
            interface=interface,
            tc_bin=args.tc_bin,
            link_rate=args.link_rate,
            netem_limit=args.netem_limit,
            jitter_ratio=jitter_ratio,
            iptables_bin=args.iptables_bin,
            ss_bin=args.ss_bin,
            burst_seed=args.burst_seed,
            use_sudo=use_sudo,
            dry_run=args.dry_run,
            netns=args.netns,
        )

    if args.teardown_at_end:
        _log("Tearing down tc and silence rules at end.")
        _clear_tc(
            interface,
            tc_bin=args.tc_bin,
            use_sudo=use_sudo,
            dry_run=args.dry_run,
            netns=args.netns,
        )
        _clear_silence(
            iptables_bin=args.iptables_bin,
            use_sudo=use_sudo,
            dry_run=args.dry_run,
            netns=args.netns,
        )
    else:
        _log("Completed schedule; final phase remains active.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        cmd_text = " ".join(str(part) for part in exc.cmd)
        print(
            f"ERROR: command failed with exit code {exc.returncode}: {cmd_text}",
            file=sys.stderr,
        )
        if exc.stderr:
            print(f"stderr: {exc.stderr.strip()}", file=sys.stderr)
        if exc.stdout:
            print(f"stdout: {exc.stdout.strip()}", file=sys.stderr)
        raise SystemExit(1)
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
