#!/usr/bin/env python3
"""Apply phase-scheduled network delay/loss faults using tc netem.

This applier intentionally handles only the <network> section of
config/bft-smart/failure_spec.xml and ignores PBFT/agent fault sections.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class DelayLoss:
    delay_ms: float
    loss_pct: float


@dataclass
class PhaseNetworkSpec:
    at_time_s: float
    directed_edges: bool
    global_fault: DelayLoss
    source_faults: Dict[int, DelayLoss]
    edge_faults: Dict[Tuple[int, int], DelayLoss]


@dataclass(frozen=True)
class HostEntry:
    replica_id: int
    ip: str


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


def _parse_delay_loss(node: ET.Element) -> DelayLoss:
    delay_ms = _parse_required_float(node, "delayMs")
    loss_pct = _parse_required_float(node, "lossPct")
    if delay_ms < 0:
        raise ValueError("delayMs must be >= 0")
    if loss_pct < 0:
        raise ValueError("lossPct must be >= 0")
    return DelayLoss(delay_ms=delay_ms, loss_pct=loss_pct)


def _parse_warmup_seconds(root: ET.Element) -> float:
    warmup_ms = _parse_optional_int(root, "warmUpTimeMs")
    if warmup_ms is not None:
        return max(0.0, warmup_ms / 1000.0)

    warmup_s = _parse_optional_float(root, "warmUpTime")
    if warmup_s is None:
        return 0.0
    return max(0.0, warmup_s)


def _parse_phases(spec_path: Path) -> Tuple[float, List[PhaseNetworkSpec]]:
    root = ET.parse(spec_path).getroot()
    warmup_s = _parse_warmup_seconds(root)
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
                    global_fault=DelayLoss(delay_ms=0.0, loss_pct=0.0),
                    source_faults={},
                    edge_faults={},
                )
            )
            continue

        directed_edges = _parse_bool(network.findtext("directedEdges"), default=True)

        global_node = network.find("global")
        global_fault = (
            _parse_delay_loss(global_node)
            if global_node is not None
            else DelayLoss(delay_ms=0.0, loss_pct=0.0)
        )

        source_faults: Dict[int, DelayLoss] = {}
        for source_node in network.findall("./sources/source"):
            src = _parse_required_int(source_node, "src")
            source_faults[src] = _parse_delay_loss(source_node)

        edge_faults: Dict[Tuple[int, int], DelayLoss] = {}
        for edge_node in network.findall("./edges/edge"):
            src = _parse_required_int(edge_node, "src")
            dst = _parse_required_int(edge_node, "dst")
            edge_faults[(src, dst)] = _parse_delay_loss(edge_node)

        phases.append(
            PhaseNetworkSpec(
                at_time_s=at_time_s,
                directed_edges=directed_edges,
                global_fault=global_fault,
                source_faults=source_faults,
                edge_faults=edge_faults,
            )
        )

        if has_at_time and index > 0 and at_time_s < phases[index - 1].at_time_s:
            raise ValueError("Phases must be non-decreasing by atTime")

    if not has_at_time:
        _log("No <atTime> found; treating <time> as phase duration (legacy behavior).")

    return warmup_s, phases


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


def _choose_fault(phase: PhaseNetworkSpec, src: int, dst: int) -> DelayLoss:
    fault = phase.global_fault
    if src in phase.source_faults:
        fault = phase.source_faults[src]
    if (src, dst) in phase.edge_faults:
        return phase.edge_faults[(src, dst)]
    if not phase.directed_edges and (dst, src) in phase.edge_faults:
        return phase.edge_faults[(dst, src)]
    return fault


def _build_ip_faults(
    phase: PhaseNetworkSpec, src_replica_id: int, hosts: Dict[int, HostEntry]
) -> Dict[str, DelayLoss]:
    ip_faults: Dict[str, DelayLoss] = {}
    for dst_replica_id, entry in sorted(hosts.items()):
        if dst_replica_id == src_replica_id:
            continue
        fault = _choose_fault(phase, src_replica_id, dst_replica_id)
        existing = ip_faults.get(entry.ip)
        if existing is None:
            ip_faults[entry.ip] = fault
            continue
        if existing != fault:
            raise ValueError(
                "Cannot express different per-replica faults for replicas sharing "
                f"the same destination IP ({entry.ip}) without namespaces/port rules."
            )
    return ip_faults


def _apply_phase(
    *,
    phase_index: int,
    phase: PhaseNetworkSpec,
    src_replica_id: int,
    hosts: Dict[int, HostEntry],
    interface: str,
    tc_bin: str,
    link_rate: str,
    use_sudo: bool,
    dry_run: bool,
    netns: str | None,
) -> None:
    ip_faults = _build_ip_faults(phase, src_replica_id, hosts)
    _log(
        f"Applying phase #{phase_index} at t={phase.at_time_s:.3f}s "
        f"on {interface}; rules={len(ip_faults)}"
    )

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

    for rule_idx, (dst_ip, fault) in enumerate(sorted(ip_faults.items()), start=1):
        class_id = 100 + rule_idx
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
                f"1:{class_id}",
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
        _run_command(
            [
                tc_bin,
                "qdisc",
                "replace",
                "dev",
                interface,
                "parent",
                f"1:{class_id}",
                "handle",
                f"{class_id}:",
                "netem",
                "delay",
                f"{fault.delay_ms:g}ms",
                "loss",
                f"{fault.loss_pct:g}%",
            ],
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
                "1",
                "u32",
                "match",
                "ip",
                "dst",
                f"{dst_ip}/32",
                "flowid",
                f"1:{class_id}",
            ],
            use_sudo=use_sudo,
            dry_run=dry_run,
            netns=netns,
        )


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
        description="Apply network delay/loss phases from failure_spec.xml using tc."
    )
    parser.add_argument(
        "--failure-spec",
        type=Path,
        default=Path("config/bft-smart/failure_spec.xml"),
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
        "--link-rate",
        type=str,
        default="10000mbit",
        help="HTB class rate used for created classes (default: 10000mbit).",
    )
    parser.add_argument(
        "--no-sudo",
        action="store_true",
        help="Do not prefix tc commands with sudo when not root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print tc commands without executing them.",
    )
    args = parser.parse_args()

    warmup_s, phases = _parse_phases(args.failure_spec)
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
    start_mono = _compute_start_mono(args.start_unix_ms)

    _log(
        "Starting network applier "
        f"(replica_id={args.replica_id}, netns={args.netns or 'host'}, "
        f"interface={interface}, warmup_s={warmup_s:.3f}, phases={len(phases)})"
    )

    for idx, phase in enumerate(phases):
        target_mono = start_mono + warmup_s + phase.at_time_s
        sleep_s = target_mono - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)
        _apply_phase(
            phase_index=idx,
            phase=phase,
            src_replica_id=args.replica_id,
            hosts=hosts,
            interface=interface,
            tc_bin=args.tc_bin,
            link_rate=args.link_rate,
            use_sudo=use_sudo,
            dry_run=args.dry_run,
            netns=args.netns,
        )

    if args.teardown_at_end:
        _log("Tearing down tc rules at end.")
        _clear_tc(
            interface,
            tc_bin=args.tc_bin,
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
