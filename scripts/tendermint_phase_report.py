#!/usr/bin/env python3
"""Build node-0 Tendermint phase, batch-size, and failure-phase tables."""

import argparse
import bisect
import datetime
import math
import re
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path


TIMESTAMP_RE = re.compile(
    r"^[A-Z]\[(?:(\d{4}-\d{2}-\d{2})\|)?(\d{2}:\d{2}:\d{2}\.\d{3})\]"
)
NEW_HEIGHT_RE = re.compile(
    r"phase enter\s+module=consensus\s+height=(\d+)\s+round=-?\d+\s+"
    r"phase=RoundStepNewHeight\b"
)
COMMITTED_RE = re.compile(r"committed state\s+module=state\s+height=(\d+)\b")
FINALIZING_RE = re.compile(
    r"finalizing commit of block\s+module=consensus\s+height=(\d+)\b"
    r".*\bnum_txs=(\d+)\b"
)
PHASE_RE = re.compile(
    r"phase enter\s+module=consensus\s+height=(\d+)\s+round=(-?\d+)\s+"
    r"phase=(RoundStep\w+)\b"
)


def _optional_float(parent, tag, default=None):
    node = parent.find(tag)
    if node is None or node.text is None or not node.text.strip():
        return default
    return float(node.text.strip())


def load_failure_schedule(path):
    root = ET.parse(path).getroot()
    warmup_ms = _optional_float(root, "warmUpTimeMs")
    if warmup_ms is not None:
        warmup_s = max(0.0, warmup_ms / 1000.0)
    else:
        warmup_s = max(0.0, _optional_float(root, "warmUpTime", 0.0))

    phase_nodes = root.findall("./phases/phase")
    has_at_time = any(node.find("atTime") is not None for node in phase_nodes)
    phase_starts_s = []
    cumulative_s = 0.0
    for node in phase_nodes:
        if has_at_time:
            at_time_s = _optional_float(node, "atTime")
            if at_time_s is None:
                at_time_s = _optional_float(node, "time")
            if at_time_s is None:
                raise ValueError("Failure phase is missing <atTime> or <time>")
        else:
            at_time_s = cumulative_s
            duration_s = _optional_float(node, "time")
            if duration_s is None:
                raise ValueError("Legacy failure phase is missing <time>")
            if duration_s < 0:
                raise ValueError("Legacy failure phase <time> must be non-negative")
            cumulative_s += duration_s

        if at_time_s < 0:
            raise ValueError("Failure phase start time must be non-negative")
        if phase_starts_s and at_time_s < phase_starts_s[-1]:
            raise ValueError("Failure phases must be ordered by start time")
        phase_starts_s.append(at_time_s)

    return warmup_s, phase_starts_s


def parse_timestamp(line, experiment_start):
    match = TIMESTAMP_RE.search(line)
    if match is None:
        return None

    date_part, time_part = match.groups()
    if date_part:
        return datetime.datetime.strptime(
            f"{date_part} {time_part}", "%Y-%m-%d %H:%M:%S.%f"
        )

    parsed_time = datetime.datetime.strptime(time_part, "%H:%M:%S.%f").time()
    timestamp = datetime.datetime.combine(experiment_start.date(), parsed_time)
    if timestamp < experiment_start - datetime.timedelta(hours=12):
        timestamp += datetime.timedelta(days=1)
    elif timestamp > experiment_start + datetime.timedelta(hours=12):
        timestamp -= datetime.timedelta(days=1)
    return timestamp


def load_node_log(path, experiment_start):
    new_height_start = {}
    committed = {}
    batch_sizes = {}
    phase_rounds = {}

    if not path.is_file():
        return new_height_start, committed, batch_sizes, phase_rounds

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            timestamp = parse_timestamp(line, experiment_start)
            if timestamp is None:
                continue

            phase_match = PHASE_RE.search(line)
            if phase_match:
                key = (int(phase_match.group(1)), int(phase_match.group(2)))
                phase = phase_match.group(3)
                phase_rounds.setdefault(key, {}).setdefault(phase, timestamp)

            height_match = NEW_HEIGHT_RE.search(line)
            if height_match:
                new_height_start.setdefault(int(height_match.group(1)), timestamp)

            committed_match = COMMITTED_RE.search(line)
            if committed_match:
                committed.setdefault(int(committed_match.group(1)), timestamp)

            finalizing_match = FINALIZING_RE.search(line)
            if finalizing_match:
                batch_sizes.setdefault(
                    int(finalizing_match.group(1)), int(finalizing_match.group(2))
                )

    return new_height_start, committed, batch_sizes, phase_rounds


def elapsed_ms(start, end):
    if start is None or end is None:
        return None
    return max(0.0, (end - start).total_seconds() * 1000.0)


def failure_phase(elapsed_s, warmup_s, phase_starts_s):
    if elapsed_s is None or elapsed_s < warmup_s:
        return -1
    phase_time_s = elapsed_s - warmup_s
    return bisect.bisect_right(phase_starts_s, phase_time_s) - 1


def round_latencies(height, round_number, phase_rounds):
    phases = phase_rounds.get((height, round_number), {})
    propose_start = phases.get("RoundStepPropose")
    prevote_start = phases.get("RoundStepPrevote")
    precommit_start = phases.get("RoundStepPrecommit")
    commit_start = phases.get("RoundStepCommit")
    next_round_start = phase_rounds.get((height, round_number + 1), {}).get(
        "RoundStepNewRound"
    )
    round_end = commit_start if commit_start is not None else next_round_start
    return (
        elapsed_ms(propose_start, prevote_start),
        elapsed_ms(prevote_start, precommit_start),
        elapsed_ms(precommit_start, round_end),
    )


def format_number(value, digits=1):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def format_table(headers, rows):
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(row):
        return " | ".join(
            cell.ljust(widths[index]) for index, cell in enumerate(row)
        )

    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))


def percentile(values, quantile):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def metric_summary(values):
    if not values:
        return ["-", "-", "-"]
    return [
        format_number(statistics.median(values)),
        format_number(percentile(values, 0.90)),
        format_number(max(values)),
    ]


def complete_height_phase_totals(height, phase_rounds):
    rounds = sorted(
        round_number
        for candidate_height, round_number in phase_rounds
        if candidate_height == height
        and "RoundStepPropose" in phase_rounds[(candidate_height, round_number)]
    )
    if not rounds:
        return None, None, None

    columns = [[], [], []]
    for round_number in rounds:
        values = round_latencies(height, round_number, phase_rounds)
        for index, value in enumerate(values):
            columns[index].append(value)

    totals = []
    for values in columns:
        totals.append(None if any(value is None for value in values) else sum(values))
    return tuple(totals)


def print_report(logs_dir, failure_spec, start_unix_ms):
    experiment_start = datetime.datetime.fromtimestamp(start_unix_ms / 1000.0)
    warmup_s, phase_starts_s = load_failure_schedule(failure_spec)
    new_height_start, committed, batch_sizes, phase_rounds = load_node_log(
        logs_dir / "node_0.log", experiment_start
    )

    round_keys = sorted(
        key for key, phases in phase_rounds.items() if "RoundStepPropose" in phases
    )
    detail_headers = [
        "height",
        "round",
        "time_from_start_n0 (s)",
        "failure_phase",
        "batch_size",
        "propose_ms_n0",
        "prevote_ms_n0",
        "precommit_ms_n0",
    ]
    detail_rows = []
    for height, round_number in round_keys:
        height_start = new_height_start.get(height)
        elapsed_s = (
            None
            if height_start is None
            else (height_start - experiment_start).total_seconds()
        )
        phase = (
            "-"
            if elapsed_s is None
            else str(failure_phase(elapsed_s, warmup_s, phase_starts_s))
        )
        propose_ms, prevote_ms, precommit_ms = round_latencies(
            height, round_number, phase_rounds
        )
        committed_in_round = (
            "RoundStepCommit" in phase_rounds.get((height, round_number), {})
        )
        batch_size = batch_sizes.get(height) if committed_in_round else None
        detail_rows.append(
            [
                str(height),
                str(round_number),
                format_number(elapsed_s, digits=3),
                phase,
                "-" if batch_size is None else str(batch_size),
                format_number(propose_ms),
                format_number(prevote_ms),
                format_number(precommit_ms),
            ]
        )

    print("==> Node 0 Tendermint height/round latency:")
    if detail_rows:
        format_table(detail_headers, detail_rows)
    else:
        print("  no node 0 phase transitions found")
    print("")

    phase_data = {
        phase: {
            "heights": [],
            "latency": [],
            "propose": [],
            "prevote": [],
            "precommit": [],
            "batch_size": [],
        }
        for phase in range(-1, len(phase_starts_s))
    }
    for height, commit_timestamp in sorted(committed.items()):
        height_start = new_height_start.get(height)
        if height_start is None:
            continue
        elapsed_s = (height_start - experiment_start).total_seconds()
        phase = failure_phase(elapsed_s, warmup_s, phase_starts_s)
        data = phase_data.setdefault(
            phase,
            {
                "heights": [],
                "latency": [],
                "propose": [],
                "prevote": [],
                "precommit": [],
                "batch_size": [],
            },
        )
        data["heights"].append(height)
        data["latency"].append(elapsed_ms(height_start, commit_timestamp))

        propose_ms, prevote_ms, precommit_ms = complete_height_phase_totals(
            height, phase_rounds
        )
        if propose_ms is not None:
            data["propose"].append(propose_ms)
        if prevote_ms is not None:
            data["prevote"].append(prevote_ms)
        if precommit_ms is not None:
            data["precommit"].append(precommit_ms)
        batch_size = batch_sizes.get(height)
        if batch_size is not None:
            data["batch_size"].append(batch_size)

    summary_headers = [
        "failure_phase",
        "height_range",
        "number_of_heights",
        "median_batch_size",
        "p90_batch_size",
        "max_batch_size",
        "median_latency",
        "p90_latency",
        "max_latency",
        "median_proposal_latency",
        "p90_proposal_latency",
        "max_proposal_latency",
        "median_prevote_latency",
        "p90_prevote_latency",
        "max_prevote_latency",
        "median_precommit_latency",
        "p90_precommit_latency",
        "max_precommit_latency",
    ]
    summary_rows = []
    for phase in sorted(phase_data):
        data = phase_data[phase]
        heights = data["heights"]
        if not heights:
            height_range = "-"
        elif heights[0] == heights[-1]:
            height_range = str(heights[0])
        else:
            height_range = f"{heights[0]}-{heights[-1]}"
        summary_rows.append(
            [str(phase), height_range, str(len(heights))]
            + metric_summary(data["batch_size"])
            + metric_summary(data["latency"])
            + metric_summary(data["propose"])
            + metric_summary(data["prevote"])
            + metric_summary(data["precommit"])
        )

    print("==> Tendermint latency by failure phase (latencies in ms):")
    format_table(summary_headers, summary_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--failure-spec", type=Path, required=True)
    parser.add_argument("--start-unix-ms", type=int, required=True)
    args = parser.parse_args()
    print_report(args.logs_dir, args.failure_spec, args.start_unix_ms)


if __name__ == "__main__":
    main()
