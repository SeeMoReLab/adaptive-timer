from abc import ABC, abstractmethod
import copy
import csv
import math
from dataclasses import dataclass, field
from enum import IntEnum
import time
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar
from sklearn.tree import DecisionTreeRegressor
from collections import deque
import numpy as np

import agent_pb2


# Model-type names retired on 2026-10-07, mapped to their replacements. The
# agent and the run scripts reject an old name with this hint instead of
# accepting it.
RENAMED_MODEL_TYPES: Dict[str, str] = {
    "random": "discrete_random",
    "pavlov": "discrete_pavlov",
    "linear": "discrete_linear",
    "cmab": "discrete_tree_ts",
    "thompson": "discrete_tree_ts",
    "cmab_reset": "discrete_tree_blocks",
    "codex_v1": "discrete_linucb",
    "codex": "discrete_linucb",
    "ppo": "discrete_ppo",
    "continuous": "continuous_gp",
    "continuous_pretrained_gp": "continuous_gp",
    "continuous_codex": "continuous_linucb",
    "latency_floor_heuristic": "continuous_heuristic",
    "continuous_latency_floor": "continuous_heuristic",
}


def reject_renamed_model_type(value: str) -> str:
    """argparse ``type`` for a model-type flag: refuses a retired name."""
    if value in RENAMED_MODEL_TYPES:
        import argparse

        raise argparse.ArgumentTypeError(
            f"model type {value!r} was renamed to {RENAMED_MODEL_TYPES[value]!r}"
        )
    return value


S = TypeVar("S")  # generic state type
A = TypeVar("A")  # generic action type


class PbftTimeoutOption(IntEnum):
    # PBFT (CppBedrock) ladder, Kevin 2026-09-29 23:41: 250/550/1200.
    TIMEOUT_250_MS = 250
    TIMEOUT_550_MS = 550
    TIMEOUT_1200_MS = 1200


PBFT_TIMEOUT_OPTIONS: Tuple[PbftTimeoutOption, ...] = (
    PbftTimeoutOption.TIMEOUT_250_MS,
    PbftTimeoutOption.TIMEOUT_550_MS,
    PbftTimeoutOption.TIMEOUT_1200_MS,
)
PBFT_TIMEOUT_VALUES_MS: Tuple[int, ...] = tuple(
    int(option) for option in PBFT_TIMEOUT_OPTIONS
)
PBFT_TIMEOUT_MS_TO_INDEX: Dict[int, int] = {
    value_ms: idx for idx, value_ms in enumerate(PBFT_TIMEOUT_VALUES_MS)
}


def set_pbft_timeout_ladder(values_ms: Sequence[int]) -> None:
    """Replace the PBFT ladder for this process (main.py and pretrain.py
    --timeout-ladder; runs/2026-10-18-pbft-ladder-variants). Every PBFT
    accessor below reads these globals at call time, so the discrete models,
    the codecs and the prior metadata all follow; call it before any model is
    built. Rungs must be distinct, ascending and inside the PBFT context
    bounds (100-3000 ms) that discrete_gp normalizes over."""
    global PBFT_TIMEOUT_VALUES_MS, PBFT_TIMEOUT_MS_TO_INDEX
    values = tuple(int(v) for v in values_ms)
    if len(values) < 2:
        raise ValueError(f"a PBFT ladder needs at least two rungs, got {values}")
    if list(values) != sorted(set(values)):
        raise ValueError(f"PBFT ladder rungs must be distinct and ascending, got {values}")
    if values[0] < 100 or values[-1] > 3000:
        raise ValueError(f"PBFT ladder rungs must lie in 100..3000 ms, got {values}")
    PBFT_TIMEOUT_VALUES_MS = values
    PBFT_TIMEOUT_MS_TO_INDEX = {value_ms: idx for idx, value_ms in enumerate(values)}


class SmartbftTimeoutOption(IntEnum):
    # SmartBFT keeps its own ladder, Kevin 2026-09-29 23:41: 300/800/1800
    # (the schedule and the 2026-09-12 dataset use it).
    TIMEOUT_300_MS = 300
    TIMEOUT_800_MS = 800
    TIMEOUT_1800_MS = 1800


SMARTBFT_TIMEOUT_OPTIONS: Tuple[SmartbftTimeoutOption, ...] = (
    SmartbftTimeoutOption.TIMEOUT_300_MS,
    SmartbftTimeoutOption.TIMEOUT_800_MS,
    SmartbftTimeoutOption.TIMEOUT_1800_MS,
)
SMARTBFT_TIMEOUT_VALUES_MS: Tuple[int, ...] = tuple(
    int(option) for option in SMARTBFT_TIMEOUT_OPTIONS
)
SMARTBFT_TIMEOUT_MS_TO_INDEX: Dict[int, int] = {
    value_ms: idx for idx, value_ms in enumerate(SMARTBFT_TIMEOUT_VALUES_MS)
}


class TimeoutOption(IntEnum):
    # One rung per link level (base / 5+50 / 5+150). Each rung clears its own
    # level's fast-path floor (150 / 275 / 620) and is punished one level down by
    # the slow-leader hold it tolerates (D150 at base, D550 at 5+50) and one
    # level up by the floor at the +150 onset. 525/1100 -> 550/1200 on
    # 2026-10-13 (Kevin, runs/2026-10-12-sbft-margins at 20,000 req/s); the
    # PBFT ladder uses the same values. 550/1200 -> 650/1600 on 2026-10-17 (Kevin,
    # runs/2026-10-16-sbft-stability, the adopted oracle's rungs): 550 was 1.0x p3's
    # slow-path bound (125 + 5x55 + 150), under the 1.1x rule; at 1300 healthy +150
    # ages reached the deadline and the straggler elected alone. PBFT keeps 550/1200.
    TIMEOUT_250_MS = 250
    TIMEOUT_650_MS = 650
    TIMEOUT_1600_MS = 1600

TIMEOUT_OPTIONS: Tuple[TimeoutOption, ...] = (
    TimeoutOption.TIMEOUT_250_MS,
    TimeoutOption.TIMEOUT_650_MS,
    TimeoutOption.TIMEOUT_1600_MS,
)
TIMEOUT_VALUES_MS: Tuple[int, ...] = tuple(int(option) for option in TIMEOUT_OPTIONS)
# The slow-path wait needs only two values, not the three the election timeout
# needs. Its cost is `S` when the wait outlasts the straggler's lag and
# `W + R` when it does not, so the best value is always either the floor -
# give up at once and pay the extra round - or the smallest value above `S`.
# A third value earns a place only when two different lags both fall inside
# the window (floor, floor + R), and R, the extra round, is 2 * one_way + 25
# ms, so that window is 39 ms wide at base delay. See adaptive_timers.MD.
# The floor is 10 since 2026-10-17 (Kevin, runs/2026-10-16-sbft-stability): with no
# straggler the n-th prepare lands 1-2 ms after the 2f+1-th, and where the slow path
# wins giving up at once is 40 ms faster than at 50.
class SbftSlowPathOption(IntEnum):
    TIMEOUT_10_MS = 10
    TIMEOUT_150_MS = 150

SBFT_SLOW_PATH_OPTIONS: Tuple[SbftSlowPathOption, ...] = (
    SbftSlowPathOption.TIMEOUT_10_MS,
    SbftSlowPathOption.TIMEOUT_150_MS,
)
SBFT_SLOW_PATH_VALUES_MS: Tuple[int, ...] = tuple(
    int(option) for option in SBFT_SLOW_PATH_OPTIONS
)
SBFT_SLOW_PATH_MS_TO_INDEX: Dict[int, int] = {
    value_ms: idx for idx, value_ms in enumerate(SBFT_SLOW_PATH_VALUES_MS)
}

TIMEOUT_MS_TO_INDEX: Dict[int, int] = {
    value_ms: idx for idx, value_ms in enumerate(TIMEOUT_VALUES_MS)
}


class TendermintProposeTimeoutOption(IntEnum):
    # One arm per regime of config/tendermint/failure_spec.xml (Kevin 2026-10-10;
    # constant-T runs of the compressed spec, scripts/cloudlab/tmp/cal/blindall/
    # tendermint/real_c120.xml, 2 draws, client judge). Only the propose timeout
    # binds in a healthy round: prevote and precommit (T/3) start only on a split
    # vote, and the propose step takes about two one-way trips (108 ms at +50,
    # 280 ms at +150).
    #   100 - proposal delay: the fastest escape from a withheld proposal.
    #   200 - +50 ms links: no missed proposals (150 misses 1.6 a window).
    #   400 - +150 ms links: 0.3 missed a window (300 misses 5, and collapses).
    # Replaced 150 / 450 / 750 (2026-09-29).
    TIMEOUT_100_MS = 100
    TIMEOUT_200_MS = 200
    TIMEOUT_400_MS = 400


TENDERMINT_PROPOSE_TIMEOUT_OPTIONS: Tuple[TendermintProposeTimeoutOption, ...] = (
    TendermintProposeTimeoutOption.TIMEOUT_100_MS,
    TendermintProposeTimeoutOption.TIMEOUT_200_MS,
    TendermintProposeTimeoutOption.TIMEOUT_400_MS,
)
TENDERMINT_PROPOSE_TIMEOUT_VALUES_MS: Tuple[int, ...] = tuple(
    int(option) for option in TENDERMINT_PROPOSE_TIMEOUT_OPTIONS
)
TENDERMINT_PROPOSE_TIMEOUT_MS_TO_INDEX: Dict[int, int] = {
    value_ms: idx
    for idx, value_ms in enumerate(TENDERMINT_PROPOSE_TIMEOUT_VALUES_MS)
}


class HotstuffTimeoutOption(IntEnum):
    # One value per link level of config/hotstuff/failure_spec.xml, each the best
    # (or tied best) there and failing one level up: 100 at base, 300 at one-way
    # 5+50 (floor about 180: 170 still locks into 3-hop rounds), 700 at 5+150
    # (floor between 460, which storms, and 530). Under the slow-leader attack a
    # lower T replaces the leader sooner: constant-T cells (client R over t > 60
    # s, 2026-10-08) base + slow leader T100 0.798 vs T160 0.761, 5+50 + slow
    # leader T300 0.469 vs T460 0.398; the no-attack phases tie. Replaced
    # 160 / 460 / 760 (2026-10-02) on 2026-10-08 (Kevin).
    TIMEOUT_100_MS = 100
    TIMEOUT_300_MS = 300
    TIMEOUT_700_MS = 700


HOTSTUFF_TIMEOUT_OPTIONS: Tuple[HotstuffTimeoutOption, ...] = (
    HotstuffTimeoutOption.TIMEOUT_100_MS,
    HotstuffTimeoutOption.TIMEOUT_300_MS,
    HotstuffTimeoutOption.TIMEOUT_700_MS,
)
HOTSTUFF_TIMEOUT_VALUES_MS: Tuple[int, ...] = tuple(
    int(option) for option in HOTSTUFF_TIMEOUT_OPTIONS
)
HOTSTUFF_TIMEOUT_MS_TO_INDEX: Dict[int, int] = {
    value_ms: idx for idx, value_ms in enumerate(HOTSTUFF_TIMEOUT_VALUES_MS)
}


class AutobahnTimeoutDelayOption(IntEnum):
    # The only Autobahn action (the fast-path wait is fixed in
    # config/autobahn/parameters.json). Healthy Autobahn has no floor; the
    # floors come from replica 3 withholding its votes, so every quorum needs
    # all three honest replicas and a premature view change starves it. They
    # are soft (about 400 at one-way 5+50, about 1000 at 5+150). One value per
    # link level: 250 at base, 500 at 5+50, 1250 at 5+150. Constant-T cells
    # (client R over t > 60 s, 2026-10-08): 5+50 T250 -0.164 (storms), T400
    # 0.260, T500 0.247, T750 0.176; with the slow leader -0.146 / 0.259 / 0.249 /
    # 0.169; 5+150 T750 storms, T1000 -0.835 (a 3 s stall), T1250 -0.873. 500
    # replaced 750 on 2026-10-08 (Kevin), within noise of 400.
    TIMEOUT_250_MS = 250
    TIMEOUT_500_MS = 500
    TIMEOUT_1250_MS = 1250


AUTOBAHN_TIMEOUT_DELAY_OPTIONS: Tuple[AutobahnTimeoutDelayOption, ...] = (
    AutobahnTimeoutDelayOption.TIMEOUT_250_MS,
    AutobahnTimeoutDelayOption.TIMEOUT_500_MS,
    AutobahnTimeoutDelayOption.TIMEOUT_1250_MS,
)
AUTOBAHN_TIMEOUT_DELAY_VALUES_MS: Tuple[int, ...] = tuple(
    int(option) for option in AUTOBAHN_TIMEOUT_DELAY_OPTIONS
)
AUTOBAHN_TIMEOUT_DELAY_MS_TO_INDEX: Dict[int, int] = {
    value_ms: idx for idx, value_ms in enumerate(AUTOBAHN_TIMEOUT_DELAY_VALUES_MS)
}


# The *_REPORT_FEATURE_FIELD_NAMES tuples are every model's input. None holds
# the applied timeout (timeout_ms, SBFT's slow_path_timeout_ms): as an input, a
# model's own last pick fed its next one (H15, the GP's two-cycle; Kevin,
# 2026-10-19, restart D9a). The dataset, the reward and the action decoding read
# it from the report itself, the SBFT heuristic from its own held action.
_BASE_REPORT_FEATURE_FIELD_NAMES: Tuple[str, ...] = (
    "total_transactions",
    "total_consensus_instances",
    "avg_consensus_latency_ms",
    "p50_consensus_latency_ms",
    "p95_consensus_latency_ms",
    "throughput_tps",
    "avg_batch_size",
    "p95_batch_size",
    "leader_change_count",
    "regency_change_count",
    "avg_inter_commit_gap_ms",
    "p50_inter_commit_gap_ms",
    "p95_inter_commit_gap_ms",
    "view_change_count",
    "no_progress_view_change_count",
)
# The 15 base fields plus the phase latencies (window means, proto fields
# 17-19): a leader's proposal hold precedes the pre-prepare, so it enters the
# consensus latency but none of the phases, and a model that reads both can
# tell a held leader from a slow link (Kevin, 2026-10-14,
# runs/2026-10-14-five-machine-models: the pretrained GP on the base fields
# alone tolerated every hold).
PBFT_REPORT_FEATURE_FIELD_NAMES: Tuple[str, ...] = _BASE_REPORT_FEATURE_FIELD_NAMES + (
    "pre_prepare_latency_ms",
    "prepare_latency_ms",
    "commit_latency_ms",
)
# SmartBFT exports no per-phase stamps; its one phase field is proposal arrival
# (a follower's VerifyProposal) -> delivery, PBFT's three phases together, for
# the same reason (Kevin, 2026-10-18: at T 1800 the pretrained discrete GP read
# the p4 hold, +50 with D300, exactly as the p5 +150 link, every base field
# equal, and kept 1800 through p4; a26_discrete_gp_pre_setA_20261016_072711).
SMARTBFT_REPORT_FEATURE_FIELD_NAMES: Tuple[str, ...] = _BASE_REPORT_FEATURE_FIELD_NAMES + (
    "proposal_to_decision_ms",
)
# SbftReport shares PbftReport's 16-field layout and adds the SBFT-specific
# phase latencies and slow-path context (see proto/agent.proto), less the
# applied wait (D9a, above). The n-th prepare fields are measured by the leader
# only (aggregator.py takes them from the report that led the most sequences).
SBFT_REPORT_FEATURE_FIELD_NAMES: Tuple[str, ...] = _BASE_REPORT_FEATURE_FIELD_NAMES + (
    "pre_prepare_latency_ms",
    "prepare_latency_ms",
    "commit_latency_ms",
    "fast_path_count",
    "slow_path_count",
    "nth_prepare_gap_ms",
    "nth_prepare_seen_count",
    "nth_prepare_unseen_count",
)
# The base fields' latency, batch and inter-commit-gap statistics (p95, not the
# p90 the list read until 2026-10-19), the violation rate in place of the view
# change counts, and the three step latencies (Kevin, 2026-10-19, restart: one
# report shape on all six).
TENDERMINT_REPORT_FEATURE_FIELD_NAMES: Tuple[str, ...] = (
    "total_transactions",
    "total_consensus_instances",
    "avg_consensus_latency_ms",
    "p50_consensus_latency_ms",
    "p95_consensus_latency_ms",
    "throughput_tps",
    "timeout_violation_rate",
    "avg_batch_size",
    "p50_batch_size",
    "p95_batch_size",
    "leader_change_count",
    "regency_change_count",
    "propose_latency_ms",
    "prevote_latency_ms",
    "precommit_latency_ms",
    "avg_inter_commit_gap_ms",
    "p50_inter_commit_gap_ms",
    "p95_inter_commit_gap_ms",
)
HOTSTUFF_REPORT_FEATURE_FIELD_NAMES: Tuple[str, ...] = (
    "total_transactions",
    "total_consensus_instances",
    "avg_consensus_latency_ms",
    "p50_consensus_latency_ms",
    "p95_consensus_latency_ms",
    "throughput_tps",
    "avg_batch_size",
    "p95_batch_size",
    "leader_change_count",
    "regency_change_count",
    "avg_inter_commit_gap_ms",
    "p50_inter_commit_gap_ms",
    "p95_inter_commit_gap_ms",
    "view_change_count",
    "no_progress_view_change_count",
    # The next leader's block receipt to its QC: a proposer's hold never enters it,
    # so it separates a slow leader from a slow link (2026-10-19, as SmartBFT's
    # proposal_to_decision_ms).
    "proposal_to_qc_ms",
)
# AutobahnReport has HotstuffReport's layout without regency_change_count.
AUTOBAHN_REPORT_FEATURE_FIELD_NAMES: Tuple[str, ...] = (
    "total_transactions",
    "total_consensus_instances",
    "avg_consensus_latency_ms",
    "p50_consensus_latency_ms",
    "p95_consensus_latency_ms",
    "throughput_tps",
    "avg_batch_size",
    "p95_batch_size",
    "leader_change_count",
    "avg_inter_commit_gap_ms",
    "p50_inter_commit_gap_ms",
    "p95_inter_commit_gap_ms",
    "view_change_count",
    "no_progress_view_change_count",
    # Prepare receipt to slot commit: the leader's hold never enters it (2026-10-19).
    "prepare_to_commit_ms",
)


def _protocol_name(protocol: int) -> str:
    try:
        return agent_pb2.Protocol.Name(protocol)
    except ValueError:
        return f"UNKNOWN({protocol})"


def _validate_protocol(protocol: int) -> int:
    if protocol not in (
        agent_pb2.PROTOCOL_PBFT,
        agent_pb2.PROTOCOL_SBFT,
        agent_pb2.PROTOCOL_TENDERMINT,
        agent_pb2.PROTOCOL_HOTSTUFF,
        agent_pb2.PROTOCOL_AUTOBAHN,
        agent_pb2.PROTOCOL_SMARTBFT,
    ):
        raise ValueError(f"Unsupported protocol: {_protocol_name(protocol)}")
    return int(protocol)


def get_report_feature_field_names(protocol: int) -> Tuple[str, ...]:
    protocol = _validate_protocol(protocol)
    if protocol == agent_pb2.PROTOCOL_PBFT:
        return PBFT_REPORT_FEATURE_FIELD_NAMES
    if protocol == agent_pb2.PROTOCOL_SMARTBFT:
        return SMARTBFT_REPORT_FEATURE_FIELD_NAMES
    if protocol == agent_pb2.PROTOCOL_SBFT:
        return SBFT_REPORT_FEATURE_FIELD_NAMES
    if protocol == agent_pb2.PROTOCOL_HOTSTUFF:
        return HOTSTUFF_REPORT_FEATURE_FIELD_NAMES
    if protocol == agent_pb2.PROTOCOL_AUTOBAHN:
        return AUTOBAHN_REPORT_FEATURE_FIELD_NAMES
    return TENDERMINT_REPORT_FEATURE_FIELD_NAMES


def get_report_feature_dim(protocol: int) -> int:
    return len(get_report_feature_field_names(protocol))


def report_message_class(protocol: int) -> type:
    protocol = _validate_protocol(protocol)
    if protocol == agent_pb2.PROTOCOL_PBFT:
        return agent_pb2.PbftReport
    if protocol == agent_pb2.PROTOCOL_SBFT:
        return agent_pb2.SbftReport
    if protocol == agent_pb2.PROTOCOL_TENDERMINT:
        return agent_pb2.TendermintReport
    if protocol == agent_pb2.PROTOCOL_HOTSTUFF:
        return agent_pb2.HotstuffReport
    if protocol == agent_pb2.PROTOCOL_AUTOBAHN:
        return agent_pb2.AutobahnReport
    if protocol == agent_pb2.PROTOCOL_SMARTBFT:
        return agent_pb2.SmartbftReport
    raise ValueError(f"Unsupported protocol: {protocol}")


def get_report_field_names(protocol: int) -> Tuple[str, ...]:
    """Every field of the protocol's report message, in field-number order.

    The dataset columns and the sharing-mode median use this, so a report field
    added to proto/agent.proto reaches both with no hand list. The model inputs
    are the *_REPORT_FEATURE_FIELD_NAMES tuples above, a subset of it without
    the applied timeouts.
    """
    fields = report_message_class(protocol).DESCRIPTOR.fields
    return tuple(f.name for f in sorted(fields, key=lambda f: f.number))


def get_timeout_values_ms(protocol: int) -> Tuple[int, ...]:
    protocol = _validate_protocol(protocol)
    if protocol == agent_pb2.PROTOCOL_PBFT:
        return PBFT_TIMEOUT_VALUES_MS
    if protocol == agent_pb2.PROTOCOL_SMARTBFT:
        return SMARTBFT_TIMEOUT_VALUES_MS
    if protocol == agent_pb2.PROTOCOL_TENDERMINT:
        return TENDERMINT_PROPOSE_TIMEOUT_VALUES_MS
    if protocol == agent_pb2.PROTOCOL_HOTSTUFF:
        return HOTSTUFF_TIMEOUT_VALUES_MS
    if protocol == agent_pb2.PROTOCOL_AUTOBAHN:
        return AUTOBAHN_TIMEOUT_DELAY_VALUES_MS
    return TIMEOUT_VALUES_MS


def get_arm_timeout_values_ms(protocol: int) -> Tuple[int, ...]:
    """The election timeout of each action index, one entry per arm.

    ``get_timeout_values_ms`` returns the distinct ladder, which is the same
    thing for every protocol whose action is one timeout. SBFT's action is the
    (election timeout, fast-path wait) pair, so its ladder is 3 long while its
    action space is 6; anything indexing a per-arm structure by action index -
    ``LatencyFloorPolicy``'s feasibility mask above all - needs this instead,
    or it silently truncates the arm set to the first ``len(ladder)`` arms and
    makes the long timeouts unplayable.
    """
    protocol = _validate_protocol(protocol)
    if protocol == agent_pb2.PROTOCOL_SBFT:
        return tuple(
            _sbft_index_to_pair(index)[0]
            for index in range(get_action_space_size(protocol))
        )
    return get_timeout_values_ms(protocol)


def timeout_ms_to_index(timeout_ms: int) -> int:
    try:
        return TIMEOUT_MS_TO_INDEX[timeout_ms]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported timeout {timeout_ms}ms. Valid options: {TIMEOUT_VALUES_MS}"
        ) from exc


def timeout_index_to_ms(index: int) -> int:
    if index < 0:
        return TIMEOUT_VALUES_MS[0]
    if index >= len(TIMEOUT_VALUES_MS):
        return TIMEOUT_VALUES_MS[-1]
    return TIMEOUT_VALUES_MS[index]


def pbft_timeout_ms_to_index(timeout_ms: int) -> int:
    try:
        return PBFT_TIMEOUT_MS_TO_INDEX[timeout_ms]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported PBFT timeout {timeout_ms}ms. "
            f"Valid options: {PBFT_TIMEOUT_VALUES_MS}"
        ) from exc


def pbft_timeout_index_to_ms(index: int) -> int:
    if index < 0:
        return PBFT_TIMEOUT_VALUES_MS[0]
    if index >= len(PBFT_TIMEOUT_VALUES_MS):
        return PBFT_TIMEOUT_VALUES_MS[-1]
    return PBFT_TIMEOUT_VALUES_MS[index]


def smartbft_timeout_ms_to_index(timeout_ms: int) -> int:
    try:
        return SMARTBFT_TIMEOUT_MS_TO_INDEX[timeout_ms]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported SmartBFT timeout {timeout_ms}ms. "
            f"Valid options: {SMARTBFT_TIMEOUT_VALUES_MS}"
        ) from exc


def smartbft_timeout_index_to_ms(index: int) -> int:
    if index < 0:
        return SMARTBFT_TIMEOUT_VALUES_MS[0]
    if index >= len(SMARTBFT_TIMEOUT_VALUES_MS):
        return SMARTBFT_TIMEOUT_VALUES_MS[-1]
    return SMARTBFT_TIMEOUT_VALUES_MS[index]


def hotstuff_timeout_ms_to_index(timeout_ms: int) -> int:
    try:
        return HOTSTUFF_TIMEOUT_MS_TO_INDEX[timeout_ms]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported HotStuff timeout {timeout_ms}ms. "
            f"Valid options: {HOTSTUFF_TIMEOUT_VALUES_MS}"
        ) from exc


def hotstuff_timeout_index_to_ms(index: int) -> int:
    if index < 0:
        return HOTSTUFF_TIMEOUT_VALUES_MS[0]
    if index >= len(HOTSTUFF_TIMEOUT_VALUES_MS):
        return HOTSTUFF_TIMEOUT_VALUES_MS[-1]
    return HOTSTUFF_TIMEOUT_VALUES_MS[index]


def autobahn_timeout_delay_ms_to_index(timeout_ms: int) -> int:
    try:
        return AUTOBAHN_TIMEOUT_DELAY_MS_TO_INDEX[timeout_ms]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported Autobahn timeout delay {timeout_ms}ms. "
            f"Valid options: {AUTOBAHN_TIMEOUT_DELAY_VALUES_MS}"
        ) from exc


def autobahn_timeout_delay_index_to_ms(index: int) -> int:
    if index < 0:
        return AUTOBAHN_TIMEOUT_DELAY_VALUES_MS[0]
    if index >= len(AUTOBAHN_TIMEOUT_DELAY_VALUES_MS):
        return AUTOBAHN_TIMEOUT_DELAY_VALUES_MS[-1]
    return AUTOBAHN_TIMEOUT_DELAY_VALUES_MS[index]


def timeout_index_to_ms_for_protocol(protocol: int, index: int) -> int:
    protocol = _validate_protocol(protocol)
    if protocol == agent_pb2.PROTOCOL_PBFT:
        return pbft_timeout_index_to_ms(index)
    if protocol == agent_pb2.PROTOCOL_SMARTBFT:
        return smartbft_timeout_index_to_ms(index)
    if protocol == agent_pb2.PROTOCOL_TENDERMINT:
        return tendermint_propose_timeout_index_to_ms(index)
    if protocol == agent_pb2.PROTOCOL_HOTSTUFF:
        return hotstuff_timeout_index_to_ms(index)
    if protocol == agent_pb2.PROTOCOL_AUTOBAHN:
        return autobahn_timeout_delay_index_to_ms(index)
    return timeout_index_to_ms(index)


def sbft_slow_path_ms_to_index(timeout_ms: int) -> int:
    try:
        return SBFT_SLOW_PATH_MS_TO_INDEX[timeout_ms]
    except KeyError:
        raise ValueError(
            f"Unsupported SBFT slow-path timeout {timeout_ms}ms. "
            f"Valid options: {SBFT_SLOW_PATH_VALUES_MS}"
        ) from None


def sbft_slow_path_index_to_ms(index: int) -> int:
    bounded = max(0, min(index, len(SBFT_SLOW_PATH_VALUES_MS) - 1))
    return SBFT_SLOW_PATH_VALUES_MS[bounded]


def _sbft_pair_to_index(election_timeout_ms: int, slow_path_timeout_ms: int) -> int:
    election_idx = timeout_ms_to_index(election_timeout_ms)
    slow_idx = sbft_slow_path_ms_to_index(slow_path_timeout_ms)
    slow_base = len(SBFT_SLOW_PATH_VALUES_MS)
    return (election_idx * slow_base) + slow_idx


def _sbft_index_to_pair(index: int) -> Tuple[int, int]:
    slow_base = len(SBFT_SLOW_PATH_VALUES_MS)
    total = len(TIMEOUT_VALUES_MS) * slow_base
    bounded = max(0, min(index, total - 1))
    election_idx = bounded // slow_base
    slow_idx = bounded % slow_base
    return (
        timeout_index_to_ms(election_idx),
        sbft_slow_path_index_to_ms(slow_idx),
    )


def tendermint_timeouts_from_propose(
    propose_timeout_ms: int,
) -> Tuple[int, int, int]:
    propose_timeout_ms = int(propose_timeout_ms)
    if propose_timeout_ms <= 0:
        raise ValueError(
            f"Tendermint propose timeout must be positive, got {propose_timeout_ms}"
        )
    phase_timeout_ms = max(1, (propose_timeout_ms + 1) // 3)
    return propose_timeout_ms, phase_timeout_ms, phase_timeout_ms


def tendermint_propose_timeout_ms_to_index(propose_timeout_ms: int) -> int:
    try:
        return TENDERMINT_PROPOSE_TIMEOUT_MS_TO_INDEX[int(propose_timeout_ms)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported Tendermint propose timeout {propose_timeout_ms}ms. "
            f"Valid options: {TENDERMINT_PROPOSE_TIMEOUT_VALUES_MS}"
        ) from exc


def tendermint_propose_timeout_index_to_ms(index: int) -> int:
    if index < 0:
        return TENDERMINT_PROPOSE_TIMEOUT_VALUES_MS[0]
    if index >= len(TENDERMINT_PROPOSE_TIMEOUT_VALUES_MS):
        return TENDERMINT_PROPOSE_TIMEOUT_VALUES_MS[-1]
    return TENDERMINT_PROPOSE_TIMEOUT_VALUES_MS[index]


def _tendermint_timeouts_to_index(
    propose_timeout_ms: int, prevote_timeout_ms: int, precommit_timeout_ms: int
) -> int:
    expected = tendermint_timeouts_from_propose(propose_timeout_ms)
    actual = (
        int(propose_timeout_ms),
        int(prevote_timeout_ms),
        int(precommit_timeout_ms),
    )
    if actual != expected:
        raise ValueError(
            "Tendermint prevote and precommit timeouts must be one-third "
            f"of propose timeout: expected {expected}, got {actual}"
        )
    return tendermint_propose_timeout_ms_to_index(propose_timeout_ms)


def _tendermint_index_to_triple(index: int) -> Tuple[int, int, int]:
    return tendermint_timeouts_from_propose(
        tendermint_propose_timeout_index_to_ms(index)
    )


def get_action_space_size(protocol: int) -> int:
    protocol = _validate_protocol(protocol)
    if protocol == agent_pb2.PROTOCOL_PBFT:
        return len(PBFT_TIMEOUT_VALUES_MS)
    if protocol == agent_pb2.PROTOCOL_SMARTBFT:
        return len(SMARTBFT_TIMEOUT_VALUES_MS)
    if protocol == agent_pb2.PROTOCOL_SBFT:
        return len(TIMEOUT_VALUES_MS) * len(SBFT_SLOW_PATH_VALUES_MS)
    if protocol == agent_pb2.PROTOCOL_TENDERMINT:
        return len(TENDERMINT_PROPOSE_TIMEOUT_VALUES_MS)
    if protocol == agent_pb2.PROTOCOL_HOTSTUFF:
        return len(HOTSTUFF_TIMEOUT_VALUES_MS)
    if protocol == agent_pb2.PROTOCOL_AUTOBAHN:
        return len(AUTOBAHN_TIMEOUT_DELAY_VALUES_MS)
    raise ValueError(f"Unsupported protocol: {_protocol_name(protocol)}")


def report_to_feature_values(protocol: int, state: object) -> List[float]:
    protocol = _validate_protocol(protocol)
    fields = get_report_feature_field_names(protocol)
    return [float(getattr(state, field_name)) for field_name in fields]


def report_to_values(protocol: int, state: object) -> List[float]:
    """Every report field (get_report_field_names), for the dataset rows."""
    return [float(getattr(state, field_name)) for field_name in get_report_field_names(protocol)]


def timeout_to_action_index(protocol: int, timeout_used: object) -> int:
    protocol = _validate_protocol(protocol)
    if protocol == agent_pb2.PROTOCOL_PBFT:
        timeout_msg = timeout_used
        return pbft_timeout_ms_to_index(int(timeout_msg.election_timeout_milliseconds))
    if protocol == agent_pb2.PROTOCOL_SMARTBFT:
        timeout_msg = timeout_used
        return smartbft_timeout_ms_to_index(int(timeout_msg.election_timeout_milliseconds))
    if protocol == agent_pb2.PROTOCOL_SBFT:
        timeout_msg = timeout_used
        return _sbft_pair_to_index(
            int(timeout_msg.election_timeout_milliseconds),
            int(timeout_msg.slow_path_timeout_milliseconds),
        )
    if protocol == agent_pb2.PROTOCOL_TENDERMINT:
        timeout_msg = timeout_used
        return _tendermint_timeouts_to_index(
            int(timeout_msg.propose_timeout_milliseconds),
            int(timeout_msg.prevote_timeout_milliseconds),
            int(timeout_msg.precommit_timeout_milliseconds),
        )
    if protocol == agent_pb2.PROTOCOL_HOTSTUFF:
        timeout_msg = timeout_used
        return hotstuff_timeout_ms_to_index(int(timeout_msg.timeout_delay_milliseconds))
    if protocol == agent_pb2.PROTOCOL_AUTOBAHN:
        timeout_msg = timeout_used
        return autobahn_timeout_delay_ms_to_index(
            int(timeout_msg.timeout_delay_milliseconds)
        )
    raise ValueError(f"Unsupported protocol: {_protocol_name(protocol)}")


def action_index_to_timeout(protocol: int, index: int) -> agent_pb2.Timeout:
    protocol = _validate_protocol(protocol)
    if protocol == agent_pb2.PROTOCOL_PBFT:
        timeout_ms = pbft_timeout_index_to_ms(index)
        return agent_pb2.Timeout(
            pbft=agent_pb2.PbftTimeout(election_timeout_milliseconds=timeout_ms)
        )
    if protocol == agent_pb2.PROTOCOL_SMARTBFT:
        timeout_ms = smartbft_timeout_index_to_ms(index)
        return agent_pb2.Timeout(
            smartbft=agent_pb2.SmartbftTimeout(
                election_timeout_milliseconds=timeout_ms
            )
        )
    if protocol == agent_pb2.PROTOCOL_SBFT:
        election_timeout_ms, slow_path_timeout_ms = _sbft_index_to_pair(index)
        return agent_pb2.Timeout(
            sbft=agent_pb2.SbftTimeout(
                election_timeout_milliseconds=election_timeout_ms,
                slow_path_timeout_milliseconds=slow_path_timeout_ms,
            )
        )
    if protocol == agent_pb2.PROTOCOL_TENDERMINT:
        propose_timeout_ms, prevote_timeout_ms, precommit_timeout_ms = (
            _tendermint_index_to_triple(index)
        )
        return agent_pb2.Timeout(
            tendermint=agent_pb2.TendermintTimeout(
                propose_timeout_milliseconds=propose_timeout_ms,
                prevote_timeout_milliseconds=prevote_timeout_ms,
                precommit_timeout_milliseconds=precommit_timeout_ms,
            )
        )
    if protocol == agent_pb2.PROTOCOL_HOTSTUFF:
        timeout_ms = hotstuff_timeout_index_to_ms(index)
        return agent_pb2.Timeout(
            hotstuff=agent_pb2.HotstuffTimeout(timeout_delay_milliseconds=timeout_ms)
        )
    if protocol == agent_pb2.PROTOCOL_AUTOBAHN:
        # The only Autobahn action is the view-change timer; the fast-path
        # wait is fixed in config/autobahn/parameters.json.
        timeout_ms = autobahn_timeout_delay_index_to_ms(index)
        return agent_pb2.Timeout(
            autobahn=agent_pb2.AutobahnTimeout(timeout_delay_milliseconds=timeout_ms)
        )
    raise ValueError(f"Unsupported protocol: {_protocol_name(protocol)}")


class FeatureExtractor(Generic[S], ABC):
    """Converts an input state into a tensor for model consumption."""

    @abstractmethod
    def featurize(self, state: S) -> np.ndarray:
        pass


class ActionCodec(Generic[A], ABC):
    """Converts between model actions and domain-specific actions."""

    @abstractmethod
    def decode(self, action: np.ndarray) -> A:
        pass

    @abstractmethod
    def encode(self, action: object) -> np.ndarray:
        pass


class TensorModel(ABC):
    """Model that consumes and produces tensors."""

    @abstractmethod
    def predict_action(self, features: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def update_model(
        self, features: np.ndarray, action: np.ndarray, reward: float
    ) -> float:
        pass

    @abstractmethod
    def reset_model(self) -> None:
        pass


class ProtocolReportFeatureExtractor(FeatureExtractor[object]):
    def __init__(self, protocol: int):
        self._protocol = _validate_protocol(protocol)

    def featurize(self, state: object) -> np.ndarray:
        features = report_to_feature_values(self._protocol, state)
        return np.asarray(features, dtype=float)


class ProtocolTimeoutActionCodec(ActionCodec[agent_pb2.Timeout]):
    """Converts discrete arm indices to and from protocol timeout actions."""

    def __init__(self, protocol: int):
        self._protocol = _validate_protocol(protocol)

    def decode(self, action: np.ndarray) -> agent_pb2.Timeout:
        action = np.asarray(action, dtype=float)
        if action.size == 1:
            option_index = int(round(float(action.reshape(-1)[0])))
        else:
            option_index = int(np.argmax(action))
        return action_index_to_timeout(self._protocol, option_index)

    def encode(self, action: object) -> np.ndarray:
        option_index = timeout_to_action_index(self._protocol, action)
        return np.asarray(float(option_index), dtype=float)


UINT32_MAX = (1 << 32) - 1


def _validate_scheduled_timeout_ms(value: object, field_name: str) -> int:
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        timeout_ms = int(value)
    elif isinstance(value, (float, np.floating)) and np.isfinite(value):
        numeric_value = float(value)
        if not numeric_value.is_integer():
            raise ValueError(
                f"{field_name} must be an integer number of milliseconds, got {value}"
            )
        timeout_ms = int(numeric_value)
    else:
        raise ValueError(
            f"{field_name} must be an integer number of milliseconds, got {value!r}"
        )

    if timeout_ms < 1 or timeout_ms > UINT32_MAX:
        raise ValueError(
            f"{field_name} must be between 1 and {UINT32_MAX} milliseconds, "
            f"got {timeout_ms}"
        )
    return timeout_ms


class ScheduledTimeoutActionCodec(ActionCodec[agent_pb2.Timeout]):
    """Converts raw scheduled millisecond values to protocol timeout actions."""

    def __init__(self, protocol: int):
        self._protocol = _validate_protocol(protocol)

    @property
    def action_dim(self) -> int:
        if self._protocol == agent_pb2.PROTOCOL_SBFT:
            return 2
        return 1

    def decode(self, action: np.ndarray) -> agent_pb2.Timeout:
        try:
            values = np.asarray(action, dtype=float).reshape(-1)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Scheduled timeout action must contain numeric values, got {action!r}"
            ) from exc
        if values.size != self.action_dim:
            raise ValueError(
                f"Expected {self.action_dim} scheduled timeout value(s), "
                f"got {values.size}"
            )

        if self._protocol == agent_pb2.PROTOCOL_PBFT:
            election_timeout_ms = _validate_scheduled_timeout_ms(
                values[0], "PBFT election timeout"
            )
            return agent_pb2.Timeout(
                pbft=agent_pb2.PbftTimeout(
                    election_timeout_milliseconds=election_timeout_ms
                )
            )

        if self._protocol == agent_pb2.PROTOCOL_SMARTBFT:
            election_timeout_ms = _validate_scheduled_timeout_ms(
                values[0], "SmartBFT election timeout"
            )
            return agent_pb2.Timeout(
                smartbft=agent_pb2.SmartbftTimeout(
                    election_timeout_milliseconds=election_timeout_ms
                )
            )

        if self._protocol == agent_pb2.PROTOCOL_SBFT:
            election_timeout_ms = _validate_scheduled_timeout_ms(
                values[0], "SBFT election timeout"
            )
            slow_path_timeout_ms = _validate_scheduled_timeout_ms(
                values[1], "SBFT slow-path timeout"
            )
            return agent_pb2.Timeout(
                sbft=agent_pb2.SbftTimeout(
                    election_timeout_milliseconds=election_timeout_ms,
                    slow_path_timeout_milliseconds=slow_path_timeout_ms,
                )
            )

        if self._protocol == agent_pb2.PROTOCOL_HOTSTUFF:
            timeout_delay_ms = _validate_scheduled_timeout_ms(
                values[0], "HotStuff timeout delay"
            )
            return agent_pb2.Timeout(
                hotstuff=agent_pb2.HotstuffTimeout(
                    timeout_delay_milliseconds=timeout_delay_ms
                )
            )

        if self._protocol == agent_pb2.PROTOCOL_AUTOBAHN:
            timeout_delay_ms = _validate_scheduled_timeout_ms(
                values[0], "Autobahn timeout delay"
            )
            return agent_pb2.Timeout(
                autobahn=agent_pb2.AutobahnTimeout(
                    timeout_delay_milliseconds=timeout_delay_ms
                )
            )

        propose_timeout_ms = _validate_scheduled_timeout_ms(
            values[0], "Tendermint propose timeout"
        )
        propose_timeout_ms, prevote_timeout_ms, precommit_timeout_ms = (
            tendermint_timeouts_from_propose(propose_timeout_ms)
        )
        return agent_pb2.Timeout(
            tendermint=agent_pb2.TendermintTimeout(
                propose_timeout_milliseconds=propose_timeout_ms,
                prevote_timeout_milliseconds=prevote_timeout_ms,
                precommit_timeout_milliseconds=precommit_timeout_ms,
            )
        )

    def encode(self, action: object) -> np.ndarray:
        if isinstance(action, agent_pb2.Timeout):
            expected_field = {
                agent_pb2.PROTOCOL_PBFT: "pbft",
                agent_pb2.PROTOCOL_SBFT: "sbft",
                agent_pb2.PROTOCOL_TENDERMINT: "tendermint",
                agent_pb2.PROTOCOL_HOTSTUFF: "hotstuff",
                agent_pb2.PROTOCOL_AUTOBAHN: "autobahn",
                agent_pb2.PROTOCOL_SMARTBFT: "smartbft",
            }[self._protocol]
            actual_field = action.WhichOneof("value")
            if actual_field != expected_field:
                raise ValueError(
                    f"Expected {expected_field} timeout, got {actual_field or 'none'}"
                )
            action = getattr(action, expected_field)

        if self._protocol in (agent_pb2.PROTOCOL_PBFT, agent_pb2.PROTOCOL_SMARTBFT):
            timeout_values = [
                _validate_scheduled_timeout_ms(
                    getattr(action, "election_timeout_milliseconds"),
                    "PBFT election timeout",
                )
            ]
        elif self._protocol == agent_pb2.PROTOCOL_SBFT:
            timeout_values = [
                _validate_scheduled_timeout_ms(
                    getattr(action, "election_timeout_milliseconds"),
                    "SBFT election timeout",
                ),
                _validate_scheduled_timeout_ms(
                    getattr(action, "slow_path_timeout_milliseconds"),
                    "SBFT slow-path timeout",
                ),
            ]
        elif self._protocol == agent_pb2.PROTOCOL_HOTSTUFF:
            timeout_values = [
                _validate_scheduled_timeout_ms(
                    getattr(action, "timeout_delay_milliseconds"),
                    "HotStuff timeout delay",
                )
            ]
        elif self._protocol == agent_pb2.PROTOCOL_AUTOBAHN:
            timeout_values = [
                _validate_scheduled_timeout_ms(
                    getattr(action, "timeout_delay_milliseconds"),
                    "Autobahn timeout delay",
                )
            ]
        else:
            propose_timeout_ms = _validate_scheduled_timeout_ms(
                getattr(action, "propose_timeout_milliseconds"),
                "Tendermint propose timeout",
            )
            expected = tendermint_timeouts_from_propose(propose_timeout_ms)
            actual = (
                propose_timeout_ms,
                int(getattr(action, "prevote_timeout_milliseconds")),
                int(getattr(action, "precommit_timeout_milliseconds")),
            )
            if actual != expected:
                raise ValueError(
                    "Tendermint prevote and precommit timeouts must be "
                    f"one-third of propose timeout: expected {expected}, got {actual}"
                )
            timeout_values = [propose_timeout_ms]

        return np.asarray(timeout_values, dtype=float)


class LinearModel(TensorModel):
    """Linear policy over protocol-specific discrete timeout options.

    Reward-weighted multinomial logistic regression: each update pushes the
    policy toward the arm that was actually taken, weighted by the reward that
    arm earned. The gradient of that objective is analytic, so nothing here
    needs a differentiation framework:

        p       = softmax(X W^T)
        dL/dz   = s * (p - onehot(taken)) / batch      s = clamp(reward, 0)/1000
        dL/dW   = dL/dz^T X

    followed by an AdamW step. Decoupled weight decay is the ``W *= 1 - lr*wd``
    that precedes the moment update - AdamW's ordering, rather than the
    L2-in-the-gradient of classic Adam.

    Replaces an earlier autograd implementation that used ``torch.optim.AdamW``
    over a ``requires_grad`` weight matrix and ``F.cross_entropy``. The two were
    checked against each other over 400 updates before the torch version was
    removed: weights agreed to 3.7e-07 and argmax agreed on 100% of 2000
    contexts. Two differences were deliberate and are kept here:

    * Arithmetic is float64 throughout. The torch version accumulated in
      float32.
    * :meth:`reset_model` clears the optimizer moments as well as the weights.
      The torch version zeroed the weights but left AdamW's state behind, so a
      reset model's first steps were still scaled by the previous run's moments.
    """

    def __init__(
        self,
        action_dim: int,
        n_features: int,
        lr: float = 0.01,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")
        if n_features <= 0:
            raise ValueError(f"n_features must be > 0, got {n_features}")
        self._action_dim = int(action_dim)
        self._n_features = int(n_features)
        self._lr = float(lr)
        self._weight_decay = float(weight_decay)
        self._betas = (float(betas[0]), float(betas[1]))
        self._eps = float(eps)
        self.reset_model()

    def predict_action(self, features: np.ndarray) -> np.ndarray:
        """Logits for every arm.

        A 1-D context returns a 1-D ``(action_dim,)`` vector, a 2-D batch
        returns ``(batch, action_dim)``.
        """
        features_np = np.asarray(features, dtype=float)
        if features_np.shape[-1] != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} features, got {features_np.shape[-1]}"
            )
        logits = features_np @ self.weights.T
        return np.asarray(logits, dtype=float)

    def update_model(
        self,
        features: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | float,
        backward: bool = True,
    ) -> float:
        features_np = np.asarray(features, dtype=float)
        if features_np.ndim == 1:
            features_np = features_np.reshape(1, -1)
        if features_np.shape[1] != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} features, got {features_np.shape[1]}"
            )

        actions_np = np.asarray(actions, dtype=float).reshape(-1)
        rewards_np = np.asarray(rewards, dtype=float).reshape(-1)

        batch_size = features_np.shape[0]
        if actions_np.size == 1 and batch_size > 1:
            actions_np = np.repeat(actions_np, batch_size)
        if rewards_np.size == 1 and batch_size > 1:
            rewards_np = np.repeat(rewards_np, batch_size)
        if actions_np.size != batch_size:
            raise ValueError(
                "actions batch size mismatch: "
                f"expected {batch_size}, got {actions_np.size}"
            )
        if rewards_np.size != batch_size:
            raise ValueError(
                "rewards batch size mismatch: "
                f"expected {batch_size}, got {rewards_np.size}"
            )

        # Out-of-range arms are clamped rather than rejected, preserving the
        # behavior of the autograd version this replaced.
        targets = np.clip(
            np.round(actions_np).astype(int), 0, self._action_dim - 1
        )
        scales = np.clip(rewards_np, 0.0, None) / 1000.0

        logits = features_np @ self.weights.T
        # log-softmax via the shift trick, the same way cross_entropy does it.
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        rows = np.arange(batch_size)
        loss = float((-log_probs[rows, targets] * scales).mean())

        if backward and loss > 0.0:
            d_logits = np.exp(log_probs)
            d_logits[rows, targets] -= 1.0
            d_logits *= scales[:, None]
            d_logits /= batch_size
            self._adamw_step(d_logits.T @ features_np)

        return loss

    def _adamw_step(self, gradient: np.ndarray) -> None:
        beta1, beta2 = self._betas
        self._step += 1
        # Decoupled decay, applied to the weights before the moment update.
        self.weights *= 1.0 - self._lr * self._weight_decay
        self._moment1 = beta1 * self._moment1 + (1.0 - beta1) * gradient
        self._moment2 = beta2 * self._moment2 + (1.0 - beta2) * gradient * gradient
        bias_correction1 = 1.0 - beta1**self._step
        bias_correction2 = 1.0 - beta2**self._step
        denominator = np.sqrt(self._moment2) / np.sqrt(bias_correction2) + self._eps
        self.weights -= (self._lr / bias_correction1) * (self._moment1 / denominator)

    def reset_model(self) -> None:
        self.weights = np.zeros(
            (self._action_dim, self._n_features), dtype=np.float64
        )
        self._moment1 = np.zeros_like(self.weights)
        self._moment2 = np.zeros_like(self.weights)
        self._step = 0


class DiscountedLinUCB(TensorModel):
    """Non-stationary contextual UCB with one discounted model per arm.

    The posterior mean exploits the timeout with the highest predicted reward.
    The confidence bonus explores arms whose reward at the current context is
    uncertain. Every observation discounts all arms toward the ridge prior, so
    an arm that is not selected gradually becomes uncertain and can be tried
    again after conditions change. Optional random exploration can protect
    against a misspecified linear reward model, but it is disabled by default
    because a permanent exploration floor prevents strict convergence.

    Raw report fields span several orders of magnitude. A deterministic
    signed-log and row normalization keeps the ridge system conditioned without
    requiring a fitted scaler whose changing coordinates would invalidate the
    accumulated sufficient statistics.

    Rewards are centred on a discounted running mean (Kevin, 2026-10-19, restart D9d;
    as ContinuousLinUCB): an unplayed arm's mean decays toward 0, which uncentred
    reads as worthless in a phase with positive rewards and as good in one with
    negative rewards, so exploration depended on the reward's sign. Centred, 0 is
    "average". The prior fit centres on the mean of its samples.
    """

    def __init__(
        self,
        action_dim: int,
        n_features: int,
        confidence_scale: float = 1.0,
        discount_factor: float = 0.95,
        # 0 since 2026-10-19 (was 0.05; Kevin, runs/2026-10-19-linucb-discount): the random share cost HotStuff
        # about 0.15 (a random low rung at 5+150 collapses rounds) and nothing measurable elsewhere.
        exploration_floor: float = 0.0,
        ridge: float = 1.0,
        reward_scale: float = 1.0,
        seed: Optional[int] = None,
        prior_theta: Optional[np.ndarray] = None,
        opening_arm: Optional[int] = None,
    ) -> None:
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")
        if n_features <= 0:
            raise ValueError(f"n_features must be > 0, got {n_features}")
        if opening_arm is not None and not 0 <= opening_arm < action_dim:
            raise ValueError(f"opening_arm must be in [0, {action_dim}), got {opening_arm}")
        if confidence_scale < 0.0:
            raise ValueError(
                f"confidence_scale must be >= 0, got {confidence_scale}"
            )
        if not 0.0 < discount_factor <= 1.0:
            raise ValueError(
                f"discount_factor must be in (0, 1], got {discount_factor}"
            )
        if not 0.0 <= exploration_floor <= 1.0:
            raise ValueError(
                f"exploration_floor must be in [0, 1], got {exploration_floor}"
            )
        if ridge <= 0.0:
            raise ValueError(f"ridge must be > 0, got {ridge}")
        if reward_scale <= 0.0:
            raise ValueError(f"reward_scale must be > 0, got {reward_scale}")

        self._action_dim = int(action_dim)
        self._n_features = int(n_features)
        self._context_dim = self._n_features + 1
        self._confidence_scale = float(confidence_scale)
        self._discount_factor = float(discount_factor)
        self._exploration_floor = float(exploration_floor)
        self._ridge = float(ridge)
        self._reward_scale = float(reward_scale)
        self._seed = seed
        # Unplayed arms are tried first, the opening arm (the ladder's middle rung; Kevin,
        # 2026-10-19, restart D10b: every model opens there) before the others in index order.
        self._unplayed_order = (
            ([int(opening_arm)] if opening_arm is not None else [])
            + [a for a in range(self._action_dim) if a != opening_arm]
        )
        # Prior mean of each arm's coefficients, in the _contexts space and in
        # reward / reward_scale units: b starts at ridge * theta0 and the
        # discount pulls it back there, so an unplayed arm's mean decays to
        # theta0 instead of to 0. Zeros (None) is the plain ridge prior.
        expected_shape = (self._action_dim, self._context_dim)
        if prior_theta is None:
            self._prior_theta = np.zeros(expected_shape, dtype=np.float64)
        else:
            self._prior_theta = np.array(prior_theta, dtype=np.float64)
            if self._prior_theta.shape != expected_shape:
                raise ValueError(
                    f"prior_theta must have shape {expected_shape}, "
                    f"got {self._prior_theta.shape}"
                )
            if not np.all(np.isfinite(self._prior_theta)):
                raise ValueError("prior_theta must contain only finite values")
        self.reset_model()

    def fit_prior_theta(
        self, features: np.ndarray, actions: np.ndarray, rewards: np.ndarray
    ) -> np.ndarray:
        """Undiscounted per-arm ridge fit, for use as ``prior_theta``.

        ``actions`` are arm indices, as update_model takes them. Each arm's
        coefficients solve ``(ridge I + X^T X) theta = X^T y`` with X the
        _contexts rows of that arm and y the rewards over reward_scale.
        """
        contexts = self._contexts(features)
        action_values = np.asarray(actions, dtype=np.float64).reshape(-1)
        reward_values = np.asarray(rewards, dtype=np.float64).reshape(-1)
        if action_values.size != len(contexts) or reward_values.size != len(contexts):
            raise ValueError(
                f"{len(contexts)} feature rows, {action_values.size} actions "
                f"and {reward_values.size} rewards"
            )
        if not np.all(np.isfinite(reward_values)):
            raise ValueError("rewards must contain only finite values")
        rounded_actions = np.rint(action_values)
        if not np.allclose(action_values, rounded_actions, rtol=0.0, atol=1e-9):
            raise ValueError("actions must be integer arm indices")
        action_indices = rounded_actions.astype(int)
        if np.any(action_indices < 0) or np.any(action_indices >= self._action_dim):
            raise ValueError(
                f"actions must be between 0 and {self._action_dim - 1}"
            )
        identity = np.eye(self._context_dim, dtype=np.float64)
        theta = np.empty((self._action_dim, self._context_dim), dtype=np.float64)
        for arm in range(self._action_dim):
            selected = action_indices == arm
            if not np.any(selected):
                raise ValueError(f"arm {arm} has no samples to fit a prior on")
            x = contexts[selected]
            y = (reward_values[selected] - float(np.mean(reward_values))) / self._reward_scale
            theta[arm] = np.linalg.solve(self._ridge * identity + x.T @ x, x.T @ y)
        return theta

    def _contexts(self, features: np.ndarray) -> np.ndarray:
        rows = np.asarray(features, dtype=np.float64)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.ndim != 2 or rows.shape[1] != self._n_features:
            actual = rows.shape[1] if rows.ndim == 2 else rows.shape
            raise ValueError(f"Expected {self._n_features} features, got {actual}")
        if not np.all(np.isfinite(rows)):
            raise ValueError("features must contain only finite values")

        transformed = np.sign(rows) * np.log1p(np.abs(rows))
        norms = np.linalg.norm(transformed, axis=1, keepdims=True)
        transformed = transformed / np.maximum(norms, 1.0)
        return np.hstack((np.ones((len(transformed), 1)), transformed))

    def _scores(self, context: np.ndarray) -> np.ndarray:
        scores = np.empty(self._action_dim, dtype=np.float64)
        for arm in range(self._action_dim):
            if self._observation_counts[arm] == 0:
                scores[arm] = np.inf
                continue
            try:
                theta = np.linalg.solve(
                    self._precision[arm], self._reward_vectors[arm]
                )
                direction = np.linalg.solve(self._precision[arm], context)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError(
                    f"discrete_linucb precision matrix for arm {arm} is singular"
                ) from exc
            predicted_reward = float(context @ theta)
            uncertainty_variance = float(
                direction @ self._covariance[arm] @ direction
            )
            uncertainty = float(np.sqrt(max(uncertainty_variance, 0.0)))
            scores[arm] = predicted_reward + self._confidence_scale * uncertainty
        return scores

    def predict_action(self, features: np.ndarray) -> np.ndarray:
        contexts = self._contexts(features)
        scores = np.vstack([self._scores(context) for context in contexts])

        all_arms_observed = bool(np.all(self._observation_counts > 0))
        for row in scores:
            if (
                all_arms_observed
                and self._exploration_floor > 0.0
                and self._rng.random() < self._exploration_floor
            ):
                selected = int(self._rng.integers(self._action_dim))
                best = float(np.max(row))
                row[selected] = np.nextafter(best, np.inf)
                continue

            best = float(np.max(row))
            if not np.isfinite(best):
                # Unplayed arms score +inf; rank them in _unplayed_order with large finite
                # scores so the codec's argmax (and a floor wrapper hiding arms under it)
                # takes the first unplayed arm in that order.
                unplayed = [a for a in self._unplayed_order if not np.isfinite(row[a])]
                for rank, arm in enumerate(unplayed):
                    row[arm] = 1e30 * (len(unplayed) - rank)
                continue
            tied = np.flatnonzero(row == best)
            if len(tied) > 1:
                selected = int(self._rng.choice(tied))
                row[selected] = np.nextafter(best, np.inf)

        return scores

    def update_model(
        self,
        features: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | float,
        backward: bool = True,
    ) -> float:
        if not backward:
            return 0.0

        contexts = self._contexts(features)
        action_values = np.asarray(actions, dtype=np.float64).reshape(-1)
        reward_values = np.asarray(rewards, dtype=np.float64).reshape(-1)
        batch_size = len(contexts)
        if action_values.size == 1 and batch_size > 1:
            action_values = np.repeat(action_values, batch_size)
        if reward_values.size == 1 and batch_size > 1:
            reward_values = np.repeat(reward_values, batch_size)
        if action_values.size != batch_size:
            raise ValueError(
                f"actions batch size mismatch: expected {batch_size}, "
                f"got {action_values.size}"
            )
        if reward_values.size != batch_size:
            raise ValueError(
                f"rewards batch size mismatch: expected {batch_size}, "
                f"got {reward_values.size}"
            )
        if not np.all(np.isfinite(action_values)):
            raise ValueError("actions must contain only finite values")
        rounded_actions = np.rint(action_values)
        if not np.allclose(action_values, rounded_actions, rtol=0.0, atol=1e-9):
            raise ValueError("actions must be integer arm indices")
        action_indices = rounded_actions.astype(int)
        if np.any(action_indices < 0) or np.any(action_indices >= self._action_dim):
            raise ValueError(
                f"actions must be between 0 and {self._action_dim - 1}"
            )
        if not np.all(np.isfinite(reward_values)):
            raise ValueError("rewards must contain only finite values")

        prior_weight = (1.0 - self._discount_factor) * self._ridge
        squared_discount = self._discount_factor * self._discount_factor
        covariance_prior_weight = (1.0 - squared_discount) * self._ridge
        identity = np.eye(self._context_dim, dtype=np.float64)
        for context, arm, reward in zip(contexts, action_indices, reward_values):
            self._precision *= self._discount_factor
            if prior_weight > 0.0:
                self._precision += prior_weight * identity[None, :, :]
            self._covariance *= squared_discount
            if covariance_prior_weight > 0.0:
                self._covariance += (
                    covariance_prior_weight * identity[None, :, :]
                )
            self._reward_vectors *= self._discount_factor
            self._reward_vectors += prior_weight * self._prior_theta

            observation = np.outer(context, context)
            self._precision[arm] += observation
            self._covariance[arm] += observation
            # The first reward is centred on itself (no baseline yet), so the first arm
            # played is not credited the whole raw reward.
            centered = float(reward) - (self._baseline() if self._baseline_weight > 0.0 else float(reward))
            self._reward_vectors[arm] += (centered / self._reward_scale) * context
            self._observation_counts[arm] += 1
            # The baseline is discounted like everything else.
            self._baseline_total = self._discount_factor * self._baseline_total + float(reward)
            self._baseline_weight = self._discount_factor * self._baseline_weight + 1.0

        return 0.0

    def reset_model(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        identity = np.eye(self._context_dim, dtype=np.float64)
        self._precision = np.repeat(
            (self._ridge * identity)[None, :, :],
            self._action_dim,
            axis=0,
        )
        self._covariance = self._precision.copy()
        self._reward_vectors = self._ridge * self._prior_theta
        self._observation_counts = np.zeros(self._action_dim, dtype=np.int64)
        self._baseline_total = 0.0
        self._baseline_weight = 0.0

    def _baseline(self) -> float:
        """The discounted mean reward so far (0 before the first)."""
        if self._baseline_weight <= 0.0:
            return 0.0
        return self._baseline_total / self._baseline_weight


class RandomTimeoutModel(TensorModel):
    """Selects a random timeout action index each prediction."""

    def __init__(self, action_dim: int, seed: Optional[int] = None):
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")
        self._action_dim = int(action_dim)
        if seed is None:
            seed = int(time.time_ns() & 0x7FFFFFFF)
        self._initial_seed = int(seed)
        self._rng = np.random.default_rng(self._initial_seed)

    def predict_action(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        if features.ndim == 1:
            option_index = int(self._rng.integers(0, self._action_dim))
            return np.asarray(float(option_index), dtype=float)

        batch_size = int(features.shape[0])
        return self._rng.integers(
            0, self._action_dim, size=batch_size
        ).astype(float)

    def update_model(
        self, features: np.ndarray, action: np.ndarray, reward: float
    ) -> float:
        del features, action, reward
        return 0.0

    def reset_model(self) -> None:
        self._rng = np.random.default_rng(self._initial_seed)

class ScheduledTimeoutModel(TensorModel):
    """Returns raw protocol timeout values from a schedule looked up by episode number."""

    def __init__(
        self,
        protocol: int,
        schedule: List[Tuple[float, object]],
        start_monotonic: Optional[float] = None,
    ):
        self._protocol = _validate_protocol(protocol)
        self._action_codec = ScheduledTimeoutActionCodec(self._protocol)
        if not schedule:
            raise ValueError("Schedule must contain at least one entry.")
        normalized_schedule: List[Tuple[float, Tuple[int, ...]]] = []
        for time_s, raw_action in schedule:
            timeout = self._action_codec.decode(raw_action)
            normalized_action = tuple(
                int(value) for value in self._action_codec.encode(timeout).reshape(-1)
            )
            normalized_schedule.append((float(time_s), normalized_action))
        self._schedule = sorted(normalized_schedule, key=lambda entry: entry[0])
        self._schedule_time_shift_s = 0.0
        self._rebuild_effective_schedule()
        self._start_time = (
            time.monotonic() if start_monotonic is None else float(start_monotonic)
        )
        # Set by set_episode_ms before the first lookup.
        self._episode_ms: Optional[int] = None
        self._active_row = -1

    @classmethod
    def from_csv(
        cls,
        protocol: int,
        path: Path,
        start_monotonic: Optional[float] = None,
    ) -> "ScheduledTimeoutModel":
        protocol = _validate_protocol(protocol)
        action_codec = ScheduledTimeoutActionCodec(protocol)
        schedule: List[Tuple[float, object]] = []
        with path.open("r", newline="") as handle:
            reader = csv.reader(handle)
            for lineno, row in enumerate(reader, start=1):
                if not row:
                    continue
                first = row[0].strip()
                if not first or first.startswith("#"):
                    continue
                try:
                    time_s = float(first)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid schedule time at line {lineno}: {row}"
                    ) from exc

                if protocol in (agent_pb2.PROTOCOL_PBFT, agent_pb2.PROTOCOL_SMARTBFT):
                    if len(row) < 2:
                        raise ValueError(
                            f"Invalid PBFT/SmartBFT schedule at line {lineno}: expected 2 columns time_s,election_timeout_ms."
                        )
                    try:
                        raw_action: object = int(row[1].strip())
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid PBFT timeout at line {lineno}: {row[1]!r}"
                        ) from exc
                elif protocol == agent_pb2.PROTOCOL_SBFT:
                    if len(row) < 3:
                        raise ValueError(
                            "Invalid SBFT schedule at line "
                            f"{lineno}: expected 3 columns "
                            "time_s,election_timeout_ms,slow_path_timeout_ms."
                        )
                    try:
                        raw_action = (
                            int(row[1].strip()),
                            int(row[2].strip()),
                        )
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid SBFT timeout at line {lineno}: {row[1:4]}"
                        ) from exc
                elif protocol == agent_pb2.PROTOCOL_HOTSTUFF:
                    if len(row) < 2:
                        raise ValueError(
                            f"Invalid HotStuff schedule at line {lineno}: expected 2 columns time_s,timeout_ms."
                        )
                    try:
                        raw_action = int(row[1].strip())
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid HotStuff timeout at line {lineno}: {row[1]!r}"
                        ) from exc
                elif protocol == agent_pb2.PROTOCOL_AUTOBAHN:
                    if len(row) != 2:
                        raise ValueError(
                            "Invalid Autobahn schedule at line "
                            f"{lineno}: expected 2 columns "
                            "time_s,timeout_delay_ms "
                            f"(got {len(row)}; the fast_path_timeout column was "
                            "removed, it is fixed in config/autobahn/parameters.json)."
                        )
                    try:
                        raw_action = int(row[1].strip())
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid Autobahn timeout at line {lineno}: {row[1]!r}"
                        ) from exc
                else:
                    if len(row) < 2:
                        raise ValueError(
                            "Invalid Tendermint schedule at line "
                            f"{lineno}: expected 2 columns "
                            "time_s,propose_timeout_ms."
                        )
                    try:
                        raw_action = int(row[1].strip())
                    except ValueError as exc:
                        raise ValueError(
                            "Invalid Tendermint propose timeout at line "
                            f"{lineno}: {row[1]!r}"
                        ) from exc
                try:
                    action_codec.decode(raw_action)
                except (ValueError, OverflowError) as exc:
                    raise ValueError(
                        f"Invalid schedule timeout at line {lineno}: {exc}"
                    ) from exc
                schedule.append((time_s, raw_action))
        if not schedule:
            raise ValueError(f"No schedule entries found in {path}.")
        return cls(protocol, schedule, start_monotonic=start_monotonic)

    def _rebuild_effective_schedule(self) -> None:
        if not self._schedule:
            raise ValueError("Schedule must contain at least one entry.")
        first_switch_time_s = float(self._schedule[0][0])
        self._schedule_time_shift_s = -min(0.0, first_switch_time_s)

    def set_episode_ms(self, episode_ms: int) -> None:
        """Bind the replicas' episode period P and check every row is on its grid.

        Episode e applies its timeouts at start + warmUp + P*(e-1), so a row
        written at time_s takes effect from the episode whose apply instant is
        time_s after the warm-up. A non-negative row off that grid would apply
        at the next multiple without saying so, and is refused.
        """
        episode_ms = int(episode_ms)
        if episode_ms <= 0:
            raise ValueError(f"episode_ms must be > 0, got {episode_ms}")
        if self._schedule[0][0] > 0.0:
            raise ValueError(
                "Schedule has no row for episode 1: its first row is at "
                f"time_s={self._schedule[0][0]:g}, it must be at or below 0."
            )
        for time_s, _ in self._schedule:
            if time_s < 0.0:
                continue
            time_ms = time_s * 1000.0
            if abs(time_ms - round(time_ms)) > 1e-6 or round(time_ms) % episode_ms != 0:
                raise ValueError(
                    f"Schedule row time_s={time_s:g} is not a multiple of the "
                    f"episode period {episode_ms / 1000.0:g} s."
                )
        self._episode_ms = episode_ms

    def action_for_episode(self, episode: int) -> np.ndarray:
        """The last row with time_s <= P*(episode-1); negative rows count for every episode."""
        if self._episode_ms is None:
            raise RuntimeError("set_episode_ms must be called before action_for_episode.")
        episode = int(episode)
        if episode < 1:
            raise ValueError(f"Episode numbers start at 1, got {episode}")
        episode_time_ms = self._episode_ms * (episode - 1)
        active_row = 0
        raw_action = self._schedule[0][1]
        for row_idx, (switch_time_s, schedule_action) in enumerate(self._schedule):
            if switch_time_s < 0.0 or round(switch_time_s * 1000.0) <= episode_time_ms:
                active_row = row_idx
                raw_action = schedule_action
            else:
                break
        if active_row != self._active_row:
            source_switch_time_s, active_action = self._schedule[active_row]
            timeout = self._action_codec.decode(active_action)
            if self._protocol == agent_pb2.PROTOCOL_PBFT:
                description = (
                    "election_timeout_ms="
                    f"{timeout.pbft.election_timeout_milliseconds}"
                )
            elif self._protocol == agent_pb2.PROTOCOL_SMARTBFT:
                description = (
                    "election_timeout_ms="
                    f"{timeout.smartbft.election_timeout_milliseconds}"
                )
            elif self._protocol == agent_pb2.PROTOCOL_SBFT:
                description = (
                    f"election_timeout_ms={timeout.sbft.election_timeout_milliseconds}, "
                    f"slow_path_timeout_ms={timeout.sbft.slow_path_timeout_milliseconds}"
                )
            elif self._protocol == agent_pb2.PROTOCOL_HOTSTUFF:
                description = (
                    "timeout_delay_ms="
                    f"{timeout.hotstuff.timeout_delay_milliseconds}"
                )
            elif self._protocol == agent_pb2.PROTOCOL_AUTOBAHN:
                description = (
                    "timeout_delay_ms="
                    f"{timeout.autobahn.timeout_delay_milliseconds}"
                )
            else:
                description = (
                    "propose_timeout_ms="
                    f"{timeout.tendermint.propose_timeout_milliseconds}, "
                    "prevote_timeout_ms="
                    f"{timeout.tendermint.prevote_timeout_milliseconds}, "
                    "precommit_timeout_ms="
                    f"{timeout.tendermint.precommit_timeout_milliseconds}"
                )
            print(
                "[scheduled-model] active schedule: "
                f"episode={episode}, "
                f"episode_time_s={episode_time_ms / 1000.0:g}, "
                f"source_time_s={source_switch_time_s:g}, "
                f"{description}"
            )
            self._active_row = active_row
        return np.asarray(raw_action, dtype=float)

    def is_warmup_active(self) -> bool:
        if self._schedule_time_shift_s <= 0.0:
            return False
        elapsed_s = time.monotonic() - self._start_time
        return 0.0 <= elapsed_s < self._schedule_time_shift_s

    def predict_action(self, features: np.ndarray) -> np.ndarray:
        del features
        raise RuntimeError(
            "ScheduledTimeoutModel is looked up by episode number; call action_for_episode."
        )

    def update_model(
        self, features: np.ndarray, action: np.ndarray, reward: float
    ) -> float:
        del features, action, reward
        return 0.0

    def reset_model(self) -> None:
        self._start_time = time.monotonic()
        self._active_row = -1

class ThompsonSamplingModel(TensorModel):
    """Per-arm bootstrapped-tree CMAB (contextual GIRO structure).

    Each arm keeps its own sliding window of (features, reward) pairs and its
    own ensemble of trees, each head fit on a bootstrap resample of that arm's
    window. Prediction draws one head per arm independently and the caller
    pulls the argmax across arms.
    """

    def __init__(
        self,
        action_dim: int,
        n_features: int,
        n_heads: int = 100,
        seed: Optional[int] = None,
    ):
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be > 0, got {n_heads}")

        self._act_dim = int(action_dim)
        self._obs_dim = int(n_features)
        self._n_heads = int(n_heads)
        self._max_bucket_size = 16
        self._seed = seed

        self.reset_model()

    def predict_action(self, features: np.ndarray, e = 0.1) -> np.ndarray:
        """e-TS: with probability e take the Thompson draw (one bootstrap head
        per arm); with probability 1-e exploit the arm with the highest
        empirical mean over its reward sliding window."""
        features_np = np.asarray(features, dtype=float)
        if features_np.ndim == 1:
            features_np = features_np.reshape(1, -1)

        batch_size = features_np.shape[0]
        predictions_stacked = np.zeros((batch_size, self._act_dim), dtype=float)
        # One independent head (bootstrap draw) per arm, per row.
        sampled_heads = self._rng.integers(
            0, self._n_heads, size=(batch_size, self._act_dim)
        )
        explore = self._rng.random(batch_size) < e

        for i in range(batch_size):
            row = features_np[i : i + 1]
            for arm in range(self._act_dim):
                if not self.is_fitted[arm]:
                    # Arm without a fitted window yet: force a pull.
                    predictions_stacked[i, arm] = 1e9
                    continue
                if explore[i]:
                    head = self.models[arm][sampled_heads[i, arm]]
                    predictions_stacked[i, arm] = head.predict(row)[0]
                else:
                    predictions_stacked[i, arm] = float(
                        np.mean(self.memory[arm]["rewards"])
                    )

        for i in range(batch_size):
            row_preds = predictions_stacked[i]
            tied_indices = np.flatnonzero(np.isclose(row_preds, np.max(row_preds)))

            if len(tied_indices) > 1:
                best_index = self._rng.choice(tied_indices)
                predictions_stacked[i, best_index] += 1e-5

        q_values = np.asarray(predictions_stacked, dtype=float)
        return q_values

    def update_model(
        self, features: np.ndarray, actions: np.ndarray, rewards: np.ndarray | float, backward: bool = True
    ) -> float:
        # An evaluation batch must not enter the replay memory: evaluate_model
        # calls this with backward=False twice per episode, so ingesting here
        # would record every episode three times over.
        if not backward:
            return 0.0
        features_np = np.asarray(features, dtype=float)
        if features_np.ndim == 1:
            features_np = features_np.reshape(1, -1)
            
        if isinstance(actions, np.ndarray):
            actions_np = np.asarray(actions, dtype=int).ravel()
        else:
            actions_np = np.array([int(actions)])
            
        if isinstance(rewards, np.ndarray):
            rewards_np = np.asarray(rewards, dtype=float).ravel()
        else:
            rewards_np = np.full(len(actions_np), float(rewards))

        if len(actions_np) == 1 and features_np.shape[0] > 1:
            actions_np = np.repeat(actions_np, features_np.shape[0])
        if len(rewards_np) == 1 and features_np.shape[0] > 1:
            rewards_np = np.repeat(rewards_np, features_np.shape[0])
        if len(actions_np) != features_np.shape[0]:
            raise ValueError(
                "actions batch size mismatch: "
                f"expected {features_np.shape[0]}, got {len(actions_np)}"
            )
        if len(rewards_np) != features_np.shape[0]:
            raise ValueError(
                "rewards batch size mismatch: "
                f"expected {features_np.shape[0]}, got {len(rewards_np)}"
            )

        for i in range(len(actions_np)):
            arm = int(actions_np[i])
            if not 0 <= arm < self._act_dim:
                raise ValueError(f"action out of range: {arm}")
            self.memory[arm]["features"].append(features_np[i])
            self.memory[arm]["rewards"].append(rewards_np[i])

        if backward:
            window_max = max(
                (max(m["rewards"]) for m in self.memory if m["rewards"]),
                default=0.0,
            )
                                    
            for arm in sorted({int(a) for a in actions_np}):
                X = np.array(self.memory[arm]["features"])
                y = np.array(self.memory[arm]["rewards"])

                            
                for model in self.models[arm]:
                    idx = self._rng.choice(len(X), size=len(X), replace=True)
                    model.fit(X[idx], y[idx])
                self.is_fitted[arm] = True

        return 0.0

    def reset_model(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self.models = [
            [
                DecisionTreeRegressor(
                    criterion='squared_error',
                    max_depth=6,
                    min_samples_leaf=6,
                    min_impurity_decrease=0.01,
                    random_state=arm_index * self._n_heads + head_index,
                )
                for head_index in range(self._n_heads)
            ]
            for arm_index in range(self._act_dim)
        ]
        self.is_fitted = [False] * self._act_dim
        self._init_buckets()

    def _init_buckets(self):
        self.memory = [
            {
                "features": deque(maxlen=self._max_bucket_size),
                "rewards": deque(maxlen=self._max_bucket_size),
            }
            for _ in range(self._act_dim)
        ]

class PavlovTimeoutModel(TensorModel):
    """Win-stay/lose-shift (Pavlov) timeout heuristic against the last reward.

    A non-learning baseline: no context, no per-arm statistics, no threshold, no
    calibration, O(1) state. Arms are ordered by ascending timeout, so for pbft
    arm 1 = 300 ms, arm 2 = 800 ms, arm 3 = 1800 ms. The run opens on arm 1 and
    the tracked reward starts at 0, so the first episode is always a win. Then,
    every episode:

        observed >= tracked -> keep the current arm
        observed <  tracked -> step to the next arm (1 -> 2 -> 3 -> 1)
        tracked = observed

    Measured cost of the strict "compare against t-1" reference: per-episode
    noise on real hardware is as large as the between-arm difference, so the
    comparison is close to a coin flip and the rule churns. Over the fixed-arm
    CloudLab runs (local/tps_rule.py, scripts/cloudlab/{300,800,1800}, 134
    episodes, 30% switch cost) it switches on 56 of 134 episodes and yields
    5444 tps, against 5935 for the best fixed arm and 8741 for a per-phase
    oracle -- i.e. as specified it does not beat leaving the timeout alone.
    local/tps_rule.py holds the variants that do, and re-measures either.
    """

    def __init__(
        self,
        action_dim: int,
        protocol: Optional[int] = None,
        start_index: Optional[int] = None,
    ):
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")

        self._action_dim = int(action_dim)
        # Walk the arms in ascending timeout order, not raw index order: for
        # sbft/tendermint the index encodes a triple, so the two differ.
        if protocol is None:
            self._order = list(range(self._action_dim))
        else:
            protocol = _validate_protocol(protocol)
            self._order = sorted(
                range(self._action_dim),
                key=lambda idx: timeout_index_to_ms_for_protocol(protocol, idx),
            )
        self._position = {idx: pos for pos, idx in enumerate(self._order)}

        if start_index is None:
            # Arm 1, the shortest timeout.
            self._start_index = self._order[0]
        else:
            self._start_index = int(start_index)
            if self._start_index not in self._position:
                raise ValueError(
                    f"start_index must be in [0, {self._action_dim}), "
                    f"got {start_index}"
                )

        self.reset_model()

    def _to_index(self, action: np.ndarray) -> int:
        if np.size(action) == 1:
            return int(round(float(np.asarray(action).reshape(-1)[0])))
        return int(np.argmax(action))

    def predict_action(self, features: np.ndarray) -> np.ndarray:
        if np.asarray(features).ndim == 1:
            return np.asarray(float(self._index), dtype=float)
        batch_size = int(np.asarray(features).shape[0])
        return np.full(batch_size, float(self._index), dtype=float)

    def update_model(
        self, features: np.ndarray, action: np.ndarray, reward: float
    ) -> float:
        del features
        arm = self._to_index(action)
        if arm in self._position:
            # Follow the timeout that actually ran: in shared or consensus mode
            # the applied arm need not be the one this replica predicted.
            self._index = arm

        value = float(reward)
        if value < self._tracked:
            self._index = self._order[
                (self._position[self._index] + 1) % len(self._order)
            ]
        self._tracked = value
        return 0.0

    def reset_model(self) -> None:
        self._tracked = 0.0
        self._index = self._start_index


@dataclass
class _BlockPolicy:
    """Q(i,m): the policy distribution frozen at the start of block m.

    ``model`` is the tree that was in force there (None for block 0, before
    anything is fitted) and ``epsilon`` is that block's exploration rate, so
    the pair reproduces the exact sampling distribution Q^{nu_m}(.|x) used at
    the time. ``length`` is block m's length, which doubles as the length of a
    replay phase carrying index m."""

    model: Optional[DecisionTreeRegressor]
    epsilon: float
    length: int


@dataclass
class _Replay:
    """One in-flight replay phase: (m, [t, t + 2^m L - 1]) from Algorithm 1."""

    index: int
    rounds_left: int
    rows: List[int] = field(default_factory=list)


class BlockedTreeBandit(TensorModel):
    """Epsilon-greedy blocks over one regression tree on (context, arm).

    Decisions run in blocks of growing length; the tree is refit between
    blocks, and a replay test against earlier block policies resets the model
    when the reward has shifted.
    """

    def __init__(
        self,
        action_dim: int,
        n_features: int,
        epsilon: float = 0.01,
        block_length: int = 8,
        max_block_length: int = 32,
        nu: float = 0.15,
        max_depth: int = 12,
        min_samples_leaf: int = 3,
        enable_replay: bool = True,
        replay_rate: float = 1.0,
        replay_var_ratio: float = 4.0,
        max_snapshots: int = 8,
        decay_epsilon: bool = True,
        seed: Optional[int] = None,
    ):
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")
        if not 0.0 < epsilon <= 1.0:
            raise ValueError(f"epsilon must be in (0, 1], got {epsilon}")
        if block_length <= 0:
            raise ValueError(f"block_length must be > 0, got {block_length}")
        if replay_rate < 0.0:
            raise ValueError(f"replay_rate must be >= 0, got {replay_rate}")
        if min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be >= 1, got {min_samples_leaf}")

        self._act_dim = int(action_dim)
        self._obs_dim = int(n_features)
        self._epsilon = float(epsilon)
        self._init_block_length = int(block_length)
        self._max_block_length = int(max_block_length)
        self._nu = float(nu)
        self._max_depth = int(max_depth)
        self._min_samples_leaf = int(min_samples_leaf)
        self._enable_replay = bool(enable_replay)
        self._replay_rate = float(replay_rate)
        self._replay_var_ratio = float(replay_var_ratio)
        self._max_snapshots = max(1, int(max_snapshots))
        self._decay_epsilon = bool(decay_epsilon)
        self._seed = seed

        self.reset_model()

    def _block_epsilon(self, block_index: int) -> float:
        if not self._decay_epsilon:
            return self._epsilon
        return float(min(1.0, self._epsilon * (2.0 ** (-0.5 * block_index))))

    def _maybe_start_replay(self) -> None:
        if not self._enable_replay:
            return
        j = self._block_index
        if j <= 0:
            return
        available = [
            m
            for m in range(min(j, len(self._block_policies)))
            if self._block_policies[m].model is not None
        ]
        if not available:
            return

        weights = np.array([2.0 ** (-0.5 * m) for m in available], dtype=float)
        p_rep = (
            self._replay_rate
            * (2.0 ** (-0.5 * j))
            * float(weights.sum())
            / float(self._init_block_length)
        )
        if self._rng.random() >= min(1.0, p_rep):
            return

        m = int(self._rng.choice(available, p=weights / weights.sum()))
        length = int(self._block_policies[m].length)
        self._replays.append(
            _Replay(index=m, rounds_left=length)
        )
        self.replay_starts += 1

    def _active_replay_index(self) -> Optional[int]:
        """M_t from line 13, then m ~ Uniform(M_t) from line 17."""
        indices = sorted({r.index for r in self._replays if r.rounds_left > 0})
        if not indices:
            return None
        return int(self._rng.choice(indices))

    def predict_action(self, features: np.ndarray, e: Optional[float] = None) -> np.ndarray:
        override_epsilon = None if e is None else float(e)
        features_np = np.asarray(features, dtype=float)
        if features_np.ndim == 1:
            features_np = features_np.reshape(1, -1)

        batch_size = features_np.shape[0]
        predictions_stacked = np.zeros((batch_size, self._act_dim), dtype=float)

        pulled_arms = {int(a) for a in self.memory["actions"]}
        missing_arms = [a for a in range(self._act_dim) if a not in pulled_arms]

        self._pending = []
        for i in range(batch_size):
            # Step 1. Randomly start a replay phase.
            self._maybe_start_replay()

            # Step 2. Sample an action, replaying Q(i,m) when M_t is non-empty.
            replay_m = self._active_replay_index()
            if replay_m is None:
                tree = self.model if self.is_fitted else None
                epsilon = (
                    self._block_epsilon(self._block_index)
                    if override_epsilon is None
                    else override_epsilon
                )
            else:
                policy = self._block_policies[replay_m]
                tree = policy.model
                epsilon = policy.epsilon
            if tree is not None:
                predictions_stacked[i, :] = tree.predict(features_np[i : i + 1])[0]

            if missing_arms:
                arm = int(self._rng.choice(missing_arms))
                missing_arms.remove(arm)
                prob = 1.0 / self._act_dim
            else:
                row_preds = predictions_stacked[i]
                tied = np.flatnonzero(np.isclose(row_preds, np.max(row_preds)))
                greedy_arm = int(self._rng.choice(tied))
                probs = np.full(self._act_dim, epsilon / self._act_dim)
                probs[greedy_arm] += 1.0 - epsilon
                arm = int(self._rng.choice(self._act_dim, p=probs))
                prob = float(probs[arm])

            predictions_stacked[i, arm] = predictions_stacked[i].max() + 1.0
            open_replays = tuple(r for r in self._replays if r.rounds_left > 0)
            self._pending.append((arm, prob, open_replays))
            for replay in open_replays:
                replay.rounds_left -= 1
            self._last_prob_by_arm[arm] = prob

        q_values = predictions_stacked
        return q_values

    def update_model(
        self, features: np.ndarray, actions: np.ndarray, rewards: np.ndarray | float, backward: bool = True
    ) -> float:
        # An evaluation batch must not enter the replay memory: evaluate_model
        # calls this with backward=False twice per episode, so ingesting here
        # would record every episode three times over. Guarding only the fit
        # below is not enough - the ingest loop also grows current_block and
        # appends rows to every open replay, which distorts the block schedule.
        if not backward:
            return 0.0
        features_np = np.asarray(features, dtype=float)
        if features_np.ndim == 1:
            features_np = features_np.reshape(1, -1)

        if isinstance(actions, np.ndarray):
            actions_np = np.asarray(actions).astype(int).flatten()
        else:
            actions_np = np.array([int(actions)])

        if isinstance(rewards, np.ndarray):
            rewards_np = np.asarray(rewards, dtype=float).flatten()
        else:
            rewards_np = np.full(len(actions_np), float(rewards))

        if len(actions_np) == 1 and features_np.shape[0] > 1:
            actions_np = np.repeat(actions_np, features_np.shape[0])
        if len(rewards_np) == 1 and features_np.shape[0] > 1:
            rewards_np = np.repeat(rewards_np, features_np.shape[0])
        if len(actions_np) != features_np.shape[0]:
            raise ValueError(
                "actions batch size mismatch: "
                f"expected {features_np.shape[0]}, got {len(actions_np)}"
            )
        if len(rewards_np) != features_np.shape[0]:
            raise ValueError(
                "rewards batch size mismatch: "
                f"expected {features_np.shape[0]}, got {len(rewards_np)}"
            )

        pending = (
            self._pending
            if len(self._pending) == features_np.shape[0]
            else None
        )
        self._pending = []

        inserted = 0
        for i in range(len(actions_np)):
            arm = int(actions_np[i])
            if not 0 <= arm < self._act_dim:
                raise ValueError(f"action out of range: {arm}")

            raw_reward = max(0.0, float(rewards_np[i]))
            if raw_reward > self.global_r_max:
                ratio = self.global_r_max / raw_reward
                for target in self.memory["rewards"]:
                    target *= ratio
                self.global_r_max = raw_reward
            scaled_reward = raw_reward / self.global_r_max

            if pending is not None and pending[i][0] == arm:
                prob = pending[i][1]
            else:
                prob = self._last_prob_by_arm.get(arm, 1.0 / self._act_dim)

            ips_target = np.zeros(self._act_dim, dtype=float)
            ips_target[arm] = scaled_reward / prob
            self.memory["features"].append(features_np[i])
            self.memory["actions"].append(arm)
            self.memory["probs"].append(prob)
            self.memory["rewards"].append(ips_target)
            row = len(self.memory["rewards"]) - 1
            self.current_block.append(row)
            if pending is not None and pending[i][0] == arm:
                for replay in pending[i][2]:
                    replay.rows.append(row)
            else:
                for replay in self._replays:
                    replay.rows.append(row)
            inserted += 1

        if backward and inserted and len(self.memory["rewards"]) > 0:
            X = np.array(self.memory["features"])
            Y = np.array(self.memory["rewards"])
            self.model.fit(X, Y)
            self.is_fitted = True
            if self._run_expired_replays(X, Y):
                return 0.0
            self._check_shift(X, Y)

        return 0.0


    def _run_expired_replays(self, X: np.ndarray, Y: np.ndarray) -> bool:
        expired = [r for r in self._replays if r.rounds_left <= 0]
        if not expired:
            return False
        self._replays = [r for r in self._replays if r.rounds_left > 0]

        prior = list(range(self._block_start_row))
        for replay in expired:
            if not replay.rows or not prior:
                continue
            self.replay_tests += 1
            if self._end_of_replay_test(replay, prior, X, Y):
                self.replay_restarts += 1
                self._hard_reset()
                return True
        return False

    def _end_of_replay_test(
        self, replay: _Replay, prior: List[int], X: np.ndarray, Y: np.ndarray
    ) -> bool:

        rows_a = np.asarray(replay.rows, dtype=int)
        rows_b = np.asarray(prior, dtype=int)
        policy = self._block_policies[replay.index]

        threshold = self._nu * float(
            np.sqrt(self._init_block_length / max(1, policy.length))
        )

        candidates = self._candidate_policies(X)
        regret_a = self._empirical_regret(rows_a, Y, candidates)
        regret_b = self._empirical_regret(rows_b, Y, candidates)
        for name, r_a in regret_a.items():
            r_b = regret_b.get(name)
            if r_b is None:
                continue
            if r_a - 4.0 * r_b >= threshold:       
                self.replay_fail_reasons["regret_worse"] += 1
                return True
            if r_b - 4.0 * r_a >= threshold:      
                self.replay_fail_reasons["regret_better"] += 1
                return True

        for actions in candidates.values():         
            v_a = self._propensity_variance(rows_a, actions, policy, X)
            v_b = self._propensity_variance(rows_b, actions, policy, X)
            if v_a - self._replay_var_ratio * v_b >= 0.0:
                self.replay_fail_reasons["variance"] += 1
                return True
        return False

    def _candidate_policies(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        n_rows = X.shape[0]
        candidates: Dict[str, np.ndarray] = {
            f"const{a}": np.full(n_rows, a, dtype=int) for a in range(self._act_dim)
        }
        for m, policy in enumerate(self._block_policies):
            if policy.model is not None:
                candidates[f"block{m}"] = np.argmax(policy.model.predict(X), axis=1)
        if self.is_fitted:
            candidates["current"] = np.argmax(self.model.predict(X), axis=1)
        return candidates

    def _empirical_regret(
        self, rows: np.ndarray, Y: np.ndarray, candidates: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        values: Dict[str, float] = {}
        for name, actions in candidates.items():
            value = self._snips(rows, actions[rows], Y)
            if value is not None:
                values[name] = value
        if not values:
            return {}
        best = max(values.values())
        return {name: best - value for name, value in values.items()}

    def _snips(
        self, rows: np.ndarray, policy_actions: np.ndarray, Y: np.ndarray
    ) -> Optional[float]:
        actions = np.asarray(self.memory["actions"], dtype=int)[rows]
        probs = np.asarray(self.memory["probs"], dtype=float)[rows]
        matched = actions == policy_actions
        if not matched.any():
            return None
        weight_sum = float((matched / probs).sum())
        if weight_sum <= 0.0:
            return None
        return float(Y[rows, policy_actions].sum() / weight_sum)

    def _propensity_variance(
        self,
        rows: np.ndarray,
        policy_actions: np.ndarray,
        policy: _BlockPolicy,
        X: np.ndarray,
    ) -> float:
        if policy.model is None:
            return float(self._act_dim)
        greedy = np.argmax(policy.model.predict(X[rows]), axis=1)
        floor = policy.epsilon / self._act_dim
        probs = np.where(policy_actions[rows] == greedy, 1.0 - policy.epsilon + floor, floor)
        return float(np.mean(1.0 / probs))

    def _check_shift(self, X: np.ndarray, Y: np.ndarray) -> None:
        pulled_arms = {int(a) for a in self.memory["actions"]}
        if len(pulled_arms) < self._act_dim:
            return
        if len(self.current_block) < self.block_length:
            return

        if self.prev_block:
            v_old = self._policy_value(self.prev_block, X, Y)
            v_new = self._policy_value(self.current_block, X, Y)
            if v_old is None or v_new is None:
                self._soft_reset()
                return
            if abs(v_old - v_new) > self._nu:
                self._hard_reset()
                return
            self.block_length = min(self.block_length * 2, self._max_block_length)
        self._soft_reset()

    def _policy_value(
        self, block: List[int], X: np.ndarray, Y: np.ndarray
    ) -> Optional[float]:
        rows = np.asarray(block, dtype=int)
        greedy = np.argmax(self.model.predict(X[rows]), axis=1)
        actions = np.asarray(self.memory["actions"])[rows]
        probs = np.asarray(self.memory["probs"])[rows]
        matched = actions == greedy
        if not matched.any():
            return None
        weight_sum = float((matched / probs).sum())
        return float(Y[rows, greedy].sum() / weight_sum)

    def reset_model(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self.global_r_max = 1.0
        self._last_prob_by_arm = {}
        self.replay_starts = 0
        self.replay_tests = 0
        self.replay_restarts = 0
        self.replay_fail_reasons = {"regret_worse": 0, "regret_better": 0, "variance": 0}
        self._hard_reset()

    def _hard_reset(self):
        self.model = DecisionTreeRegressor(
            criterion='squared_error',
            max_depth=self._max_depth,
            min_samples_leaf=self._min_samples_leaf,
            random_state=42,
        )
        self.is_fitted = False
        self.memory = {
                "features": list(),
                "actions": list(),
                "probs": list(),
                "rewards": list(),
            }
        self.block_length = self._init_block_length
        self.current_block = list()
        self.prev_block = list()
        self._pending = list()
        self._block_index = 0
        self._block_start_row = 0
        self._replays = list()
        self._block_policies = [
            _BlockPolicy(model=None, epsilon=self._block_epsilon(0), length=self.block_length)
        ]

    def _soft_reset(self):
        self.prev_block = self.current_block
        self.current_block = list()
        self._block_index += 1
        self._block_start_row = len(self.memory["rewards"])
        self._replays = list()
        self._block_policies.append(
            _BlockPolicy(
                model=copy.deepcopy(self.model) if self.is_fitted else None,
                epsilon=self._block_epsilon(self._block_index),
                length=self.block_length,
            )
        )

        for stale in self._block_policies[: -self._max_snapshots]:
            stale.model = None


# ---------------------------------------------------------------------------
# PPO
#
# Policy gradient rather than a bandit: the models above all estimate a reward
# per arm and then argmax it, which needs a correct reward model everywhere in
# the action space. PPO never estimates the reward - it moves the probability
# of an action up or down by the sign of its advantage, so a misordering it
# never samples costs nothing, and the per-arm staleness that strands an arm in
# a per-arm ridge model has no equivalent here: one shared network means every
# update moves every arm's logit.
#
# What it pays for that is sample efficiency, and this environment is poor in
# samples: a 2160 s run at a 5 s reward window is about 107 decisions, roughly
# 15 per failure phase. That number drives every default below - a linear
# policy head by default (51 parameters for PBFT, the same count as discrete_linucb),
# a 8-decision update, and a uniform-mixture exploration floor carried over
# from discrete_linucb so that no arm's probability can reach zero and strand it.
#
# No torch. learning_agent's numeric dependencies are numpy, scipy and
# sklearn, and this code runs on the cluster nodes; a network this small does
# not need an autograd engine, and the gradients below are written out.
# ---------------------------------------------------------------------------


def signed_log_context(rows: np.ndarray, n_features: int) -> np.ndarray:
    """DiscountedLinUCB's context transform, as a function.

    Raw report fields span several orders of magnitude - transaction counts in
    the hundreds of thousands beside view-change counts of 0 or 1 - so a
    network on raw rows would see one input dominate. Signed-log compresses the
    scale, the row norm removes the overall level (which is mostly the offered
    load), and the leading 1 is the bias column.

    Deliberately the same transform discrete_linucb uses, so that a PPO-versus-discrete_linucb
    comparison is a comparison of the two learning rules and not of two
    different featurizations.
    """
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != n_features:
        actual = values.shape[1] if values.ndim == 2 else values.shape
        raise ValueError(f"Expected {n_features} features, got {actual}")
    if not np.all(np.isfinite(values)):
        raise ValueError("features must contain only finite values")
    transformed = np.sign(values) * np.log1p(np.abs(values))
    norms = np.linalg.norm(transformed, axis=1, keepdims=True)
    transformed = transformed / np.maximum(norms, 1.0)
    return np.hstack((np.ones((len(transformed), 1)), transformed))


class _TinyMLP:
    """tanh MLP with hand-written backprop, seeded from a numpy generator.

    ``sizes`` is (input, *hidden, output); with no hidden sizes this is a
    single affine map, which is the default both PPO models use. Weights are
    drawn from the caller's generator rather than a global one so that every
    replica in sharing mode builds bit-identical parameters.
    """

    def __init__(
        self,
        sizes: Sequence[int],
        rng: np.random.Generator,
        output_scale: float = 0.01,
        output_bias: float = 0.0,
    ) -> None:
        self.sizes = tuple(int(size) for size in sizes)
        if len(self.sizes) < 2 or any(size <= 0 for size in self.sizes):
            raise ValueError(f"invalid layer sizes {self.sizes}")
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        last = len(self.sizes) - 2
        for index in range(len(self.sizes) - 1):
            fan_in, fan_out = self.sizes[index], self.sizes[index + 1]
            # The output layer starts near zero so the initial policy is close
            # to uniform: a run has ~107 decisions and cannot afford to spend
            # its first phase walking back a random initial preference.
            scale = output_scale if index == last else math.sqrt(1.0 / fan_in)
            self.weights.append(
                rng.normal(0.0, scale, size=(fan_in, fan_out)).astype(np.float64)
            )
            bias = np.zeros(fan_out, dtype=np.float64)
            if index == last:
                bias += float(output_bias)
            self.biases.append(bias)

    @property
    def parameters(self) -> List[np.ndarray]:
        return self.weights + self.biases

    def forward(self, inputs: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """(output, per-layer inputs) - the second is what backward() needs."""
        layer_inputs = [np.asarray(inputs, dtype=np.float64)]
        last = len(self.weights) - 1
        output = layer_inputs[0]
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            output = layer_inputs[index] @ weight + bias
            if index < last:
                layer_inputs.append(np.tanh(output))
        return output, layer_inputs

    def backward(
        self, layer_inputs: List[np.ndarray], delta: np.ndarray
    ) -> List[np.ndarray]:
        """Gradients for :attr:`parameters`, given dLoss/dOutput."""
        weight_grads: List[Optional[np.ndarray]] = [None] * len(self.weights)
        bias_grads: List[Optional[np.ndarray]] = [None] * len(self.biases)
        for index in range(len(self.weights) - 1, -1, -1):
            inputs = layer_inputs[index]
            weight_grads[index] = inputs.T @ delta
            bias_grads[index] = delta.sum(axis=0)
            if index > 0:
                # tanh' = 1 - tanh^2, and layer_inputs[index] is that tanh.
                delta = (delta @ self.weights[index].T) * (1.0 - inputs * inputs)
        return list(weight_grads) + list(bias_grads)


class _Adam:
    """Adam over a fixed list of numpy arrays, updated in place."""

    def __init__(
        self,
        parameters: Sequence[np.ndarray],
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self._lr = float(lr)
        self._beta1 = float(beta1)
        self._beta2 = float(beta2)
        self._eps = float(eps)
        self._step_count = 0
        self._first = [np.zeros_like(value) for value in parameters]
        self._second = [np.zeros_like(value) for value in parameters]

    def step(
        self, parameters: Sequence[np.ndarray], gradients: Sequence[np.ndarray]
    ) -> None:
        self._step_count += 1
        bias1 = 1.0 - self._beta1**self._step_count
        bias2 = 1.0 - self._beta2**self._step_count
        for index, (value, grad) in enumerate(zip(parameters, gradients)):
            self._first[index] *= self._beta1
            self._first[index] += (1.0 - self._beta1) * grad
            self._second[index] *= self._beta2
            self._second[index] += (1.0 - self._beta2) * grad * grad
            step = (self._first[index] / bias1) / (
                np.sqrt(self._second[index] / bias2) + self._eps
            )
            value -= self._lr * step


def _clip_global_norm(
    gradients: List[np.ndarray], max_norm: float
) -> List[np.ndarray]:
    if max_norm <= 0.0:
        return gradients
    total = math.sqrt(sum(float(np.sum(g * g)) for g in gradients))
    if total <= max_norm or total == 0.0:
        return gradients
    factor = max_norm / total
    return [g * factor for g in gradients]


class PPOBase(TensorModel):
    """Clipped-objective PPO shared by the discrete and continuous policies.

    Subclasses supply the policy head: how many numbers the actor emits, how to
    sample from them, and the log-density, entropy and their derivatives with
    respect to those numbers. Everything structural lives here - the rollout
    buffer, the advantage estimate, the clipped surrogate, the value loss, the
    KL early stop and Adam.

    **One decision is one step, and the default discount is zero.** The reward
    an episode receives is measured over the window that follows its action, so
    the immediate reward already contains that action's effect and the natural
    estimator is ``A = r - V(s)``. ``gamma > 0`` and GAE are implemented and
    reachable, but a longer horizon is not free here: a phase is about 15
    decisions, so a discount with a memory longer than that mixes two regimes'
    rewards into one return. discrete_linucb's knob ablation found discounting to be
    its single worst-performing knob, which is the same effect seen from the
    other side.

    **Old log-probabilities are recomputed at flush, not stored at sampling.**
    The parameters only change inside :meth:`_flush`, so when a flush begins the
    live parameters *are* the ones every buffered decision was drawn under, and
    evaluating them there is exact rather than an approximation. It also means
    the interface does not have to carry a log-probability from
    ``predict_action`` to ``update_model``, which it has no slot for.

    **The defaults are tuned for 107 decisions, and two of them are unusual.**
    ``update_every=4`` rather than a few thousand steps: there is no large
    rollout to collect, and four decisions is the smallest batch an advantage
    can be standardized over. ``entropy_coef=0.3`` is some thirty times the
    textbook value, and it is doing a different job here - with a regime that
    changes every ~15 decisions, a policy that commits is a policy that is
    wrong for the whole of the next phase. On a seven-phase synthetic built
    from the measured per-(phase, arm) throughputs of
    local/pbft_c6525100g_utah_2 (10 seeds), raising it from 0.0 to 0.3 moved
    mean throughput 24084 -> 24840 at 15 decisions per phase and per-phase
    accuracy 0.546 -> 0.615 at 150, and cost nothing at either. Entropy is the
    only thing in PPO that answers non-stationarity: there is no discount on
    past evidence the way discrete_linucb has one, so a collapsed policy would never
    find out the regime had moved.

    **The action trained on is the action that ran.** A latency floor, or any
    other constraint layered on top, can replace the sampled action before it
    reaches the cluster, and the agent reports back what was applied. That
    action is what enters the buffer, so the ratio is taken between the new and
    old policy *at the action the environment actually saw*. The behaviour
    density in that case is the constrained one rather than the policy's own,
    which this does not attempt to correct for; the constraint is treated as
    part of the environment.
    """

    def __init__(
        self,
        n_features: int,
        head_dim: int,
        hidden_sizes: Sequence[int] = (),
        actor_lr: float = 0.05,
        critic_lr: float = 0.05,
        clip_ratio: float = 0.2,
        epochs: int = 10,
        update_every: int = 4,
        gamma: float = 0.0,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.3,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.02,
        # 0 since 2026-10-19 (was 0.05; Kevin, runs/2026-10-19-linucb-discount): as LinUCB's, the uniform share
        # cost HotStuff up to 0.38 and nothing measurable elsewhere.
        exploration_floor: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        if n_features <= 0:
            raise ValueError(f"n_features must be > 0, got {n_features}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be > 0, got {head_dim}")
        if not 0.0 < clip_ratio < 1.0:
            raise ValueError(f"clip_ratio must be in (0, 1), got {clip_ratio}")
        if epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {epochs}")
        if update_every <= 0:
            raise ValueError(f"update_every must be > 0, got {update_every}")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        if not 0.0 <= gae_lambda <= 1.0:
            raise ValueError(f"gae_lambda must be in [0, 1], got {gae_lambda}")
        if entropy_coef < 0.0:
            raise ValueError(f"entropy_coef must be >= 0, got {entropy_coef}")
        if not 0.0 <= exploration_floor < 1.0:
            raise ValueError(
                f"exploration_floor must be in [0, 1), got {exploration_floor}"
            )
        if target_kl < 0.0:
            raise ValueError(f"target_kl must be >= 0, got {target_kl}")

        self._n_features = int(n_features)
        self._context_dim = self._n_features + 1
        self._head_dim = int(head_dim)
        self._hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self._actor_lr = float(actor_lr)
        self._critic_lr = float(critic_lr)
        self._clip_ratio = float(clip_ratio)
        self._epochs = int(epochs)
        self._update_every = int(update_every)
        self._gamma = float(gamma)
        self._gae_lambda = float(gae_lambda)
        self._entropy_coef = float(entropy_coef)
        self._max_grad_norm = float(max_grad_norm)
        self._target_kl = float(target_kl)
        self._exploration_floor = float(exploration_floor)
        self._seed = seed
        self.reset_model()

    # -- head interface -----------------------------------------------------

    def _head_terms(
        self, head: np.ndarray, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(log_prob, entropy, d log_prob / d head, d entropy / d head)."""
        raise NotImplementedError

    @property
    def _output_bias(self) -> float:
        return 0.0

    # -- plumbing -----------------------------------------------------------

    def _contexts(self, features: np.ndarray) -> np.ndarray:
        return signed_log_context(features, self._n_features)

    def _actor_head(self, contexts: np.ndarray) -> np.ndarray:
        head, _ = self._actor.forward(contexts)
        return head

    def _values(self, contexts: np.ndarray) -> np.ndarray:
        values, _ = self._critic.forward(contexts)
        return values.reshape(-1)

    def _advantages(
        self, rewards: np.ndarray, values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """GAE(lambda). At the default gamma=0 this is exactly r - V(s).

        The buffer is a slice out of a continuing stream, not an episode, so
        the final decision has no successor to bootstrap from and is treated as
        terminal. With gamma=0 that choice is inert; with gamma>0 it makes the
        last entry of each flush slightly pessimistic.
        """
        count = len(rewards)
        advantages = np.zeros(count, dtype=np.float64)
        running = 0.0
        for index in range(count - 1, -1, -1):
            next_value = values[index + 1] if index + 1 < count else 0.0
            delta = rewards[index] + self._gamma * next_value - values[index]
            running = delta + self._gamma * self._gae_lambda * running
            advantages[index] = running
        return advantages, advantages + values

    def _flush(self) -> None:
        contexts = np.vstack(self._buffer_contexts)
        actions = np.asarray(self._buffer_actions, dtype=np.float64)
        rewards = np.asarray(self._buffer_rewards, dtype=np.float64)
        self._buffer_contexts.clear()
        self._buffer_actions.clear()
        self._buffer_rewards.clear()

        values = self._values(contexts)
        advantages, returns = self._advantages(rewards, values)
        spread = float(advantages.std())
        if spread > 1e-8:
            advantages = (advantages - advantages.mean()) / spread
        else:
            # Every decision in this batch did equally well. Nothing to learn
            # about the policy from it; the critic still gets the batch.
            advantages = np.zeros_like(advantages)

        # Exact, not an approximation: parameters have not moved since these
        # decisions were drawn. See the class docstring.
        old_log_probs, _, _, _ = self._head_terms(
            self._actor_head(contexts), actions
        )

        batch = float(len(rewards))
        for _ in range(self._epochs):
            head, layer_inputs = self._actor.forward(contexts)
            log_probs, entropy, d_log_probs, d_entropy = self._head_terms(
                head, actions
            )
            ratios = np.exp(np.clip(log_probs - old_log_probs, -20.0, 20.0))
            unclipped = ratios * advantages
            clipped = (
                np.clip(ratios, 1.0 - self._clip_ratio, 1.0 + self._clip_ratio)
                * advantages
            )
            # d(-min(unclipped, clipped))/d log_prob. The clipped branch has
            # zero gradient only where the ratio is actually outside the band;
            # inside it the two branches are the same number.
            inside = (ratios > 1.0 - self._clip_ratio) & (
                ratios < 1.0 + self._clip_ratio
            )
            active = (unclipped <= clipped) | inside
            surrogate_grad = -(advantages * ratios) * active / batch
            delta = (
                surrogate_grad[:, None] * d_log_probs
                - (self._entropy_coef / batch) * d_entropy
            )
            actor_grads = self._actor.backward(layer_inputs, delta)
            self._actor_optimizer.step(
                self._actor.parameters,
                _clip_global_norm(actor_grads, self._max_grad_norm),
            )

            predicted, critic_inputs = self._critic.forward(contexts)
            critic_delta = (predicted.reshape(-1) - returns)[:, None] / batch
            critic_grads = self._critic.backward(critic_inputs, critic_delta)
            self._critic_optimizer.step(
                self._critic.parameters,
                _clip_global_norm(critic_grads, self._max_grad_norm),
            )

            # Sample KL(old || new). Positive and roughly the reverse of the
            # usual estimator, but it is the one that is cheap here.
            approx_kl = float(np.mean(old_log_probs - log_probs))
            if self._target_kl > 0.0 and approx_kl > self._target_kl:
                break

    # -- TensorModel --------------------------------------------------------

    def update_model(
        self,
        features: np.ndarray,
        actions: np.ndarray,
        rewards: "np.ndarray | float",
        backward: bool = True,
    ) -> float:
        if not backward:
            return 0.0

        contexts = self._contexts(features)
        # Values per decision: 1, or 2 for ContinuousPPO on SBFT (timeout, wait).
        width = int(getattr(self, "_action_width", 1))
        action_values = np.asarray(actions, dtype=np.float64).reshape(-1)
        reward_values = np.asarray(rewards, dtype=np.float64).reshape(-1)
        batch_size = len(contexts)
        if action_values.size == width and batch_size > 1:
            action_values = np.tile(action_values, batch_size)
        if reward_values.size == 1 and batch_size > 1:
            reward_values = np.repeat(reward_values, batch_size)
        if action_values.size != batch_size * width:
            raise ValueError(
                f"actions batch size mismatch: expected {batch_size} x {width}, "
                f"got {action_values.size}"
            )
        if reward_values.size != batch_size:
            raise ValueError(
                f"rewards batch size mismatch: expected {batch_size}, "
                f"got {reward_values.size}"
            )
        if not np.all(np.isfinite(action_values)):
            raise ValueError("actions must contain only finite values")
        if not np.all(np.isfinite(reward_values)):
            raise ValueError("rewards must contain only finite values")

        rows = action_values.reshape(batch_size, width)
        for context, action, reward in zip(contexts, rows, reward_values):
            self._buffer_contexts.append(context.reshape(1, -1))
            self._buffer_actions.append(
                self._store_action(float(action[0]) if width == 1 else action)
            )
            self._buffer_rewards.append(float(reward))
            if len(self._buffer_rewards) >= self._update_every:
                self._flush()
        return 0.0

    def _store_action(self, action: float) -> float:
        """Map a reported action to the form the head's log-density takes."""
        return float(action)

    # -- checkpoint ---------------------------------------------------------

    def _named_parameters(self) -> Dict[str, np.ndarray]:
        named: Dict[str, np.ndarray] = {}
        for prefix, network in (("actor", self._actor), ("critic", self._critic)):
            for index, (weight, bias) in enumerate(
                zip(network.weights, network.biases)
            ):
                named[f"{prefix}_w{index}"] = weight
                named[f"{prefix}_b{index}"] = bias
        return named

    def state_arrays(self) -> Dict[str, np.ndarray]:
        """Copies of the actor and critic weights and biases.

        Optimizer moments, the rollout buffer and the RNG are not included: a
        loaded model starts with fresh Adam state and an empty buffer.
        """
        return {name: value.copy() for name, value in self._named_parameters().items()}

    def load_state_arrays(self, arrays: Dict[str, np.ndarray]) -> None:
        """Write :meth:`state_arrays` output into this model's parameters.

        In place, so the arrays Adam holds stay the network's. Every key and
        shape must match exactly.
        """
        named = self._named_parameters()
        missing = sorted(set(named) - set(arrays))
        extra = sorted(set(arrays) - set(named))
        if missing or extra:
            raise ValueError(
                f"PPO state keys do not match: missing {missing}, unexpected {extra}"
            )
        loaded = {}
        for name, target in named.items():
            value = np.asarray(arrays[name], dtype=np.float64)
            if value.shape != target.shape:
                raise ValueError(
                    f"PPO state {name} has shape {value.shape}, "
                    f"the model needs {target.shape}"
                )
            if not np.all(np.isfinite(value)):
                raise ValueError(f"PPO state {name} contains non-finite values")
            loaded[name] = value
        for name, target in named.items():
            target[...] = loaded[name]

    def reset_model(self) -> None:
        # The first decision after a reset is the opening action, not a sample (D10b).
        self._opened = False
        self._rng = np.random.default_rng(self._seed)
        sizes = (self._context_dim,) + self._hidden_sizes
        self._actor = _TinyMLP(
            sizes + (self._head_dim,),
            self._rng,
            output_bias=self._output_bias,
        )
        self._critic = _TinyMLP(sizes + (1,), self._rng)
        self._actor_optimizer = _Adam(self._actor.parameters, self._actor_lr)
        self._critic_optimizer = _Adam(self._critic.parameters, self._critic_lr)
        self._buffer_contexts: List[np.ndarray] = []
        self._buffer_actions: List[float] = []
        self._buffer_rewards: List[float] = []


class DiscretePPO(PPOBase):
    """PPO with a softmax policy over the discrete timeout ladder.

    The actor emits one logit per arm; the arm is sampled, not argmaxed, so
    exploration is the policy's own entropy rather than a bolted-on rule. The
    critic is a state-value baseline over the same context, which is what makes
    the update a comparison between "this arm" and "this context's average arm"
    instead of between arms whose rewards were measured under different load.

    **Why this is not another bandit.** Every model above learns a reward per
    arm and plays the argmax, so an arm that has not been pulled lately holds a
    stale estimate and comes back only on a confidence bonus - the starvation
    that has repeatedly stranded the long arm in this codebase. A softmax
    policy has no per-arm memory to go stale: one shared network means a single
    update moves every logit, and the floor below guarantees the probability of
    any arm is bounded away from zero no matter what the logits do.

    **The exploration floor is mixed into the distribution, not applied on top
    of it.** ``p = (1 - e) * softmax(z) + e / K`` is the behaviour policy, and
    it is also the density the ratio and the entropy are computed from, so the
    importance weights stay exact. Bolting an epsilon-greedy branch on after
    sampling - which is what discrete_linucb does - would make the behaviour policy
    something the update does not know about. At the default ``e = 0.05`` and
    three arms no arm can fall below a 1.7% chance of being tried.

    Sampled rather than greedy also means the reported score row has to carry
    the *sampled* arm as its maximum, since the codec argmaxes it. The row is
    the mixed probability vector with the sampled entry raised by one
    representable step, the same device DiscountedLinUCB uses to break ties.
    """

    def __init__(
        self,
        action_dim: int,
        n_features: int,
        hidden_sizes: Sequence[int] = (),
        opening_arm: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")
        if opening_arm is not None and not 0 <= opening_arm < action_dim:
            raise ValueError(f"opening_arm must be in [0, {action_dim}), got {opening_arm}")
        self._action_dim = int(action_dim)
        self._opening_arm = opening_arm
        super().__init__(
            n_features=n_features,
            head_dim=self._action_dim,
            hidden_sizes=hidden_sizes,
            **kwargs,
        )

    def _probabilities(
        self, head: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """(softmax, softmax mixed with uniform at the exploration floor)."""
        shifted = head - np.max(head, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        soft = exponentials / np.sum(exponentials, axis=1, keepdims=True)
        if self._exploration_floor <= 0.0:
            return soft, soft
        mixed = (1.0 - self._exploration_floor) * soft + (
            self._exploration_floor / self._action_dim
        )
        return soft, mixed

    def _head_terms(
        self, head: np.ndarray, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        soft, mixed = self._probabilities(head)
        indices = np.rint(actions).astype(int)
        rows = np.arange(len(indices))
        chosen = mixed[rows, indices]
        log_probs = np.log(np.maximum(chosen, 1e-12))

        # d mixed / d logit = (1 - e) * (diag(soft) - soft soft^T), so
        # d log mixed_a / d logit = (1 - e) * soft_a / mixed_a * (onehot - soft).
        one_hot = np.zeros_like(soft)
        one_hot[rows, indices] = 1.0
        scale = (1.0 - self._exploration_floor) * soft[rows, indices] / np.maximum(
            chosen, 1e-12
        )
        d_log_probs = scale[:, None] * (one_hot - soft)

        log_mixed = np.log(np.maximum(mixed, 1e-12))
        entropy = -np.sum(mixed * log_mixed, axis=1)
        # dH/dz_j = -(1 - e) * soft_j * (c_j - sum_k c_k soft_k), c = 1 + log p.
        coefficients = 1.0 + log_mixed
        weighted = np.sum(coefficients * soft, axis=1, keepdims=True)
        d_entropy = (
            -(1.0 - self._exploration_floor) * soft * (coefficients - weighted)
        )
        return log_probs, entropy, d_log_probs, d_entropy

    def predict_action(self, features: np.ndarray) -> np.ndarray:
        contexts = self._contexts(features)
        _, mixed = self._probabilities(self._actor_head(contexts))
        scores = np.array(mixed, dtype=np.float64, copy=True)
        for row in scores:
            if not self._opened and self._opening_arm is not None:
                # The opening arm (the ladder's middle rung, D10b) as the first decision.
                selected = int(self._opening_arm)
                self._opened = True
            else:
                selected = int(self._rng.choice(self._action_dim, p=row / row.sum()))
            best = float(np.max(row))
            # The codec argmaxes this row, so the sample has to be its maximum.
            row[selected] = np.nextafter(best, np.inf)
        return scores

    def _store_action(self, action: float) -> float:
        index = int(round(action))
        if index < 0 or index >= self._action_dim:
            raise ValueError(
                f"actions must be between 0 and {self._action_dim - 1}, got {action}"
            )
        return float(index)
