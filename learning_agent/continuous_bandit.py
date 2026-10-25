from collections import deque
from dataclasses import dataclass
import math
import os
from typing import Any, Deque, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import betaln, digamma, polygamma

import agent_pb2
from latency_floor import LatencyFloor, latency_indices, signal_fields
from reward_utils import reward_normalizer_tps
from models import (
    ActionCodec,
    PPOBase,
    TensorModel,
    get_timeout_values_ms,
    tendermint_timeouts_from_propose,
)


class ContinuousTimeoutActionCodec(ActionCodec[agent_pb2.Timeout]):
    """Maps normalized continuous actions to protocol timeout messages."""

    def __init__(
        self,
        protocol: int,
        bounds_ms: Tuple[int, int] = (100, 10000),
        wait_bounds_ms: Optional[Tuple[int, int]] = None,
    ):
        self._protocol = int(protocol)
        # timeout_values = get_timeout_values_ms(self._protocol)
        # lower_ms = min(timeout_values)
        # upper_ms = max(timeout_values)
        lower_ms, upper_ms = int(bounds_ms[0]), int(bounds_ms[1])
        if lower_ms <= 0 or upper_ms <= lower_ms:
            raise ValueError(
                f"Invalid continuous timeout bounds: {lower_ms}..{upper_ms}ms"
            )

        if self._protocol in (
            agent_pb2.PROTOCOL_PBFT,
            agent_pb2.PROTOCOL_SMARTBFT,
            agent_pb2.PROTOCOL_TENDERMINT,
            agent_pb2.PROTOCOL_HOTSTUFF,
            agent_pb2.PROTOCOL_AUTOBAHN,
        ):
            if wait_bounds_ms is not None:
                raise ValueError(
                    "wait_bounds_ms is SBFT's fast-path wait; "
                    f"{agent_pb2.Protocol.Name(self._protocol)} has one timeout dimension"
                )
            self._action_dim = 1
            self._bounds_ms: Tuple[Tuple[int, int], ...] = ((lower_ms, upper_ms),)
        elif self._protocol == agent_pb2.PROTOCOL_SBFT:
            # Dimension 0 the election timeout T, dimension 1 the fast-path wait W.
            if wait_bounds_ms is None:
                raise ValueError("SBFT continuous actions need wait_bounds_ms")
            wait_lower_ms, wait_upper_ms = int(wait_bounds_ms[0]), int(wait_bounds_ms[1])
            if wait_lower_ms <= 0 or wait_upper_ms <= wait_lower_ms:
                raise ValueError(
                    f"Invalid continuous wait bounds: {wait_lower_ms}..{wait_upper_ms}ms"
                )
            self._action_dim = 2
            self._bounds_ms = ((lower_ms, upper_ms), (wait_lower_ms, wait_upper_ms))
        else:
            raise ValueError(
                "Continuous timeout actions currently support only PBFT, "
                "SmartBFT, SBFT, Tendermint, HotStuff and Autobahn"
            )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def protocol(self) -> int:
        return self._protocol

    @property
    def bounds_ms(self) -> Tuple[Tuple[int, int], ...]:
        return self._bounds_ms

    def normalize_ms(self, timeout_ms: int, dimension: int = 0) -> float:
        """Public form of the log-space mapping that decode() inverts."""
        return self._normalize_value(int(timeout_ms), dimension)

    def denormalize_ms(self, normalized: float, dimension: int = 0) -> int:
        """Inverse of :meth:`normalize_ms`, without building a Timeout.

        A model that reasons in milliseconds - one whose action space is
        defined relative to a measured latency rather than to the codec's own
        bounds - needs to read back the timeout behind an action it is handed,
        which is otherwise only reachable through decode() and a protocol
        switch on the message it returns.
        """
        return self._denormalize_value(float(normalized), dimension)

    def _normalize_value(self, timeout_ms: int, dimension: int) -> float:
        lower_ms, upper_ms = self._bounds_ms[dimension]
        if timeout_ms < lower_ms or timeout_ms > upper_ms:
            raise ValueError(
                f"Timeout {timeout_ms}ms is outside continuous bounds "
                f"{lower_ms}..{upper_ms}ms"
            )
        numerator = math.log(float(timeout_ms)) - math.log(float(lower_ms))
        denominator = math.log(float(upper_ms)) - math.log(float(lower_ms))
        return numerator / denominator

    def _denormalize_value(self, normalized: float, dimension: int) -> int:
        lower_ms, upper_ms = self._bounds_ms[dimension]
        bounded = min(1.0, max(0.0, float(normalized)))
        log_value = math.log(float(lower_ms)) + bounded * (
            math.log(float(upper_ms)) - math.log(float(lower_ms))
        )
        return min(upper_ms, max(lower_ms, int(round(math.exp(log_value)))))

    def decode(self, action: np.ndarray) -> agent_pb2.Timeout:
        values = np.asarray(action, dtype=float).reshape(-1)
        if values.size != self._action_dim:
            raise ValueError(
                f"Expected {self._action_dim} continuous action values, "
                f"got {values.size}"
            )
        timeout_values = [
            self._denormalize_value(float(values[index].item()), index)
            for index in range(self._action_dim)
        ]

        if self._protocol == agent_pb2.PROTOCOL_PBFT:
            return agent_pb2.Timeout(
                pbft=agent_pb2.PbftTimeout(
                    election_timeout_milliseconds=timeout_values[0]
                )
            )
        if self._protocol == agent_pb2.PROTOCOL_SMARTBFT:
            return agent_pb2.Timeout(
                smartbft=agent_pb2.SmartbftTimeout(
                    election_timeout_milliseconds=timeout_values[0]
                )
            )
        if self._protocol == agent_pb2.PROTOCOL_SBFT:
            return agent_pb2.Timeout(
                sbft=agent_pb2.SbftTimeout(
                    election_timeout_milliseconds=timeout_values[0],
                    slow_path_timeout_milliseconds=timeout_values[1],
                )
            )
        if self._protocol == agent_pb2.PROTOCOL_HOTSTUFF:
            return agent_pb2.Timeout(
                hotstuff=agent_pb2.HotstuffTimeout(
                    timeout_delay_milliseconds=timeout_values[0]
                )
            )
        if self._protocol == agent_pb2.PROTOCOL_AUTOBAHN:
            return agent_pb2.Timeout(
                autobahn=agent_pb2.AutobahnTimeout(
                    timeout_delay_milliseconds=timeout_values[0]
                )
            )
        if self._protocol != agent_pb2.PROTOCOL_TENDERMINT:
            raise ValueError(f"Unsupported protocol {self._protocol}")
        propose_timeout_ms, prevote_timeout_ms, precommit_timeout_ms = (
            tendermint_timeouts_from_propose(timeout_values[0])
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
            if self._protocol == agent_pb2.PROTOCOL_PBFT:
                action = action.pbft
            elif self._protocol == agent_pb2.PROTOCOL_SMARTBFT:
                action = action.smartbft
            elif self._protocol == agent_pb2.PROTOCOL_SBFT:
                action = action.sbft
            elif self._protocol == agent_pb2.PROTOCOL_HOTSTUFF:
                action = action.hotstuff
            elif self._protocol == agent_pb2.PROTOCOL_AUTOBAHN:
                action = action.autobahn
            elif self._protocol == agent_pb2.PROTOCOL_TENDERMINT:
                action = action.tendermint
            else:
                raise ValueError(f"Unsupported protocol {self._protocol}")

        if self._protocol in (
            agent_pb2.PROTOCOL_PBFT,
            agent_pb2.PROTOCOL_SMARTBFT,
        ):
            timeout_values = [
                int(getattr(action, "election_timeout_milliseconds"))
            ]
        elif self._protocol == agent_pb2.PROTOCOL_SBFT:
            timeout_values = [
                int(getattr(action, "election_timeout_milliseconds")),
                int(getattr(action, "slow_path_timeout_milliseconds")),
            ]
        elif self._protocol in (
            agent_pb2.PROTOCOL_HOTSTUFF,
            agent_pb2.PROTOCOL_AUTOBAHN,
        ):
            timeout_values = [int(getattr(action, "timeout_delay_milliseconds"))]
        else:
            propose_timeout_ms = int(
                getattr(action, "propose_timeout_milliseconds")
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

        normalized = [
            self._normalize_value(timeout_ms, index)
            for index, timeout_ms in enumerate(timeout_values)
        ]
        return np.asarray(normalized, dtype=float)


class NormalizedLatencyFloor:
    """:class:`LatencyFloor` expressed in the codec's normalized action space.

    The continuous models search in [0, 1] over log-spaced milliseconds, so the
    constraint has to be mapped through the same transform before it can bound
    a candidate set.
    """

    def __init__(
        self,
        action_codec: ContinuousTimeoutActionCodec,
        latency_index: Union[int, Sequence[int]],
        n_features: int,
        instances_index: Optional[int] = None,
        latency_floor: Optional[LatencyFloor] = None,
    ) -> None:
        latency_index = latency_indices(latency_index, n_features)
        if instances_index is not None and not 0 <= int(instances_index) < int(
            n_features
        ):
            raise ValueError(
                f"instances_index must be in [0, {n_features}), got {instances_index}"
            )
        self._action_codec = action_codec
        self._latency_index = latency_index
        self._instances_index = (
            None if instances_index is None else int(instances_index)
        )
        self._latency_floor = (
            LatencyFloor() if latency_floor is None else latency_floor
        )

    @property
    def floor_ms(self) -> float:
        return self._latency_floor.floor_ms

    def observe_features(self, row: np.ndarray) -> None:
        alive = True
        if self._instances_index is not None:
            alive = float(row[self._instances_index]) > 0.0
        self._latency_floor.observe(sum(float(row[i]) for i in self._latency_index), alive=alive)

    def lower_bound(self) -> float:
        """Smallest normalized action the constraint permits, in [0, 1)."""
        floor_ms = self._latency_floor.floor_ms
        if floor_ms <= 0.0:
            return 0.0
        lower_ms, upper_ms = self._action_codec.bounds_ms[0]
        if floor_ms <= lower_ms:
            return 0.0
        # A floor above the whole range would leave nothing to play; keep the
        # longest timeout available, as feasible_timeout_mask does.
        if floor_ms >= upper_ms:
            return 1.0
        return self._action_codec.normalize_ms(int(math.ceil(floor_ms)))

    def reset(self) -> None:
        self._latency_floor.reset()


@dataclass(frozen=True)
class HeuristicSpec:
    """continuous_heuristic's constants on one protocol.

    Kevin (2026-10-08): one rule, with per-protocol constants taken from the
    fixed-timeout calibration cells and the mechanics, never from oracle or
    evaluation runs. Cells: scripts/cloudlab/tmp/cal/pt/cells and sbcal (one spec
    phase per run at a constant timeout), judged by cal/pt/cell_judge.py.

    A window that committed sets T = ``multiplier`` x the ``quantile`` of
    ``signal`` over the last ``window`` committing windows, clamped to
    ``bounds_ms``. ``signal`` is one report field, or several joined by "+" and
    summed per window (latency_floor.signal_fields; SBFT, 2026-10-14). A window with at least ``storm_view_changes`` view changes is
    a storm: T doubles, and the T that stormed is not played again until the
    estimate falls by a third (a faster link). A window that committed nothing
    takes the feature window's length as its latency (ContinuousHeuristicTimeoutModel,
    2026-10-19). Any other window holds T. With
    ``storm_view_changes`` None the rule has no storm branch: on that protocol
    the signal itself rises steeply under a too-low T, and the view-change count
    is not a storm signal (Autobahn and HotStuff time out every few slots even
    at their best T).

    Shared choices (2026-10-08, documented after the review; values unchanged):
    ``bounds_ms`` are each protocol's ContextSpec action range, the range the
    GP and the other continuous models search, so every model plays the same
    space. ``window`` 1 follows a link change within one window; the replays of
    the cells (cal/pt/heur_replay.py) converge to the clean region from both
    sides at window 1 on SmartBFT, HotStuff and Autobahn. ``quantile`` has no
    effect at window 1. ``storm_view_changes`` 3 is a design choice, not a
    value the cells separate per window. Since restart D14 (2026-10-19) the count
    on PBFT, SBFT and SmartBFT covers the trailing 20 s episode, about 2x a 10 s
    count and including the warm-up's hold re-pins, so 3 is a looser test than it
    was per 7 s window (about 0.7x a 10 s count, 2026-10-09 to 10-19). The cells
    were measured per 10 s feature window: the SBFT storm cells run 2-16 view
    changes (c1 W500 dips to 2) and an escaping cell reaches 4 (c1 W150);
    SmartBFT storm cells run 0-20 per window and clean or best cells reach 5-7
    (cal_L4_T800, cal_L4_T1200). Per cell the counts do separate (PBFT,
    CppBedrock/adaptive_timers.MD 2026-09 Utah: storms 40-140 per phase, escapes
    1-6 per 120 s phase), so a storm is a sustained high count, and a single
    window at 3 can misfire either way (goal review, 2026-10-08).
    """

    signal: str
    multiplier: float
    window: int
    quantile: float
    bounds_ms: Tuple[int, int]
    storm_view_changes: Optional[int]


# k, all six alike (2026-10-10): fitted blind by scripts/cloudlab/tmp/cal/blindall/<protocol>/fit/
# heur_fit_sig.py on constant-T runs of blind.xml (stratified draws, none of the evaluation spec's
# values): per blind phase the rule's fixed point T = k x estimate(T), scored by the client judge's R;
# per k the median over the blind phases; k is the low edge of the plateau around the best (within
# 0.01), rounded up to 2 decimals. The link phases bound k from below and are well sampled; the attack
# phases that bound it from above are one or two draws. The real spec was a check only.
# Tendermint refit at 5,000 tx/s (2026-10-12, runs/2026-10-12-tm-recal5k, blindall bl5k_ runs, tps/5000): low edge
# 0.5146 -> 0.52 (was 0.53 at 4,000 tx/s; eval R +0.556 vs +0.557, indistinguishable).
# HotStuff refit at 105,000 tx/s (2026-10-12, runs/2026-10-12-recal3, blindall bl512_ runs, tps/105000): low edge
# 0.8276 -> 0.83 (was 0.96 at 10,000 tx/s; real-spec eval R +0.326 vs +0.315 at 0.96; Kevin approved the 0.13 move).
# Autobahn refit at 140,000 tx/s (2026-10-12, runs/2026-10-12-recal3, bl512_ runs): low edge 0.8959 -> 0.90 (was 0.94;
# real-spec eval R +0.3351 vs +0.3347, indistinguishable).
# Restart refit (2026-10-20, scripts/cloudlab/tmp/cal/restart/s3/heur/SUMMARY.md; restart-cal S1 runs): the blind score
# is the mean over the blind phases, not the median (RUNSPEC answer 12: the median let HotStuff and Tendermint fit
# k = 0.01, T 100 everywhere), with k >= the latency floor's share c when the signal is the floor's; D7 low edge
# unchanged; the signal is the hold-free one unless the end-to-end one has the strictly better blind mean (answer 11).
HEURISTIC_SPECS: Dict[int, HeuristicSpec] = {
    # A request's age at execution (acceptance to execution, so a little under the
    # reported latency): clean at T = 0.85 x p95 at 5+75 (300 vs 354) and 0.93 x
    # at 5+175 (600 vs 648), every T from 100 clean at base. The slow leader
    # inflates the latency only while T
    # keeps it (base + D120: T 150 escapes, 250 keeps it at p95 244; + D400:
    # T <= 400 escapes, 600 keeps it at 569), so the estimate is the minimum
    # over 4 windows, which an escape window resets to the healthy latency.
    # k 1.26 -> 1.21 (2026-10-13, runs/2026-10-13-recal-sbft-pbft S4): low edge of the blind
    # plateau at 20,000 req/s with 512 B requests (bl20_ cells; eval-best on the real cells 1.33).
    # Restart (2026-10-21, restart-cal7 calibration of the relay-on-skip engine, read as the 2f+1 aggregate the live rule
    # sees): the phase sum (hold-free, the floor's signal) k 8.231107 -> 8.24, blind mean +0.756, real-spec
    # +0.723; the p95 refit gave k 0.219597 (blind +0.752, real-spec +0.555) and the previous p95 k 1.21
    # scored blind +0.716, real-spec +0.734. Kevin chose the hold-free signal so that every protocol with one
    # uses it. Was p95 1.21.
    agent_pb2.PROTOCOL_PBFT: HeuristicSpec(
        "pre_prepare_latency_ms+prepare_latency_ms+commit_latency_ms", 3.96, 4, 0.0, (100, 3000), 3),
    # The sum of the three phase latencies (pre-prepare, prepare, commit; window
    # means, as the replicas report them) instead of the consensus p95 (Kevin,
    # 2026-10-13, runs/2026-10-13-recal-sbft-pbft): a leader's proposal hold
    # never enters the phase latencies (it precedes the pre-prepare), the link and
    # the straggler do, so a held leader is deposed at every k and the +150 link is
    # reached through the slow-path bound below. k 1.76 is the low edge of the blind
    # plateau at 20,000 req/s with 512 B requests (bl20_ cells; the consensus-p95
    # signal's blind low edge was 0.76, and its real-spec score was lower).
    # The slow-path bound and the wait W are in ContinuousHeuristicTimeoutModel
    # (SBFT_* below).
    # Restart (2026-10-21, restart-cal7 calibration of the relay-on-skip engine): k 4.282058 -> 4.29 (the only signal),
    # blind mean +0.581, real-spec 0.486. Was 1.76.
    agent_pb2.PROTOCOL_SBFT: HeuristicSpec(
        "pre_prepare_latency_ms+prepare_latency_ms+commit_latency_ms", 4.2, 4, 0.0, (100, 3000), 3),
    # A request's timer from arrival (forward T/2, then complain T/2), closed-loop
    # client. Cells with a 50 s base lead-in: at 5+50 T <= 600 storms and 800 is
    # clean (p95 351); at 5+150 1000 storms, 1200-1800 stall for 3-15 s and 2400
    # is clean (p95 967); the lowest clean ratios there are 2.28 and 2.48.
    # At base every T from 100 is clean (p95 41); under the 300 ms proposal delay
    # 300 is best and 800 keeps the slow leader. Storms run 3-18 view changes a
    # window, depositions 0-2.
    # Restart: proposal_to_decision_ms (hold-free) k 5.21981 -> 5.22, blind mean +0.306 against the p95's +0.305
    # (k 1.56); real-spec +0.218. Was p95 1.39.
    agent_pb2.PROTOCOL_SMARTBFT: HeuristicSpec(
        "proposal_to_decision_ms", 5.22, 1, 0.5, (100, 3000), 3),
    # The propose timeout of each round (prevote and precommit at T/3). p50, not
    # the mean: under the proposer hold only the attacked heights are slow, and
    # p50 stays flat in T while the mean follows it.
    # Restart: k 0.524849 -> 0.53 (the only signal; c 0.33 not binding), blind mean +0.566, real-spec +0.565.
    agent_pb2.PROTOCOL_TENDERMINT: HeuristicSpec(
        "p50_consensus_latency_ms", 0.53, 1, 0.5, (100, 3000), None),
    # One round is 2h (3h when a Nagle-delayed vote follows a timeout) and the
    # reported latency is three rounds. Floors: 160 storms and 200 is clean at
    # 5+50 (avg 333), 460 storms and 530 is clean at 5+150 (avg 934), so the
    # floor sits at 0.5-0.6 x the latency; under the slow-leader hold (relative
    # to the live timeout) 200 still loses at 5+50 + X400 and 300 is best, and
    # at base the lowest T wins.
    # Restart: proposal_to_qc_ms (hold-free) k 4.282058 -> 4.29, blind mean +0.292 against the avg's +0.290
    # (k 0.62); real-spec -0.130 on both signals (avg k 0.83 scored +0.274 on the same eval runs: the fitted k lands
    # T at 217-244 in the real spec's p2-p4 instead of about 300). Was avg 0.83.
    agent_pb2.PROTOCOL_HOTSTUFF: HeuristicSpec(
        "proposal_to_qc_ms", 4.29, 1, 0.5, (100, 3000), None),
    # Header first seen to commit, about one slot: the best measured T is 0.75 x
    # it at 5+50 (400 at 533) and 0.76 x at 5+150 (1000 at 1308); below the soft
    # floor the latency climbs steeply (250 at 5+50: 989). Base is flat in T (100-750).
    # Restart: prepare_to_commit_ms (hold-free) k 1.229821 -> 1.23, blind mean +0.351, real-spec +0.255. The blind
    # score kept the avg (+0.355, k 0.75, real-spec +0.245), but Kevin chose the hold-free signal so that every
    # protocol with one uses it (2026-10-20). Was avg 0.90.
    agent_pb2.PROTOCOL_AUTOBAHN: HeuristicSpec(
        "prepare_to_commit_ms", 1.23, 1, 0.5, (100, 3000), None),
}

# SBFT's slow path takes about 125 + 5h + W (CLAUDE.md hop model); a T under it
# elects on every slow-path commit (cells: 5+50 at T 525 storms with W 300 and
# 500, 5+150 at T 1100 with W 300). T stays above it by this margin once the slow
# path is in use (at least SBFT_SLOW_SHARE of the recent commits).
# Design margin over the hop-model bound: CLAUDE.md puts the floor at about 1.1 x the
# bound; 1.15 adds a little more. Not fitted to a cell.
SBFT_SLOW_PATH_MARGIN = 1.15
# Any slow-path commits in the recent windows: the cells are 0% slow when the wait
# catches or no straggler exists (c0, c2, c5 at W >= 150, three of four c3 W150 draws)
# and slow when it does not (c1, c4 W50: 100%); a storming wait mixes the paths (c4
# W300: 5-20% a window), so 0.05 is a design threshold at the low edge of that mix.
SBFT_SLOW_SHARE = 0.05
# The wait: catch the straggler's prepare only when its lag g is shorter than the
# extra slow-path round R = 2h + 25 (cells: g ~80 at 5+50 caught at W >= 150,
# better than the slow path; g ~300 at 5+50 lost to W <= 50; g ~100 at 5+150
# caught). W = g + SBFT_WAIT_SLACK_MS, else SBFT_WAIT_MIN_MS. g is the largest
# seen n-th prepare gap over the last SBFT_GAP_WINDOWS windows; the gap is
# observable up to the leader's execution, about W + 2h after the quorum, so a
# catchable gap is seen even at the minimum wait.
# The slack: with jitterRatio 0 and W = 50 the seen gap is constant to the millisecond
# in the cells (c3 84, c5 100, c6 300); at other W it varies (c3 79-84, c5 96-100, and
# storming cells 72-520), so 20 ms is a design margin over the non-storming spread.
# The windows: the gap is seen only while a leader that receives the straggler's
# prepare leads (c3 at W 50: seen in 2 of 6 windows, the others unseen), so the
# catch is held across leader changes for 3 windows (one leader re-pin tick is 60 s).
# The low rung of the SBFT ladder (SBFT_SLOW_PATH_VALUES_MS 10/150 since 2026-10-17;
# Kevin, 2026-10-19: minimum 10): with no lag to catch the 4th prepare lands 1-2 ms after
# the 3rd, and giving up at once is about 40 ms faster than W 50 in the slow-path phases
# (runs/2026-10-16-sbft-stability). The gap stays observable at W 10: up to about W + 2h.
SBFT_WAIT_MIN_MS = 10
SBFT_WAIT_SLACK_MS = 20
SBFT_GAP_WINDOWS = 3


def heuristic_spec(protocol: int) -> HeuristicSpec:
    if protocol not in HEURISTIC_SPECS:
        raise ValueError(f"continuous_heuristic has no constants for protocol {protocol}")
    return HEURISTIC_SPECS[protocol]


# The latency floor (Kevin, 2026-10-19, restart D8): a timeout below what a healthy request
# needs elects on every request. Floor = multiplier x the protocol's hold-free signal +
# offset, i.e. the protocol's share c of the time its timer covers, rebuilt by mechanics
# from a signal a leader's proposal hold never enters (a tolerated hold enters every
# end-to-end latency, lifts the floor over the timeout that would depose the leader and
# locks the learner out of it; floorfix/floor_check.py, 2026-10-18). Not fitted. Checked on
# the archived oracle runs (scripts/cloudlab/tmp/cal/restart/d8_floor.py): never above the
# oracle's T in any phase, hold phases included.
# - PBFT, SBFT (c 1, the timer runs from a request's arrival to its execution): the phase sum
#   starts at the pre-prepare's receipt, so it carries 2 of the 3 hops; x 1.5 adds the
#   pre-prepare hop, + 100 the proposal interval the oldest request waits (CLAUDE.md).
# - SmartBFT (c 1): one proposal at a time (view.go inFlightProposal), so a request waits out
#   the decision in flight and then its own, 3 hops each; the field carries 2: 2 x 1.5 = 3.
# - HotStuff (c 1/3, one round of the three a block needs): a round is two hops and the
#   field, the next leader's receipt of a block to its QC, is the second: x 2.
# - Autobahn (c 1/2, one slot of two): the field, Prepare receipt to commit, is the slot.
# - Tendermint (c 1/3): no hold-free field; the p50 never sees the one held height in four.
# Every floor reads the last committing window: the 4-window minimum PBFT and SBFT used
# existed only because p95 carried the hold.
@dataclass(frozen=True)
class FloorSpec:
    signal: str
    multiplier: float
    offset_ms: float = 0.0


FLOOR_SPECS: Dict[int, FloorSpec] = {
    agent_pb2.PROTOCOL_PBFT: FloorSpec("pre_prepare_latency_ms+prepare_latency_ms+commit_latency_ms", 1.5, 100.0),
    agent_pb2.PROTOCOL_SBFT: FloorSpec("pre_prepare_latency_ms+prepare_latency_ms+commit_latency_ms", 1.5, 100.0),
    agent_pb2.PROTOCOL_SMARTBFT: FloorSpec("proposal_to_decision_ms", 3.0),
    agent_pb2.PROTOCOL_TENDERMINT: FloorSpec("p50_consensus_latency_ms", 0.33),
    agent_pb2.PROTOCOL_HOTSTUFF: FloorSpec("proposal_to_qc_ms", 2.0),
    agent_pb2.PROTOCOL_AUTOBAHN: FloorSpec("prepare_to_commit_ms", 1.0),
}


def floor_spec(protocol: int) -> FloorSpec:
    if protocol not in FLOOR_SPECS:
        raise ValueError(f"no latency floor is defined for protocol {protocol}")
    return FLOOR_SPECS[protocol]


def opening_action_ms(protocol: int) -> Tuple[int, ...]:
    """Every model's first action (Kevin, 2026-10-19): the ladder's middle rung, and on SBFT
    the minimum wait."""
    arms_ms = sorted(int(v) for v in get_timeout_values_ms(protocol))
    opening = (arms_ms[len(arms_ms) // 2],)
    if protocol == agent_pb2.PROTOCOL_SBFT:
        opening = opening + (SBFT_WAIT_MIN_MS,)
    return opening


def make_latency_floor(protocol: int, scale: float = 1.0, window: int = 1) -> LatencyFloor:
    """The protocol's floor (FLOOR_SPECS) on the last committing window, ``scale`` x the
    whole floor (1 = as defined; main.py and pretrain.py both build it here). ``window``
    > 1 takes the median over that many committing windows (main.py --latency-floor-window,
    PBFT study E6, 2026-10-24)."""
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"floor scale must be > 0, got {scale}")
    spec = floor_spec(protocol)
    return LatencyFloor(
        multiplier=scale * spec.multiplier,
        window=window,
        quantile=0.5,
        field=spec.signal,
        offset_ms=scale * spec.offset_ms,
    )



class ContinuousHeuristicTimeoutModel(TensorModel):
    """The per-protocol rule of :class:`HeuristicSpec`; on SBFT also the wait W.

    SBFT: W is the straggler catch above; T is the larger of multiplier x the
    p95 estimate and, once the slow path is in use or the rule waits beyond the
    minimum, SBFT_SLOW_PATH_MARGIN x (125 + 5h + W). h is the one-way delay, from
    the minimum p50 of the recent committing windows: (p50 - 100 - caught gap) / 3
    on the fast path, (p50 - 125 - W) / 5 on the slow one.

    ``multiplier``, ``window`` and ``quantile`` override the protocol's constants
    (None keeps them). ``latency_floor`` is the optional constraint every model
    takes (--latency-floor-multiplier): T is never below it.

    A window that committed nothing, or committed without a measurement of the
    signal (restart D15), takes its own length (``empty_window_ms``, the
    feature window) as its latency, as the reward does (Kevin, 2026-10-19): T =
    multiplier x that length, clamped, at once, whatever the window and quantile (a
    4-window minimum would otherwise wait out three more empty windows). The empty
    window is not kept in the window, so the first committing window sets T again.

    Not a bandit: nothing is explored or learned. Until a window commits, T is
    the initial value.
    """

    def __init__(
        self,
        protocol: int,
        action_codec: ContinuousTimeoutActionCodec,
        feature_names: Sequence[str],
        initial_timeout_ms: Tuple[int, ...],
        multiplier: Optional[float] = None,
        window: Optional[int] = None,
        quantile: Optional[float] = None,
        latency_floor: Optional[NormalizedLatencyFloor] = None,
        empty_window_ms: Optional[float] = None,
    ):
        if empty_window_ms is None or not math.isfinite(empty_window_ms) or empty_window_ms <= 0.0:
            raise ValueError(f"empty_window_ms (the feature window) must be > 0, got {empty_window_ms}")
        self._empty_window_ms = float(empty_window_ms)
        spec = heuristic_spec(protocol)
        spec = HeuristicSpec(
            spec.signal,
            spec.multiplier if multiplier is None else float(multiplier),
            spec.window if window is None else int(window),
            spec.quantile if quantile is None else float(quantile),
            spec.bounds_ms,
            spec.storm_view_changes,
        )
        if not math.isfinite(spec.multiplier) or spec.multiplier <= 0.0:
            raise ValueError(f"multiplier must be > 0, got {spec.multiplier}")
        if spec.window <= 0:
            raise ValueError(f"window must be > 0, got {spec.window}")
        if not 0.0 <= spec.quantile <= 1.0:
            raise ValueError(f"quantile must be in [0, 1], got {spec.quantile}")
        self._sbft = protocol == agent_pb2.PROTOCOL_SBFT
        want_dim = 2 if self._sbft else 1
        if action_codec.action_dim != want_dim:
            raise ValueError(
                f"continuous_heuristic needs a {want_dim}-dimensional codec on this "
                f"protocol, got {action_codec.action_dim}"
            )
        names = list(feature_names)
        self._signal_fields = signal_fields(spec.signal)
        required = [*self._signal_fields, "total_consensus_instances"]
        if spec.storm_view_changes is not None:
            required.append("view_change_count")
        if self._sbft:
            required += ["p50_consensus_latency_ms", "fast_path_count", "slow_path_count",
                         "nth_prepare_gap_ms", "nth_prepare_seen_count"]
        missing = [f for f in required if f not in names]
        if missing:
            raise ValueError(f"continuous_heuristic needs report fields {missing}")
        self._idx = {name: i for i, name in enumerate(names)}
        self._obs_dim = len(names)
        self._spec = spec
        self._action_codec = action_codec
        self._latency_floor = latency_floor
        if len(initial_timeout_ms) != want_dim:
            raise ValueError(f"initial_timeout_ms needs {want_dim} values, got {initial_timeout_ms}")
        self._initial = tuple(int(v) for v in initial_timeout_ms)
        self.reset_model()

    def _field(self, row: np.ndarray, name: str) -> float:
        return float(row[self._idx[name]])

    def _clamp_t(self, timeout_ms: float) -> int:
        lower_ms, upper_ms = self._spec.bounds_ms
        return int(min(upper_ms, max(lower_ms, round(timeout_ms))))

    def _estimate(self, values: Deque[float]) -> Optional[float]:
        if not values:
            return None
        return float(np.quantile(np.asarray(values, dtype=float), self._spec.quantile))

    def _sbft_wait_and_bound(
        self, row: np.ndarray, committed: bool
    ) -> Tuple[float, Optional[float], Optional[float]]:
        """(W, slow-path lower bound on T or None, h or None) from this window and
        the recent ones."""
        seen = self._field(row, "nth_prepare_seen_count") > 0.0
        self._gaps.append(self._field(row, "nth_prepare_gap_ms") if seen else None)
        if committed:
            fast = self._field(row, "fast_path_count")
            slow = self._field(row, "slow_path_count")
            if fast + slow > 0.0:
                self._slow_shares.append(slow / (fast + slow))
        p50_min = min(self._p50s) if self._p50s else None
        if p50_min is None:
            return float(self._held[1]), None, None
        slow_in_use = bool(self._slow_shares) and max(self._slow_shares) >= SBFT_SLOW_SHARE
        # The wait this window ran under is the one held since the last decision: the
        # features carry no applied timeout (models.py, restart D9a).
        wait_now = float(self._held[1])
        gaps = [g for g in self._gaps if g is not None]
        gap = max(gaps) if gaps else None
        # Design choice: the latest window's majority path picks the hop formula.
        if slow_in_use and self._slow_shares[-1] >= 0.5:
            h = (p50_min - 125.0 - wait_now) / 5.0
        else:
            # A fast-path commit waited for the straggler's prepare when it caught it.
            h = (p50_min - 100.0 - min(gap or 0.0, wait_now)) / 3.0
        h = max(1.0, h)
        extra_round_ms = 2.0 * h + 25.0
        if gap is not None and gap + SBFT_WAIT_SLACK_MS < extra_round_ms:
            wait_ms = gap + SBFT_WAIT_SLACK_MS
        else:
            wait_ms = float(SBFT_WAIT_MIN_MS)
        # A wait longer than the minimum can miss and take the slow path too, so T
        # covers the slow path whenever the rule waits, not only once it is in use.
        waiting = wait_ms > SBFT_WAIT_MIN_MS
        bound = SBFT_SLOW_PATH_MARGIN * (125.0 + 5.0 * h + wait_ms) if (slow_in_use or waiting) else None
        return wait_ms, bound, h

    def _step(self, row: np.ndarray) -> Tuple[int, ...]:
        spec = self._spec
        committed = self._field(row, "total_consensus_instances") > 0.0
        self._seen_commit = self._seen_commit or committed
        signal = sum(self._field(row, name) for name in self._signal_fields)
        p50 = self._field(row, "p50_consensus_latency_ms")
        # main.py fills a window that committed without a latency sample with the window's
        # own length in every consensus latency, so p50 at that length marks it as empty too
        # (D15): it takes the empty branch and is not kept in the window.
        measured = committed and signal > 0.0 and p50 < self._empty_window_ms
        if measured:
            self._signals.append(signal)
            if self._sbft and p50 > 0.0:
                self._p50s.append(p50)
        estimate = self._estimate(self._signals)
        held_t = self._held[0]
        storm = (
            spec.storm_view_changes is not None
            and self._field(row, "view_change_count") >= spec.storm_view_changes
        )
        wait_ms, bound, h = (
            self._sbft_wait_and_bound(row, committed) if self._sbft else (None, None, None)
        )
        if self._seen_commit and not measured:
            # Nothing committed, or no measurement of the signal in a window that committed:
            # the window's own length as its latency (class docstring).
            timeout_ms = self._clamp_t(spec.multiplier * self._empty_window_ms)
            wait_ms = self._held[1] if self._sbft else None
        elif storm:
            self._storm_ms = max(self._storm_ms or 0, held_t)
            self._storm_estimate = estimate
            timeout_ms = self._clamp_t(2 * held_t)
            if self._sbft:
                wait_ms = SBFT_WAIT_MIN_MS
        elif committed and estimate is not None:
            # Design choice: a fall of a third in the estimate counts as a faster
            # link (the spec's link steps change the latency by 2x or more).
            if (self._storm_ms is not None and self._storm_estimate is not None
                    and estimate < self._storm_estimate * 2.0 / 3.0):
                self._storm_ms = self._storm_estimate = None
            target = spec.multiplier * estimate
            if bound is not None:
                target = max(target, bound)
            if self._storm_ms is not None:
                target = max(target, 2 * self._storm_ms)
            timeout_ms = self._clamp_t(target)
        else:
            timeout_ms = held_t
            wait_ms = self._held[1] if self._sbft else None
        if self._latency_floor is not None:
            self._latency_floor.observe_features(row)
            floor_ms = float(self._latency_floor.floor_ms)
            if floor_ms > timeout_ms:
                timeout_ms = self._clamp_t(math.ceil(floor_ms))
        if self._sbft:
            lower_w, upper_w = self._action_codec.bounds_ms[1]
            return (timeout_ms, int(min(upper_w, max(lower_w, round(wait_ms)))))
        return (timeout_ms,)

    def predict_action(self, features: np.ndarray) -> np.ndarray:
        features_np = np.asarray(features, dtype=float)
        single = features_np.ndim == 1
        if single:
            features_np = features_np.reshape(1, -1)
        if features_np.shape[1] != self._obs_dim:
            raise ValueError(f"Expected {self._obs_dim} features, got {features_np.shape[1]}")
        actions = []
        for row in features_np:
            self._held = self._step(row)
            actions.append([self._action_codec.normalize_ms(v, d) for d, v in enumerate(self._held)])
        result = np.asarray(actions, dtype=float)
        return result[0] if single else result

    def update_model(
        self,
        features: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | float,
        backward: bool = True,
    ) -> float:
        """No-op: the rule has no parameters to move. The reward still reaches
        the agent's CSV, so a run is comparable with a learned one."""
        return 0.0

    def reset_model(self) -> None:
        # Before the first commit an empty window is the start-up, not a stall.
        self._seen_commit = False
        self._signals: Deque[float] = deque(maxlen=self._spec.window)
        self._p50s: Deque[float] = deque(maxlen=self._spec.window)
        self._gaps: Deque[Optional[float]] = deque(maxlen=SBFT_GAP_WINDOWS)
        self._slow_shares: Deque[float] = deque(maxlen=SBFT_GAP_WINDOWS)
        self._storm_ms: Optional[int] = None
        self._storm_estimate: Optional[float] = None
        if self._latency_floor is not None:
            self._latency_floor.reset()
        self._held: Tuple[int, ...] = self._initial


# --- Simulator-pretrained Gaussian-process bandit -----------------------------

_PRIOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_models")

# Action range of --model-type=continuous_gp on Tendermint, and where its simulator writes the prior.
CONTINUOUS_TENDERMINT_BOUNDS_MS: Tuple[int, int] = (100, 3000)
DEFAULT_TENDERMINT_PRIOR_PATH = os.path.join(_PRIOR_DIR, "continuous_tendermint_prior.npz")


@dataclass(frozen=True)
class ContextSpec:
    """What --model-type=continuous_gp and discrete_gp read and play on one protocol.

    The context vector is log1p of ``log_fields`` followed by ``raw_fields``
    verbatim, in that order; ``length_scales`` has one GP length scale per
    context dimension (inf drops the field from the kernel; the prior still
    reads it). ``bounds_ms`` is the codec's action range (SBFT's election
    timeout), ``wait_bounds_ms`` SBFT's fast-path wait range (None on the
    one-dimensional protocols), ``prior_path`` where the simulator writes the
    GP prior by default (input_dim = context_dim + action_dim, trained on the
    agent's reward design; the agent never loads it implicitly, only a path
    given with --prior-path) and ``simulator`` the script that trains it.
    """

    log_fields: Tuple[str, ...]
    raw_fields: Tuple[str, ...]
    length_scales: Tuple[float, ...]
    bounds_ms: Tuple[int, int]
    prior_path: str
    simulator: str
    wait_bounds_ms: Optional[Tuple[int, int]] = None

    def __post_init__(self) -> None:
        if len(self.length_scales) != len(self.log_fields) + len(self.raw_fields):
            raise ValueError(
                f"{len(self.length_scales)} length scales for "
                f"{len(self.log_fields) + len(self.raw_fields)} context fields"
            )

    @property
    def context_dim(self) -> int:
        return len(self.length_scales)


# Log-scaled report fields the model reads, in TENDERMINT_REPORT_FEATURE_FIELD_NAMES
# terms, plus the violation rate.
_TENDERMINT_CONTEXT_LOG_FIELDS: Tuple[str, ...] = (
    "avg_consensus_latency_ms",
    "p50_consensus_latency_ms",
    "p95_consensus_latency_ms",
    "propose_latency_ms",
    "prevote_latency_ms",
    "precommit_latency_ms",
    "throughput_tps",
    "avg_batch_size",
    "p50_inter_commit_gap_ms",
)
# GP length scales per context dimension: 0.3 in log units is a factor of 1.35.
# The GP tells contexts apart only by p50 latency, throughput and violation
# rate. The other fields swing from one feature window to the next within a
# phase (p3 prevote latency 77-143 ms in 8 s windows; the feature window is 7 s
# since 2026-10-09), which made every window a new context and
# kept the GP from pooling a phase's evidence. inf drops a field from the
# kernel; the prior still reads every field.
_TENDERMINT_CONTEXT_LENGTH_SCALES: Tuple[float, ...] = (
    math.inf,  # avg_consensus_latency_ms
    0.3,  # p50_consensus_latency_ms
    math.inf, math.inf, math.inf, math.inf,  # p95, propose, prevote, precommit
    0.2,  # throughput_tps
    math.inf, math.inf,  # avg_batch_size, p50_inter_commit_gap_ms
    0.15,  # timeout_violation_rate
)

# PBFT and SmartBFT share the 15 base report fields (PBFT_REPORT_FEATURE_FIELD_NAMES).
# view_change_count stands in for Tendermint's violation rate. timeout_ms, the
# applied election timeout, is left out on purpose: the model's own last pick
# in the context drove a two-cycle (H15); since restart D9a no model's input
# holds it.
_PBFT_CONTEXT_LOG_FIELDS: Tuple[str, ...] = (
    "avg_consensus_latency_ms",
    "p50_consensus_latency_ms",
    "p95_consensus_latency_ms",
    "throughput_tps",
    "avg_batch_size",
    "p50_inter_commit_gap_ms",
    "view_change_count",
)
# Same reading as the Tendermint scales; 0.7 on the log1p count is a factor
# of 2 in view changes (quiet windows report 0; measured per 10 s window, a
# PBFT storm 3-5 and a SmartBFT storm about 18; since restart D14 the count covers
# the trailing 20 s episode on PBFT, SBFT and SmartBFT, about 2x that; on log1p a
# factor of 2 is 0.7 whatever the window).
_PBFT_CONTEXT_LENGTH_SCALES: Tuple[float, ...] = (
    math.inf,  # avg_consensus_latency_ms
    0.3,  # p50_consensus_latency_ms
    math.inf,  # p95_consensus_latency_ms
    0.2,  # throughput_tps
    math.inf, math.inf,  # avg_batch_size, p50_inter_commit_gap_ms
    0.7,  # view_change_count
)
# PBFT only: the phase latencies beside the consensus latency, read by the
# prior and not by the kernel (as SBFT's). A proposal hold precedes the
# pre-prepare, so under a hold the consensus latency rises and the phases do
# not, while a slow link raises both: the one observable that separates the
# two (Kevin, 2026-10-14, runs/2026-10-14-five-machine-models; the prior on the
# base fields read a 200 ms hold at base as a +50 link and a 550 ms hold as a
# +150 link, and played 800-2000 ms in every hold phase).
_PBFT_PHASE_CONTEXT_LOG_FIELDS: Tuple[str, ...] = _PBFT_CONTEXT_LOG_FIELDS + (
    "pre_prepare_latency_ms",
    "prepare_latency_ms",
    "commit_latency_ms",
)
_PBFT_PHASE_CONTEXT_LENGTH_SCALES: Tuple[float, ...] = _PBFT_CONTEXT_LENGTH_SCALES + (
    math.inf, math.inf, math.inf,  # pre_prepare, prepare, commit latency: prior only
)
# SmartBFT: the same separation from its one phase field, proposal arrival ->
# delivery (PBFT's three phases together; SMARTBFT_REPORT_FEATURE_FIELD_NAMES).
_SMARTBFT_CONTEXT_LOG_FIELDS: Tuple[str, ...] = _PBFT_CONTEXT_LOG_FIELDS + (
    "proposal_to_decision_ms",
)
_SMARTBFT_CONTEXT_LENGTH_SCALES: Tuple[float, ...] = _PBFT_CONTEXT_LENGTH_SCALES + (
    math.inf,  # proposal_to_decision_ms: prior only
)
# HotStuff and Autobahn: the same separation from their hold-free fields (2026-10-19;
# the base fields alone read a slow leader as a slow link).
_HOTSTUFF_CONTEXT_LOG_FIELDS: Tuple[str, ...] = _PBFT_CONTEXT_LOG_FIELDS + ("proposal_to_qc_ms",)
_AUTOBAHN_CONTEXT_LOG_FIELDS: Tuple[str, ...] = _PBFT_CONTEXT_LOG_FIELDS + ("prepare_to_commit_ms",)
_HOLD_FREE_CONTEXT_LENGTH_SCALES: Tuple[float, ...] = _PBFT_CONTEXT_LENGTH_SCALES + (
    math.inf,  # the hold-free field: prior only
)
# HotStuff and Autobahn count round timeouts as view changes: 43 a window at T about 100
# and 0 at 200 or more in healthy HotStuff phases, 19-69 on Autobahn even at its best T, so
# the count follows the model's own T, the self-feedback H15 removed by dropping the
# applied timeout. Prior only there (Kevin, 2026-10-19, restart D9b).
_ROUND_TIMEOUT_CONTEXT_LENGTH_SCALES: Tuple[float, ...] = (
    _HOLD_FREE_CONTEXT_LENGTH_SCALES[:6] + (math.inf,) + _HOLD_FREE_CONTEXT_LENGTH_SCALES[7:]
)

# SBFT's report is PBFT's plus the phase latencies and path counts
# (SBFT_REPORT_FEATURE_FIELD_NAMES); the fast and slow path counts join
# view_change_count in the kernel. pre_prepare_latency_ms (pre-prepare to
# certificate) carries the model's own W on a miss, so the prior reads it but
# the kernel does not. The (T, W) applied while the context was measured is not
# part of it: the prior's 3-episode return already handles a T that escapes a
# delayed leader, and with the applied pair as an input the policy cycled
# through its own last pick (runs/2026-10-01-sbft-continuous, 22:55: three of
# five fits good without it, one or two with it). The leader's n-th prepare
# gap and its seen/unseen counts tell a miss the wait could catch (a gap under
# the slow path) from one it cannot (unseen by execution, as with a crashed
# replica); the prior reads them, the kernel does not.
_SBFT_CONTEXT_LOG_FIELDS: Tuple[str, ...] = (
    "avg_consensus_latency_ms",
    "p50_consensus_latency_ms",
    "p95_consensus_latency_ms",
    "throughput_tps",
    "avg_batch_size",
    "p50_inter_commit_gap_ms",
    "view_change_count",
    "pre_prepare_latency_ms",
    "prepare_latency_ms",
    "commit_latency_ms",  # completes the phase sum (2026-10-14; see _PBFT_PHASE_CONTEXT_LOG_FIELDS)
    "fast_path_count",
    "slow_path_count",
    "nth_prepare_gap_ms",
    "nth_prepare_seen_count",
    "nth_prepare_unseen_count",
)
_SBFT_CONTEXT_LENGTH_SCALES: Tuple[float, ...] = (
    math.inf,  # avg_consensus_latency_ms
    0.3,  # p50_consensus_latency_ms
    math.inf,  # p95_consensus_latency_ms
    0.2,  # throughput_tps
    math.inf, math.inf,  # avg_batch_size, p50_inter_commit_gap_ms
    0.7,  # view_change_count
    math.inf, math.inf, math.inf,  # pre_prepare_latency_ms, prepare_latency_ms, commit_latency_ms
    # The path counts follow the model's own W (and flip deep/shallow at 5+150):
    # prior only (Kevin, 2026-10-19, restart D9c; 0.7 until then).
    math.inf,  # fast_path_count
    math.inf,  # slow_path_count
    math.inf, math.inf, math.inf,  # nth_prepare_gap_ms, _seen_count, _unseen_count
)

CONTEXT_SPECS: Dict[int, ContextSpec] = {
    agent_pb2.PROTOCOL_TENDERMINT: ContextSpec(
        log_fields=_TENDERMINT_CONTEXT_LOG_FIELDS,
        raw_fields=("timeout_violation_rate",),
        length_scales=_TENDERMINT_CONTEXT_LENGTH_SCALES,
        bounds_ms=CONTINUOUS_TENDERMINT_BOUNDS_MS,
        prior_path=DEFAULT_TENDERMINT_PRIOR_PATH,
        simulator="tendermint_sim.py",
    ),
    # Top rung 1200; base p50 94-168 across hosts, so 100 sits under the floor.
    agent_pb2.PROTOCOL_PBFT: ContextSpec(
        log_fields=_PBFT_PHASE_CONTEXT_LOG_FIELDS,
        raw_fields=(),
        length_scales=_PBFT_PHASE_CONTEXT_LENGTH_SCALES,
        bounds_ms=(100, 3000),
        prior_path=os.path.join(_PRIOR_DIR, "continuous_pbft_prior.npz"),
        simulator="pbft_sim.py",
    ),
    # 1800 is the only rung that survives 5+150 and 800 still storms there.
    # One range for every protocol and model, 100-3000 (Kevin, 2026-10-19; was 100-4000
    # here, 100-2000 on HotStuff).
    agent_pb2.PROTOCOL_SMARTBFT: ContextSpec(
        log_fields=_SMARTBFT_CONTEXT_LOG_FIELDS,
        raw_fields=(),
        length_scales=_SMARTBFT_CONTEXT_LENGTH_SCALES,
        bounds_ms=(100, 3000),
        prior_path=os.path.join(_PRIOR_DIR, "continuous_smartbft_prior.npz"),
        simulator="smartbft_sim.py",
    ),
    # Two-dimensional action: election timeout T in bounds_ms, fast-path wait
    # W in wait_bounds_ms.
    agent_pb2.PROTOCOL_SBFT: ContextSpec(
        log_fields=_SBFT_CONTEXT_LOG_FIELDS,
        raw_fields=(),
        length_scales=_SBFT_CONTEXT_LENGTH_SCALES,
        bounds_ms=(100, 3000),
        prior_path=os.path.join(_PRIOR_DIR, "continuous_sbft_prior.npz"),
        simulator="sbft_sim.py",
        wait_bounds_ms=(10, 500),
    ),
    # HotStuff's report is PBFT's field for field (models.py). A healthy round
    # is 2h, so the 100 ms lower bound sits at the block-delay floor of a 5 ms
    # link; the shared 100-3000 range (2026-10-19; 2000 until then).
    agent_pb2.PROTOCOL_HOTSTUFF: ContextSpec(
        log_fields=_HOTSTUFF_CONTEXT_LOG_FIELDS,
        raw_fields=(),
        length_scales=_ROUND_TIMEOUT_CONTEXT_LENGTH_SCALES,
        bounds_ms=(100, 3000),
        prior_path=os.path.join(_PRIOR_DIR, "continuous_hotstuff_prior.npz"),
        simulator="hotstuff_sim.py",
    ),
    # Autobahn's report is HotStuff's without regency_change_count, so the
    # PBFT context fields all exist. Healthy Autobahn has no floor; under vote
    # withholding 5+150 needs about 1000 ms, and 3000 is three times that.
    agent_pb2.PROTOCOL_AUTOBAHN: ContextSpec(
        log_fields=_AUTOBAHN_CONTEXT_LOG_FIELDS,
        raw_fields=(),
        length_scales=_ROUND_TIMEOUT_CONTEXT_LENGTH_SCALES,
        bounds_ms=(100, 3000),
        prior_path=os.path.join(_PRIOR_DIR, "continuous_autobahn_prior.npz"),
        simulator="autobahn_sim.py",
    ),
}


def context_spec(protocol: int) -> ContextSpec:
    spec = CONTEXT_SPECS.get(int(protocol))
    if spec is None:
        raise ValueError(
            "--model-type=continuous_gp/discrete_gp has no context spec for "
            f"{agent_pb2.Protocol.Name(int(protocol))}; it runs on PBFT, SmartBFT, "
            "SBFT, Tendermint, HotStuff and Autobahn"
        )
    return spec


def report_context(
    spec: ContextSpec, field_names: Tuple[str, ...], features: np.ndarray
) -> np.ndarray:
    """Context vector for one report: log1p of spec.log_fields, then spec.raw_fields.

    The model's own previous timeout is deliberately not part of it: with it,
    each replica's last pick fed back into its next one and drove a two-cycle
    between low and high timeouts in p3.
    """
    row = np.asarray(features, dtype=float).reshape(-1)
    values = [
        math.log1p(max(0.0, float(row[field_names.index(name)])))
        for name in spec.log_fields
    ]
    values.extend(float(row[field_names.index(name)]) for name in spec.raw_fields)
    return np.asarray(values, dtype=float)


def tendermint_context(field_names: Tuple[str, ...], features: np.ndarray) -> np.ndarray:
    return report_context(CONTEXT_SPECS[agent_pb2.PROTOCOL_TENDERMINT], field_names, features)


class NumpyMLP:
    """ReLU MLP regressor evaluated in numpy, loaded from an .npz of weights."""

    def __init__(self, path: str):
        data = np.load(path)
        count = int(data["n_layers"])
        self._weights = [data[f"W{i}"] for i in range(count)]
        self._biases = [data[f"b{i}"] for i in range(count)]
        self._x_mean = data["x_mean"]
        self._x_std = data["x_std"]
        self._y_mean = float(data["y_mean"])
        self._y_std = float(data["y_std"])
        # Every prior is written by train_gp_prior, which records all of these; a file
        # without them predates the 2026-10-19 restart and is refused, not read with
        # defaults.
        missing = sorted(set(GP_PRIOR_KEYS) - set(data.files))
        if missing:
            raise ValueError(f"Prior {path} lacks {missing}; retrain it (train_gp_prior)")
        self.protocol = str(data["protocol"])
        self.input_names = tuple(str(v) for v in data["input_names"])
        self.action_bounds_ms = tuple(tuple(int(v) for v in row) for row in data["action_bounds_ms"])
        self.code_hash = str(data["code_hash"])
        self.reward_design = str(data["reward_design"])
        # N of reward_tps_latency the prior was fit under (reward_utils.REWARD_NORMALIZER_TPS).
        self.reward_normalizer_tps = float(data["reward_normalizer_tps"])
        # Episodes the action was held for in training (<sim> train --hold-episodes);
        # the target was the mean reward over the hold.
        self.hold_episodes = int(data["hold_episodes"])
        # Rewards below this were fit as this value (<sim> train --target-floor); -inf
        # when the prior was fit to the raw reward.
        self.target_floor = float(data["target_floor"])

    @property
    def input_dim(self) -> int:
        return int(self._x_mean.size)

    @property
    def y_std(self) -> float:
        """Standard deviation of the training target, the reward's natural scale."""
        return self._y_std

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        h = (np.asarray(inputs, dtype=float) - self._x_mean) / self._x_std
        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            h = h @ w + b
            if i < len(self._weights) - 1:
                h = np.maximum(h, 0.0)
        out = h.reshape(-1) * self._y_std + self._y_mean
        return out


# What every GP prior records besides its weights (NumpyMLP refuses a file without them).
GP_PRIOR_KEYS: Tuple[str, ...] = (
    "protocol", "input_names", "action_bounds_ms", "code_hash", "sim_constants",
    "reward_design", "reward_normalizer_tps", "hold_episodes", "target_floor",
)


def gp_prior_input_names(protocol: int) -> Tuple[str, ...]:
    """The prior's input columns: the protocol's context, then its normalised action(s)."""
    spec = CONTEXT_SPECS[int(protocol)]
    actions = ("timeout_position", "wait_position") if spec.wait_bounds_ms else ("timeout_position",)
    return tuple(spec.log_fields) + tuple(spec.raw_fields) + actions


def train_gp_prior(
    protocol: int,
    x: np.ndarray,
    y: np.ndarray,
    out_path: str,
    *,
    seed: int,
    design: str,
    hold: int,
    target_floor: float,
    windows: Any,
    sim_constants: Dict[str, Any],
    code_hash: str,
    hidden: Tuple[int, ...] = (128, 128, 64),
    extra: Optional[Dict[str, Any]] = None,
) -> float:
    """Fit the MLP prior of the GP models on simulator samples and write it; returns
    the validation RMSE. The one writer for all six simulators (Kevin, 2026-10-19,
    restart): the file records what NumpyMLP and the GP check on load (protocol,
    input columns, the action range the positions are normalised over, reward,
    hold and floor) and, for the record, the simulator constants the runs were
    drawn around and the code_hash of the tree (scripts/cloudlab/code_state.py)
    that trained it, which the harness compares with FREEZE. ``windows`` is the
    simulator's PriorWindows; ``extra`` adds simulator-specific scalars.
    """
    from sklearn.neural_network import MLPRegressor
    import json

    if not str(code_hash).strip():
        raise ValueError("code_hash is required (scripts/cloudlab/code_state.py)")
    if int(hold) < 1:
        raise ValueError(f"hold must be >= 1, got {hold}")
    spec = CONTEXT_SPECS[int(protocol)]
    names = gp_prior_input_names(protocol)
    if x.ndim != 2 or x.shape[1] != len(names):
        raise ValueError(f"x has shape {x.shape}; the prior takes {len(names)} inputs {names}")
    # Rewards below the floor are fit as the floor. SimReturnGPModel applies the
    # same floor to the online reward, so the residual compares like with like, and
    # the fit stops spending its error on collapses instead of on the structure that
    # ranks the usable timeouts.
    y = np.maximum(y, target_floor)
    x_mean, x_std = x.mean(axis=0), x.std(axis=0) + 1e-9
    y_mean, y_std = float(y.mean()), float(y.std() + 1e-9)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    cut = int(0.9 * len(x))
    tr, va = order[:cut], order[cut:]
    mlp = MLPRegressor(
        hidden_layer_sizes=tuple(hidden),
        activation="relu",
        alpha=1e-5,
        batch_size=512,
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=seed,
    )
    mlp.fit((x[tr] - x_mean) / x_std, (y[tr] - y_mean) / y_std)
    pred = mlp.predict((x[va] - x_mean) / x_std) * y_std + y_mean
    rmse = float(np.sqrt(np.mean((pred - y[va]) ** 2)))
    protocol_name = agent_pb2.Protocol.Name(int(protocol))[len("PROTOCOL_"):].lower()
    bounds = [spec.bounds_ms] + ([spec.wait_bounds_ms] if spec.wait_bounds_ms else [])
    arrays: Dict[str, Any] = {"n_layers": np.asarray(len(mlp.coefs_))}
    for i, (w, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        arrays[f"W{i}"] = w
        arrays[f"b{i}"] = b
    arrays.update(
        x_mean=x_mean,
        x_std=x_std,
        y_mean=np.asarray(y_mean),
        y_std=np.asarray(y_std),
        protocol=np.asarray(protocol_name),
        input_names=np.asarray(names),
        action_bounds_ms=np.asarray(bounds, dtype=np.int64),
        code_hash=np.asarray(str(code_hash)),
        sim_constants=np.asarray(json.dumps(sim_constants, sort_keys=True, default=float)),
        reward_design=np.asarray(design),
        reward_normalizer_tps=np.asarray(reward_normalizer_tps(protocol_name)),
        hold_episodes=np.asarray(int(hold)),
        target_floor=np.asarray(float(target_floor)),
    )
    for key, value in (extra or {}).items():
        if key in arrays:
            raise ValueError(f"extra key {key} would overwrite a prior field")
        arrays[key] = np.asarray(value)
    arrays.update(windows.arrays())
    np.savez(out_path, **arrays)
    return rmse


# Default floor for an untrained prior. SimReturnGPModel needs a finite one -
# it applies the same floor to the online reward so the residual compares like
# with like - and -2 is the value the simulators' --target-floor help quotes.
DEFAULT_UNTRAINED_TARGET_FLOOR = -2.0


class ZeroPrior:
    """NumpyMLP's surface with no simulator knowledge: it predicts 0 everywhere.

    The "untrained" arm of the pretraining ablation. The GP machinery is
    unchanged - same grid, same kernel, same confidence bound - but the
    residual it fits is the reward itself rather than the reward's distance
    from a simulator's guess, so everything the model knows it learned online.
    """

    def __init__(
        self,
        input_dim: int,
        reward_design: str,
        target_floor: float = DEFAULT_UNTRAINED_TARGET_FLOOR,
    ):
        self._input_dim = int(input_dim)
        self.reward_design = str(reward_design)
        self.hold_episodes = 1
        self.target_floor = float(target_floor)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def y_std(self) -> float:
        """1.0: with no fitted target there is no scale to carry, so the GP's
        constants act on the reward in its own units."""
        return 1.0

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return np.zeros(len(np.atleast_2d(np.asarray(inputs, dtype=float))))


class SimPriorGPModel(TensorModel):
    """Continuous contextual bandit: simulator prior plus online GP residual.

    The reward of timeout ``a`` in context ``z`` is modelled as
    ``prior(z, a) + f(z, a)``. ``prior`` is an MLP trained offline on the
    protocol's simulator (``ContextSpec.simulator``; randomized environments,
    never the failure spec);
    ``f`` is a Gaussian process fitted online to what the prior got wrong on
    this deployment. ``f`` has a level shared by every timeout in a context
    (``level_std``) plus a part local in the timeout, so a context-wide miss
    of the prior (as in p3, where it is optimistic at every timeout) lowers
    untried timeouts too instead of only the one played. Each decision plays,
    over a log-spaced grid of timeouts,
    the argmax of the lower confidence bound ``prior + mean - confidence * sd``
    of the posterior: a timeout nobody has tried in this context is discounted
    by its uncertainty, so after a failure the model moves toward timeouts it
    has evidence for rather than toward an untried one the prior is optimistic
    about, and a context it has not seen at all falls back on the prior.

    The action is the normalized log timeout of ``action_codec``; on SBFT it
    is the pair (election timeout, fast-path wait), searched over the product
    grid ``grid_size`` x ``wait_grid_size`` with a product RBF kernel over the
    two. The context, scales, and prior come from
    ``CONTEXT_SPECS[action_codec.protocol]``.
    There is no floor, clamp, or rule other than the codec's range.

    ``axes_ms`` (discrete_gp) replaces the log-spaced grid with given timeouts
    in ms, one tuple per action dimension, normalized by the codec; on SBFT
    the grid is their T-major product, so on the ladder rungs grid row i is
    discrete arm i. ``grid_size`` and ``wait_grid_size`` are then unused.
    """

    def __init__(
        self,
        action_codec: ContinuousTimeoutActionCodec,
        field_names: Tuple[str, ...],
        prior_path: Optional[str],
        reward_design: str,
        grid_size: int = 64,
        signal_std: float = 0.08,
        noise_std: float = 0.04,
        action_length_scale: float = 0.1,
        memory_size: int = 256,
        confidence: float = 1.0,
        level_std: float = 0.2,
        wait_grid_size: int = 16,
        latency_floor: Optional[NormalizedLatencyFloor] = None,
        axes_ms: Optional[Tuple[Sequence[int], ...]] = None,
    ):
        action_dim = int(action_codec.action_dim)
        if action_dim not in (1, 2):
            raise ValueError(
                f"SimPriorGPModel supports one or two timeout dimensions, got {action_dim}"
            )
        sizes = (("memory_size", memory_size),)
        if axes_ms is None:
            sizes = (("grid_size", grid_size), ("wait_grid_size", wait_grid_size)) + sizes
        elif len(axes_ms) != action_dim:
            raise ValueError(
                f"axes_ms has {len(axes_ms)} axes, the codec {action_dim} dimensions"
            )
        elif any(len(axis) == 0 for axis in axes_ms):
            raise ValueError(f"axes_ms has an empty axis: {axes_ms}")
        for name, value in sizes:
            if int(value) <= 1:
                raise ValueError(f"{name} must be > 1, got {value}")
        for name, value in (
            ("signal_std", signal_std),
            ("noise_std", noise_std),
            ("action_length_scale", action_length_scale),
            ("confidence", confidence),
            ("level_std", level_std),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be > 0, got {value}")

        self._action_dim = action_dim
        # The optional constraint every model takes: grid points whose
        # election timeout is under it are not scored (see _feasible_mask).
        self._latency_floor = latency_floor
        self._spec = context_spec(action_codec.protocol)
        self._field_names = tuple(field_names)
        self._obs_dim = len(self._field_names)
        for name in self._spec.log_fields + self._spec.raw_fields:
            if name not in self._field_names:
                raise ValueError(
                    f"Context field {name} is not in the report fields given"
                )
        context_dim_for_prior = self._spec.context_dim
        if prior_path is None:
            # Untrained: no simulator pretraining, the GP fits the reward itself.
            self._prior = ZeroPrior(
                input_dim=context_dim_for_prior + action_dim,
                reward_design=reward_design,
            )
        elif not os.path.isfile(prior_path):
            raise FileNotFoundError(
                f"Prior {prior_path} does not exist; train it with "
                f"{self._spec.simulator} train --reward-design {reward_design}"
            )
        else:
            self._prior = NumpyMLP(prior_path)
        if prior_path is not None:
            protocol_name = agent_pb2.Protocol.Name(int(action_codec.protocol))[len("PROTOCOL_"):].lower()
            expected_bounds = tuple(tuple(int(v) for v in b) for b in action_codec.bounds_ms)
            for what, found, wanted in (
                ("protocol", self._prior.protocol, protocol_name),
                ("input_names", self._prior.input_names, gp_prior_input_names(action_codec.protocol)),
                ("action_bounds_ms", self._prior.action_bounds_ms, expected_bounds),
            ):
                if found != wanted:
                    raise ValueError(
                        f"Prior {prior_path} has {what} {found}, the agent has {wanted}; retrain it "
                        f"with {self._spec.simulator} train"
                    )
        if self._prior.reward_design != reward_design:
            raise ValueError(
                f"Prior {prior_path} was trained on {self._prior.reward_design}, "
                f"but the agent runs {reward_design}; retrain it with "
                f"{self._spec.simulator} train --reward-design {reward_design}"
            )
        if reward_design == "reward_tps_latency" and prior_path is not None:
            expected = reward_normalizer_tps(
                agent_pb2.Protocol.Name(int(action_codec.protocol))[len("PROTOCOL_"):].lower()
            )
            if self._prior.reward_normalizer_tps != expected:
                raise ValueError(
                    f"Prior {prior_path} was fit under reward_tps_latency with tps / "
                    f"{self._prior.reward_normalizer_tps:g}, but the agent "
                    f"divides by {expected:g}; retrain it with {self._spec.simulator} train "
                    f"--reward-design {reward_design}"
                )
        context_dim = self._spec.context_dim
        if self._prior.input_dim != context_dim + action_dim:
            raise ValueError(
                f"Prior {prior_path} takes {self._prior.input_dim} inputs, "
                f"expected {context_dim + action_dim} (context + action)"
            )
        if axes_ms is None:
            axes = tuple(
                np.linspace(0.0, 1.0, int(size))
                for size in (grid_size, wait_grid_size)[:action_dim]
            )
        else:
            axes = tuple(
                np.array([action_codec.normalize_ms(ms, dim) for ms in axis], dtype=float)
                for dim, axis in enumerate(axes_ms)
            )
        # Where a flat prior opens: the grid point nearest the ladder's middle rung (and on
        # SBFT the minimum wait), as the heuristic and discrete models open (Kevin,
        # 2026-10-19; the middle of the log grid until then, above Tendermint's top rung).
        opening = opening_action_ms(action_codec.protocol)
        if action_dim == 1:
            self._grid = axes[0]
            self._mid_index = int(np.argmin(np.abs(axes[0] - action_codec.normalize_ms(opening[0], 0))))
        else:
            # One (T, W) row per pair, T major.
            timeouts, waits = axes
            self._grid = np.column_stack(
                (np.repeat(timeouts, waits.size), np.tile(waits, timeouts.size))
            )
            t_index = int(np.argmin(np.abs(timeouts - action_codec.normalize_ms(opening[0], 0))))
            w_index = int(np.argmin(np.abs(waits - action_codec.normalize_ms(opening[1], 1))))
            self._mid_index = t_index * waits.size + w_index
            self._grid_axes = (timeouts, waits)
        self._signal_var = float(signal_std) ** 2
        self._noise_var = float(noise_std) ** 2
        self._scales = np.asarray(
            self._spec.length_scales + (float(action_length_scale),) * action_dim,
            dtype=float,
        )
        self._memory_size = int(memory_size)
        self._confidence = float(confidence)
        # Shared by every timeout in a context: how far the prior's level can
        # be off there as a whole (the Tendermint prior's simulator holdout
        # RMSE, 0.204; another protocol's prior passes its own).
        self._level_var = float(level_std) ** 2
        self.reset_model()

    def _kernel(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = self._action_dim
        diff = (a[:, None, :] - b[None, :, :]) / self._scales
        sq = diff * diff
        context = np.exp(-0.5 * np.sum(sq[:, :, :-d], axis=2))
        # Product RBF over the action dimensions; at d = 1 the sum is the one term.
        action = np.exp(-0.5 * np.sum(sq[:, :, -d:], axis=2))
        return context * (self._level_var + self._signal_var * action)

    def _kernel_grid(self, a: np.ndarray, context: np.ndarray) -> np.ndarray:
        """``_kernel(a, _inputs(context, _grid))`` for d = 2, with the context factor
        computed once per row of ``a`` instead of once per (row, grid point), since
        every grid row carries the same context. Equal to ``_kernel`` up to rounding."""
        d = self._action_dim
        dc = (a[:, :-d] - context.reshape(1, -1)) / self._scales[:-d]
        context_factor = np.exp(-0.5 * np.sum(dc * dc, axis=1))
        # The grid is the product of its T and W axes (T major) and the action kernel a
        # product of per-axis RBFs, so it factors into a (rows, T) and a (rows, W) table.
        timeouts, waits = self._grid_axes
        dt = (a[:, -2, None] - timeouts[None, :]) / self._scales[-2]
        dw = (a[:, -1, None] - waits[None, :]) / self._scales[-1]
        action = (
            np.exp(-0.5 * dt * dt)[:, :, None] * np.exp(-0.5 * dw * dw)[:, None, :]
        ).reshape(len(a), -1)
        return context_factor[:, None] * (self._level_var + self._signal_var * action)

    def _inputs(self, context: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return np.hstack(
            (
                np.repeat(context.reshape(1, -1), len(actions), axis=0),
                actions.reshape(-1, self._action_dim),
            )
        )

    def _choose(self, context: np.ndarray) -> np.ndarray | float:
        """The chosen grid row: a float at d = 1, a (T, W) pair at d = 2."""
        grid_inputs = self._inputs(context, self._grid)
        prior = self._prior.predict(grid_inputs)
        if not self._residuals:
            # A flat prior (ZeroPrior, before anything is learned) makes argmax
            # pick index 0, the shortest timeout, on every protocol. Nothing in
            # the model prefers it, so open in the middle of the range instead.
            if float(np.ptp(prior)) == 0.0:
                return self._grid[self._opening_index()]
            return self._grid[self._best_feasible(prior)]

        x = np.asarray(self._inputs_seen, dtype=float)
        y = np.asarray(self._residuals, dtype=float)
        k_xx = self._kernel(x, x) + self._noise_var * np.eye(len(x))
        k_xg = (
            self._kernel(x, grid_inputs)
            if self._action_dim == 1
            else self._kernel_grid(x, context)
        )
        chol = np.linalg.cholesky(k_xx)
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
        v = (
            np.linalg.solve(chol, k_xg)
            if self._action_dim == 1
            else solve_triangular(chol, k_xg, lower=True)
        )
        mean = k_xg.T @ alpha
        prior_var = self._level_var + self._signal_var
        sd = np.sqrt(np.clip(prior_var - np.sum(v * v, axis=0), 0.0, None))
        return self._grid[self._best_feasible(prior + mean - self._confidence * sd)]

    def _feasible_mask(self) -> np.ndarray:
        """Grid rows the latency floor permits: election timeout at or above
        its lower bound. A floor above the whole range leaves the longest
        timeout, as feasible_timeout_mask does on the discrete ladder."""
        timeouts = self._grid if self._action_dim == 1 else self._grid[:, 0]
        if self._latency_floor is None:
            return np.ones(len(timeouts), dtype=bool)
        mask = timeouts >= self._latency_floor.lower_bound() - 1e-12
        if not mask.any():
            mask = timeouts >= timeouts.max()
        return mask

    def _best_feasible(self, scores: np.ndarray) -> int:
        return int(np.argmax(np.where(self._feasible, scores, -np.inf)))

    def _opening_index(self) -> int:
        """Where a flat prior opens: the middle of the range, or the shortest
        permitted timeout when the floor rules the middle out."""
        if self._feasible[self._mid_index]:
            return self._mid_index
        return int(np.flatnonzero(self._feasible)[0])

    def predict_action(self, features: np.ndarray) -> np.ndarray:
        row = np.asarray(features, dtype=float).reshape(-1)
        if row.size != self._obs_dim:
            raise ValueError(f"Expected {self._obs_dim} features, got {row.size}")
        if self._latency_floor is not None:
            self._latency_floor.observe_features(row)
        self._feasible = self._feasible_mask()
        action = self._choose(report_context(self._spec, self._field_names, row))
        # A copy, never a view of the grid.
        return np.array(action, dtype=float).reshape(-1)

    def update_model(
        self,
        features: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | float,
        backward: bool = True,
    ) -> float:
        row = np.asarray(features, dtype=float).reshape(-1)
        if row.size != self._obs_dim:
            raise ValueError(f"Expected {self._obs_dim} features, got {row.size}")
        action = np.asarray(actions, dtype=float).reshape(-1)
        if action.size != self._action_dim:
            raise ValueError(
                f"Expected {self._action_dim} action values, got {action.size}"
            )
        reward = float(np.asarray(rewards, dtype=float).reshape(-1)[0])
        inputs = np.append(report_context(self._spec, self._field_names, row), action)
        residual = reward - float(self._prior.predict(inputs.reshape(1, -1))[0])
        self._inputs_seen.append(inputs)
        self._residuals.append(residual)
        return residual

    def reset_model(self) -> None:
        self._inputs_seen: deque[np.ndarray] = deque(maxlen=self._memory_size)
        self._residuals: deque[float] = deque(maxlen=self._memory_size)
        if self._latency_floor is not None:
            self._latency_floor.reset()
        self._feasible = self._feasible_mask()


class SimReturnGPModel(SimPriorGPModel):
    """SimPriorGPModel for rewards made of throughput and latency only
    (reward_tps_latency): a floored, scale-free fit.

    Such a reward scores a collapse at -5 and lower, and a prior fit to the raw
    reward spends its error on those tails and loses the 0.1-sized structure
    that ranks the usable timeouts (the p3 context of the tps_latency H18 draw
    read flat from 450 to 3000 ms). The prior here is fit with a target floor
    (tendermint_sim train --target-floor): rewards below the floor count as
    the floor, a collapse is a collapse. The model applies the same floor to
    the online reward, so the residual compares like with like; above the
    floor the reward is untouched, so the ranking of usable timeouts is the
    reward's own.

    The GP residual, its kernel variances and the confidence bound are in reward
    units (Kevin, 2026-10-19, restart D9e: every protocol's reward is tps / N minus
    latency in seconds, one scale, so one set of constants means the same on all
    six). Until then they were in units of each prior's training-target standard
    deviation (0.83-1.23 across the protocols' priors, 1.0 untrained), so the same
    constants meant a different width on each protocol and between a pretrained
    row and its untrained pair; the values (signal 0.13, noise 0.066, level 0.5)
    are those constants at the mean of that spread, about 1.

    The prior may also be fit to the held-action return (--hold-episodes h >
    1): the mean reward of holding the action h episodes from the context. On
    Tendermint that leans high at base under this reward, so its prior is per
    episode (hold 1). On SBFT the cost of a high T under a proposal delay lands
    at the delay's next re-pin, up to one interval after the T that will pay it
    was chosen, so a per-episode reward cannot see it. With h > 1 the online
    residual is taken against the matching target: the mean of the h rewards
    from that decision on, recorded h - 1 episodes later. At h = 1 the update
    is the per-episode one, unchanged. A term linear in the action was
    also tried in the residual (to extrapolate a slope to untried timeouts);
    replayed on real histories it changed nothing where the reward carried no
    slope and its extrapolations from two or three noisy points were wild, so
    the residual is H18's: a context-shared level plus a local action term.
    """

    def __init__(
        self,
        action_codec: ContinuousTimeoutActionCodec,
        field_names: Tuple[str, ...],
        prior_path: Optional[str],
        reward_design: str,
        grid_size: int = 64,
        signal_std: float = 0.13,
        noise_std: float = 0.066,
        action_length_scale: float = 0.1,
        memory_size: int = 256,
        confidence: float = 1.0,
        level_std: float = 0.5,
        wait_grid_size: int = 16,
        latency_floor: Optional[NormalizedLatencyFloor] = None,
        axes_ms: Optional[Tuple[Sequence[int], ...]] = None,
    ):
        super().__init__(
            action_codec,
            field_names,
            prior_path,
            reward_design,
            grid_size=grid_size,
            signal_std=signal_std,
            noise_std=noise_std,
            action_length_scale=action_length_scale,
            memory_size=memory_size,
            confidence=confidence,
            level_std=level_std,
            wait_grid_size=wait_grid_size,
            latency_floor=latency_floor,
            axes_ms=axes_ms,
        )
        if not math.isfinite(self._prior.target_floor):
            raise ValueError(
                f"Prior {prior_path} was fit to the raw reward; SimReturnGPModel needs a "
                f"floored fit: {self._spec.simulator} train --target-floor -2 "
                f"--reward-design {reward_design}"
            )
        self._floor = float(self._prior.target_floor)
        # Reward units on every protocol (class docstring); the prior predicts in them.
        self._y_scale = 1.0
        self._hold = int(self._prior.hold_episodes)
        if self._hold < 1:
            raise ValueError(f"Prior {prior_path} has hold_episodes {self._hold}")
        # Decisions whose h-episode return is not complete yet, (inputs, raw reward),
        # are kept in self._pending (reset_model).

    def reset_model(self) -> None:
        super().reset_model()
        self._pending = deque()

    def _prior_scaled(self, inputs: np.ndarray) -> np.ndarray:
        return self._prior.predict(inputs) / self._y_scale

    def _choose(self, context: np.ndarray) -> np.ndarray | float:
        grid_inputs = self._inputs(context, self._grid)
        prior = self._prior_scaled(grid_inputs)
        if not self._residuals:
            # See SimPriorGPModel._choose: a flat prior carries no preference,
            # so open in the middle of the range rather than at index 0.
            if float(np.ptp(prior)) == 0.0:
                return self._grid[self._opening_index()]
            return self._grid[self._best_feasible(prior)]
        x = np.asarray(self._inputs_seen, dtype=float)
        y = np.asarray(self._residuals, dtype=float)
        k_xx = self._kernel(x, x) + self._noise_var * np.eye(len(x))
        k_xg = (
            self._kernel(x, grid_inputs)
            if self._action_dim == 1
            else self._kernel_grid(x, context)
        )
        chol = np.linalg.cholesky(k_xx)
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
        v = (
            np.linalg.solve(chol, k_xg)
            if self._action_dim == 1
            else solve_triangular(chol, k_xg, lower=True)
        )
        mean = k_xg.T @ alpha
        prior_var = self._level_var + self._signal_var
        sd = np.sqrt(np.clip(prior_var - np.sum(v * v, axis=0), 0.0, None))
        return self._grid[self._best_feasible(prior + mean - self._confidence * sd)]

    def update_model(
        self,
        features: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | float,
        backward: bool = True,
    ) -> float:
        row = np.asarray(features, dtype=float).reshape(-1)
        if row.size != self._obs_dim:
            raise ValueError(f"Expected {self._obs_dim} features, got {row.size}")
        action = np.asarray(actions, dtype=float).reshape(-1)
        if action.size != self._action_dim:
            raise ValueError(
                f"Expected {self._action_dim} action values, got {action.size}"
            )
        inputs = np.append(report_context(self._spec, self._field_names, row), action)
        if self._hold == 1:
            reward = max(self._floor, float(np.asarray(rewards, dtype=float).reshape(-1)[0]))
            residual = reward / self._y_scale - float(self._prior_scaled(inputs.reshape(1, -1))[0])
            self._inputs_seen.append(inputs)
            self._residuals.append(residual)
            return residual
        # h > 1: the oldest pending decision gets the mean of its own reward and the
        # h - 1 that followed it, floored as the prior's targets were.
        self._pending.append((inputs, float(np.asarray(rewards, dtype=float).reshape(-1)[0])))
        if len(self._pending) < self._hold:
            return float("nan")
        target = max(self._floor, float(np.mean([r for _, r in self._pending])))
        oldest, _ = self._pending.popleft()
        residual = target / self._y_scale - float(self._prior_scaled(oldest.reshape(1, -1))[0])
        self._inputs_seen.append(oldest)
        self._residuals.append(residual)
        return residual


# Candidate timeouts scored at each decision. Spread over the arm range in the
# codec's log space, 65 points put neighbours within about 2.5% of each other -
# far finer than a single 5s window can resolve, so the grid is not what limits
# the choice.
DEFAULT_CANDIDATE_COUNT = 65


# Basis centres over the action range (Kevin, 2026-10-19: every continuous model plays the
# protocol's whole range, 100-3000, not the ladder's span): log-spaced, nine across 100-3000
# (a ratio of 1.53 between neighbours, the density of the refined three-rung ladder it replaces).
LINUCB_BASIS_CENTERS = 9
# SBFT's fast-path wait W (Kevin, 2026-10-19, restart D10c: the continuous models choose W
# too): a separate basis over the wait range, added to the T basis (option b, so T and W
# enter the reward additively), 4 log-spaced centres across 10-500, and 16 candidate waits.
LINUCB_WAIT_BASIS_CENTERS = 4
LINUCB_WAIT_CANDIDATES = 16


def _log_grid_ms(lower_ms: int, upper_ms: int, count: int) -> Tuple[int, ...]:
    """``count`` timeouts spaced evenly in log from ``lower_ms`` to ``upper_ms``."""
    if count < 2:
        raise ValueError(f"count must be >= 2, got {count}")
    ratio = (float(upper_ms) / float(lower_ms)) ** (1.0 / (count - 1))
    return tuple(sorted({int(round(lower_ms * ratio ** i)) for i in range(count)}))


class ContinuousLinUCB(TensorModel):
    """Discounted LinUCB over a continuous timeout.

    ``DiscountedLinUCB`` (``discrete_linucb``) fits one ridge model per arm, which is
    the same thing as one shared model over ``one_hot(arm)`` outer-product the
    context. This is that model with the one-hot replaced by a smooth basis
    over the log-timeout::

        z(x, a) = phi(a) (x) x          (Kronecker product)

    Everything else is carried over unchanged: the signed-log row-normalized
    context, the ridge prior, the discount on every statistic, the UCB
    half-width from the discounted covariance sandwich, the random exploration
    floor, and playing each basis centre once before anything is fitted, which
    is what discrete_linucb's ``+inf`` score for an unpulled arm amounts to. Place the
    centres at the arms and shrink their width to zero and the two models are
    the same model; the parameter count is already identical, ``len(centres) *
    (n_features + 1)``.

    What the smooth basis buys is that neighbouring timeouts share evidence.
    On the ladder an arm is only informed by its own pulls, so an arm avoided
    for a phase keeps a stale estimate and comes back on the confidence bonus
    alone. Here a reward earned at 600ms moves the estimate at 550ms and 700ms
    with it, and the exploration bonus is a bonus for a *region* of the action
    space that has not been tried lately.

    **The basis sums to one at every timeout, and this is not cosmetic.** The
    prediction is then ``sum_j w_j(a) (theta_j . x)``, a weighted average of
    the per-centre reward models, so it cannot exceed the best centre and the
    fitted optimum is always at one. Normalizing the bumps to unit L2 norm
    instead - which looks more like the one-hot it replaces, and keeps the
    confidence bonus from favouring the ends - is a trap: the prediction
    becomes an inner product that rewards spreading weight over several
    positive coefficients, so a blend between two decent centres scores above
    either of them, and since the coefficients are built from the action being
    played, whatever the model plays becomes its own argmax. Measured on
    local/pbft_c6525100g_utah_2/continuous_codex_mixed, that locked the agent
    at 370ms - a timeout no baseline plays - for 104 of 107 episodes and cost
    4600 tps against the same model's heuristic.

    **The centres are LINUCB_BASIS_CENTERS log-spaced timeouts across the
    action range** (2026-10-19; the arms plus the midpoint between each pair
    until then, at the same density). With centres only at the arms, an average
    over them is maximised at an arm and the continuous action space buys
    nothing; at this density the model can settle between two of them when
    that is where the reward is (measured on a synthetic reward paying only in
    700-900ms: 90% of decisions inside it with the ladder midpoints, 0%
    without). Each centre costs ``n_features + 1`` coefficients.

    The bumps are half a centre-spacing wide, so neighbours overlap at about
    14%: wide enough to share evidence, narrow enough that adjacent centres
    still say different things.

    **The one departure from discrete_linucb: rewards are centred.** The ridge prior
    is zero, and the discount pulls an unplayed timeout's estimate toward it,
    so zero is what an untried timeout is assumed to be worth. That is only
    sane if the rewards are centred on zero, and they are not: reward_tps is
    1.0 at its 1000 tps knee and this cluster runs at 40000, so a reward is 2
    to 10 and a forgotten timeout reads as worse than the worst thing ever
    measured. Measured on the first run of
    local/pbft_c6525100g_utah_2/continuous_codex_mixed: by the last episode
    the timeout being played scored 5.35 against 0.11 for the longest one,
    while the confidence bonus meant to bring it back tops out at 1.24 - so
    the agent held one timeout for 102 of 107 episodes and never answered a
    phase change. Subtracting a discounted mean of recent rewards makes the
    prior mean "average", which is what optimism in the face of uncertainty
    assumes: a timeout not tried lately is worth finding out about, not
    written off. discrete_linucb centres the same way since 2026-10-19 (restart
    D9d); pass ``center_rewards=False`` to reproduce the uncentred model here.

    Centring is also why the discount is 0.90 rather than discrete_linucb's 0.95.
    The effective memory is ``1 / (1 - gamma)`` observations, and a phase of
    this spec is about 15 episodes at a 5s window, so 0.95 remembers 20 - it
    cannot forget a phase inside that phase. 0.95 was nonetheless right for
    discrete_linucb, and measurably so: with an uncentred reward, forgetting decays
    an estimate toward "worthless", so faster forgetting buries the arm it was
    supposed to free. Once the prior means "average", the two pull the same
    way. On a seven-phase benchmark whose paying timeout moves under the agent
    (3 seeds, 15 episodes per phase, rewards on this hardware's 2..10 scale),
    total regret: uncentred 434 at 0.95 and 456 at 0.85 - the ordering the
    hardware showed for discrete_linucb - against centred 320 at 0.95 and 184 at
    0.90.

    **Neither of those two fixes answers how the agent got short in the first
    place, and nothing in this class does.** The initial design plays every
    centre once, and on that run all five decisions landed inside phase 0,
    where the client's offered rate caps throughput: the five timeouts read
    39114..40049 tps, a 2.4% spread, so reward_mixed's
    ``window / (window + timeout + latency)`` discount was the only signal
    left in the sweep - 9.85, 9.81, 9.27, 8.94, 8.34 at 250, 371, 550, 812 and
    1200ms. The sweep fits the reward's own penalty gradient rather than the
    protocol, and the way back up is then the exploration floor alone: 0.05
    over a log-spaced grid whose span above 500ms is 55.8% of it is 2.8% per
    decision, or 0.42 expected probes above 500ms in a 15-episode phase.
    Phases 1, 3, 4 and 6 of that run drew none; the 497ms probe that landed in
    phase 5 scored 2.66 against 1.95 for the twelve episodes the held timeout
    played there and was one sample against forty. So the fault phases are
    never sampled at the length that pays - phase 4 committed 9641 tps here
    against 13234 for the same model at 550ms under reward_tps and 15139 for
    the oracle. Interleaving the design sweep across phases, and drawing the
    exploration floor over the centres rather than over the candidate grid,
    are the two things that would change that; neither is done here.

    The action space is the protocol's whole range (the codec's ContextSpec
    bounds, 100-3000 ms on every protocol; Kevin, 2026-10-19), as for every
    continuous model; until then it was the ladder's span, so that it played no
    timeout a discrete baseline could not. A latency floor, when one is configured,
    trims the bottom of that range exactly as ``feasible_timeout_mask`` trims
    the ladder - the model is not otherwise aware of it, and with no floor
    configured it is a bandit over the range and nothing else.
    """

    def __init__(
        self,
        action_codec: ContinuousTimeoutActionCodec,
        n_features: int,
        basis_centers_ms: Optional[Sequence[int]] = None,
        clamp_ms: Optional[Tuple[int, int]] = None,
        candidate_count: int = DEFAULT_CANDIDATE_COUNT,
        confidence_scale: float = 1.0,
        # 0.95 and no random share since 2026-10-19 (were 0.90 / 0.05; Kevin, runs/2026-10-19-linucb-discount):
        # one discount for both LinUCB models (0.90-0.98 within 0.01); the random share cost
        # HotStuff about 0.10 and nothing measurable elsewhere.
        discount_factor: float = 0.95,
        exploration_floor: float = 0.0,
        ridge: float = 1.0,
        reward_scale: float = 1.0,
        center_rewards: bool = True,
        latency_floor: Optional[NormalizedLatencyFloor] = None,
        seed: Optional[int] = None,
        prior_theta: Optional[np.ndarray] = None,
    ) -> None:
        if n_features <= 0:
            raise ValueError(f"n_features must be > 0, got {n_features}")
        if action_codec.action_dim not in (1, 2):
            raise ValueError(
                "ContinuousLinUCB places a timeout (and on SBFT a wait) on a line, so it "
                f"needs a 1- or 2-dimensional codec, got {action_codec.action_dim}"
            )
        if candidate_count < 2:
            raise ValueError(
                f"candidate_count must be >= 2, got {candidate_count}"
            )
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

        codec_lower_ms, codec_upper_ms = action_codec.bounds_ms[0]
        if clamp_ms is None:
            # The protocol's whole range, as every continuous model (2026-10-19).
            clamp_ms = (codec_lower_ms, codec_upper_ms)
        lower_ms, upper_ms = (int(clamp_ms[0]), int(clamp_ms[1]))
        if upper_ms <= lower_ms:
            raise ValueError(f"Invalid clamp bounds: {lower_ms}..{upper_ms}ms")
        if lower_ms < codec_lower_ms or upper_ms > codec_upper_ms:
            raise ValueError(
                f"Clamp bounds {lower_ms}..{upper_ms}ms fall outside the codec "
                f"bounds {codec_lower_ms}..{codec_upper_ms}ms"
            )

        centers_ms = tuple(
            sorted(
                int(value)
                for value in (basis_centers_ms or _log_grid_ms(lower_ms, upper_ms, LINUCB_BASIS_CENTERS))
            )
        )
        if not centers_ms:
            raise ValueError("basis_centers_ms must not be empty")
        if centers_ms[0] < lower_ms or centers_ms[-1] > upper_ms:
            raise ValueError(
                f"Basis centres {centers_ms} fall outside the action range "
                f"{lower_ms}..{upper_ms}ms"
            )

        self._action_codec = action_codec
        self._obs_dim = int(n_features)
        self._lower_ms = lower_ms
        self._upper_ms = upper_ms
        self._centers_ms = centers_ms
        self._candidate_count = int(candidate_count)
        self._confidence_scale = float(confidence_scale)
        self._discount_factor = float(discount_factor)
        self._exploration_floor = float(exploration_floor)
        self._ridge = float(ridge)
        self._reward_scale = float(reward_scale)
        self._center_rewards = bool(center_rewards)
        self._latency_floor = latency_floor
        self._seed = seed

        self._context_dim = self._obs_dim + 1
        self._centers = np.asarray(
            [self._position_of(value) for value in centers_ms], dtype=np.float64
        )
        self._n_basis = len(self._centers)
        # Half the mean centre spacing; see the class docstring for why not
        # a full one.
        self._basis_width = (
            1.0
            if self._n_basis < 2
            else float(
                0.5
                * (self._centers[-1] - self._centers[0])
                / (self._n_basis - 1)
            )
        )
        if self._basis_width <= 0.0:
            self._basis_width = 1.0
        # SBFT: the wait W on its own basis, added to the timeout's (LINUCB_WAIT_BASIS_CENTERS).
        self._two_d = action_codec.action_dim == 2
        self._n_wait_basis = 0
        self._held_wait_ms: Optional[int] = None
        if self._two_d:
            self._wait_lower_ms, self._wait_upper_ms = (int(v) for v in action_codec.bounds_ms[1])
            self._wait_centers_ms = _log_grid_ms(
                self._wait_lower_ms, self._wait_upper_ms, LINUCB_WAIT_BASIS_CENTERS
            )
            self._wait_centers = np.asarray(
                [self._wait_position_of(v) for v in self._wait_centers_ms], dtype=np.float64
            )
            self._n_wait_basis = len(self._wait_centers)
            self._wait_basis_width = float(
                0.5 * (self._wait_centers[-1] - self._wait_centers[0]) / (self._n_wait_basis - 1)
            )
            self._wait_positions = np.linspace(0.0, 1.0, LINUCB_WAIT_CANDIDATES, dtype=np.float64)
            opening = opening_action_ms(action_codec.protocol)
            # The design sweep holds the other dimension at the opening action.
            self._opening_positions = (
                self._position_of(opening[0]), self._wait_position_of(opening[1])
            )
        self._opening_position = self._position_of(opening_action_ms(action_codec.protocol)[0])
        self._feature_dim = (self._n_basis + self._n_wait_basis) * self._context_dim
        self._positions = np.linspace(
            0.0, 1.0, self._candidate_count, dtype=np.float64
        )
        # Prior mean of the coefficients, in the _joint layout and in the
        # units of the update's target (centred if center_rewards, over
        # reward_scale): b starts at ridge * theta0 and the discount pulls it
        # back there. Zeros (None) is the plain ridge prior.
        if prior_theta is None:
            self._prior_theta = np.zeros(self._feature_dim, dtype=np.float64)
        else:
            self._prior_theta = np.array(prior_theta, dtype=np.float64)
            if self._prior_theta.shape != (self._feature_dim,):
                raise ValueError(
                    f"prior_theta must have shape ({self._feature_dim},), "
                    f"got {self._prior_theta.shape}"
                )
            if not np.all(np.isfinite(self._prior_theta)):
                raise ValueError("prior_theta must contain only finite values")

        self.reset_model()

    @property
    def floor_ms(self) -> float:
        return 0.0 if self._latency_floor is None else self._latency_floor.floor_ms

    def fit_prior_theta(
        self, features: np.ndarray, actions: np.ndarray, rewards: np.ndarray
    ) -> np.ndarray:
        """Undiscounted ridge fit, for use as ``prior_theta``.

        ``actions`` are codec-normalised, as update_model takes them, and the
        inputs are built exactly as update_model builds them. With
        center_rewards the target is the reward minus the mean of ``rewards``,
        the level the online baseline settles at, over reward_scale.
        """
        rows = np.asarray(features, dtype=float)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        positions = self._action_positions(actions, len(rows))
        reward_values = np.asarray(rewards, dtype=np.float64).reshape(-1)
        if reward_values.size != len(rows):
            raise ValueError(
                f"{len(rows)} feature rows and {reward_values.size} rewards"
            )
        if not np.all(np.isfinite(reward_values)):
            raise ValueError("rewards must contain only finite values")
        inputs = np.vstack(
            [
                self._joint(self._context(row), position[None]).reshape(-1)
                for row, position in zip(rows, positions)
            ]
        )
        targets = reward_values
        if self._center_rewards:
            targets = targets - float(np.mean(reward_values))
        targets = targets / self._reward_scale
        identity = np.eye(self._feature_dim, dtype=np.float64)
        return np.linalg.solve(
            self._ridge * identity + inputs.T @ inputs, inputs.T @ targets
        )

    def _position_of(self, timeout_ms: float) -> float:
        """Where a timeout sits in the action range, in [0, 1] log-space."""
        value = float(timeout_ms)
        if not math.isfinite(value) or value <= 0.0:
            return 0.0
        position = (math.log(value) - math.log(float(self._lower_ms))) / (
            math.log(float(self._upper_ms)) - math.log(float(self._lower_ms))
        )
        return float(min(1.0, max(0.0, position)))

    def _timeout_of(self, position: float) -> int:
        bounded = min(1.0, max(0.0, float(position)))
        log_value = math.log(float(self._lower_ms)) + bounded * (
            math.log(float(self._upper_ms)) - math.log(float(self._lower_ms))
        )
        return min(
            self._upper_ms, max(self._lower_ms, int(round(math.exp(log_value))))
        )

    def _wait_position_of(self, wait_ms: float) -> float:
        """SBFT: where a wait sits in the wait range, in [0, 1] log-space."""
        value = float(wait_ms)
        if not math.isfinite(value) or value <= 0.0:
            return 0.0
        position = (math.log(value) - math.log(float(self._wait_lower_ms))) / (
            math.log(float(self._wait_upper_ms)) - math.log(float(self._wait_lower_ms))
        )
        return float(min(1.0, max(0.0, position)))

    def _wait_of(self, position: float) -> int:
        bounded = min(1.0, max(0.0, float(position)))
        log_value = math.log(float(self._wait_lower_ms)) + bounded * (
            math.log(float(self._wait_upper_ms)) - math.log(float(self._wait_lower_ms))
        )
        return min(self._wait_upper_ms, max(self._wait_lower_ms, int(round(math.exp(log_value)))))

    def _action_positions(self, actions: np.ndarray, batch_size: int) -> np.ndarray:
        """Codec-normalised actions -> positions: (n, 1), or (n, 2) with the wait on SBFT.

        The agent trains on the timeout the replicas actually ran with, which is not
        always the one this model asked for; a value outside the action range lands on
        the nearest end of it."""
        width = 2 if self._two_d else 1
        values = np.asarray(actions, dtype=np.float64).reshape(-1)
        if values.size == width and batch_size > 1:
            values = np.tile(values, batch_size)
        if values.size != batch_size * width:
            raise ValueError(
                f"actions batch size mismatch: expected {batch_size} x {width}, got {values.size}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("actions must contain only finite values")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("Continuous actions must be normalized to [0, 1]")
        values = values.reshape(batch_size, width)
        positions = np.empty_like(values)
        for i, row in enumerate(values):
            positions[i, 0] = self._position_of(self._action_codec.denormalize_ms(float(row[0]), 0))
            if self._two_d:
                positions[i, 1] = self._wait_position_of(
                    self._action_codec.denormalize_ms(float(row[1]), 1)
                )
        return positions

    def _lower_position(self) -> float:
        """Smallest position the latency floor permits, in [0, 1].

        A floor above the whole range leaves the longest timeout, which is what
        feasible_timeout_mask leaves on the discrete ladder.
        """
        if self._latency_floor is None:
            return 0.0
        floor_ms = float(self._latency_floor.floor_ms)
        if not math.isfinite(floor_ms) or floor_ms <= self._lower_ms:
            return 0.0
        if floor_ms >= self._upper_ms:
            return 1.0
        return self._position_of(math.ceil(floor_ms))

    def _candidates(self, lower_position: float) -> np.ndarray:
        if lower_position <= 0.0:
            return self._positions
        feasible = self._positions[self._positions >= lower_position]
        if len(feasible) == 0:
            return np.asarray([lower_position], dtype=np.float64)
        # The bound itself is a legal timeout and is the shortest one left, so
        # it is offered even when the grid steps over it.
        return np.concatenate(([lower_position], feasible))

    def _context(self, row: np.ndarray) -> np.ndarray:
        values = np.asarray(row, dtype=np.float64).reshape(-1)
        if values.size != self._obs_dim:
            raise ValueError(
                f"Expected {self._obs_dim} features, got {values.size}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("features must contain only finite values")
        transformed = np.sign(values) * np.log1p(np.abs(values))
        norm = float(np.linalg.norm(transformed))
        transformed = transformed / max(norm, 1.0)
        return np.concatenate(([1.0], transformed))

    def _basis(self, positions: np.ndarray) -> np.ndarray:
        values = np.asarray(positions, dtype=np.float64).reshape(-1, 1)
        distances = (values - self._centers.reshape(1, -1)) / self._basis_width
        weights = np.exp(-0.5 * distances * distances)
        totals = np.sum(weights, axis=1, keepdims=True)
        return weights / np.maximum(totals, 1e-12)

    def _wait_basis(self, positions: np.ndarray) -> np.ndarray:
        values = np.asarray(positions, dtype=np.float64).reshape(-1, 1)
        distances = (values - self._wait_centers.reshape(1, -1)) / self._wait_basis_width
        weights = np.exp(-0.5 * distances * distances)
        totals = np.sum(weights, axis=1, keepdims=True)
        return weights / np.maximum(totals, 1e-12)

    def _joint(self, context: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Inputs for positions (n,) / (n, 1), or (n, 2) on SBFT: the timeout's basis
        and, on SBFT, the wait's basis beside it, each times the context."""
        values = np.asarray(positions, dtype=np.float64)
        if self._two_d:
            values = values.reshape(-1, 2)
            basis = np.concatenate(
                (self._basis(values[:, 0]), self._wait_basis(values[:, 1])), axis=1
            )
        else:
            basis = self._basis(values.reshape(-1))
        return basis[:, :, None] * context[None, None, :]

    def _scores(self, context: np.ndarray, positions: np.ndarray) -> np.ndarray:
        inputs = self._joint(context, positions).reshape(len(positions), -1)
        try:
            theta = np.linalg.solve(self._precision, self._reward_vector)
            directions = np.linalg.solve(self._precision, inputs.T).T
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "continuous_linucb precision matrix is singular"
            ) from exc
        predicted = inputs @ theta
        variances = np.einsum(
            "ij,jk,ik->i", directions, self._covariance, directions
        )
        return predicted + self._confidence_scale * np.sqrt(
            np.maximum(variances, 0.0)
        )

    def _select_position(
        self, context: np.ndarray, lower_position: float
    ) -> float:
        if self._design_index == 0:
            # The opening action first (the ladder's middle rung, and on SBFT the minimum
            # wait; Kevin, 2026-10-19, restart D10b: every model opens there).
            self._design_index += 1
            if not self._two_d:
                return np.asarray([max(lower_position, self._opening_position)])
            return np.asarray([max(lower_position, self._opening_positions[0]), self._opening_positions[1]])
        if self._design_index <= self._n_basis + self._n_wait_basis:
            # One decision per basis centre before anything is fitted: the
            # continuous form of discrete_linucb scoring an unpulled arm at +inf, and
            # deterministic, so replicas sharing a feature vector still draw
            # the same action. On SBFT each timeout centre at the opening wait, then
            # each wait centre at the opening timeout.
            index = self._design_index - 1
            self._design_index += 1
            if index < self._n_basis:
                centre = float(self._centers[index])
                if not self._two_d:
                    return np.asarray([max(lower_position, centre)])
                return np.asarray([max(lower_position, centre), self._opening_positions[1]])
            return np.asarray([
                max(lower_position, self._opening_positions[0]),
                float(self._wait_centers[index - self._n_basis]),
            ])

        timeouts = self._candidates(lower_position)
        if self._two_d:
            candidates = np.column_stack((
                np.repeat(timeouts, len(self._wait_positions)),
                np.tile(self._wait_positions, len(timeouts)),
            ))
        else:
            candidates = timeouts.reshape(-1, 1)
        if self._exploration_floor > 0.0 and self._rng.random() < (
            self._exploration_floor
        ):
            return candidates[int(self._rng.integers(len(candidates)))].copy()

        scores = self._scores(context, candidates)
        best = float(np.max(scores))
        tied = np.flatnonzero(scores >= best - 1e-12)
        return candidates[int(self._rng.choice(tied))].copy()

    def predict_action(self, features: np.ndarray) -> np.ndarray:
        rows = np.asarray(features, dtype=float)
        single = rows.ndim == 1
        if single:
            rows = rows.reshape(1, -1)
        if rows.shape[1] != self._obs_dim:
            raise ValueError(
                f"Expected {self._obs_dim} features, got {rows.shape[1]}"
            )

        actions = []
        for row in rows:
            if self._latency_floor is not None:
                self._latency_floor.observe_features(row)
            position = self._select_position(
                self._context(row), self._lower_position()
            )
            self._held_timeout_ms = self._timeout_of(float(position[0]))
            action = [self._action_codec.normalize_ms(self._held_timeout_ms, 0)]
            if self._two_d:
                self._held_wait_ms = self._wait_of(float(position[1]))
                action.append(self._action_codec.normalize_ms(self._held_wait_ms, 1))
            actions.append(action)

        result = np.asarray(actions, dtype=float)
        return result[0] if single else result

    def _baseline(self) -> float:
        """What an unmeasured timeout is assumed to be worth.

        Zero - the ridge prior itself - when centring is off.
        """
        if not self._center_rewards or self._baseline_weight <= 0.0:
            return 0.0
        return self._baseline_total / self._baseline_weight

    def _observe_reward(self, reward: float) -> None:
        """Fold one reward into the baseline, discounted like everything else."""
        self._baseline_total = (
            self._discount_factor * self._baseline_total + reward
        )
        self._baseline_weight = self._discount_factor * self._baseline_weight + 1.0

    def _discount(self) -> None:
        """One observation's decay toward the ridge prior.

        Precision at gamma and covariance at gamma squared, as in
        DiscountedLinUCB: that is what keeps the confidence half-width a
        variance rather than a count of pulls.
        """
        identity = np.eye(self._feature_dim, dtype=np.float64)
        prior_weight = (1.0 - self._discount_factor) * self._ridge
        squared_discount = self._discount_factor * self._discount_factor
        covariance_prior_weight = (1.0 - squared_discount) * self._ridge
        self._precision *= self._discount_factor
        if prior_weight > 0.0:
            self._precision += prior_weight * identity
        self._covariance *= squared_discount
        if covariance_prior_weight > 0.0:
            self._covariance += covariance_prior_weight * identity
        self._reward_vector *= self._discount_factor
        self._reward_vector += prior_weight * self._prior_theta

    def update_model(
        self,
        features: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | float,
        backward: bool = True,
    ) -> float:
        if not backward:
            return 0.0

        rows = np.asarray(features, dtype=float)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.shape[1] != self._obs_dim:
            raise ValueError(
                f"Expected {self._obs_dim} features, got {rows.shape[1]}"
            )

        batch_size = len(rows)
        positions = self._action_positions(actions, batch_size)
        reward_values = np.asarray(rewards, dtype=np.float64).reshape(-1)
        if reward_values.size == 1 and batch_size > 1:
            reward_values = np.repeat(reward_values, batch_size)
        if reward_values.size != batch_size:
            raise ValueError(
                f"rewards batch size mismatch: expected {batch_size}, "
                f"got {reward_values.size}"
            )
        if not np.all(np.isfinite(reward_values)):
            raise ValueError("rewards must contain only finite values")

        for row, position, reward in zip(rows, positions, reward_values):
            inputs = self._joint(self._context(row), position[None]).reshape(-1)
            # With centring, the first reward is centred on itself (no baseline yet), so
            # the first timeout played is not credited the whole raw reward.
            first = self._center_rewards and self._baseline_weight <= 0.0
            centered = 0.0 if first else float(reward) - self._baseline()
            self._discount()
            self._precision += np.outer(inputs, inputs)
            self._covariance += np.outer(inputs, inputs)
            self._reward_vector += (centered / self._reward_scale) * inputs
            self._observe_reward(float(reward))

        return 0.0

    def reset_model(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        if self._latency_floor is not None:
            self._latency_floor.reset()
        identity = np.eye(self._feature_dim, dtype=np.float64)
        self._precision = self._ridge * identity
        self._covariance = self._precision.copy()
        self._reward_vector = self._ridge * self._prior_theta
        self._design_index = 0
        self._baseline_total = 0.0
        self._baseline_weight = 0.0
        self._held_timeout_ms = self._lower_ms
        if self._two_d:
            self._held_wait_ms = self._wait_lower_ms


class ContinuousPPO(PPOBase):
    """PPO with a Beta policy over the continuous timeout.

    :class:`DiscretePPO`'s softmax is replaced by a Beta density over the
    position of the timeout in the arm range, measured in log space exactly as
    :class:`ContinuousLinUCB` measures it. The actor emits two numbers,
    ``alpha = 1 + softplus(z0)`` and ``beta = 1 + softplus(z1)``; holding both
    at or above one keeps the density unimodal and finite at the ends, so the
    policy can concentrate anywhere in the range but cannot put an atom on a
    boundary and get stuck there.

    **Beta rather than a clipped Gaussian.** The action is bounded on both
    sides and the obvious alternative - sample a Gaussian, clip to the range -
    piles the clipped mass onto the two end points and reports a density that
    does not integrate to what was actually sampled, which shows up as a biased
    importance ratio precisely at the ends of the ladder where this spec's
    interesting arms live. The Beta is supported on exactly the interval being
    sampled, so no correction is needed.

    **It starts almost uniform, and that is the point.** The output bias is set
    so that ``alpha = beta ~ 1.13`` before any learning: a very broad bump over
    the whole arm range. ``ContinuousLinUCB`` instead opens with a design sweep
    of one decision per basis centre, and its own docstring records what that
    cost - all five landed inside phase 0, where the offered load caps
    throughput and the five timeouts differ by 2.4%, so the sweep fitted the
    reward's penalty gradient rather than the protocol and the model never
    recovered. A broad policy keeps sampling the whole range as the phases go
    by, so the long timeouts get tried inside the phases that pay for them.

    **The action space is the protocol's whole range** (the codec's ContextSpec
    bounds, 100-3000 ms; 2026-10-19, the ladder's span until then), as for
    :class:`ContinuousLinUCB`. A latency
    floor, when configured, raises the bottom of the range; the sampled
    position is clamped up to it, which puts a small atom on the floor itself,
    and that is treated as part of the environment in the same way the discrete
    models treat ``feasible_timeout_mask``.
    """

    def __init__(
        self,
        action_codec: ContinuousTimeoutActionCodec,
        n_features: int,
        clamp_ms: Optional[Tuple[int, int]] = None,
        hidden_sizes: Sequence[int] = (),
        latency_floor: Optional[NormalizedLatencyFloor] = None,
        initial_concentration_bias: float = -2.0,
        **kwargs: Any,
    ) -> None:
        if action_codec.action_dim not in (1, 2):
            raise ValueError(
                "ContinuousPPO draws a timeout (and on SBFT a wait), so it needs a "
                f"1- or 2-dimensional codec, got {action_codec.action_dim}"
            )
        # SBFT (Kevin, 2026-10-19, restart D10c): a second, independent Beta over the wait
        # range; the action's log-probability and entropy are the sums over the two.
        self._two_d = action_codec.action_dim == 2
        self._action_width = action_codec.action_dim
        if self._two_d:
            self._wait_lower_ms, self._wait_upper_ms = (int(v) for v in action_codec.bounds_ms[1])
            self._wait_log_lower = math.log(float(self._wait_lower_ms))
            self._wait_log_span = math.log(float(self._wait_upper_ms)) - self._wait_log_lower
        self._action_codec = action_codec
        self._latency_floor = latency_floor
        self._initial_concentration_bias = float(initial_concentration_bias)

        codec_lower_ms, codec_upper_ms = action_codec.bounds_ms[0]
        if clamp_ms is None:
            # The protocol's whole range, as every continuous model (2026-10-19).
            clamp_ms = (codec_lower_ms, codec_upper_ms)
        lower_ms, upper_ms = int(clamp_ms[0]), int(clamp_ms[1])
        if lower_ms < codec_lower_ms or upper_ms > codec_upper_ms:
            raise ValueError(
                f"clamp {lower_ms}..{upper_ms}ms is outside the codec bounds "
                f"{codec_lower_ms}..{codec_upper_ms}ms"
            )
        if upper_ms <= lower_ms:
            raise ValueError(
                f"clamp upper bound must exceed lower, got {lower_ms}..{upper_ms}"
            )
        self._lower_ms = lower_ms
        self._upper_ms = upper_ms
        self._log_lower = math.log(float(lower_ms))
        self._log_span = math.log(float(upper_ms)) - self._log_lower

        super().__init__(
            n_features=n_features,
            head_dim=2 * action_codec.action_dim,  # (alpha, beta) per dimension
            hidden_sizes=hidden_sizes,
            **kwargs,
        )

    @property
    def _output_bias(self) -> float:
        return self._initial_concentration_bias

    # -- diagnostics --------------------------------------------------------

    @property
    def floor_ms(self) -> float:
        return 0.0 if self._latency_floor is None else self._latency_floor.floor_ms

    # -- action range -------------------------------------------------------

    def _position_of(self, timeout_ms: float) -> float:
        value = float(timeout_ms)
        if not math.isfinite(value) or value <= 0.0:
            return 0.0
        position = (math.log(value) - self._log_lower) / self._log_span
        return float(min(1.0, max(0.0, position)))

    def _timeout_of(self, position: float) -> int:
        bounded = min(1.0, max(0.0, float(position)))
        value = math.exp(self._log_lower + bounded * self._log_span)
        return min(self._upper_ms, max(self._lower_ms, int(round(value))))

    def _lower_position(self) -> float:
        """Smallest position the latency floor permits, in [0, 1]."""
        if self._latency_floor is None:
            return 0.0
        floor_ms = float(self._latency_floor.floor_ms)
        if not math.isfinite(floor_ms) or floor_ms <= self._lower_ms:
            return 0.0
        if floor_ms >= self._upper_ms:
            return 1.0
        return self._position_of(math.ceil(floor_ms))

    # -- Beta head ----------------------------------------------------------

    def _concentrations(self, head: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(alpha, beta) = 1 + softplus(head), computed without overflow; each (n, d),
        the head laid out alpha, beta per dimension."""
        values = np.asarray(head, dtype=np.float64)
        softplus = np.logaddexp(0.0, values)
        concentrations = 1.0 + softplus
        return concentrations[:, 0::2], concentrations[:, 1::2]

    def _wait_position_of(self, wait_ms: float) -> float:
        value = float(wait_ms)
        if not math.isfinite(value) or value <= 0.0:
            return 0.0
        position = (math.log(value) - self._wait_log_lower) / self._wait_log_span
        return float(min(1.0, max(0.0, position)))

    def _wait_of(self, position: float) -> int:
        bounded = min(1.0, max(0.0, float(position)))
        value = math.exp(self._wait_log_lower + bounded * self._wait_log_span)
        return min(self._wait_upper_ms, max(self._wait_lower_ms, int(round(value))))

    def _head_terms(
        self, head: np.ndarray, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        alpha, beta = self._concentrations(head)
        positions = np.clip(
            np.asarray(actions, dtype=np.float64).reshape(len(alpha), -1), 1e-6, 1.0 - 1e-6
        )
        log_u = np.log(positions)
        log_1mu = np.log1p(-positions)

        log_density = (
            (alpha - 1.0) * log_u + (beta - 1.0) * log_1mu - betaln(alpha, beta)
        )
        density = np.exp(np.clip(log_density, -700.0, 700.0))
        floor = self._exploration_floor
        mixture = (1.0 - floor) * density + floor
        log_probs = np.log(np.maximum(mixture, 1e-300))

        # d log f / d alpha and d log f / d beta for the Beta density.
        digamma_sum = digamma(alpha + beta)
        d_log_density_alpha = log_u - (digamma(alpha) - digamma_sum)
        d_log_density_beta = log_1mu - (digamma(beta) - digamma_sum)
        # The mixture only rescales them: d log q = (1-e) f d log f / q.
        share = (1.0 - floor) * density / np.maximum(mixture, 1e-300)
        d_log_alpha = share * d_log_density_alpha
        d_log_beta = share * d_log_density_beta

        # Entropy of the Beta itself. The uniform component only ever adds
        # entropy, so bonusing the Beta's own is the conservative choice and,
        # unlike the mixture's, it is closed form.
        entropy = (
            betaln(alpha, beta)
            - (alpha - 1.0) * digamma(alpha)
            - (beta - 1.0) * digamma(beta)
            + (alpha + beta - 2.0) * digamma_sum
        )
        trigamma_sum = polygamma(1, alpha + beta)
        d_entropy_alpha = -(alpha - 1.0) * polygamma(1, alpha) + (
            alpha + beta - 2.0
        ) * trigamma_sum
        d_entropy_beta = -(beta - 1.0) * polygamma(1, beta) + (
            alpha + beta - 2.0
        ) * trigamma_sum

        # Chain through alpha = 1 + softplus(z): d alpha / d z = sigmoid(z).
        slope = 1.0 / (1.0 + np.exp(-np.asarray(head, dtype=np.float64)))
        # Per dimension above, each (n, d); the action's terms are the sums over the
        # dimensions (independent Betas), the gradients back in the head's layout.
        d_log_probs = np.empty_like(slope)
        d_log_probs[:, 0::2] = d_log_alpha * slope[:, 0::2]
        d_log_probs[:, 1::2] = d_log_beta * slope[:, 1::2]
        d_entropy = np.empty_like(slope)
        d_entropy[:, 0::2] = d_entropy_alpha * slope[:, 0::2]
        d_entropy[:, 1::2] = d_entropy_beta * slope[:, 1::2]
        return log_probs.sum(axis=1), entropy.sum(axis=1), d_log_probs, d_entropy

    # -- TensorModel --------------------------------------------------------

    def predict_action(self, features: np.ndarray) -> np.ndarray:
        rows = np.asarray(features, dtype=float)
        single = rows.ndim == 1
        if single:
            rows = rows.reshape(1, -1)
        if rows.shape[1] != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} features, got {rows.shape[1]}"
            )

        actions = []
        for row in rows:
            if self._latency_floor is not None:
                self._latency_floor.observe_features(row)
            head = self._actor_head(self._contexts(row.reshape(1, -1)))
            alpha, beta = self._concentrations(head)
            positions = []
            if not self._opened:
                # The opening action (the ladder's middle rung, and on SBFT the minimum
                # wait; D10b) as the first decision.
                opening = opening_action_ms(self._action_codec.protocol)
                positions.append(self._position_of(opening[0]))
                if self._two_d:
                    positions.append(self._wait_position_of(opening[1]))
                self._opened = True
            for d in range(len(positions), self._action_width):
                if (
                    self._exploration_floor > 0.0
                    and self._rng.random() < self._exploration_floor
                ):
                    positions.append(float(self._rng.random()))
                else:
                    positions.append(float(self._rng.beta(float(alpha[0, d]), float(beta[0, d]))))
            # The latency floor bounds the timeout only.
            positions[0] = max(positions[0], self._lower_position())
            self._held_timeout_ms = self._timeout_of(positions[0])
            action = [self._action_codec.normalize_ms(self._held_timeout_ms, 0)]
            if self._two_d:
                self._held_wait_ms = self._wait_of(positions[1])
                action.append(self._action_codec.normalize_ms(self._held_wait_ms, 1))
            actions.append(action)

        result = np.asarray(actions, dtype=float)
        return result[0] if single else result

    def _store_action(self, action: "float | np.ndarray") -> "float | np.ndarray":
        """Codec-normalized action -> position in the action range (on SBFT a
        (timeout, wait) pair of positions).

        The agent reports the timeout the replicas ran with, normalized against
        the codec's bounds; the millisecond value is recovered and re-expressed
        on the policy's range, and a value outside it lands on the nearest end,
        which is what clamping already did on the way out.
        """
        values = np.asarray(action, dtype=np.float64).reshape(-1)
        if values.size != self._action_width:
            raise ValueError(f"expected {self._action_width} action values, got {values.size}")
        if np.any(values < 0.0) or np.any(values > 1.0) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"Continuous actions must be normalized to [0, 1], got {values.tolist()}"
            )
        position = self._position_of(self._action_codec.denormalize_ms(float(values[0]), 0))
        if not self._two_d:
            return position
        return np.asarray(
            [position, self._wait_position_of(self._action_codec.denormalize_ms(float(values[1]), 1))]
        )

    def reset_model(self) -> None:
        super().reset_model()
        if self._latency_floor is not None:
            self._latency_floor.reset()
        self._held_timeout_ms = self._lower_ms
        if self._two_d:
            self._held_wait_ms = self._wait_lower_ms
