"""Consensus latency as a feasibility constraint rather than as a reward.

The reward shapings in :mod:`reward_utils` that folded consensus latency into
the objective (``design_1`` through ``design_3``) were retired because latency
mostly reports the delay environment the run is currently in, not the quality
of the timeout that was chosen: a phase with 150ms of injected network delay
reads as high latency under every arm, so the learner ends up ranking phases
instead of ranking arms.

The same measurement is still worth having, one level down, as a domain
constraint on the action space. If the last windows committed with their slow
rounds taking N ms, then a timeout below N ms cannot be met by the protocol: the forward timer expires before the round can finish, the replicas
raise a view change, and throughput collapses for reasons that have nothing to
do with the fault being injected. Those actions are not a gamble worth taking -
they are known-bad before they are played. So bound the action space below by
the latency the protocol is already achieving and let the learner explore only
above it, where the arms actually differ.

The estimator is deliberately not a mean, because the measurement is
contaminated by the arm at both ends. A too-short timeout raises latency
through the view changes it causes; a too-long one raises it because a slow
leader is left in place for most of the timer. Either way a mean-based floor
ratchets upward - a bad window raises the floor, which hides the short arms,
which keeps the floor high - and the ratchet would be driven by the agent's own
last choice rather than by the environment. A low quantile over a sliding
window estimates instead "how long a round takes when it is working", which is
the quantity the constraint is about, and it comes back down as soon as enough
healthy windows accumulate. See DEFAULT_LATENCY_FLOOR_QUANTILE for the measured
reason the quantile has to be as low as it is. Windows that committed nothing
are skipped entirely: they report a latency of 0, which is an absence of
measurement, not evidence that consensus is fast.

The default field is the window's mean, avg_consensus_latency_ms - see
latency_feature_indices, which both main.py and simulator.py call without
overriding it. That matches what ContinuousThresholdTimeoutModel and
ProportionalThresholdExpert read, so the constraint and the heuristics are
compared on the same measurement.

An earlier revision of this comment claimed p95, on the argument that what a
timeout has to cover is the slow round, not the average one - a bound set at
the mean fires on about half the rounds by construction. The code never
followed it, and it mostly did not matter: the two are identical while a window
is homogeneous (measured: byte-identical floor behaviour on the continuous-delay
hardware runs, 33/107 binds and 0 oracle-arm exclusions for avg, p50 and p95
alike; on data/smartbft_wallclock at k=1.0, q=0.50 both fields bind 257/1260
with per-arm median floors 37/349/638 for avg against 44/366/644 for p95). They
diverge once a window mixes fast and slow rounds, where the mean reports the
burst diluted by its duty cycle and the p95 reports its magnitude. Which is
better is genuinely unsettled, because the tail has a detection threshold at
the duty cycle: below about 5% duty a burst does not reach the 95th percentile
at all, and there the mean is the more sensitive statistic. The fault phases
run at 2.5-10% duty, straddling that. Pass latency_field="p95_consensus_latency_ms"
to latency_feature_indices to measure the other side.

One thing this constraint is not: a way to win an arm comparison. On the
current SmartBFT arms it mostly does not bind - a 35ms round time puts the
floor far below the 300ms arm - so it is worth turning on for the continuous
models, whose action space reaches down to 100ms, and in phases where latency
genuinely rises. It removes a known-bad region of the action space; it does not
rank what is left.
"""

from collections import deque
import inspect
import math
from typing import Deque, Optional, Sequence, Tuple

import numpy as np

from models import TensorModel


# Multiplier applied to the estimated latency. 1.0 is the constraint as stated:
# do not go below what the protocol is already achieving. Above 1.0 it buys
# headroom for the jitter around that estimate.
DEFAULT_LATENCY_FLOOR_K = 1.0

# Windows kept for the quantile. Long enough that a burst of degraded windows
# cannot carry the quantile on its own, short enough to follow a regime change.
DEFAULT_LATENCY_FLOOR_WINDOW = 16

# Quantile of the window taken as the healthy round time. Measured on
# data/smartbft_wallclock, where avg_consensus_latency_ms per arm is:
#
#     arm      p10    p25   median
#     300ms     35     36       37
#     800ms     35     36      348
#    1800ms     37    347      638
#
# (p95_consensus_latency_ms has the same shape: 41/42/44, 42/43/365,
# 44/363/644, so the argument below does not depend on which field is used.)
#
# Two pressures push this in opposite directions.
#
# Downward: the p10 column above is flat across the arms and the p25 column is
# not. By the quartile the estimate has already absorbed the latency the long
# arm causes rather than measures, because a slow leader under an 1800ms timer
# holds requests for most of that timer before it is replaced. A floor built on
# a contaminated estimate can hide the 300ms arm precisely when 300ms is the
# arm committing in 44ms.
#
# Upward: under intermittent faults a large share of windows contain no burst
# at all, and a low quantile selects exactly those. Simulating the
# failure_spec_intermittent.xml burst schedule (5s burst per 20s interval, 40%
# of feature windows untouched), the floor binds 0/200 decisions at 0.10 and
# 129/200 at 0.50 - at 0.10 the constraint is not conservative, it is absent.
#
# 0.50 is where those meet. Measured against the hardware CSVs with the p95
# field - not the avg field this module actually defaults to, though on
# data/smartbft_wallclock the two bind on exactly the same decisions - it binds
# 46/107 decisions and excludes an oracle arm on 0 of them,
# while 0.75 excludes one on 21/107; 0.10 binds 33/107, also with 0 exclusions,
# but never reaches phase 3. So 0.50 is the largest value measured safe and the
# smallest that survives an intermittent fault.
#
# This is a calibration, not a resolution: the contamination that motivates the
# downward pressure is still there, and the estimator that would remove it -
# per-arm windows, floor = the minimum across arms of each arm's own low
# quantile - is not implemented. Re-check the exclusion count on any new spec
# rather than assuming this transfers.
DEFAULT_LATENCY_FLOOR_QUANTILE = 0.50


class LatencyFloor:
    """Lower bound in milliseconds, estimated from observed consensus latency.

    ``floor_ms`` is 0.0 until a window commits something, so the constraint is
    inert on a cold start and the learner keeps its full action space.
    """

    def __init__(
        self,
        multiplier: float = DEFAULT_LATENCY_FLOOR_K,
        window: int = DEFAULT_LATENCY_FLOOR_WINDOW,
        quantile: float = DEFAULT_LATENCY_FLOOR_QUANTILE,
        max_floor_ms: Optional[float] = None,
    ) -> None:
        if not math.isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError(f"multiplier must be > 0, got {multiplier}")
        if window <= 0:
            raise ValueError(f"window must be > 0, got {window}")
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"quantile must be in [0, 1], got {quantile}")
        if max_floor_ms is not None and max_floor_ms <= 0.0:
            raise ValueError(f"max_floor_ms must be > 0, got {max_floor_ms}")

        self._multiplier = float(multiplier)
        self._window = int(window)
        self._quantile = float(quantile)
        self._max_floor_ms = None if max_floor_ms is None else float(max_floor_ms)
        self.reset()

    @property
    def multiplier(self) -> float:
        return self._multiplier

    @property
    def observation_count(self) -> int:
        return len(self._latencies)

    def observe(self, latency_ms: float, alive: bool = True) -> None:
        """Record one window's consensus latency.

        A window that committed nothing (``alive=False``) or reports a
        non-positive or non-finite latency carries no measurement and is
        dropped rather than recorded as a fast round.
        """
        if not alive:
            return
        value = float(latency_ms)
        if not math.isfinite(value) or value <= 0.0:
            return
        self._latencies.append(value)

    @property
    def floor_ms(self) -> float:
        """The current lower bound, or 0.0 when nothing has been observed."""
        if not self._latencies:
            return 0.0
        healthy_ms = float(
            np.quantile(np.asarray(self._latencies, dtype=float), self._quantile)
        )
        floor_ms = self._multiplier * healthy_ms
        if self._max_floor_ms is not None:
            floor_ms = min(floor_ms, self._max_floor_ms)
        return floor_ms

    def permits(self, timeout_ms: float) -> bool:
        return float(timeout_ms) >= self.floor_ms

    def reset(self) -> None:
        self._latencies: Deque[float] = deque(maxlen=self._window)


def feasible_timeout_mask(
    timeouts_ms: Sequence[int], floor_ms: float
) -> np.ndarray:
    """Boolean mask over discrete arms, never all-False.

    A floor above every arm would leave the agent with nothing to play, so the
    largest arm always survives: the constraint is about not choosing a timeout
    that is knowably too short, and when they all are, the longest is the least
    wrong.
    """
    values = np.asarray(timeouts_ms, dtype=float)
    if values.size == 0:
        raise ValueError("timeouts_ms must not be empty")
    mask = values >= float(floor_ms)
    if not mask.any():
        mask = values >= values.max()
    return mask


class LatencyFloorPolicy(TensorModel):
    """Applies :class:`LatencyFloor` to any discrete model's choice.

    The wrapped model keeps its own exploration; this only hides the arms the
    measured latency has already ruled out. When the model asks for one of
    them, the highest-scoring feasible arm is played instead, and for the
    one-hot outputs most of the models here produce, that is the shortest arm
    at or above the floor - the nearest thing to what the model wanted.

    The learner is not told it was overridden, and does not need to be: the
    online agent trains on the timeout the replicas actually ran with
    (``main._apply_reward_payload_locked`` encodes it back out of the report),
    so the update lands on the arm that was played.
    """

    def __init__(
        self,
        model: TensorModel,
        timeouts_ms: Sequence[int],
        latency_index: int,
        n_features: int,
        instances_index: Optional[int] = None,
        latency_floor: Optional[LatencyFloor] = None,
    ) -> None:
        if n_features <= 0:
            raise ValueError(f"n_features must be > 0, got {n_features}")
        if not 0 <= int(latency_index) < n_features:
            raise ValueError(
                f"latency_index must be in [0, {n_features}), got {latency_index}"
            )
        if instances_index is not None and not 0 <= int(instances_index) < n_features:
            raise ValueError(
                f"instances_index must be in [0, {n_features}), got {instances_index}"
            )
        timeouts = tuple(int(value) for value in timeouts_ms)
        if not timeouts:
            raise ValueError("timeouts_ms must not be empty")

        self._model = model
        self._timeouts = timeouts
        self._latency_index = int(latency_index)
        self._instances_index = (
            None if instances_index is None else int(instances_index)
        )
        self._n_features = int(n_features)
        self._latency_floor = (
            LatencyFloor() if latency_floor is None else latency_floor
        )
        self._takes_backward = (
            "backward" in inspect.signature(model.update_model).parameters
        )
        self._override_count = 0

    @property
    def model(self) -> TensorModel:
        return self._model

    @property
    def latency_floor(self) -> LatencyFloor:
        return self._latency_floor

    @property
    def override_count(self) -> int:
        """Decisions where the floor moved the model off its preferred arm."""
        return self._override_count

    def _observe(self, row: np.ndarray) -> None:
        alive = True
        if self._instances_index is not None:
            alive = float(row[self._instances_index]) > 0.0
        self._latency_floor.observe(float(row[self._latency_index]), alive=alive)

    def _preferred_index(self, scores: np.ndarray) -> int:
        if scores.size == 1:
            index = int(round(float(scores.reshape(-1)[0])))
            return min(len(self._timeouts) - 1, max(0, index))
        return int(np.argmax(scores))

    def _shortest_feasible(self, mask: np.ndarray) -> int:
        feasible = np.flatnonzero(mask)
        return int(feasible[np.argmin([self._timeouts[i] for i in feasible])])

    def _constrain_row(self, scores: np.ndarray, mask: np.ndarray) -> int:
        """Index of the best feasible arm for one row of model output."""
        preferred = self._preferred_index(scores)
        if mask[preferred]:
            return preferred
        if scores.size == 1:
            return self._shortest_feasible(mask)

        masked = np.where(mask, scores, -np.inf)
        tied = np.flatnonzero(masked == float(np.max(masked)))
        # A one-hot row leaves every feasible arm tied at 0 once its own arm is
        # masked out, so break the tie toward the shortest feasible timeout:
        # the nearest thing to the shorter one the model asked for.
        return int(tied[np.argmin([self._timeouts[i] for i in tied])])

    def predict_action(self, features: np.ndarray) -> np.ndarray:
        rows = np.asarray(features, dtype=float)
        single = rows.ndim == 1
        if single:
            rows = rows.reshape(1, -1)
        if rows.shape[1] != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} features, got {rows.shape[1]}"
            )

        for row in rows:
            self._observe(row)

        predicted = np.asarray(self._model.predict_action(features), dtype=float)
        batch = predicted.reshape(len(rows), -1)
        mask = feasible_timeout_mask(self._timeouts, self._latency_floor.floor_ms)

        constrained = np.zeros((len(rows), len(self._timeouts)), dtype=float)
        for index, scores in enumerate(batch):
            selected = self._constrain_row(scores, mask)
            if self._preferred_index(scores) != selected:
                self._override_count += 1
            constrained[index, selected] = 1.0

        return constrained[0] if single else constrained

    def update_model(
        self,
        features: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | float,
        backward: bool = True,
    ) -> float:
        """Pass the update straight through to the wrapped model.

        The floor constrains what is played, not what is learned from. Models
        without a ``backward`` parameter take one row at a time, so a batch is
        replayed in order for them, as simulator.update_model_batch does.
        """
        if self._takes_backward:
            return self._model.update_model(features, actions, rewards, backward)

        features_np = np.asarray(features)
        actions_np = np.asarray(actions)
        rewards_np = np.asarray(rewards, dtype=float)
        rows = int(features_np.shape[0]) if features_np.ndim > 1 else 1
        total = 0.0
        for index in range(rows):
            row_features = features_np[index] if features_np.ndim > 1 else features_np
            row_action = actions_np[index] if actions_np.ndim > 1 else actions_np
            row_reward = rewards_np[index] if rewards_np.ndim > 0 else rewards_np
            total += float(
                self._model.update_model(row_features, row_action, float(row_reward))
            )
        return total / max(rows, 1)

    def reset_model(self) -> None:
        self._latency_floor.reset()
        self._override_count = 0
        self._model.reset_model()


def latency_feature_indices(
    feature_names: Sequence[str],
    latency_field: str = "avg_consensus_latency_ms",
) -> Tuple[int, Optional[int]]:
    """(latency index, liveness index) for a protocol's feature vector."""
    names = list(feature_names)
    latency_index = names.index(latency_field)
    instances_index = (
        names.index("total_consensus_instances")
        if "total_consensus_instances" in names
        else None
    )
    return latency_index, instances_index
