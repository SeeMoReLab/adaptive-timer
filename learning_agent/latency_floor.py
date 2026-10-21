from collections import deque
import inspect
import math
from typing import Deque, Optional, Sequence, Tuple, Union

import numpy as np

from models import TensorModel


# Multiplier applied to the estimated latency. 1.0 is the constraint as stated:
# do not go below what the protocol is already achieving. Above 1.0 it buys
# headroom for the jitter around that estimate.
DEFAULT_LATENCY_FLOOR_MULTIPLIER = 1.0

# Windows kept for the quantile (pretrain.py and the simulators; the live agent takes
# continuous_heuristic's window, main.py). At 1 the floor is the
# latest committing window's latency and the quantile has no effect. A longer
# window (it was 16 until 2026-10-07) keeps a burst of degraded windows from
# carrying the quantile on its own, at the cost of following a regime change
# more slowly.
DEFAULT_LATENCY_FLOOR_WINDOW = 1

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
#
# Those measurements were taken with a 16-window floor. At the 1-window default
# the quantile has no effect; it starts to matter with a longer window
# (pretrain.py --latency-floor-window), where 0.50 is the value measured above.
# The live agent takes continuous_heuristic's window and quantile instead (main.py).
DEFAULT_LATENCY_FLOOR_QUANTILE = 0.50


class LatencyFloor:
    """Lower bound in milliseconds, estimated from observed consensus latency.

    ``floor_ms`` is 0.0 until a window commits something, so the constraint is
    inert on a cold start and the learner keeps its full action space.
    """

    def __init__(
        self,
        multiplier: float = DEFAULT_LATENCY_FLOOR_MULTIPLIER,
        window: int = DEFAULT_LATENCY_FLOOR_WINDOW,
        quantile: float = DEFAULT_LATENCY_FLOOR_QUANTILE,
        max_floor_ms: Optional[float] = None,
        field: str = "avg_consensus_latency_ms",
        offset_ms: float = 0.0,
    ) -> None:
        if not math.isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError(f"multiplier must be > 0, got {multiplier}")
        if window <= 0:
            raise ValueError(f"window must be > 0, got {window}")
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"quantile must be in [0, 1], got {quantile}")
        if max_floor_ms is not None and max_floor_ms <= 0.0:
            raise ValueError(f"max_floor_ms must be > 0, got {max_floor_ms}")
        if not math.isfinite(offset_ms) or offset_ms < 0.0:
            raise ValueError(f"offset_ms must be >= 0, got {offset_ms}")

        self._multiplier = float(multiplier)
        self._window = int(window)
        self._quantile = float(quantile)
        self._max_floor_ms = None if max_floor_ms is None else float(max_floor_ms)
        # A constant part of the time the timer covers that the signal does not carry
        # (continuous_bandit.FLOOR_SPECS: PBFT/SBFT's proposal interval).
        self._offset_ms = float(offset_ms)
        self.field = field  # the report field observe() is fed from (model_builder)
        self.reset()

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
        floor_ms = self._multiplier * healthy_ms + self._offset_ms
        if self._max_floor_ms is not None:
            floor_ms = min(floor_ms, self._max_floor_ms)
        return floor_ms

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
        latency_index: Union[int, Sequence[int]],
        n_features: int,
        instances_index: Optional[int] = None,
        latency_floor: Optional[LatencyFloor] = None,
    ) -> None:
        if n_features <= 0:
            raise ValueError(f"n_features must be > 0, got {n_features}")
        latency_index = latency_indices(latency_index, n_features)
        if instances_index is not None and not 0 <= int(instances_index) < n_features:
            raise ValueError(
                f"instances_index must be in [0, {n_features}), got {instances_index}"
            )
        timeouts = tuple(int(value) for value in timeouts_ms)
        if not timeouts:
            raise ValueError("timeouts_ms must not be empty")

        self._model = model
        self._timeouts = timeouts
        self._latency_index = latency_index
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

    @property
    def model(self) -> TensorModel:
        return self._model

    def _observe(self, row: np.ndarray) -> None:
        alive = True
        if self._instances_index is not None:
            alive = float(row[self._instances_index]) > 0.0
        self._latency_floor.observe(sum(float(row[i]) for i in self._latency_index), alive=alive)

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
        self._model.reset_model()


def signal_fields(signal: str) -> Tuple[str, ...]:
    """The report fields of a latency signal: one field name, or several joined by
    "+" and summed per window (SBFT's three phase latencies, 2026-10-14)."""
    fields = tuple(name.strip() for name in signal.split("+"))
    if not signal or any(not name for name in fields):
        raise ValueError(f"signal must be report field names joined by '+', got {signal!r}")
    return fields


def latency_indices(latency_index: Union[int, Sequence[int]], n_features: int) -> Tuple[int, ...]:
    """The feature indices a floor sums for its latency, each checked against n_features."""
    indices = (int(latency_index),) if isinstance(latency_index, int) else tuple(int(i) for i in latency_index)
    if not indices or any(not 0 <= i < int(n_features) for i in indices):
        raise ValueError(f"latency_index must be in [0, {n_features}), got {latency_index}")
    return indices


def latency_feature_indices(
    feature_names: Sequence[str],
    latency_field: str = "avg_consensus_latency_ms",
) -> Tuple[Tuple[int, ...], Optional[int]]:
    """(latency indices, liveness index) for a protocol's feature vector; the
    latency is the sum of the fields of ``latency_field`` (see signal_fields)."""
    names = list(feature_names)
    latency_index = tuple(names.index(name) for name in signal_fields(latency_field))
    instances_index = (
        names.index("total_consensus_instances")
        if "total_consensus_instances" in names
        else None
    )
    return latency_index, instances_index
