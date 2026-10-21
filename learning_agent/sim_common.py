"""The parts of a randomized training run every simulator draws the same way.

Kevin, 2026-10-19 (restart D11): one pretraining recipe for all six protocols. Each
simulator keeps its protocol's own fault kinds (stragglers, vote withholding, relative
holds, the silent proposer) and its fitted physics; what they share is drawn here:

- the offered load, as a factor of the protocol's rate N (reward_utils), 0.5-1.5;
- the first fault at 40-400 s and the gaps between faults, 40-400 s;
- an extra link delay, log-uniform 10-250 ms (the real specs use +50 and +150);
- a leader's proposal hold, log-uniform 50-1500 ms (the real specs use 300-600);
- the base one-way delay, 2-10 ms (5 in every real spec);
- no netem jitter: every real spec sets jitterRatio 0 (2026-10-19);
- the initial timeout: a quarter of the runs start at the protocol's opening action
  (the ladder's middle rung, where every model and engine starts), the rest anywhere in
  the action range, log-uniform;
- an initial backlog in 30% of the runs where a simulator models one;
- every simulator constant not fitted on the blind cells is drawn 0.5-2x around a fitted
  or cited value (``around``).

The values are design choices at the union of the per-simulator ranges they replace, not
fits; DESIGN.md section 6 lists them.
"""

import math
import os
from typing import List, Tuple

import numpy as np

LOAD_FACTOR_RANGE: Tuple[float, float] = (0.5, 1.5)
FIRST_FAULT_S: Tuple[float, float] = (40.0, 400.0)
FAULT_GAP_S: Tuple[float, float] = (40.0, 400.0)
LINK_EXTRA_MS: Tuple[float, float] = (10.0, 250.0)
HOLD_MS: Tuple[float, float] = (50.0, 1500.0)
BASE_DELAY_MS: Tuple[float, float] = (2.0, 10.0)
JITTER_RATIO = 0.0
OPENING_START_SHARE = 0.25
BACKLOG_SHARE = 0.30
CONSTANT_FACTOR: Tuple[float, float] = (0.5, 2.0)


def log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    if not 0.0 < lo <= hi:
        raise ValueError(f"log_uniform needs 0 < lo <= hi, got {lo}, {hi}")
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def around(rng: np.random.Generator, value: float, lo: float = CONSTANT_FACTOR[0],
           hi: float = CONSTANT_FACTOR[1]) -> float:
    """A constant scaled by a log-uniform factor in [lo, hi] (default 0.5-2)."""
    if value <= 0:
        raise ValueError(f"cannot draw around a non-positive value {value}")
    return float(value * log_uniform(rng, lo, hi))


def draw_load_tps(rng: np.random.Generator, rate_tps: float) -> float:
    return float(rate_tps * rng.uniform(*LOAD_FACTOR_RANGE))


def draw_base_delay_ms(rng: np.random.Generator) -> float:
    return float(rng.uniform(*BASE_DELAY_MS))


def draw_link_extra_ms(rng: np.random.Generator) -> float:
    return log_uniform(rng, *LINK_EXTRA_MS)


def draw_hold_ms(rng: np.random.Generator) -> float:
    return log_uniform(rng, *HOLD_MS)


def fault_start_times(rng: np.random.Generator, total_s: float) -> List[float]:
    """Onsets of the faults of a run of total_s seconds after the base phase at 0."""
    times: List[float] = []
    t = float(rng.uniform(*FIRST_FAULT_S))
    while t < total_s:
        times.append(t)
        t += float(rng.uniform(*FAULT_GAP_S))
    return times


def draw_initial_action_ms(rng: np.random.Generator, protocol: int) -> Tuple[float, ...]:
    """The action a training run starts with (module docstring): the whole opening
    action in OPENING_START_SHARE of the runs, otherwise each dimension log-uniform
    over its range (the timeout, and on SBFT the fast-path wait)."""
    from continuous_bandit import context_spec, opening_action_ms

    if rng.random() < OPENING_START_SHARE:
        return tuple(float(v) for v in opening_action_ms(protocol))
    spec = context_spec(protocol)
    bounds = [spec.bounds_ms] + ([spec.wait_bounds_ms] if spec.wait_bounds_ms else [])
    return tuple(log_uniform(rng, float(lo), float(hi)) for lo, hi in bounds)


def draw_initial_timeout_ms(rng: np.random.Generator, protocol: int) -> float:
    """draw_initial_action_ms on a one-timeout protocol."""
    action = draw_initial_action_ms(rng, protocol)
    if len(action) != 1:
        raise ValueError(f"protocol {protocol} has {len(action)} action dimensions; use draw_initial_action_ms")
    return action[0]


def draw_initial_backlog(rng: np.random.Generator, max_pending: int) -> int:
    if rng.random() < BACKLOG_SHARE:
        return int(rng.integers(0, max_pending + 1))
    return 0


def schedule_rows(protocol_name: str) -> List[Tuple[float, Tuple[float, ...]]]:
    """The rows of learning_agent/config/schedule_<protocol>.csv, the scheduled oracle,
    as (time_s, action values), checked as ScheduledTimeoutModel requires them: in
    order, the leading row negative (the warm-up row), every other row a multiple of
    the 20 s episode period."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config",
                        f"schedule_{protocol_name}.csv")
    rows: List[Tuple[float, Tuple[float, ...]]] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values = [float(v) for v in line.split(",")]
            rows.append((values[0], tuple(values[1:])))
    if not rows or rows[0][0] >= 0:
        raise ValueError(f"{path}: the first row must be the negative warm-up row")
    times = [t for t, _ in rows]
    if times != sorted(times) or any(t % 20 for t in times[1:]):
        raise ValueError(f"{path}: rows must be in order and multiples of 20 s after the warm-up row")
    return rows


def scheduled_action(rows: List[Tuple[float, Tuple[float, ...]]], apply_s: float,
                     warmup_s: float) -> Tuple[float, ...]:
    """The oracle's action for an episode applied at sim time apply_s, as
    ScheduledTimeoutModel picks it: the last row with time_s <= apply_s - warm-up."""
    rel = apply_s - warmup_s
    chosen = [action for at, action in rows if at <= rel + 1e-6]
    if not chosen:
        raise ValueError(f"no schedule row at {rel:.1f} s after the warm-up")
    return chosen[-1]


__all__ = [
    "LOAD_FACTOR_RANGE", "FIRST_FAULT_S", "FAULT_GAP_S", "LINK_EXTRA_MS", "HOLD_MS",
    "BASE_DELAY_MS", "JITTER_RATIO", "OPENING_START_SHARE", "BACKLOG_SHARE", "CONSTANT_FACTOR",
    "log_uniform", "around", "draw_load_tps", "draw_base_delay_ms", "draw_link_extra_ms",
    "draw_hold_ms", "fault_start_times", "draw_initial_action_ms", "draw_initial_timeout_ms",
    "draw_initial_backlog",
    "schedule_rows", "scheduled_action",
]
