"""The windows a prior was trained under, recorded in the prior and checked on load.

Kevin (2026-10-09): every prior records the window split it was trained
under, and loading one whose windows differ from the run's, or that records
none, is an error. The agent is told three lengths (main.py
--feature-window-ms, --reward-window-ms and --episode-ms; the reply wait and
the warm-up only through the episode), so those three are recorded and
checked. A simulator's come from its FEATURE_S, REPLY_S, WARMUP_S and
REWARD_S.

Every prior is a flat .npz (each simulator's GP prior from its train command,
model_builder.save_model_prior's LinUCB and PPO checkpoints), so the record is
the same three entries in each: ``windows.feature_ms``, ``windows.reward_ms``
and ``windows.episode_ms``.
"""

from dataclasses import dataclass
import os
from typing import Dict

import numpy as np

_PREFIX = "windows."


def _ms(seconds: float) -> int:
    ms = round(float(seconds) * 1000.0)
    if abs(ms - float(seconds) * 1000.0) > 1e-6:
        raise ValueError(f"window {seconds} s is not a whole number of ms")
    return int(ms)


@dataclass(frozen=True)
class PriorWindows:
    """Feature window, reward window and episode period, in ms."""

    feature_ms: int
    reward_ms: int
    episode_ms: int

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}")

    @classmethod
    def from_seconds(
        cls, feature_s: float, reply_s: float, warmup_s: float, reward_s: float
    ) -> "PriorWindows":
        """A simulator's windows (its FEATURE_S, REPLY_S, WARMUP_S, REWARD_S);
        the episode is their sum, as on every engine."""
        return cls(
            feature_ms=_ms(feature_s),
            reward_ms=_ms(reward_s),
            episode_ms=_ms(feature_s + reply_s + warmup_s + reward_s),
        )

    def as_dict(self) -> Dict[str, int]:
        return {
            "feature_ms": self.feature_ms,
            "reward_ms": self.reward_ms,
            "episode_ms": self.episode_ms,
        }

    def arrays(self) -> Dict[str, np.ndarray]:
        """The record, as entries to add to a prior's .npz."""
        return {_PREFIX + key: np.asarray(value) for key, value in self.as_dict().items()}


def check_prior_windows(path: str, expected: PriorWindows) -> None:
    """Refuse the .npz at ``path`` unless it records exactly ``expected``."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prior {path} does not exist")
    with np.load(path, allow_pickle=False) as data:
        stored = {
            key[len(_PREFIX):]: int(data[key]) for key in data.files if key.startswith(_PREFIX)
        }
    if not stored:
        raise ValueError(
            f"Prior {path} records no windows; retrain it under the run's windows "
            f"{expected.as_dict()}"
        )
    if stored != expected.as_dict():
        raise ValueError(
            f"Prior {path} was trained under windows {stored}, the current setup has "
            f"{expected.as_dict()}"
        )
