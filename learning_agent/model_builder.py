"""The agent's model, built from its settings in one place.

main.serve and the simulators both call :func:`build_model`, so a simulator
trains and evaluates exactly the model the agent runs: same hyperparameters,
same seed, same codec, same latency-floor wiring.

Priors. ``continuous_gp`` and ``discrete_gp`` load a simulator-trained GP
prior (NumpyMLP .npz, validated by SimPriorGPModel); the two share one .npz. The LinUCB and PPO models load a checkpoint
written by :func:`save_model_prior`: the LinUCB models a prior mean theta0
(``prior_theta``), the PPO models their actor and critic weights. A
checkpoint carries the setup it was trained under, and loading refuses one
whose model type, protocol, reward design, features, action space or
hyperparameters differ from the current ones. Every prior also records the
windows it was trained under, and build_model refuses one whose windows differ
from the run's or that records none (prior_windows.check_prior_windows).
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import agent_pb2
from continuous_bandit import (
    SBFT_WAIT_MIN_MS,
    opening_action_ms,
    ContinuousHeuristicTimeoutModel,
    ContinuousLinUCB,
    ContinuousPPO,
    ContinuousTimeoutActionCodec,
    NormalizedLatencyFloor,
    SimReturnGPModel,
    context_spec,
    heuristic_spec,
)
from latency_floor import LatencyFloor, LatencyFloorPolicy, latency_feature_indices
from prior_windows import PriorWindows, check_prior_windows
from models import (
    SBFT_SLOW_PATH_VALUES_MS,
    ActionCodec,
    BlockedTreeBandit,
    DiscountedLinUCB,
    DiscretePPO,
    LinearModel,
    PavlovTimeoutModel,
    ProtocolTimeoutActionCodec,
    RandomTimeoutModel,
    ScheduledTimeoutActionCodec,
    ScheduledTimeoutModel,
    TensorModel,
    ThompsonSamplingModel,
    action_index_to_timeout,
    get_action_space_size,
    get_arm_timeout_values_ms,
    get_report_feature_dim,
    get_report_feature_field_names,
    get_timeout_values_ms,
)
from reward_utils import get_reward_design, reward_normalizer_tps


# Model types that take --prior-path. Any other model type given a prior is
# an error rather than a prior silently ignored.
PRETRAINABLE_MODEL_TYPES = (
    "continuous_gp",
    "discrete_gp",
    "discrete_linucb",
    "continuous_linucb",
    "discrete_ppo",
    "continuous_ppo",
)

# Model types whose codec is built with the model; every other one but
# scheduled uses ProtocolTimeoutActionCodec over the discrete arms.
# discrete_gp plays only the arms but emits them through the continuous codec.
_CONTINUOUS_CODEC_MODEL_TYPES = (
    "continuous_gp",
    "discrete_gp",
    "continuous_linucb",
    "continuous_heuristic",
    "continuous_ppo",
)

# Class each checkpoint-loading model type builds, before any floor wrapper.
_CHECKPOINT_MODEL_CLASSES = {
    "discrete_linucb": DiscountedLinUCB,
    "continuous_linucb": ContinuousLinUCB,
    "discrete_ppo": DiscretePPO,
    "continuous_ppo": ContinuousPPO,
}

_META_PREFIX = "meta."
_ARRAY_PREFIX = "array."
_PROVENANCE_PREFIX = "provenance."


@dataclass(frozen=True)
class BuiltModel:
    """The model, the codec its actions go through, and the startup lines
    describing it (main prints them)."""

    model: TensorModel
    action_codec: ActionCodec
    description: List[str]


def opening_arm_index(protocol: int) -> int:
    """The discrete arm of the opening action (continuous_bandit.opening_action_ms: the
    ladder's middle rung, and on SBFT the minimum wait), where the discrete LinUCB and PPO
    models open (Kevin, 2026-10-19, restart D10b)."""
    opening = tuple(opening_action_ms(protocol))
    matches = []
    for index in range(get_action_space_size(protocol)):
        ((_, timeout),) = action_index_to_timeout(protocol, index).ListFields()
        values = tuple(int(v) for _, v in timeout.ListFields())
        if values[:len(opening)] == opening:
            matches.append(index)
    if len(matches) != 1:
        raise ValueError(f"opening action {opening} matches arms {matches} on {protocol}")
    return matches[0]


def prior_metadata(model_type: str, protocol: int, model: TensorModel) -> Dict[str, object]:
    """What a LinUCB or PPO checkpoint must agree on with ``model``.

    ``model`` is the unwrapped model (not a LatencyFloorPolicy). ``timeouts_ms``
    is what defines the action space: the arm values for the discrete models,
    the basis centres for continuous_linucb (with the range their positions
    are measured in), the clamp range for continuous_ppo.
    """
    expected_class = _CHECKPOINT_MODEL_CLASSES.get(model_type)
    if expected_class is None:
        raise ValueError(
            f"--model-type={model_type} has no checkpoint prior; those that do "
            f"are {', '.join(sorted(_CHECKPOINT_MODEL_CLASSES))}"
        )
    if type(model) is not expected_class:
        raise ValueError(
            f"--model-type={model_type} builds {expected_class.__name__}, "
            f"got {type(model).__name__}"
        )
    metadata: Dict[str, object] = {
        "model_type": model_type,
        "protocol": agent_pb2.Protocol.Name(int(protocol)),
        "reward_design": get_reward_design(),
        # N of reward_tps_latency; priors fit before 2026-10-12 lack it (tps / 10,000) and are refused.
        "reward_normalizer_tps": reward_normalizer_tps(
            agent_pb2.Protocol.Name(int(protocol))[len("PROTOCOL_"):].lower()
        ),
        "feature_names": tuple(get_report_feature_field_names(protocol)),
    }
    if model_type == "discrete_linucb":
        metadata["timeouts_ms"] = tuple(get_arm_timeout_values_ms(protocol))
        metadata["discount_factor"] = model._discount_factor
        metadata["ridge"] = model._ridge
        metadata["reward_scale"] = model._reward_scale
    elif model_type == "continuous_linucb":
        metadata["timeouts_ms"] = tuple(model._centers_ms)
        metadata["action_range_ms"] = (model._lower_ms, model._upper_ms)
        metadata["discount_factor"] = model._discount_factor
        metadata["ridge"] = model._ridge
        metadata["reward_scale"] = model._reward_scale
        metadata["center_rewards"] = model._center_rewards
    elif model_type == "discrete_ppo":
        metadata["timeouts_ms"] = tuple(get_arm_timeout_values_ms(protocol))
    else:
        metadata["timeouts_ms"] = (model._lower_ms, model._upper_ms)
    if model_type in ("continuous_linucb", "continuous_ppo") and model._two_d:
        # SBFT: the wait is the model's second dimension (2026-10-19).
        metadata["wait_range_ms"] = (model._wait_lower_ms, model._wait_upper_ms)
        if model_type == "continuous_linucb":
            metadata["wait_centers_ms"] = tuple(model._wait_centers_ms)
    return metadata


def save_model_prior(
    path: str,
    model_type: str,
    protocol: int,
    model: TensorModel,
    arrays: Dict[str, np.ndarray],
    *,
    windows: PriorWindows,
    **provenance: object,
) -> None:
    """Write a LinUCB or PPO prior for :func:`build_model` to load.

    ``model`` is a model built for the setup the prior is for (its metadata is
    taken from it, see :func:`prior_metadata`). ``arrays`` is ``{"theta":
    model.fit_prior_theta(...)}`` for the LinUCB models and
    ``model.state_arrays()`` for the PPO models. ``windows`` are the
    simulator's windows the prior was trained under. ``provenance`` (runs,
    seed, simulator, ...) is stored and never validated.
    """
    if not str(path).endswith(".npz"):
        raise ValueError(f"Prior path {path} must end in .npz")
    if not arrays:
        raise ValueError("arrays must not be empty")
    payload: Dict[str, np.ndarray] = {}
    for key, value in prior_metadata(model_type, protocol, model).items():
        payload[_META_PREFIX + key] = np.asarray(value)
    for key, value in arrays.items():
        payload[_ARRAY_PREFIX + key] = np.asarray(value, dtype=np.float64)
    for key, value in provenance.items():
        stored = np.asarray(value)
        if stored.dtype == object:
            raise ValueError(
                f"provenance {key} must be a string, number or array of them, "
                f"got {type(value).__name__}"
            )
        payload[_PROVENANCE_PREFIX + key] = stored
    payload.update(windows.arrays())
    np.savez(path, **payload)


def load_model_prior(
    path: str, model_type: str, protocol: int, model: TensorModel
) -> Dict[str, np.ndarray]:
    """The arrays of a :func:`save_model_prior` file, after checking every
    metadata field against :func:`prior_metadata` of ``model`` (including the
    current reward design)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prior {path} does not exist")
    expected = prior_metadata(model_type, protocol, model)
    with np.load(path, allow_pickle=False) as data:
        contents = {key: data[key] for key in data.files}
    stored = {
        key[len(_META_PREFIX):]: value
        for key, value in contents.items()
        if key.startswith(_META_PREFIX)
    }
    # Field by field first, model_type leading, so a prior of another model
    # type is named as that rather than as a different set of fields.
    for key, value in expected.items():
        if key not in stored:
            continue
        wanted = np.asarray(value)
        found = stored[key]
        if wanted.shape != found.shape or not np.array_equal(wanted, found):
            raise ValueError(
                f"Prior {path} has {key}={found.tolist()!r}, the current setup "
                f"has {key}={wanted.tolist()!r}"
            )
    missing = sorted(set(expected) - set(stored))
    extra = sorted(set(stored) - set(expected))
    if missing or extra:
        raise ValueError(
            f"Prior {path} metadata does not match: missing {missing}, "
            f"unexpected {extra}"
        )
    arrays = {
        key[len(_ARRAY_PREFIX):]: value
        for key, value in contents.items()
        if key.startswith(_ARRAY_PREFIX)
    }
    if not arrays:
        raise ValueError(f"Prior {path} holds no arrays")
    return arrays


def _prior_theta(arrays: Dict[str, np.ndarray], path: str) -> np.ndarray:
    if set(arrays) != {"theta"}:
        raise ValueError(
            f"Prior {path} for a LinUCB model must hold exactly 'theta', "
            f"got {sorted(arrays)}"
        )
    return arrays["theta"]


def build_model(
    protocol: int,
    model_type: str,
    *,
    prior_path: Optional[str] = None,
    prior_windows: Optional[PriorWindows] = None,
    latency_floor: Optional[LatencyFloor] = None,
    schedule_file: Optional[Path] = None,
    start_monotonic: Optional[float] = None,
    heuristic_multiplier: Optional[float] = None,
    heuristic_window: Optional[int] = None,
    heuristic_quantile: Optional[float] = None,
    linucb_discount: Optional[float] = None,
    linucb_exploration: Optional[float] = None,
    ppo_exploration: Optional[float] = None,
) -> BuiltModel:
    """The model main.serve runs for ``model_type`` on ``protocol``.

    ``prior_path`` None is an untrained model; a path is loaded, and only by
    PRETRAINABLE_MODEL_TYPES, after checking that it was trained under
    ``prior_windows``, the run's windows (required with a prior).
    ``latency_floor`` None disables the floor. The
    GP's BLAS thread limit is process-global and left to the caller.
    """
    if prior_path is not None:
        if not str(prior_path).strip():
            raise ValueError("prior_path is empty; pass None for no prior")
        if model_type not in PRETRAINABLE_MODEL_TYPES:
            raise ValueError(
                f"--prior-path is not supported with --model-type={model_type}; "
                f"it applies to {', '.join(PRETRAINABLE_MODEL_TYPES)}"
            )
        if prior_windows is None:
            raise ValueError("a prior needs prior_windows, the windows of the run it is loaded into")
        check_prior_windows(prior_path, prior_windows)

    # Domain constraint on the action space, taken by every model except the
    # scheduled oracle: a timeout under the part of the consensus latency it
    # must cover cannot be met, so it is not worth a decision. The floor reads
    # its own field (LatencyFloor.field; main.py builds it with make_latency_floor, on
    # FLOOR_SPECS' hold-free signal). Discrete models get it as a wrapper that hides the arms under it;
    # continuous models are handed it and search above it.
    latency_index, instances_index = latency_feature_indices(
        get_report_feature_field_names(protocol),
        latency_floor.field if latency_floor is not None else "avg_consensus_latency_ms",
    )
    # The LinUCB discount and random-exploration share; None keeps each class's
    # default (discount 0.95 and no random exploration, both models; study of 2026-10-19).
    if (linucb_discount is not None or linucb_exploration is not None) and model_type not in (
        "discrete_linucb", "continuous_linucb"
    ):
        raise ValueError(f"linucb_discount/linucb_exploration apply to the LinUCB models, not {model_type}")
    linucb_overrides = {k: float(v) for k, v in (("discount_factor", linucb_discount),
                                                  ("exploration_floor", linucb_exploration)) if v is not None}
    # PPO's uniform share mixed into its policy; None keeps the class default (0, since 2026-10-19).
    if ppo_exploration is not None and model_type not in ("discrete_ppo", "continuous_ppo"):
        raise ValueError(f"ppo_exploration applies to the PPO models, not {model_type}")
    ppo_overrides = {} if ppo_exploration is None else {"exploration_floor": float(ppo_exploration)}
    if latency_floor is not None and model_type == "scheduled":
        raise ValueError(
            "--model-type=scheduled replays its schedule unchanged and takes no "
            "latency floor; pass --latency-floor-multiplier 0"
        )

    def normalized_floor(action_codec: ActionCodec) -> Optional[NormalizedLatencyFloor]:
        if latency_floor is None:
            return None
        return NormalizedLatencyFloor(
            action_codec=action_codec,
            latency_index=latency_index,
            n_features=get_report_feature_dim(protocol),
            instances_index=instances_index,
            latency_floor=latency_floor,
        )

    # Every continuous model acts on the protocol's ContextSpec range (100-3000 ms; on SBFT
    # also the wait, 10-500 ms, which continuous_linucb and continuous_ppo choose as their
    # second dimension since 2026-10-19; it was T / 8 until then).
    def continuous_codec() -> ContinuousTimeoutActionCodec:
        spec = context_spec(protocol)
        return ContinuousTimeoutActionCodec(
            protocol, bounds_ms=spec.bounds_ms, wait_bounds_ms=spec.wait_bounds_ms
        )

    def checkpoint(model: TensorModel) -> Dict[str, np.ndarray]:
        return load_model_prior(prior_path, model_type, protocol, model)

    description: List[str] = []
    action_codec: Optional[ActionCodec] = None
    if model_type == "continuous_heuristic":
        # Per-protocol rule (HEURISTIC_SPECS); on SBFT it also sets the wait W.
        hspec = heuristic_spec(protocol)
        if protocol == agent_pb2.PROTOCOL_SBFT:
            action_codec = ContinuousTimeoutActionCodec(
                protocol, bounds_ms=hspec.bounds_ms,
                wait_bounds_ms=context_spec(protocol).wait_bounds_ms,
            )
        else:
            action_codec = ContinuousTimeoutActionCodec(protocol, bounds_ms=hspec.bounds_ms)
        initial = opening_action_ms(protocol)
        model = ContinuousHeuristicTimeoutModel(
            protocol=protocol,
            action_codec=action_codec,
            feature_names=get_report_feature_field_names(protocol),
            initial_timeout_ms=initial,
            multiplier=heuristic_multiplier,
            window=heuristic_window,
            quantile=heuristic_quantile,
            latency_floor=normalized_floor(action_codec),
            # A window that committed nothing takes the feature window's length (2026-10-19).
            empty_window_ms=(None if prior_windows is None else prior_windows.feature_ms),
        )
        k = hspec.multiplier if heuristic_multiplier is None else heuristic_multiplier
        description.append(
            f"[continuous_heuristic] {k:g} x {hspec.signal} over recent committing "
            f"windows, clamped to {hspec.bounds_ms[0]}..{hspec.bounds_ms[1]}ms"
            + (", storm doubling" if hspec.storm_view_changes is not None else "")
            + (", SBFT wait from the n-th prepare gap" if protocol == agent_pb2.PROTOCOL_SBFT else "")
        )
    elif model_type in ("continuous_gp", "discrete_gp"):
        # Refuses a protocol without a context spec; wait_bounds_ms is SBFT's
        # fast-path wait (None elsewhere). No prior path is a zero prior.
        # SimPriorGPModel (H18) stays in the code, unwired.
        # discrete_gp is the same model scoring only the ladder rungs (on SBFT
        # the T-major product with the waits, the arm order), with the same prior.
        spec = context_spec(protocol)
        action_codec = ContinuousTimeoutActionCodec(
            protocol, bounds_ms=spec.bounds_ms, wait_bounds_ms=spec.wait_bounds_ms
        )
        axes_ms = None
        if model_type == "discrete_gp":
            axes_ms = (tuple(get_timeout_values_ms(protocol)),)
            if protocol == agent_pb2.PROTOCOL_SBFT:
                axes_ms = axes_ms + (tuple(SBFT_SLOW_PATH_VALUES_MS),)
        model = SimReturnGPModel(
            action_codec=action_codec,
            field_names=get_report_feature_field_names(protocol),
            prior_path=prior_path,
            reward_design=get_reward_design(),
            latency_floor=normalized_floor(action_codec),
            axes_ms=axes_ms,
        )
        prior_text = "untrained (zero prior)" if prior_path is None else prior_path
        if model_type == "continuous_gp":
            description.append(f"[continuous_gp] prior: {prior_text}")
        else:
            # Grid row i must emit exactly discrete arm i, or the model is not
            # playing the ladder the other discrete models play.
            n_arms = get_action_space_size(protocol)
            if len(model._grid) != n_arms:
                raise RuntimeError(
                    f"discrete_gp grid has {len(model._grid)} rows, expected {n_arms} arms"
                )
            for index in range(n_arms):
                emitted = action_codec.decode(model._grid[index])
                expected = action_index_to_timeout(protocol, index)
                if emitted != expected:
                    raise RuntimeError(
                        f"discrete_gp grid row {index} decodes to {emitted}, "
                        f"arm {index} is {expected}"
                    )
            rungs_text = " x ".join(str(axis) for axis in axes_ms)
            description.append(f"[discrete_gp] rungs {rungs_text}ms, prior: {prior_text}")
    elif model_type == "discrete_random":
        # Seeded like the other models so that, in sharing mode, every node
        # draws the same action from the same agreed features. Reward
        # aggregation is only meaningful when the fleet explores in lockstep.
        model = RandomTimeoutModel(
            action_dim=get_action_space_size(protocol),
            seed=42,
        )
    elif model_type == "discrete_linear":
        model = LinearModel(
            action_dim=get_action_space_size(protocol),
            n_features=get_report_feature_dim(protocol),
            lr=0.05,
        )
    elif model_type == "discrete_pavlov":
        # Win-stay/lose-shift baseline: no context and no learning, so it takes
        # neither n_features nor a seed. protocol only fixes the arm ordering.
        model = PavlovTimeoutModel(
            action_dim=get_action_space_size(protocol),
            protocol=protocol,
        )
    elif model_type == "discrete_tree_ts":
        model = ThompsonSamplingModel(
            action_dim=get_action_space_size(protocol),
            n_features=get_report_feature_dim(protocol),
            seed=42,
        )
    elif model_type == "discrete_tree_blocks":
        model = BlockedTreeBandit(
            action_dim=get_action_space_size(protocol),
            n_features=get_report_feature_dim(protocol),
            seed=42,
        )
    elif model_type == "discrete_linucb":

        def discrete_linucb(prior_theta: Optional[np.ndarray]) -> DiscountedLinUCB:
            return DiscountedLinUCB(
                action_dim=get_action_space_size(protocol),
                n_features=get_report_feature_dim(protocol),
                seed=42,
                prior_theta=prior_theta,
                opening_arm=opening_arm_index(protocol),
                **linucb_overrides,
            )

        model = discrete_linucb(None)
        if prior_path is not None:
            model = discrete_linucb(_prior_theta(checkpoint(model), prior_path))
            description.append(f"[discrete_linucb] prior mean: {prior_path}")
    elif model_type == "discrete_ppo":
        model = DiscretePPO(
            action_dim=get_action_space_size(protocol),
            n_features=get_report_feature_dim(protocol),
            seed=42,
            opening_arm=opening_arm_index(protocol),
            **ppo_overrides,
        )
        if prior_path is not None:
            model.load_state_arrays(checkpoint(model))
        description.append(
            "[discrete_ppo] softmax policy over "
            f"{tuple(sorted(get_timeout_values_ms(protocol)))}, "
            "linear actor + value baseline, on-policy from "
            + ("scratch" if prior_path is None else f"the prior {prior_path}")
        )
    elif model_type == "continuous_linucb":
        action_codec = continuous_codec()

        def continuous_linucb(prior_theta: Optional[np.ndarray]) -> ContinuousLinUCB:
            return ContinuousLinUCB(
                action_codec=action_codec,
                n_features=get_report_feature_dim(protocol),
                seed=42,
                latency_floor=normalized_floor(action_codec),
                prior_theta=prior_theta,
                **linucb_overrides,
            )

        model = continuous_linucb(None)
        if prior_path is not None:
            model = continuous_linucb(_prior_theta(checkpoint(model), prior_path))
        description.append(
            "[continuous_linucb] discounted LinUCB over "
            f"{model._lower_ms}..{model._upper_ms}ms, basis centres {model._centers_ms}"
            + (f", wait basis {model._wait_centers_ms} ms" if model._two_d else "")
        )
        if prior_path is not None:
            description.append(f"[continuous_linucb] prior mean: {prior_path}")
    elif model_type == "continuous_ppo":
        action_codec = continuous_codec()
        model = ContinuousPPO(
            action_codec=action_codec,
            n_features=get_report_feature_dim(protocol),
            seed=42,
            latency_floor=normalized_floor(action_codec),
            **ppo_overrides,
        )
        if prior_path is not None:
            model.load_state_arrays(checkpoint(model))
        description.append(
            "[continuous_ppo] Beta policy over "
            f"{model._lower_ms}..{model._upper_ms}ms"
            + (f" and a second over the wait {model._wait_lower_ms}..{model._wait_upper_ms}ms" if model._two_d else "")
            + ", linear actor + value baseline, "
            "on-policy from "
            + ("scratch" if prior_path is None else f"the prior {prior_path}")
        )
    elif model_type == "scheduled":
        if schedule_file is None:
            raise ValueError("--schedule-file is required when --model-type=scheduled")
        model = ScheduledTimeoutModel.from_csv(
            protocol,
            schedule_file,
            start_monotonic=start_monotonic,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Discrete models get the floor as a wrapper that hides the arms under it;
    # the continuous models and discrete_gp were handed it above, and
    # scheduled refused it.
    # A prior was loaded into the inner model above, before this wrap.
    #
    # In sharing mode the floor stays identical on every replica for free: it
    # is estimated from the consensus-agreed feature vector, which is the same
    # on all of them, so constraining the action space does not break the
    # lockstep the aggregated reward depends on.
    if (
        latency_floor is not None
        and model_type.startswith("discrete_")
        and model_type not in _CONTINUOUS_CODEC_MODEL_TYPES
    ):
        model = LatencyFloorPolicy(
            model=model,
            # Per arm, not the distinct ladder: SBFT has 6 arms over a 3-rung
            # ladder, and the shorter vector left arms 3-5 - both 1100 rungs -
            # unreachable.
            timeouts_ms=get_arm_timeout_values_ms(protocol),
            latency_index=latency_index,
            n_features=get_report_feature_dim(protocol),
            instances_index=instances_index,
            latency_floor=latency_floor,
        )

    if model_type == "scheduled":
        action_codec = ScheduledTimeoutActionCodec(protocol)
    elif model_type not in _CONTINUOUS_CODEC_MODEL_TYPES:
        action_codec = ProtocolTimeoutActionCodec(protocol)
    if action_codec is None:
        raise RuntimeError(f"no action codec was built for {model_type}")
    return BuiltModel(model=model, action_codec=action_codec, description=description)
