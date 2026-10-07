"""The agent's model, built from its settings in one place.

main.serve and the simulators both call :func:`build_model`, so a simulator
trains and evaluates exactly the model the agent runs: same hyperparameters,
same seed, same codec, same latency-floor wiring.

Priors. ``continuous_gp`` loads a simulator-trained GP prior (NumpyMLP .npz,
validated by SimPriorGPModel). The LinUCB and PPO models load a checkpoint
written by :func:`save_model_prior`: the LinUCB models a prior mean theta0
(``prior_theta``), the PPO models their actor and critic weights. A
checkpoint carries the setup it was trained under, and loading refuses one
whose model type, protocol, reward design, features, action space or
hyperparameters differ from the current ones.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import agent_pb2
from continuous_bandit import (
    DEFAULT_HEURISTIC_QUANTILE,
    DEFAULT_HEURISTIC_WINDOW,
    DEFAULT_PROPORTIONAL_K,
    DEFAULT_WAIT_DIVISOR,
    ContinuousHeuristicTimeoutModel,
    ContinuousLinUCB,
    ContinuousPPO,
    ContinuousTimeoutActionCodec,
    DerivedWaitActionCodec,
    NormalizedLatencyFloor,
    SimReturnGPModel,
    context_spec,
)
from latency_floor import LatencyFloor, LatencyFloorPolicy, latency_feature_indices
from models import (
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
    get_action_space_size,
    get_arm_timeout_values_ms,
    get_report_feature_dim,
    get_report_feature_field_names,
    get_timeout_values_ms,
)
from reward_utils import get_reward_design


# Model types that take --prior-path. Any other model type given a prior is
# an error rather than a prior silently ignored.
PRETRAINABLE_MODEL_TYPES = (
    "continuous_gp",
    "discrete_linucb",
    "continuous_linucb",
    "discrete_ppo",
    "continuous_ppo",
)

# Model types whose codec is built with the model; every other one but
# scheduled uses ProtocolTimeoutActionCodec over the discrete arms.
_CONTINUOUS_CODEC_MODEL_TYPES = (
    "continuous_gp",
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
    if model_type in ("continuous_linucb", "continuous_ppo") and isinstance(
        model._action_codec, DerivedWaitActionCodec
    ):
        # SBFT: the wait played beside each T is T / divisor.
        metadata["wait_divisor"] = model._action_codec.divisor
    return metadata


def save_model_prior(
    path: str,
    model_type: str,
    protocol: int,
    model: TensorModel,
    arrays: Dict[str, np.ndarray],
    **provenance: object,
) -> None:
    """Write a LinUCB or PPO prior for :func:`build_model` to load.

    ``model`` is a model built for the setup the prior is for (its metadata is
    taken from it, see :func:`prior_metadata`). ``arrays`` is ``{"theta":
    model.fit_prior_theta(...)}`` for the LinUCB models and
    ``model.state_arrays()`` for the PPO models. ``provenance`` (runs, seed,
    simulator, ...) is stored and never validated.
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
    latency_floor: Optional[LatencyFloor] = None,
    schedule_file: Optional[Path] = None,
    start_monotonic: Optional[float] = None,
    heuristic_multiplier: float = DEFAULT_PROPORTIONAL_K,
    heuristic_window: int = DEFAULT_HEURISTIC_WINDOW,
    heuristic_quantile: float = DEFAULT_HEURISTIC_QUANTILE,
    wait_divisor: float = DEFAULT_WAIT_DIVISOR,
) -> BuiltModel:
    """The model main.serve runs for ``model_type`` on ``protocol``.

    ``prior_path`` None is an untrained model; a path is loaded, and only by
    PRETRAINABLE_MODEL_TYPES. ``latency_floor`` None disables the floor. The
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

    # Domain constraint on the action space, taken by every model except the
    # scheduled oracle: a timeout under the consensus latency the replicas are
    # already achieving cannot be met, so it is not worth a decision. The
    # field is the window mean, avg_consensus_latency_ms, the same one
    # continuous_heuristic reads, so the rule and the constraint are on one
    # measurement. Discrete models get it as a wrapper that hides the arms
    # under it; continuous models are handed it and search above it.
    latency_index, instances_index = latency_feature_indices(
        get_report_feature_field_names(protocol)
    )
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

    # SBFT's action is two-dimensional. The models below each place one
    # timeout on a line and refuse a 2-D codec, so on SBFT they get the
    # election timeout and the fast-path wait is derived from it,
    # W = T / --wait-divisor; everywhere else this is the ordinary codec.
    # continuous_gp searches both dimensions and does not use this.
    def one_dimensional_codec() -> ActionCodec:
        if protocol != agent_pb2.PROTOCOL_SBFT:
            return ContinuousTimeoutActionCodec(protocol)
        sbft_spec = context_spec(protocol)
        return DerivedWaitActionCodec(
            ContinuousTimeoutActionCodec(
                protocol,
                bounds_ms=sbft_spec.bounds_ms,
                wait_bounds_ms=sbft_spec.wait_bounds_ms,
            ),
            divisor=wait_divisor,
        )

    def checkpoint(model: TensorModel) -> Dict[str, np.ndarray]:
        return load_model_prior(prior_path, model_type, protocol, model)

    description: List[str] = []
    action_codec: Optional[ActionCodec] = None
    if model_type == "continuous_heuristic":
        action_codec = one_dimensional_codec()
        arms_ms = get_timeout_values_ms(protocol)
        model = ContinuousHeuristicTimeoutModel(
            action_codec=action_codec,
            n_features=get_report_feature_dim(protocol),
            latency_index=latency_index,
            instances_index=instances_index,
            multiplier=heuristic_multiplier,
            window=heuristic_window,
            quantile=heuristic_quantile,
            clamp_ms=(min(arms_ms), max(arms_ms)),
            initial_timeout_ms=sorted(arms_ms)[len(arms_ms) // 2],
            latency_floor=normalized_floor(action_codec),
        )
        description.append(
            f"[continuous_heuristic] {heuristic_multiplier:g} x the "
            f"{heuristic_quantile:g} quantile of the last {heuristic_window} "
            f"committing windows' latency, clamped to {min(arms_ms)}..{max(arms_ms)}ms"
        )
    elif model_type == "continuous_gp":
        # Refuses a protocol without a context spec; wait_bounds_ms is SBFT's
        # fast-path wait (None elsewhere). No prior path is a zero prior.
        # SimPriorGPModel (H18) stays in the code, unwired.
        spec = context_spec(protocol)
        action_codec = ContinuousTimeoutActionCodec(
            protocol, bounds_ms=spec.bounds_ms, wait_bounds_ms=spec.wait_bounds_ms
        )
        model = SimReturnGPModel(
            action_codec=action_codec,
            field_names=get_report_feature_field_names(protocol),
            prior_path=prior_path,
            reward_design=get_reward_design(),
            latency_floor=normalized_floor(action_codec),
        )
        description.append(
            "[continuous_gp] prior: "
            + ("untrained (zero prior)" if prior_path is None else prior_path)
        )
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
        action_codec = one_dimensional_codec()
        arms_ms = get_timeout_values_ms(protocol)

        def continuous_linucb(prior_theta: Optional[np.ndarray]) -> ContinuousLinUCB:
            return ContinuousLinUCB(
                action_codec=action_codec,
                n_features=get_report_feature_dim(protocol),
                seed=42,
                latency_floor=normalized_floor(action_codec),
                prior_theta=prior_theta,
            )

        model = continuous_linucb(None)
        if prior_path is not None:
            model = continuous_linucb(_prior_theta(checkpoint(model), prior_path))
        description.append(
            "[continuous_linucb] discounted LinUCB over "
            f"{min(arms_ms)}..{max(arms_ms)}ms, basis centred on the arms "
            f"{tuple(sorted(arms_ms))}"
        )
        if prior_path is not None:
            description.append(f"[continuous_linucb] prior mean: {prior_path}")
    elif model_type == "continuous_ppo":
        action_codec = one_dimensional_codec()
        arms_ms = get_timeout_values_ms(protocol)
        model = ContinuousPPO(
            action_codec=action_codec,
            n_features=get_report_feature_dim(protocol),
            seed=42,
            latency_floor=normalized_floor(action_codec),
        )
        if prior_path is not None:
            model.load_state_arrays(checkpoint(model))
        description.append(
            "[continuous_ppo] Beta policy over "
            f"{min(arms_ms)}..{max(arms_ms)}ms, linear actor + value baseline, "
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
    # the continuous models were handed it above, and scheduled refused it.
    # A prior was loaded into the inner model above, before this wrap.
    #
    # In sharing mode the floor stays identical on every replica for free: it
    # is estimated from the consensus-agreed feature vector, which is the same
    # on all of them, so constraining the action space does not break the
    # lockstep the aggregated reward depends on.
    if latency_floor is not None and model_type.startswith("discrete_"):
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
