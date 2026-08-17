import argparse
import csv
import inspect
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from tqdm import tqdm
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import time

import agent_pb2
from data_loader import (
    CsvTrainingData, 
    DatasetLoader, 
    _extract_action_index, 
    _extract_features, 
    _extract_label, 
    _is_empty_row,
    )
from reward_utils import calculate_reward_from_report_row

from models import (
    CMAB,
    LinearModel,
    MultiRF,
    NeighborhoodRobustBandit,
    PavlovTimeoutModel,
    QuadraticRF,
    SingleRF,
    ThompsonSamplingModel,
    ThompsonSamplingModelV1,
    ThompsonSamplingModelV2,
    ThompsonSamplingModelV3,
    TensorModel,
    QuadraticRewardScaler,
    RewardTransformer,
    get_action_space_size,
    get_report_feature_dim,
    get_report_feature_field_names,
    get_timeout_values_ms,
    timeout_index_to_ms_for_protocol,
)

from generator import DataGenerator

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

PROTOCOL_NAME_TO_ENUM = {
    "pbft": agent_pb2.PROTOCOL_PBFT,
    "sbft": agent_pb2.PROTOCOL_SBFT,
    "tendermint": agent_pb2.PROTOCOL_TENDERMINT,
}

MODEL_NAME_TO_TENSORMODEL = {
    "linear": LinearModel,
    # Contextual multi-armed bandits, oldest to newest. "cmab" is the current
    # model; v1 through v3 are its predecessors, kept so a run can be compared
    # against the iteration it came from.
    "cmab": ThompsonSamplingModel,
    "cmab_v1": ThompsonSamplingModelV1,
    "cmab_v2": ThompsonSamplingModelV2,
    "cmab_v3": ThompsonSamplingModelV3,
    # Blocked CMAB with replay-triggered resets. Not an iteration of "cmab" -
    # a separate line, boosted-tree based, so it needs xgboost installed.
    "cmab_reset": CMAB,
    # Contextual bandit with local commitment and a region-level prior
    # for ties that reward evidence cannot resolve.
    "claude_v1": NeighborhoodRobustBandit,
    # Win-stay/lose-shift heuristic. Context-free and non-learning, so it is a
    # floor to beat rather than an iteration of the bandits above.
    "pavlov": PavlovTimeoutModel,
    # BFTBrain random-forest baselines, named after its --num-models values.
    "quadratic_rf": QuadraticRF,
    "multi_rf": MultiRF,
    "single_rf": SingleRF,
}

# Models whose constructor takes a seed, so a run is reproducible from --seed
# alone rather than from the global RNG state. ThompsonSamplingModelV1 is
# absent because it has no seed parameter - it draws from the global numpy RNG.
SEEDED_MODELS = (
    ThompsonSamplingModel,
    ThompsonSamplingModelV2,
    ThompsonSamplingModelV3,
    CMAB,
    QuadraticRF,
    MultiRF,
    SingleRF,
    NeighborhoodRobustBandit,
)

# Heuristics with no contextual model, so no n_features and no seed. The
# protocol is passed instead: it fixes the arm ordering they step through.
CONTEXT_FREE_MODELS = (PavlovTimeoutModel,)

DEFAULT_PBFT_PHASES = "0;1;2;3;4;5;0;2;1;5;4;2;1"

def set_seed(seed: int) -> None:
    """Seed the global RNGs.

    Only affects code that reaches for module-level randomness - the loader's
    shuffle, and any model that has not been given an explicit ``seed=``.
    Models constructed from SEEDED_MODELS carry their own Generator and are
    unaffected by this.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def _parse_protocol(value: str) -> int:
    protocol = PROTOCOL_NAME_TO_ENUM.get(value.strip().lower().replace("-", "_"))
    if protocol is None:
        valid = ", ".join(sorted({"pbft", "sbft", "tendermint"}))
        raise argparse.ArgumentTypeError(
            f"Unsupported protocol '{value}'. Valid options: {valid}"
        )
    return protocol

def _parse_model(value: str) -> TensorModel:
    model = MODEL_NAME_TO_TENSORMODEL.get(value.strip().lower())
    if model is None:
        valid = ", ".join(sorted(MODEL_NAME_TO_TENSORMODEL))
        raise argparse.ArgumentTypeError(
            f"Unsupported model '{value}'. Valid options: {valid}"
        )
    return model

def _takes_batch_update(model: TensorModel) -> bool:
    """True for models whose update_model consumes a whole batch.

    Two signatures live in models.py. The learned models take
    (features, actions, rewards, backward=) and vectorise internally. The
    heuristics follow TensorModel's declared signature (features, action,
    reward) and are inherently sequential -- PavlovTimeoutModel compares each
    reward against the previous one and steps one arm on a loss -- so their
    batches must be replayed a row at a time, in order. Detected by signature
    rather than by class so a new model needs no entry here.
    """
    return "backward" in inspect.signature(model.update_model).parameters


def update_model_batch(
    model: TensorModel,
    features: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    backward: bool = True,
) -> float:
    if _takes_batch_update(model):
        return model.update_model(features, actions, rewards, backward=backward)

    # Sequential replay. The heuristics are updated even on the eval pass: their
    # policy *is* the reward comparison, so skipping it would pin them to one arm.
    features = np.asarray(features)
    actions = np.asarray(actions)
    rewards = np.asarray(rewards)
    rows = int(features.shape[0]) if features.ndim > 1 else 1
    total = 0.0
    for i in range(rows):
        row_features = features[i] if features.ndim > 1 else features
        row_action = actions[i] if actions.ndim > 0 else actions
        row_reward = rewards[i] if rewards.ndim > 0 else rewards
        total += float(model.update_model(row_features, row_action, float(row_reward)))
    return total / max(rows, 1)


def predict_logits(
    model: TensorModel, features: np.ndarray, action_dim: int
) -> np.ndarray:
    """Always return (batch, action_dim) so argmax(axis=1) is valid.

    Learned models emit per-arm scores already. The heuristics emit a 1-D array
    of chosen arm indices, which argmax(axis=1) cannot consume, so promote it to
    one-hot -- argmax then recovers exactly the arm the heuristic picked.
    """
    out = np.asarray(model.predict_action(features))
    if out.ndim == 2:
        return out
    index = np.clip(np.rint(out.reshape(-1)).astype(int), 0, action_dim - 1)
    return np.eye(action_dim, dtype=float)[index]


def _parse_acceptable(value: str) -> Dict[int, set]:
    """Parse ``0:*;2:300,800;3:800,1800`` into {phase: {timeout_ms, ...}}.

    A phase can have more than one timeout that is genuinely correct - either
    because the alternatives are statistically indistinguishable in reward, or
    because the workload tolerates both. Scoring against a single argmax marks
    those choices wrong and understates any model that picks them. Phases left
    out of the spec keep the single-best-timeout behaviour; ``*`` accepts any
    timeout, for phases where the choice does not matter.
    """
    acceptable: Dict[int, set] = {}
    for clause in value.split(";"):
        clause = clause.strip()
        if not clause:
            continue
        if ":" not in clause:
            raise argparse.ArgumentTypeError(
                f"Expected 'phase:timeout[,timeout...]', got {clause!r}"
            )
        phase_text, timeouts_text = clause.split(":", 1)
        try:
            phase = int(phase_text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Bad phase in {clause!r}")
        if timeouts_text.strip() == "*":
            acceptable[phase] = None  # filled in once the protocol is known
            continue
        try:
            acceptable[phase] = {int(t) for t in timeouts_text.split(",") if t.strip()}
        except ValueError:
            raise argparse.ArgumentTypeError(f"Bad timeout list in {clause!r}")
        if not acceptable[phase]:
            raise argparse.ArgumentTypeError(f"No timeouts given in {clause!r}")
    return acceptable


def _resolve_acceptable(
    acceptable: Optional[Dict[int, set]], protocol: int
) -> Optional[Dict[int, set]]:
    """Expand ``*`` and reject timeouts the protocol does not offer."""
    if acceptable is None:
        return None
    valid = set(get_timeout_values_ms(protocol))
    resolved = {}
    for phase, timeouts in acceptable.items():
        if timeouts is None:
            resolved[phase] = set(valid)
            continue
        unknown = timeouts - valid
        if unknown:
            raise argparse.ArgumentTypeError(
                f"Phase {phase}: timeouts {sorted(unknown)} are not valid for this "
                f"protocol. Valid: {sorted(valid)}"
            )
        resolved[phase] = timeouts
    return resolved


def _parse_data(
    reader: csv.DictReader,
    protocol: int,
    feature_field_names: Sequence[str],
    timeout_values_ms: Sequence[int],
    phase_options: Dict[int, int],
    reward_transform: Optional[Callable[[float], float]] = None,
    acceptable: Optional[Dict[int, set]] = None,
) -> CsvTrainingData:
    reward_transform = reward_transform or (lambda value: value)

    features: List[List[float]] = []
    actions: List[float] = []
    rewards: List[float] = []
    labels: List[float] = []
    accepted: List[List[float]] = []

    if reader.fieldnames is None:
        raise ValueError(f"Missing a header row.")

    for row_number, row in enumerate(reader, start=2):
        if _is_empty_row(row):
            continue

        feature_values = _extract_features(
            protocol=protocol,
            row=row,
            feature_field_names=feature_field_names,
            row_number=row_number,
        )
        
        action_index = _extract_action_index(
            protocol=protocol,
            row=row,
            timeout_values_ms=timeout_values_ms,
            row_number=row_number,
        )
        simulated_reward = row.get("simulated_reward")
        if simulated_reward is not None and simulated_reward.strip() != "":
            reward_value = float(simulated_reward)
        else:
            reward_value = calculate_reward_from_report_row(
                protocol, row, row_number=row_number
            )

        label_index = _extract_label(
            row=row, 
            phase_options=phase_options,
            row_number=row_number
        )

        features.append(feature_values)
        actions.append(float(action_index))
        rewards.append(float(reward_transform(reward_value)))
        labels.append(float(label_index))

        if acceptable is not None:
            phase_value = int(float(row["phase"]))
            allowed = acceptable.get(phase_value)
            row_accept = [0.0] * len(timeout_values_ms)
            if allowed is None:
                row_accept[int(label_index)] = 1.0
            else:
                for timeout_ms in allowed:
                    row_accept[timeout_values_ms.index(timeout_ms)] = 1.0
            accepted.append(row_accept)

    if not features:
        raise ValueError("No completed episode rows found.")

    return CsvTrainingData(
        features=features,
        actions=actions,
        rewards=rewards,
        labels=labels,
        accepted=accepted or None,
    )

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End to End Learning Agent Simulation"
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default='pbft',
        help="Protocol: pbft | sbft | tendermint",
    )
    parser.add_argument(
        "--model",
        type=_parse_model,
        default='linear',
        # Listed from the registry so a new model never has to be named twice.
        help="Model: " + " | ".join(sorted(MODEL_NAME_TO_TENSORMODEL)),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Observation sample directory. Defaults to "
            "learning_agent/data/<protocol>."
        ),
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=Path(__file__).resolve().parent / "plot",
        help="Plot Image Directory path.",
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        default= 'experiment',
        help="Name of experiment instance",
    )

    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1000,
        help="Number of episodes per phase to generate"
    )
    
    parser.add_argument(
        "--phases",
        type=str,
        default=None,
        help=(
            "Phase sequence. PBFT retains its established default; other "
            "protocols default to every non-warmup phase in the dataset."
        ),
    )

    parser.add_argument(
        "--acceptable-timeouts",
        type=_parse_acceptable,
        default=None,
        help=(
            "Phases where more than one timeout counts as correct, e.g. "
            "'0:*;2:300,800;3:800,1800;4:800,1800;8:*'. Phases not listed are "
            "scored against the single best timeout, which is the default "
            "behaviour."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Rows handed to the model per predict_action/update_model call. "
            "Defaults to 1 because that is what production does: each node "
            "runs its own agent and featurize() returns a single report "
            "(sharing mode aggregates to one report first). A batch here only "
            "ever groups one episode's per-node rows, so anything above 1 lets "
            "a model see several nodes' contexts in a single call and pool or "
            "coordinate across them - which inflates simulator results by "
            "exploiting something the real agent never has."
        ),
    )
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reward-scaler",
        choices=("raw", "quadratic"),
        default="raw",
        help="Optional additional scaling for the reward derived from the reward report.",
    )

    return parser

def generate_data(
    generator: DataGenerator,
    protocol: str,
    phase: int,
    current_start_episode: int,
    scaler: RewardTransformer,
    batch_size: int,
    phase_options: Dict[int, int],
    num_episode: int = 1,
    prev_timeout: List[int] = None,
    acceptable: Optional[Dict[int, set]] = None,
) -> Tuple[DatasetLoader, int]:
    phase_data, new_start_episode = generator.generate_local_report(
        phase_index=int(phase),
        current_start_episode=current_start_episode,
        num_episode=num_episode,
        prev_timeout=prev_timeout,
    )
    parsed_data = _parse_data(
        reader=phase_data,
        protocol=PROTOCOL_NAME_TO_ENUM[protocol],
        feature_field_names=get_report_feature_field_names(
            PROTOCOL_NAME_TO_ENUM[protocol]
        ),
        timeout_values_ms=get_timeout_values_ms(PROTOCOL_NAME_TO_ENUM[protocol]),
        phase_options=phase_options,
        reward_transform=scaler,
        acceptable=acceptable,
    )

    loader = DatasetLoader(
        features=parsed_data.features,
        actions=parsed_data.actions,
        rewards=parsed_data.rewards,
        labels=parsed_data.labels,
        batch_size=batch_size,
        shuffle=True,
        accepted=parsed_data.accepted,
    )

    return loader, new_start_episode

def train_model(
    protocol: int,
    model: TensorModel,
    train_loader: DatasetLoader,
    eval_loader: DatasetLoader,
    epochs: int,
    save_path: Optional[Path] = None,
    early_stop: Optional[int] = 20,
) -> Tuple[TensorModel, Dict]:
    
    best_eval_metrics = float("-inf")
    count = 0

    stats = {'best_epoch':0, 'best_accuracy':0.0, 'best_loss':0.0}
    for e in range(epochs):
        train_loss = _run_epoch(model, train_loader, backward=True)
        metrics,_ = evaluate_model(
            protocol=protocol,
            model=model,
            eval_loader= eval_loader,
        )
        if metrics['accuracy'] > best_eval_metrics:
            best_eval_metrics = metrics['accuracy']

            stats['best_epoch'] = e
            stats['best_accuracy'] = metrics['accuracy']
            stats['best_loss'] = metrics['weighted_loss']
            count = 0
        else:
            count += 1
            if count == early_stop:
                break

    return model, stats

def _run_epoch(model: TensorModel, loader: DatasetLoader, backward: bool) -> float:
    loss_sum = 0.0
    batch_count = 0

    for batch in loader:
        loss_sum += update_model_batch(
            model,
            batch["features"],
            batch["actions"],
            batch["rewards"],
            backward=backward,
        )
        batch_count += 1

    if batch_count == 0:
        return 0.0
    
    return loss_sum / batch_count

def evaluate_model(
    protocol: int,
    model: TensorModel, eval_loader: DatasetLoader
) -> Tuple[Dict[str, float], List[int]]:

    total = 0
    correct = 0
    weighted_loss_sum = 0.0
    # Accumulated across batches, not overwritten by the last one. This list is
    # handed to the generator as prev_timeout, so dropping earlier batches
    # silently discards the decisions those rows made - visible whenever
    # batch_size is below the node count.
    curr_timeout: List[int] = []

    for batch in eval_loader:
        weighted_loss_sum += update_model_batch(
            model, batch["features"], batch["actions"], batch["rewards"], backward=False
        )
        logits = predict_logits(
            model, batch["features"], get_action_space_size(protocol)
        )
        predicted = np.argmax(logits, axis=1)
        expected = np.rint(batch["labels"]).astype(int)
        total += int(expected.size)
        accepted = batch.get("accepted")
        if accepted is None:
            correct += int((predicted == expected).sum())
        else:
            # A phase can have more than one timeout that is genuinely right;
            # scoring against a single argmax marks those choices wrong.
            correct += int(accepted[np.arange(len(predicted)), predicted].sum())
        curr_timeout.extend(
            timeout_index_to_ms_for_protocol(protocol, idx)
            for idx in predicted.tolist()
        )

    accuracy = (correct / total) if total else 0.0

    return {
        "test samples": float(total),
        "accuracy": float(accuracy),
        "weighted_loss": float(weighted_loss_sum / max(len(eval_loader), 1)),

    }, curr_timeout

def plot_accuracy(result: Dict[Tuple[int, str], float], save_path: str = 'accuracy_plot.pdf') -> None:
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 12,            
        'axes.labelsize': 14,         
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'axes.linewidth': 1.0,       
        'lines.linewidth': 2.0,       
        'lines.markersize': 7,        
        'figure.figsize': (10.0, 6.0), 
        'axes.spines.top': False,     
        'axes.spines.right': False
    })
    
    fig, ax = plt.subplots()
    
    sorted_items = sorted(result.items(), key=lambda item: item[0][0])
    
    phases = [str(k[1]) for k, v in sorted_items]
    accuracies = [v for k, v in sorted_items]

    x_coords = list(range(len(accuracies)))
    single_color = '#1f77b4'

    ax.plot(x_coords, accuracies, marker='o', linestyle='-', 
            color=single_color, label='Accuracy', alpha=0.8)
    
    x_ticks = []
    x_labels = []
    
    for i in range(len(accuracies)):
        x_ticks.append(i)
        x_labels.append(phases[i])

            
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel('Phase')
    ax.set_ylabel('Accuracy')

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)

    ax.grid(True, linestyle=':', alpha=0.6, axis='y')
    ax.legend(frameon=False, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_stage_accuracies(data_dict: Dict[Tuple[int, str], List[float]], save_path: str = 'stage_accuracies.png') -> None:
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 12,              
        'axes.labelsize': 14,         
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'axes.linewidth': 1.0,        
        'lines.linewidth': 1,       
        'lines.markersize': 3,    
        'figure.figsize': (12.0, 6.0), 
        'axes.spines.top': False,     
        'axes.spines.right': False
    })

    fig, ax = plt.subplots()
    
    current_x = 0
    x_ticks = []
    x_labels = []
    
    single_color = '#1f77b4' 
    
    for i, (key, accuracies) in enumerate(data_dict.items()):
        stage_idx, phase_id = key
        n_points = len(accuracies)
        
        x_coords = range(current_x, current_x + n_points)
        
        label = 'Accuracy' if i == 0 else None
        
        ax.plot(x_coords, accuracies, marker='o', linestyle='-', 
                color=single_color, 
                label=label,
                alpha=0.5,
                linewidth=0.5) 
        
        middle_x = current_x + (n_points / 2) - 0.5
        x_ticks.append(middle_x)
        x_labels.append(f'{phase_id}')
        
        current_x += n_points
        if i < len(data_dict) - 1:
            ax.axvline(x=current_x - 0.5, color='gray', linestyle='--', alpha=1.0)

    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel('Phase')
    ax.set_ylabel('Accuracy')
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    
    ax.grid(True, linestyle=':', alpha=0.6, axis='y')
    ax.legend(frameon=False, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_prediction_phases(data_dict: Dict[Tuple[int, str], List[float]], save_path: str = 'prediction_phases.png', y_ticks: List[float] = None) -> None:
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 12,              
        'axes.labelsize': 14,         
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'axes.linewidth': 1.0,        
        'lines.linewidth': 1,       
        'lines.markersize': 3,    
        'figure.figsize': (12.0, 6.0), 
        'axes.spines.top': False,     
        'axes.spines.right': False
    })

    fig, ax = plt.subplots()
    
    current_x = 0
    x_ticks = []
    x_labels = []
    all_predictions = []
    
    single_color = '#1f77b4' 
    
    sorted_keys = sorted(data_dict.items(), key=lambda item: item[0][0])
    
    for i, (key, predictions) in enumerate(sorted_keys):
        stage_idx, phase_id = key
        n_points = len(predictions)
        all_predictions.extend(predictions)
        
        x_coords = range(current_x, current_x + n_points)
        
        label = 'Predicted Timeout' if i == 0 else None
        
        ax.plot(x_coords, predictions, marker='o', linestyle='', 
                color=single_color, 
                label=label,
                alpha=0.5) 
        
        middle_x = current_x + (n_points / 2) - 0.5
        x_ticks.append(middle_x)
        x_labels.append(f'{phase_id}')
        
        current_x += n_points
        
        if i < len(sorted_keys) - 1:
            ax.axvline(x=current_x - 0.5, color='gray', linestyle='--', alpha=1.0)

    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    else:
        unique_preds = np.unique(all_predictions)
        ax.set_yticks(unique_preds)
        y_min, y_max = min(unique_preds), max(unique_preds)
        padding = (y_max - y_min) * 0.1 if y_max != y_min else 10
        ax.set_ylim(y_min - padding, y_max + padding)

    ax.set_xlabel('Phase')
    ax.set_ylabel('Predicted Value')
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    
    ax.grid(True, linestyle=':', alpha=0.6, axis='y')
    ax.legend(frameon=False, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main() -> None:
    args = _build_parser().parse_args()
    set_seed(args.seed)
    protocol_name = args.protocol.strip().lower().replace("-", "_")
    protocol_id = _parse_protocol(protocol_name)
    data_dir = args.data_dir or Path(__file__).resolve().parent / "data" / protocol_name

    # 1. Initialization
    generator = DataGenerator(
        protocol=protocol_name,
        directory_path=data_dir,
    )
    if args.phases is not None:
        phase_spec = args.phases
    elif protocol_id == agent_pb2.PROTOCOL_PBFT:
        phase_spec = DEFAULT_PBFT_PHASES
    else:
        phase_spec = ";".join(
            str(phase) for phase in sorted(generator.phase_options) if phase >= 0
        )
    print(
        f"Initiating Experiment [Model:] {args.model} "
        f"[Protocol:] {protocol_name} [Phases:] {phase_spec}"
    )
    if args.batch_size > 1:
        print(
            f"WARNING: --batch-size {args.batch_size} hands the model several "
            "nodes' reports in one call. Production calls predict_action with a "
            "single report, so results above 1 may not transfer."
        )
    scaler = QuadraticRewardScaler.transform if args.reward_scaler == "quadratic" else None
    acceptable = _resolve_acceptable(args.acceptable_timeouts, protocol_id)
    if acceptable:
        print(
            "Scoring these phases against multiple acceptable timeouts: "
            + ", ".join(
                f"{phase}->{sorted(timeouts)}" for phase, timeouts in sorted(acceptable.items())
            )
        )

    if args.model in CONTEXT_FREE_MODELS:
        model_kwargs = {
            "action_dim": get_action_space_size(protocol_id),
            "protocol": protocol_id,
        }
    else:
        model_kwargs = {
            "action_dim": get_action_space_size(protocol_id),
            "n_features": get_report_feature_dim(protocol_id),
        }
    if args.model in SEEDED_MODELS:
        model_kwargs["seed"] = args.seed
    model = args.model(**model_kwargs)

    os.makedirs(args.plot_dir, exist_ok=True)
    exp_path = os.path.join(args.plot_dir, args.experiment_name)
    os.makedirs(exp_path, exist_ok=True)

    current_start_episode = 0
    idx = 0
    phases = phase_spec.split(';')
    use_tqdm = HAS_TQDM and args.verbose
    curr_timeout = None
    
    accuracy = []
    result = {}
    raw_result = {}
    predictions = {}
    

    convergence_stats = {} 
    patience = 5  

    overall_inference_times = []
    overall_training_times = []

    for phase in phases:
        idx += 1
        phase_accuracy = []
        
        phase_converged = False
        stable_count = 0
        last_pred = None
        converged_ep = -1
        converged_time_ms = -1.0
        
        phase_start_time = time.perf_counter() 
        
        if use_tqdm:
            episode_iterator = tqdm(
                range(args.num_episodes), 
                desc=f"PHASE {phase} ({idx}/{len(phases)})"
            )
        else:
            episode_iterator = range(args.num_episodes)
        
        raw_result[(idx, phase)] = []
        predictions[(idx, phase)] = []
        
        for phase_eps in episode_iterator:
            phase_loader, current_start_episode = generate_data(
                generator=generator, 
                protocol=protocol_name,
                phase=phase, 
                current_start_episode=current_start_episode, 
                scaler=scaler,       
                batch_size=args.batch_size,
                phase_options=generator.phase_options,
                num_episode=1,
                prev_timeout=curr_timeout,
                acceptable=acceptable,
            )
            
            inf_start_t = time.perf_counter()
            metrics, curr_timeout = evaluate_model(
                protocol=protocol_id,
                model=model,         
                eval_loader=phase_loader,
            )
            overall_inference_times.append((time.perf_counter() - inf_start_t) * 1000)
            
            predictions[(idx, phase)].append(curr_timeout)
            phase_accuracy.append(metrics['accuracy'])
            raw_result[(idx, phase)].append(metrics['accuracy'])

            if not phase_converged:
                if curr_timeout == last_pred:
                    stable_count += 1
                else:
                    stable_count = 1  
                    last_pred = curr_timeout

                if stable_count >= patience:
                    phase_converged = True
                    converged_ep = (phase_eps + 1) - patience + 1 
                    converged_time_ms = (time.perf_counter() - phase_start_time) * 1000 

            train_start_t = time.perf_counter()
            model, _ = train_model(
                protocol=protocol_id,
                model=model,
                train_loader=phase_loader,
                eval_loader=phase_loader,
                epochs=args.epochs,
            )
            overall_training_times.append((time.perf_counter() - train_start_t) * 1000)
            
            if use_tqdm:
                episode_iterator.set_postfix({
                    'Pred Acc': f"{sum(phase_accuracy)/len(phase_accuracy):.4f}",
                })
            
        if not phase_converged:
            converged_ep = args.num_episodes
            converged_time_ms = (time.perf_counter() - phase_start_time) * 1000
            

        convergence_stats[(idx, phase)] = {
            'ep': converged_ep, 
            'time_ms': converged_time_ms, 
            'converged': phase_converged,
        }
        
        phase_avg_acc = sum(phase_accuracy) / len(phase_accuracy)
        result[(idx, phase)] = phase_avg_acc
        accuracy.append(phase_avg_acc)

    plot_accuracy(result=result, save_path=os.path.join(exp_path, 'summary_accuracies.png'))
    plot_stage_accuracies(data_dict=raw_result, save_path=os.path.join(exp_path, 'episode_accuracies.png'))
    plot_prediction_phases(
        data_dict=predictions,
        save_path=os.path.join(exp_path, 'predictions.png'),
        y_ticks=list(get_timeout_values_ms(protocol_id)),
    )

    print(f"\n[SUMMARY]")
    print(
        f"[Experiment Configs:]\tprotocol={protocol_name}, "
        f"num_episodes={args.num_episodes}, phases={phase_spec}"
    )
    
    eps_values = [stat['ep'] for stat in convergence_stats.values()]
    time_values = [stat['time_ms'] for stat in convergence_stats.values()]

    avg_eps = np.mean(eps_values)
    std_eps = np.std(eps_values)
    med_eps = np.median(eps_values)
    max_eps = np.max(eps_values)

    avg_ms = np.mean(time_values)
    std_ms = np.std(time_values)
    med_ms = np.median(time_values)
    max_ms = np.max(time_values)
    

    avg_inf = np.mean(overall_inference_times)
    std_inf = np.std(overall_inference_times)
    med_inf = np.median(overall_inference_times)
    max_inf = np.max(overall_inference_times)

    avg_train = np.mean(overall_training_times)
    std_train = np.std(overall_training_times)
    med_train = np.median(overall_training_times)
    max_train = np.max(overall_training_times)

    print(
        f"[Overall Metrics:]\t"
        f"accuracy={(sum(accuracy) / len(accuracy)):.4f} (std={np.std(accuracy):.4f}), "
        f"conv_eps={avg_eps:.1f} (std={std_eps:.1f}, med={med_eps:.1f}, max={max_eps:.1f}), "
        f"conv_time={avg_ms:.2f} (std={std_ms:.2f}, med={med_ms:.2f}, max={max_ms:.2f}), "
        f"[Computation Overheads:]\t"
        f"inference={avg_inf:.2f}ms (std={std_inf:.2f}, med={med_inf:.2f}, max={max_inf:.2f}), "
        f"training={avg_train:.2f}ms (std={std_train:.2f}, med={med_train:.2f}, max={max_train:.2f})"
    )
    
    print(f"\n[Convergence Metrics per Phase:]")
    for (i, p), stats in convergence_stats.items():
        status = "Converged" if stats['converged'] else "Failed to Converge"
        print(f"  Phase {p} ({i}/{len(phases)}): {status} at Episode {stats['ep']} | Converge Time: {stats['time_ms']:.2f} ms")

    print(
        f"\n[Model Configs:]\t"
        f"model={model.__class__.__name__}, "
        f"epochs={args.epochs}, "
        f"batch_size={args.batch_size}, "
        f"seed={args.seed}, "
        f"Data Generation temperature= 0.5"
    )

if __name__ == "__main__":
    main()
    
