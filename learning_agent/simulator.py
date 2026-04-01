import argparse
import csv
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import torch
from tqdm import tqdm
import numpy as np
import os
import random
import matplotlib.pyplot as plt

import agent_pb2
from data_loader import (
    CsvTrainingData, 
    DatasetLoader, 
    _extract_action_index, 
    _extract_features, 
    _extract_label, 
    _extract_reward, 
    _is_empty_row, 
    _is_matched_row
    )

from models import (
    TIMEOUT_VALUES_MS,
    PHASE_INDEX_TO_TIMEOUT_INDEX,
    TIMEOUT_INDEX_TO_MS,
    LinearTorchModel,
    SVMTorchModel,
    PPOTorchModel,
    ThompsonSamplingModel,
    TensorModel,
    QuadraticRewardScaler,
    RewardTransformer,
    get_action_space_size,
    get_report_feature_dim,
    get_report_feature_field_names,
)

from generator import DataGenerator

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

PROTOCOL_NAME_TO_ENUM = {
    "pbft": agent_pb2.PROTOCOL_PBFT,
    "two_pc_over_pbft": agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT,
    "2pc": agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT,
    "sbft": agent_pb2.PROTOCOL_SBFT,
}

MODEL_NAME_TO_TENSORMODEL = {
    "linear": LinearTorchModel,
    "svm": SVMTorchModel,
    "ppo": PPOTorchModel,
    "cmab": ThompsonSamplingModel
}

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def _parse_protocol(value: str) -> int:
    protocol = PROTOCOL_NAME_TO_ENUM.get(value.strip().lower().replace("-", "_"))
    if protocol is None:
        valid = ", ".join(sorted({"pbft", "two_pc_over_pbft", "sbft"}))
        raise argparse.ArgumentTypeError(
            f"Unsupported protocol '{value}'. Valid options: {valid}"
        )
    return protocol

def _parse_model(value: str) -> TensorModel:
    model = MODEL_NAME_TO_TENSORMODEL.get(value.strip().lower())
    if model is None:
        valid = ", ".join(sorted({"linear", 'svm',"ppo","cmab"}))
        raise argparse.ArgumentTypeError(
            f"Unsupported model '{value}'. Valid options: {valid}"
        )
    return model

def _parse_data(
    reader: csv.DictReader,
    protocol: int,
    feature_field_names: Sequence[str],
    timeout_values_ms: Sequence[int],
    phase_options: Dict[int, int],
    reward_transform: Optional[Callable[[float], float]] = None,
) -> CsvTrainingData:
    reward_transform = reward_transform or (lambda value: value)

    features: List[List[float]] = []
    actions: List[float] = []
    rewards: List[float] = []
    labels: List[float] = []

    if reader.fieldnames is None:
        raise ValueError(f"Missing a header row.")

    for row_number, row in enumerate(reader, start=2):
        if _is_empty_row(row):
            continue
        if not _is_matched_row(row):
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
        reward_value = _extract_reward(row=row, row_number=row_number)

        label_index = _extract_label(
            row=row, 
            phase_options=phase_options,
            row_number=row_number
        )

        features.append(feature_values)
        actions.append(float(action_index))
        rewards.append(float(reward_transform(reward_value)))
        labels.append(float(label_index))


    if not features:
        raise ValueError(
            f"No usable rows found "
            "Rows with matched_episode_features=0 are skipped."
        )

    return CsvTrainingData(features=features, actions=actions, rewards=rewards, labels=labels)

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End to End Learning Agent Simulation"
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default='pbft',
        help="Protocol: pbft | two_pc_over_pbft | sbft",
    )
    parser.add_argument(
        "--model",
        type=_parse_model,
        default='linear',
        help="Model: linear | svm",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Observation Sample Directory path.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=Path(__file__).resolve().parent / "plot",
        help="Plot Image Directory path.",
    )
    
    parser.add_argument(
        "--summary-plot-filename",
        type=str,
        default= 'lineplot.png',
        help="Summary plot Image Filename.",
    )

    parser.add_argument(
        "--complete-plot-filename",
        type=str,
        default= 'complet_lineplot.png',
        help="Stages plot Image Filename.",
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
        default="0;1;2;3;4;5;0;2;1;5;4;2;1",
        help="Phase sequence.",
    )

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reward-scaler",
        choices=("raw", "quadratic"),
        default="raw",
        help="How to scale reward column before training.",
    )

    return parser

def generate_data(
        generator: DataGenerator, protocol: str, phase: int, current_start_episode: int, scaler: RewardTransformer, batch_size: int, num_episode:int = 1, prev_timeout: List[int] = None
        )-> DatasetLoader:
    phase_data, new_start_episode = generator.generate_phase(phase_index=int(phase), current_start_episode=current_start_episode, num_episode=num_episode, prev_timeout = prev_timeout)
    parsed_data = _parse_data(
            reader=phase_data,
            protocol=PROTOCOL_NAME_TO_ENUM[protocol],
            feature_field_names=get_report_feature_field_names(PROTOCOL_NAME_TO_ENUM[protocol]),
            timeout_values_ms=TIMEOUT_VALUES_MS,
            phase_options=PHASE_INDEX_TO_TIMEOUT_INDEX, 
            reward_transform= scaler,
    )  

    loader = DatasetLoader(
        features=parsed_data.features,
        actions=parsed_data.actions,
        rewards=parsed_data.rewards,
        labels=parsed_data.labels,
        batch_size=batch_size,
        shuffle=True,
    )

    return loader, new_start_episode

def train_model(
    model: TensorModel,
    train_loader: DatasetLoader,
    eval_loader: DatasetLoader,
    epochs: int,
    save_path: Optional[Path] = None,
    early_stop: Optional[int] = 20,
) -> Tuple[TensorModel, Dict]:
    
    best_weights = model.save()
    best_eval_metrics = float("-inf")
    count = 0

    stats = {'best_epoch':0, 'best_accuracy':0.0, 'best_loss':0.0}
    for e in range(epochs):
        train_loss = _run_epoch(model, train_loader, backward=True)
        metrics,_ = evaluate_model(
            model=model,
            eval_loader= eval_loader,
        )
        if metrics['accuracy'] > best_eval_metrics:
            best_eval_metrics = metrics['accuracy']
            best_weights = model.save()

            stats['best_epoch'] = e
            stats['best_accuracy'] = metrics['accuracy']
            stats['best_loss'] = metrics['weighted_loss']
            count = 0
        else:
            count += 1
            if count == early_stop:
                break
            
    model.load(best_weights)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_weights, save_path)

    return model, stats

def _run_epoch(model: TensorModel, loader: DatasetLoader, backward: bool) -> float:
    loss_sum = 0.0
    batch_count = 0

    for batch in loader:
        loss_sum += model.update_model(
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
    model: TensorModel, eval_loader: DatasetLoader
) -> Tuple[Dict[str, float], List[int]]:

    total = 0
    correct = 0
    weighted_loss_sum = 0.0

    for batch in eval_loader:
        weighted_loss_sum += model.update_model(
            features = batch["features"], actions = batch["actions"], rewards = batch["rewards"], backward=False
        )
        logits = model.predict_action(batch["features"])
        predicted = torch.argmax(logits, dim=1)
        expected = torch.round(batch["labels"]).long()
        total += int(expected.numel())
        correct += int((predicted == expected).sum().item())
        curr_timeout = [TIMEOUT_INDEX_TO_MS[idx] for idx in predicted.tolist()]

    accuracy = (correct / total) if total else 0.0

    return {
        "test samples": float(total),
        "accuracy": float(accuracy),
        "weighted_loss": float(weighted_loss_sum / max(len(eval_loader), 1)),

    }, curr_timeout

def plot_accuracy(result: Dict[str, float], save_path: str = 'accuracy_plot.pdf') -> None:
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
    
    phases = [str(p[1]) for p in list(result.keys())] 
    accuracies = list(result.values())

    x_positions = range(len(accuracies))

    fig, ax = plt.subplots()
    
    ax.plot(x_positions, accuracies, marker='o', color='#1f77b4', linestyle='-', label='Accuracy')
    
    ax.set_xticks(x_positions)
    
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('Phase')
    ax.set_ylabel('Accuracy')

    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_stage_accuracies(data_dict: Dict[Tuple[int, str], List[float]], save_path: str = 'stage_accuracies.pdf') -> None:
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
    

def main() -> None:
    set_seed(11)

    args = _build_parser().parse_args()
    print(f'Initiating Experiment [Model:] {args.model} [Protocol:] {args.protocol} [Phases:] {args.phases}')

    generator = DataGenerator(
            protocol=args.protocol, 
            directory_path=args.data_dir, 
        )
    scaler = QuadraticRewardScaler.transform if args.reward_scaler == "quadratic" else None

    model = args.model(
        action_dim=get_action_space_size(_parse_protocol(args.protocol)),
        n_features=get_report_feature_dim(_parse_protocol(args.protocol)),
        lr=args.lr,
    )

    os.makedirs(args.plot_dir, exist_ok=True)
    current_start_episode = 0
    idx = 0
    phases = args.phases.split(';')
    use_tqdm = HAS_TQDM and args.verbose
    curr_timeout = None
    accuracy = []
    result= {}
    raw_result = {}
    for phase in phases:
        idx += 1
        phase_accuracy, phase_loss = [], []
        print('='*100)
        if use_tqdm:
            episode_iterator = tqdm(
                range(args.num_episodes), 
                desc=f"PHASE {phase} ({idx}/{len(phases)})"
            )
        else:
            episode_iterator = range(args.num_episodes)
        
        raw_result[(idx,phase)] = []
        for phase_eps in episode_iterator:
            phase_loader, current_start_episode = generate_data(
                generator=generator, 
                protocol=args.protocol, 
                phase=phase, 
                current_start_episode=current_start_episode, 
                scaler=scaler,       
                batch_size=args.batch_size,
                num_episode=1,
                prev_timeout = curr_timeout
            )
            metrics, curr_timeout = evaluate_model(
                model=model,         
                eval_loader=phase_loader,
            )

            phase_accuracy.append(metrics['accuracy'])
            raw_result[(idx,phase)].append(metrics['accuracy'])

            model, _ = train_model(
                model=model,
                train_loader=phase_loader,
                eval_loader=phase_loader,
                epochs=args.epochs,
            )
            
            if use_tqdm:
                episode_iterator.set_postfix({
                    'Pred Acc': f"{sum(phase_accuracy)/len(phase_accuracy):.4f}",
                })
            
        
        result[(idx,phase)] = sum(phase_accuracy)/len(phase_accuracy)
        accuracy.append(sum(phase_accuracy)/len(phase_accuracy))

        
    plot_accuracy(result= result, save_path = os.path.join(args.plot_dir, args.summary_plot_filename))
    plot_stage_accuracies(data_dict= raw_result, save_path = os.path.join(args.plot_dir, args.complete_plot_filename))

    print(
        f"\n[SUMMARY]]\n"
        f"[Experiment Configs:]\t"
        f"protocol={args.protocol}, "
        f"num_episodes={args.num_episodes}, "
        f"phases={args.phases}, "
        f"accuracy={(sum(accuracy) / len(accuracy)):.4f}, "
        f"std={np.std(accuracy):.4f},"
        f'plot filename: {os.path.join(args.plot_dir, args.summary_plot_filename)}\t'


        f"[Model Configs:]\t"
        f"model={model.__class__.__name__}, "
        f"lr={args.lr}, "
        f"epochs={args.epochs}, "
        f"batch_size={args.batch_size}, "
        f"seed={args.seed}, "
        )
        

if __name__ == "__main__":
    main()
    