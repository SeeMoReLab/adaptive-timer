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
import time

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
    PHASE_INDEX_TO_TIMEOUT_INDEX,
    LinearTorchModel,
    ThompsonSamplingModel,
    ThompsonSamplingModelOld,
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
    "two_pc_over_pbft": agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT,
    "2pc": agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT,
    "sbft": agent_pb2.PROTOCOL_SBFT,
    "tendermint": agent_pb2.PROTOCOL_TENDERMINT,
}

MODEL_NAME_TO_TENSORMODEL = {
    "linear": LinearTorchModel,
    "cmab": ThompsonSamplingModel,
    "cmab_old": ThompsonSamplingModelOld,
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
        valid = ", ".join(sorted({"pbft", "two_pc_over_pbft", "sbft", "tendermint"}))
        raise argparse.ArgumentTypeError(
            f"Unsupported protocol '{value}'. Valid options: {valid}"
        )
    return protocol

def _parse_model(value: str) -> TensorModel:
    model = MODEL_NAME_TO_TENSORMODEL.get(value.strip().lower())
    if model is None:
        valid = ", ".join(sorted({"linear", 'svm',"ppo","cmab", "cmab_old"}))
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
        help="Protocol: pbft | two_pc_over_pbft | sbft | tendermint",
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
        default="0;1;2;3;4;5;0;2;1;5;4;2;1",
        help="Phase sequence.",
    )

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
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
    phase_data, new_start_episode = generator.generate_local_report(phase_index=int(phase), current_start_episode=current_start_episode, num_episode=num_episode, prev_timeout = prev_timeout)
    parsed_data = _parse_data(
            reader=phase_data,
            protocol=PROTOCOL_NAME_TO_ENUM[protocol],
            feature_field_names=get_report_feature_field_names(PROTOCOL_NAME_TO_ENUM[protocol]),
            timeout_values_ms=get_timeout_values_ms(PROTOCOL_NAME_TO_ENUM[protocol]),
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
    protocol: int,
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
        curr_timeout = [
            timeout_index_to_ms_for_protocol(protocol, idx)
            for idx in predicted.tolist()
        ]

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
    set_seed(11)

    args = _build_parser().parse_args()
    print(f'Initiating Experiment [Model:] {args.model} [Protocol:] {args.protocol} [Phases:] {args.phases} ')

    # 1. Initialization
    generator = DataGenerator(
        protocol=args.protocol, 
        directory_path=args.data_dir, 
    )
    scaler = QuadraticRewardScaler.transform if args.reward_scaler == "quadratic" else None

    if args.model in (ThompsonSamplingModel, ThompsonSamplingModelOld):
        model = args.model(
            action_dim=get_action_space_size(_parse_protocol(args.protocol)),
            n_features=get_report_feature_dim(_parse_protocol(args.protocol)),
        )
    else:
        model = args.model(
        action_dim=get_action_space_size(_parse_protocol(args.protocol)),
        n_features=get_report_feature_dim(_parse_protocol(args.protocol)),
    )

    os.makedirs(args.plot_dir, exist_ok=True)
    exp_path = os.path.join(args.plot_dir, args.experiment_name)
    os.makedirs(exp_path, exist_ok=True)

    current_start_episode = 0
    idx = 0
    phases = args.phases.split(';')
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
                protocol=args.protocol, 
                phase=phase, 
                current_start_episode=current_start_episode, 
                scaler=scaler,       
                batch_size=args.batch_size,
                num_episode=1,
                prev_timeout=curr_timeout
            )
            
            inf_start_t = time.perf_counter()
            metrics, curr_timeout = evaluate_model(
                protocol=PROTOCOL_NAME_TO_ENUM[args.protocol],
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
                protocol=PROTOCOL_NAME_TO_ENUM[args.protocol],
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
    plot_prediction_phases(data_dict=predictions, save_path=os.path.join(exp_path, 'predictions.png'), y_ticks=[300, 800, 1800])

    print(f"\n[SUMMARY]")
    print(f"[Experiment Configs:]\tprotocol={args.protocol}, num_episodes={args.num_episodes}, phases={args.phases}")
    
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
    
