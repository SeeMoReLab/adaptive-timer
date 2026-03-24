import argparse
import csv
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import torch
from tqdm import tqdm
import numpy as np

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
    LinearTorchModel,
    SVMTorchModel,
    PPOTorchModel,
    TensorModel,
    QuadraticRewardScaler,
    RewardTransformer,
    get_action_space_size,
    get_report_feature_dim,
    get_report_feature_field_names,
)

from train_pipeline import _run_epoch, evaluate_linear_model_on_csv, set_seed, train_linear_model_from_csv
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
    "ppo": PPOTorchModel
}

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
        valid = ", ".join(sorted({"linear", 'svm'}))
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
    parser.add_argument("--epochs", type=int, default=100)
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
    
        metrics,_ = evaluate_linear_model_on_csv(
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

    current_start_episode = 0
    idx = 0
    phases = args.phases.split(';')
    use_tqdm = HAS_TQDM and args.verbose
    curr_timeout = None
    accuracy = []
    for phase in phases:
        idx += 1
        phase_accuracy, phase_loss = [], []
        if use_tqdm:
            episode_iterator = tqdm(
                range(args.num_episodes), 
                desc=f"PHASE {phase} ({idx}/{len(phases)})"
            )
        else:
            episode_iterator = range(args.num_episodes)
            
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

            metrics, curr_timeout = evaluate_linear_model_on_csv(
                model=model,         
                eval_loader=phase_loader,
            )

            phase_accuracy.append(metrics['accuracy'])
            phase_loss.append(metrics['weighted_loss'])


            model, _ = train_model(
                model=model,
                train_loader=phase_loader,
                eval_loader=phase_loader,
                epochs=args.epochs,
            )
            
            if use_tqdm:
                episode_iterator.set_postfix({
                    'Pred Acc': f"{sum(phase_accuracy)/len(phase_accuracy):.4f}",
                    'Pred Loss': f"{sum(phase_loss)/len(phase_loss):.4f}",
                })

        accuracy.append(sum(phase_accuracy)/len(phase_accuracy))

    print(
        f"\n[SUMMARY]]\n"
        f"[Experiment Configs:]\t"
        f"protocol={args.protocol}, "
        f"num_episodes={args.num_episodes}, "
        f"phases={args.phases}, "
        f"accuracy={(sum(accuracy) / len(accuracy)):.4f}, "
        f"std={np.std(accuracy):.4f}"

        f"[Model Configs:]\t"
        f"model={model.__class__.__name__}, "
        f"lr={args.lr}, "
        f"epochs={args.epochs}, "
        f"batch_size={args.batch_size}, "
        f"seed={args.seed}, "
        )
        

if __name__ == "__main__":
    main()
    
