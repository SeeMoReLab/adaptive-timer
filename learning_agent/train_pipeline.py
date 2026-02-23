import argparse
from pathlib import Path
from typing import Callable, Dict, Optional
import random
import torch

import agent_pb2
from data_loader import DatasetLoader, load_training_data_from_csv, split_train_eval_loaders
from models import (
    TIMEOUT_VALUES_MS,
    PHASE_INDEX_TO_TIMEOUT_INDEX,
    LinearTorchModel,
    DQNTorchModel,
    TensorModel,
    QuadraticRewardScaler,
    get_action_space_size,
    get_report_feature_dim,
    get_report_feature_field_names,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

PROTOCOL_NAME_TO_ENUM = {
    "pbft": agent_pb2.PROTOCOL_PBFT,
    "two_pc_over_pbft": agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT,
    "2pc": agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT,
    "sbft": agent_pb2.PROTOCOL_SBFT,
}

MODEL_NAME_TO_TENSORMODEL = {
    "linear": LinearTorchModel,
    "dqn": DQNTorchModel,
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
        valid = ", ".join(sorted({"linear", "dqn"}))
        raise argparse.ArgumentTypeError(
            f"Unsupported model '{value}'. Valid options: {valid}"
        )
    return model

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and test timeout model from CSV files."
    )
    parser.add_argument(
        "--protocol",
        type=_parse_protocol,
        default=agent_pb2.PROTOCOL_PBFT,
        help="Protocol: pbft | two_pc_over_pbft | sbft",
    )
    parser.add_argument(
        "--model",
        type=_parse_model,
        default='linear',
        help="Model: linear | dqn",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "sample_pbft_large.csv",
        help="Training CSV path.",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=None,
        help="Optional test CSV path. If omitted, train CSV is reused.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reward-scaler",
        choices=("raw", "quadratic"),
        default="raw",
        help="How to scale reward column before training.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def run_pipeline(
    protocol: int,
    model: TensorModel,
    train_csv: Path,
    test_csv: Optional[Path],
    epochs: int,
    batch_size: int,
    eval_ratio: float,
    lr: float,
    seed: int,
    reward_scaler: str,
    verbose: bool,
) -> None:
    protocol = int(protocol)
    reward_transform = (
        QuadraticRewardScaler.transform if reward_scaler == "quadratic" else None
    )
    model = train_linear_model_from_csv(
        protocol=protocol,
        model=model,
        train_csv=train_csv,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        eval_ratio=eval_ratio,
        verbose=verbose,
        reward_transform=reward_transform,
        seed=seed,
    )
    metrics = evaluate_linear_model_on_csv(
        model=model,
        protocol=protocol,
        path=test_csv or train_csv,
        batch_size=batch_size,
    )
    protocol_name = agent_pb2.Protocol.Name(protocol)
    print(
        "[csv-train-test] "
        f"protocol={protocol_name}, "
        f"train_csv={train_csv}, "
        f"test_csv={test_csv or train_csv}, "
        f"samples={int(metrics['samples'])}, "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"f1_macro={metrics['f1_macro']:.4f}, "
        f"weighted_loss={metrics['weighted_loss']:.6f}"
    )


def train_linear_model_from_csv(
    protocol: int,
    model: TensorModel,
    train_csv: Path,
    lr: float,
    epochs: int,
    batch_size: int,
    eval_ratio: float,
    verbose: bool,
    reward_transform: Optional[Callable[[float], float]],
    seed: Optional[int],
) -> TensorModel:
    data = load_training_data_from_csv(
        path=train_csv,
        protocol=protocol,
        feature_field_names=get_report_feature_field_names(protocol),
        timeout_values_ms=TIMEOUT_VALUES_MS,
        phase_options=PHASE_INDEX_TO_TIMEOUT_INDEX,
        reward_transform=reward_transform,
    )
    train_loader, eval_loader = split_train_eval_loaders(
        data=data,
        batch_size=batch_size,
        eval_ratio=eval_ratio,
        seed=seed,
    )

    
    tensor_model = model(
        action_dim=get_action_space_size(protocol),
        n_features=get_report_feature_dim(protocol),
        lr=lr,
    )


    best_weights = tensor_model.save()
    best_eval_loss = float("inf")

    show_progress = bool(verbose and tqdm is not None)
    epochs_iter = tqdm(range(epochs), desc="Training") if show_progress else range(epochs)
    for e in epochs_iter:
        train_loss = _run_epoch(tensor_model, train_loader, backward=True)
        eval_loss = (
            train_loss
            if eval_loader is None
            else _run_epoch(tensor_model, eval_loader, backward=False)
        )
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_weights = tensor_model.save()
        if show_progress:
            epochs_iter.set_postfix(
                Epoch=f"{e + 1}/{epochs}",
                Train_Loss=f"{train_loss:.7f}",
                Eval_Loss=f"{eval_loss:.7f}",
            )

   
    tensor_model.load(best_weights)

    return tensor_model


def evaluate_linear_model_on_csv(
    model: TensorModel, protocol: int, path: Path, batch_size: int = 64
) -> Dict[str, float]:
    data = load_training_data_from_csv(
        path=path,
        protocol=protocol,
        feature_field_names=get_report_feature_field_names(protocol),
        timeout_values_ms=TIMEOUT_VALUES_MS,
        phase_options=PHASE_INDEX_TO_TIMEOUT_INDEX
    )
    loader = DatasetLoader(
        features=data.features,
        actions=data.actions,
        rewards=data.rewards,
        labels=data.labels,
        batch_size=batch_size,
        shuffle=False,
    )

    total = 0
    correct = 0
    num_classes = len(TIMEOUT_VALUES_MS)
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)
    f1_sum = 0.0

    weighted_loss_sum = 0.0
    for batch in loader:
        weighted_loss_sum += model.update_model(
            batch["features"], batch["actions"], batch["rewards"], backward=False
        )
        logits = model.predict_action(batch["features"])
        predicted = torch.argmax(logits, dim=1)
        expected = torch.round(batch["labels"]).long()
        total += int(expected.numel())
        correct += int((predicted == expected).sum().item())

        for c in range(num_classes):
            tp[c] += ((predicted == c) & (expected == c)).sum().item()
            fp[c] += ((predicted == c) & (expected != c)).sum().item()
            fn[c] += ((predicted != c) & (expected == c)).sum().item()

    for c in range(num_classes):
        p_denom = tp[c] + fp[c]
        r_denom = tp[c] + fn[c]

        precision = tp[c] / p_denom if p_denom > 0 else 0.0
        recall = tp[c] / r_denom if r_denom > 0 else 0.0

        f1_denom = precision + recall
        f1_class = 2 * (precision * recall) / f1_denom if f1_denom > 0 else 0.0
        f1_sum += f1_class

    accuracy = (correct / total) if total else 0.0
    f1 = f1_sum / num_classes

    return {
        "samples": float(total),
        "accuracy": float(accuracy),
        "f1_macro": float(f1),
        "weighted_loss": float(weighted_loss_sum / max(len(loader), 1)),
    }


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def main() -> None:
    args = _build_parser().parse_args()
    set_seed(11)
    run_pipeline(
        protocol=args.protocol,
        model=args.model,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_ratio=args.eval_ratio,
        lr=args.lr,
        seed=args.seed,
        reward_scaler=args.reward_scaler,
        verbose=True,
    )


if __name__ == "__main__":
    main()
