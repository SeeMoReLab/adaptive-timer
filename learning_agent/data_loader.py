import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

import agent_pb2

PBFT_LEGACY_FEATURE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "total_transactions": ("processed_transactions",),
    "avg_consensus_latency_ms": ("avg_message_delay",),
    "p95_consensus_latency_ms": ("p95_message_delay",),
    "p99_consensus_latency_ms": ("p99_message_delay",),
    "phase_propose_avg_delay_ms": ("avg_propose_delay",),
    "phase_propose_p95_delay_ms": ("avg_propose_delay",),
    "phase_write_avg_delay_ms": ("avg_write_delay",),
    "phase_write_p95_delay_ms": ("avg_write_delay",),
    "phase_accept_avg_delay_ms": ("avg_accept_delay",),
    "phase_accept_p95_delay_ms": ("avg_accept_delay",),
    "phase_post_decision_avg_delay_ms": ("avg_post_decision_delay",),
    "phase_post_decision_p95_delay_ms": ("avg_post_decision_delay",),
}


@dataclass(frozen=True)
class CsvTrainingData:
    features: List[List[float]]
    actions: List[float]
    rewards: List[float]


class DatasetLoader:
    def __init__(
        self,
        features: Sequence[Sequence[float]],
        actions: Sequence[float],
        rewards: Sequence[float],
        batch_size: int = 1,
        shuffle: bool = False,
    ) -> None:
        if len(features) != len(actions) or len(actions) != len(rewards):
            raise ValueError("features/actions/rewards length mismatch")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not features:
            raise ValueError("Dataset must contain at least one sample.")

        self._features = torch.tensor(features, dtype=torch.float32)
        self._actions = torch.tensor(actions, dtype=torch.float32)
        self._rewards = torch.tensor(rewards, dtype=torch.float32)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self._all_indices = list(range(len(features)))
        self._indices: List[int] = []
        self._cursor = 0

    def __len__(self) -> int:
        total = len(self._all_indices)
        return (total + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> "DatasetLoader":
        self._indices = list(self._all_indices)
        if self.shuffle:
            random.shuffle(self._indices)
        self._cursor = 0
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        if self._cursor >= len(self._indices):
            raise StopIteration

        batch_idx = self._indices[self._cursor : self._cursor + self.batch_size]
        self._cursor += self.batch_size
        index_tensor = torch.tensor(batch_idx, dtype=torch.long)
        return {
            "features": self._features.index_select(0, index_tensor),
            "actions": self._actions.index_select(0, index_tensor),
            "rewards": self._rewards.index_select(0, index_tensor),
        }


def split_train_eval_loaders(
    data: CsvTrainingData,
    batch_size: int,
    eval_ratio: float = 0.2,
    seed: Optional[int] = None,
) -> Tuple[DatasetLoader, Optional[DatasetLoader]]:
    if not (0.0 <= eval_ratio < 1.0):
        raise ValueError(f"eval_ratio must be in [0.0, 1.0), got {eval_ratio}")

    num_samples = len(data.features)
    if num_samples == 0:
        raise ValueError("Cannot split empty dataset.")

    indices = list(range(num_samples))
    rng = random.Random(seed)
    rng.shuffle(indices)

    if eval_ratio == 0.0 or num_samples == 1:
        split_idx = num_samples
    else:
        split_idx = int(num_samples * (1.0 - eval_ratio))
        split_idx = min(max(split_idx, 1), num_samples - 1)

    train_idx = indices[:split_idx]
    eval_idx = indices[split_idx:]

    train_loader = DatasetLoader(
        features=[data.features[i] for i in train_idx],
        actions=[data.actions[i] for i in train_idx],
        rewards=[data.rewards[i] for i in train_idx],
        batch_size=batch_size,
        shuffle=True,
    )

    eval_loader: Optional[DatasetLoader] = None
    if eval_idx:
        eval_loader = DatasetLoader(
            features=[data.features[i] for i in eval_idx],
            actions=[data.actions[i] for i in eval_idx],
            rewards=[data.rewards[i] for i in eval_idx],
            batch_size=batch_size,
            shuffle=False,
        )

    return train_loader, eval_loader


def load_training_data_from_csv(
    path: Path,
    protocol: int,
    feature_field_names: Sequence[str],
    timeout_values_ms: Sequence[int],
    reward_transform: Optional[Callable[[float], float]] = None,
) -> CsvTrainingData:
    reward_transform = reward_transform or (lambda value: value)

    features: List[List[float]] = []
    actions: List[float] = []
    rewards: List[float] = []

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {path} is missing a header row.")

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

            features.append(feature_values)
            actions.append(float(action_index))
            rewards.append(float(reward_transform(reward_value)))

    if not features:
        raise ValueError(
            f"No usable rows found in {path}. "
            "Rows with matched_episode_features=0 are skipped."
        )

    return CsvTrainingData(features=features, actions=actions, rewards=rewards)


def _is_empty_row(row: Dict[str, str]) -> bool:
    return all((value is None or not str(value).strip()) for value in row.values())


def _is_matched_row(row: Dict[str, str]) -> bool:
    matched_raw = row.get("matched_episode_features")
    if matched_raw is None:
        return True
    normalized = matched_raw.strip().lower()
    return normalized in {"1", "true", "yes", "y"}


def _extract_features(
    protocol: int,
    row: Dict[str, str],
    feature_field_names: Sequence[str],
    row_number: int,
) -> List[float]:
    feature_values: List[float] = []
    for field_name in feature_field_names:
        raw_value = _lookup_feature_raw_value(protocol, row, field_name)
        if raw_value is None:
            raise ValueError(
                f"Missing feature '{field_name}' at CSV line {row_number}."
            )
        try:
            feature_values.append(float(raw_value))
        except ValueError as exc:
            raise ValueError(
                f"Invalid feature value for '{field_name}' at CSV line {row_number}: {raw_value!r}"
            ) from exc
    return feature_values


def _lookup_feature_raw_value(
    protocol: int, row: Dict[str, str], field_name: str
) -> Optional[str]:
    direct = row.get(field_name)
    if direct is not None and direct.strip() != "":
        return direct

    if protocol == agent_pb2.PROTOCOL_PBFT:
        aliases = PBFT_LEGACY_FEATURE_ALIASES.get(field_name, ())
        for alias in aliases:
            aliased = row.get(alias)
            if aliased is not None and aliased.strip() != "":
                return aliased
        if field_name == "total_consensus_instances":
            total_raw = row.get("processed_transactions") or row.get("total_transactions")
            batch_raw = row.get("avg_batch_size")
            if total_raw and batch_raw:
                try:
                    total_transactions = float(total_raw)
                    avg_batch = max(float(batch_raw), 1.0)
                    return str(max(1.0, total_transactions / avg_batch))
                except ValueError:
                    return None
    return None


def _extract_action_index(
    protocol: int,
    row: Dict[str, str],
    timeout_values_ms: Sequence[int],
    row_number: int,
) -> int:
    if protocol == agent_pb2.PROTOCOL_PBFT:
        election_timeout_ms = _parse_timeout_ms(
            row=row,
            candidates=(
                "election_timeout_milliseconds_used",
                "timeout_milliseconds_used",
                "election_timeout_milliseconds",
            ),
            row_number=row_number,
        )
        return _timeout_ms_to_index(election_timeout_ms, timeout_values_ms, row_number)

    if protocol == agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT:
        election_timeout_ms = _parse_timeout_ms(
            row=row,
            candidates=(
                "election_timeout_milliseconds_used",
                "election_timeout_milliseconds",
            ),
            row_number=row_number,
        )
        prepare_timeout_ms = _parse_timeout_ms(
            row=row,
            candidates=(
                "prepare_timeout_milliseconds_used",
                "prepare_timeout_milliseconds",
            ),
            row_number=row_number,
        )
        election_idx = _timeout_ms_to_index(
            election_timeout_ms, timeout_values_ms, row_number
        )
        prepare_idx = _timeout_ms_to_index(
            prepare_timeout_ms, timeout_values_ms, row_number
        )
        return election_idx * len(timeout_values_ms) + prepare_idx

    if protocol == agent_pb2.PROTOCOL_SBFT:
        election_timeout_ms = _parse_timeout_ms(
            row=row,
            candidates=(
                "election_timeout_milliseconds_used",
                "election_timeout_milliseconds",
            ),
            row_number=row_number,
        )
        slow_timeout_ms = _parse_timeout_ms(
            row=row,
            candidates=(
                "slow_path_timeout_milliseconds_used",
                "slow_path_timeout_milliseconds",
            ),
            row_number=row_number,
        )
        batch_timeout_ms = _parse_timeout_ms(
            row=row,
            candidates=(
                "batch_timeout_milliseconds_used",
                "batch_timeout_milliseconds",
            ),
            row_number=row_number,
        )
        election_idx = _timeout_ms_to_index(
            election_timeout_ms, timeout_values_ms, row_number
        )
        slow_idx = _timeout_ms_to_index(slow_timeout_ms, timeout_values_ms, row_number)
        batch_idx = _timeout_ms_to_index(batch_timeout_ms, timeout_values_ms, row_number)
        base = len(timeout_values_ms)
        return (election_idx * base * base) + (slow_idx * base) + batch_idx

    raise ValueError(f"Unsupported protocol: {protocol}")


def _parse_timeout_ms(
    row: Dict[str, str], candidates: Sequence[str], row_number: int
) -> int:
    for field_name in candidates:
        raw_value = row.get(field_name)
        if raw_value is None or raw_value.strip() == "":
            continue
        try:
            return int(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid timeout '{field_name}' at CSV line {row_number}: {raw_value!r}"
            ) from exc
    raise ValueError(
        f"Missing timeout column at CSV line {row_number}. "
        f"Tried: {', '.join(candidates)}"
    )


def _timeout_ms_to_index(
    timeout_ms: int, timeout_values_ms: Sequence[int], row_number: int
) -> int:
    try:
        return list(timeout_values_ms).index(timeout_ms)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported timeout {timeout_ms}ms at CSV line {row_number}. "
            f"Valid options: {tuple(timeout_values_ms)}"
        ) from exc


def _extract_reward(row: Dict[str, str], row_number: int) -> float:
    for field_name in ("reward", "total_transactions"):
        raw_value = row.get(field_name)
        if raw_value is None or raw_value.strip() == "":
            continue
        try:
            return float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid reward value for '{field_name}' at CSV line {row_number}: {raw_value!r}"
            ) from exc
    raise ValueError(
        f"Missing reward value at CSV line {row_number}. Tried: reward,total_transactions"
    )
