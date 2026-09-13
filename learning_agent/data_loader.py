from collections import defaultdict
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import agent_pb2
from models import (
    AUTOBAHN_CAR_TIMEOUT_VALUES_MS,
    AUTOBAHN_FAST_PATH_TIMEOUT_VALUES_MS,
    AUTOBAHN_TIMEOUT_DELAY_VALUES_MS,
    tendermint_timeouts_from_propose,
)
from reward_utils import reward_from_report_row

PBFT_LEGACY_FEATURE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "total_transactions": ("processed_transactions",),
    "avg_consensus_latency_ms": ("avg_message_delay",),
    "p50_consensus_latency_ms": ("median_message_delay", "p50_message_delay"),
    "p95_consensus_latency_ms": ("p95_message_delay",),
}


@dataclass(frozen=True)
class CsvTrainingData:
    features: List[List[float]]
    actions: List[float]
    rewards: List[float]
    labels: List[float]
    # Optional (n_samples, action_dim) 0/1 matrix marking every timeout that
    # counts as correct for the row's phase, not just the single best one.
    accepted: Optional[List[List[float]]] = None


class DatasetLoader:
    def __init__(
        self,
        features: Sequence[Sequence[float]],
        actions: Sequence[float],
        rewards: Sequence[float],
        labels: Sequence[float],
        batch_size: int = 1,
        shuffle: bool = False,
        accepted: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        if len(features) != len(actions) or len(actions) != len(rewards):
            raise ValueError("features/actions/rewards length mismatch")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not features:
            raise ValueError("Dataset must contain at least one sample.")

        self._features = np.asarray(features, dtype=float)
        self._actions = np.asarray(actions, dtype=float)
        self._rewards = np.asarray(rewards, dtype=float)
        self._labels = np.asarray(labels, dtype=float)
        self._accepted = None if accepted is None else np.asarray(accepted, dtype=float)
        if self._accepted is not None and len(self._accepted) != len(features):
            raise ValueError(
                f"accepted length mismatch: expected {len(features)}, "
                f"got {len(self._accepted)}"
            )
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

    def __next__(self) -> Dict[str, np.ndarray]:
        if self._cursor >= len(self._indices):
            raise StopIteration

        batch_idx = self._indices[self._cursor : self._cursor + self.batch_size]
        self._cursor += self.batch_size
        indices = np.asarray(batch_idx, dtype=int)
        return {
            "features": self._features[indices],
            "actions": self._actions[indices],
            "rewards": self._rewards[indices],
            "labels": self._labels[indices],
            "accepted": None if self._accepted is None else self._accepted[indices],
        }


def split_train_eval_loaders(
    data, 
    batch_size: int,
    eval_ratio: float = 0.2,
    seed: Optional[int] = None,
) -> Tuple[DatasetLoader, Optional[DatasetLoader]]:
    if not (0.0 <= eval_ratio < 1.0):
        raise ValueError(f"eval_ratio must be in [0.0, 1.0), got {eval_ratio}")

    num_samples = len(data.features)
    if num_samples == 0:
        raise ValueError("Cannot split empty dataset.")

    rng = random.Random(seed)

    if eval_ratio == 0.0 or num_samples == 1:
        train_idx = list(range(num_samples))
        rng.shuffle(train_idx)
        eval_idx = []
    else:
        label_to_indices = defaultdict(list)
        for i, label in enumerate(data.labels):
            label_val = label.item() if hasattr(label, 'item') else label
            label_to_indices[label_val].append(i)

        train_idx = []
        eval_idx = []

        #Stratified Sampling
        for label_val, idxs in label_to_indices.items():
            rng.shuffle(idxs)
            n_class = len(idxs)

            if n_class == 1:
                train_idx.extend(idxs)
            else:
                split_idx = int(n_class * (1.0 - eval_ratio))
                split_idx = max(1, min(split_idx, n_class - 1))

                # eval should have smaller ratio
                train_idx.extend(idxs[:split_idx])
                eval_idx.extend(idxs[split_idx:])

        rng.shuffle(train_idx)
        rng.shuffle(eval_idx)

    train_loader = DatasetLoader(
        features=[data.features[i] for i in train_idx],
        actions=[data.actions[i] for i in train_idx],
        rewards=[data.rewards[i] for i in train_idx],
        labels=[data.labels[i] for i in train_idx],
        batch_size=batch_size,
        shuffle=True,
    )
    
    eval_loader = None
    if eval_idx:
        eval_loader = DatasetLoader(
            features=[data.features[i] for i in eval_idx],
            actions=[data.actions[i] for i in eval_idx],
            rewards=[data.rewards[i] for i in eval_idx],
            labels=[data.labels[i] for i in eval_idx],
            batch_size=batch_size,
            shuffle=False, # Eval loader doesn't strictly need shuffling
        )

    return train_loader, eval_loader


def load_training_data_from_csv(
    path: Path,
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

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {path} is missing a header row.")

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
        raise ValueError(f"No completed episode rows found in {path}.")

    return CsvTrainingData(features=features, actions=actions, rewards=rewards, labels=labels)


def _is_empty_row(row: Dict[str, str]) -> bool:
    return all((value is None or not str(value).strip()) for value in row.values())


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
    direct = row.get(f"feature_{field_name}")
    if direct is not None and direct.strip() != "":
        return direct

    if protocol in (agent_pb2.PROTOCOL_PBFT, agent_pb2.PROTOCOL_SMARTBFT):
        aliases = PBFT_LEGACY_FEATURE_ALIASES.get(field_name, ())
        for alias in aliases:
            aliased = row.get(f"feature_{alias}")
            if aliased is not None and aliased.strip() != "":
                return aliased
        if field_name == "total_consensus_instances":
            total_raw = row.get("feature_processed_transactions") or row.get(
                "feature_total_transactions"
            )
            batch_raw = row.get("feature_avg_batch_size")
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
    if protocol in (agent_pb2.PROTOCOL_PBFT, agent_pb2.PROTOCOL_SMARTBFT):
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

    if protocol == agent_pb2.PROTOCOL_TENDERMINT:
        propose_timeout_ms = _parse_timeout_ms(
            row=row,
            candidates=(
                "propose_timeout_milliseconds_used",
                "propose_timeout_milliseconds",
            ),
            row_number=row_number,
        )
        prevote_timeout_ms = _parse_timeout_ms(
            row=row,
            candidates=(
                "prevote_timeout_milliseconds_used",
                "prevote_timeout_milliseconds",
            ),
            row_number=row_number,
        )
        precommit_timeout_ms = _parse_timeout_ms(
            row=row,
            candidates=(
                "precommit_timeout_milliseconds_used",
                "precommit_timeout_milliseconds",
            ),
            row_number=row_number,
        )
        expected_timeouts = tendermint_timeouts_from_propose(propose_timeout_ms)
        actual_timeouts = (
            propose_timeout_ms,
            prevote_timeout_ms,
            precommit_timeout_ms,
        )
        if actual_timeouts != expected_timeouts:
            raise ValueError(
                "Tendermint prevote and precommit timeouts must be one-third "
                f"of propose timeout at CSV line {row_number}: "
                f"expected {expected_timeouts}, got {actual_timeouts}"
            )
        return _timeout_ms_to_index(
            propose_timeout_ms, timeout_values_ms, row_number
        )

    if protocol == agent_pb2.PROTOCOL_HOTSTUFF:
        timeout_delay_ms = _parse_timeout_ms(
            row=row,
            candidates=(
                "timeout_delay_milliseconds_used",
                "timeout_delay_milliseconds",
            ),
            row_number=row_number,
        )
        return _timeout_ms_to_index(timeout_delay_ms, timeout_values_ms, row_number)

    if protocol == agent_pb2.PROTOCOL_AUTOBAHN:
        # The three Autobahn knobs have distinct value sets, so the shared
        # timeout_values_ms argument cannot cover them; the per-knob option
        # tuples from models are used instead.
        timeout_delay_ms = _parse_timeout_ms(
            row=row,
            candidates=(
                "timeout_delay_milliseconds_used",
                "timeout_delay_milliseconds",
            ),
            row_number=row_number,
        )
        car_timeout_ms = _parse_timeout_ms(
            row=row,
            candidates=(
                "car_timeout_milliseconds_used",
                "car_timeout_milliseconds",
            ),
            row_number=row_number,
        )
        fast_path_timeout_ms = _parse_timeout_ms(
            row=row,
            candidates=(
                "fast_path_timeout_milliseconds_used",
                "fast_path_timeout_milliseconds",
            ),
            row_number=row_number,
        )
        delay_idx = _timeout_ms_to_index(
            timeout_delay_ms, AUTOBAHN_TIMEOUT_DELAY_VALUES_MS, row_number
        )
        car_idx = _timeout_ms_to_index(
            car_timeout_ms, AUTOBAHN_CAR_TIMEOUT_VALUES_MS, row_number
        )
        fast_idx = _timeout_ms_to_index(
            fast_path_timeout_ms, AUTOBAHN_FAST_PATH_TIMEOUT_VALUES_MS, row_number
        )
        car_base = len(AUTOBAHN_CAR_TIMEOUT_VALUES_MS)
        fast_base = len(AUTOBAHN_FAST_PATH_TIMEOUT_VALUES_MS)
        return (delay_idx * car_base * fast_base) + (car_idx * fast_base) + fast_idx

    raise ValueError(f"Unsupported protocol: {protocol}")


def _parse_timeout_ms(
    row: Dict[str, str], candidates: Sequence[str], row_number: int
) -> int:
    for field_name in candidates:
        raw_value = row.get(field_name)
        if raw_value is None or raw_value.strip() == "":
            continue
        try:
            return int(float(raw_value))
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
    return reward_from_report_row(row, row_number)

def _extract_label(row: Dict[str, str], phase_options: Dict[int, int], row_number: int) -> float:
    for field_name in ("phase",):
        raw_value = row.get("phase")
        if raw_value is None or raw_value.strip() == "":
            continue
        try:
            return label_to_timeout_index(int(raw_value), phase_options= phase_options, row_number=row_number)
        except ValueError as exc:
            raise ValueError(
                f"Invalid label index for '{field_name}' at CSV line {row_number}: {raw_value!r}"
            ) from exc
    raise ValueError(
        f"Missing label index at CSV line {row_number}. Tried: phase"
    )

def label_to_timeout_index(
        label_index: int, phase_options: Dict[int, int], row_number: int
    ) -> int:
        result = phase_options.get(label_index)
        if result is None:
            raise ValueError(
                f"Unsupported label index {label_index} at CSV line {row_number}. "
                f"Valid options: {tuple(phase_options)}"
            )
        return result
