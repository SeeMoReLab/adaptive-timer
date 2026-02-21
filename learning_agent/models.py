from abc import ABC, abstractmethod
import csv
from enum import IntEnum
import time
from pathlib import Path
from typing import Dict, Generic, List, Optional, Tuple, TypeVar
import random
import copy
from tqdm import tqdm

import torch
import torch.nn.functional as F

import agent_pb2

S = TypeVar("S")  # generic state type
A = TypeVar("A")  # generic action type


class TimeoutOption(IntEnum):
    TIMEOUT_300_MS = 300
    TIMEOUT_600_MS = 600
    TIMEOUT_1800_MS = 1800


TIMEOUT_OPTIONS: Tuple[TimeoutOption, ...] = (
    TimeoutOption.TIMEOUT_300_MS,
    TimeoutOption.TIMEOUT_600_MS,
    TimeoutOption.TIMEOUT_1800_MS,
)
TIMEOUT_VALUES_MS: Tuple[int, ...] = tuple(int(option) for option in TIMEOUT_OPTIONS)
TIMEOUT_MS_TO_INDEX: Dict[int, int] = {
    value_ms: idx for idx, value_ms in enumerate(TIMEOUT_VALUES_MS)
}

PBFT_REPORT_FEATURE_FIELD_NAMES: Tuple[str, ...] = (
    "total_transactions",
    "total_consensus_instances",
    "avg_consensus_latency_ms",
    "p95_consensus_latency_ms",
    "p99_consensus_latency_ms",
    "throughput_tps",
    "timeout_violation_rate",
    "avg_batch_size",
    "p95_batch_size",
    "leader_change_count",
    "regency_change_count",
    "phase_propose_avg_delay_ms",
    "phase_propose_p95_delay_ms",
    "phase_write_avg_delay_ms",
    "phase_write_p95_delay_ms",
    "phase_accept_avg_delay_ms",
    "phase_accept_p95_delay_ms",
    "phase_post_decision_avg_delay_ms",
    "phase_post_decision_p95_delay_ms",
)
TWO_PC_OVER_PBFT_REPORT_FEATURE_FIELD_NAMES: Tuple[str, ...] = (
    "total_transactions",
    "single_shard_transactions",
    "cross_shard_transactions",
    "prepare_ok_count",
    "prepare_fail_count",
    "commit_ok_count",
    "abort_count",
    "in_doubt_count",
    "prepare_timeout_count",
    "commit_timeout_unknown_outcome_count",
    "abort_best_effort_timeout_count",
    "lock_contention_count",
    "prepare_timeout_rate",
    "abort_rate",
    "commit_rate",
    "lock_contention_rate",
    "avg_end_to_end_latency_ms",
    "p95_end_to_end_latency_ms",
    "p99_end_to_end_latency_ms",
    "avg_prepare_latency_ms",
    "p95_prepare_latency_ms",
    "avg_commit_latency_ms",
    "p95_commit_latency_ms",
    "avg_lock_wait_ms",
    "p95_lock_wait_ms",
    "avg_lock_hold_ms",
    "p95_lock_hold_ms",
    "reason_prepare_lock_busy_count",
    "reason_prepare_validation_fail_count",
    "reason_prepare_timeout_count",
    "reason_prepare_other_count",
    "reason_commit_timeout_unknown_outcome_count",
    "reason_abort_best_effort_timeout_count",
    "reason_other_count",
)
SBFT_REPORT_FEATURE_FIELD_NAMES: Tuple[str, ...] = (
    "total_transactions",
    "total_consensus_instances",
    "avg_consensus_latency_ms",
    "p95_consensus_latency_ms",
    "p99_consensus_latency_ms",
    "throughput_tps",
    "timeout_violation_rate",
    "avg_batch_size",
    "p95_batch_size",
    "leader_change_count",
    "regency_change_count",
)


def _protocol_name(protocol: int) -> str:
    try:
        return agent_pb2.Protocol.Name(protocol)
    except ValueError:
        return f"UNKNOWN({protocol})"


def _validate_protocol(protocol: int) -> int:
    if protocol not in (
        agent_pb2.PROTOCOL_PBFT,
        agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT,
        agent_pb2.PROTOCOL_SBFT,
    ):
        raise ValueError(f"Unsupported protocol: {_protocol_name(protocol)}")
    return int(protocol)


def get_report_feature_field_names(protocol: int) -> Tuple[str, ...]:
    protocol = _validate_protocol(protocol)
    if protocol == agent_pb2.PROTOCOL_PBFT:
        return PBFT_REPORT_FEATURE_FIELD_NAMES
    if protocol == agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT:
        return TWO_PC_OVER_PBFT_REPORT_FEATURE_FIELD_NAMES
    return SBFT_REPORT_FEATURE_FIELD_NAMES


def get_report_feature_dim(protocol: int) -> int:
    return len(get_report_feature_field_names(protocol))


def timeout_ms_to_index(timeout_ms: int) -> int:
    try:
        return TIMEOUT_MS_TO_INDEX[timeout_ms]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported timeout {timeout_ms}ms. Valid options: {TIMEOUT_VALUES_MS}"
        ) from exc


def timeout_index_to_ms(index: int) -> int:
    if index < 0:
        return TIMEOUT_VALUES_MS[0]
    if index >= len(TIMEOUT_VALUES_MS):
        return TIMEOUT_VALUES_MS[-1]
    return TIMEOUT_VALUES_MS[index]


def _two_pc_pair_to_index(election_timeout_ms: int, prepare_timeout_ms: int) -> int:
    election_idx = timeout_ms_to_index(election_timeout_ms)
    prepare_idx = timeout_ms_to_index(prepare_timeout_ms)
    return election_idx * len(TIMEOUT_VALUES_MS) + prepare_idx


def _two_pc_index_to_pair(index: int) -> Tuple[int, int]:
    total = len(TIMEOUT_VALUES_MS) * len(TIMEOUT_VALUES_MS)
    bounded = max(0, min(index, total - 1))
    election_idx = bounded // len(TIMEOUT_VALUES_MS)
    prepare_idx = bounded % len(TIMEOUT_VALUES_MS)
    return timeout_index_to_ms(election_idx), timeout_index_to_ms(prepare_idx)


def _sbft_triple_to_index(
    election_timeout_ms: int, slow_path_timeout_ms: int, batch_timeout_ms: int
) -> int:
    election_idx = timeout_ms_to_index(election_timeout_ms)
    slow_idx = timeout_ms_to_index(slow_path_timeout_ms)
    batch_idx = timeout_ms_to_index(batch_timeout_ms)
    base = len(TIMEOUT_VALUES_MS)
    return (election_idx * base * base) + (slow_idx * base) + batch_idx


def _sbft_index_to_triple(index: int) -> Tuple[int, int, int]:
    base = len(TIMEOUT_VALUES_MS)
    total = base * base * base
    bounded = max(0, min(index, total - 1))
    election_idx = bounded // (base * base)
    rem = bounded % (base * base)
    slow_idx = rem // base
    batch_idx = rem % base
    return (
        timeout_index_to_ms(election_idx),
        timeout_index_to_ms(slow_idx),
        timeout_index_to_ms(batch_idx),
    )


def get_action_space_size(protocol: int) -> int:
    protocol = _validate_protocol(protocol)
    if protocol == agent_pb2.PROTOCOL_PBFT:
        return len(TIMEOUT_VALUES_MS)
    if protocol == agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT:
        return len(TIMEOUT_VALUES_MS) * len(TIMEOUT_VALUES_MS)
    return len(TIMEOUT_VALUES_MS) * len(TIMEOUT_VALUES_MS) * len(TIMEOUT_VALUES_MS)


def report_to_feature_values(protocol: int, state: object) -> List[float]:
    protocol = _validate_protocol(protocol)
    fields = get_report_feature_field_names(protocol)
    return [float(getattr(state, field_name)) for field_name in fields]


def timeout_to_action_index(protocol: int, timeout_used: object) -> int:
    protocol = _validate_protocol(protocol)
    if protocol == agent_pb2.PROTOCOL_PBFT:
        timeout_msg = timeout_used
        return timeout_ms_to_index(int(timeout_msg.election_timeout_milliseconds))
    if protocol == agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT:
        timeout_msg = timeout_used
        return _two_pc_pair_to_index(
            int(timeout_msg.election_timeout_milliseconds),
            int(timeout_msg.prepare_timeout_milliseconds),
        )

    timeout_msg = timeout_used
    return _sbft_triple_to_index(
        int(timeout_msg.election_timeout_milliseconds),
        int(timeout_msg.slow_path_timeout_milliseconds),
        int(timeout_msg.batch_timeout_milliseconds),
    )


def action_index_to_timeout(protocol: int, index: int) -> agent_pb2.Timeout:
    protocol = _validate_protocol(protocol)
    if protocol == agent_pb2.PROTOCOL_PBFT:
        timeout_ms = timeout_index_to_ms(index)
        return agent_pb2.Timeout(
            pbft=agent_pb2.PbftTimeout(election_timeout_milliseconds=timeout_ms)
        )
    if protocol == agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT:
        election_timeout_ms, prepare_timeout_ms = _two_pc_index_to_pair(index)
        return agent_pb2.Timeout(
            two_pc_over_pbft=agent_pb2.TwoPcOverPbftTimeout(
                election_timeout_milliseconds=election_timeout_ms,
                prepare_timeout_milliseconds=prepare_timeout_ms,
            )
        )

    election_timeout_ms, slow_path_timeout_ms, batch_timeout_ms = _sbft_index_to_triple(
        index
    )
    return agent_pb2.Timeout(
        sbft=agent_pb2.SbftTimeout(
            election_timeout_milliseconds=election_timeout_ms,
            slow_path_timeout_milliseconds=slow_path_timeout_ms,
            batch_timeout_milliseconds=batch_timeout_ms,
        )
    )


def describe_action(protocol: int, index: int) -> str:
    protocol = _validate_protocol(protocol)
    if protocol == agent_pb2.PROTOCOL_PBFT:
        return f"election_timeout_ms={timeout_index_to_ms(index)}"
    if protocol == agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT:
        election_timeout_ms, prepare_timeout_ms = _two_pc_index_to_pair(index)
        return (
            f"election_timeout_ms={election_timeout_ms}, "
            f"prepare_timeout_ms={prepare_timeout_ms}"
        )
    election_timeout_ms, slow_path_timeout_ms, batch_timeout_ms = _sbft_index_to_triple(
        index
    )
    return (
        f"election_timeout_ms={election_timeout_ms}, "
        f"slow_path_timeout_ms={slow_path_timeout_ms}, "
        f"batch_timeout_ms={batch_timeout_ms}"
    )

def produce_dataloader(features: List[agent_pb2.Report], actions: List[torch.Tensor], rewards:List[float], batch_size:int, eval_ratio: float)->Tuple['DatasetLoader', 'DatasetLoader']:
    num_samples = len(features)
    indices = list(range(num_samples))
    random.shuffle(indices)

    split_idx = int(num_samples * (1-eval_ratio))

    train_indices = indices[:split_idx]
    eval_indices = indices[split_idx:]

    train_loader = DatasetLoader(
        features=[features[i] for i in train_indices],
        actions=[actions[i] for i in train_indices],
        rewards=[rewards[i] for i in train_indices],
        batch_size=batch_size,
        shuffle=True
    )

    eval_loader = DatasetLoader(
        features=[features[i] for i in eval_indices],
        actions=[actions[i] for i in eval_indices],
        rewards=[rewards[i] for i in eval_indices],
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, eval_loader


class FeatureExtractor(Generic[S], ABC):
    """Converts an input state into a tensor for model consumption."""

    @abstractmethod
    def featurize(self, state: S) -> torch.Tensor:
        pass


class ActionDecoder(Generic[A], ABC):
    """Converts a model output tensor into a domain-specific action."""

    @abstractmethod
    def decode(self, action: torch.Tensor) -> A:
        pass


class TensorModel(ABC):
    """Model that consumes and produces tensors."""

    @abstractmethod
    def predict_action(self, features: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def update_model(
        self, features: torch.Tensor, action: torch.Tensor, reward: float
    ) -> None:
        pass

    @abstractmethod
    def reset_model(self) -> None:
        pass

class RewardTransformer(ABC):
    """Converts throughput result into float reward."""

    @abstractmethod
    def transform(self, throughput: int) -> float:
        pass

class BaseLoader(ABC):
    """Dataset Loader"""

    @abstractmethod
    def __iter__(self)-> 'BaseLoader':
        pass

    @abstractmethod
    def __next__(self) -> Dict[str, torch.Tensor]:
        pass


class ProtocolReportFeatureExtractor(FeatureExtractor[object]):
    def __init__(self, protocol: int):
        self._protocol = _validate_protocol(protocol)

    def featurize(self, state: object) -> torch.Tensor:
        features = report_to_feature_values(self._protocol, state)
        return torch.tensor(features, dtype=torch.float32)


class ProtocolTimeoutActionDecoder(ActionDecoder[agent_pb2.Timeout]):
    """Decodes model output into a protocol-specific timeout action."""

    def __init__(self, protocol: int):
        self._protocol = _validate_protocol(protocol)

    def decode(self, action: torch.Tensor) -> agent_pb2.Timeout:
        if action.numel() == 1:
            option_index = int(round(float(action.item())))
        else:
            option_index = int(torch.argmax(action).item())
        return action_index_to_timeout(self._protocol, option_index)

class QuadraticRewardScaler(RewardTransformer):
    """Quadratic shaping below 1000, linear above. Prior work"""

    @classmethod
    def transform(self,throughput: int) -> float:
        value = float(max(0, throughput))
        if value < 1000.0:
            return (value / 1000.0) ** 2 * 1000.0
        return value

class DatasetLoader(BaseLoader):
    """Training dataset loader"""
    def __init__(self, 
        features: List[agent_pb2.Report], actions:List[torch.Tensor], rewards:List[float], batch_size:Optional[int] = 1, shuffle:Optional[bool] = False
    )->None:
        self.features = features
        self.actions = actions
        self.rewards = rewards
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.all_indices = list(range(len(self.features)))

    def __iter__(self) -> 'DatasetLoader':
        self.current_indices = list(self.all_indices)

        if self.shuffle:
            random.shuffle(self.current_indices)

        self.cursor = 0
        return self
    
    def __next__(self) -> Dict[str, torch.Tensor]:
        if self.cursor >= len(self.current_indices):
            raise StopIteration
        
        batch_idx = self.current_indices[self.cursor : self.cursor + self.batch_size]
        self.cursor += self.batch_size
        
        return {
            "features": torch.tensor([report_to_feature_values(self.features[i]) for i in batch_idx], dtype=torch.float32),
            "actions": torch.tensor([self.actions[i] for i in batch_idx], dtype=torch.float32),
            "rewards": torch.tensor([self.rewards[i] for i in batch_idx], dtype=torch.float32)
        }
            

class LinearTorchModel(TensorModel):
    """Linear classifier over protocol-specific discrete timeout options."""

    def __init__(self, action_dim: int, n_features: int, lr: float = 0.01):
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")
        self._action_dim = int(action_dim)
        self.weights = torch.zeros((self._action_dim, n_features), requires_grad=True)
        self.optimizer = torch.optim.SGD([self.weights], lr=lr)

    def predict_action(self, features: torch.Tensor) -> torch.Tensor:
        return torch.mv(self.weights, features)


    def update_model(
        self, features: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, backward: bool = True
    ) -> None:
        target_index = torch.round(actions).long().view(-1)
        targets = torch.clamp(target_index, 0, len(TIMEOUT_VALUES_MS) - 1)
        logits = torch.matmul(features, self.weights.t())
        reward_scales = torch.clamp(rewards, min=0.0) / 1000.0
        loss_vector = F.cross_entropy(logits, targets, reduction='none')
        loss = (loss_vector * reward_scales.squeeze()).mean()

        if loss > 0:
            if backward:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            return loss.item() 
        
        return 0.0

    def reset_model(self) -> None:
        torch.nn.init.zeros_(self.weights)

    @classmethod
    def from_csv(
        cls, n_features:int, lr:float, path:Path, epoch: int, batch_size:Optional[int] = 1, eval_ratio:Optional[float] = 0.2, verbose:Optional[bool] = False
    )-> "LinearTorchModel":
        reports:List[agent_pb2.Report] = []
        rewards: List[float] = []
        actions: List[torch.Tensor] = []
        
        model = cls(n_features = n_features, lr = lr)
        
        with path.open("r", newline="", encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader, None)
            for lineno, row in enumerate(reader, start=1):
                if not row:
                    continue
                if len(row) != 23:
                    raise ValueError(
                        f"Invalid csv file at line {lineno}: expected 23 columns."
                    )
                try:
                    # Need to clarify usage
                    # reward_episode = int(row[0])
                    # report_node_id = int(row[1])
                    # matched_episode = int(row[22])

                    action_index = timeout_ms_to_index(int(row[20]))
                    actions.append(torch.tensor(float(action_index), dtype=torch.float32))
                    rewards.append(QuadraticRewardScaler.transform(float(row[21])))
                    reports.append(
                            agent_pb2.Report(
                                processed_transactions = int(row[2]),
                                avg_message_delay = float(row[3]),
                                max_message_delay = float(row[4]),
                                min_message_delay = float(row[5]),
                                std_message_delay = float(row[6]),
                                p95_message_delay = float(row[7]),
                                p99_message_delay = float(row[8]),
                                throughput_tps = float(row[9]),
                                timeout_violation_rate = float(row[10]),
                                no_op_rate = float(row[11]),
                                avg_batch_size = float(row[12]),
                                p95_batch_size = float(row[13]),
                                leader_change_count = int(row[14]),
                                regency_change_count = int(row[15]),
                                avg_propose_delay = float(row[16]),
                                avg_write_delay = float(row[17]),
                                avg_accept_delay = float(row[18]),
                                avg_post_decision_delay = float(row[19])
                            )
                        )
                
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid values at line {lineno}: {row}"
                    ) from exc
                
        train_loader, eval_loader = produce_dataloader(
                features = reports,
                actions = actions,
                rewards = rewards,
                batch_size = batch_size,
                eval_ratio= eval_ratio
            )

        global_loss = float('inf')       
        if verbose:
            pbar = tqdm(range(epoch), desc="Training")
        else:
            pbar = range(epoch)

        for e in pbar:
            loss,eval_loss = 0,0
            for idx, batch in enumerate(train_loader):
                loss += model.update_model(batch["features"], batch["actions"], batch["rewards"])
            current_loss = loss / (idx+1)

            for idx, batch in enumerate(eval_loader):
                eval_loss += model.update_model(batch["features"], batch["actions"], batch["rewards"], backward=False)
            current_eval_loss = eval_loss / (idx+1)

            if current_eval_loss < global_loss:
                global_loss = current_eval_loss
                best_weights = model.weights.clone().detach()
                if verbose:
                    pbar.set_postfix(Epoch = f"{e+1}/{epoch}", Train_Loss=f"{current_loss:.7f}", Eval_Loss = f"{current_eval_loss:.7f}")

        with torch.no_grad():
            model.weights.copy_(best_weights)

        return model
    
class ScheduledTimeoutModel(TensorModel):
    """Returns protocol-specific timeout option indices from a time-based schedule."""

    def __init__(
        self,
        protocol: int,
        schedule: List[Tuple[float, int]],
        start_monotonic: Optional[float] = None,
    ):
        self._protocol = _validate_protocol(protocol)
        if not schedule:
            raise ValueError("Schedule must contain at least one entry.")
        self._schedule = sorted(schedule, key=lambda entry: entry[0])
        if self._schedule[0][0] < 0:
            raise ValueError("Schedule times must be >= 0 seconds.")
        self._start_time = (
            time.monotonic() if start_monotonic is None else float(start_monotonic)
        )
        self._active_row = -1

    @classmethod
    def from_csv(
        cls,
        protocol: int,
        path: Path,
        start_monotonic: Optional[float] = None,
    ) -> "ScheduledTimeoutModel":
        protocol = _validate_protocol(protocol)
        schedule: List[Tuple[float, int]] = []
        with path.open("r", newline="") as handle:
            reader = csv.reader(handle)
            for lineno, row in enumerate(reader, start=1):
                if not row:
                    continue
                first = row[0].strip()
                if not first or first.startswith("#"):
                    continue
                try:
                    time_s = float(first)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid schedule time at line {lineno}: {row}"
                    ) from exc
                if time_s < 0:
                    raise ValueError(
                        f"Invalid schedule at line {lineno}: time_s must be >= 0."
                    )

                if protocol == agent_pb2.PROTOCOL_PBFT:
                    if len(row) < 2:
                        raise ValueError(
                            f"Invalid PBFT schedule at line {lineno}: expected 2 columns time_s,election_timeout_ms."
                        )
                    timeout_ms = int(row[1].strip())
                    option_index = timeout_ms_to_index(timeout_ms)
                elif protocol == agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT:
                    if len(row) < 3:
                        raise ValueError(
                            "Invalid 2PC-over-PBFT schedule at line "
                            f"{lineno}: expected 3 columns "
                            "time_s,election_timeout_ms,prepare_timeout_ms."
                        )
                    election_timeout_ms = int(row[1].strip())
                    prepare_timeout_ms = int(row[2].strip())
                    option_index = _two_pc_pair_to_index(
                        election_timeout_ms, prepare_timeout_ms
                    )
                else:
                    if len(row) < 4:
                        raise ValueError(
                            "Invalid SBFT schedule at line "
                            f"{lineno}: expected 4 columns "
                            "time_s,election_timeout_ms,slow_path_timeout_ms,batch_timeout_ms."
                        )
                    election_timeout_ms = int(row[1].strip())
                    slow_path_timeout_ms = int(row[2].strip())
                    batch_timeout_ms = int(row[3].strip())
                    option_index = _sbft_triple_to_index(
                        election_timeout_ms,
                        slow_path_timeout_ms,
                        batch_timeout_ms,
                    )
                schedule.append((time_s, option_index))
        if not schedule:
            raise ValueError(f"No schedule entries found in {path}.")
        return cls(protocol, schedule, start_monotonic=start_monotonic)

    def set_start_monotonic(self, start_monotonic: float) -> None:
        self._start_time = float(start_monotonic)
        self._active_row = -1

    def _current_option_index(self) -> int:
        elapsed_s = time.monotonic() - self._start_time
        active_row = 0
        option_index = self._schedule[0][1]
        for row_idx, (switch_time_s, schedule_option_index) in enumerate(self._schedule):
            if elapsed_s >= switch_time_s:
                active_row = row_idx
                option_index = schedule_option_index
            else:
                break
        if active_row != self._active_row:
            switch_time_s, active_option_index = self._schedule[active_row]
            print(
                "[scheduled-model] active schedule: "
                f"time_s>={switch_time_s:g}, "
                f"{describe_action(self._protocol, active_option_index)}"
            )
            self._active_row = active_row
        return option_index

    def predict_action(self, features: torch.Tensor) -> torch.Tensor:
        del features
        option_index = self._current_option_index()
        return torch.tensor(float(option_index), dtype=torch.float32)

    def update_model(
        self, features: torch.Tensor, action: torch.Tensor, reward: float
    ) -> None:
        del features, action, reward

    def reset_model(self) -> None:
        self._start_time = time.monotonic()
        self._active_row = -1
        
if __name__ == "__main__":
    random.seed(42) 
    model = LinearTorchModel.from_csv(
        n_features = REPORT_FEATURE_DIM, 
        lr = 1e-9,
        path = Path('data/sample_large.csv'), 
        epoch = 30, 
        batch_size = 4,
        verbose = True,
        eval_ratio = 0.2
    )
