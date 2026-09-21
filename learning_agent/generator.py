import os
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde
import csv
import io

import agent_pb2
from models import (
    autobahn_fast_path_timeout_index_to_ms,
    autobahn_timeout_delay_ms_to_index,
    get_report_feature_field_names,
    get_timeout_values_ms,
    tendermint_timeouts_from_propose,
)
from reward_utils import calculate_reward_from_report_row


PROTOCOL_NAME_TO_ENUM = {
    "pbft": agent_pb2.PROTOCOL_PBFT,
    "sbft": agent_pb2.PROTOCOL_SBFT,
    "tendermint": agent_pb2.PROTOCOL_TENDERMINT,
    "hotstuff": agent_pb2.PROTOCOL_HOTSTUFF,
    "autobahn": agent_pb2.PROTOCOL_AUTOBAHN,
    "smartbft": agent_pb2.PROTOCOL_SMARTBFT,
}

SIMULATED_REWARD_COLUMN = "simulated_reward"
CALCULATED_REWARD_COLUMN = "_calculated_reward"


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class BaseDataGenerator(ABC):
    
    @abstractmethod
    def read_data(self) -> Optional[pd.DataFrame]:
        pass

    @abstractmethod
    def generate_local_report(self) -> Optional[pd.DataFrame]:
        pass

class DataGenerator(BaseDataGenerator):
    def __init__(self, protocol: str, directory_path: str) -> None:
        self.directory_path = directory_path
        self.df: Optional[pd.DataFrame] = None
        self.protocol = protocol.strip().lower().replace("-", "_")
        try:
            self.protocol_id = PROTOCOL_NAME_TO_ENUM[self.protocol]
        except KeyError as exc:
            raise ValueError(f"Unsupported protocol: {protocol}") from exc
        self.reward_models = {}
        self.reward_residuals = {}

        self.feature_field_names = get_report_feature_field_names(self.protocol_id)
        self.feature_columns = tuple(
            f"feature_{field_name}" for field_name in self.feature_field_names
        )
        self.ordinal_columns = tuple(
            column
            for column in self.feature_columns
            if column.endswith("_count")
            or column
            in (
                "feature_total_transactions",
                "feature_total_consensus_instances",
            )
        )
        self.timeout_values_ms = get_timeout_values_ms(self.protocol_id)
        self.primary_timeout_column = self._primary_timeout_column()
        self.phase_options: Dict[int, int] = {}

        self.read_data()

    def read_data(self):
        csv_files = []
        with os.scandir(path=self.directory_path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith('.csv'):
                    csv_files.append(entry.path) 

        if not csv_files:
            print("No CSV files found in the current directory.")
            return None

        dataframes = [pd.read_csv(file) for file in sorted(csv_files)]
        self.df = pd.concat(dataframes, ignore_index=True)

        required_columns = {
            "phase",
            "report_node_id",
            self.primary_timeout_column,
            *self.feature_columns,
        }
        missing = sorted(required_columns.difference(self.df.columns))
        if missing:
            raise ValueError(
                f"{self.protocol} simulator data is missing columns: "
                f"{', '.join(missing)}"
            )

        self.df[CALCULATED_REWARD_COLUMN] = [
            calculate_reward_from_report_row(
                self.protocol_id, row.to_dict(), row_number=index + 2
            )
            for index, row in self.df.iterrows()
        ]
        self.phase_options = self._derive_phase_options()

        print(f"Successfully read {len(csv_files)} files!")
        return self.df

    def _primary_timeout_column(self) -> str:
        if self.protocol_id == agent_pb2.PROTOCOL_TENDERMINT:
            return "propose_timeout_milliseconds_used"
        if self.protocol_id in (
            agent_pb2.PROTOCOL_HOTSTUFF,
            agent_pb2.PROTOCOL_AUTOBAHN,
        ):
            return "timeout_delay_milliseconds_used"
        return "election_timeout_milliseconds_used"

    def _derive_phase_options(self) -> Dict[int, int]:
        if self.df is None:
            return {}
        means = self.df.groupby(
            ["phase", self.primary_timeout_column]
        )[CALCULATED_REWARD_COLUMN].mean()
        options = {}
        for phase, phase_means in means.groupby(level=0):
            by_timeout = phase_means.droplevel(0)
            best_timeout = int(by_timeout.idxmax())
            try:
                options[int(phase)] = self.timeout_values_ms.index(best_timeout)
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported {self.protocol} timeout {best_timeout}ms "
                    f"in phase {int(phase)}"
                ) from exc
        return options
    
    def generate_local_report(
        self,
        phase_index: int = 0,
        current_start_episode: int = 0,
        verbose: bool = False,
        num_episode: int = 1,
        prev_timeout: Optional[List[int]] = None,
    ) -> Tuple[csv.DictReader, int]:
        if self.df is None or self.df.empty:
            print("No data available. Please run read_data() first.")
            return None

        grouped_df = self.df[self.df['phase'] == phase_index]
        if grouped_df.empty:
            raise ValueError(
                f"No completed {self.protocol} episode rows found for phase "
                f"{phase_index}."
            )
        
        conf_stats = (
            grouped_df.groupby(self.primary_timeout_column)[CALCULATED_REWARD_COLUMN]
            .agg(["mean", "std", "median", "var", "min", "max"])
            .to_dict(orient="index")
        )
        
        unique_nodes = grouped_df['report_node_id'].dropna().unique()
        
        feature_cols = list(self.feature_columns)
        original_phase_data = grouped_df[feature_cols]

        synthetic_episodes = np.arange(
            current_start_episode, current_start_episode + num_episode
        )
        skeleton_index = pd.MultiIndex.from_product(
            [synthetic_episodes, unique_nodes],
            names=["episode", "report_node_id"],
        )
        trial = pd.DataFrame(index=skeleton_index).reset_index()
        total_rows = len(trial)

        if verbose:
            print(f"Phase {phase_index}: Generating {total_rows} rows...")

        synthetic_features = self._generate_kde(
            original_df=original_phase_data,
            num_samples=total_rows,
            ordinal_cols=list(self.ordinal_columns)
        )
        
        for col in feature_cols:
            trial[col] = synthetic_features[col].values

        selected_timeouts = self._selected_timeouts(prev_timeout, total_rows)
        self._set_timeout_columns(trial, selected_timeouts)

        trial[SIMULATED_REWARD_COLUMN] = 0.0

        for idx, row in trial.iterrows():
            timeout_val = row[self.primary_timeout_column]
            single_reward = self._sample_reward_by_timeout(
                phase_index, timeout_val, conf_stats, n_samples=1
            )[0]
            trial.at[idx, SIMULATED_REWARD_COLUMN] = single_reward

        res = trial.assign(phase=phase_index)

        csv_buffer = io.StringIO()

        res.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        reader = csv.DictReader(csv_buffer)

        current_start_episode += num_episode
        return reader, current_start_episode

    def _selected_timeouts(
        self, prev_timeout: Optional[Sequence[int]], total_rows: int
    ) -> np.ndarray:
        if prev_timeout is None:
            return np.full(total_rows, self.timeout_values_ms[0], dtype=int)
        selected = np.asarray(prev_timeout, dtype=int).reshape(-1)
        if selected.size == 1:
            return np.full(total_rows, int(selected[0]), dtype=int)
        if selected.size == total_rows:
            return selected
        if total_rows % selected.size == 0:
            return np.tile(selected, total_rows // selected.size)
        raise ValueError(
            f"Cannot assign {selected.size} predicted timeouts to "
            f"{total_rows} generated rows."
        )

    def _set_timeout_columns(
        self, trial: pd.DataFrame, selected_timeouts: Sequence[int]
    ) -> None:
        if self.protocol_id in (agent_pb2.PROTOCOL_PBFT, agent_pb2.PROTOCOL_SMARTBFT):
            trial["election_timeout_milliseconds_used"] = selected_timeouts
            return
        if self.protocol_id == agent_pb2.PROTOCOL_SBFT:
            trial["election_timeout_milliseconds_used"] = selected_timeouts
            trial["slow_path_timeout_milliseconds_used"] = selected_timeouts
            return
        if self.protocol_id == agent_pb2.PROTOCOL_HOTSTUFF:
            trial["timeout_delay_milliseconds_used"] = selected_timeouts
            return
        if self.protocol_id == agent_pb2.PROTOCOL_AUTOBAHN:
            # The knobs have distinct value sets, so the SBFT convention of
            # copying the selected timeout into every column would produce
            # invalid actions; fast-path instead follows the selected timeout
            # delay at the same option index.
            delay_indices = [
                autobahn_timeout_delay_ms_to_index(int(timeout))
                for timeout in selected_timeouts
            ]
            trial["timeout_delay_milliseconds_used"] = selected_timeouts
            trial["fast_path_timeout_milliseconds_used"] = [
                autobahn_fast_path_timeout_index_to_ms(index)
                for index in delay_indices
            ]
            return

        timeout_triples = [
            tendermint_timeouts_from_propose(int(timeout))
            for timeout in selected_timeouts
        ]
        trial["propose_timeout_milliseconds_used"] = [
            triple[0] for triple in timeout_triples
        ]
        trial["prevote_timeout_milliseconds_used"] = [
            triple[1] for triple in timeout_triples
        ]
        trial["precommit_timeout_milliseconds_used"] = [
            triple[2] for triple in timeout_triples
        ]

    def _generate_kde(
        self, 
        original_df: pd.DataFrame, 
        num_samples: int, 
        ordinal_cols: List[str],
        noise_scale: float = 0.
    ) -> pd.DataFrame:
        """
        Multivariate KDE on continuous columns with adjustable Gaussian noise.
        """
        df_clean = original_df.dropna(axis=1, how='all')
        if df_clean.empty:
            return pd.DataFrame()

        continuous_cols = [
            c for c in df_clean.columns
            if c not in ordinal_cols and len(np.unique(df_clean[c].dropna())) > 1
        ]
        ordinal_cols_present = [
            c for c in df_clean.columns
            if c in ordinal_cols or len(np.unique(df_clean[c].dropna())) <= 1
        ]

        res = {}
        if continuous_cols:
            for col in continuous_cols:
                orig_vals = df_clean[col].dropna().values
                kde = gaussian_kde(orig_vals)
                samples = kde.resample(size=num_samples)[0]
                
                if noise_scale > 0:
                    col_std = np.std(orig_vals)
                    noise = np.random.normal(loc=0.0, scale=noise_scale * col_std, size=num_samples)
                    samples += noise

                res[col] = np.clip(samples, orig_vals.min(), orig_vals.max())

        for col in ordinal_cols_present:
            orig_vals = df_clean[col].dropna().values
            res[col] = np.random.choice(orig_vals, size=num_samples, replace=True)

        return pd.DataFrame(res)
    
    def _sample_reward_by_timeout(
            self, idx, timeout_val: float, conf_stats: dict, n_samples: int, temperature:float = 0.5
            ) -> np.ndarray:
        """
        Generates rewards using a Normal distribution based on historical mean and std, 
        """
        if n_samples == 0:
            return np.array([])

        stats = conf_stats.get(timeout_val, {})
        mean = stats.get('mean', 0.0)
        std = stats.get('std', 0.0)
        min_val = stats.get('min', 0.0)
        max_val = stats.get('max', 0.0)

        if pd.isna(std) or std == 0:
            jitter = abs(mean * temperature) 
            return np.random.uniform(mean - jitter, mean + jitter, size=n_samples)
        else:
            sampled_rewards = np.random.normal(
                loc=mean, 
                scale=(std * temperature), 
                size=n_samples
            )
            min_val = stats.get('min', 0.0)
            max_val = stats.get('max', 1.0) 
            sampled_rewards = np.clip(sampled_rewards, min_val, max_val)
            return sampled_rewards

     
