import os
import random
from abc import ABC, abstractmethod
from typing import  List, Optional, Tuple
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde
import csv
import io

from models import PBFT_REPORT_FEATURE_FIELD_NAMES, TWO_PC_OVER_PBFT_REPORT_FEATURE_FIELD_NAMES

LOCAL_PBFT_COLUMNS: Tuple[str, ...] = (
    "report_node_id",
    "election_timeout_milliseconds_used",
    'matched_episode_features'
)

LOCAL_2PC_OVER_PBFT_COLUMNS: Tuple[str, ...] = (
    "report_node_id",
    "participant_shard_id",
    "election_timeout_milliseconds_used",
    "prepare_timeout_milliseconds_used",
    "matched_episode_features",
)

REPORT_ORDINAL_COLUMNS_PBFT: Tuple[str, ...]= [
    'total_transactions',
    'total_consensus_instances',
    'matched_episode_features',
    'leader_change_count', 
    'regency_change_count'
]

REPORT_ORDINAL_COLUMNS_2PC_OVER_PBFT: Tuple[str, ...]= [
    "reason_prepare_lock_busy_count",
    "reason_prepare_validation_fail_count",
    "reason_prepare_timeout_count",
    "reason_prepare_other_count",
    "reason_commit_timeout_unknown_outcome_count",
    "reason_abort_best_effort_timeout_count",
    "reason_other_count",
]

STRUCTURAL_COLUMNS_PBFT: List[str] = ['reward_episode', 'report_node_id']

STRUCTURAL_COLUMNS_2PC_OVER_PBFT: List[str] = ['reward_episode', 'participant_shard_id']

# Per-node params: faults (proposal/network delay) hit nodes unequally, so each
# report_node_id gets its own {phase: (p_fail, jitter)}. Applied by node when the
# node id is known; PHASE_NOISE_PARAMS (pooled over nodes) is the fallback for
# unknown nodes / other protocols.
PHASE_NOISE_PARAMS_BY_NODE = {
    1: {0: (0.03, 0.025), 1: (0.08, 0.055), 2: (0.11, 0.060), 3: (0.28, 0.296), 4: (0.25, 0.201), 5: (0.21, 0.350), 6: (0.29, 0.350), 7: (0.28, 0.350), 8: (0.00, 0.004)},
    2: {0: (0.01, 0.028), 1: (0.12, 0.018), 2: (0.07, 0.089), 3: (0.31, 0.346), 4: (0.31, 0.198), 5: (0.36, 0.254), 6: (0.23, 0.350), 7: (0.38, 0.350), 8: (0.00, 0.004)},
    3: {0: (0.01, 0.026), 1: (0.12, 0.026), 2: (0.18, 0.075), 3: (0.20, 0.350), 4: (0.40, 0.148), 5: (0.25, 0.350), 6: (0.10, 0.350), 7: (0.31, 0.350), 8: (0.00, 0.005)},
    4: {0: (0.01, 0.027), 1: (0.17, 0.014), 2: (0.10, 0.018), 3: (0.35, 0.329), 4: (0.30, 0.309), 5: (0.38, 0.350), 6: (0.35, 0.346), 7: (0.29, 0.350), 8: (0.00, 0.005)},
}
# PHASE_NOISE_PARAMS = {
#     0: (0.02, 0.032),
#     1: (0.13, 0.022),
#     2: (0.11, 0.074),
#     3: (0.29, 0.329),
#     4: (0.31, 0.261),
#     5: (0.32, 0.350),
#     6: (0.30, 0.350),
#     7: (0.30, 0.350),
#     8: (0.00, 0.005),
# }
# DEFAULT_NOISE_PARAMS = (0.25, 0.30)
FAIL_RATIO_MAX = 0.45


def _matched_rows(df: pd.DataFrame) -> pd.DataFrame:
    if 'matched_episode_features' not in df.columns:
        return df
    matched = df['matched_episode_features'].astype(str).str.strip().str.lower()
    return df[matched.isin({'1', 'true', 'yes', 'y'})]


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

    @abstractmethod
    def generate_tpc_report(self) -> Optional[pd.DataFrame]:
        pass


class DataGenerator(BaseDataGenerator):
    def __init__(self, protocol: str, directory_path: str, reward_noise: bool = True) -> None:
        self.directory_path = directory_path
        self.df: Optional[pd.DataFrame] = None
        self.protocol = protocol
        self.reward_models = {}
        self.reward_residuals = {}
        self.reward_noise = reward_noise

        self.local_col_name = LOCAL_PBFT_COLUMNS + PBFT_REPORT_FEATURE_FIELD_NAMES
        self.local_col_ordinal = REPORT_ORDINAL_COLUMNS_PBFT
        self.local_structure = STRUCTURAL_COLUMNS_PBFT
        self.protocol =protocol.lower()

        if self.protocol == 'two_pc_over_pbft':
            self.tpc_col_name = LOCAL_2PC_OVER_PBFT_COLUMNS + TWO_PC_OVER_PBFT_REPORT_FEATURE_FIELD_NAMES
            self.tpc_col_ordinal = REPORT_ORDINAL_COLUMNS_2PC_OVER_PBFT
            self.tpc_structure = STRUCTURAL_COLUMNS_2PC_OVER_PBFT

        
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

        dataframes = [pd.read_csv(file) for file in csv_files]
        self.df = pd.concat(dataframes, ignore_index=True)

        print(f"Successfully read {len(csv_files)} files!")
        return self.df
    
    def generate_local_report(self, phase_index: int = 0, current_start_episode: int = 0, verbose: bool = False, num_episode:int = 1, prev_timeout: Optional[List[int]] = None )-> csv.DictReader:
        if self.df is None or self.df.empty:
            print("No data available. Please run read_data() first.")
            return None

        grouped_df = self.df[self.df['phase'] == phase_index]
        grouped_df = _matched_rows(grouped_df)
        if grouped_df.empty:
            raise ValueError(
                f"No matched PBFT rows found for phase {phase_index}. "
                "The simulator requires rows with matched_episode_features=1."
            )
        
        conf_stats = grouped_df.groupby('election_timeout_milliseconds_used')['reward'].agg(
                ['mean', 'std', 'median', 'var', 'min', 'max']
        ).to_dict(orient='index')
        
        unique_nodes = grouped_df['report_node_id'].dropna().unique()
        
        res = {}

        feature_cols = [c for c in self.local_col_name if c not in self.local_structure]
        original_phase_data = grouped_df[feature_cols]

        synthetic_episodes = np.arange(current_start_episode, current_start_episode + num_episode)
        skeleton_index = pd.MultiIndex.from_product(
            [synthetic_episodes, unique_nodes],
            names=['reward_episode', 'report_node_id']
        )
        trial = pd.DataFrame(index=skeleton_index).reset_index()
        total_rows = len(trial)

        if verbose:
            print(f"Phase {phase_index}: Generating {total_rows} rows...")

        synthetic_features = self._generate_kde(
            original_df=original_phase_data,
            num_samples=total_rows,
            ordinal_cols=self.local_col_ordinal
        )
        
        for col in feature_cols:
            trial[col] = synthetic_features[col].values

        if prev_timeout != None:
            trial['election_timeout_milliseconds_used'] = prev_timeout
        else:
            trial['election_timeout_milliseconds_used'] = 300

        trial['reward'] = 0.0

        for idx, row in trial.iterrows():
            timeout_val = row['election_timeout_milliseconds_used']
            single_reward = self._sample_reward_by_timeout(
                phase_index, timeout_val, conf_stats, n_samples=1,
                node_id=row['report_node_id'],
            )[0]
            trial.at[idx, 'reward'] = single_reward

        res = trial.assign(phase=phase_index)

        csv_buffer = io.StringIO()

        res.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        reader = csv.DictReader(csv_buffer)

        current_start_episode += num_episode
        return reader, current_start_episode

    def generate_tpc_report(self, phase_index: int = 0, current_start_episode: int = 0, verbose: bool = False, num_episode:int = 1, prev_timeout: Optional[List[int]] = None )-> csv.DictReader:
        if self.df is None or self.df.empty:
            print("No data available. Please run read_data() first.")
            return None

        grouped_df = self.df[self.df['phase'] == phase_index]
        
        grouped_df = _matched_rows(grouped_df)
        conf_stats = grouped_df.groupby('prepare_timeout_milliseconds_used')['reward'].agg(
                ['mean', 'std', 'median', 'var', 'min', 'max']
        ).to_dict(orient='index')
        
        unique_nodes = grouped_df['participant_shard_id'].dropna().unique()
        
        res = {}

        feature_cols = [c for c in self.tpc_col_name if c not in self.tpc_structure]
        original_phase_data = grouped_df[feature_cols]

        synthetic_episodes = np.arange(current_start_episode, current_start_episode + num_episode)
        skeleton_index = pd.MultiIndex.from_product(
            [synthetic_episodes, unique_nodes],
            names=['reward_episode', 'participant_shard_id']
        )
        trial = pd.DataFrame(index=skeleton_index).reset_index()
        total_rows = len(trial)

        if verbose:
            print(f"Phase {phase_index}: Generating {total_rows} rows...")

        synthetic_features = self._generate_kde(
            original_df=original_phase_data,
            num_samples=total_rows,
            ordinal_cols=self.local_col_ordinal
        )
        
        for col in feature_cols:
            trial[col] = synthetic_features[col].values

        if prev_timeout != None:
            trial['prepare_timeout_milliseconds_used'] = prev_timeout
        else:
            trial['prepare_timeout_milliseconds_used'] = 150

        trial['reward'] = 0.0

        for idx, row in trial.iterrows():
            timeout_val = row['prepare_timeout_milliseconds_used']
            single_reward = self._sample_reward_by_timeout(phase_index,timeout_val, conf_stats, n_samples=1)[0]
            trial.at[idx, 'reward'] = single_reward

        res = trial.assign(phase=phase_index)

        csv_buffer = io.StringIO()

        res.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        reader = csv.DictReader(csv_buffer)

        current_start_episode += num_episode
        return reader, current_start_episode
    
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
    
    def _apply_deployment_noise(self, rewards: np.ndarray, phase_index: int, node_id=None) -> np.ndarray:
        """
        Deployment noise: with probability p_fail the episode degrades (reward
        drops to Uniform(0, 0.45) x its value), otherwise relative Gaussian
        jitter is applied. Uses the calling node's per-node params when known,
        falling back to the pooled per-phase params.
        """
        if not self.reward_noise or rewards.size == 0:
            return rewards

        params = None
        if node_id is not None:
            node_map = PHASE_NOISE_PARAMS_BY_NODE.get(int(node_id))
            if node_map is not None:
                params = node_map.get(int(phase_index))
        if params is None:
            params = PHASE_NOISE_PARAMS.get(int(phase_index), DEFAULT_NOISE_PARAMS)
        p_fail, jitter = params

        n = rewards.shape[0]
        noisy = rewards * (1.0 + np.random.normal(0.0, jitter, size=n))
        failed = np.random.random(n) < p_fail
        noisy[failed] = rewards[failed] * np.random.uniform(0.0, FAIL_RATIO_MAX, size=int(failed.sum()))
        return np.maximum(noisy, 0.0)

    def _sample_reward_by_timeout(
            self, idx, timeout_val: float, conf_stats: dict, n_samples: int, temperature:float = 0.5, node_id=None
            ) -> np.ndarray:
        """
        Generates rewards using a Normal distribution based on historical mean and std,
        then applies deployment-calibrated noise so the oracle arm is not always best.
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
            sampled_rewards = np.random.uniform(mean - jitter, mean + jitter, size=n_samples)
        else:
            sampled_rewards = np.random.normal(
                loc=mean,
                scale=(std * temperature),
                size=n_samples
            )
            min_val = stats.get('min', 0.0)
            max_val = stats.get('max', 1.0)
            sampled_rewards = np.clip(sampled_rewards, min_val, max_val)

        return self._apply_deployment_noise(sampled_rewards, idx, node_id=node_id)

     
