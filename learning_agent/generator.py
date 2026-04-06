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

PBFT_FEATURE_FIELD: Tuple[str, ...] = (
    "report_node_id",
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
    "election_timeout_milliseconds_used",
    'matched_episode_features')

PBFT_REPORT_ORDINAL_COLUMNS: List[str] = [
    'total_transactions',
    'total_consensus_instances',
    'election_timeout_milliseconds_used',
    'matched_episode_features'
]

STRUCTURAL_COLUMNS: List[str] = ['reward_episode', 'report_node_id']

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class BaseDataGenerator(ABC):
    
    @abstractmethod
    def _read_data(self) -> Optional[pd.DataFrame]:
        pass

    @abstractmethod
    def generate_phase(self) -> Optional[pd.DataFrame]:
        pass


class DataGenerator(BaseDataGenerator):
    def __init__(self, protocol: str, directory_path: str) -> None:
        self.directory_path = directory_path
        self.df: Optional[pd.DataFrame] = None
        self.protocol = protocol
        self.reward_models = {}
        self.reward_residuals = {}
        
        if protocol.lower() == 'pbft':
            self.col_name = PBFT_FEATURE_FIELD
            self.col_ordinal = PBFT_REPORT_ORDINAL_COLUMNS
        else:
            raise ValueError("Unknown protocol! Supported: pbft")
        
        self._read_data()
    
    def _read_data(self) -> Optional[pd.DataFrame]:
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
    
    def generate_phase(self, phase_index: int = 0, current_start_episode: int = 0, verbose: bool = False, num_episode:int = 1, prev_timeout: Optional[List[int]] = None )-> csv.DictReader:
        if self.df is None or self.df.empty:
            print("No data available. Please run read_data() first.")
            return None

        grouped_df = self.df[self.df['phase'] == phase_index]
        
        conf_stats = grouped_df.groupby('election_timeout_milliseconds_used')['reward'].agg(
                ['mean', 'std', 'median', 'var', 'min', 'max']
        ).to_dict(orient='index')
        
        unique_nodes = grouped_df['report_node_id'].dropna().unique()
        
        res = {}

        feature_cols = [c for c in self.col_name if c not in STRUCTURAL_COLUMNS]
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
            ordinal_cols=self.col_ordinal
        )
        
        for col in feature_cols:
            trial[col] = synthetic_features[col].values

        if prev_timeout != None:
            trial['election_timeout_milliseconds_used'] = prev_timeout
        else:
            trial['election_timeout_milliseconds_used'] = 150

        trial['reward'] = 0.0 

        for idx, row in trial.iterrows():
            timeout_val = row['election_timeout_milliseconds_used']
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

     