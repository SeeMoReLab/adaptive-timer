import os
import random
from abc import ABC, abstractmethod
from typing import  List, Optional, Tuple
import argparse

import numpy as np
import pandas as pd
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
    def __init__(self, protocol: str, directory_path: str, num_episode_each_phase:int = 1000) -> None:
        self.directory_path = directory_path
        self.df: Optional[pd.DataFrame] = None
        self.protocol = protocol
        self.num_episodes_per_phase = num_episode_each_phase

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
      
    def generate_phase(self, phase_index: int = 0, current_start_episode: int = 0, verbose: bool = False)-> csv.DictReader:
        if self.df is None or self.df.empty:
            print("No data available. Please run read_data() first.")
            return None

        grouped_df = self.df[self.df['phase'] == phase_index]

        conf = {
                timeout: group['reward'].dropna().values
                for timeout, group in grouped_df.groupby('election_timeout_milliseconds_used')
            }
        
        conf_stats = grouped_df.groupby('election_timeout_milliseconds_used')['reward'].agg(
                ['mean', 'std', 'median', 'var', 'min', 'max']
        ).to_dict(orient='index')
        
        unique_nodes = grouped_df['report_node_id'].dropna().unique()
        
        res = {}

        feature_cols = [c for c in self.col_name if c not in STRUCTURAL_COLUMNS]
        original_phase_data = grouped_df[feature_cols]

        synthetic_episodes = np.arange(current_start_episode, current_start_episode + self.num_episodes_per_phase)
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

        available_timeouts = list(conf.keys())

        if current_start_episode ==0:
            trial['election_timeout_milliseconds_used'] = 300
        else:
            trial['election_timeout_milliseconds_used'] = trial['election_timeout_milliseconds_used'].apply(
                lambda t: min(available_timeouts, key=lambda x: abs(x - t))
            )

        rewards = np.empty(total_rows)
        for timeout_val, group_idx in trial.groupby('election_timeout_milliseconds_used').groups.items():
            n = len(group_idx)
            rewards[group_idx] = self._sample_reward_by_timeout(timeout_val, conf_stats, n)

        trial['reward'] = rewards
        res = trial
        res = res.assign(phase=phase_index)

        csv_buffer = io.StringIO()

        res.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        reader = csv.DictReader(csv_buffer)

        
        current_start_episode += self.num_episodes_per_phase
            
        return reader, current_start_episode

    def _generate_kde(self, original_df: pd.DataFrame, num_samples: int, ordinal_cols: List[str]) -> pd.DataFrame:
        """
        Multivariate KDE on continuous columns
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
                res[col] = np.clip(samples, orig_vals.min(), orig_vals.max())

        for col in ordinal_cols_present:
            orig_vals = df_clean[col].dropna().values
            res[col] = np.random.choice(orig_vals, size=num_samples, replace=True)

        return pd.DataFrame(res)
    
    def _sample_reward_by_timeout(
            self, timeout_val: float, conf_stats: dict, n_samples: int, temperature:float = 1.5
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


        # Add randomness if sample data small
        if pd.isna(std) or std == 0:
            jitter = abs(mean * 0.05) 
            jitter = jitter if jitter > 0 else 0.1 
            return np.random.uniform(mean - jitter, mean + jitter, size=n_samples)

        # Generate Normal/Gaussian distribution
        sampled_rewards = np.random.normal(loc=mean, scale=(std * temperature), size=n_samples)

        # Inject noise
        anomaly_mask = np.random.rand(n_samples) < 0.1
        if np.any(anomaly_mask):
            num_anomalies = np.sum(anomaly_mask)
            random_spikes = np.random.uniform(min_val, max_val, size=num_anomalies)
            sampled_rewards[anomaly_mask] = random_spikes

        return sampled_rewards
    
