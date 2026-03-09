import os
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, TypeVar
import argparse

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

PBFT_FEATURE_FIELD: Tuple[str, ...] = (
    "report_node_id",
    "phase",
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
    def read_data(self, save: bool = False, filename: str = "original.csv") -> Optional[pd.DataFrame]:
        pass

    @abstractmethod
    def generate_data(self, save: bool = False, filename: str = "synthetic.csv") -> Optional[pd.DataFrame]:
        pass

    @abstractmethod
    def save(self, df: pd.DataFrame, filename: str) -> None:
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
    
    def read_data(self, save: bool = False, filename: str = "original.csv") -> Optional[pd.DataFrame]:
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
        
        if save:
            self.save(self.df, f'{self.protocol}_{filename}')

        print(f"Successfully read {len(csv_files)} files!")
        return self.df
        
    def generate_data(self, save: bool = False, filename: str = "synthetic.csv") -> Optional[pd.DataFrame]:
        if self.df is None or self.df.empty:
            print("No data available. Please run read_data() first.")
            return None

        phase_df = {}
        for phase_name, phase_data in self.df.groupby('phase'):
            phase_df[phase_name] = phase_data
            print(f"Phase '{phase_name}' has {len(phase_data)} rows.")

        conf = {}
        conf_stats = {}

        for key, items in phase_df.items():
            conf[key] = {
                timeout: group['reward'].dropna().values
                for timeout, group in items.groupby('election_timeout_milliseconds_used')
            }
            conf_stats[key] = items.groupby('election_timeout_milliseconds_used')['reward'].agg(
                ['mean', 'std', 'median', 'var', 'min', 'max']
            ).to_dict()
        
        first_phase_key = list(phase_df.keys())[0]
        unique_nodes = phase_df[first_phase_key]['report_node_id'].dropna().unique()
        
        res = {}
        current_start_episode = 1

        for key in sorted(phase_df.keys()):
            feature_cols = [c for c in self.col_name if c not in STRUCTURAL_COLUMNS]
            original_phase_data = phase_df[key][feature_cols]

            synthetic_episodes = np.arange(current_start_episode, current_start_episode + self.num_episodes_per_phase)
            skeleton_index = pd.MultiIndex.from_product(
                [synthetic_episodes, unique_nodes],
                names=['reward_episode', 'report_node_id']
            )
            trial = pd.DataFrame(index=skeleton_index).reset_index()
            total_rows = len(trial)

            print(f"Phase {key}: Generating {total_rows} rows...")

            synthetic_features = self._generate_kde(
                original_df=original_phase_data,
                num_samples=total_rows,
                ordinal_cols=self.col_ordinal
            )
            
            for col in feature_cols:
                trial[col] = synthetic_features[col].values

            available_timeouts = list(conf[key].keys())
            trial['election_timeout_milliseconds_used'] = trial['election_timeout_milliseconds_used'].apply(
                lambda t: min(available_timeouts, key=lambda x: abs(x - t))
            )

            rewards = np.empty(total_rows)
            for timeout_val, group_idx in trial.groupby('election_timeout_milliseconds_used').groups.items():
                n = len(group_idx)
                rewards[group_idx] = self._sample_reward_by_timeout(timeout_val, conf[key], n)

            trial['reward'] = rewards
            res[key] = trial
            current_start_episode += self.num_episodes_per_phase

        df_synthetic = pd.concat(
            [df.assign(phase=key) for key, df in res.items()],
            ignore_index=True
        )

        if save:
            self.save(df_synthetic, f'{self.protocol}_{filename}')
            
        return df_synthetic

    def save(self, df: pd.DataFrame, filename: str) -> None:
        df.to_csv(os.path.join(self.directory_path, filename), index=False)

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
    
    def _sample_reward_by_timeout(self, timeout_val: float, phase_rewards: dict, n_samples: int) -> np.ndarray:
        """
        Quantile resample: maps uniform random draws onto the sorted original reward values
        """
        orig_vals = np.sort(phase_rewards.get(timeout_val, np.array([0.0])))

        if len(orig_vals) == 0 or n_samples == 0:
            return np.array([0.0] * n_samples)

        if len(np.unique(orig_vals)) <= 1:
            return np.full(n_samples, orig_vals[0])

        u = np.random.uniform(0, 1, size=n_samples)
        indices = (u * (len(orig_vals) - 1)).astype(int)
        return orig_vals[indices]
    

if __name__ == '__main__':
    seed_everything(42)
    parser = argparse.ArgumentParser(description="Generate synthetic data for PBFT protocol.")
    parser.add_argument(
        '-p', '--protocol', 
        type=str, 
        default='pbft', 
        help="The protocol to use (default: pbft)"
    )
    parser.add_argument(
        '-d', '--data_dir', 
        type=str, 
        default='./data/pbft', 
        help="Directory containing the original CSV files (default: ./data)"
    )
    parser.add_argument(
        '-n', '--num_episodes', 
        type=int, 
        default=1000, 
        help="Number of episodes per phase to generate (default: 1000)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Directory '{args.data_dir}' not found.\nPlace original'{args.protocol}' CSV files inside '{args.data_dir}' and run again.")
    else:
        print(f"Initializing DataGenerator for protocol: {args.protocol}")
        
        generator = DataGenerator(
            protocol=args.protocol, 
            directory_path=args.data_dir, 
            num_episode_each_phase=args.num_episodes
        )
        df_original = generator.read_data(save=True)
        
        if df_original is not None:
            df_synthetic = generator.generate_data(save=True)
            
            if df_synthetic is not None:
                print(f"Original data shape: {df_original.shape}")
                print(f"Synthetic data shape: {df_synthetic.shape}")
        else:
            print("\nProcess aborted: No original data could be loaded. Please check your directory and CSV files.")
