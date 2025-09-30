"""
Feature Engineering Module for Horse Racing Prediction

Based on EDA findings:
- Tier 1: odds_win, odds_place, burden_weight
- Tier 2: jockey, trainer, horse_age
- Tier 3: gate_no, track_cond_pct
"""

import pandas as pd
import numpy as np
from typing import Tuple


class FeatureEngineer:
    """Feature engineering based on EDA insights"""

    def __init__(self):
        self.jockey_stats = {}
        self.trainer_stats = {}
        self.horse_stats = {}
        self.global_avg_pos = 6.18  # From EDA
        self.global_win_rate = 0.0951  # 9.51% from EDA

    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """
        Compute statistics from training data

        Args:
            df: Training dataframe with race results

        Returns:
            self
        """
        # Compute jockey statistics
        jockey_agg = df.groupby('jockey_name').agg({
            'finish_pos': ['mean', 'count'],
        }).reset_index()
        jockey_agg.columns = ['jockey_name', 'jockey_avg_pos', 'jockey_races']

        # Calculate win rate
        wins = df[df['finish_pos'] == 1].groupby('jockey_name').size()
        jockey_agg = jockey_agg.merge(
            wins.rename('jockey_wins').reset_index(),
            on='jockey_name',
            how='left'
        )
        jockey_agg['jockey_wins'] = jockey_agg['jockey_wins'].fillna(0)
        jockey_agg['jockey_win_rate'] = (
            jockey_agg['jockey_wins'] / jockey_agg['jockey_races']
        )
        self.jockey_stats = jockey_agg

        # Compute trainer statistics
        trainer_agg = df.groupby('trainer_name').agg({
            'finish_pos': ['mean', 'count'],
        }).reset_index()
        trainer_agg.columns = ['trainer_name', 'trainer_avg_pos', 'trainer_races']

        wins = df[df['finish_pos'] == 1].groupby('trainer_name').size()
        trainer_agg = trainer_agg.merge(
            wins.rename('trainer_wins').reset_index(),
            on='trainer_name',
            how='left'
        )
        trainer_agg['trainer_wins'] = trainer_agg['trainer_wins'].fillna(0)
        trainer_agg['trainer_win_rate'] = (
            trainer_agg['trainer_wins'] / trainer_agg['trainer_races']
        )
        self.trainer_stats = trainer_agg

        # Compute horse statistics
        horse_agg = df.groupby('horse_name').agg({
            'finish_pos': ['mean', 'count'],
        }).reset_index()
        horse_agg.columns = ['horse_name', 'horse_avg_pos', 'horse_races']

        wins = df[df['finish_pos'] == 1].groupby('horse_name').size()
        horse_agg = horse_agg.merge(
            wins.rename('horse_wins').reset_index(),
            on='horse_name',
            how='left'
        )
        horse_agg['horse_wins'] = horse_agg['horse_wins'].fillna(0)
        horse_agg['horse_win_rate'] = (
            horse_agg['horse_wins'] / horse_agg['horse_races']
        )
        self.horse_stats = horse_agg

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataframe with engineered features

        Args:
            df: Input dataframe

        Returns:
            DataFrame with added features
        """
        df = df.copy()

        # Merge jockey stats
        df = df.merge(
            self.jockey_stats[['jockey_name', 'jockey_avg_pos', 'jockey_win_rate', 'jockey_races']],
            on='jockey_name',
            how='left'
        )

        # Merge trainer stats
        df = df.merge(
            self.trainer_stats[['trainer_name', 'trainer_avg_pos', 'trainer_win_rate', 'trainer_races']],
            on='trainer_name',
            how='left'
        )

        # Merge horse stats
        df = df.merge(
            self.horse_stats[['horse_name', 'horse_avg_pos', 'horse_win_rate', 'horse_races']],
            on='horse_name',
            how='left'
        )

        # Fill missing values for new jockeys/trainers/horses with global averages
        df['jockey_avg_pos'] = df['jockey_avg_pos'].fillna(self.global_avg_pos)
        df['jockey_win_rate'] = df['jockey_win_rate'].fillna(self.global_win_rate)
        df['jockey_races'] = df['jockey_races'].fillna(0)

        df['trainer_avg_pos'] = df['trainer_avg_pos'].fillna(self.global_avg_pos)
        df['trainer_win_rate'] = df['trainer_win_rate'].fillna(self.global_win_rate)
        df['trainer_races'] = df['trainer_races'].fillna(0)

        df['horse_avg_pos'] = df['horse_avg_pos'].fillna(self.global_avg_pos)
        df['horse_win_rate'] = df['horse_win_rate'].fillna(self.global_win_rate)
        df['horse_races'] = df['horse_races'].fillna(0)

        # Create elite jockey/trainer indicators (top 10% threshold from EDA)
        df['is_elite_jockey'] = (df['jockey_win_rate'] > 0.13).astype(int)
        df['is_elite_trainer'] = (df['trainer_win_rate'] > 0.12).astype(int)

        # Age features (peak age 2-3 from EDA)
        df['is_young_horse'] = (df['horse_age'] <= 3).astype(int)

        # Gate position features (inner gates have slight advantage)
        df['is_inner_gate'] = (df['gate_no'] <= 3).astype(int)

        # Interaction features
        df['elite_combo'] = df['is_elite_jockey'] * df['is_elite_trainer']

        # Odds-based features (log transform for heavy tail distribution)
        df['log_odds_win'] = np.log1p(df['odds_win'])
        df['log_odds_place'] = np.log1p(df['odds_place'])

        # Inverse odds (implied probability)
        df['implied_prob_win'] = 1 / df['odds_win']
        df['implied_prob_place'] = 1 / df['odds_place']

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(df).transform(df)


def get_feature_columns() -> list:
    """
    Returns list of feature columns to use for modeling

    Based on EDA findings:
    - Tier 1: odds, burden_weight
    - Tier 2: jockey/trainer stats, age
    - Tier 3: gate, track condition
    """
    return [
        # Tier 1: Odds (strongest predictors, ρ=0.519)
        'odds_win',
        'odds_place',
        'log_odds_win',
        'log_odds_place',
        'implied_prob_win',
        'implied_prob_place',
        'burden_weight',

        # Tier 2: People and age
        'jockey_win_rate',
        'jockey_avg_pos',
        'jockey_races',
        'trainer_win_rate',
        'trainer_avg_pos',
        'trainer_races',
        'horse_win_rate',
        'horse_avg_pos',
        'horse_races',
        'horse_age',
        'is_elite_jockey',
        'is_elite_trainer',
        'is_young_horse',

        # Tier 3: Position and conditions
        'gate_no',
        'is_inner_gate',
        'track_cond_pct',

        # Interactions
        'elite_combo',
    ]


def prepare_target(df: pd.DataFrame, target_type: str = 'binary') -> pd.Series:
    """
    Prepare target variable

    Args:
        df: DataFrame with finish_pos column
        target_type: 'binary' (win/lose) or 'top3' (top 3 vs rest)

    Returns:
        Target series
    """
    if target_type == 'binary':
        return (df['finish_pos'] == 1).astype(int)
    elif target_type == 'top3':
        return (df['finish_pos'] <= 3).astype(int)
    else:
        raise ValueError(f"Unknown target_type: {target_type}")


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically (time-series aware)

    Args:
        df: Input dataframe with trd_dt column
        test_size: Fraction for test set
        val_size: Fraction for validation set (from remaining after test split)

    Returns:
        train_df, val_df, test_df
    """
    df = df.sort_values('trd_dt').reset_index(drop=True)

    n = len(df)
    test_idx = int(n * (1 - test_size))
    val_idx = int(test_idx * (1 - val_size))

    train_df = df.iloc[:val_idx].copy()
    val_df = df.iloc[val_idx:test_idx].copy()
    test_df = df.iloc[test_idx:].copy()

    return train_df, val_df, test_df