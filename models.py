"""
Model implementations for horse racing prediction
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
import lightgbm as lgb
from typing import Optional, Dict, Any


class OddsBaselineModel(BaseEstimator, ClassifierMixin):
    """
    Baseline model using only odds (market efficiency)

    Uses implied probability from odds_win as prediction
    No training required - pure market-based prediction
    """

    def __init__(self):
        self.classes_ = np.array([0, 1])

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """No training needed for baseline"""
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return implied probabilities from odds_win

        Args:
            X: DataFrame with odds_win column

        Returns:
            Array of shape (n_samples, 2) with [P(lose), P(win)]
        """
        implied_prob = 1 / X['odds_win'].values
        # Return probabilities for both classes
        proba = np.column_stack([1 - implied_prob, implied_prob])
        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class (always 0 for imbalanced data)"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)


class LogisticBaselineModel:
    """
    Logistic regression baseline using only odds_win

    Simple linear model to establish baseline performance
    """

    def __init__(self, **kwargs):
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            **kwargs
        )
        self.classes_ = np.array([0, 1])

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Train on odds_win only"""
        # Use only odds_win for baseline
        X_odds = X[['odds_win']].values
        self.model.fit(X_odds, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        X_odds = X[['odds_win']].values
        return self.model.predict_proba(X_odds)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class"""
        X_odds = X[['odds_win']].values
        return self.model.predict(X_odds)


class LGBMModel:
    """
    LightGBM model for horse racing prediction

    Handles non-linear relationships and feature interactions
    Uses class balancing for imbalanced data
    """

    def __init__(
        self,
        feature_columns: list,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize LightGBM model

        Args:
            feature_columns: List of feature column names to use
            params: LightGBM parameters (optional)
        """
        self.feature_columns = feature_columns

        # Default parameters optimized for binary classification
        default_params = {
            'objective': 'binary',
            'metric': ['binary_logloss', 'auc'],
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 6,
            'min_child_samples': 20,
            'verbose': -1,
            'random_state': 42,
        }

        if params:
            default_params.update(params)

        self.params = default_params
        self.model = None
        self.classes_ = np.array([0, 1])

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
        verbose_eval: int = 50,
    ):
        """
        Train LightGBM model

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (for early stopping)
            y_val: Validation labels
            num_boost_round: Maximum number of boosting rounds
            early_stopping_rounds: Stop if no improvement for N rounds
            verbose_eval: Print evaluation every N rounds
        """
        # Calculate scale_pos_weight for class imbalance (90.49 / 9.51 ≈ 9.5)
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        scale_pos_weight = neg_count / pos_count

        params = self.params.copy()
        params['scale_pos_weight'] = scale_pos_weight

        # Prepare datasets
        train_data = lgb.Dataset(
            X_train[self.feature_columns],
            label=y_train,
            free_raw_data=False
        )

        valid_sets = [train_data]
        valid_names = ['train']

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(
                X_val[self.feature_columns],
                label=y_val,
                reference=train_data,
                free_raw_data=False
            )
            valid_sets.append(val_data)
            valid_names.append('valid')

        # Train model
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(verbose_eval),
            ],
        )

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities

        Args:
            X: Features dataframe

        Returns:
            Array of shape (n_samples, 2) with [P(lose), P(win)]
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")

        proba_win = self.model.predict(X[self.feature_columns])
        proba = np.column_stack([1 - proba_win, proba_win])
        return proba

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class

        Args:
            X: Features dataframe
            threshold: Classification threshold (default 0.5)

        Returns:
            Predicted classes
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] > threshold).astype(int)

    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance

        Args:
            importance_type: 'gain' or 'split'

        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model must be trained before getting feature importance")

        importance = self.model.feature_importance(importance_type=importance_type)

        return pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)