"""
Evaluation module for horse racing prediction models
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional


def evaluate_model(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Comprehensive evaluation of binary classification model

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities (n_samples, 2) or (n_samples,)
        threshold: Classification threshold
        model_name: Name of the model for display

    Returns:
        Dictionary of metrics
    """
    # Handle probability format
    if y_pred_proba.ndim == 2:
        proba_pos = y_pred_proba[:, 1]
    else:
        proba_pos = y_pred_proba

    # Predicted classes
    y_pred = (proba_pos > threshold).astype(int)

    # Calculate metrics
    metrics = {
        'model_name': model_name,
        'roc_auc': roc_auc_score(y_true, proba_pos),
        'log_loss': log_loss(y_true, proba_pos),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn),
        'true_positive': int(tp),
    })

    # Specificity
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

    return metrics


def print_metrics(metrics: Dict[str, Any]):
    """
    Print evaluation metrics in a formatted way

    Args:
        metrics: Dictionary of metrics from evaluate_model
    """
    print(f"\n{'='*60}")
    print(f"Model: {metrics['model_name']}")
    print(f"{'='*60}")
    print(f"ROC-AUC:       {metrics['roc_auc']:.4f}")
    print(f"Log Loss:      {metrics['log_loss']:.4f}")
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print(f"Precision:     {metrics['precision']:.4f}")
    print(f"Recall:        {metrics['recall']:.4f}")
    print(f"F1 Score:      {metrics['f1_score']:.4f}")
    print(f"Specificity:   {metrics['specificity']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {metrics['true_negative']:5d}  FP: {metrics['false_positive']:5d}")
    print(f"  FN: {metrics['false_negative']:5d}  TP: {metrics['true_positive']:5d}")
    print(f"{'='*60}\n")


def compare_models(metrics_list: list) -> pd.DataFrame:
    """
    Compare multiple models side by side

    Args:
        metrics_list: List of metrics dictionaries

    Returns:
        DataFrame with comparison
    """
    df = pd.DataFrame(metrics_list)

    # Select key metrics
    key_metrics = [
        'model_name', 'roc_auc', 'log_loss', 'accuracy',
        'precision', 'recall', 'f1_score'
    ]

    return df[key_metrics].sort_values('roc_auc', ascending=False)


def plot_roc_curves(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    save_path: Optional[str] = None
):
    """
    Plot ROC curves for multiple models

    Args:
        y_true: True labels
        predictions: Dict mapping model name to predicted probabilities
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(10, 8))

    for model_name, y_pred_proba in predictions.items():
        # Handle probability format
        if y_pred_proba.ndim == 2:
            proba_pos = y_pred_proba[:, 1]
        else:
            proba_pos = y_pred_proba

        fpr, tpr, _ = roc_curve(y_true, proba_pos)
        auc = roc_auc_score(y_true, proba_pos)

        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', linewidth=2)

    # Plot diagonal (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")

    plt.show()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    save_path: Optional[str] = None
):
    """
    Plot feature importance

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to display
        save_path: Path to save figure (optional)
    """
    # Select top N features
    plot_df = importance_df.head(top_n)

    plt.figure(figsize=(10, max(6, top_n * 0.3)))
    sns.barplot(data=plot_df, x='importance', y='feature', palette='viridis')

    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance (Gain)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")

    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix heatmap

    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_path: Path to save figure (optional)
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Lose', 'Win'],
        yticklabels=['Lose', 'Win'],
        cbar_kws={'label': 'Count'}
    )

    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.show()


def calculate_expected_value(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    odds: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate expected value of betting strategy

    Args:
        y_true: True outcomes (1 = win, 0 = lose)
        y_pred_proba: Predicted probabilities
        odds: Win odds for each race
        threshold: Betting threshold (only bet if predicted prob > threshold)

    Returns:
        Dictionary with EV metrics
    """
    # Handle probability format
    if y_pred_proba.ndim == 2:
        proba_pos = y_pred_proba[:, 1]
    else:
        proba_pos = y_pred_proba

    # Identify bets
    bet_mask = proba_pos > threshold

    if bet_mask.sum() == 0:
        return {
            'num_bets': 0,
            'num_wins': 0,
            'win_rate': 0.0,
            'total_profit': 0.0,
            'roi': 0.0,
        }

    # Calculate results
    num_bets = bet_mask.sum()
    actual_wins = y_true[bet_mask]
    bet_odds = odds[bet_mask]

    # Profit per bet (assuming $1 bet)
    profits = np.where(actual_wins == 1, bet_odds - 1, -1)

    total_profit = profits.sum()
    roi = (total_profit / num_bets) * 100  # Percentage

    return {
        'num_bets': int(num_bets),
        'num_wins': int(actual_wins.sum()),
        'win_rate': float(actual_wins.mean()),
        'total_profit': float(total_profit),
        'roi': float(roi),
    }