"""
Training script for horse racing prediction models

Usage:
    python train.py --data race_results.parquet --output-dir models/
"""

import argparse
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime

from feature_engineering import (
    FeatureEngineer,
    get_feature_columns,
    prepare_target,
    split_data
)
from models import OddsBaselineModel, LogisticBaselineModel, LGBMModel
from evaluation import (
    evaluate_model,
    print_metrics,
    compare_models,
    plot_roc_curves,
    plot_feature_importance,
)


def load_and_prepare_data(data_path: str) -> pd.DataFrame:
    """
    Load and prepare raw data

    Args:
        data_path: Path to parquet file

    Returns:
        Cleaned dataframe
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)

    print(f"Loaded {len(df):,} records")
    print(f"Date range: {df['trd_dt'].min()} to {df['trd_dt'].max()}")

    # Select required columns
    required_cols = [
        'trd_dt', 'finish_pos', 'odds_win', 'odds_place', 'burden_weight',
        'horse_age', 'gate_no', 'track_cond_pct', 'jockey_name',
        'trainer_name', 'horse_name'
    ]

    # Check for missing required columns
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df[required_cols].copy()

    # Handle missing values
    print("\nHandling missing values...")
    print(f"Missing before: {df.isnull().sum().sum()}")

    # Fill track_cond_pct with median (8.0 from EDA)
    df['track_cond_pct'] = df['track_cond_pct'].fillna(8.0)

    # Drop rows with missing critical values
    critical_cols = ['finish_pos', 'odds_win', 'odds_place', 'jockey_name', 'trainer_name', 'horse_name']
    df = df.dropna(subset=critical_cols)

    print(f"Missing after: {df.isnull().sum().sum()}")
    print(f"Records after cleaning: {len(df):,}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Train horse racing prediction models'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='race_results.parquet',
        help='Path to race results parquet file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plots (useful for non-interactive environments)'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("="*80)
    print("Horse Racing Prediction Model Training")
    print("="*80)

    # Load and prepare data
    df = load_and_prepare_data(args.data)

    # Split data chronologically
    print("\nSplitting data chronologically...")
    train_df, val_df, test_df = split_data(df, test_size=0.2, val_size=0.1)

    print(f"Train set: {len(train_df):,} records ({train_df['trd_dt'].min()} to {train_df['trd_dt'].max()})")
    print(f"Val set:   {len(val_df):,} records ({val_df['trd_dt'].min()} to {val_df['trd_dt'].max()})")
    print(f"Test set:  {len(test_df):,} records ({test_df['trd_dt'].min()} to {test_df['trd_dt'].max()})")

    # Feature engineering
    print("\nFeature engineering...")
    feature_engineer = FeatureEngineer()

    # Fit on train, transform all sets
    train_df = feature_engineer.fit_transform(train_df)
    val_df = feature_engineer.transform(val_df)
    test_df = feature_engineer.transform(test_df)

    # Save feature engineer
    fe_path = output_dir / 'feature_engineer.pkl'
    with open(fe_path, 'wb') as f:
        pickle.dump(feature_engineer, f)
    print(f"Saved feature engineer to {fe_path}")

    # Prepare features and target
    feature_cols = get_feature_columns()
    print(f"\nUsing {len(feature_cols)} features")

    # Prepare targets
    y_train = prepare_target(train_df, target_type='binary')
    y_val = prepare_target(val_df, target_type='binary')
    y_test = prepare_target(test_df, target_type='binary')

    print(f"\nClass distribution:")
    print(f"Train: {y_train.mean()*100:.2f}% wins")
    print(f"Val:   {y_val.mean()*100:.2f}% wins")
    print(f"Test:  {y_test.mean()*100:.2f}% wins")

    # ========== Train Models ==========

    print("\n" + "="*80)
    print("Training Models")
    print("="*80)

    # 1. Odds Baseline Model
    print("\n[1/3] Training Odds Baseline Model...")
    odds_baseline = OddsBaselineModel()
    odds_baseline.fit(train_df[feature_cols], y_train)

    # 2. Logistic Baseline Model
    print("[2/3] Training Logistic Baseline Model...")
    logistic_baseline = LogisticBaselineModel()
    logistic_baseline.fit(train_df[feature_cols], y_train)

    # 3. LightGBM Model
    print("[3/3] Training LightGBM Model...")
    lgbm_model = LGBMModel(feature_columns=feature_cols)
    lgbm_model.fit(
        train_df[feature_cols],
        y_train,
        X_val=val_df[feature_cols],
        y_val=y_val,
        num_boost_round=500,
        early_stopping_rounds=50,
        verbose_eval=50
    )

    # Save models
    print("\nSaving models...")
    with open(output_dir / 'odds_baseline.pkl', 'wb') as f:
        pickle.dump(odds_baseline, f)
    with open(output_dir / 'logistic_baseline.pkl', 'wb') as f:
        pickle.dump(logistic_baseline, f)
    with open(output_dir / 'lgbm_model.pkl', 'wb') as f:
        pickle.dump(lgbm_model, f)

    # Save feature columns
    with open(output_dir / 'feature_columns.json', 'w') as f:
        json.dump(feature_cols, f, indent=2)

    print(f"Models saved to {output_dir}/")

    # ========== Evaluation ==========

    print("\n" + "="*80)
    print("Model Evaluation on Test Set")
    print("="*80)

    # Predictions
    odds_pred = odds_baseline.predict_proba(test_df[feature_cols])
    logistic_pred = logistic_baseline.predict_proba(test_df[feature_cols])
    lgbm_pred = lgbm_model.predict_proba(test_df[feature_cols])

    # Evaluate each model
    odds_metrics = evaluate_model(y_test, odds_pred, model_name="Odds Baseline")
    logistic_metrics = evaluate_model(y_test, logistic_pred, model_name="Logistic Baseline")
    lgbm_metrics = evaluate_model(y_test, lgbm_pred, model_name="LightGBM")

    # Print metrics
    print_metrics(odds_metrics)
    print_metrics(logistic_metrics)
    print_metrics(lgbm_metrics)

    # Compare models
    print("\n" + "="*80)
    print("Model Comparison")
    print("="*80)
    comparison = compare_models([odds_metrics, logistic_metrics, lgbm_metrics])
    print(comparison.to_string(index=False))

    # Save metrics
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            'odds_baseline': odds_metrics,
            'logistic_baseline': logistic_metrics,
            'lightgbm': lgbm_metrics,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # ========== Visualizations ==========

    if not args.no_plots:
        print("\n" + "="*80)
        print("Generating Visualizations")
        print("="*80)

        # ROC curves
        plot_roc_curves(
            y_test,
            {
                'Odds Baseline': odds_pred,
                'Logistic Baseline': logistic_pred,
                'LightGBM': lgbm_pred,
            },
            save_path=output_dir / 'roc_curves.png'
        )

        # Feature importance
        importance_df = lgbm_model.get_feature_importance()
        print("\nTop 10 Features:")
        print(importance_df.head(10).to_string(index=False))

        plot_feature_importance(
            importance_df,
            top_n=20,
            save_path=output_dir / 'feature_importance.png'
        )

        # Save importance to CSV
        importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)

    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}/")
    print("\nFiles created:")
    print("  - feature_engineer.pkl")
    print("  - odds_baseline.pkl")
    print("  - logistic_baseline.pkl")
    print("  - lgbm_model.pkl")
    print("  - feature_columns.json")
    print("  - metrics.json")
    print("  - feature_importance.csv")
    if not args.no_plots:
        print("  - roc_curves.png")
        print("  - feature_importance.png")


if __name__ == '__main__':
    main()