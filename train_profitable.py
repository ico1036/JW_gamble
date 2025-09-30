"""
Profit-Focused Training Script for Horse Racing Betting

Evaluates models based on ROI and profitability, not just accuracy
"""

import argparse
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict

from feature_engineering import (
    FeatureEngineer,
    get_feature_columns,
    prepare_target,
    split_data
)
from models import OddsBaselineModel, LogisticBaselineModel, LGBMModel
from evaluation import evaluate_model, print_metrics
from betting_strategy import (
    simulate_betting,
    find_optimal_threshold,
    analyze_by_odds_range,
    calculate_expected_value,
)


def load_and_prepare_data(data_path: str) -> pd.DataFrame:
    """Load and prepare raw data"""
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

    df = df[required_cols].copy()

    # Handle missing values
    df['track_cond_pct'] = df['track_cond_pct'].fillna(8.0)
    critical_cols = ['finish_pos', 'odds_win', 'odds_place', 'jockey_name', 'trainer_name', 'horse_name']
    df = df.dropna(subset=critical_cols)

    print(f"Records after cleaning: {len(df):,}")
    return df


def print_betting_results(name: str, result: Dict):
    """Print betting simulation results"""
    print(f"\n{'='*80}")
    print(f"{name} - Betting Simulation Results")
    print(f"{'='*80}")
    print(f"Initial Bankroll:  ₩{result['initial_bankroll']:,.0f}")
    print(f"Final Bankroll:    ₩{result['final_bankroll']:,.0f}")
    print(f"Total Profit:      ₩{result['total_profit']:,.0f} ({result['profit_pct']:+.2f}%)")
    print(f"\nBetting Statistics:")
    print(f"  Total Bets:      {result['total_bets']:,}")
    print(f"  Wins:            {result['total_wins']:,}")
    print(f"  Win Rate:        {result['win_rate']*100:.2f}%")
    print(f"  Total Wagered:   ₩{result['total_wagered']:,.0f}")
    print(f"  Total Returned:  ₩{result['total_returned']:,.0f}")
    print(f"  ROI:             {result['roi']:+.2f}%")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Train profit-focused horse racing betting models'
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
        default='models_profitable',
        help='Directory to save models and results'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("="*80)
    print("PROFIT-FOCUSED HORSE RACING BETTING MODEL")
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
    train_df = feature_engineer.fit_transform(train_df)
    val_df = feature_engineer.transform(val_df)
    test_df = feature_engineer.transform(test_df)

    feature_cols = get_feature_columns()
    y_train = prepare_target(train_df, target_type='binary')
    y_val = prepare_target(val_df, target_type='binary')
    y_test = prepare_target(test_df, target_type='binary')

    print(f"\nTest set class distribution: {y_test.mean()*100:.2f}% wins")

    # ========== Train Models ==========
    print("\n" + "="*80)
    print("TRAINING MODELS")
    print("="*80)

    # 1. Odds Baseline
    print("\n[1/3] Odds Baseline Model...")
    odds_baseline = OddsBaselineModel()
    odds_baseline.fit(train_df[feature_cols], y_train)

    # 2. Logistic Baseline
    print("[2/3] Logistic Baseline Model...")
    logistic_baseline = LogisticBaselineModel()
    logistic_baseline.fit(train_df[feature_cols], y_train)

    # 3. LightGBM
    print("[3/3] LightGBM Model...")
    lgbm_model = LGBMModel(feature_columns=feature_cols)
    lgbm_model.fit(
        train_df[feature_cols], y_train,
        X_val=val_df[feature_cols], y_val=y_val,
        num_boost_round=500,
        early_stopping_rounds=50,
        verbose_eval=100
    )

    # ========== Profitability Analysis on Test Set ==========
    print("\n" + "="*80)
    print("PROFITABILITY ANALYSIS - TEST SET")
    print("="*80)

    # Get predictions
    odds_pred = odds_baseline.predict_proba(test_df[feature_cols])[:, 1]
    logistic_pred = logistic_baseline.predict_proba(test_df[feature_cols])[:, 1]
    lgbm_pred = lgbm_model.predict_proba(test_df[feature_cols])[:, 1]

    test_odds = test_df['odds_win'].values

    # Strategy 1: Fixed bet on all predictions
    print("\n" + "="*80)
    print("STRATEGY 1: Fixed Bet (₩1,000) - All Races")
    print("="*80)

    odds_result_all = simulate_betting(
        y_test, odds_pred, test_odds,
        strategy='threshold',
        prob_threshold=0.0,  # Bet on all
        bet_size_method='fixed',
        fixed_bet_size=1000,
    )
    print_betting_results("Odds Baseline (All Bets)", odds_result_all)

    lgbm_result_all = simulate_betting(
        y_test, lgbm_pred, test_odds,
        strategy='threshold',
        prob_threshold=0.0,
        bet_size_method='fixed',
        fixed_bet_size=1000,
    )
    print_betting_results("LightGBM (All Bets)", lgbm_result_all)

    # Strategy 2: Positive EV only
    print("\n" + "="*80)
    print("STRATEGY 2: Positive Expected Value (EV > 0)")
    print("="*80)

    odds_result_ev = simulate_betting(
        y_test, odds_pred, test_odds,
        strategy='ev',
        ev_threshold=0.0,
        bet_size_method='fixed',
        fixed_bet_size=1000,
    )
    print_betting_results("Odds Baseline (EV > 0)", odds_result_ev)

    lgbm_result_ev = simulate_betting(
        y_test, lgbm_pred, test_odds,
        strategy='ev',
        ev_threshold=0.0,
        bet_size_method='fixed',
        fixed_bet_size=1000,
    )
    print_betting_results("LightGBM (EV > 0)", lgbm_result_ev)

    # Strategy 3: Optimized threshold
    print("\n" + "="*80)
    print("STRATEGY 3: Optimized EV Threshold (Maximizes ROI)")
    print("="*80)

    print("\nOptimizing Odds Baseline threshold...")
    odds_opt_thresh, odds_opt_result = find_optimal_threshold(
        y_test, odds_pred, test_odds,
        strategy='ev',
        threshold_range=(-0.05, 0.15),
        num_steps=40,
        bet_size_method='fixed',
        fixed_bet_size=1000,
    )
    print(f"Optimal threshold: EV > {odds_opt_thresh:.4f}")
    print_betting_results("Odds Baseline (Optimized)", odds_opt_result)

    print("\nOptimizing LightGBM threshold...")
    lgbm_opt_thresh, lgbm_opt_result = find_optimal_threshold(
        y_test, lgbm_pred, test_odds,
        strategy='ev',
        threshold_range=(-0.05, 0.15),
        num_steps=40,
        bet_size_method='fixed',
        fixed_bet_size=1000,
    )
    print(f"Optimal threshold: EV > {lgbm_opt_thresh:.4f}")
    print_betting_results("LightGBM (Optimized)", lgbm_opt_result)

    # Strategy 4: Kelly Criterion
    print("\n" + "="*80)
    print("STRATEGY 4: Kelly Criterion (Dynamic Bet Sizing)")
    print("="*80)

    odds_kelly = simulate_betting(
        y_test, odds_pred, test_odds,
        strategy='ev',
        ev_threshold=0.0,
        bet_size_method='kelly',
        kelly_fraction=0.25,  # Quarter Kelly (conservative)
    )
    print_betting_results("Odds Baseline (Kelly 0.25)", odds_kelly)

    lgbm_kelly = simulate_betting(
        y_test, lgbm_pred, test_odds,
        strategy='ev',
        ev_threshold=0.0,
        bet_size_method='kelly',
        kelly_fraction=0.25,
    )
    print_betting_results("LightGBM (Kelly 0.25)", lgbm_kelly)

    # Analysis by odds range
    print("\n" + "="*80)
    print("PROFITABILITY BY ODDS RANGE")
    print("="*80)

    print("\nOdds Baseline:")
    odds_range_df = analyze_by_odds_range(y_test, odds_pred, test_odds)
    print(odds_range_df.to_string(index=False))

    print("\n\nLightGBM:")
    lgbm_range_df = analyze_by_odds_range(y_test, lgbm_pred, test_odds)
    print(lgbm_range_df.to_string(index=False))

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    results = {
        'timestamp': datetime.now().isoformat(),
        'test_set': {
            'n_races': len(y_test),
            'n_wins': int(y_test.sum()),
            'win_rate': float(y_test.mean()),
        },
        'strategies': {
            'odds_baseline_all': odds_result_all,
            'odds_baseline_ev': odds_result_ev,
            'odds_baseline_optimized': odds_opt_result,
            'odds_baseline_kelly': odds_kelly,
            'lgbm_all': lgbm_result_all,
            'lgbm_ev': lgbm_result_ev,
            'lgbm_optimized': lgbm_opt_result,
            'lgbm_kelly': lgbm_kelly,
        },
        'optimal_thresholds': {
            'odds_baseline': odds_opt_thresh,
            'lgbm': lgbm_opt_thresh,
        },
    }

    # Remove bankroll history and bet details (too large)
    for strategy_name, strategy_result in results['strategies'].items():
        if 'bankroll_history' in strategy_result:
            del strategy_result['bankroll_history']
        if 'bet_details' in strategy_result:
            del strategy_result['bet_details']

    results_path = output_dir / 'profitability_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    # Save odds range analysis
    odds_range_df.to_csv(output_dir / 'odds_baseline_by_range.csv', index=False)
    lgbm_range_df.to_csv(output_dir / 'lgbm_by_range.csv', index=False)

    # Save models
    with open(output_dir / 'odds_baseline.pkl', 'wb') as f:
        pickle.dump(odds_baseline, f)
    with open(output_dir / 'lgbm_model.pkl', 'wb') as f:
        pickle.dump(lgbm_model, f)
    with open(output_dir / 'feature_engineer.pkl', 'wb') as f:
        pickle.dump(feature_engineer, f)

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}/")


if __name__ == '__main__':
    main()