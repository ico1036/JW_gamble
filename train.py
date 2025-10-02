"""
Train Machine Learning Models for Horse Racing Prediction

Models:
1. Logistic Regression (Baseline)
2. LightGBM (Advanced)

핵심 전략:
- ROI 최적화 (threshold tuning)
- 배당 범위별 모델 (optional)
- Overfitting 방지 (regularization, early stopping)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, classification_report
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

from evaluation import (
    evaluate_strategy, evaluate_by_odds_range,
    print_evaluation, analyze_elite_effect, calculate_roi
)


def load_data():
    """데이터 및 피처 로드"""
    train_df = pd.read_parquet('data/train.parquet')
    val_df = pd.read_parquet('data/val.parquet')
    test_df = pd.read_parquet('data/test.parquet')

    with open('data/feature_columns.json', 'r') as f:
        feature_cols = json.load(f)

    print(f"✓ Data loaded:")
    print(f"  - Train: {len(train_df):,}")
    print(f"  - Val:   {len(val_df):,}")
    print(f"  - Test:  {len(test_df):,}")
    print(f"  - Features: {len(feature_cols)}")

    return train_df, val_df, test_df, feature_cols


def optimize_threshold_for_roi(y_true, y_proba, odds, thresholds=None, min_bets=100):
    """
    ROI 최적화를 위한 threshold 탐색

    Args:
        y_true: 실제값
        y_proba: 예측 확률
        odds: 배당
        thresholds: 탐색할 threshold 리스트
        min_bets: 최소 베팅 수 (이보다 적으면 제외)

    Returns:
        best_threshold, best_roi, threshold_results
    """
    if thresholds is None:
        # 확률 분포에 맞춰 threshold 범위 조정
        min_t = max(0.05, y_proba.min() - 0.01)
        max_t = min(0.9, y_proba.max() + 0.01)
        thresholds = np.arange(min_t, max_t, 0.01)

    results = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        roi_metrics = calculate_roi(
            pd.Series(y_pred),
            pd.Series(y_true.values),
            pd.Series(odds.values)
        )

        # min_bets 이상인 경우만 추가
        if roi_metrics['num_bets'] >= min_bets:
            results.append({
                'threshold': threshold,
                'roi': roi_metrics['roi'],
                'num_bets': roi_metrics['num_bets'],
                'precision': roi_metrics['precision']
            })

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print(f"⚠️  Warning: No threshold with >= {min_bets} bets. Using lowest threshold.")
        # min_bets 무시하고 다시 계산
        return optimize_threshold_for_roi(y_true, y_proba, odds, thresholds, min_bets=0)

    best_idx = results_df['roi'].idxmax()
    best_threshold = results_df.iloc[best_idx]['threshold']
    best_roi = results_df.iloc[best_idx]['roi']

    return best_threshold, best_roi, results_df


def train_logistic_regression(train_df, val_df, feature_cols):
    """로지스틱 회귀 모델 훈련"""
    print("\n" + "=" * 80)
    print("Training Logistic Regression")
    print("=" * 80)

    # 데이터 준비
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['top3']
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df['top3']

    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 모델 훈련
    model = LogisticRegression(
        C=0.1,  # Regularization
        max_iter=1000,
        random_state=42,
        class_weight='balanced'  # 불균형 데이터 처리
    )
    model.fit(X_train_scaled, y_train)

    # 예측
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
    y_val_proba = model.predict_proba(X_val_scaled)[:, 1]

    # 평가
    print(f"\nTrain ROC-AUC: {roc_auc_score(y_train, y_train_proba):.4f}")
    print(f"Val ROC-AUC: {roc_auc_score(y_val, y_val_proba):.4f}")

    # Threshold 최적화 (Val set)
    print(f"\nOptimizing threshold for ROI (Val set)...")
    best_threshold, best_roi, threshold_results = optimize_threshold_for_roi(
        y_val, y_val_proba, val_df['odds_place']
    )

    print(f"✓ Best threshold: {best_threshold:.3f}")
    print(f"✓ Best Val ROI: {best_roi:.2f}%")

    # Top 5 thresholds
    top5 = threshold_results.nlargest(5, 'roi')
    print(f"\nTop 5 thresholds:")
    print(top5.to_string(index=False))

    return model, scaler, best_threshold


def train_lightgbm(train_df, val_df, feature_cols):
    """LightGBM 모델 훈련"""
    print("\n" + "=" * 80)
    print("Training LightGBM")
    print("=" * 80)

    # 데이터 준비
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['top3']
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df['top3']

    # LightGBM Dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # 파라미터
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 64,
        'max_depth': 8,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'min_split_gain': 0.001,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'seed': 42
    }

    # 훈련
    print("\nTraining...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=50)
        ]
    )

    # 예측
    y_train_proba = model.predict(X_train, num_iteration=model.best_iteration)
    y_val_proba = model.predict(X_val, num_iteration=model.best_iteration)

    # 평가
    print(f"\nTrain ROC-AUC: {roc_auc_score(y_train, y_train_proba):.4f}")
    print(f"Val ROC-AUC: {roc_auc_score(y_val, y_val_proba):.4f}")

    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    print(f"\nTop 10 Important Features:")
    print(importance_df.head(10).to_string(index=False))

    # Threshold 최적화
    print(f"\nOptimizing threshold for ROI (Val set)...")
    best_threshold, best_roi, threshold_results = optimize_threshold_for_roi(
        y_val, y_val_proba, val_df['odds_place']
    )

    print(f"✓ Best threshold: {best_threshold:.3f}")
    print(f"✓ Best Val ROI: {best_roi:.2f}%")

    return model, best_threshold, importance_df


def evaluate_model(model, model_name, threshold, test_df, feature_cols, scaler=None):
    """모델 평가"""
    print("\n" + "=" * 80)
    print(f"Evaluating {model_name} on Test Set")
    print("=" * 80)

    X_test = test_df[feature_cols].fillna(0)

    # 예측
    if scaler is not None:  # Logistic Regression
        X_test_scaled = scaler.transform(X_test)
        y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:  # LightGBM
        y_test_proba = model.predict(X_test, num_iteration=model.best_iteration)

    y_test_pred = (y_test_proba >= threshold).astype(int)

    # ROC-AUC
    auc = roc_auc_score(test_df['top3'], y_test_proba)
    print(f"Test ROC-AUC: {auc:.4f}")

    # 평가
    eval_df = evaluate_strategy(
        pd.Series(y_test_pred),
        test_df['top3'],
        test_df['odds_place'],
        model_name
    )
    print_evaluation(eval_df)

    # 배당 범위별
    odds_range_df = evaluate_by_odds_range(
        pd.Series(y_test_pred),
        test_df['top3'],
        test_df['odds_place'],
        test_df['odds_win']
    )
    print(f"\nOdds Range Analysis:")
    print(odds_range_df.to_string(index=False))
    print("=" * 80)

    # 엘리트 효과
    analyze_elite_effect(test_df, pd.Series(y_test_pred))

    return y_test_pred, y_test_proba


def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("Machine Learning Models Training")
    print("=" * 80)

    # 데이터 로드
    train_df, val_df, test_df, feature_cols = load_data()

    # 1. Logistic Regression
    lr_model, lr_scaler, lr_threshold = train_logistic_regression(
        train_df, val_df, feature_cols
    )

    # 2. LightGBM
    lgb_model, lgb_threshold, importance_df = train_lightgbm(
        train_df, val_df, feature_cols
    )

    # Test set 평가
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION")
    print("=" * 80)

    lr_pred, lr_proba = evaluate_model(
        lr_model, "Logistic Regression", lr_threshold,
        test_df, feature_cols, lr_scaler
    )

    lgb_pred, lgb_proba = evaluate_model(
        lgb_model, "LightGBM", lgb_threshold,
        test_df, feature_cols
    )

    # 결과 저장
    output_dir = Path('models')
    output_dir.mkdir(exist_ok=True)

    # 모델 저장
    import pickle
    with open(output_dir / 'logistic_regression.pkl', 'wb') as f:
        pickle.dump({'model': lr_model, 'scaler': lr_scaler, 'threshold': lr_threshold}, f)

    lgb_model.save_model(str(output_dir / 'lightgbm.txt'))

    # Feature importance 저장
    importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)

    # 예측 결과 저장
    test_df['lr_proba'] = lr_proba
    test_df['lr_pred'] = lr_pred
    test_df['lgb_proba'] = lgb_proba
    test_df['lgb_pred'] = lgb_pred

    test_df[[
        'trd_dt', 'race_no', 'horse_name', 'odds_win', 'odds_place', 'top3',
        'lr_proba', 'lr_pred', 'lgb_proba', 'lgb_pred'
    ]].to_csv(output_dir / 'test_predictions.csv', index=False)

    print(f"\n✓ Models and predictions saved to {output_dir}/")


if __name__ == '__main__':
    main()
