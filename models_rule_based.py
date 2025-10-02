"""
Rule-based Models for Horse Racing Prediction

EDA 인사이트 기반 룰베이스 전략:
1. 50배+ 엘리트 전략 (Primary): 고배당 + 엘리트 기수/조련사
2. 1-2배 저배당 전략 (Secondary): 이미 수익 가능
3. 하이브리드 전략: 1 + 2 조합
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from evaluation import (
    evaluate_strategy, evaluate_by_odds_range,
    print_evaluation, analyze_elite_effect
)


class RuleBasedModel:
    """룰베이스 모델 베이스 클래스"""

    def __init__(self, name: str):
        self.name = name

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """예측 (하위 클래스에서 구현)"""
        raise NotImplementedError

    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        """평가"""
        predictions = self.predict(df)
        return evaluate_strategy(
            predictions,
            df['top3'],
            df['odds_place'],
            self.name
        )


class Strategy50xElite(RuleBasedModel):
    """
    50배+ 엘리트 전략

    조건:
    - 단승 배당 50배 이상
    - 엘리트 기수 OR 엘리트 조련사

    EDA 근거:
    - 엘리트 기수 Top3: 11.3% (베이스라인 4.7%의 2.4배)
    - 엘리트 조련사 Top3: 11.7% (베이스라인 4.7%의 2.5배)
    - 필요 정확도: 7.4% (평균 배당 16.86배 기준)
    - 갭: +3.9%p ~ +4.3%p (충분히 수익 가능!)
    """

    def __init__(self, min_odds: float = 50.0, elite_threshold: str = 'any'):
        """
        Args:
            min_odds: 최소 배당 (기본 50배)
            elite_threshold: 'any' (기수 OR 조련사), 'both' (기수 AND 조련사)
        """
        super().__init__(f"50x+ Elite ({elite_threshold.upper()})")
        self.min_odds = min_odds
        self.elite_threshold = elite_threshold

    def predict(self, df: pd.DataFrame) -> pd.Series:
        # 50배 이상 고배당
        high_odds_mask = df['odds_win'] >= self.min_odds

        # 엘리트 조건
        if self.elite_threshold == 'any':
            elite_mask = (df['jockey_elite'] == 1) | (df['trainer_elite'] == 1)
        elif self.elite_threshold == 'both':
            elite_mask = (df['jockey_elite'] == 1) & (df['trainer_elite'] == 1)
        else:
            raise ValueError(f"Invalid elite_threshold: {self.elite_threshold}")

        predictions = (high_odds_mask & elite_mask).astype(int)
        return predictions


class Strategy1to2x(RuleBasedModel):
    """
    1-2배 저배당 전략

    조건:
    - 단승 배당 1~2배

    EDA 근거:
    - 실제 Top3: 53.6%
    - 필요 정확도: 44.7%
    - 갭: -8.9%p (이미 수익 가능!)
    - 이론 ROI: +20.1%
    """

    def __init__(self, min_odds: float = 1.0, max_odds: float = 2.0):
        super().__init__(f"Low Odds ({min_odds}-{max_odds}x)")
        self.min_odds = min_odds
        self.max_odds = max_odds

    def predict(self, df: pd.DataFrame) -> pd.Series:
        predictions = (
            (df['odds_win'] >= self.min_odds) &
            (df['odds_win'] < self.max_odds)
        ).astype(int)
        return predictions


class StrategyHybrid(RuleBasedModel):
    """
    하이브리드 전략

    조건:
    - (50배+ 엘리트) OR (1-2배 저배당)

    두 가지 수익 가능한 전략을 결합
    """

    def __init__(self):
        super().__init__("Hybrid (50x Elite + 1-2x)")
        self.strategy_50x = Strategy50xElite(min_odds=50, elite_threshold='any')
        self.strategy_1to2x = Strategy1to2x(min_odds=1, max_odds=2)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        pred_50x = self.strategy_50x.predict(df)
        pred_1to2x = self.strategy_1to2x.predict(df)

        predictions = ((pred_50x == 1) | (pred_1to2x == 1)).astype(int)
        return predictions


class StrategyEliteScore(RuleBasedModel):
    """
    엘리트 스코어 전략

    조건:
    - 50배+ 고배당
    - 엘리트 스코어 기반 필터링
      - 기수 50x Top3 비율
      - 조련사 50x Top3 비율
      - 스코어 임계값 이상만 베팅
    """

    def __init__(self, min_odds: float = 50.0, score_threshold: float = 0.10):
        super().__init__(f"Elite Score (threshold={score_threshold:.1%})")
        self.min_odds = min_odds
        self.score_threshold = score_threshold

    def predict(self, df: pd.DataFrame) -> pd.Series:
        # 50배 이상
        high_odds_mask = df['odds_win'] >= self.min_odds

        # 엘리트 스코어 = max(기수 Top3 비율, 조련사 Top3 비율)
        elite_score = df[['jockey_50x_top3_rate', 'trainer_50x_top3_rate']].max(axis=1)

        predictions = (high_odds_mask & (elite_score >= self.score_threshold)).astype(int)
        return predictions


def load_data():
    """데이터 로드"""
    train_df = pd.read_parquet('data/train.parquet')
    val_df = pd.read_parquet('data/val.parquet')
    test_df = pd.read_parquet('data/test.parquet')

    print(f"✓ Data loaded:")
    print(f"  - Train: {len(train_df):,}")
    print(f"  - Val:   {len(val_df):,}")
    print(f"  - Test:  {len(test_df):,}")

    return train_df, val_df, test_df


def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("Rule-Based Models Evaluation")
    print("=" * 80)

    # 데이터 로드
    train_df, val_df, test_df = load_data()

    # 전략 정의
    strategies = [
        Strategy50xElite(min_odds=50, elite_threshold='any'),
        Strategy50xElite(min_odds=50, elite_threshold='both'),
        Strategy1to2x(min_odds=1, max_odds=2),
        StrategyHybrid(),
        StrategyEliteScore(min_odds=50, score_threshold=0.10),
        StrategyEliteScore(min_odds=50, score_threshold=0.12),
    ]

    # 베이스라인: 모든 50배+ 베팅
    class Baseline50x(RuleBasedModel):
        def predict(self, df):
            return (df['odds_win'] >= 50).astype(int)

    strategies.insert(0, Baseline50x("Baseline (All 50x+)"))

    # Validation set 평가
    print("\n" + "=" * 80)
    print("Validation Set Evaluation")
    print("=" * 80)

    val_results = []
    for strategy in strategies:
        result = strategy.evaluate(val_df)
        val_results.append(result)

    val_results_df = pd.concat(val_results, ignore_index=True)
    print_evaluation(val_results_df, "Validation Set Results")

    # Test set 평가 (최종)
    print("\n" + "=" * 80)
    print("Test Set Evaluation (Final)")
    print("=" * 80)

    test_results = []
    for strategy in strategies:
        result = strategy.evaluate(test_df)
        test_results.append(result)

    test_results_df = pd.concat(test_results, ignore_index=True)
    print_evaluation(test_results_df, "Test Set Results")

    # 베스트 전략 선택 (Validation ROI 기준)
    val_results_df['ROI_numeric'] = val_results_df['ROI'].str.rstrip('%').astype(float)
    best_idx = val_results_df['ROI_numeric'].idxmax()
    best_strategy = strategies[best_idx]

    print(f"\n🏆 Best Strategy (by Val ROI): {best_strategy.name}")
    print(f"   Val ROI: {val_results_df.iloc[best_idx]['ROI']}")
    print(f"   Test ROI: {test_results_df.iloc[best_idx]['ROI']}")

    # 엘리트 효과 분석
    best_predictions = best_strategy.predict(test_df)
    analyze_elite_effect(test_df, best_predictions)

    # 배당 범위별 분석
    print("\n" + "=" * 80)
    print(f"Odds Range Analysis - {best_strategy.name}")
    print("=" * 80)

    odds_range_df = evaluate_by_odds_range(
        best_predictions,
        test_df['top3'],
        test_df['odds_place'],
        test_df['odds_win']
    )
    print(odds_range_df.to_string(index=False))
    print("=" * 80)

    # 결과 저장
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)

    val_results_df.to_csv(output_dir / 'rule_based_val_results.csv', index=False)
    test_results_df.to_csv(output_dir / 'rule_based_test_results.csv', index=False)

    # 베스트 전략 예측 저장
    test_predictions = best_strategy.predict(test_df)
    test_df['rule_based_prediction'] = test_predictions
    test_df[['trd_dt', 'race_no', 'horse_name', 'odds_win', 'odds_place', 'top3', 'rule_based_prediction']].to_csv(
        output_dir / 'rule_based_predictions.csv', index=False
    )

    print(f"\n✓ Results saved to {output_dir}/")


if __name__ == '__main__':
    main()
