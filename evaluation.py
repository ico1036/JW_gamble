"""
모델 평가 모듈

핵심 지표:
- ROI (Return on Investment): 수익률
- Precision: 정확도
- Expected Value: 기댓값
- Break-even 달성 여부
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


PAYOUT_RATE = 0.8  # 마사회 수수료 20%


def calculate_roi(predictions: pd.Series, actuals: pd.Series, odds: pd.Series,
                  bet_amount: float = 1000) -> Dict:
    """
    ROI 계산

    Args:
        predictions: 예측값 (1: 베팅, 0: 베팅 안함)
        actuals: 실제값 (1: Top3, 0: 실패)
        odds: 복승 배당 (odds_place)
        bet_amount: 베팅 금액 (기본 1000원)

    Returns:
        dict: ROI, 총 수익, 총 베팅액, 정확도 등
    """
    # 베팅한 경우만
    bet_mask = predictions == 1

    if bet_mask.sum() == 0:
        return {
            'roi': 0.0,
            'total_return': 0.0,
            'total_bet': 0.0,
            'profit': 0.0,
            'precision': 0.0,
            'num_bets': 0,
            'num_wins': 0
        }

    bet_actuals = actuals[bet_mask]
    bet_odds = odds[bet_mask]

    # 수익 계산
    num_bets = bet_mask.sum()
    num_wins = bet_actuals.sum()

    total_bet = num_bets * bet_amount
    total_return = (bet_actuals * bet_odds * bet_amount * PAYOUT_RATE).sum()
    profit = total_return - total_bet
    roi = (profit / total_bet * 100) if total_bet > 0 else 0

    precision = num_wins / num_bets if num_bets > 0 else 0

    return {
        'roi': roi,
        'total_return': total_return,
        'total_bet': total_bet,
        'profit': profit,
        'precision': precision,
        'num_bets': num_bets,
        'num_wins': num_wins
    }


def calculate_breakeven_precision(odds: pd.Series) -> float:
    """
    Break-even precision 계산

    필요 정확도 = 1 / (평균 배당 × 환급률)
    """
    avg_odds = odds.mean()
    breakeven = 1 / (avg_odds * PAYOUT_RATE)
    return breakeven


def calculate_expected_value(predictions: pd.Series, actuals: pd.Series,
                            odds: pd.Series) -> Dict:
    """
    기댓값 계산

    EV = (승리확률 × 배당 × 환급률) - 1
    EV > 0: 기댓값 플러스 (장기적 수익)
    EV < 0: 기댓값 마이너스 (장기적 손해)
    """
    bet_mask = predictions == 1

    if bet_mask.sum() == 0:
        return {'ev': 0.0, 'positive_ev_ratio': 0.0, 'avg_odds': 0.0, 'precision': 0.0}

    bet_actuals = actuals[bet_mask]
    bet_odds = odds[bet_mask]

    precision = bet_actuals.mean()
    avg_odds = bet_odds.mean()

    ev = (precision * avg_odds * PAYOUT_RATE) - 1

    # 각 베팅의 개별 EV
    individual_evs = (bet_actuals * bet_odds * PAYOUT_RATE) - 1
    positive_ev_ratio = (individual_evs > 0).mean()

    return {
        'ev': ev,
        'positive_ev_ratio': positive_ev_ratio,
        'avg_odds': avg_odds,
        'precision': precision
    }


def evaluate_strategy(predictions: pd.Series, actuals: pd.Series,
                     odds: pd.Series, strategy_name: str = "Strategy") -> pd.DataFrame:
    """
    전략 종합 평가

    Args:
        predictions: 예측값
        actuals: 실제값
        odds: 배당
        strategy_name: 전략 이름

    Returns:
        DataFrame: 평가 결과
    """
    roi_metrics = calculate_roi(predictions, actuals, odds)
    ev_metrics = calculate_expected_value(predictions, actuals, odds)

    bet_mask = predictions == 1
    if bet_mask.sum() > 0:
        breakeven = calculate_breakeven_precision(odds[bet_mask])
    else:
        breakeven = 0.0

    results = {
        'Strategy': strategy_name,
        'Num Bets': roi_metrics['num_bets'],
        'Precision': f"{roi_metrics['precision']:.1%}",
        'Breakeven': f"{breakeven:.1%}",
        'Gap': f"{(roi_metrics['precision'] - breakeven):.1%}",
        'ROI': f"{roi_metrics['roi']:.2f}%",
        'EV': f"{ev_metrics['ev']:.4f}",
        'Avg Odds': f"{ev_metrics['avg_odds']:.2f}x",
        'Total Bet': f"{roi_metrics['total_bet']:,.0f}원",
        'Total Return': f"{roi_metrics['total_return']:,.0f}원",
        'Profit': f"{roi_metrics['profit']:,.0f}원"
    }

    return pd.DataFrame([results])


def evaluate_by_odds_range(predictions: pd.Series, actuals: pd.Series,
                          odds: pd.Series, odds_win: pd.Series) -> pd.DataFrame:
    """배당 범위별 평가"""
    odds_ranges = [
        (0, 2, '1-2x'),
        (2, 3, '2-3x'),
        (3, 5, '3-5x'),
        (5, 10, '5-10x'),
        (10, 20, '10-20x'),
        (20, 50, '20-50x'),
        (50, 1000, '50x+')
    ]

    results = []

    for min_odds, max_odds, label in odds_ranges:
        mask = (odds_win >= min_odds) & (odds_win < max_odds) & (predictions == 1)

        if mask.sum() > 0:
            range_preds = predictions[mask]
            range_actuals = actuals[mask]
            range_odds = odds[mask]

            roi_metrics = calculate_roi(range_preds, range_actuals, range_odds)

            results.append({
                'Odds Range': label,
                'Num Bets': roi_metrics['num_bets'],
                'Precision': f"{roi_metrics['precision']:.1%}",
                'ROI': f"{roi_metrics['roi']:.2f}%",
                'Avg Odds': f"{range_odds.mean():.2f}x",
                'Profit': f"{roi_metrics['profit']:,.0f}원"
            })

    return pd.DataFrame(results)


def print_evaluation(eval_df: pd.DataFrame, title: str = "Evaluation Results"):
    """평가 결과 출력"""
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)
    print(eval_df.to_string(index=False))
    print("=" * 80)


def analyze_elite_effect(df: pd.DataFrame, predictions: pd.Series):
    """엘리트 기수/조련사 효과 분석"""

    # 인덱스 일치
    predictions = predictions.reset_index(drop=True)
    df_reset = df.reset_index(drop=True)

    # 50배+ 고배당만
    high_odds_mask = (df_reset['odds_win'] >= 50) & (predictions == 1)
    high_odds_df = df_reset[high_odds_mask].copy()

    if len(high_odds_df) == 0:
        print("\n⚠️  50배+ 베팅이 없습니다.")
        return

    print("\n" + "=" * 80)
    print("50배+ 고배당 엘리트 효과 분석")
    print("=" * 80)

    # 엘리트 조합별 분석
    cases = [
        ('Elite Jockey + Elite Trainer',
         (high_odds_df['jockey_elite'] == 1) & (high_odds_df['trainer_elite'] == 1)),
        ('Elite Jockey Only',
         (high_odds_df['jockey_elite'] == 1) & (high_odds_df['trainer_elite'] == 0)),
        ('Elite Trainer Only',
         (high_odds_df['jockey_elite'] == 0) & (high_odds_df['trainer_elite'] == 1)),
        ('No Elite',
         (high_odds_df['jockey_elite'] == 0) & (high_odds_df['trainer_elite'] == 0)),
    ]

    results = []
    for case_name, mask in cases:
        if mask.sum() > 0:
            case_df = high_odds_df[mask]
            top3_rate = case_df['top3'].mean()
            avg_odds = case_df['odds_place'].mean()

            # ROI 계산
            roi_metrics = calculate_roi(
                pd.Series([1] * len(case_df)),
                case_df['top3'].reset_index(drop=True),
                case_df['odds_place'].reset_index(drop=True)
            )

            results.append({
                'Case': case_name,
                'Count': len(case_df),
                'Top3 Rate': f"{top3_rate:.1%}",
                'Avg Odds': f"{avg_odds:.2f}x",
                'ROI': f"{roi_metrics['roi']:.2f}%"
            })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("=" * 80)


if __name__ == '__main__':
    # 테스트 코드
    np.random.seed(42)

    # 샘플 데이터 생성
    n = 1000
    predictions = pd.Series(np.random.binomial(1, 0.3, n))
    actuals = pd.Series(np.random.binomial(1, 0.1, n))
    odds = pd.Series(np.random.uniform(5, 20, n))

    # 평가
    eval_df = evaluate_strategy(predictions, actuals, odds, "Test Strategy")
    print_evaluation(eval_df)
