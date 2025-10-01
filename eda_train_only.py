"""
EDA - Train Set Only (2013-2023)
Look-ahead bias 완전 제거

절대 규칙: Test set (2024-2025) 절대 안 봄!
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_style('whitegrid')


def main():
    print("=" * 100)
    print("EDA - Train Set Only (2013-2023)")
    print("Look-ahead bias 제거 - Test set 절대 안 봄!")
    print("=" * 100)

    # 데이터 로드
    df = pd.read_parquet('data/race_results_full.parquet')

    # Train/Test Split (시계열)
    train = df[df['trd_dt'].dt.year <= 2023].copy()
    test = df[df['trd_dt'].dt.year >= 2024].copy()

    print(f"\n전체 데이터: {len(df):,}개")
    print(f"  기간: {df['trd_dt'].min()} ~ {df['trd_dt'].max()}")

    print(f"\n✓ Train set (2013-2023): {len(train):,}개")
    print(f"  기간: {train['trd_dt'].min()} ~ {train['trd_dt'].max()}")

    print(f"\n✗ Test set (2024-2025): {len(test):,}개 - 절대 안 봄!")
    print(f"  기간: {test['trd_dt'].min()} ~ {test['trd_dt'].max()}")

    # 이제부터 Train만 사용!
    df = train

    # Top 3 여부
    df['is_top3'] = (df['finish_pos'] <= 3).astype(int)

    # 배당 범위
    df['odds_range'] = pd.cut(
        df['odds_win'],
        bins=[0, 2, 3, 5, 10, 20, 50, 1000],
        labels=['1-2x', '2-3x', '3-5x', '5-10x', '10-20x', '20-50x', '50x+']
    )

    # ========================================
    # 1. 배당 범위별 Top 3 비율
    # ========================================
    print("\n" + "=" * 100)
    print("[1] 배당 범위별 Top 3 비율 (Train Set)")
    print("=" * 100)

    odds_stats = df.groupby('odds_range', observed=True).agg({
        'is_top3': ['sum', 'count', 'mean'],
        'odds_place': 'mean'
    }).reset_index()
    odds_stats.columns = ['odds_range', 'top3_count', 'total', 'top3_rate', 'avg_place_odds']

    # Break-even precision (환급률 80%)
    odds_stats['breakeven_precision'] = 1.0 / (odds_stats['avg_place_odds'] * 0.8)
    odds_stats['gap'] = odds_stats['breakeven_precision'] - odds_stats['top3_rate']

    print(f"\n{'배당 범위':>10s} {'전체':>8s} {'Top3':>8s} {'Top3비율':>10s} {'복승배당':>10s} {'필요정확도':>12s} {'부족분':>10s}")
    print("-" * 100)
    for _, row in odds_stats.iterrows():
        gap_str = f"{row['gap']:+.1%}" if row['gap'] > 0 else f"✓ {row['gap']:+.1%}"
        print(f"{row['odds_range']:>10s} {row['total']:8.0f} {row['top3_count']:8.0f} "
              f"{row['top3_rate']:9.1%} {row['avg_place_odds']:9.2f}배 "
              f"{row['breakeven_precision']:11.1%} {gap_str:>12s}")

    # ========================================
    # 2. 고배당 말 (10배+) 상세 분석
    # ========================================
    for min_odds, label in [(10, '10배+'), (20, '20배+'), (50, '50배+')]:
        print(f"\n" + "=" * 100)
        print(f"[2] {label} 고배당 말 분석")
        print("=" * 100)

        df_longshot = df[df['odds_win'] >= min_odds].copy()

        baseline_top3 = df_longshot['is_top3'].mean()
        avg_place_odds = df_longshot['odds_place'].mean()
        breakeven = 1.0 / (avg_place_odds * 0.8)

        print(f"\n전체: {len(df_longshot):,}마리")
        print(f"Top 3: {df_longshot['is_top3'].sum():,}마리 ({baseline_top3:.1%})")
        print(f"평균 복승 배당: {avg_place_odds:.2f}배")
        print(f"Break-even 필요: {breakeven:.1%}")
        print(f"향상 필요: {breakeven - baseline_top3:+.1%}p")

        # 특징별 분석
        print(f"\n--- 말 나이별 Top 3 비율 ---")
        age_stats = df_longshot.groupby('horse_age')['is_top3'].agg(['sum', 'count', 'mean']).reset_index()
        age_stats = age_stats[age_stats['count'] >= 100].sort_values('mean', ascending=False)
        print(f"{'나이':>6s} {'전체':>8s} {'Top3':>8s} {'비율':>8s} {'향상':>10s}")
        for _, row in age_stats.head(10).iterrows():
            improvement = row['mean'] - baseline_top3
            print(f"{row['horse_age']:>6.0f} {row['count']:8.0f} {row['sum']:8.0f} "
                  f"{row['mean']:7.1%} {improvement:+9.1%}p")

        # 부담중량별
        print(f"\n--- 부담중량별 Top 3 비율 ---")
        df_longshot['burden_bin'] = pd.cut(df_longshot['burden_weight'], bins=[45, 52, 54, 56, 58, 61])
        burden_stats = df_longshot.groupby('burden_bin', observed=True)['is_top3'].agg(['sum', 'count', 'mean']).reset_index()
        burden_stats = burden_stats[burden_stats['count'] >= 100].sort_values('mean', ascending=False)
        print(f"{'부담중량':>10s} {'전체':>8s} {'Top3':>8s} {'비율':>8s} {'향상':>10s}")
        for _, row in burden_stats.iterrows():
            improvement = row['mean'] - baseline_top3
            print(f"{str(row['burden_bin']):>10s} {row['count']:8.0f} {row['sum']:8.0f} "
                  f"{row['mean']:7.1%} {improvement:+9.1%}p")

        # 게이트별
        print(f"\n--- 게이트 위치별 Top 3 비율 ---")
        df_longshot['gate_bin'] = pd.cut(df_longshot['gate_no'], bins=[0, 3, 6, 9, 17])
        gate_stats = df_longshot.groupby('gate_bin', observed=True)['is_top3'].agg(['sum', 'count', 'mean']).reset_index()
        gate_stats = gate_stats[gate_stats['count'] >= 100].sort_values('mean', ascending=False)
        print(f"{'게이트':>10s} {'전체':>8s} {'Top3':>8s} {'비율':>8s} {'향상':>10s}")
        for _, row in gate_stats.iterrows():
            improvement = row['mean'] - baseline_top3
            print(f"{str(row['gate_bin']):>10s} {row['count']:8.0f} {row['sum']:8.0f} "
                  f"{row['mean']:7.1%} {improvement:+9.1%}p")

        # 기수별 (10번 이상 출전)
        print(f"\n--- 기수별 Top 3 비율 (10번 이상) ---")
        jockey_stats = df_longshot.groupby('jockey_name')['is_top3'].agg(['sum', 'count', 'mean']).reset_index()
        jockey_stats = jockey_stats[jockey_stats['count'] >= 10].sort_values('mean', ascending=False)
        print(f"{'기수':>10s} {'전체':>8s} {'Top3':>8s} {'비율':>8s} {'향상':>10s}")
        for _, row in jockey_stats.head(10).iterrows():
            improvement = row['mean'] - baseline_top3
            print(f"{row['jockey_name']:>10s} {row['count']:8.0f} {row['sum']:8.0f} "
                  f"{row['mean']:7.1%} {improvement:+9.1%}p")

        # 조련사별
        print(f"\n--- 조련사별 Top 3 비율 (10번 이상) ---")
        trainer_stats = df_longshot.groupby('trainer_name')['is_top3'].agg(['sum', 'count', 'mean']).reset_index()
        trainer_stats = trainer_stats[trainer_stats['count'] >= 10].sort_values('mean', ascending=False)
        print(f"{'조련사':>10s} {'전체':>8s} {'Top3':>8s} {'비율':>8s} {'향상':>10s}")
        for _, row in trainer_stats.head(10).iterrows():
            improvement = row['mean'] - baseline_top3
            print(f"{row['trainer_name']:>10s} {row['count']:8.0f} {row['sum']:8.0f} "
                  f"{row['mean']:7.1%} {improvement:+9.1%}p")

    # ========================================
    # 3. 핵심 인사이트
    # ========================================
    print(f"\n" + "=" * 100)
    print("[3] 핵심 인사이트 (Train Set 2013-2023)")
    print("=" * 100)

    print(f"\n1. 수익 가능한 배당 범위:")
    profitable = odds_stats[odds_stats['gap'] < 0]
    if len(profitable) > 0:
        for _, row in profitable.iterrows():
            print(f"   {row['odds_range']}: Top 3 {row['top3_rate']:.1%} >= 필요 {row['breakeven_precision']:.1%}")
    else:
        print("   없음 - 모든 범위에서 부족")

    print(f"\n2. 가장 유망한 배당 범위:")
    most_promising = odds_stats.nsmallest(3, 'gap')
    for _, row in most_promising.iterrows():
        print(f"   {row['odds_range']}: 부족분 {row['gap']:.1%}p (가장 달성 가능)")

    print(f"\n3. Train Set 통계:")
    print(f"   - 총 경주일: {df['trd_dt'].nunique()}일")
    print(f"   - 총 경주: {len(df):,}회")
    print(f"   - 평균 Top 3 비율: {df['is_top3'].mean():.1%}")
    print(f"   - 말: {df['horse_name'].nunique():,}마리")
    print(f"   - 기수: {df['jockey_name'].nunique()}명")
    print(f"   - 조련사: {df['trainer_name'].nunique()}명")

    print(f"\n" + "=" * 100)
    print("EDA 완료 - Test set은 절대 안 봄!")
    print("다음: 모델링 전략 수립")
    print("=" * 100)


if __name__ == '__main__':
    main()
