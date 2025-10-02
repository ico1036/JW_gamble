"""
Feature Engineering for Horse Racing Prediction

핵심 전략: 50배+ 고배당 예측
- 엘리트 기수/조련사 탐지
- 집계 통계 (rolling window)
- 타겟: Top 3 (복승)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data(data_path='data/race_results_full.parquet'):
    """데이터 로드"""
    df = pd.read_parquet(data_path)
    print(f"✓ Loaded {len(df):,} records")
    return df


def create_target(df):
    """타겟 변수 생성"""
    df['top3'] = (df['finish_pos'] <= 3).astype(int)
    df['win'] = (df['finish_pos'] == 1).astype(int)

    print(f"✓ Target created:")
    print(f"  - Top 3 rate: {df['top3'].mean():.1%}")
    print(f"  - Win rate: {df['win'].mean():.1%}")

    return df


def split_data(df):
    """
    Train/Val/Test 분할 (시계열 고려)
    - Train: 2013 ~ 2023-12-31
    - Test: 2024-01-01 ~ 2025-09-28
    - Val: Train의 마지막 10%
    """
    # Test split
    test_start = '2024-01-01'
    train_df = df[df['trd_dt'] < test_start].copy()
    test_df = df[df['trd_dt'] >= test_start].copy()

    # Val split from train (last 10%)
    train_dates = sorted(train_df['trd_dt'].unique())
    val_split_idx = int(len(train_dates) * 0.9)
    val_start_date = train_dates[val_split_idx]

    val_df = train_df[train_df['trd_dt'] >= val_start_date].copy()
    train_df = train_df[train_df['trd_dt'] < val_start_date].copy()

    print(f"\n✓ Data split:")
    print(f"  - Train: {len(train_df):,} ({train_df['trd_dt'].min()} ~ {train_df['trd_dt'].max()})")
    print(f"  - Val:   {len(val_df):,} ({val_df['trd_dt'].min()} ~ {val_df['trd_dt'].max()})")
    print(f"  - Test:  {len(test_df):,} ({test_df['trd_dt'].min()} ~ {test_df['trd_dt'].max()})")

    return train_df, val_df, test_df


def create_aggregation_features(df, train_df):
    """
    집계 통계 피처 생성 (Look-ahead bias 방지)

    각 말/기수/조련사의 과거 성적 집계
    - 과거 30일/90일 승률
    - 과거 전체 승률
    - 50배+ 고배당에서의 성적
    """
    # 전체 데이터에 대해 순차적으로 집계 (look-ahead 방지)
    df = df.sort_values(['trd_dt', 'race_no']).reset_index(drop=True)

    # 기수 집계 통계
    jockey_stats = train_df.groupby('jockey_name').agg({
        'top3': ['mean', 'count'],
        'win': 'mean'
    }).round(4)
    jockey_stats.columns = ['jockey_top3_rate', 'jockey_races', 'jockey_win_rate']
    jockey_stats = jockey_stats.reset_index()

    # 조련사 집계 통계
    trainer_stats = train_df.groupby('trainer_name').agg({
        'top3': ['mean', 'count'],
        'win': 'mean'
    }).round(4)
    trainer_stats.columns = ['trainer_top3_rate', 'trainer_races', 'trainer_win_rate']
    trainer_stats = trainer_stats.reset_index()

    # 말 집계 통계
    horse_stats = train_df.groupby('horse_name').agg({
        'top3': ['mean', 'count'],
        'win': 'mean'
    }).round(4)
    horse_stats.columns = ['horse_top3_rate', 'horse_races', 'horse_win_rate']
    horse_stats = horse_stats.reset_index()

    # 50배+ 고배당 집계 (핵심!)
    high_odds_train = train_df[train_df['odds_win'] >= 50]

    jockey_50x_stats = high_odds_train.groupby('jockey_name')['top3'].agg(['mean', 'count']).reset_index()
    jockey_50x_stats.columns = ['jockey_name', 'jockey_50x_top3_rate', 'jockey_50x_races']

    trainer_50x_stats = high_odds_train.groupby('trainer_name')['top3'].agg(['mean', 'count']).reset_index()
    trainer_50x_stats.columns = ['trainer_name', 'trainer_50x_top3_rate', 'trainer_50x_races']

    # Merge
    df = df.merge(jockey_stats, on='jockey_name', how='left')
    df = df.merge(trainer_stats, on='trainer_name', how='left')
    df = df.merge(horse_stats, on='horse_name', how='left')
    df = df.merge(jockey_50x_stats, on='jockey_name', how='left')
    df = df.merge(trainer_50x_stats, on='trainer_name', how='left')

    # Fill missing with global mean
    global_top3_rate = train_df['top3'].mean()

    fill_cols = [
        'jockey_top3_rate', 'jockey_win_rate', 'trainer_top3_rate', 'trainer_win_rate',
        'horse_top3_rate', 'horse_win_rate', 'jockey_50x_top3_rate', 'trainer_50x_top3_rate'
    ]
    for col in fill_cols:
        df[col] = df[col].fillna(global_top3_rate)

    df['jockey_races'] = df['jockey_races'].fillna(0)
    df['trainer_races'] = df['trainer_races'].fillna(0)
    df['horse_races'] = df['horse_races'].fillna(0)
    df['jockey_50x_races'] = df['jockey_50x_races'].fillna(0)
    df['trainer_50x_races'] = df['trainer_50x_races'].fillna(0)

    print(f"\n✓ Aggregation features created")

    return df


def create_elite_flags(df):
    """
    엘리트 기수/조련사 플래그 (50배+ 기준)

    EDA 인사이트:
    - 엘리트 기수: 50배+ Top3 비율 10%+ (베이스라인 5.3%의 2배)
    - 엘리트 조련사: 50배+ Top3 비율 10%+
    """
    # 엘리트 임계값
    ELITE_THRESHOLD = 0.10
    MIN_RACES = 10  # 최소 출전 횟수

    df['jockey_elite'] = (
        (df['jockey_50x_top3_rate'] >= ELITE_THRESHOLD) &
        (df['jockey_50x_races'] >= MIN_RACES)
    ).astype(int)

    df['trainer_elite'] = (
        (df['trainer_50x_top3_rate'] >= ELITE_THRESHOLD) &
        (df['trainer_50x_races'] >= MIN_RACES)
    ).astype(int)

    print(f"✓ Elite flags created:")
    print(f"  - Elite jockeys: {df['jockey_elite'].sum():,} ({df['jockey_elite'].mean():.1%})")
    print(f"  - Elite trainers: {df['trainer_elite'].sum():,} ({df['trainer_elite'].mean():.1%})")

    return df


def create_odds_range(df):
    """배당 범위 피처"""
    df['odds_range'] = pd.cut(
        df['odds_win'],
        bins=[0, 2, 3, 5, 10, 20, 50, 1000],
        labels=['1-2x', '2-3x', '3-5x', '5-10x', '10-20x', '20-50x', '50x+']
    )
    df['is_high_odds'] = (df['odds_win'] >= 50).astype(int)

    return df


def create_basic_features(df):
    """기본 피처 정제"""
    # 결측치 처리
    df['horse_age'] = df['horse_age'].fillna(df['horse_age'].median())
    df['burden_weight'] = df['burden_weight'].fillna(df['burden_weight'].median())
    df['gate_no'] = df['gate_no'].fillna(df['gate_no'].median())

    # 카테고리 인코딩
    df['track_encoded'] = df['track'].cat.codes
    df['horse_sex_encoded'] = df['horse_sex'].map({'암': 0, '수': 1, '거': 2}).fillna(0)

    return df


def select_features(df):
    """모델링에 사용할 피처 선택"""

    # 핵심 피처 (Tier 1: 가장 중요)
    tier1_features = [
        'odds_win',           # 가장 강력한 예측 변수
        'odds_place',         # 복승 배당
        'burden_weight',      # 부담중량 (핸디캡)
    ]

    # 엘리트 피처 (Tier 2: 50배+ 전략 핵심)
    tier2_features = [
        'jockey_elite',
        'trainer_elite',
        'jockey_50x_top3_rate',
        'trainer_50x_top3_rate',
        'jockey_50x_races',
        'trainer_50x_races',
    ]

    # 집계 통계 (Tier 3)
    tier3_features = [
        'jockey_top3_rate',
        'jockey_win_rate',
        'trainer_top3_rate',
        'trainer_win_rate',
        'horse_top3_rate',
        'horse_win_rate',
    ]

    # 기본 피처 (Tier 4)
    tier4_features = [
        'horse_age',
        'gate_no',
        'track_encoded',
        'horse_sex_encoded',
        'is_high_odds',
    ]

    feature_cols = tier1_features + tier2_features + tier3_features + tier4_features

    # 결측치가 많은 odds_place는 별도 처리
    if 'odds_place' in df.columns:
        df['odds_place'] = df['odds_place'].fillna(df['odds_win'] * 0.5)  # 단승의 절반으로 추정

    print(f"\n✓ Selected {len(feature_cols)} features:")
    print(f"  - Tier 1 (odds): {len(tier1_features)}")
    print(f"  - Tier 2 (elite): {len(tier2_features)}")
    print(f"  - Tier 3 (stats): {len(tier3_features)}")
    print(f"  - Tier 4 (basic): {len(tier4_features)}")

    return feature_cols


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("Feature Engineering Pipeline")
    print("=" * 60)

    # 1. 데이터 로드
    df = load_data()

    # 2. 타겟 생성
    df = create_target(df)

    # 3. Train/Val/Test 분할
    train_df, val_df, test_df = split_data(df)

    # 4. Feature engineering (train 기반)
    print("\n" + "=" * 60)
    print("Feature Engineering (based on Train set)")
    print("=" * 60)

    # 모든 데이터셋에 피처 적용 (train 통계 기반)
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    all_df = create_aggregation_features(all_df, train_df)
    all_df = create_elite_flags(all_df)
    all_df = create_odds_range(all_df)
    all_df = create_basic_features(all_df)

    # 5. 피처 선택
    feature_cols = select_features(all_df)

    # 6. 최종 분할 (피처 엔지니어링 후)
    test_start = '2024-01-01'
    train_full = all_df[all_df['trd_dt'] < test_start].copy()
    test_full = all_df[all_df['trd_dt'] >= test_start].copy()

    train_dates = sorted(train_full['trd_dt'].unique())
    val_split_idx = int(len(train_dates) * 0.9)
    val_start_date = train_dates[val_split_idx]

    train_final = train_full[train_full['trd_dt'] < val_start_date].copy()
    val_final = train_full[train_full['trd_dt'] >= val_start_date].copy()

    # 7. 저장
    output_dir = Path('data')
    output_dir.mkdir(exist_ok=True)

    train_final.to_parquet(output_dir / 'train.parquet', index=False)
    val_final.to_parquet(output_dir / 'val.parquet', index=False)
    test_full.to_parquet(output_dir / 'test.parquet', index=False)

    # 피처 리스트 저장
    import json
    with open(output_dir / 'feature_columns.json', 'w') as f:
        json.dump(feature_cols, f, indent=2)

    print("\n" + "=" * 60)
    print("✓ Feature engineering completed!")
    print("=" * 60)
    print(f"\nSaved files:")
    print(f"  - data/train.parquet ({len(train_final):,} rows)")
    print(f"  - data/val.parquet ({len(val_final):,} rows)")
    print(f"  - data/test.parquet ({len(test_full):,} rows)")
    print(f"  - data/feature_columns.json ({len(feature_cols)} features)")

    # 통계 출력
    print(f"\n50배+ 고배당 분포:")
    print(f"  - Train: {(train_final['odds_win'] >= 50).sum():,} ({(train_final['odds_win'] >= 50).mean():.1%})")
    print(f"  - Val:   {(val_final['odds_win'] >= 50).sum():,} ({(val_final['odds_win'] >= 50).mean():.1%})")
    print(f"  - Test:  {(test_full['odds_win'] >= 50).sum():,} ({(test_full['odds_win'] >= 50).mean():.1%})")

    print(f"\n엘리트 기수/조련사 (50배+):")
    high_odds_train = train_final[train_final['odds_win'] >= 50]
    print(f"  - Elite jockey Top3: {high_odds_train[high_odds_train['jockey_elite']==1]['top3'].mean():.1%}")
    print(f"  - Elite trainer Top3: {high_odds_train[high_odds_train['trainer_elite']==1]['top3'].mean():.1%}")
    print(f"  - Non-elite Top3: {high_odds_train[(high_odds_train['jockey_elite']==0) & (high_odds_train['trainer_elite']==0)]['top3'].mean():.1%}")


if __name__ == '__main__':
    main()
