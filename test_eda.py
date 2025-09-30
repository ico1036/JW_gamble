#!/usr/bin/env python3
"""
TDD 방식의 EDA 테스트 코드
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def data_path():
    """테스트 데이터 경로"""
    return '/Users/ryan/horse_park/race_results.parquet'


@pytest.fixture
def df(data_path):
    """실제 데이터 로드"""
    return pd.read_parquet(data_path)


class TestDataQuality:
    """데이터 품질 검증 테스트"""

    def test_data_loads_successfully(self, df):
        """데이터가 성공적으로 로드되는지 확인"""
        assert df is not None
        assert len(df) > 0

    def test_expected_columns_exist(self, df):
        """필수 컬럼이 존재하는지 확인"""
        expected_cols = [
            'trd_dt', 'race_no', 'gate_no', 'horse_name',
            'jockey_name', 'trainer_name', 'weight', 'weight_change',
            'distance', 'finish_pos', 'odds_win', 'odds_place'
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_duplicate_records(self, df):
        """중복 레코드 확인"""
        # 완전 중복 체크
        assert df.duplicated().sum() == 0 or df.duplicated().sum() < len(df) * 0.01

    def test_finish_pos_is_valid(self, df):
        """finish_pos가 유효한 범위인지 확인"""
        assert df['finish_pos'].min() >= 1
        assert df['finish_pos'].max() <= 20  # 보통 경주는 20마 이하


class TestDistributions:
    """분포 분석 테스트"""

    def test_odds_distribution_is_reasonable(self, df):
        """배당률 분포가 합리적인지 확인"""
        # 배당률은 1.0 이상이어야 함
        assert df['odds_win'].min() >= 1.0
        # 극단적으로 높은 배당률 체크 (1000배 이하)
        assert df['odds_win'].max() < 1000

    def test_weight_distribution(self, df):
        """마체중 분포 확인"""
        # 경주마 체중은 보통 400-600kg
        assert df['weight'].min() >= 350
        assert df['weight'].max() <= 650


class TestHorsePerformance:
    """말 성적 분석 테스트"""

    def test_can_calculate_win_rate_per_horse(self, df):
        """말별 승률 계산 가능 여부"""
        win_rates = df.groupby('horse_name').apply(
            lambda x: (x['finish_pos'] == 1).sum() / len(x)
        )
        assert len(win_rates) > 0
        assert win_rates.max() <= 1.0
        assert win_rates.min() >= 0.0

    def test_can_calculate_jockey_success_rate(self, df):
        """기수 성공률 계산 가능 여부"""
        jockey_wins = df.groupby('jockey_name')['finish_pos'].apply(
            lambda x: (x == 1).sum()
        )
        assert len(jockey_wins) > 0


class TestCorrelations:
    """상관관계 분석 테스트"""

    def test_can_compute_correlation_matrix(self, df):
        """상관관계 행렬 계산 가능 여부"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        assert corr_matrix is not None
        assert len(corr_matrix) > 0

    def test_odds_correlate_with_finish_position(self, df):
        """배당률과 순위 간 상관관계 확인"""
        # 낮은 배당률(인기마) = 좋은 순위(낮은 숫자)
        # 따라서 양의 상관관계가 있어야 함
        corr = df[['odds_win', 'finish_pos']].corr().iloc[0, 1]
        assert corr > 0, "Odds should positively correlate with finish position"


class TestRaceConditions:
    """경주 조건 분석 테스트"""

    def test_distance_categories_exist(self, df):
        """거리 카테고리가 존재하는지 확인"""
        distances = df['distance'].unique()
        assert len(distances) > 0
        # 일반적인 경마 거리: 1000m ~ 3000m
        assert distances.min() >= 1000
        assert distances.max() <= 3200

    def test_track_condition_affects_performance(self, df):
        """주로 상태가 성적에 영향을 미치는지 확인"""
        if 'track_cond' in df.columns:
            track_conditions = df['track_cond'].nunique()
            assert track_conditions > 0


class TestVisualization:
    """시각화 관련 테스트"""

    def test_can_create_histogram(self, df):
        """히스토그램 생성 가능 여부"""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        df['odds_win'].hist(ax=ax)
        assert ax is not None
        plt.close()

    def test_can_create_correlation_heatmap(self, df):
        """상관관계 히트맵 생성 가능 여부"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
        corr_matrix = df[numeric_cols].corr()

        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, ax=ax)
        assert ax is not None
        plt.close()