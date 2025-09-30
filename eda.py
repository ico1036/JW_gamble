#!/usr/bin/env python3
"""
경마 데이터 탐색적 데이터 분석 (EDA)
대학원 수준의 체계적인 분석
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
import warnings

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 스타일 설정
sns.set_style("whitegrid")
sns.set_palette("husl")


class HorseRacingEDA:
    """경마 데이터 EDA 클래스"""

    def __init__(self, data_path: str):
        """
        Args:
            data_path: Parquet 파일 경로
        """
        self.data_path = data_path
        self.df = None
        self.output_dir = Path('eda_output')
        self.output_dir.mkdir(exist_ok=True)
        self.figures_dir = self.output_dir / 'figures'
        self.figures_dir.mkdir(exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """데이터 로드"""
        print("=" * 80)
        print("1. 데이터 로딩")
        print("=" * 80)
        self.df = pd.read_parquet(self.data_path)
        print(f"✓ 데이터 로드 완료: {len(self.df):,}개 레코드")
        print(f"✓ 기간: {self.df['trd_dt'].min()} ~ {self.df['trd_dt'].max()}")
        print(f"✓ 컬럼 수: {len(self.df.columns)}개\n")
        return self.df

    def data_quality_report(self) -> Dict:
        """데이터 품질 리포트"""
        print("=" * 80)
        print("2. 데이터 품질 분석")
        print("=" * 80)

        report = {}

        # 기본 정보
        print(f"총 레코드 수: {len(self.df):,}")
        print(f"총 컬럼 수: {len(self.df.columns)}")
        print(f"메모리 사용량: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")

        # 결측치 분석
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        missing_df = pd.DataFrame({
            '결측치 수': missing,
            '결측 비율(%)': missing_pct
        }).query('`결측치 수` > 0').sort_values('결측치 수', ascending=False)

        if len(missing_df) > 0:
            print("결측치 현황:")
            print(missing_df)
        else:
            print("✓ 결측치 없음")

        report['missing'] = missing_df

        # 중복 레코드
        duplicates = self.df.duplicated().sum()
        print(f"\n중복 레코드: {duplicates:,}개 ({duplicates/len(self.df)*100:.2f}%)")
        report['duplicates'] = duplicates

        # 데이터 타입
        print("\n데이터 타입 분포:")
        dtype_counts = self.df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count}개")

        report['dtypes'] = dtype_counts

        # 고유값 분석
        print("\n주요 범주형 변수 고유값:")
        categorical_cols = ['horse_name', 'jockey_name', 'trainer_name',
                            'track', 'grade', 'race_type', 'weather', 'track_cond']
        for col in categorical_cols:
            if col in self.df.columns:
                nunique = self.df[col].nunique()
                print(f"  {col}: {nunique:,}개")

        print()
        return report

    def descriptive_statistics(self) -> pd.DataFrame:
        """기술 통계량"""
        print("=" * 80)
        print("3. 기술 통계량")
        print("=" * 80)

        # 수치형 변수
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        desc = self.df[numeric_cols].describe()

        print("수치형 변수 기술통계:")
        print(desc.round(2))
        print()

        # 추가 통계량 (왜도, 첨도)
        skew = self.df[numeric_cols].skew()
        kurt = self.df[numeric_cols].kurt()

        print("분포 형태:")
        print(f"{'변수':<20} {'왜도(Skewness)':<15} {'첨도(Kurtosis)':<15}")
        print("-" * 50)
        for col in numeric_cols:
            print(f"{col:<20} {skew[col]:>14.3f} {kurt[col]:>14.3f}")

        print()
        return desc

    def analyze_target_variable(self) -> Dict:
        """타겟 변수(finish_pos) 분석"""
        print("=" * 80)
        print("4. 타겟 변수 분석 (finish_pos)")
        print("=" * 80)

        result = {}

        # 분포
        finish_counts = self.df['finish_pos'].value_counts().sort_index()
        print("순위별 분포:")
        print(finish_counts.head(10))

        # 우승 비율
        win_rate = (self.df['finish_pos'] == 1).sum() / len(self.df) * 100
        top3_rate = (self.df['finish_pos'] <= 3).sum() / len(self.df) * 100

        print(f"\n우승(1등) 비율: {win_rate:.2f}%")
        print(f"상위 3위 이내 비율: {top3_rate:.2f}%")

        result['finish_counts'] = finish_counts
        result['win_rate'] = win_rate
        result['top3_rate'] = top3_rate

        # 시각화
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 순위 분포
        axes[0].bar(finish_counts.index[:15], finish_counts.values[:15])
        axes[0].set_xlabel('Finish Position')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Distribution of Finish Positions')
        axes[0].axvline(x=3.5, color='red', linestyle='--', alpha=0.5, label='Top 3')
        axes[0].legend()

        # 우승 vs 비우승
        win_counts = pd.Series([
            (self.df['finish_pos'] == 1).sum(),
            (self.df['finish_pos'] > 1).sum()
        ], index=['Winner', 'Non-winner'])
        axes[1].pie(win_counts, labels=win_counts.index, autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Winner vs Non-winner Distribution')

        plt.tight_layout()
        plt.savefig(self.figures_dir / '01_target_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n✓ 시각화 저장: {self.figures_dir / '01_target_distribution.png'}\n")

        return result

    def analyze_correlations(self) -> pd.DataFrame:
        """상관관계 분석"""
        print("=" * 80)
        print("5. 상관관계 분석")
        print("=" * 80)

        # 수치형 변수 선택
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numeric_cols].corr()

        # finish_pos와의 상관관계
        finish_corr = corr_matrix['finish_pos'].sort_values()
        print("finish_pos와의 상관관계 (Pearson):")
        print(finish_corr)

        # 통계적 유의성 검증
        print("\n통계적 유의성 (p-value < 0.05):")
        for col in numeric_cols:
            if col != 'finish_pos':
                data_clean = self.df[[col, 'finish_pos']].dropna()
                if len(data_clean) > 0:
                    corr, pval = pearsonr(data_clean[col], data_clean['finish_pos'])
                    if pval < 0.05:
                        print(f"  {col}: r={corr:.3f}, p={pval:.4f} ***")

        # 히트맵
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Correlation Matrix of Numerical Variables', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(self.figures_dir / '02_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n✓ 시각화 저장: {self.figures_dir / '02_correlation_heatmap.png'}\n")

        return corr_matrix

    def analyze_horse_performance(self) -> Dict:
        """말 성적 분석"""
        print("=" * 80)
        print("6. 말 성적 분석")
        print("=" * 80)

        result = {}

        # 말별 통계
        horse_stats = self.df.groupby('horse_name').agg({
            'finish_pos': ['count', 'mean', 'min'],
            'trd_dt': ['min', 'max']
        }).reset_index()
        horse_stats.columns = ['horse_name', 'races', 'avg_pos', 'best_pos', 'first_race', 'last_race']

        # 승률 계산
        wins = self.df[self.df['finish_pos'] == 1].groupby('horse_name').size()
        horse_stats['wins'] = horse_stats['horse_name'].map(wins).fillna(0)
        horse_stats['win_rate'] = (horse_stats['wins'] / horse_stats['races'] * 100).round(2)

        # 최소 5회 이상 출전한 말만
        horse_stats_filtered = horse_stats[horse_stats['races'] >= 5].copy()

        print(f"총 말: {len(horse_stats):,}마리")
        print(f"5회 이상 출전: {len(horse_stats_filtered):,}마리")

        # 상위 승률
        top_horses = horse_stats_filtered.nlargest(10, 'win_rate')
        print("\nTop 10 승률 (5회 이상 출전):")
        print(top_horses[['horse_name', 'races', 'wins', 'win_rate', 'avg_pos']])

        result['horse_stats'] = horse_stats
        result['top_horses'] = top_horses

        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 출전 횟수 분포
        axes[0, 0].hist(horse_stats['races'], bins=50, edgecolor='black')
        axes[0, 0].set_xlabel('Number of Races')
        axes[0, 0].set_ylabel('Number of Horses')
        axes[0, 0].set_title('Distribution of Race Counts per Horse')
        axes[0, 0].axvline(horse_stats['races'].median(), color='red', linestyle='--',
                           label=f'Median: {horse_stats["races"].median():.0f}')
        axes[0, 0].legend()

        # 승률 분포 (5회 이상)
        axes[0, 1].hist(horse_stats_filtered['win_rate'], bins=30, edgecolor='black')
        axes[0, 1].set_xlabel('Win Rate (%)')
        axes[0, 1].set_ylabel('Number of Horses')
        axes[0, 1].set_title('Distribution of Win Rates (≥5 races)')

        # 평균 순위 분포
        axes[1, 0].hist(horse_stats_filtered['avg_pos'], bins=30, edgecolor='black')
        axes[1, 0].set_xlabel('Average Finish Position')
        axes[1, 0].set_ylabel('Number of Horses')
        axes[1, 0].set_title('Distribution of Average Finish Positions')

        # Top 10 승률 막대그래프
        top_10 = horse_stats_filtered.nlargest(10, 'win_rate')
        axes[1, 1].barh(range(len(top_10)), top_10['win_rate'])
        axes[1, 1].set_yticks(range(len(top_10)))
        axes[1, 1].set_yticklabels(top_10['horse_name'], fontsize=8)
        axes[1, 1].set_xlabel('Win Rate (%)')
        axes[1, 1].set_title('Top 10 Horses by Win Rate')
        axes[1, 1].invert_yaxis()

        plt.tight_layout()
        plt.savefig(self.figures_dir / '03_horse_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n✓ 시각화 저장: {self.figures_dir / '03_horse_performance.png'}\n")

        return result

    def analyze_jockey_trainer(self) -> Dict:
        """기수 및 조교사 분석"""
        print("=" * 80)
        print("7. 기수 및 조교사 분석")
        print("=" * 80)

        result = {}

        # 기수 통계
        jockey_stats = self.df.groupby('jockey_name').agg({
            'finish_pos': ['count', 'mean'],
        }).reset_index()
        jockey_stats.columns = ['jockey_name', 'races', 'avg_pos']

        wins = self.df[self.df['finish_pos'] == 1].groupby('jockey_name').size()
        jockey_stats['wins'] = jockey_stats['jockey_name'].map(wins).fillna(0)
        jockey_stats['win_rate'] = (jockey_stats['wins'] / jockey_stats['races'] * 100).round(2)

        # 조교사 통계
        trainer_stats = self.df.groupby('trainer_name').agg({
            'finish_pos': ['count', 'mean'],
        }).reset_index()
        trainer_stats.columns = ['trainer_name', 'races', 'avg_pos']

        wins = self.df[self.df['finish_pos'] == 1].groupby('trainer_name').size()
        trainer_stats['wins'] = trainer_stats['trainer_name'].map(wins).fillna(0)
        trainer_stats['win_rate'] = (trainer_stats['wins'] / trainer_stats['races'] * 100).round(2)

        # 최소 출전 횟수 필터
        jockey_filtered = jockey_stats[jockey_stats['races'] >= 50]
        trainer_filtered = trainer_stats[trainer_stats['races'] >= 50]

        print(f"총 기수: {len(jockey_stats)}명 (50회 이상: {len(jockey_filtered)}명)")
        print(f"총 조교사: {len(trainer_stats)}명 (50회 이상: {len(trainer_filtered)}명)")

        # Top 기수
        top_jockeys = jockey_filtered.nlargest(10, 'win_rate')
        print("\nTop 10 기수 (승률, 50회 이상):")
        print(top_jockeys)

        # Top 조교사
        top_trainers = trainer_filtered.nlargest(10, 'win_rate')
        print("\nTop 10 조교사 (승률, 50회 이상):")
        print(top_trainers)

        result['jockey_stats'] = jockey_stats
        result['trainer_stats'] = trainer_stats

        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 기수 승률 Top 10
        axes[0, 0].barh(range(len(top_jockeys)), top_jockeys['win_rate'])
        axes[0, 0].set_yticks(range(len(top_jockeys)))
        axes[0, 0].set_yticklabels(top_jockeys['jockey_name'], fontsize=9)
        axes[0, 0].set_xlabel('Win Rate (%)')
        axes[0, 0].set_title('Top 10 Jockeys by Win Rate (≥50 races)')
        axes[0, 0].invert_yaxis()

        # 기수 평균 순위
        axes[0, 1].scatter(jockey_filtered['races'], jockey_filtered['avg_pos'], alpha=0.6)
        axes[0, 1].set_xlabel('Number of Races')
        axes[0, 1].set_ylabel('Average Finish Position')
        axes[0, 1].set_title('Jockey Experience vs Performance')

        # 조교사 승률 Top 10
        axes[1, 0].barh(range(len(top_trainers)), top_trainers['win_rate'])
        axes[1, 0].set_yticks(range(len(top_trainers)))
        axes[1, 0].set_yticklabels(top_trainers['trainer_name'], fontsize=9)
        axes[1, 0].set_xlabel('Win Rate (%)')
        axes[1, 0].set_title('Top 10 Trainers by Win Rate (≥50 races)')
        axes[1, 0].invert_yaxis()

        # 조교사 평균 순위
        axes[1, 1].scatter(trainer_filtered['races'], trainer_filtered['avg_pos'], alpha=0.6)
        axes[1, 1].set_xlabel('Number of Races')
        axes[1, 1].set_ylabel('Average Finish Position')
        axes[1, 1].set_title('Trainer Experience vs Performance')

        plt.tight_layout()
        plt.savefig(self.figures_dir / '04_jockey_trainer.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n✓ 시각화 저장: {self.figures_dir / '04_jockey_trainer.png'}\n")

        return result

    def analyze_race_conditions(self) -> Dict:
        """경주 조건 분석"""
        print("=" * 80)
        print("8. 경주 조건 분석")
        print("=" * 80)

        result = {}

        # 거리별 분석
        distance_stats = self.df.groupby('distance').agg({
            'finish_pos': 'count',
            'finish_time': 'mean',
            'odds_win': 'mean'
        }).reset_index()
        distance_stats.columns = ['distance', 'races', 'avg_time', 'avg_odds']

        print("거리별 통계:")
        print(distance_stats.sort_values('distance'))

        # 주로 상태별
        if 'track_cond' in self.df.columns:
            track_stats = self.df.groupby('track_cond')['finish_pos'].count()
            print(f"\n주로 상태별 경주 수:")
            print(track_stats)

        # 날씨별
        if 'weather' in self.df.columns:
            weather_stats = self.df.groupby('weather')['finish_pos'].count()
            print(f"\n날씨별 경주 수:")
            print(weather_stats)

        # 등급별
        if 'grade' in self.df.columns:
            grade_stats = self.df.groupby('grade')['finish_pos'].count()
            print(f"\n등급별 경주 수:")
            print(grade_stats)

        result['distance_stats'] = distance_stats

        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 거리별 경주 수
        axes[0, 0].bar(distance_stats['distance'], distance_stats['races'])
        axes[0, 0].set_xlabel('Distance (m)')
        axes[0, 0].set_ylabel('Number of Races')
        axes[0, 0].set_title('Race Count by Distance')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 거리별 평균 시간
        axes[0, 1].plot(distance_stats['distance'], distance_stats['avg_time'], marker='o')
        axes[0, 1].set_xlabel('Distance (m)')
        axes[0, 1].set_ylabel('Average Finish Time (s)')
        axes[0, 1].set_title('Average Finish Time by Distance')
        axes[0, 1].grid(True, alpha=0.3)

        # 주로 상태별 (있는 경우)
        if 'track_cond' in self.df.columns:
            track_counts = self.df['track_cond'].value_counts()
            axes[1, 0].bar(range(len(track_counts)), track_counts.values)
            axes[1, 0].set_xticks(range(len(track_counts)))
            axes[1, 0].set_xticklabels(track_counts.index, rotation=45)
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Race Count by Track Condition')

        # 날씨별 (있는 경우)
        if 'weather' in self.df.columns:
            weather_counts = self.df['weather'].value_counts()
            axes[1, 1].pie(weather_counts, labels=weather_counts.index,
                           autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Race Distribution by Weather')

        plt.tight_layout()
        plt.savefig(self.figures_dir / '05_race_conditions.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n✓ 시각화 저장: {self.figures_dir / '05_race_conditions.png'}\n")

        return result

    def analyze_odds_vs_results(self) -> Dict:
        """배당률 vs 실제 결과 분석"""
        print("=" * 80)
        print("9. 배당률 vs 실제 결과 분석")
        print("=" * 80)

        result = {}

        # 배당률 구간별 승률
        bins = [1, 2, 3, 5, 10, 20, 50, 1000]
        labels = ['1-2', '2-3', '3-5', '5-10', '10-20', '20-50', '50+']
        self.df['odds_bin'] = pd.cut(self.df['odds_win'], bins=bins, labels=labels)

        odds_analysis = self.df.groupby('odds_bin').agg({
            'finish_pos': ['count', lambda x: (x == 1).sum(), lambda x: (x <= 3).sum()]
        }).reset_index()
        odds_analysis.columns = ['odds_bin', 'count', 'wins', 'top3']
        odds_analysis['win_rate'] = (odds_analysis['wins'] / odds_analysis['count'] * 100).round(2)
        odds_analysis['top3_rate'] = (odds_analysis['top3'] / odds_analysis['count'] * 100).round(2)

        print("배당률 구간별 승률:")
        print(odds_analysis)

        # 상관관계
        corr, pval = spearmanr(self.df['odds_win'], self.df['finish_pos'])
        print(f"\n배당률-순위 상관관계 (Spearman): r={corr:.3f}, p={pval:.6f}")

        result['odds_analysis'] = odds_analysis

        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 배당률 분포
        axes[0, 0].hist(self.df['odds_win'], bins=50, edgecolor='black')
        axes[0, 0].set_xlabel('Win Odds')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Distribution of Win Odds')
        axes[0, 0].set_xlim(0, 50)

        # 배당률 구간별 승률
        axes[0, 1].bar(range(len(odds_analysis)), odds_analysis['win_rate'])
        axes[0, 1].set_xticks(range(len(odds_analysis)))
        axes[0, 1].set_xticklabels(odds_analysis['odds_bin'])
        axes[0, 1].set_xlabel('Odds Range')
        axes[0, 1].set_ylabel('Win Rate (%)')
        axes[0, 1].set_title('Win Rate by Odds Range')

        # 배당률 vs 순위 산점도
        sample = self.df.sample(min(5000, len(self.df)))
        axes[1, 0].scatter(sample['odds_win'], sample['finish_pos'], alpha=0.3)
        axes[1, 0].set_xlabel('Win Odds')
        axes[1, 0].set_ylabel('Finish Position')
        axes[1, 0].set_title('Win Odds vs Finish Position')
        axes[1, 0].set_xlim(0, 50)

        # 배당률 구간별 Top3 비율
        axes[1, 1].plot(range(len(odds_analysis)), odds_analysis['top3_rate'],
                        marker='o', linewidth=2, markersize=8)
        axes[1, 1].set_xticks(range(len(odds_analysis)))
        axes[1, 1].set_xticklabels(odds_analysis['odds_bin'])
        axes[1, 1].set_xlabel('Odds Range')
        axes[1, 1].set_ylabel('Top 3 Rate (%)')
        axes[1, 1].set_title('Top 3 Finish Rate by Odds Range')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.figures_dir / '06_odds_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n✓ 시각화 저장: {self.figures_dir / '06_odds_analysis.png'}\n")

        return result

    def analyze_weight_effects(self) -> Dict:
        """마체중 및 증감 효과 분석"""
        print("=" * 80)
        print("10. 마체중 및 증감 효과 분석")
        print("=" * 80)

        result = {}

        # 체중 구간별 분석
        weight_bins = [400, 450, 480, 510, 540, 600]
        weight_labels = ['400-450', '450-480', '480-510', '510-540', '540+']
        self.df['weight_bin'] = pd.cut(self.df['weight'], bins=weight_bins, labels=weight_labels)

        weight_analysis = self.df.groupby('weight_bin').agg({
            'finish_pos': ['count', 'mean', lambda x: (x == 1).sum()]
        }).reset_index()
        weight_analysis.columns = ['weight_bin', 'count', 'avg_pos', 'wins']
        weight_analysis['win_rate'] = (weight_analysis['wins'] / weight_analysis['count'] * 100).round(2)

        print("체중 구간별 성적:")
        print(weight_analysis)

        # 체중 증감 효과
        weight_change_analysis = self.df.groupby('weight_change').agg({
            'finish_pos': ['count', 'mean', lambda x: (x == 1).sum()]
        }).reset_index()
        weight_change_analysis.columns = ['weight_change', 'count', 'avg_pos', 'wins']
        weight_change_analysis['win_rate'] = (
            weight_change_analysis['wins'] / weight_change_analysis['count'] * 100
        ).round(2)
        weight_change_analysis = weight_change_analysis[weight_change_analysis['count'] >= 10]

        print("\n체중 증감별 성적 (10회 이상):")
        print(weight_change_analysis.sort_values('weight_change').head(20))

        # 통계 검정
        normal_weight = self.df[self.df['weight'].between(480, 510)]['finish_pos']
        light_weight = self.df[self.df['weight'] < 480]['finish_pos']
        heavy_weight = self.df[self.df['weight'] > 510]['finish_pos']

        stat, pval = stats.f_oneway(normal_weight, light_weight, heavy_weight)
        print(f"\n체중 그룹 간 순위 차이 (ANOVA): F={stat:.3f}, p={pval:.6f}")

        result['weight_analysis'] = weight_analysis
        result['weight_change_analysis'] = weight_change_analysis

        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 체중 분포
        axes[0, 0].hist(self.df['weight'], bins=40, edgecolor='black')
        axes[0, 0].set_xlabel('Horse Weight (kg)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Distribution of Horse Weight')
        axes[0, 0].axvline(self.df['weight'].mean(), color='red', linestyle='--',
                           label=f'Mean: {self.df["weight"].mean():.0f}kg')
        axes[0, 0].legend()

        # 체중 구간별 평균 순위
        axes[0, 1].bar(range(len(weight_analysis)), weight_analysis['avg_pos'])
        axes[0, 1].set_xticks(range(len(weight_analysis)))
        axes[0, 1].set_xticklabels(weight_analysis['weight_bin'])
        axes[0, 1].set_xlabel('Weight Range (kg)')
        axes[0, 1].set_ylabel('Average Finish Position')
        axes[0, 1].set_title('Average Finish Position by Weight Range')

        # 체중 증감 분포
        axes[1, 0].hist(self.df['weight_change'], bins=50, edgecolor='black')
        axes[1, 0].set_xlabel('Weight Change (kg)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Distribution of Weight Change')
        axes[1, 0].axvline(0, color='red', linestyle='--', label='No change')
        axes[1, 0].legend()

        # 체중 증감별 평균 순위
        wc_plot = weight_change_analysis[(weight_change_analysis['weight_change'] >= -10) &
                                          (weight_change_analysis['weight_change'] <= 10)]
        axes[1, 1].plot(wc_plot['weight_change'], wc_plot['avg_pos'], marker='o')
        axes[1, 1].set_xlabel('Weight Change (kg)')
        axes[1, 1].set_ylabel('Average Finish Position')
        axes[1, 1].set_title('Average Finish Position by Weight Change')
        axes[1, 1].axvline(0, color='red', linestyle='--', alpha=0.3)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.figures_dir / '07_weight_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n✓ 시각화 저장: {self.figures_dir / '07_weight_analysis.png'}\n")

        return result

    def run_full_analysis(self) -> Dict:
        """전체 EDA 실행"""
        print("\n" + "=" * 80)
        print("경마 데이터 탐색적 데이터 분석 (EDA)")
        print("=" * 80 + "\n")

        results = {}

        results['data'] = self.load_data()
        results['quality'] = self.data_quality_report()
        results['descriptive'] = self.descriptive_statistics()
        results['target'] = self.analyze_target_variable()
        results['correlations'] = self.analyze_correlations()
        results['horse'] = self.analyze_horse_performance()
        results['jockey_trainer'] = self.analyze_jockey_trainer()
        results['conditions'] = self.analyze_race_conditions()
        results['odds'] = self.analyze_odds_vs_results()
        results['weight'] = self.analyze_weight_effects()

        print("=" * 80)
        print("EDA 완료!")
        print(f"결과 저장 위치: {self.output_dir}")
        print(f"시각화 파일: {self.figures_dir}")
        print("=" * 80)

        return results