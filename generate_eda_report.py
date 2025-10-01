#!/usr/bin/env python3
"""
EDA 리포트 생성 - 시각화 포함
Train Set Only (2013-2023)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path
import platform

# 한글 폰트 설정
def setup_korean_font():
    """한글 폰트 자동 설정"""
    system = platform.system()

    if system == 'Darwin':  # macOS
        font_path = '/System/Library/Fonts/AppleSDGothicNeo.ttc'
        if Path(font_path).exists():
            # 폰트 파일 직접 등록
            font_entry = fm.FontEntry(fname=font_path, name='AppleSDGothicNeo')
            fm.fontManager.ttflist.append(font_entry)

            # rcParams 설정
            plt.rcParams['font.family'] = 'AppleSDGothicNeo'
            plt.rcParams['font.sans-serif'] = ['AppleSDGothicNeo', 'Apple SD Gothic Neo']
            plt.rcParams['axes.unicode_minus'] = False

            print(f"✓ 한글 폰트 설정: AppleSDGothicNeo ({font_path})")
            return True
        else:
            print(f"⚠ AppleSDGothicNeo 폰트를 찾을 수 없습니다")

    # 기본 설정
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    return False

sns.set_style('whitegrid')
sns.set_palette('husl')
setup_korean_font()  # seaborn 이후에 폰트 설정

# 출력 디렉토리
output_dir = Path('eda_output')
output_dir.mkdir(exist_ok=True)

def main():
    print("=" * 100)
    print("EDA 리포트 생성 - Train Set Only (2013-2023)")
    print("=" * 100)

    # 데이터 로드
    df = pd.read_parquet('data/race_results_full.parquet')

    # Train/Test Split
    train = df[df['trd_dt'].dt.year <= 2023].copy()
    test = df[df['trd_dt'].dt.year >= 2024].copy()

    print(f"\n✓ Train set: {len(train):,}개 (2013-2023)")
    print(f"✗ Test set: {len(test):,}개 (2024-2025) - 절대 안 봄!")

    # 이제부터 Train만 사용
    df = train

    # Top 3 여부
    df['is_top3'] = (df['finish_pos'] <= 3).astype(int)

    # 배당 범위
    df['odds_range'] = pd.cut(
        df['odds_win'],
        bins=[0, 2, 3, 5, 10, 20, 50, 1000],
        labels=['1-2x', '2-3x', '3-5x', '5-10x', '10-20x', '20-50x', '50x+']
    )

    print("\n" + "=" * 100)
    print("시각화 생성 중...")
    print("=" * 100)

    # ==================================================
    # 1. 배당 범위별 Top 3 비율 vs 필요 정확도
    # ==================================================
    print("\n[1/7] 배당 범위별 수익성 분석...")

    odds_stats = df.groupby('odds_range', observed=True).agg({
        'is_top3': ['sum', 'count', 'mean'],
        'odds_place': 'mean'
    }).reset_index()
    odds_stats.columns = ['odds_range', 'top3_count', 'total', 'top3_rate', 'avg_place_odds']
    odds_stats['breakeven_precision'] = 1.0 / (odds_stats['avg_place_odds'] * 0.8)
    odds_stats['gap'] = odds_stats['breakeven_precision'] - odds_stats['top3_rate']

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(odds_stats))
    width = 0.35

    bars1 = ax.bar(x - width/2, odds_stats['top3_rate'] * 100, width,
                   label='실제 Top 3 비율', color='#2ecc71')
    bars2 = ax.bar(x + width/2, odds_stats['breakeven_precision'] * 100, width,
                   label='필요 정확도 (break-even)', color='#e74c3c')

    ax.set_xlabel('배당 범위', fontsize=12, fontweight='bold')
    ax.set_ylabel('Top 3 비율 (%)', fontsize=12, fontweight='bold')
    ax.set_title('배당 범위별 Top 3 비율 vs Break-even 필요 정확도\n(Train Set 2013-2023)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(odds_stats['odds_range'])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # 수익/손실 표시
    for i, (actual, needed) in enumerate(zip(odds_stats['top3_rate'], odds_stats['breakeven_precision'])):
        if actual >= needed:
            ax.text(i, max(actual, needed) * 100 + 2, '✓ 수익가능',
                   ha='center', fontsize=10, color='green', fontweight='bold')
        else:
            gap = (needed - actual) * 100
            ax.text(i, max(actual, needed) * 100 + 2, f'+{gap:.1f}%p',
                   ha='center', fontsize=9, color='red')

    plt.tight_layout()
    plt.savefig(output_dir / '1_odds_range_profitability.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ 저장: {output_dir / '1_odds_range_profitability.png'}")

    # ==================================================
    # 2. 50x+ 엘리트 기수 Top 10
    # ==================================================
    print("\n[2/7] 50x+ 고배당 엘리트 기수 분석...")

    df_50x = df[df['odds_win'] >= 50].copy()
    baseline_50x = df_50x['is_top3'].mean()

    jockey_stats = df_50x.groupby('jockey_name')['is_top3'].agg(['sum', 'count', 'mean']).reset_index()
    jockey_stats = jockey_stats[jockey_stats['count'] >= 10].sort_values('mean', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(jockey_stats)), jockey_stats['mean'] * 100, color='#3498db')
    ax.axvline(baseline_50x * 100, color='red', linestyle='--', linewidth=2, label=f'전체 평균 {baseline_50x:.1%}')

    ax.set_yticks(range(len(jockey_stats)))
    ax.set_yticklabels(jockey_stats['jockey_name'])
    ax.set_xlabel('Top 3 비율 (%)', fontsize=12, fontweight='bold')
    ax.set_title('50배+ 고배당 엘리트 기수 Top 10\n(10회 이상 출전, Train 2013-2023)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    # 수치 표시
    for i, (pct, cnt) in enumerate(zip(jockey_stats['mean'], jockey_stats['count'])):
        ax.text(pct * 100 + 0.5, i, f'{pct:.1%} ({cnt}회)',
               va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / '2_elite_jockeys_50x.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ 저장: {output_dir / '2_elite_jockeys_50x.png'}")

    # ==================================================
    # 3. 50x+ 엘리트 조련사 Top 10
    # ==================================================
    print("\n[3/7] 50x+ 고배당 엘리트 조련사 분석...")

    trainer_stats = df_50x.groupby('trainer_name')['is_top3'].agg(['sum', 'count', 'mean']).reset_index()
    trainer_stats = trainer_stats[trainer_stats['count'] >= 10].sort_values('mean', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(trainer_stats)), trainer_stats['mean'] * 100, color='#9b59b6')
    ax.axvline(baseline_50x * 100, color='red', linestyle='--', linewidth=2, label=f'전체 평균 {baseline_50x:.1%}')

    ax.set_yticks(range(len(trainer_stats)))
    ax.set_yticklabels(trainer_stats['trainer_name'])
    ax.set_xlabel('Top 3 비율 (%)', fontsize=12, fontweight='bold')
    ax.set_title('50배+ 고배당 엘리트 조련사 Top 10\n(10회 이상, Train 2013-2023)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    # 수치 표시
    for i, (pct, cnt) in enumerate(zip(trainer_stats['mean'], trainer_stats['count'])):
        ax.text(pct * 100 + 0.5, i, f'{pct:.1%} ({cnt}회)',
               va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / '3_elite_trainers_50x.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ 저장: {output_dir / '3_elite_trainers_50x.png'}")

    # ==================================================
    # 4. 50x+ 말 나이별 Top 3 비율
    # ==================================================
    print("\n[4/7] 50x+ 말 나이별 분석...")

    age_stats = df_50x.groupby('horse_age')['is_top3'].agg(['sum', 'count', 'mean']).reset_index()
    age_stats = age_stats[age_stats['count'] >= 100].sort_values('horse_age')

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(age_stats['horse_age'].astype(str), age_stats['mean'] * 100, color='#e67e22')
    ax.axhline(baseline_50x * 100, color='red', linestyle='--', linewidth=2, label=f'전체 평균 {baseline_50x:.1%}')

    ax.set_xlabel('말 나이 (세)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Top 3 비율 (%)', fontsize=12, fontweight='bold')
    ax.set_title('50배+ 고배당 말 나이별 Top 3 비율\n(100회 이상, Train 2013-2023)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # 수치 표시
    for i, (age, pct, cnt) in enumerate(zip(age_stats['horse_age'], age_stats['mean'], age_stats['count'])):
        ax.text(i, pct * 100 + 0.2, f'{pct:.1%}\n({cnt:,})',
               ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / '4_horse_age_50x.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ 저장: {output_dir / '4_horse_age_50x.png'}")

    # ==================================================
    # 5. 50x+ 부담중량별 Top 3 비율
    # ==================================================
    print("\n[5/7] 50x+ 부담중량별 분석...")

    df_50x['burden_bin'] = pd.cut(df_50x['burden_weight'], bins=[45, 52, 54, 56, 58, 61])
    burden_stats = df_50x.groupby('burden_bin', observed=True)['is_top3'].agg(['sum', 'count', 'mean']).reset_index()
    burden_stats = burden_stats[burden_stats['count'] >= 100]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(burden_stats)), burden_stats['mean'] * 100, color='#16a085')
    ax.axhline(baseline_50x * 100, color='red', linestyle='--', linewidth=2, label=f'전체 평균 {baseline_50x:.1%}')

    ax.set_xticks(range(len(burden_stats)))
    ax.set_xticklabels([str(b) for b in burden_stats['burden_bin']], rotation=0)
    ax.set_xlabel('부담중량 (kg)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Top 3 비율 (%)', fontsize=12, fontweight='bold')
    ax.set_title('50배+ 고배당 부담중량별 Top 3 비율\n(100회 이상, Train 2013-2023)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # 수치 표시
    for i, (pct, cnt) in enumerate(zip(burden_stats['mean'], burden_stats['count'])):
        ax.text(i, pct * 100 + 0.2, f'{pct:.1%}\n({cnt:,})',
               ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / '5_burden_weight_50x.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ 저장: {output_dir / '5_burden_weight_50x.png'}")

    # ==================================================
    # 6. 연도별 데이터 분포
    # ==================================================
    print("\n[6/7] 연도별 데이터 분포...")

    df['year'] = df['trd_dt'].dt.year
    yearly_stats = df.groupby('year').agg({
        'is_top3': ['sum', 'count', 'mean'],
        'odds_win': 'mean'
    }).reset_index()
    yearly_stats.columns = ['year', 'top3_count', 'total_races', 'top3_rate', 'avg_odds']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 상단: 레코드 수
    ax1.bar(yearly_stats['year'], yearly_stats['total_races'], color='#3498db', alpha=0.7)
    ax1.set_xlabel('연도', fontsize=12, fontweight='bold')
    ax1.set_ylabel('레코드 수', fontsize=12, fontweight='bold')
    ax1.set_title('연도별 데이터 분포 (Train 2013-2023)', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(axis='y', alpha=0.3)

    for year, count in zip(yearly_stats['year'], yearly_stats['total_races']):
        ax1.text(year, count + 100, f'{count:,}', ha='center', fontsize=9)

    # 하단: Top 3 비율 & 평균 배당
    ax2_twin = ax2.twinx()

    line1 = ax2.plot(yearly_stats['year'], yearly_stats['top3_rate'] * 100,
                     marker='o', linewidth=2, color='#2ecc71', label='Top 3 비율')
    line2 = ax2_twin.plot(yearly_stats['year'], yearly_stats['avg_odds'],
                          marker='s', linewidth=2, color='#e74c3c', label='평균 단승 배당')

    ax2.set_xlabel('연도', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Top 3 비율 (%)', fontsize=12, fontweight='bold', color='#2ecc71')
    ax2_twin.set_ylabel('평균 단승 배당 (배)', fontsize=12, fontweight='bold', color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#2ecc71')
    ax2_twin.tick_params(axis='y', labelcolor='#e74c3c')
    ax2.grid(alpha=0.3)

    # 범례 통합
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, fontsize=10, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_dir / '6_yearly_trends.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ 저장: {output_dir / '6_yearly_trends.png'}")

    # ==================================================
    # 7. 배당 vs Top 3 확률 (Scatter)
    # ==================================================
    print("\n[7/7] 배당 vs Top 3 확률 관계...")

    # 배당 구간별 집계
    df['odds_bin'] = pd.cut(df['odds_win'], bins=np.arange(0, 101, 5))
    scatter_data = df.groupby('odds_bin', observed=True)['is_top3'].agg(['mean', 'count']).reset_index()
    scatter_data = scatter_data[scatter_data['count'] >= 50]  # 샘플 50개 이상
    scatter_data['odds_mid'] = scatter_data['odds_bin'].apply(lambda x: x.mid)

    fig, ax = plt.subplots(figsize=(12, 7))
    scatter = ax.scatter(scatter_data['odds_mid'], scatter_data['mean'] * 100,
                        s=scatter_data['count']/10, alpha=0.6, color='#3498db')

    # 이론적 확률 (1/odds)
    x_theory = np.linspace(1, 100, 100)
    y_theory = (1 / x_theory) * 100
    ax.plot(x_theory, y_theory, 'r--', linewidth=2, label='이론적 확률 (1/배당)', alpha=0.7)

    ax.set_xlabel('단승 배당 (배)', fontsize=12, fontweight='bold')
    ax.set_ylabel('실제 Top 3 비율 (%)', fontsize=12, fontweight='bold')
    ax.set_title('단승 배당 vs 실제 Top 3 비율\n(Train 2013-2023, 점 크기 = 샘플 수)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / '7_odds_vs_top3_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ 저장: {output_dir / '7_odds_vs_top3_scatter.png'}")

    print("\n" + "=" * 100)
    print("시각화 완료! 총 7개 그래프 생성")
    print(f"저장 위치: {output_dir.absolute()}")
    print("=" * 100)

    # 기본 통계 출력 (MD 파일 작성용)
    print("\n" + "=" * 100)
    print("기본 통계 요약")
    print("=" * 100)

    print(f"\n데이터셋 크기:")
    print(f"  - 총 레코드: {len(df):,}개")
    print(f"  - 기간: {df['trd_dt'].min().date()} ~ {df['trd_dt'].max().date()}")
    print(f"  - 경주일: {df['trd_dt'].nunique()}일")
    print(f"  - 말: {df['horse_name'].nunique():,}마리")
    print(f"  - 기수: {df['jockey_name'].nunique()}명")
    print(f"  - 조련사: {df['trainer_name'].nunique()}명")

    print(f"\n컬럼별 결측치:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    for col in df.columns:
        if missing[col] > 0:
            print(f"  - {col}: {missing[col]:,}개 ({missing_pct[col]}%)")

    print(f"\n배당 통계:")
    print(f"  - 평균 단승 배당: {df['odds_win'].mean():.2f}배")
    print(f"  - 중앙값: {df['odds_win'].median():.2f}배")
    print(f"  - 최소: {df['odds_win'].min():.2f}배")
    print(f"  - 최대: {df['odds_win'].max():.2f}배")

    print(f"\nTop 3 통계:")
    print(f"  - Top 3 비율: {df['is_top3'].mean():.1%}")
    print(f"  - 1등 비율: {(df['finish_pos'] == 1).mean():.1%}")

    print("\n" + "=" * 100)


if __name__ == '__main__':
    main()
