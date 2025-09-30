#!/usr/bin/env python3
"""
EDA 실행 스크립트
"""
from eda import HorseRacingEDA

if __name__ == '__main__':
    # EDA 실행
    eda = HorseRacingEDA('/Users/ryan/horse_park/race_results.parquet')
    results = eda.run_full_analysis()

    print("\n✓ 모든 분석 완료")