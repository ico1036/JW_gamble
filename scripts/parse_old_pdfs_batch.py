#!/usr/bin/env python3
"""
2013-2020년 구 버전 PDF 전체 파싱
"""
import os
from glob import glob
from pathlib import Path
import pandas as pd
from parse_pdf_old_format import parse_single_pdf_old


def main():
    pdf_dir = '/Users/ryan/horse_park'
    output_path = '/Users/ryan/horse_park/race_results_old.parquet'

    # 2013-2020년 PDF만 선택
    all_pdfs = sorted(glob(os.path.join(pdf_dir, '*.pdf')))
    old_pdfs = [p for p in all_pdfs if '2013' <= Path(p).stem[:4] <= '2020']

    print(f"=" * 100)
    print(f"2013-2020년 구 버전 PDF 파싱")
    print(f"=" * 100)
    print(f"\n총 PDF 파일: {len(old_pdfs)}개\n")

    all_dfs = []
    success_count = 0
    error_count = 0
    total_records = 0

    for i, pdf_path in enumerate(old_pdfs, 1):
        filename = Path(pdf_path).name

        try:
            df = parse_single_pdf_old(pdf_path)

            if len(df) > 0:
                all_dfs.append(df)
                success_count += 1
                total_records += len(df)
                print(f"[{i}/{len(old_pdfs)}] ✓ {filename:20s}: {len(df):4d}개 레코드")
            else:
                print(f"[{i}/{len(old_pdfs)}] ✗ {filename:20s}: 레코드 없음")
                error_count += 1

        except Exception as e:
            print(f"[{i}/{len(old_pdfs)}] ✗ {filename:20s}: ERROR - {str(e)[:50]}")
            error_count += 1

        # 진행상황 요약 (매 50파일마다)
        if i % 50 == 0:
            print(f"\n진행: {i}/{len(old_pdfs)} ({i/len(old_pdfs)*100:.1f}%)")
            print(f"성공: {success_count}, 실패: {error_count}, 누적 레코드: {total_records:,}\n")

    # 전체 DataFrame 합치기
    print(f"\n" + "=" * 100)
    print(f"파싱 완료!")
    print(f"성공: {success_count}, 실패: {error_count}\n")

    if all_dfs:
        print("DataFrame 병합 중...")
        final_df = pd.concat(all_dfs, ignore_index=True)

        print(f"\n최종 레코드 수: {len(final_df):,}개")
        print(f"기간: {final_df['trd_dt'].min()} ~ {final_df['trd_dt'].max()}")

        # Parquet 저장
        print(f"\nParquet 저장 중: {output_path}")
        final_df.to_parquet(output_path, index=False, compression='snappy')

        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"저장 완료! 파일 크기: {file_size:.2f}MB")

        # 통계
        print(f"\n기본 통계:")
        print(f"- 총 경주일: {final_df['trd_dt'].nunique()}일")
        print(f"- 총 출전: {len(final_df):,}회")
        print(f"- 말: {final_df['horse_name'].nunique()}마리")
        print(f"- 기수: {final_df['jockey_name'].nunique()}명")
        print(f"- 조교사: {final_df['trainer_name'].nunique()}명")

        # 연도별
        print(f"\n연도별 레코드 수:")
        print(final_df.groupby(final_df['trd_dt'].dt.year).size())

    else:
        print("ERROR: 파싱된 데이터가 없습니다!")

    print(f"\n" + "=" * 100)


if __name__ == '__main__':
    main()
