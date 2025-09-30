#!/usr/bin/env python3
"""
전체 PDF 배치 파싱 스크립트 - 1,147개 파일
"""
import os
from pathlib import Path
from glob import glob
import pandas as pd
from parse_pdf_complete import parse_single_pdf
from dotenv import load_dotenv

load_dotenv()


def main():
    pdf_dir = '/Users/ryan/horse_park'
    output_path = '/Users/ryan/horse_park/race_results.parquet'

    # PDF 파일 목록
    pdf_files = sorted(glob(os.path.join(pdf_dir, '*.pdf')))

    print(f"총 PDF 파일: {len(pdf_files)}개\n")

    all_dfs = []
    success_count = 0
    error_count = 0

    for i, pdf_path in enumerate(pdf_files, 1):
        filename = Path(pdf_path).name

        try:
            df = parse_single_pdf(pdf_path)

            if len(df) > 0:
                all_dfs.append(df)
                success_count += 1
                print(f"[{i}/{len(pdf_files)}] ✓ {filename}: {len(df)}개 레코드")
            else:
                print(f"[{i}/{len(pdf_files)}] ✗ {filename}: 레코드 없음")
                error_count += 1

        except Exception as e:
            print(f"[{i}/{len(pdf_files)}] ✗ {filename}: 에러 - {e}")
            error_count += 1

        # 진행상황 요약 (매 100파일마다)
        if i % 100 == 0:
            print(f"\n진행: {i}/{len(pdf_files)} ({i/len(pdf_files)*100:.1f}%)")
            print(f"성공: {success_count}, 실패: {error_count}\n")

    # 전체 DataFrame 합치기
    print(f"\n모든 파일 처리 완료!")
    print(f"성공: {success_count}, 실패: {error_count}\n")

    if all_dfs:
        print("DataFrame 병합 중...")
        final_df = pd.concat(all_dfs, ignore_index=True)

        print(f"\n최종 레코드 수: {len(final_df):,}개")
        print(f"기간: {final_df['trd_dt'].min()} ~ {final_df['trd_dt'].max()}")
        print(f"\n컬럼: {list(final_df.columns)}")

        # Parquet 저장
        print(f"\nParquet 저장 중: {output_path}")
        final_df.to_parquet(output_path, index=False, compression='snappy')

        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"저장 완료! 파일 크기: {file_size:.2f}MB")

        # 샘플 데이터
        print(f"\n샘플 데이터:")
        print(final_df.head(10))

        # 통계
        print(f"\n기본 통계:")
        print(f"- 총 경주일: {final_df['trd_dt'].nunique()}일")
        print(f"- 총 경주: {final_df.groupby('trd_dt')['race_no'].max().sum()}개")
        print(f"- 총 출전: {len(final_df)}회")
        print(f"- 말: {final_df['horse_name'].nunique()}마리")
        print(f"- 기수: {final_df['jockey_name'].nunique()}명")
        print(f"- 조교사: {final_df['trainer_name'].nunique()}명")

    else:
        print("ERROR: 파싱된 데이터가 없습니다!")


if __name__ == '__main__':
    main()