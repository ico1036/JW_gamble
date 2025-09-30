#!/usr/bin/env python3
"""
경마 성적표 PDF 파싱 스크립트
"""
import re
import os
from pathlib import Path
from datetime import datetime
import pdfplumber
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def extract_race_date(pdf_path: str) -> str:
    """파일명에서 경주일자 추출 (YYYY-MM-DD)"""
    filename = Path(pdf_path).stem  # 2025-09-28
    return filename


def parse_track_condition(cond_text: str) -> tuple[str, float]:
    """
    주로상태 파싱
    예: '불량[20%]' -> ('불량', 20.0)
    """
    match = re.search(r'([가-힣]+)\[(\d+)%\]', cond_text)
    if match:
        return match.group(1), float(match.group(2))
    return cond_text, None


def parse_horse_info(horse_text: str) -> tuple[str, str]:
    """
    말 정보 파싱
    예: '한 수' -> ('한', '수')
    """
    parts = horse_text.strip().split()
    if len(parts) >= 2:
        return parts[0], parts[1]  # 산지, 성별
    return None, None


def parse_weight_change(weight_text: str) -> int:
    """
    마체중 증감 파싱
    예: '( +5)' -> 5, '(-11)' -> -11
    """
    match = re.search(r'\(([+-]?\d+)\)', weight_text)
    if match:
        return int(match.group(1))
    return 0


def parse_finish_time(time_text: str) -> float:
    """
    주파시간을 초 단위로 변환
    예: '1:12.7' -> 72.7
    """
    if not time_text or ':' not in time_text:
        return None
    try:
        parts = time_text.split(':')
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    except:
        return None


def parse_margin(margin_text: str) -> float:
    """
    착차 파싱
    예: '½' -> 0.5, '2½' -> 2.5, '코' -> 0.1
    """
    if not margin_text or margin_text == '':
        return 0.0

    margin_text = margin_text.strip()

    # 특수 표현
    if margin_text == '코':
        return 0.1
    if margin_text == '목':
        return 0.25

    # 분수 처리
    margin_text = margin_text.replace('½', '.5')
    margin_text = margin_text.replace('¾', '.75')
    margin_text = margin_text.replace('¼', '.25')

    try:
        return float(margin_text)
    except:
        return 0.0


def parse_pdf_page(pdf_page, trd_dt: str) -> list[dict]:
    """
    PDF 페이지에서 경주 데이터 추출
    """
    text = pdf_page.extract_text()

    if not text:
        return []

    # 경주별로 섹션 분리 (제 N경주)
    race_sections = re.split(r'제\s*(\d+)경주', text)

    records = []

    # race_sections는 [전, 1, 후, 2, 후, ...] 형태
    for i in range(1, len(race_sections), 2):
        race_no = int(race_sections[i])
        race_text = race_sections[i + 1] if i + 1 < len(race_sections) else ''

        # 경주 기본 정보 추출
        # 등급, 거리, 별정/핸디캡, 날씨, 주로상태
        grade_match = re.search(r'(국\d등급|혼\d등급|\d등급|국OPEN)', race_text)
        grade = grade_match.group(1) if grade_match else None

        distance_match = re.search(r'(\d{4})M', race_text)
        distance = int(distance_match.group(1)) if distance_match else None

        race_type = '핸디캡' if '핸디캡' in race_text else '별정'

        weather_match = re.search(r'(비|흐림|맑음)', race_text)
        weather = weather_match.group(1) if weather_match else None

        track_cond_match = re.search(r'([가-힣]+)\[(\d+)%\]', race_text)
        if track_cond_match:
            track_cond = track_cond_match.group(1)
            track_cond_pct = float(track_cond_match.group(2))
        else:
            track_cond = None
            track_cond_pct = None

        # 출전마 정보는 테이블 형태로 파싱 (복잡함)
        # 일단 간단한 버전으로 구현

    return records


def parse_single_pdf(pdf_path: str) -> pd.DataFrame:
    """
    단일 PDF 파일 파싱
    """
    trd_dt = extract_race_date(pdf_path)

    all_records = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            records = parse_pdf_page(page, trd_dt)
            all_records.extend(records)

    # DataFrame 생성
    df = pd.DataFrame(all_records)

    return df


def main():
    # 테스트: 샘플 파일 파싱
    pdf_path = '/Users/ryan/horse_park/2025-09-28.pdf'

    print(f"파싱 중: {pdf_path}")

    df = parse_single_pdf(pdf_path)

    print(f"\n추출된 레코드: {len(df)}개")
    print(df.head())

    # Parquet 저장
    output_path = '/Users/ryan/horse_park/test_output.parquet'
    df.to_parquet(output_path, index=False)
    print(f"\n저장 완료: {output_path}")


if __name__ == '__main__':
    main()