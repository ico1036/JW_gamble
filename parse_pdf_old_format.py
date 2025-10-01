#!/usr/bin/env python3
"""
경마 성적표 PDF 파싱 스크립트 - 2013-2020년 구 버전

구조: Row-based (각 row가 한 마리의 정보)
"""
import re
from pathlib import Path
from datetime import datetime
import pdfplumber
import pandas as pd


def extract_race_date(pdf_path: str) -> str:
    """파일명에서 경주일자 추출 (YYYY-MM-DD)"""
    filename = Path(pdf_path).stem
    return filename


def parse_weight_change(weight_text: str) -> tuple[int, int]:
    """
    마체중 증감 파싱
    예: '523( +5)' -> (523, 5)
    """
    if not weight_text:
        return None, None
    match = re.search(r'(\d+)\(([+-]?\d+)\)', weight_text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def parse_finish_time(time_text: str) -> float:
    """주파시간을 초 단위로 변환"""
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
    """착차 파싱"""
    if not margin_text or margin_text.strip() == '':
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


def parse_race_header(header_text: str) -> dict:
    """
    경주 헤더 정보 파싱
    예: '제 1경주 국6등급 1000M 별정 연령오픈 일반 흐림 다습[12%]'
    """
    info = {}

    # 경주번호
    race_no_match = re.search(r'제\s*(\d+)경주', header_text)
    info['race_no'] = int(race_no_match.group(1)) if race_no_match else None

    # 등급
    grade_match = re.search(r'(국\d등급|혼\d등급|\d등급|국OPEN)', header_text)
    info['grade'] = grade_match.group(1) if grade_match else None

    # 거리
    distance_match = re.search(r'(\d{4})M', header_text)
    info['distance'] = int(distance_match.group(1)) if distance_match else None

    # 경주구분
    info['race_type'] = '핸디캡' if '핸디캡' in header_text else '별정'

    # 날씨
    weather_match = re.search(r'(비|흐림|맑음)', header_text)
    info['weather'] = weather_match.group(1) if weather_match else None

    # 주로상태
    track_cond_match = re.search(r'([가-힣]+)\[(\d+)%\]', header_text)
    if track_cond_match:
        info['track_cond'] = track_cond_match.group(1)
        info['track_cond_pct'] = float(track_cond_match.group(2))
    else:
        info['track_cond'] = None
        info['track_cond_pct'] = None

    return info


def parse_table_old_format(table: list, trd_dt: str, track: str = '서울') -> list[dict]:
    """
    2013-2020년 구 버전 테이블 파싱 (Row-based)

    각 row가 한 마리의 정보:
    [0]=순위, [1]=게이트, [2]=말이름, [5]=산지, [6]=성별, [7]=나이,
    [8]=부담중량, [10]=기수, [12]=조련사, [16]=마체중, [17]=주파시간,
    [18]=착차, [20]=후3F, [21]=단승배당, [23]=복승배당
    """
    if not table or len(table) < 4:
        return []

    # 헤더 파싱 (row 1)
    header_text = table[1][0] if len(table) > 1 and table[1][0] else ''
    race_info = parse_race_header(header_text)

    # 데이터 행 찾기 (row 4부터)
    records = []

    for row_idx in range(4, len(table)):
        row = table[row_idx]

        # 빈 row이거나 데이터가 아니면 스킵
        if not row or not row[0] or not row[0].strip().isdigit():
            break

        try:
            # 순위가 숫자인지 확인
            rank = int(row[0].strip())

            # 마체중/증감
            weight, weight_change = parse_weight_change(row[16]) if len(row) > 16 else (None, None)

            record = {
                'trd_dt': datetime.strptime(trd_dt, '%Y-%m-%d'),
                'race_no': race_info['race_no'],
                'gate_no': int(row[1]) if len(row) > 1 and row[1] else None,
                'horse_name': row[2].strip() if len(row) > 2 and row[2] else None,
                'horse_origin': row[5].strip() if len(row) > 5 and row[5] else None,
                'horse_sex': row[6].strip() if len(row) > 6 and row[6] else None,
                'horse_age': int(row[7]) if len(row) > 7 and row[7] and row[7].strip().isdigit() else None,
                'jockey_name': row[10].strip() if len(row) > 10 and row[10] else None,
                'trainer_name': row[12].strip() if len(row) > 12 and row[12] else None,
                'weight': weight,
                'weight_change': weight_change,
                'track': track,
                'distance': race_info['distance'],
                'grade': race_info['grade'],
                'race_type': race_info['race_type'],
                'weather': race_info['weather'],
                'track_cond': race_info['track_cond'],
                'track_cond_pct': race_info['track_cond_pct'],
                'burden_weight': float(row[8]) if len(row) > 8 and row[8] else None,
                'rating': None,  # 구 버전에는 레이팅 없음
                'finish_pos': rank,
                'finish_time': parse_finish_time(row[17]) if len(row) > 17 else None,
                'margin': parse_margin(row[18]) if len(row) > 18 else 0.0,
                'odds_win': float(row[21]) if len(row) > 21 and row[21] else None,
                'odds_place': float(row[23]) if len(row) > 23 and row[23] else None,
            }

            records.append(record)

        except Exception as e:
            # 파싱 에러는 조용히 넘김 (매출액 등 다른 정보일 수 있음)
            continue

    return records


def parse_single_pdf_old(pdf_path: str) -> pd.DataFrame:
    """단일 PDF 파일 파싱 (구 버전)"""
    trd_dt = extract_race_date(pdf_path)
    all_records = []

    track = '서울'  # 기본값

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            try:
                tables = page.extract_tables()

                for table in tables:
                    records = parse_table_old_format(table, trd_dt, track)
                    all_records.extend(records)

            except Exception as e:
                # 페이지 에러는 조용히 넘김
                continue

    df = pd.DataFrame(all_records)

    # 데이터 타입 최적화
    if len(df) > 0:
        df['trd_dt'] = pd.to_datetime(df['trd_dt'])
        df['race_no'] = df['race_no'].astype('Int8')
        df['gate_no'] = df['gate_no'].astype('Int8')
        df['horse_age'] = df['horse_age'].astype('Int8')
        df['weight'] = df['weight'].astype('Int16')
        df['weight_change'] = df['weight_change'].astype('Int8')
        df['distance'] = df['distance'].astype('Int16')
        df['finish_pos'] = df['finish_pos'].astype('Int8')

        # Category 타입
        df['horse_name'] = df['horse_name'].astype('category')
        df['horse_origin'] = df['horse_origin'].astype('category')
        df['horse_sex'] = df['horse_sex'].astype('category')
        df['jockey_name'] = df['jockey_name'].astype('category')
        df['trainer_name'] = df['trainer_name'].astype('category')
        df['track'] = df['track'].astype('category')
        df['grade'] = df['grade'].astype('category')
        df['race_type'] = df['race_type'].astype('category')
        df['weather'] = df['weather'].astype('category')
        df['track_cond'] = df['track_cond'].astype('category')

    return df


def main():
    # 테스트
    pdf_path = '/Users/ryan/horse_park/2020-01-04.pdf'

    print(f"파싱 중: {pdf_path}\n")

    df = parse_single_pdf_old(pdf_path)

    print(f"추출된 레코드: {len(df)}개\n")
    print(df.head(20))
    print(f"\n컬럼: {list(df.columns)}")
    print(f"\n샘플 데이터:")
    print(df[['trd_dt', 'race_no', 'horse_name', 'jockey_name', 'finish_pos', 'odds_win']].head(10))


if __name__ == '__main__':
    main()
