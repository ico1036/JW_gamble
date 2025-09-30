#!/usr/bin/env python3
"""
경마 성적표 PDF 완전 파싱 스크립트
"""
import re
from pathlib import Path
from datetime import datetime
import pdfplumber
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def extract_race_date(pdf_path: str) -> str:
    """파일명에서 경주일자 추출 (YYYY-MM-DD)"""
    filename = Path(pdf_path).stem
    return filename


def parse_weight_change(weight_text: str) -> tuple[int, int]:
    """
    마체중 증감 파싱
    예: '523( +5)' -> (523, 5)
    """
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
    예: '제 1경주 국6등급 1200M 별정 2세 일반 비 불량[20%] 주로①'
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

    # 레이팅 범위
    rating_match = re.search(r'레이팅\s*(\d+)\s*~\s*(\d+)', header_text)
    if rating_match:
        info['has_rating'] = True
    else:
        info['has_rating'] = False

    return info


def parse_table(table: list, trd_dt: str, track: str = '서울') -> list[dict]:
    """
    테이블에서 레코드 추출
    """
    if not table or len(table) < 4:
        return []

    # 헤더 파싱
    header_text = table[0][0] if table[0][0] else ''
    race_info = parse_race_header(header_text)

    # 데이터 행 찾기 (보통 3번 또는 4번 인덱스)
    data_row_idx = None
    for idx, row in enumerate(table):
        # 순위 컬럼이 '1\n2\n3...' 형태인 행 찾기
        if row[0] and '1\n2' in row[0]:
            data_row_idx = idx
            break

    if data_row_idx is None:
        return []

    data_row = table[data_row_idx]

    # 각 컬럼을 newline으로 split
    ranks = data_row[0].strip().split('\n') if data_row[0] else []
    gate_nos = data_row[1].strip().split('\n') if data_row[1] else []
    horse_names = data_row[2].strip().split('\n') if data_row[2] else []
    origins = data_row[5].strip().split('\n') if data_row[5] else []
    sexes = data_row[6].strip().split('\n') if data_row[6] else []
    ages = data_row[7].strip().split('\n') if data_row[7] else []
    burden_weights = data_row[8].strip().split('\n') if data_row[8] else []
    ratings = data_row[9].strip().split('\n') if data_row[9] and race_info['has_rating'] else []
    jockeys = data_row[10].strip().split('\n') if data_row[10] else []
    trainers = data_row[12].strip().split('\n') if data_row[12] else []
    weights_raw = data_row[16].strip().split('\n') if data_row[16] else []
    finish_times_raw = data_row[17].strip().split('\n') if data_row[17] else []
    margins_raw = data_row[18].strip().split('\n') if data_row[18] else []
    last_3fs = data_row[20].strip().split('\n') if data_row[20] else []
    odds_wins = data_row[21].strip().split('\n') if data_row[21] else []
    odds_places = data_row[23].strip().split('\n') if data_row[23] else []

    # 레코드 생성
    records = []
    num_horses = len(ranks)

    for i in range(num_horses):
        try:
            # 마체중/증감
            weight, weight_change = parse_weight_change(weights_raw[i]) if i < len(weights_raw) else (None, None)

            # 레이팅
            rating = float(ratings[i]) if race_info['has_rating'] and i < len(ratings) and ratings[i] else None

            record = {
                'trd_dt': datetime.strptime(trd_dt, '%Y-%m-%d'),
                'race_no': race_info['race_no'],
                'gate_no': int(gate_nos[i]) if i < len(gate_nos) else None,
                'horse_name': horse_names[i] if i < len(horse_names) else None,
                'horse_origin': origins[i] if i < len(origins) else None,
                'horse_sex': sexes[i] if i < len(sexes) else None,
                'horse_age': int(ages[i]) if i < len(ages) and ages[i] else None,
                'jockey_name': jockeys[i] if i < len(jockeys) else None,
                'trainer_name': trainers[i] if i < len(trainers) else None,
                'weight': weight,
                'weight_change': weight_change,
                'track': track,
                'distance': race_info['distance'],
                'grade': race_info['grade'],
                'race_type': race_info['race_type'],
                'weather': race_info['weather'],
                'track_cond': race_info['track_cond'],
                'track_cond_pct': race_info['track_cond_pct'],
                'burden_weight': float(burden_weights[i]) if i < len(burden_weights) and burden_weights[i] else None,
                'rating': rating,
                'finish_pos': int(ranks[i]) if i < len(ranks) else None,
                'finish_time': parse_finish_time(finish_times_raw[i]) if i < len(finish_times_raw) else None,
                'margin': parse_margin(margins_raw[i]) if i < len(margins_raw) else 0.0,
                'odds_win': float(odds_wins[i]) if i < len(odds_wins) and odds_wins[i] else None,
                'odds_place': float(odds_places[i]) if i < len(odds_places) and odds_places[i] else None,
            }

            records.append(record)

        except Exception as e:
            print(f"  레코드 파싱 에러 (경주 {race_info['race_no']}, 마번 {i+1}): {e}")
            continue

    return records


def parse_single_pdf(pdf_path: str) -> pd.DataFrame:
    """단일 PDF 파일 파싱"""
    trd_dt = extract_race_date(pdf_path)
    all_records = []

    # 경주장 (서울/부산/제주) - 파일명이나 첫 페이지에서 추출 가능
    track = '서울'  # 기본값

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            try:
                tables = page.extract_tables()

                for table in tables:
                    records = parse_table(table, trd_dt, track)
                    all_records.extend(records)

            except Exception as e:
                print(f"  페이지 {page_num + 1} 파싱 에러: {e}")
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
    # 테스트: 샘플 파일 파싱
    pdf_path = '/Users/ryan/horse_park/2025-09-28.pdf'

    print(f"파싱 중: {pdf_path}\n")

    df = parse_single_pdf(pdf_path)

    print(f"추출된 레코드: {len(df)}개\n")
    print(df.head(20))
    print(f"\n컬럼: {list(df.columns)}")
    print(f"\n데이터 타입:\n{df.dtypes}")

    # Parquet 저장
    output_path = '/Users/ryan/horse_park/test_output.parquet'
    df.to_parquet(output_path, index=False)
    print(f"\n저장 완료: {output_path}")


if __name__ == '__main__':
    main()