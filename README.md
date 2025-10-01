# 한국 경마 예측 프로젝트

한국마사회(KRA) 경마 데이터 분석 및 예측 모델링 프로젝트

## 📁 프로젝트 구조

```
horse_park/
├── data/
│   ├── pdfs/                      # PDF 원본 데이터 (1,113개, 2013-2025)
│   └── race_results_full.parquet  # 파싱된 전체 데이터 (63,846 레코드)
│
├── scripts/                       # 데이터 수집 스크립트
│   ├── parse_pdf_complete.py      # 2021-2025 PDF 파서
│   ├── parse_pdf_old_format.py    # 2013-2020 PDF 파서
│   ├── parse_all_pdfs.py          # 일괄 파싱 (신규)
│   └── parse_old_pdfs_batch.py    # 일괄 파싱 (구버전)
│
├── eda_output/                    # EDA 시각화 결과
│   ├── 1_odds_range_profitability.png
│   ├── 2_elite_jockeys_50x.png
│   ├── 3_elite_trainers_50x.png
│   ├── 4_horse_age_50x.png
│   ├── 5_burden_weight_50x.png
│   ├── 6_yearly_trends.png
│   └── 7_odds_vs_top3_scatter.png
│
├── eda_train_only.py              # Train Set EDA 분석
├── generate_eda_report.py         # 시각화 생성 스크립트
│
├── EDA_REPORT_FINAL.md            # 📊 종합 EDA 리포트 (14,000+ 단어)
├── MODELING_STRATEGY.md           # 📋 모델링 전략 문서
│
├── pyproject.toml                 # Python 프로젝트 설정
├── uv.lock                        # 패키지 잠금 파일
└── README.md                      # 이 파일

```

## 🎯 프로젝트 개요

### 데이터셋
- **기간**: 2013-2025 (12년)
- **레코드**: 63,846개
- **Train Set**: 53,395개 (2013-2023)
- **Test Set**: 10,451개 (2024-2025)

### 핵심 발견
- 시장 효율성 매우 높음 (배당 ≈ 실제 확률)
- **50배+ 고배당**이 가장 유망한 타겟
  - 필요 개선: 단 +2.1%p (5.3% → 7.4%)
  - 평균 복승 배당: 16.86배
  - 예상 ROI: +20-40%

### 예측 신호
- **엘리트 기수**: 박재이 16.7% (+11.4%p), 윤태혁 15.4% (+10.1%p)
- **엘리트 조련사**: 이준철 12.5% (+7.2%p), 이희영 12.2% (+6.9%p)
- **부담중량**: 54-56kg (+1.1%p)
- **말 나이**: 3-5세 (+0.4%p)

## 📖 시작하기

### 1. 환경 설정

```bash
# UV 패키지 관리자 사용
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
uv sync
```

### 2. EDA 실행

```bash
# Train Set EDA (콘솔 출력)
uv run python eda_train_only.py

# 시각화 생성
uv run python generate_eda_report.py
```

### 3. 리포트 확인

- **종합 리포트**: [EDA_REPORT_FINAL.md](EDA_REPORT_FINAL.md)
- **모델링 전략**: [MODELING_STRATEGY.md](MODELING_STRATEGY.md)

## 🔬 방법론

### Look-ahead Bias 제거
- **절대 규칙**: Train Set (2013-2023)만 사용
- Test Set (2024-2025)는 최종 검증 시 1회만 사용
- Rolling window 방식으로 feature engineering

### 평가 전략
- Walk-forward Cross-Validation (2016-2023)
- 평가 지표: Top 3 Precision, Coverage, ROI

## 🎲 추천 전략

### Primary: 50배+ Rule-based Model

**점수 시스템**:
- 엘리트 기수 (Top 3 10%+, 10회+): +3점
- 엘리트 조련사 (Top 3 10%+, 10회+): +2점
- 부담중량 54-56kg: +1점
- 말 나이 3-5세: +1점
- 게이트 1-3번: +1점

**베팅 규칙**: 4점 이상

**예상 성능**:
- Precision: 10-15%
- ROI: +20-40%

## 📊 데이터 컬럼 설명

### 경주 정보
- `trd_dt`: 경주일자
- `race_no`: 경주 번호
- `track`: 경주장
- `distance`: 거리 (m)
- `grade`: 등급

### 말 정보
- `horse_name`: 말 이름
- `horse_age`: 나이
- `horse_sex`: 성별
- `weight`: 마체중

### 참가자
- `jockey_name`: 기수
- `trainer_name`: 조련사
- `burden_weight`: 부담중량

### 결과
- `finish_pos`: 순위
- `odds_win`: 단승 배당
- `odds_place`: 복승 배당

## 📈 다음 단계

1. ✅ 데이터 수집 (PDF 파싱)
2. ✅ EDA (Train Set Only)
3. ✅ 전략 수립
4. ⏳ Feature Engineering (Rolling Window)
5. ⏳ 모델 구현 (Rule-based)
6. ⏳ Walk-forward CV
7. ⏳ Test Set 최종 검증

## 📝 라이선스

This project is for educational and research purposes only.

## 👥 기여

Ryan (ico1036)

---

**Last Updated**: 2025-10-01
**Status**: EDA 완료, 모델링 준비 중
