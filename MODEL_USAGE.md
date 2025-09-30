# Horse Racing Prediction Model - Usage Guide

## 개요

EDA 분석 결과를 바탕으로 구현된 한국 경마 예측 모델입니다. 배당률(odds)을 주요 특징으로 사용하며, 기수, 조련사, 말의 과거 성적을 aggregation 통계로 활용합니다.

## 구현된 모델

### 1. Odds Baseline Model
- **설명**: 배당률의 implied probability를 그대로 사용
- **학습**: 불필요 (시장 효율성 기반)
- **성능**: ROC-AUC 0.815
- **용도**: 시장 효율성 확인 및 베이스라인

### 2. Logistic Baseline Model
- **설명**: odds_win만 사용한 로지스틱 회귀
- **학습**: 필요 (단순)
- **성능**: ROC-AUC 0.815
- **용도**: 단순 선형 모델 베이스라인

### 3. LightGBM Model
- **설명**: 24개 특징을 사용한 gradient boosting
- **학습**: 필요 (복잡)
- **성능**: ROC-AUC 0.749
- **용도**: 비선형 관계 및 특징 상호작용 학습

## 사용된 특징 (24개)

### Tier 1: 배당률 (가장 강력한 예측변수)
- `odds_win`, `odds_place`
- `log_odds_win`, `log_odds_place` (로그 변환)
- `implied_prob_win`, `implied_prob_place` (implied probability)
- `burden_weight` (부담중량, 과거 성적의 프록시)

### Tier 2: 기수/조련사/말 통계
- `jockey_win_rate`, `jockey_avg_pos`, `jockey_races`
- `trainer_win_rate`, `trainer_avg_pos`, `trainer_races`
- `horse_win_rate`, `horse_avg_pos`, `horse_races`
- `horse_age`
- `is_elite_jockey`, `is_elite_trainer`, `is_young_horse`

### Tier 3: 경주 조건
- `gate_no`, `is_inner_gate`
- `track_cond_pct`

### 상호작용
- `elite_combo` (elite_jockey × elite_trainer)

## 설치

```bash
# 의존성 설치
uv sync

# 또는 개별 패키지 설치
uv add lightgbm scikit-learn pandas numpy matplotlib seaborn
```

## 데이터 준비

필수 컬럼:
- `trd_dt`: 경주 날짜
- `finish_pos`: 최종 순위 (1위, 2위, ...)
- `odds_win`, `odds_place`: 배당률
- `burden_weight`: 부담중량
- `horse_age`: 말 나이
- `gate_no`: 게이트 번호
- `track_cond_pct`: 트랙 수분 함량
- `jockey_name`, `trainer_name`, `horse_name`: 이름

## 모델 학습

### 기본 사용법

```bash
uv run python train.py --data race_results.parquet --output-dir models
```

### 옵션

- `--data`: 입력 데이터 경로 (기본: `race_results.parquet`)
- `--output-dir`: 모델 저장 디렉토리 (기본: `models`)
- `--no-plots`: 시각화 비활성화 (비대화형 환경용)

### 출력 파일

학습 후 `models/` 디렉토리에 다음 파일이 생성됩니다:

- `feature_engineer.pkl`: Feature engineering 객체 (통계량 포함)
- `odds_baseline.pkl`: Odds Baseline 모델
- `logistic_baseline.pkl`: Logistic Baseline 모델
- `lgbm_model.pkl`: LightGBM 모델
- `feature_columns.json`: 사용된 특징 목록
- `metrics.json`: 평가 지표 (ROC-AUC, log-loss 등)
- `roc_curves.png`: ROC 곡선 비교 (--no-plots 미사용시)
- `feature_importance.png`: 특징 중요도 (--no-plots 미사용시)
- `feature_importance.csv`: 특징 중요도 CSV (--no-plots 미사용시)

## 예측 (Python 코드)

```python
import pickle
import pandas as pd

# 모델 및 feature engineer 로드
with open('models/feature_engineer.pkl', 'rb') as f:
    feature_engineer = pickle.load(f)

with open('models/lgbm_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 새로운 데이터 로드
new_data = pd.read_parquet('new_races.parquet')

# Feature engineering
new_data = feature_engineer.transform(new_data)

# 예측
probabilities = model.predict_proba(new_data)
win_probs = probabilities[:, 1]  # 승리 확률

# 결과 출력
new_data['win_probability'] = win_probs
print(new_data[['horse_name', 'odds_win', 'win_probability']].head())
```

## 성능 결과

### Test Set (2024-12-15 ~ 2025-09-28, 4,767 경주)

| 모델 | ROC-AUC | Log Loss | Precision | Recall | F1-Score |
|------|---------|----------|-----------|--------|----------|
| **Odds Baseline** | **0.815** | **0.262** | 0.511 | 0.155 | 0.238 |
| Logistic Baseline | 0.815 | 0.269 | 0.000 | 0.000 | 0.000 |
| LightGBM | 0.749 | 0.305 | 0.000 | 0.000 | 0.000 |

### 주요 발견

1. **Odds Baseline이 가장 우수**: ROC-AUC 0.815
   - 시장 효율성이 매우 높음 (EDA 결과와 일치)
   - 추가 특징이 오히려 성능을 떨어뜨림

2. **클래스 불균형 문제**: 승자 비율 9.75%
   - Precision은 낮지만 ROC-AUC는 높음
   - 0.5 threshold에서는 대부분 negative 예측

3. **LightGBM 성능 저하**:
   - 더 많은 특징이 오히려 noise로 작용
   - Overfitting 가능성
   - Hyperparameter 튜닝 필요

## 개선 방향

### 1. 모델 개선
- **Threshold 조정**: 클래스 불균형에 맞춘 threshold (0.1-0.2)
- **Hyperparameter 튜닝**: LightGBM params 최적화
- **Feature Selection**: 중요도 낮은 특징 제거
- **Calibration**: Platt scaling, isotonic regression

### 2. Feature Engineering 개선
- **Time-series features**: 최근 N경기 성적 (rolling window)
- **Race-specific features**: 같은 경주 내 상대적 순위
- **Interaction features**: 더 복잡한 상호작용
- **Past performance**: 말의 과거 성적 직접 사용 (leakage 방지)

### 3. 데이터 보강
- Rating 데이터 추가 (현재 100% missing)
- Pedigree 정보
- Workout times
- 다른 트랙 데이터 (부산, 제주)

### 4. 평가 지표 확장
- **Expected Value**: 베팅 수익성
- **Calibration plot**: 확률 calibration 시각화
- **Top-K accuracy**: Exacta/Trifecta 예측

## 베팅 전략 (EDA 기반)

EDA 분석에서 발견된 시장 비효율성:

1. **단승 1-3배당**: 2% 저평가 (수익 기회)
2. **5-20배당 범위**: 1.3% edge
3. **50배당 이상**: 0.9% 과대평가 (회피)

### Kelly Criterion 적용

```python
# 예시: 1.5배당, 52.5% 승률
odds = 1.5
true_prob = 0.525
kelly_fraction = (odds * true_prob - 1) / (odds - 1)
# = 0.075 (bankroll의 7.5% 베팅)
```

## 프로젝트 구조

```
horse_park/
├── race_results.parquet          # 원본 데이터
├── eda_output/                   # EDA 결과
│   ├── EDA_REPORT.md
│   └── figures/
├── feature_engineering.py        # Feature engineering 모듈
├── models.py                     # 모델 구현
├── evaluation.py                 # 평가 모듈
├── train.py                      # 학습 스크립트
├── models/                       # 학습된 모델 저장
│   ├── feature_engineer.pkl
│   ├── *_model.pkl
│   └── metrics.json
└── MODEL_USAGE.md               # 이 문서
```

## 참고 문헌

- EDA_REPORT.md: 탐색적 데이터 분석 결과
- pyproject.toml: 의존성 목록

## 라이센스

Internal use only

---

**생성일**: 2025-09-30
**데이터셋**: KRA 경마 결과 (2021-2025, 23,833건)
**최고 성능**: Odds Baseline (ROC-AUC 0.815)