# 모델링 전략

## 현재 상황 (Train Set 2013-2023 분석)

### 수익성 분석

| 배당 범위 | Top 3 비율 | 필요 정확도 | 갭 | 평균 복승배당 |
|---------|-----------|-----------|-----|------------|
| 1-2x | 53.6% | 44.7% | **✓ -8.9%p** | 2.80배 |
| 50x+ | 5.3% | 7.4% | **+2.1%p** | 16.86배 |
| 20-50x | 11.1% | 20.7% | +9.6%p | 6.03배 |
| 10-20x | 17.2% | 35.6% | +18.4%p | 3.51배 |

**결론:**
- 1-2x는 이미 수익 가능하지만 배당이 낮음 (2.80배)
- **50x+가 가장 유망**: 단 +2.1%p만 개선하면 수익, 평균 배당 16.86배

---

## 전략 1: 50x+ 고배당 집중 (Primary)

### 목표
- 50x+ 배당 말 중 Top 3에 들 확률 **7.4%+ 말 찾기**
- 현재 베이스라인: 5.3%
- 필요 개선: +2.1%p

### 예측 신호 (Train Set에서 확인된 패턴)

#### 1. 엘리트 기수 (10번 이상 출전)
| 기수 | Top 3 비율 | 향상 | 샘플 |
|-----|-----------|-----|------|
| 박재이 | 16.7% | +11.4%p | 12 |
| 윤태혁 | 15.4% | +10.1%p | 13 |
| 김덕현 | 13.2% | +7.9%p | 38 |
| 부민호 | 12.9% | +7.6%p | 70 |
| 마누엘 | 11.5% | +6.2%p | 52 |

**임계값**: Top 3 비율 10%+ (베이스라인 5.3%의 2배)

#### 2. 엘리트 조련사 (10번 이상)
| 조련사 | Top 3 비율 | 향상 | 샘플 |
|-------|-----------|-----|------|
| 이준철 | 12.5% | +7.2%p | 16 |
| 이희영 | 12.2% | +6.9%p | 131 |
| 송문길 | 10.8% | +5.5%p | 102 |
| 리카디 | 10.7% | +5.4%p | 56 |
| 김순근 | 10.7% | +5.4%p | 56 |

**임계값**: Top 3 비율 10%+

#### 3. 말 특성
- **부담중량**: 54-56kg (6.4%, +1.1%p)
- **나이**: 3-5세 (5.7%, +0.4%p)
- **게이트**: 1-3번 (6.2%, +0.9%p)

### Feature Engineering (Train Only)

각 기수/조련사의 과거 성적을 **train set에서만** 계산:

```python
# 기수 통계 (해당 경주 이전 데이터만 사용)
jockey_stats = {
    'total_races_50x': 출전 횟수 (50x+ 배당),
    'top3_rate_50x': Top 3 비율 (50x+ 배당),
    'total_races_all': 전체 출전 횟수,
    'top3_rate_all': 전체 Top 3 비율,
}

# 조련사 통계 (해당 경주 이전 데이터만 사용)
trainer_stats = {
    'total_races_50x': 출전 횟수 (50x+ 배당),
    'top3_rate_50x': Top 3 비율 (50x+ 배당),
    'total_races_all': 전체 출전 횟수,
    'top3_rate_all': 전체 Top 3 비율,
}

# 말 자체 통계
horse_stats = {
    'total_races': 출전 횟수,
    'top3_rate': Top 3 비율,
    'avg_finish_pos': 평균 등수,
}
```

**중요:** Look-ahead bias 방지 위해 Rolling window 방식 사용!

### 모델 접근법

#### Option A: Rule-based Model (Simple, Interpretable)
```python
def predict_50x_top3(horse_data, jockey_stats, trainer_stats):
    score = 0

    # 엘리트 기수 (+3점)
    if jockey_stats['top3_rate_50x'] >= 0.10 and jockey_stats['total_races_50x'] >= 10:
        score += 3

    # 엘리트 조련사 (+2점)
    if trainer_stats['top3_rate_50x'] >= 0.10 and trainer_stats['total_races_50x'] >= 10:
        score += 2

    # 부담중량 54-56kg (+1점)
    if 54 <= horse_data['burden_weight'] <= 56:
        score += 1

    # 나이 3-5세 (+1점)
    if horse_data['horse_age'] in [3, 4, 5]:
        score += 1

    # 게이트 1-3번 (+1점)
    if horse_data['gate_no'] <= 3:
        score += 1

    # 임계값: 점수 4점 이상이면 베팅
    return score >= 4

# 예상 성능: 점수 4점+ 말들의 Top 3 비율 > 7.4%
```

#### Option B: LightGBM (More flexible, but risk of overfitting)
```python
features = [
    # 기수 통계
    'jockey_top3_rate_50x',
    'jockey_total_races_50x',
    'jockey_top3_rate_all',

    # 조련사 통계
    'trainer_top3_rate_50x',
    'trainer_total_races_50x',
    'trainer_top3_rate_all',

    # 말 특성
    'horse_age',
    'burden_weight',
    'gate_no',
    'horse_top3_rate',

    # 경주 조건
    'distance',
    'track_cond_pct',
]

# LightGBM with conservative hyperparameters
lgbm_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'max_depth': 3,  # 얕게 (과적합 방지)
    'num_leaves': 7,
    'min_data_in_leaf': 100,  # 크게 (과적합 방지)
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
}
```

**권장**: Option A (Rule-based)부터 시작
- 이유: 샘플 크기가 작음 (50x+ = 7,507개)
- 해석 가능성 중요
- 과적합 위험 최소화

---

## 전략 2: 20-50x 중배당 (Secondary)

### 목표
- 20-50x 배당 말 중 Top 3 확률 **20.7%+ 찾기**
- 현재 베이스라인: 11.1%
- 필요 개선: +9.6%p (50x+보다 더 어려움)

### 예측 신호 (20x+ 기준)
- 엘리트 기수: 다실바 46.2%, 이효식 20.0%, 박재이 17.6%
- 엘리트 조련사: 이준철 18.3%, 리카디 15.6%, 김순근 15.3%
- 부담중량 56-58kg: 10.7% (+2.1%p)
- 나이 5세: 10.2% (+1.7%p)

**Note**: 개선 폭이 크고 샘플이 더 많아서 (16,911개) 모델 학습이 더 안정적일 수 있음

---

## 전략 3: 1-2x 저배당 최적화 (Bonus)

### 목표
- 이미 수익 가능 (53.6% > 44.7%)
- 추가 개선 시 더 안정적 수익 가능

### 문제점
- 평균 배당 2.80배로 낮음
- 수익률: (53.6% × 2.80 × 0.8) - 100% = **+20.1%** (괜찮음!)
- 샘플 크기 작음 (5,937개)

**결론**: 보조 전략으로만 활용

---

## 검증 전략

### 1. Train Set Cross-Validation (2013-2023)
```python
# Time-series walk-forward validation
years = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

for test_year in years[3:]:  # 2016년부터 테스트
    train_years = years[:years.index(test_year)]

    # Train: 2013 ~ test_year-1
    # Test: test_year

    # 평가지표:
    # - Top 3 Precision (score 4+ 말들의 Top 3 비율)
    # - Coverage (score 4+ 말이 전체의 몇 %?)
    # - ROI (실제 수익률)
```

### 2. Test Set Final Validation (2024-2025)
- **절대 규칙**: 모델 개발 완료 후 1회만 실행!
- Test set 10,451개 (2024-2025)
- 평가:
  - Top 3 Precision >= 7.4% (50x+)
  - ROI > 0%
  - 실제 베팅 시뮬레이션

---

## Success Criteria

### Minimum (최소 목표)
- 50x+ 배당에서 Top 3 Precision **7.4%+** (break-even)
- ROI >= 0%

### Target (목표)
- 50x+ 배당에서 Top 3 Precision **9.0%+** (20% profit margin)
- ROI >= 20%

### Stretch (도전 목표)
- 50x+ 배당에서 Top 3 Precision **10.0%+** (35% profit margin)
- ROI >= 35%

---

## 다음 단계

1. **Feature Engineering**: 기수/조련사 통계 계산 (Rolling window)
2. **Rule-based Model 구현**: 점수 기반 필터링
3. **Train Set CV**: 2016-2023 walk-forward 검증
4. **하이퍼파라미터 튜닝**: 점수 임계값 최적화
5. **Final Test**: 2024-2025 데이터로 최종 검증
6. **수익성 분석**: 실제 베팅 시뮬레이션

**예상 소요 시간**: 2-3일
**핵심 원칙**: Look-ahead bias 절대 금지!
