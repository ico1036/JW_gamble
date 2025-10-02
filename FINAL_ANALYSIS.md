# 🏇 경마 예측 프로젝트 최종 분석

## 📊 프로젝트 개요

**목표**: 한국마사회(KRA) 경마 데이터를 활용한 수익 창출 모델 개발

**데이터**:
- 기간: 2013-02-22 ~ 2025-09-28 (약 12년)
- 레이스: 2,540개
- 레코드: 63,846개
- Train: 2013-2023 (53,395개)
- Test: 2024-2025 (10,451개)

**목표 변수**: Top 3 (복승) - finish_pos <= 3

---

## 🔬 실험한 모델

### 1. 룰베이스 모델

| 전략 | 설명 | Test Precision | Test ROI | 상태 |
|------|------|----------------|----------|------|
| Baseline (All 50x+) | 50배+ 모두 베팅 | 4.9% | -53.49% | ❌ |
| 50x+ Elite (ANY) | 50배+ 엘리트 기수 OR 조련사 | 4.2% | -58.13% | ❌ |
| 50x+ Elite (BOTH) | 50배+ 엘리트 기수 AND 조련사 | 0.0% | -100.00% | ❌ |
| Low Odds (1-2x) | 1-2배 저배당 | 82.9% | -26.05% | ❌ |
| Hybrid | 50x Elite + 1-2x 조합 | 61.2% | -34.88% | ❌ |

**결론**: 모든 룰베이스 전략 실패

### 2. 머신러닝 모델

#### Logistic Regression
- Train ROC-AUC: 0.8423
- Val ROC-AUC: 0.6657
- Test ROC-AUC: 0.6714
- **Test ROI: -35.62%** ❌
- Threshold: 0.89 (매우 보수적)
- 베팅 수: 554개

#### LightGBM
- Train ROC-AUC: 0.8507
- Val ROC-AUC: 0.7369
- Test ROC-AUC: 0.7304
- **Test ROI: -33.25%** ❌
- Threshold: 0.229
- 베팅 수: 1,113개

**결론**: 모든 머신러닝 모델 실패

---

## 💡 실패 원인 분석

### 1. 데이터 분포 불일치 (가장 치명적)

#### Train Set (2013-2023)
- odds_place 결측: **38.0%**
- 1-2배 구간: 22.1%만 odds_place 데이터 존재
- 해당 레이스들의 평균 odds_place: **2.89배**

#### Test Set (2024-2025)
- odds_place 결측: **0%** (완벽)
- 1-2배 구간: 100% odds_place 데이터 존재
- 실제 평균 odds_place: **1.12배**

**결과**:
- EDA 기반 ROI 예측: **+20.1%** (잘못된 계산!)
- 실제 Test ROI: **-26.05%**
- **차이: 46.1%p** (치명적 오류)

**원인**:
- 2013-2023년: 일부 레이스만 복승 발매 (고배당 레이스 위주?)
- 2024-2025년: 모든 레이스 복승 발매 (정책 변경)
- Train/Test 데이터 생성 과정이 근본적으로 다름

### 2. 시장 효율성 (Efficient Market Hypothesis)

**배당(odds)이 가장 강력한 예측 변수**:
- ROC-AUC 0.7~0.85 달성
- 하지만 ROI는 여전히 마이너스

**의미**:
- 시장이 이미 모든 정보를 반영
- 기수, 조련사, 말의 과거 성적 → 이미 배당에 반영됨
- 추가 피처로는 배당을 뛰어넘는 예측 불가능

### 3. 높은 수수료 (20%)

**수익 공식**:
```
ROI = (승리확률 × 배당 × 환급률) - 1
환급률 = 80% (마사회 수수료 20%)
```

**예시: 1-2배 저배당**
- Precision: 82.9%
- 평균 배당: 1.12배
- ROI = (82.9% × 1.12 × 0.8) - 1 = **-26.05%**

**Break-even 조건**:
- 필요 정확도 = 1 / (배당 × 0.8)
- 1.12배 배당 → 필요 정확도: **111.6%** (불가능!)

### 4. Overfitting

**엘리트 기수/조련사 전략**:
- EDA (Train): 엘리트 기수 Top3 11.3%
- Test 결과: 엘리트 기수 Top3 4.2%
- **차이: 7.1%p 하락**

**원인**:
- Train set 기반으로 "엘리트" 선정
- Test set에서는 해당 기수들의 성적 하락
- Train 특정 패턴에 Overfitting

### 5. LightGBM Early Stopping 문제

- Iteration 2에서 early stopping 발생
- Val AUC가 너무 빨리 피크 후 하락
- 모델이 제대로 학습하지 못함

**원인**:
- Train/Val 분포 차이로 인한 불안정한 학습
- 데이터가 너무 noisy
- 패턴이 매우 약함 (배당 외에 유의미한 신호 부족)

---

## 📈 모델별 성능 비교

| 모델 | Val ROI | Test ROI | Test Precision | Test AUC | 베팅 수 | 평균 배당 |
|------|---------|----------|----------------|----------|---------|----------|
| **베이스라인** |||||||
| All 50x+ | -61.60% | -53.49% | 4.9% | - | 1,566 | 14.38x |
| All 1-2x | - | -26.05% | 82.9% | - | 316 | 1.12x |
| **룰베이스** |||||||
| 50x+ Elite (ANY) | -72.11% | -58.13% | 4.2% | - | 120 | 14.25x |
| Hybrid | -43.67% | -34.88% | 61.2% | - | 436 | 4.73x |
| **머신러닝** |||||||
| Logistic Regression | -24.40% | -35.62% | 30.9% | 0.6714 | 554 | 3.68x |
| LightGBM | -26.79% | -33.25% | 58.8% | 0.7304 | 1,113 | 1.93x |

**최선의 결과**:
- Logistic Regression (threshold 0.89)
- Val ROI: -24.40%
- Test ROI: -35.62%
- **여전히 큰 손실**

---

## 🚨 핵심 교훈

### 1. "The House Always Wins"

**수학적 증명**:
- 환급률 80% (수수료 20%)
- 장기적으로 베팅 금액의 20%는 무조건 손실
- 아무리 정확하게 예측해도 수수료를 극복 불가능

**실제 결과**:
- 1-2배 저배당: 82.9% 정확도에도 -26.05% ROI
- Break-even 필요 정확도: 111.6% (불가능)

### 2. 시장은 생각보다 효율적

- 배당 자체가 이미 매우 정확한 확률 반영
- 추가 정보(기수, 조련사, 과거 성적)를 넣어도 배당을 뛰어넘지 못함
- ROC-AUC 0.7~0.85 달성했지만 ROI는 여전히 마이너스

### 3. EDA의 함정

**문제**:
- EDA는 Train set만 보고 수행 (look-ahead bias 방지)
- Train/Test 데이터 분포가 다르면 EDA 인사이트 무용지물
- **데이터 생성 과정의 안정성 검증 필수**

**교훈**:
- Train set 일부 데이터(22.1%)로 계산한 ROI +20.1%
- 실제 Test set 전체 데이터로는 ROI -26.05%
- **46.1%p 차이!**

### 4. Overfitting은 어디에나 있다

- 룰베이스 모델도 Overfitting 가능
- "엘리트 기수" 선정이 Train 기반 → Test에서 실패
- 단순한 규칙도 데이터에 과적합될 수 있음

---

## 📊 Feature Importance (LightGBM)

| Rank | Feature | Importance | 설명 |
|------|---------|------------|------|
| 1 | horse_top3_rate | 8,271 | 말의 과거 Top3 비율 |
| 2 | odds_win | 5,750 | 단승 배당 (가장 강력) |
| 3 | odds_place | 2,538 | 복승 배당 |
| 4 | trainer_50x_races | 1,254 | 조련사 50x+ 출전 횟수 |
| 5 | jockey_50x_races | 925 | 기수 50x+ 출전 횟수 |
| 6 | trainer_top3_rate | 417 | 조련사 Top3 비율 |
| 7 | jockey_win_rate | 128 | 기수 승률 |
| 8 | jockey_top3_rate | 84 | 기수 Top3 비율 |
| 9 | trainer_win_rate | 72 | 조련사 승률 |
| 10 | trainer_50x_top3_rate | 69 | 조련사 50x+ Top3 비율 |

**분석**:
- 배당(odds)이 가장 중요 (rank 2, 3)
- 과거 성적도 중요하지만 배당을 뛰어넘지 못함
- 엘리트 관련 피처는 상대적으로 덜 중요

---

## 💰 현실적 조언

### ❌ 하지 말아야 할 것

1. **경마로 돈 벌기를 기대하지 마세요**
   - 수학적으로 장기 수익 불가능
   - 수수료 20%를 극복할 수 없음

2. **"이번엔 다를 것"이라고 생각하지 마세요**
   - EDA에서 수익 가능해 보여도 실제로는 손실
   - Train/Test 분포 차이로 인한 착시

3. **더 복잡한 모델을 시도하지 마세요**
   - LightGBM도 실패
   - 딥러닝도 마찬가지일 것
   - 근본적인 문제는 시장 효율성과 수수료

### ✅ 할 수 있는 것

1. **오락으로 즐기기**
   - 소액 베팅으로 재미만 추구
   - 수익 기대는 금물

2. **데이터 분석 학습**
   - 경마 데이터로 ML 공부
   - Feature engineering, 모델링 연습
   - 하지만 실제 수익은 기대하지 말 것

3. **시장 효율성 연구**
   - 배당이 얼마나 정확한지 분석
   - Favorite-longshot bias 연구
   - 학술적 목적으로만

---

## 📁 프로젝트 구조

```
horse_park/
├── data/
│   ├── race_results_full.parquet    # 원본 데이터
│   ├── train.parquet                # Train set (피처 포함)
│   ├── val.parquet                  # Validation set
│   ├── test.parquet                 # Test set
│   └── feature_columns.json         # 피처 리스트
├── models/
│   ├── logistic_regression.pkl      # Logistic 모델
│   ├── lightgbm.txt                 # LightGBM 모델
│   ├── feature_importance.csv       # Feature importance
│   └── test_predictions.csv         # Test 예측 결과
├── results/
│   ├── rule_based_val_results.csv   # 룰베이스 Val 결과
│   ├── rule_based_test_results.csv  # 룰베이스 Test 결과
│   └── rule_based_predictions.csv   # 룰베이스 예측
├── eda_output/                      # EDA 시각화 (7개)
├── feature_engineering.py           # 피처 생성
├── models_rule_based.py             # 룰베이스 모델
├── train.py                         # ML 모델 훈련
├── evaluation.py                    # 평가 모듈
├── EDA_REPORT_FINAL.md             # EDA 리포트
├── CRITICAL_FINDING.md             # 중대 발견 사항
└── FINAL_ANALYSIS.md               # 최종 분석 (본 문서)
```

---

## 🎯 결론

**경마로 돈을 버는 것은 거의 불가능합니다.**

**이유**:
1. ⚠️ **높은 수수료 (20%)**: 장기적으로 극복 불가능
2. 📊 **시장 효율성**: 배당이 이미 모든 정보 반영
3. 📉 **데이터 분포 불일치**: Train/Test 데이터 생성 과정 차이
4. 🔄 **Overfitting**: 모든 모델이 Train 데이터에 과적합
5. 💸 **수학적 한계**: Break-even 조건이 불가능한 수준

**최종 점수**:
- 최선의 모델 (Logistic Regression): **-24.40% ROI (Val), -35.62% ROI (Test)**
- 모든 모델: **마이너스 ROI**

**"The house always wins." - 카지노 격언**

---

## 🔬 추가 연구 방향 (학술적 목적)

1. **시장 효율성 분석**
   - Favorite-longshot bias 정량화
   - 배당의 예측력 분석
   - 시간대별 시장 효율성 변화

2. **Anomaly Detection**
   - 배당과 실제 확률의 큰 차이 탐지
   - 하지만 수익은 여전히 어려울 것

3. **Multi-class Classification**
   - Top 3 대신 정확한 순위 예측
   - Ordinal regression 적용

4. **Survival Analysis**
   - 말의 전성기 예측
   - 은퇴 시점 예측

**하지만 어느 것도 장기적 수익을 보장하지 않습니다.**

---

**⚠️ 최종 경고**: 경마는 오락일 뿐입니다. 투자 수단으로 생각하지 마세요. 수학적으로 손실이 거의 확정되어 있습니다.
