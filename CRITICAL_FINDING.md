# 🚨 중대 발견: 데이터 분포 불일치

## 문제의 핵심

### Train Set (2013-2023)
- **odds_place 결측: 38.0%**
- 1-2배 구간: 22.1%만 odds_place 데이터 존재
- 해당 레이스들의 평균 odds_place: **2.89배**
- EDA에서 이를 기반으로 ROI +20.1% 계산 ❌

### Test Set (2024-2025)  
- **odds_place 결측: 0%** (완벽)
- 1-2배 구간: 100% odds_place 데이터 존재
- 실제 평균 odds_place: **1.12배**
- 실제 ROI: **-26.05%** ✓

## 왜 이런 일이?

**데이터 수집 정책 변화:**
- 2013-2023: 일부 레이스만 복승 발매 (고배당 레이스 위주로 선택적 발매?)
- 2024-2025: 모든 레이스 복승 발매

**결과:**
- Train/Test 데이터 분포가 근본적으로 다름
- Train 기반 EDA 인사이트가 Test에서 작동하지 않음

## 룰베이스 모델 실패 분석

### 전략별 결과 (Test Set)

| 전략 | Precision | 평균 배당 | ROI | 상태 |
|------|-----------|----------|-----|------|
| Baseline (All 50x+) | 4.9% | 14.38x | -53.49% | ❌ 실패 |
| 50x+ Elite (ANY) | 4.2% | 14.25x | -58.13% | ❌ 실패 |
| 50x+ Elite (BOTH) | 0.0% | 14.25x | -100.00% | ❌ 완전 실패 |
| Low Odds (1-2x) | 82.9% | 1.12x | -26.05% | ❌ 실패 |
| Hybrid | 61.2% | 4.73x | -34.88% | ❌ 실패 |

### 실패 원인

**1. Low Odds (1-2x) 전략:**
- EDA 예측: ROI +20.1% (평균 배당 2.89x 기반)
- 실제 결과: ROI -26.05% (평균 배당 1.12x)
- **원인: Train/Test 데이터 분포 차이**

**2. 50배+ 엘리트 전략:**
- EDA 예측: 엘리트 Top3 11.3%
- 실제 결과: 엘리트 Top3 4.2%
- **원인: Overfitting (Train 기반 엘리트 선정이 Test에서 무효)**

## 다음 단계

### 폐기할 전략:
- ❌ 1-2배 저배당 전략 (데이터 분포 불일치)
- ❌ 단순 엘리트 플래그 (Overfitting)

### 시도할 방향:
1. **머신러닝 모델** (Logistic Regression, XGBoost, LightGBM)
   - 복잡한 패턴 학습
   - Overfitting 방지 (regularization, cross-validation)

2. **피처 재설계**
   - 엘리트 플래그 → 연속형 스코어로 변경
   - Rolling window 통계 추가
   - 인터랙션 피처

3. **평가 기준 재정립**
   - ROI 최적화 (Precision만으로는 부족)
   - Expected Value (EV) 기반 베팅

## 교훈

**"EDA의 함정"**
- EDA는 Train set만 보고 수행
- Test set은 보지 않음 (look-ahead bias 방지)
- 하지만 Train/Test 분포가 다르면 EDA 인사이트가 무용지물
- **데이터 분포 안정성 검증 필수**
