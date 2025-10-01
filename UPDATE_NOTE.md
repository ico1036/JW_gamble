# 업데이트 노트 (2025-10-01)

## 한글 폰트 수정 완료 ✅

### 변경사항
EDA 리포트의 모든 시각화가 한글 폰트 문제 해결 후 재생성되었습니다.

### 영향을 받는 파일
- `eda_output/1_odds_range_profitability.png` - 배당 범위별 수익성
- `eda_output/2_elite_jockeys_50x.png` - 50x+ 엘리트 기수
- `eda_output/3_elite_trainers_50x.png` - 50x+ 엘리트 조련사
- `eda_output/4_horse_age_50x.png` - 말 나이별 분석
- `eda_output/5_burden_weight_50x.png` - 부담중량별 분석
- `eda_output/6_yearly_trends.png` - 연도별 트렌드
- `eda_output/7_odds_vs_top3_scatter.png` - 배당 vs 확률

### Before → After
- **이전**: 한글 텍스트가 빈 네모(□□□)로 표시
- **현재**: 한글 텍스트 완벽 렌더링 (AppleSDGothicNeo 폰트)

### 리포트 확인
`EDA_REPORT_FINAL.md`의 모든 이미지 링크는 업데이트된 시각화를 가리킵니다.

### 기술적 세부사항
- 폰트: AppleSDGothicNeo (macOS 시스템 폰트)
- 경로: `/System/Library/Fonts/AppleSDGothicNeo.ttc`
- 해결 방법: FontEntry 직접 등록 + seaborn 이후 설정
