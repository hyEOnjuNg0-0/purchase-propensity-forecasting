# Step 4 EDA 및 문제 타당성 검증

## 핵심 요약

- 대상 파일: `data\2019-Oct.csv`
- 예측 window: 30분
- 분석 row 수: 2000000
- 세션 수: 446930
- 라벨링 sample 수: 1919411
- positive sample 비율: 0.060393
- 최상위 sequence pattern: view

## 리포트용 chart 후보

- `eda_session_length_purchase_rate.csv`: 세션 길이별 구매율
- `eda_sequence_pattern_purchase_rate.csv`: 초기 event sequence pattern별 구매율
- `eda_price_band_purchase_rate.csv`: 가격대별 30분 내 구매율
- `eda_category_conversion.csv`: category별 purchase/view 전환율
- `eda_hourly_purchase_rate.csv`: 시간대별 30분 내 구매율
- `eda_positive_negative_sample_comparison.csv`: positive/negative sample 차이

## 문제 타당성 판단 포인트

- positive 비율과 가격대/시간대/행동 pattern별 차이가 baseline 모델의 학습 신호 후보이다.
- category 전환율은 descriptive EDA이며, 모델 feature에는 기준 시점 이후 정보가 들어가지 않도록 Step 5에서 별도 검증한다.
- sequence pattern 집계는 첫 purchase 전 이벤트만 사용해 사후 구매 이벤트가 pattern에 섞이지 않도록 제한했다.
