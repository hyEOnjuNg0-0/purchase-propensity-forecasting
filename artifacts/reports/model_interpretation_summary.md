# Step 10 최종 모델 비교 및 해석

## 모델 비교 결론

- test PR-AUC 기준 최상위 모델은 LightGBM (none)이며 PR-AUC는 0.303770이다.
- 동일 sample 비교 점검: matched_by_split_counts.
- GRU는 최고 baseline보다 test PR-AUC가 0.130308 낮다. 현재 설정에서는 짧은 prefix 요약 feature와 class imbalance 대응만으로도 강한 baseline이 형성됐고, GRU는 더 긴 학습이나 sequence 표현 튜닝이 필요할 수 있다.

## LightGBM 주요 feature 해석

- 해석 대상 전략: `none`
- `last_price`: 최근 상품 가격은 구매 장벽과 상품군 차이를 함께 반영한다.
- `user_past_event_count`: 과거 이벤트 수는 사용자의 전체 활동성과 재방문 가능성을 나타낸다.
- `hour`: 시간대는 쇼핑 맥락과 구매 전환 패턴의 일중 변동을 반영한다.
- `unique_product_count`: 고유 상품 수는 비교 탐색 범위와 관심 강도를 나타낸다.
- `session_elapsed_minutes`: 세션 경과 시간은 사용자가 탐색에 투자한 누적 시간을 나타낸다.

## 해석 범위

- Step 10에서는 복잡한 SHAP, attention map, embedding clustering을 수행하지 않는다.
- 해석은 동일 split metric 비교와 LightGBM feature importance 기반의 짧은 행동 관점 요약으로 제한한다.
