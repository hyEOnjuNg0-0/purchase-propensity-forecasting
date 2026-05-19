# Step 3 라벨링 설계 및 TDD

## 라벨 정의

- 예측 window: 기준 시점 이후 30분
- positive: 같은 `user_session`에서 `(cutoff_time, cutoff_time + window]` 구간에 purchase 발생
- sample 정책: 첫 purchase 이벤트와 그 이후 이벤트는 학습 sample에서 제외
- feature 범위: prefix feature는 기준 시점까지의 이벤트만 사용

## 라벨 분포

- 후보 이벤트 수: 42,448,764
- 라벨링 sample 수: 40,653,641
- positive: 2,753,084 (0.067720)
- negative: 37,900,557 (0.932280)
- 첫 purchase 이후 제외: 1,795,121
- 세션 수: 9,244,421
- purchase 포함 세션 수: 629,560

## 누수 방지 확인

- 라벨 계산은 기준 시점 이후 purchase 시각만 참조한다.
- prefix feature 생성은 기준 시점 이후 이벤트를 포함하지 않는다.
- raw `user_id`, `user_session`, 원문 `event_time`은 모델 입력 feature가 아니라 key와 검증 용도로만 사용한다.
