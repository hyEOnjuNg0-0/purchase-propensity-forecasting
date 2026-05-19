# 데이터 검증 로직 테스트 후보

- 필수 컬럼이 누락되면 `schema_required_columns`가 실패해야 한다.
- 허용되지 않은 `event_type`이 있으면 unexpected event로 기록해야 한다.
- 파싱 불가능한 `event_time`은 실패 건수에 포함해야 한다.
- `price <= 0`은 이상 가격 건수에 포함해야 한다.
- 완전 동일 row가 반복되면 중복 row로 계산해야 한다.
- 같은 `user_session` 내 관측 순서상 시간이 감소하면 시간 역전으로 기록해야 한다.
- purchase 포함 세션과 비purchase 세션의 기본 통계는 분리 산출해야 한다.
- 극단 세션 기준 이상인 세션은 `data_quality_extreme_sessions.csv`에 포함해야 한다.
