# Step 2 데이터 신뢰성 검증

## 핵심 결과

- 필수 컬럼 누락: 없음
- event_time 파싱 실패: 0 rows
- 허용 범위 밖 event_type: 없음
- `price <= 0`: 68,673 rows (0.001618)
- 완전 중복 row: 30,220 rows (0.000712)
- 세션 내 시간 역전: 0 events, 0 sessions
- 극단 세션 기준: 100 events 이상
- 극단 세션 수: 2,487
- 최대 세션 길이: 1,159 events

## 모델링 전 결정 필요 사항

- `brand`, `category_code` 결측은 unknown category 처리 여부를 Step 5에서 확정한다.
- `user_session` 결측 row는 세션 기반 라벨링 대상에서 제외하는 정책을 우선 검토한다.
- 중복 row가 존재하면 라벨링 전 제거 여부와 제거 기준을 문서화한다.
- 시간 역전 세션은 prefix 생성 전에 `event_time` 기준 정렬을 강제한다.
- 극단적으로 긴 세션은 bot성 행동 또는 장기 세션 여부를 샘플링 검토한다.
