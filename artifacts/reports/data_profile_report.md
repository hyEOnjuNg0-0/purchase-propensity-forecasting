# Step 1 데이터 프로파일링

## 요약

- 대상 파일: `data\2019-Oct.csv`
- 전체 row count: 42,448,764
- CSV 파일 크기: 5.28 GB
- 추정 pandas memory footprint: 15.37 GB
- event_time 범위: 2019-10-01T00:00:00+00:00 ~ 2019-10-31T23:59:59+00:00
- purchase event 비율: 0.017500
- purchase 포함 세션 비율: 0.068102

## 산출물

- `data_profile_summary.csv`: Step 1 핵심 요약표
- `data_profile_event_type_distribution.csv`: event_type 분포
- `data_profile_missing_values.csv`: 컬럼별 결측 현황
- `data_quality_issues_draft.csv`: 데이터 품질 이슈 초안

## 다음 검증 후보

- 필수 컬럼 존재 여부를 자동 검증한다.
- `price <= 0` 이상치를 확인한다.
- 완전 중복 row 비율과 세션 내 시간 역전 여부를 확인한다.
- 극단적으로 긴 세션과 purchase/비purchase 세션의 차이를 비교한다.
