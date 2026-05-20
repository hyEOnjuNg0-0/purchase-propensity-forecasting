# Feature Artifact Reference

이 문서는 Step 6 검증 대상인 `feature_split_summary.csv`와 `sequence_feature_dataset.parquet`의 컬럼 이름, 의미, 활용 목적을 정리한다. 기준 artifact는 `2019-10-10`까지의 이벤트로 생성한 Step 5 결과다.

## 1. `artifacts/reports/feature_split_summary.csv`

이 파일은 시간 기준 train/validation/test split의 sample 수와 label 분포를 요약한다. Step 7 이후 모델 학습과 평가에서 split별 class imbalance를 확인하는 기준 report다.

### 1.1 컬럼 정의

| 컬럼명 | 타입 | 설명 |
| --- | --- | --- |
| `split` | string | 시간 기준 데이터 구간이다. 값은 `train`, `validation`, `test` 중 하나다. |
| `sample_count` | integer | 해당 split에 포함된 prefix sample 수다. |
| `positive_count` | integer | 해당 split에서 30분 이내 purchase label이 1인 sample 수다. |
| `negative_count` | integer | 해당 split에서 30분 이내 purchase label이 0인 sample 수다. |
| `positive_ratio` | float | `positive_count / sample_count`로 계산한 positive label 비율이다. |
| `cutoff_time_min` | datetime string | 해당 split에 포함된 가장 이른 예측 기준 시점이다. |
| `cutoff_time_max` | datetime string | 해당 split에 포함된 가장 늦은 예측 기준 시점이다. |

### 1.2 현재 생성 결과

| split | sample_count | positive_count | negative_count | positive_ratio | cutoff_time_min | cutoff_time_max |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| train | 8,605,918 | 572,332 | 8,033,586 | 0.066504 | 2019-10-01T00:00:00+00:00 | 2019-10-08T04:53:25+00:00 |
| validation | 1,844,114 | 120,396 | 1,723,718 | 0.065287 | 2019-10-08T04:53:26+00:00 | 2019-10-09T12:15:02+00:00 |
| test | 1,844,115 | 107,733 | 1,736,382 | 0.058420 | 2019-10-09T12:15:03+00:00 | 2019-10-10T23:59:59+00:00 |

## 2. `artifacts/features/sequence_feature_dataset.parquet`

이 파일은 sequence model 입력용 artifact다. `sample_id`를 기준으로 `sample_index.csv`와 연결하며, 각 sequence 컬럼은 예측 기준 시점까지 관측된 prefix만 포함한다. 현재 artifact는 12,294,147 rows, 123 row groups로 생성되어 있다.

### 2.1 컬럼 정의

| 컬럼명 | 타입 | 모델 역할 | 설명 |
| --- | --- | --- | --- |
| `sample_id` | string | key | 모델 간 평가 sample을 연결하는 고유 key다. `sample_index.csv`와 `tabular_feature_dataset.csv`의 `sample_id`와 동일한 집합을 가져야 한다. |
| `event_type_sequence` | string sequence | sequence_input | 기준 시점까지의 이벤트 타입 prefix sequence다. 예: `view`, `cart`, `remove_from_cart`. |
| `product_id_sequence` | string sequence | sequence_input | 기준 시점까지의 상품 ID prefix sequence다. sequence model의 item embedding 입력 후보로 사용한다. |
| `category_id_sequence` | string sequence | sequence_input | 기준 시점까지의 카테고리 ID prefix sequence다. sequence model의 category embedding 입력 후보로 사용한다. |
| `price_bin_sequence` | string sequence | sequence_input | 기준 시점까지의 가격 구간 prefix sequence다. 연속 price 원문 대신 bin label sequence를 사용한다. |
| `time_gap_minutes_sequence` | string sequence | sequence_input | 기준 시점까지의 이벤트 간 시간 간격 sequence다. 각 token은 직전 이벤트와의 차이를 분 단위 문자열로 저장한다. |

### 2.2 사용 및 검증 기준

| 기준 | 설명 |
| --- | --- |
| 공통 sample 계약 | `sample_id` 집합은 `sample_index.csv`, `tabular_feature_dataset.csv`, `sequence_feature_dataset.parquet`에서 모두 일치해야 한다. |
| 누수 방지 | sequence 컬럼은 기준 시점 이후 이벤트를 포함하지 않는다. |
| 길이 제한 | sequence는 `--max-sequence-length 50` 설정에 따라 최근 50개 prefix token으로 제한된다. |
| 모델 입력 제외 컬럼 | `user_id`, `user_session`, `cutoff_time`, `split`, `label`, `minutes_until_purchase`는 이 parquet에 포함하지 않는다. |
| 연결 방식 | 학습 시 label과 split은 `sample_index.csv`에서 `sample_id`로 join해서 사용한다. |
