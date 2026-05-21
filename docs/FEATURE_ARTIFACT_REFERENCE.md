# Feature Artifact Reference

Step 5에서 생성한 모델 입력 artifact의 컬럼 요약이다. `sample_id`는 dataset 연결용 key이며, 모델 feature로 학습하지 않는다. label과 split은 `sample_index.csv`에서 join하되, feature matrix에는 넣지 않는다.

## `artifacts/features/tabular_feature_dataset.csv`

Logistic Regression, LightGBM 같은 tabular baseline 모델 입력용 feature다.

| 컬럼명 | 타입 | 설명 |
| --- | --- | --- |
| `sample_id` | string | `sample_index.csv`와 연결하는 sample key |
| `prefix_length` | integer | 기준 시점까지 관측된 이벤트 수 |
| `last_event_type` | category | 기준 시점의 마지막 이벤트 타입 |
| `session_elapsed_minutes` | float | 세션 시작부터 기준 시점까지 경과 시간 |
| `time_since_previous_event_minutes` | float | 직전 이벤트 이후 경과 시간 |
| `hour` | integer | 기준 시점의 UTC hour |
| `event_count_view` | integer | prefix 내 `view` 누적 횟수 |
| `event_count_cart` | integer | prefix 내 `cart` 누적 횟수 |
| `event_count_remove_from_cart` | integer | prefix 내 `remove_from_cart` 누적 횟수 |
| `unique_product_count` | integer | prefix 내 고유 상품 수 |
| `unique_category_count` | integer | prefix 내 고유 카테고리 수 |
| `last_price` | float | 기준 시점 이벤트의 가격 |
| `last_price_bin` | category | 기준 시점 이벤트의 가격 구간 |
| `user_past_event_count` | integer | 기준 시점 이전 해당 사용자의 누적 이벤트 수 |
| `user_past_purchase_count` | integer | 기준 시점 이전 해당 사용자의 누적 purchase 수 |
| `user_past_cart_count` | integer | 기준 시점 이전 해당 사용자의 누적 cart 수 |

## `artifacts/features/sequence_feature_dataset.parquet`

GRU, SASRec 같은 sequence model 입력용 feature다. sequence는 `--max-sequence-length 50` 설정에 따라 최근 50개 prefix token으로 제한된다.

| 컬럼명 | 타입 | 설명 |
| --- | --- | --- |
| `sample_id` | string | `sample_index.csv`와 연결하는 sample key |
| `event_type_sequence` | string sequence | 기준 시점까지의 이벤트 타입 sequence |
| `product_id_sequence` | string sequence | 기준 시점까지의 상품 ID sequence |
| `category_id_sequence` | string sequence | 기준 시점까지의 카테고리 ID sequence |
| `price_bin_sequence` | string sequence | 기준 시점까지의 가격 구간 sequence |
| `time_gap_minutes_sequence` | string sequence | 기준 시점까지의 이벤트 간 시간 간격 sequence |

Step 9 GRU 모델은 위 sequence 입력 컬럼 전체를 사용한다. `event_type_sequence`, `product_id_sequence`, `category_id_sequence`, `price_bin_sequence`는 train split에서 fit한 vocabulary로 embedding index를 만들고, `time_gap_minutes_sequence`는 train split에서 fit한 `log1p` 표준화 값으로 GRU 입력에 결합한다.

## 학습 입력 제외 컬럼

아래 컬럼은 artifact 연결, 평가, audit 용도로만 사용하며 모델 입력 feature로 학습하지 않는다.

| 컬럼명 | 위치 | 학습 입력 제외 이유 |
| --- | --- | --- |
| `sample_id` | 모든 feature artifact | dataset 연결 key이며 예측 신호로 사용하면 안 된다. |
| `user_session` | `sample_index.csv` | 세션 식별자라 모델이 세션 자체를 외울 수 있어 제외한다. |
| `user_id` | `sample_index.csv` | 사용자 식별자라 raw ID 과적합과 누수 위험이 있어 제외한다. |
| `cutoff_time` | `sample_index.csv` | 원문 timestamp는 제외하고 `hour`, elapsed/gap feature만 사용한다. |
| `split` | `sample_index.csv` | train/validation/test 분할용 metadata이며 feature가 아니다. |
| `label` | `sample_index.csv` | 예측 target이며 feature가 아니다. |
| `minutes_until_purchase` | `sample_index.csv` | positive sample 진단용 값이며 feature가 아니다. |

학습 시에는 `sample_id`로 `sample_index.csv`를 join해 `split`과 `label`만 가져오고, 모델 입력 `X`에서는 `sample_id`와 위 metadata 컬럼을 모두 제거한다.
