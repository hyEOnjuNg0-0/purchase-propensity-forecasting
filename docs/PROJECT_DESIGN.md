# 프로젝트 설계서

이 문서는 PurchaseConversionPrediction 프로젝트의 문제 정의, 데이터 검증, 라벨링, feature 설계, 모델링 전략, 평가 설계, 해석 계획을 정의한다.

## 1. 프로젝트 개요

이 프로젝트는 Kaggle의 `eCommerce behavior data from multi category store` 중 `2019-Oct.csv`를 사용하여 사용자 행동 시퀀스 기반 구매 시점 예측 문제를 정의하고, 데이터 신뢰성 검증부터 모델 비교와 행동 패턴 해석까지 수행하는 데이터 사이언스 포트폴리오 프로젝트이다.

운영 서비스화는 범위에서 제외하되, 포트폴리오 결과를 검토하기 쉽도록 Streamlit 기반 결과 대시보드와 간단한 Docker 실행 환경을 포함한다. 최종 목표는 문제 정의 능력, 데이터 검증 역량, 실험 설계 역량, 모델 해석 능력을 Streamlit 대시보드와 재현 가능한 분석 코드로 보여주는 것이다.

## 2. 문제 정의

### 2.1 예측 목표

세션 내 특정 시점까지 관측된 사용자 행동 이력을 기반으로, 해당 시점 이후 30분 이내에 `purchase` 이벤트가 발생하는지 예측한다.

- 입력: 기준 시점 이전의 사용자 행동 시퀀스
- 출력: 향후 30분 내 구매 발생 여부
- 기본 예측 윈도우: 30분
- 민감도 분석 후보: 10분, 60분

### 2.2 예측 단위

기본 예측 단위는 `user_session` 내 prefix sequence이다.

예시:

```text
view -> cart -> view
```

위와 같이 기준 시점까지의 행동 이력이 주어졌을 때, 기준 시점 이후 30분 안에 같은 세션에서 `purchase`가 발생하면 positive label로 정의한다.

### 2.3 라벨링 정책

Step 3의 기본 라벨링 정책은 다음과 같다.

- 기준 시점: 세션 내 각 이벤트 시점이며, prefix feature에는 해당 시점까지 관측된 이벤트만 포함한다.
- positive label: 같은 `user_session`에서 `(cutoff_time, cutoff_time + 30분]` 구간에 첫 `purchase`가 발생하면 1로 정의한다.
- negative label: 같은 구간에 `purchase`가 발생하지 않으면 0으로 정의한다.
- purchase 이후 sample 정책: 첫 `purchase` 이벤트와 그 이후 같은 세션의 이벤트는 학습 sample에서 제외한다. 이미 구매가 관측된 이후의 이벤트를 입력에 포함하면 구매 예측 문제가 사후 설명 문제로 바뀔 수 있기 때문이다.
- 세션 경계: 라벨은 같은 `user_session` 내부 이벤트만 참조하며, 다른 세션의 purchase는 포함하지 않는다.
- 누수 방지: `prefix_event_types`와 후속 feature 후보는 기준 시점 이후 이벤트를 포함하지 않아야 하며, 단위 테스트로 검증한다.

### 2.4 포함 범위와 제외 범위

포함 범위:

- `2019-Oct.csv` 단일 월 분석
- 세션 기반 sequence 생성
- 향후 30분 내 purchase 발생 여부 라벨링
- 데이터 신뢰성 검증
- Logistic Regression, LightGBM, GRU, SASRec 모델 비교
- TiSASRec 선택 고도화 검토
- attention 또는 embedding 기반 행동 패턴 분석
- 분석 결과 확인용 Streamlit 대시보드
- 로컬 재현성을 위한 Dockerfile

제외 범위:

- 실시간 추론 API 또는 웹 서비스 구현
- 운영 목적의 Streamlit 배포
- 2019-Nov 데이터를 활용한 운영 수준의 out-of-time 검증
- 개인화 추천 시스템 구현
- 구매 금액 예측 또는 상품 추천 문제로의 확장

## 3. 데이터 명세

### 3.1 원천 데이터

- 파일: `data/2019-Oct.csv`
- 보조 파일: `data/2019-Nov.csv`
- 기준 분석 범위: `2019-Oct.csv`
- 데이터 출처: Kaggle `eCommerce behavior data from multi category store`

### 3.2 주요 컬럼

| 컬럼 | 의미 | 활용 |
| --- | --- | --- |
| `event_time` | 이벤트 발생 시각 | 시계열 정렬, 시간 차이 계산, 라벨 생성 |
| `event_type` | 사용자 행동 종류 | 행동 sequence feature |
| `product_id` | 상품 ID | item embedding, 다양성 feature |
| `category_id` | 카테고리 ID | category embedding, 카테고리 선호 |
| `category_code` | 카테고리 경로 | 상위 카테고리 파생 feature |
| `brand` | 브랜드 | 브랜드 선호, 결측 검증 |
| `price` | 가격 | 가격대 feature, 소비 패턴 |
| `user_id` | 사용자 ID | 사용자 단위 이력 집계, leakage 검증, raw ID 모델 입력 제외 |
| `user_session` | 세션 ID | 세션 단위 sequence 구성, 라벨 생성 key, raw ID 모델 입력 제외 |

## 4. 데이터 신뢰성 검증

### 4.1 스키마 검증

- 필수 컬럼 존재 여부 확인
- 컬럼별 dtype 검증
- `event_time` timezone 및 parsing 가능 여부 검증
- `event_type` 허용 값 확인: `view`, `cart`, `remove_from_cart`, `purchase`

### 4.2 결측 및 이상치 검증

- 컬럼별 결측률 계산
- `brand`, `category_code` 결측률의 모델 영향 분석
- `price <= 0` 또는 비정상 가격 분포 확인
- 같은 `user_session` 내 시간 역전 여부 확인

### 4.3 중복 및 정합성 검증

- 완전 중복 row 비율 확인
- 동일 사용자, 동일 세션, 동일 시각 이벤트의 중복 패턴 확인
- `purchase` 이벤트가 없는 세션과 있는 세션의 비율 확인
- 세션 길이 분포 및 극단적으로 긴 세션 확인

### 4.4 라벨 신뢰성 검증

- label 생성 시 기준 시점 이후 정보가 feature에 포함되지 않도록 검증
- 기준 시점 이후 30분 내 `purchase`만 positive로 사용
- 세션 종료 이후 이벤트를 라벨에 잘못 포함하지 않도록 검증
- 동일 세션 내 여러 purchase가 존재할 때 첫 purchase 기준과 전체 purchase 기준의 차이 확인

### 4.5 데이터 누수 방지

- 시간 기반 split 사용
- train 기간에서 fit한 encoder/scaler만 validation/test에 적용
- 라벨 생성 이후 feature 생성 과정에서 미래 이벤트가 섞이지 않도록 테스트 작성
- 집계 feature는 기준 시점 이전 이벤트만 사용

## 5. Feature 설계

### 5.0 공통 Sample 계약

baseline 모델과 sequence model은 저장 포맷과 입력 feature가 달라도 같은 평가 sample을 사용해야 공정하게 비교할 수 있다. 이를 위해 모든 모델용 dataset은 공통 `sample_id`를 기준으로 연결한다.

공통 sample index는 다음 컬럼을 가진다.

| 컬럼 | 역할 |
| --- | --- |
| `sample_id` | 모델 간 평가를 연결하는 고유 key |
| `user_session` | 세션 단위 라벨 검증 및 audit key |
| `user_id` | 사용자 단위 누수 검증 및 audit key |
| `cutoff_time` | 예측 기준 시점 |
| `split` | 시간 기준 train/validation/test 구분 |
| `label` | 향후 30분 내 purchase 발생 여부 |
| `minutes_until_purchase` | positive sample의 구매까지 남은 시간 |

`sample_id`, `user_session`, `user_id`, `cutoff_time`은 모델 입력 feature로 직접 사용하지 않고, dataset 연결과 검증을 위한 audit/key 컬럼으로만 사용한다.

### 5.1 공통 Feature

- 세션 내 이벤트 순서
- 기준 시점까지의 sequence length
- 마지막 이벤트 타입
- 이벤트 타입별 누적 횟수
- 마지막 이벤트 이후 경과 시간
- 세션 시작 이후 경과 시간
- 고유 상품 수, 고유 카테고리 수, 고유 브랜드 수
- 평균 가격, 최대 가격, 최근 가격

### 5.2 Sequence Feature

- `event_type` sequence
- `product_id` sequence
- `category_id` sequence
- `price` bin sequence
- 이벤트 간 time gap sequence
- sequence model 입력은 최근 `max_sequence_length`개 prefix로 제한한다.
- sequence feature는 CSV 문자열 반복 저장을 피하고, 모델 학습에 적합한 별도 artifact로 저장한다.

### 5.3 Time-aware Feature

- 연속 time gap을 log transform 후 binning
- 기준 시점의 hour, day of week
- 세션 내 상대 시간 위치

### 5.4 Raw ID 컬럼 처리 정책

`user_id`와 `user_session`은 모델 입력 feature로 직접 사용하지 않는다. 두 컬럼은 고유 식별자 성격이 강해 그대로 학습에 투입하면 특정 사용자나 세션을 외우는 방식의 과적합과 데이터 누수 위험이 크다.

- `user_session`
  - 사용 목적: 세션 단위 이벤트 묶음, 세션 내 정렬, prefix sequence 생성, 30분 purchase label 생성
  - 모델 입력: 제외
- `user_id`
  - 사용 목적: 사용자 단위 과거 행동 집계 feature 생성 후보, train/validation/test leakage 검증
  - 모델 입력: raw ID는 제외
  - 허용 후보: 기준 시점 이전 정보만 사용한 `user_past_session_count`, `user_past_purchase_count`, `user_past_cart_count`, `user_days_since_last_event`
- `event_time`
  - 사용 목적: 정렬, 시간 차이, label window 계산, 시간 파생 feature 생성
  - 모델 입력: 원문 timestamp는 제외하고 hour, day of week, elapsed time, time gap 등으로 변환
- `product_id`, `category_id`
  - baseline tabular 모델에서는 raw high-cardinality ID 직접 입력을 피하고 count/diversity/최근 item feature 또는 인코딩 정책을 별도로 둔다.
  - sequence model에서는 embedding 입력 후보로 사용할 수 있다.

모든 사용자 단위 집계 feature는 기준 시점 이전 이벤트만 사용해야 하며, train 기간에서 fit한 encoder/scaler만 validation/test에 적용한다.

### 5.5 Feature Artifact 분리 정책

Step 5 이후 feature artifact는 하나의 거대한 CSV로 모든 feature를 저장하지 않고, 목적에 따라 분리한다.

```text
artifacts/features
|-- sample_index.csv
|-- tabular_feature_dataset.csv
`-- sequence_feature_dataset.parquet
```

- `sample_index.csv`
  - 모든 모델이 공유하는 평가 기준이다.
  - `sample_id`, `user_session`, `user_id`, `cutoff_time`, `split`, `label`, `minutes_until_purchase`를 포함한다.
  - 모델 입력이 아니라 평가 정렬, 오류 분석, Streamlit 표시 연결에 사용한다.
- `tabular_feature_dataset.csv`
  - Logistic Regression, LightGBM baseline용 dataset이다.
  - `sample_id`와 tabular feature만 포함한다.
  - `event_type_sequence`, `product_id_sequence`, `category_id_sequence`, `price_bin_sequence`, `time_gap_minutes_sequence`처럼 긴 prefix 문자열은 저장하지 않는다.
- `sequence_feature_dataset.parquet`
  - GRU, SASRec 같은 sequence model용 dataset이다.
  - `sample_id`와 최근 `max_sequence_length`개 sequence feature를 포함한다.
  - CSV 문자열 반복 저장 대신 압축 가능한 columnar artifact로 저장한다.

이 분리 정책은 파일 크기를 줄이기 위한 단순 최적화가 아니라, 모델별 입력 표현을 명확히 분리하고 동일 `sample_id`/`label`/`split` 기준으로 성능을 비교하기 위한 계약이다.

## 6. 모델링 계획

### 6.1 Baseline 모델

1. Logistic Regression
   - 목적: 단순 feature 기반 선형 baseline 확보
   - 해석: coefficient 또는 odds ratio 중심

2. LightGBM
   - 목적: tabular feature 기반 강한 baseline 확보
   - 해석: feature importance, SHAP 분석 후보

### 6.2 Deep Learning 모델

1. GRU
   - 입력: event, category, price bin, time gap embedding
   - 목적: sequential pattern이 baseline 대비 성능을 개선하는지 검증
   - vocabulary 정책: `product_id_sequence`와 `category_id_sequence`는 train split 빈도 기준 상위 10,000개 토큰만 embedding vocabulary에 포함하고 나머지는 `<unknown>`으로 처리한다.

2. SASRec
   - 입력: sequence embedding + positional encoding
   - 목적: self-attention 기반 sequence modeling 성능 검증

3. TiSASRec
   - 상태: 선택 고도화
   - 목적: absolute position뿐 아니라 time interval 정보를 attention 구조에 반영
   - 일정 리스크가 높을 경우 설계와 후속 개선안으로 문서화

## 7. 평가 설계

### 7.1 Split 전략

`2019-Oct.csv` 내부에서 시간 기반 split을 사용한다.

- Train: 10월 초반 70%
- Validation: 10월 중반 15%
- Test: 10월 후반 15%

정확한 날짜 경계는 이벤트 수와 positive label 비율을 확인한 뒤 결정한다.

### 7.2 평가 지표

구매 예측은 class imbalance가 예상되므로 accuracy를 핵심 지표로 사용하지 않는다.

- Primary: PR-AUC
- Secondary: ROC-AUC, F1, Recall@K, Precision@K
- Calibration: reliability curve, Brier score

### 7.3 비교 관점

- 단순 baseline 대비 LightGBM 개선 여부
- tabular baseline 대비 sequence model 개선 여부
- sequence length별 성능 차이
- 구매까지 남은 시간 구간별 성능 차이
- 10분, 30분, 60분 label window 민감도

### 7.4 모델 비교 조건

baseline 모델과 sequence model의 성능 비교는 입력 feature 형식이 같아서 가능한 것이 아니라, 동일한 평가 sample을 예측하기 때문에 가능하다.

비교 시 반드시 다음 조건을 맞춘다.

- 동일 `sample_id` 집합
- 동일 `label`
- 동일 train/validation/test split
- 동일 metric 계산 코드
- 동일 평가 시점과 threshold 정책

따라서 성능 비교의 해석은 "같은 구매 예측 문제에서 tabular summary 표현과 sequence 표현 중 어느 쪽이 더 유효한가"로 둔다. sequence model이 baseline보다 높으면 행동 순서 정보의 추가 효용을 보여주고, 낮거나 비슷하면 현재 문제에서는 요약 feature가 충분히 강하거나 sequence 학습 이득이 제한적이라는 근거로 해석한다.

## 8. 해석 및 분석 계획

### 8.1 데이터 분석

- 이벤트 타입별 전환 흐름
- 세션 길이와 구매율 관계
- 가격대와 구매율 관계
- 카테고리별 구매 전환 차이
- 구매 세션과 비구매 세션의 행동 패턴 차이

### 8.2 모델 해석

- LightGBM feature importance 또는 SHAP
- GRU hidden vector 기반 사용자 행동 패턴 clustering
- SASRec attention map 시각화
- UMAP 또는 t-SNE 기반 sequence representation 시각화

### 8.3 대시보드 핵심 메시지

- 데이터 신뢰성 검증을 통해 라벨과 feature 생성의 타당성을 확보했다.
- 단순 이진 분류가 아니라 시간 제한이 있는 구매 발생 문제로 정의했다.
- baseline부터 sequence model까지 단계적으로 비교하여 모델 복잡도의 필요성을 검증했다.
- attention과 embedding 분석으로 예측 결과를 행동 패턴 관점에서 해석했다.

## 9. Streamlit 결과 대시보드

Streamlit은 운영 서비스가 아니라 포트폴리오 리뷰어가 결과물을 빠르게 확인하기 위한 presentation layer로 둔다.

### 9.1 표시 대상

- 프로젝트 문제 정의와 예측 단위
- 데이터 신뢰성 검증 요약
- EDA 핵심 시각화
- 모델별 성능 비교표
- 선택 모델의 주요 해석 결과
- 한계점과 후속 개선안

### 9.2 Artifact 계약

Streamlit은 원천 CSV를 직접 처리하지 않고, 분석 pipeline이 생성한 artifact만 읽는다.

권장 artifact:

```text
artifacts
|-- features
|   |-- sample_index.csv
|   |-- tabular_feature_dataset.csv
|   `-- sequence_feature_dataset.parquet
|-- reports
|   |-- data_quality_summary.csv
|   |-- label_distribution.csv
|   |-- labeling_policy.csv
|   |-- model_metrics.csv
|   `-- project_summary.md
|-- figures
|   |-- event_distribution.png
|   |-- model_comparison.png
|   `-- representation_umap.png
`-- models
    `-- best_model_metadata.json
```

artifact가 없을 경우 임의의 모의 데이터를 생성하지 않고, 필요한 파일 목록과 생성 절차를 안내한다.

## 10. 완료 기준

- 데이터 신뢰성 검증 결과가 문서화되어 있다.
- 라벨링과 누수 방지 로직에 대한 테스트가 존재한다.
- Logistic Regression, LightGBM, GRU, SASRec 중 최소 3개 모델 결과가 비교되어 있다.
- PR-AUC 중심의 평가 결과가 train/validation/test 기준으로 정리되어 있다.
- attention 또는 embedding 기반 해석 시각화가 최소 1개 이상 포함되어 있다.
- Streamlit에서 실제 artifact 기반 결과를 확인할 수 있다.
- Docker로 Streamlit 대시보드를 실행할 수 있다.
- 한계점과 후속 개선안이 Streamlit 대시보드에서 명확히 확인된다.
