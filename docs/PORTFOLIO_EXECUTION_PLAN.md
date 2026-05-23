# 개인 과제 실행 계획

## 1. 목표

`2019-Oct.csv`를 대상으로 세션 내 행동 이력을 이용해 "향후 30분 내 구매 확률"을 예측하는 개인 과제를 완성한다. 결과물은 복잡한 연구 프로젝트가 아니라, 데이터 검증부터 baseline과 최소 딥러닝 sequence 모델 비교까지를 짧고 명확하게 보여주는 Streamlit 보고서로 구성한다.

핵심 메시지는 다음 세 가지로 제한한다.

- 30분 내 구매 예측 문제를 누수 없이 정의했다.
- tabular baseline과 GRU sequence 모델을 같은 sample 기준으로 비교했다.
- Streamlit에서 데이터 품질, 라벨 분포, EDA, 모델 성능, 한계점을 한 흐름으로 확인할 수 있다.

## 2. 최종 산출물

| 산출물 | 설명 |
| --- | --- |
| Streamlit 결과 보고서 | 문제 정의, 데이터 검증, EDA, 모델 비교, 한계점 표시 |
| 재현 가능한 script | 데이터 검증, 전처리, baseline 학습, GRU 학습, 평가 실행 흐름 |
| 실험 결과 표 | Logistic Regression, LightGBM, GRU의 PR-AUC, ROC-AUC, F1, Recall@K, Precision@K |
| 시각화 | EDA chart, 모델 성능 비교, LightGBM feature importance |
| Docker 실행 환경 | Streamlit 보고서 실행용 최소 Dockerfile |
| 기술 문서 | `TECHSPEC.md`, `PROJECT_DESIGN.md`, `ARCHITECTURE.md`, 실행 계획 문서 |

## 3. 범위

### 3.1 포함 범위

- `2019-Oct.csv` 단일 월 분석
- 세션 기반 sequence 생성
- 향후 30분 내 purchase 발생 여부 라벨링
- 데이터 신뢰성 검증
- Logistic Regression, LightGBM, GRU 모델 비교
- LightGBM feature importance 기반 간단한 해석
- Streamlit 결과 대시보드
- Docker 기반 대시보드 실행 환경

### 3.2 제외 범위

- 서비스 API 또는 웹앱 구현
- 운영 목적의 Streamlit 배포
- 실시간 inference pipeline
- 2019-Nov 기반 최종 검증
- 추천 시스템 구현
- 대규모 분산 처리 인프라 구축
- SASRec, TiSASRec 필수 구현
- attention map, embedding clustering, SHAP 등 고비용 해석 분석

## 4. 성공 기준

개인 과제 기준으로 다음 항목을 만족하면 완료로 판단한다.

- 문제 정의가 비즈니스 관점과 모델링 관점에서 모두 설명되어 있다.
- 데이터 신뢰성 검증 결과가 체크리스트와 수치로 정리되어 있다.
- 라벨 생성 방식과 데이터 누수 방지 전략이 명확하다.
- Logistic Regression, LightGBM, GRU의 성능 결과가 동일 split과 동일 metric으로 비교되어 있다.
- GRU가 baseline보다 낮거나 비슷하더라도, sequence 모델을 학습하고 결과를 해석했다.
- 성능이 기대보다 낮더라도 원인 분석과 후속 개선안이 Streamlit에서 명확히 확인된다.
- LightGBM feature importance 또는 오류 구간 분석 중 최소 1개 이상의 해석 결과가 포함되어 있다.
- Streamlit 대시보드가 실제 분석 artifact를 읽어 결과를 표시한다.
- Docker로 Streamlit 대시보드를 실행할 수 있다.

## 5. 실행 일정

### Step 1: 문제 정의 및 데이터 프로파일링

목표:

- 프로젝트 문제 정의를 확정한다.
- 원천 데이터의 크기, 스키마, 결측, 기본 분포를 확인한다.

작업:

- `2019-Oct.csv` row count, column dtype, memory footprint 확인
- `event_time` parsing 및 시간 범위 확인
- `event_type` 분포 확인
- 사용자 수, 세션 수, 상품 수, 카테고리 수 확인
- purchase 비율과 purchase 포함 세션 비율 확인

산출물:

- 데이터 프로파일링 요약표
- 데이터 품질 이슈 초안

### Step 2: 데이터 신뢰성 검증

목표:

- 모델링 전에 데이터가 신뢰 가능한지 검증한다.

작업:

- 필수 컬럼 존재 여부 검증
- 결측률 및 이상 가격 검증
- 중복 row 비율 확인
- 세션 내 시간 역전 여부 확인
- 극단적으로 긴 세션과 비정상 세션 탐색
- 구매 세션과 비구매 세션의 기본 차이 확인

산출물:

- 데이터 신뢰성 검증 artifact
- 검증 로직 테스트 후보 목록

### Step 3: 라벨링 설계 및 TDD

목표:

- 향후 30분 내 purchase label을 누수 없이 생성한다.

작업:

- 세션 내 prefix sequence 생성 규칙 정의
- 기준 시점 이후 30분 내 purchase 여부 라벨 생성
- 기준 시점 이후 이벤트가 feature에 포함되지 않는지 테스트 작성
- purchase 이후 시점의 sample 포함 여부 정책 결정
- positive/negative label 비율 확인

산출물:

- 라벨링 함수
- 라벨링 단위 테스트
- 라벨 분포표

### Step 4: EDA 및 문제 타당성 검증

목표:

- 라벨링된 문제의 데이터 사이언스적 타당성을 검증한다.

작업:

- 세션 길이별 구매율 분석
- event sequence pattern별 구매율 분석
- 가격대별 구매율 분석
- 카테고리별 전환율 분석
- 시간대별 구매율 분석
- positive/negative sample 차이 분석

산출물:

- EDA notebook 또는 script
- Streamlit 표시용 핵심 chart 후보

### Step 5: Feature Engineering 및 artifact 생성

목표:

- baseline과 sequence model에서 같은 sample을 공유하되, 모델별 입력 목적에 맞는 feature artifact를 구축한다.

작업:

- 공통 `sample_id` 생성
- `sample_index.csv` 생성
- tabular feature 생성
- baseline용 `tabular_feature_dataset.csv` 생성
- sequence model용 event/category/product/time gap sequence 생성
- sequence feature는 최근 `max_sequence_length`개 prefix로 제한
- sequence model용 `sequence_feature_dataset.parquet` 생성
- price binning 설계
- raw `user_id`, `user_session`, `event_time`은 모델 입력 feature에서 제외
- `user_session`은 sequence/prefix/label 생성 key로만 사용
- `user_id`는 기준 시점 이전 정보 기반 사용자 과거 행동 집계 feature 후보로만 사용
- train/validation/test split 생성
- encoder/scaler fit 범위 검증
- baseline/sequence dataset의 `sample_id`, `label`, `split` 일치 여부 검증

산출물:

- `artifacts/features/sample_index.csv`
- `artifacts/features/tabular_feature_dataset.csv`
- `artifacts/features/sequence_feature_dataset.parquet`
- feature dictionary
- raw ID 컬럼 제외 정책
- leakage 방지 체크리스트

### Step 6: Feature artifact 검증 및 재생성 안정화

목표:

- Step 7 학습을 시작하기 전에 Step 5에서 생성한 feature artifact가 정상 상태인지 한 번 점검한다.

범위:

- Step 6은 feature 구조를 다시 설계하거나 별도 artifact 분리 로직을 추가하는 단계가 아니다.
- `sample_index.csv`, `tabular_feature_dataset.csv`, `sequence_feature_dataset.parquet` 생성 책임은 Step 5에 둔다.
- Step 6 자체를 별도 포트폴리오 산출물로 제시하지 않는다.

작업:

- `sequence_feature_dataset.parquet`가 정상적으로 읽히는지 확인
- `sample_index.csv`, `tabular_feature_dataset.csv`, `sequence_feature_dataset.parquet`의 `sample_id` 집합 일치 여부 확인
- `sample_index.csv`의 `label`, `split` 분포와 `feature_split_summary.csv` 일치 여부 확인
- 후속 Step 7 baseline 학습에서 필요한 입력 컬럼과 제외 컬럼을 `feature_dictionary.csv` 기준으로 점검
- 문제가 있으면 Step 5 feature 생성 명령을 같은 설정으로 재실행하고 다시 확인

완료 기준:

- 세 feature artifact가 모두 존재하고 비어 있지 않다.
- parquet 파일을 정상적으로 읽을 수 있다.
- 세 artifact의 `sample_id` 기준이 서로 일치한다.
- `label`, `split` 분포가 학습을 진행할 수 있는 수준으로 확인된다.
- 별도 산출물 파일은 만들지 않고, 이상이 있을 때만 원인과 조치 내용을 기록한다.

### Step 7: Baseline 모델 구현

목표:

- 단순 baseline으로 문제의 난이도와 최소 성능 기준을 잡는다.

작업:

- `tabular_feature_dataset.csv`와 `sample_index.csv` 로드
- Logistic Regression 학습
- LightGBM 학습
- PR-AUC, ROC-AUC, F1, Recall@K 평가
- class imbalance 대응 전략 비교
- feature importance 확인

산출물:

- baseline 성능표
- feature importance 초안

### Step 8: Baseline 결과 정리 및 Streamlit 골격 작성

목표:

- Step 7 결과를 Streamlit 보고서에 바로 표시할 수 있는 형태로 정리한다.

작업:

- `model_metrics.csv`, `baseline_feature_importance.csv`, `baseline_model_status.csv` 존재 여부 확인
- Logistic Regression과 LightGBM의 test 성능을 하나의 표로 정리
- LightGBM feature importance 상위 feature를 Streamlit 표시용으로 정리
- Streamlit 보고서 기본 구조 작성
- artifact가 없을 때 모의 데이터 대신 생성 명령을 안내하도록 처리

산출물:

- Streamlit 보고서 초안
- baseline 결과 화면
- 누락 artifact 안내 메시지

### Step 9: GRU sequence 모델 구현

목표:

- sequence feature artifact 전체를 직접 입력하는 딥러닝 모델을 학습하고 baseline과 비교할 수 있게 한다.

범위:

- 필수 모델은 GRU 1개로 제한한다.
- 복잡한 튜닝보다 동일 sample, 동일 split, 동일 metric 비교와 입력 feature 계약 준수를 우선한다.
- SASRec과 TiSASRec은 필수 구현에서 제외하고 후속 개선안으로 둔다.

작업:

- `sequence_feature_dataset.parquet`와 `sample_index.csv` 로드
- `event_type_sequence`, `product_id_sequence`, `category_id_sequence`, `price_bin_sequence`, `time_gap_minutes_sequence`를 모두 포함하는 GRU dataset 구성
- categorical sequence는 train split 기준 vocabulary로 변환하고, time gap은 train split 기준 `log1p` 표준화 값으로 변환
- 작은 fixture 기반 단위 테스트로 dataset shape, padding, unknown token, numeric sequence, label 연결 검증
- GRU classifier 학습
- PR-AUC, ROC-AUC, F1, Recall@K, Precision@K 평가
- 학습 설정과 결과를 artifact로 저장

산출물:

- GRU 학습 코드
- GRU 테스트
- `artifacts/reports/gru_model_metrics.csv`
- `artifacts/reports/gru_model_status.csv`

### Step 10: 최종 모델 비교 및 간단한 해석

목표:

- Logistic Regression, LightGBM, GRU 결과를 하나의 비교표로 통합하고, Streamlit에서 읽기 쉬운 결론을 만든다.

작업:

- 세 모델의 metric artifact를 동일 schema로 통합
- 동일 `sample_id`, `label`, `split` 기준 비교였는지 점검
- 가장 중요한 metric은 PR-AUC로 표시하고 ROC-AUC, F1, Recall@K, Precision@K는 보조 지표로 둔다.
- LightGBM feature importance 상위 항목을 구매 행동 관점으로 짧게 해석
- GRU가 baseline 대비 개선됐는지, 개선되지 않았다면 가능한 원인을 정리
- 복잡한 SHAP, attention map, embedding clustering은 수행하지 않는다.

산출물:

- `artifacts/reports/final_model_comparison.csv`
- `artifacts/reports/model_interpretation_summary.md`
- Streamlit 모델 비교 화면

### Step 11: Streamlit 결과 보고서 완성

목표:

- 리뷰어가 별도 문서를 읽지 않아도 Streamlit에서 과제의 전체 흐름을 이해하도록 한다.

작업:

- Overview: 문제 정의, 예측 단위, 30분 label window 설명
- Data Quality: 스키마, 결측, 이상치, 세션 정합성 검증 결과 표시
- Labeling: 라벨 정책, label 분포, 누수 방지 체크 표시
- EDA: 세션 길이, sequence pattern, 가격대, 카테고리, 시간대별 구매율 중 핵심 chart 표시
- Model Results: Logistic Regression, LightGBM, GRU 성능 비교표와 chart 표시
- Interpretation: feature importance와 모델 비교 결론 표시
- Limitations: 데이터 한계, 모델 한계, 후속 개선안 표시

산출물:

- 완성된 `app/streamlit_app.py`
- Streamlit 입력 artifact 목록
- 누락 artifact 안내 메시지

### Step 12: Docker, 실행 순서, 마감 점검

목표:

- Streamlit 보고서가 재현 가능하게 실행되는지 확인하고 과제를 마감한다.

작업:

- 전체 실행 순서 점검
- `.\scripts\run_ptf.ps1 python -m pytest` 실행
- `.\scripts\run_ptf.ps1 streamlit run app/streamlit_app.py` 실행 확인
- Dockerfile이 Streamlit 보고서 실행에 필요한 최소 의존성을 포함하는지 확인
- 문서와 Streamlit 문구가 "30분 내 구매 확률 예측" 주제에 맞게 일관적인지 점검
- SASRec, TiSASRec, SHAP, attention 분석은 후속 개선안으로 정리

산출물:

- 최종 Streamlit 결과 보고서
- Docker 실행 명령: `docker build -t purchase-conversion-prediction-report .`, `docker run --rm -p 8501:8501 purchase-conversion-prediction-report`
- 테스트 결과
- 후속 개선안 요약: SASRec, TiSASRec, SHAP, attention/embedding 분석은 필수 구현 밖의 후속 개선안으로 표시

## 6. 리스크와 대응

| 리스크 | 영향 | 대응 |
| --- | --- | --- |
| 데이터 크기로 인한 처리 지연 | 일정 지연 | 2019-Oct 단일 월 유지, 필요 시 세션 단위 stratified sample 사용 |
| feature CSV 크기 폭증 | 디스크 사용량 증가 및 후속 학습 지연 | 공통 `sample_index.csv`, baseline용 tabular CSV, sequence용 parquet artifact로 분리하고 긴 prefix 문자열 반복 저장을 피함 |
| positive label 부족 | metric 불안정 | PR-AUC 중심 평가, threshold-independent metric 사용 |
| GRU 학습 시간이 길어짐 | 마감 지연 | sequence length와 sample 상한 옵션을 두고, 최소 학습 결과를 우선 확보 |
| GRU 성능 개선 미미 | 딥러닝 설득력 약화 | baseline 대비 성능 개선 여부보다 sequence 정보의 한계와 문제 특성을 명확히 해석 |
| 성능 개선 미미 | 포트폴리오 설득력 저하 | 모델 복잡도 대비 성능, 오류 분석, 데이터 한계를 중심으로 해석 |
| Streamlit 화면이 분석 품질을 흐림 | 포트폴리오 메시지 약화 | 학술적 리포트 형식 대신 대시보드 안에서 문제 정의, 검증, 결과, 해석을 짧고 명확하게 연결 |
| Docker 환경 구성 지연 | 마감 지연 | 학습 환경이 아니라 Streamlit 실행 환경만 컨테이너화 |

## 7. Streamlit 권장 화면 구성

1. Overview: 문제 정의, 예측 단위, 사용 데이터
2. Data Quality: 스키마, 결측, 이상치, 세션 정합성 검증 결과
3. Labeling: 30분 purchase label 정책, label 분포, 누수 방지 체크
4. EDA: 세션 길이, 행동 sequence, 가격대, 카테고리, 시간대별 구매율
5. Features: 공통 sample index, tabular feature, sequence feature, raw ID 제외 정책
6. Model Results: Logistic Regression, LightGBM, GRU 성능 비교
7. Interpretation: feature importance와 baseline 대비 GRU 결과 해석
8. Limitations: 데이터 한계, 성능 한계, 후속 개선안

## 8. 남은 열린 질문

- `purchase` 이후 같은 세션의 이벤트를 학습 sample로 포함할지 여부
- `remove_from_cart` 이벤트를 독립 행동으로 유지할지, cart 관련 파생 feature로 요약할지 여부
- session 길이가 1인 sample을 포함할지 여부
- `brand`, `category_code` 결측을 별도 unknown category로 처리할지 여부
- Streamlit에 포함할 EDA chart 우선순위
