# 아키텍처

```text
C:\PurchaseTimeForecasting
|-- .dockerignore
|-- .gitignore
|-- Agents.md
|-- Dockerfile
|-- pytest.ini
|-- requirements.txt
|-- app
|   `-- streamlit_app.py
|-- artifacts
|   `-- reports
|       |-- data_quality_extreme_sessions.csv
|       |-- data_quality_missing_values.csv
|       |-- data_quality_report.md
|       |-- data_quality_session_comparison.csv
|       |-- data_quality_summary.csv
|       |-- data_quality_test_candidates.md
|       |-- data_profile_event_type_distribution.csv
|       |-- data_profile_missing_values.csv
|       |-- data_profile_report.md
|       |-- data_profile_summary.csv
|       |-- data_quality_issues_draft.csv
|       |-- eda_category_conversion.csv
|       |-- eda_hourly_purchase_rate.csv
|       |-- eda_positive_negative_sample_comparison.csv
|       |-- eda_price_band_purchase_rate.csv
|       |-- eda_problem_validity_summary.csv
|       |-- eda_report.md
|       |-- eda_sequence_pattern_purchase_rate.csv
|       |-- eda_session_length_purchase_rate.csv
|       |-- label_distribution.csv
|       |-- labeling_policy.csv
|       `-- labeling_report.md
|-- data
|   |-- 2019-Oct.csv
|-- scripts
|   |-- create_labels.py
|   |-- run_eda.py
|   |-- profile_data.py
|   |-- validate_data_quality.py
|   `-- run_ptf.ps1
|-- src
|   `-- purchase_time_forecasting
|       |-- __init__.py
|       |-- data_quality.py
|       |-- data_profiling.py
|       |-- exploratory_analysis.py
|       `-- labeling.py
|-- tests
|   |-- test_data_quality.py
|   |-- test_data_profiling.py
|   |-- test_exploratory_analysis.py
|   `-- test_labeling.py
`-- docs
    |-- ARCHITECTURE.md
    |-- PORTFOLIO_EXECUTION_PLAN.md
    |-- PROJECT_DESIGN.md
    `-- TECHSPEC.md
```

## 구성 변경 내역

- `src/purchase_time_forecasting/data_profiling.py`: Step 1 데이터 프로파일링 핵심 로직이다. 대용량 CSV를 chunk 단위로 읽어 row count, dtype, memory footprint, 시간 범위, event_type 분포, 고유값 수, purchase 비율을 계산한다.
- `scripts/profile_data.py`: `ptf` 환경 래퍼로 실행하는 Step 1 CLI 진입점이다.
- `tests/test_data_profiling.py`: 프로파일링 계산과 artifact 저장 계약을 검증하는 테스트다.
- `src/purchase_time_forecasting/data_quality.py`: Step 2 데이터 신뢰성 검증 핵심 로직이다. 필수 컬럼, 결측률, 이상 가격, 완전 중복 row, 세션 내 시간 역전, 극단 세션, purchase/비purchase 세션 차이를 검증한다.
- `scripts/validate_data_quality.py`: `ptf` 환경 래퍼로 실행하는 Step 2 CLI 진입점이다.
- `tests/test_data_quality.py`: 데이터 신뢰성 검증 계산과 artifact 저장 계약을 검증하는 pytest 테스트다.
- `src/purchase_time_forecasting/labeling.py`: Step 3 라벨링 핵심 로직이다. 세션 내 prefix 라벨을 생성하고, 대용량 CSV에 대해 첫 purchase 기준 30분 window 라벨 분포를 chunk 단위로 계산한다.
- `scripts/create_labels.py`: `ptf` 환경 래퍼로 실행하는 Step 3 CLI 진입점이다.
- `tests/test_labeling.py`: prefix feature가 기준 시점 이후 이벤트를 포함하지 않는지, 30분 window와 세션 경계를 지키는지, artifact 저장 계약을 검증하는 pytest 테스트다.
- `src/purchase_time_forecasting/exploratory_analysis.py`: Step 4 EDA 및 문제 타당성 검증 핵심 로직이다. 세션 길이, 초기 sequence pattern, 가격대, category, 시간대, positive/negative sample 차이를 첫 purchase 기준 라벨 window로 집계한다.
- `scripts/run_eda.py`: `ptf` 환경 래퍼로 실행하는 Step 4 CLI 진입점이다.
- `tests/test_exploratory_analysis.py`: EDA 집계와 artifact 저장 계약을 검증하는 pytest 테스트다.
- `pytest.ini`: pytest 임시 파일을 저장소 내부 `.pytest_tmp`에 생성하도록 고정한다.
- `artifacts/reports`: Step 1/2/3/4 실행 결과를 저장하는 리포트 artifact 디렉터리다.
