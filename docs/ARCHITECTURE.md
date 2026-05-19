# 아키텍처

```text
C:\PurchaseTimeForecasting
|-- .dockerignore
|-- .gitignore
|-- Agents.md
|-- Dockerfile
|-- requirements.txt
|-- app
|   `-- streamlit_app.py
|-- artifacts
|   `-- reports
|       |-- data_profile_event_type_distribution.csv
|       |-- data_profile_missing_values.csv
|       |-- data_profile_report.md
|       |-- data_profile_summary.csv
|       `-- data_quality_issues_draft.csv
|-- data
|   |-- 2019-Oct.csv
|-- scripts
|   |-- profile_data.py
|   `-- run_ptf.ps1
|-- src
|   `-- purchase_time_forecasting
|       |-- __init__.py
|       `-- data_profiling.py
|-- tests
|   `-- test_data_profiling.py
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
- `artifacts/reports`: Step 1 실행 결과를 저장하는 리포트 artifact 디렉터리다.
