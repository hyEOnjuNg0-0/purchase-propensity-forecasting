# 아키텍처 문서

## 1. 현재 디렉토리 구조

```text
C:\PurchaseTimeForecasting
|-- .dockerignore
|-- .gitignore
|-- Agents.md
|-- Dockerfile
|-- requirements.txt
|-- app
|   `-- streamlit_app.py
|-- data
|   |-- 2019-Oct.csv
|   `-- 2019-Nov.csv
`-- docs
    |-- ARCHITECTURE.md
    |-- PORTFOLIO_EXECUTION_PLAN.md
    |-- PROJECT_DESIGN.md
    |-- TECHSPEC.md
    `-- inital_plan.md
```

## 2. 문서 구조

| 파일 | 역할 |
| --- | --- |
| `docs/inital_plan.md` | 최초 러프 프로젝트 아이디어 |
| `docs/TECHSPEC.md` | 사용 언어, 도구, 라이브러리, 모델, 실행 환경 기술 스펙 |
| `docs/PROJECT_DESIGN.md` | 문제 정의, 데이터 명세, 검증 전략, 라벨링, 모델링 및 평가 설계 |
| `docs/PORTFOLIO_EXECUTION_PLAN.md` | 2주 단위 실행 계획과 산출물 기준 |
| `docs/ARCHITECTURE.md` | 저장소 구조와 문서/향후 코드 구조 설명 |

## 3. 실행 진입점

| 파일 | 역할 |
| --- | --- |
| `app/streamlit_app.py` | 분석 결과 artifact를 읽어 보여주는 Streamlit 대시보드 |
| `Dockerfile` | Streamlit 대시보드 실행용 최소 Docker 환경 |
| `.dockerignore` | 대용량 원천 데이터와 산출물이 Docker build context에 포함되지 않도록 제외 |
| `.gitignore` | 대용량 데이터, 산출물, 일부 계획 문서, 로컬 개발 파일을 Git 추적에서 제외 |
| `requirements.txt` | 대시보드와 향후 분석 코드의 Python 의존성 |

## 4. 예정 코드 구조

향후 구현 단계에서는 Clean Architecture 원칙에 따라 다음 구조를 권장한다.

```text
src
|-- domain
|   |-- labeling.py
|   |-- schema.py
|   `-- validation.py
|-- application
|   |-- build_dataset.py
|   |-- run_eda.py
|   |-- train_model.py
|   `-- evaluate_model.py
|-- infrastructure
|   |-- data_loader.py
|   |-- feature_store.py
|   `-- artifact_repository.py
|-- models
|   |-- baseline.py
|   |-- gru.py
|   |-- sasrec.py
|   `-- tisasrec.py
`-- visualization
    |-- attention.py
    |-- clustering.py
    `-- report_figures.py
```

테스트 구조는 다음과 같이 둔다.

```text
tests
|-- domain
|   |-- test_labeling.py
|   `-- test_validation.py
|-- application
|   `-- test_build_dataset.py
`-- models
    `-- test_sequence_batch.py
```

## 5. 설계 원칙

- `domain`: 라벨링, 데이터 검증, 핵심 규칙처럼 외부 프레임워크에 의존하지 않는 순수 로직을 둔다.
- `application`: 분석 실행 흐름과 use case orchestration을 담당한다.
- `infrastructure`: CSV 로딩, artifact 저장, 외부 라이브러리 연동을 담당한다.
- `models`: 모델 구현을 담당하되 데이터 로딩과 평가 리포트 생성 책임을 갖지 않는다.
- `visualization`: 리포트용 시각화 생성 책임을 담당한다.
- `app`: Streamlit presentation layer를 담당한다. 원천 데이터 처리와 모델 학습 로직을 포함하지 않고, 생성된 artifact만 읽는다.

## 6. Docker 실행

로컬 실행:

```powershell
streamlit run app/streamlit_app.py
```

Docker 실행:

```powershell
docker build -t purchase-time-forecasting .
docker run --rm -p 8501:8501 -v ${PWD}/artifacts:/app/artifacts purchase-time-forecasting
```

원천 데이터가 필요한 분석 script를 컨테이너에서 실행할 경우 `data/`도 별도 볼륨으로 마운트한다.