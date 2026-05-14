# 기술 스펙

이 문서는 프로젝트 구현에 사용하는 언어, 도구, 라이브러리, 모델, 실행 환경만 정의한다. 문제 정의, 데이터 검증, 라벨링, 평가 설계는 `docs/PROJECT_DESIGN.md`를 따른다.

## 1. 언어와 런타임

| 항목 | 선택 |
| --- | --- |
| 주 언어 | Python |
| Python 버전 | 3.11 |
| 문서 언어 | 한국어 |
| 주석 언어 | 한국어 |
| Shell 기준 | PowerShell |

## 2. 데이터 처리 도구

| 용도 | 도구 |
| --- | --- |
| DataFrame 처리 | pandas |
| 수치 연산 | numpy |
| 대용량 CSV 샘플링 후보 | pandas chunk processing |
| 데이터 검증 | pytest 기반 검증 로직 |
| 실험 artifact 저장 | CSV, JSON, PNG, Markdown |

## 3. 머신러닝 라이브러리

| 용도 | 라이브러리 |
| --- | --- |
| 전통 ML baseline | scikit-learn |
| Gradient Boosting baseline | LightGBM |
| 딥러닝 모델 | PyTorch |
| 평가 지표 | scikit-learn metrics |
| 차원 축소 | UMAP, scikit-learn t-SNE |
| 모델 해석 | SHAP |

## 4. 시각화 도구

| 용도 | 도구 |
| --- | --- |
| 정적 시각화 | matplotlib, seaborn |
| 인터랙티브 결과 확인 | Streamlit |
| 리포트용 figure 저장 | PNG |

## 5. 모델 스펙

### 5.1 Baseline 모델

| 모델 | 목적 | 라이브러리 |
| --- | --- | --- |
| Logistic Regression | 선형 baseline | scikit-learn |
| LightGBM | tabular feature 기반 강한 baseline | LightGBM |

### 5.2 Sequence 모델

| 모델 | 목적 | 라이브러리 |
| --- | --- | --- |
| GRU | 행동 sequence 기반 recurrent baseline | PyTorch |
| SASRec | self-attention 기반 sequence model | PyTorch |
| TiSASRec | time interval-aware attention 고도화 후보 | PyTorch |

### 5.3 입력 Embedding

- `event_type` embedding
- `product_id` embedding
- `category_id` embedding
- `price_bin` embedding
- `time_gap_bin` embedding
- positional encoding

## 6. 평가 지표

| 구분 | 지표 |
| --- | --- |
| Primary metric | PR-AUC |
| Secondary metrics | ROC-AUC, F1, Recall@K, Precision@K |
| Calibration | Brier score, reliability curve |

## 7. Streamlit 스펙

| 항목 | 선택 |
| --- | --- |
| 앱 위치 | `app/streamlit_app.py` |
| 기본 포트 | 8501 |
| 입력 데이터 | `artifacts/` 하위 실제 분석 결과 |
| 원천 CSV 직접 처리 | 하지 않음 |
| 모의 데이터 생성 | 하지 않음 |

Streamlit이 읽는 권장 artifact는 다음과 같다.

```text
artifacts
|-- reports
|   |-- project_summary.md
|   |-- data_quality_summary.csv
|   `-- model_metrics.csv
|-- figures
|   |-- event_distribution.png
|   |-- model_comparison.png
|   `-- representation_umap.png
`-- models
    `-- best_model_metadata.json
```

## 8. Docker 스펙

| 항목 | 선택 |
| --- | --- |
| Base image | `python:3.11-slim` |
| 실행 대상 | Streamlit 대시보드 |
| Dockerfile | `Dockerfile` |
| 제외 파일 | `.dockerignore` |
| 노출 포트 | 8501 |

Docker는 학습용 GPU 환경이나 운영 serving 환경이 아니라 결과 확인용 대시보드 실행 환경으로 제한한다.

