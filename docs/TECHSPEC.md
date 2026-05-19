# 기술 스펙

## 1. 실행 환경

| 항목 | 값 |
| --- | --- |
| 언어 | Python |
| Python 버전 | 3.11 |
| Shell | PowerShell |
| 문서/주석 언어 | 한국어 |

## 2. 핵심 라이브러리

| 구분 | 라이브러리 |
| --- | --- |
| 전통 ML | scikit-learn, LightGBM |
| 딥러닝 | PyTorch |
| 평가 | scikit-learn metrics |
| 차원 축소 | UMAP, scikit-learn t-SNE |
| 모델 해석 | SHAP |
| 시각화 | matplotlib, seaborn, Streamlit |

## 3. 모델

| 분류 | 모델 |
| --- | --- |
| Baseline | Logistic Regression, LightGBM |
| Sequence | GRU, SASRec, TiSASRec |

## 4. 평가 지표

| 분류 | 지표 |
| --- | --- |
| Primary | PR-AUC |
| Secondary | ROC-AUC, F1, Recall@K, Precision@K |
| Calibration | Brier score, reliability curve |

## 5. 실행 산출물/배포

| 항목 | 값 |
| --- | --- |
| 대시보드 엔트리 | `app/streamlit_app.py` |
| 기본 포트 | 8501 |
| Docker Base image | `python:3.11-slim` |
| Docker 실행 대상 | Streamlit 대시보드 |
