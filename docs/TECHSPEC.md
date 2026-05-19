# 기술 스펙

## 1. 실행 환경

| 항목 | 값 |
| --- | --- |
| 언어 | Python |
| Python 버전 | 3.10.20 |
| Conda 환경 | `ptf` |
| Shell | PowerShell |
| 문서/주석 언어 | 한국어 |
| 테스트 실행 | `.\scripts\run_ptf.ps1 python -m pytest` |

## 2. 핵심 라이브러리

| 구분 | 라이브러리 |
| --- | --- |
| 데이터 처리 | pandas 2.3.3 |
| 테스트 | pytest 9.0.3 |
| 대시보드 | Streamlit 1.57.0 |

## 3. 후속 모델링/분석 예정 라이브러리

| 구분 | 라이브러리 |
| --- | --- |
| 전통 ML | scikit-learn, LightGBM |
| 딥러닝 | PyTorch |
| 평가 | scikit-learn metrics |
| 차원 축소 | UMAP, scikit-learn t-SNE |
| 모델 해석 | SHAP |
| 시각화 | matplotlib, seaborn |

## 4. 모델

| 분류 | 모델 |
| --- | --- |
| Baseline | Logistic Regression, LightGBM |
| Sequence | GRU, SASRec, TiSASRec |

## 5. 평가 지표

| 분류 | 지표 |
| --- | --- |
| Primary | PR-AUC |
| Secondary | ROC-AUC, F1, Recall@K, Precision@K |
| Calibration | Brier score, reliability curve |

## 6. 실행 산출물/배포

| 항목 | 값 |
| --- | --- |
| 대시보드 엔트리 | `app/streamlit_app.py` |
| 기본 포트 | 8501 |
| Docker Base image | `python:3.10-slim` |
| Docker 실행 대상 | Streamlit 대시보드 |
