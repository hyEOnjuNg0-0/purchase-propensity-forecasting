# PurchaseConversionPrediction

세션 내 사용자 행동 이력을 기반으로 향후 30분 내 구매 가능성을 예측한 결과를 Streamlit 보고서로 확인하는 프로젝트입니다.

## Streamlit 결과 보고서 Docker 실행

이 Docker 이미지는 모델 학습용이 아니라, 이미 생성된 분석 결과 artifact를 읽어 Streamlit 결과 보고서를 실행하기 위한 용도입니다.

### 1. Docker 준비

Windows에서는 먼저 Docker Desktop을 실행하고 Docker engine이 정상 실행 중인지 확인합니다.

```powershell
docker --version
```

### 2. 저장소 받기

```powershell
git clone <repo-url>
cd PurchaseConversionPrediction
```

이미 저장소를 받은 상태라면 프로젝트 루트에서 실행합니다.

```powershell
cd C:\PurchaseConversionPrediction
```

### 3. 이미지 빌드

```powershell
docker build -t purchase-conversion-prediction-report .
```

### 4. 보고서 실행

```powershell
docker run --rm -p 8501:8501 purchase-conversion-prediction-report
```

### 5. 브라우저에서 확인

```text
http://localhost:8501
```

종료하려면 Docker를 실행한 터미널에서 `Ctrl + C`를 누릅니다.

## 주의할 점

- Docker 이미지는 Streamlit 결과 보고서 확인용입니다. 모델 학습이나 feature 재생성용 환경이 아닙니다.
- 원천 데이터 `data/2019-Oct.csv`와 대용량 `artifacts/features`는 Docker 이미지에 포함하지 않습니다.
- 보고서는 저장소에 포함된 `artifacts/reports` 결과 artifact를 읽어 표시합니다.
- 분석 결과를 처음부터 재생성하려면 Docker가 아니라 로컬 conda `ptf` 환경과 원천 데이터가 필요합니다.
