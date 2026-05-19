# Step 5 Feature Engineering

## 핵심 요약

- 대상 파일: `C:\PurchaseTimeForecasting\data\2019-Oct.csv`
- 입력 row 제한: 20,000
- 종료 일시 필터: 없음
- feature sample 수: 19,441
- raw `user_id`, `user_session`, `cutoff_time`은 모델 입력에서 제외하고 audit/key 용도로만 유지한다.
- sequence feature는 기준 시점까지의 prefix만 포함한다.
- encoder/scaler fit 범위는 train split으로 제한한다.

## 산출물

- `artifacts/features/feature_dataset.csv`: baseline/sequence 모델 공용 feature dataset
- `feature_dictionary.csv`: feature별 모델 입력 역할과 누수 정책
- `feature_leakage_checklist.csv`: Step 5 누수 방지 체크리스트
- `feature_transformer_scope.csv`: train split 기준 encoder/scaler fit 범위
- `feature_split_summary.csv`: split별 label 분포

## Split 요약

- train: 10,503 samples, positive ratio 0.045701
- validation: 4,281 samples, positive ratio 0.046952
- test: 4,657 samples, positive ratio 0.024479
