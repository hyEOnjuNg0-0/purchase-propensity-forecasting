# Step 7 Baseline Models

## 핵심 요약

- 입력 artifact: `sample_index.csv`, `tabular_feature_dataset.csv`
- `sample_id`로 label/split을 join하고 audit metadata는 학습 입력에서 제외한다.
- categorical feature는 train split에서 관측한 값만 one-hot 인코딩한다.
- numeric 결측 대체와 표준화 통계는 train split에서만 계산한다.

## 모델 상태

- logistic_regression / none: trained (sklearn class_weight=none)
- lightgbm / none: trained (lightgbm scale_pos_weight=1.000000)
- logistic_regression / balanced: trained (sklearn class_weight=balanced)
- lightgbm / balanced: trained (lightgbm scale_pos_weight=14.036584)

## Validation PR-AUC

- logistic_regression / none: PR-AUC 0.250463, ROC-AUC 0.712400, F1 0.151206
- lightgbm / none: PR-AUC 0.297095, ROC-AUC 0.745729, F1 0.182267
- logistic_regression / balanced: PR-AUC 0.244111, ROC-AUC 0.721175, F1 0.273178
- lightgbm / balanced: PR-AUC 0.296197, ROC-AUC 0.746300, F1 0.255459
