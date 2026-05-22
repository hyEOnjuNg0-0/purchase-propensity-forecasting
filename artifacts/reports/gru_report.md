# Step 9 GRU Sequence Model

## 핵심 요약

- 입력 artifact: `sample_index.csv`, `sequence_feature_dataset.parquet`
- 입력 feature: `event_type_sequence`, `product_id_sequence`, `category_id_sequence`, `price_bin_sequence`, `time_gap_minutes_sequence`
- `sample_id`, `label`, `split`은 baseline과 동일 계약을 따른다.

## 모델 상태

- gru / pos_weight: trained (sequence_features=event_type_sequence+product_id_sequence+category_id_sequence+price_bin_sequence+time_gap_minutes_sequence, vocab_size=(event_type_sequence=4, product_id_sequence=10002, category_id_sequence=558, price_bin_sequence=9), max_sequence_length=50, embedding_dim=8, hidden_dim=16, epochs=3, batch_size=512, learning_rate=0.001000, device=cuda)

## Validation History

- epoch 1: train_loss 1.129510, validation PR-AUC 0.217547
- epoch 2: train_loss 1.020060, validation PR-AUC 0.199576
- epoch 3: train_loss 0.964017, validation PR-AUC 0.177210

## Test Metrics

- PR-AUC 0.173462, ROC-AUC 0.680508, F1 0.190954, Recall@K 0.310954, Precision@K 0.207545
