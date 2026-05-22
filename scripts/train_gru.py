"""Step 9 GRU 모델 학습 실행 스크립트."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from purchase_time_forecasting.sequence_modeling import (  # noqa: E402
    GruTrainingPolicy,
    SequenceDatasetPolicy,
    load_gru_event_type_dataset,
    train_gru_classifier,
    validate_gru_device,
    write_gru_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 9 sequence feature artifact 전체를 사용하는 GRU 모델을 학습한다."
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "features",
        help="Step 5 sequence feature artifact 디렉터리",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "reports",
        help="GRU 결과 artifact 저장 디렉터리",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help="빠른 검증용으로 split별 앞쪽 N개 sample만 사용한다.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=50,
        help="GRU 입력에 사용할 최근 sequence feature 최대 길이",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=8,
        help="categorical sequence feature별 embedding 차원",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=16,
        help="GRU hidden state 차원",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="학습 batch 크기",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="학습 epoch 수",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Adam optimizer learning rate",
    )
    parser.add_argument(
        "--top-k-fraction",
        type=float,
        default=0.1,
        help="Recall@K/Precision@K 계산에 사용할 상위 비율",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="F1 계산에 사용할 classification threshold",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="PyTorch device. 예: auto, cpu, cuda, gpu(cuda alias)",
    )
    parser.add_argument(
        "--disable-pos-weight",
        action="store_true",
        help="class imbalance 보정용 pos_weight를 사용하지 않는다.",
    )
    return parser.parse_args()


def log_progress(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def main() -> None:
    args = parse_args()
    log_progress("GRU 학습 스크립트 시작")
    device = validate_gru_device(args.device)
    log_progress(f"GRU device 요청 검증 완료: {args.device} -> {device}")
    dataset_policy = SequenceDatasetPolicy(
        max_sequence_length=args.max_sequence_length,
    )
    dataset = load_gru_event_type_dataset(
        args.features_dir,
        policy=dataset_policy,
        max_samples_per_split=args.max_samples_per_split,
        progress_callback=log_progress,
    )
    training_policy = GruTrainingPolicy(
        max_sequence_length=args.max_sequence_length,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        threshold=args.threshold,
        top_k_fraction=args.top_k_fraction,
        use_pos_weight=not args.disable_pos_weight,
        device=device,
    )
    result = train_gru_classifier(
        dataset,
        policy=training_policy,
        progress_callback=log_progress,
    )
    log_progress("GRU artifact 저장 시작")
    write_gru_artifacts(result, args.reports_dir)
    log_progress("GRU artifact 저장 완료")

    print(f"gru_sample_count={dataset.sample_count}")
    print(f"reports_dir={args.reports_dir}")
    for row in result.model_status.to_dict("records"):
        print(
            f"{row['model_name']}:{row['class_imbalance_strategy']}="
            f"{row['status']}"
        )
    log_progress("GRU 학습 스크립트 완료")


if __name__ == "__main__":
    main()
