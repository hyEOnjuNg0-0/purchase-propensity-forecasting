"""Step 7 baseline 모델 학습 실행 스크립트."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from purchase_conversion_prediction.baseline_modeling import (  # noqa: E402
    BaselineTrainingPolicy,
    load_baseline_dataset,
    train_baseline_models,
    write_baseline_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 7 Logistic Regression 및 LightGBM baseline을 학습한다."
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "features",
        help="Step 5 feature artifact 디렉터리",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "reports",
        help="baseline 결과 artifact 저장 디렉터리",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help="빠른 검증용으로 split별 실제 sample을 label 균형에 맞춰 최대 N개만 사용한다.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="제한 샘플 로드 시 pandas read_csv chunk 크기",
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
        "--logistic-max-iter",
        type=int,
        default=300,
        help="sklearn Logistic Regression 최대 반복 횟수",
    )
    parser.add_argument(
        "--lightgbm-n-estimators",
        type=int,
        default=200,
        help="LightGBM tree 수",
    )
    parser.add_argument(
        "--skip-lightgbm",
        action="store_true",
        help="LightGBM 의존성이 없거나 smoke test만 필요할 때 LightGBM 학습을 생략한다.",
    )
    return parser.parse_args()


def log_progress(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def main() -> None:
    args = parse_args()
    log_progress("baseline 학습 스크립트 시작")
    dataset = load_baseline_dataset(
        args.features_dir,
        max_samples_per_split=args.max_samples_per_split,
        chunksize=args.chunksize,
        progress_callback=log_progress,
    )
    policy = BaselineTrainingPolicy(
        threshold=args.threshold,
        top_k_fraction=args.top_k_fraction,
        logistic_max_iter=args.logistic_max_iter,
        lightgbm_n_estimators=args.lightgbm_n_estimators,
        train_lightgbm=not args.skip_lightgbm,
    )
    result = train_baseline_models(
        dataset,
        policy=policy,
        progress_callback=log_progress,
    )
    log_progress("baseline artifact 저장 시작")
    write_baseline_artifacts(result, args.reports_dir)
    log_progress("baseline artifact 저장 완료")

    print(f"baseline_sample_count={len(dataset)}")
    print(f"reports_dir={args.reports_dir}")
    for row in result.model_status.to_dict("records"):
        print(
            f"{row['model_name']}:{row['class_imbalance_strategy']}="
            f"{row['status']}"
        )
    log_progress("baseline 학습 스크립트 완료")


if __name__ == "__main__":
    main()
