"""Step 5 Feature Engineering 실행 스크립트."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from purchase_time_forecasting.feature_engineering import (  # noqa: E402
    FeatureEngineeringPolicy,
    build_feature_artifacts,
    build_feature_dataset_from_csv,
    build_feature_dataset_from_csv_streaming,
    build_split_summary_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2019-Oct.csv Step 5 feature dataset 및 문서화 artifact를 생성한다."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data" / "2019-Oct.csv",
        help="feature 생성 대상 원천 CSV 경로",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "features",
        help="feature dataset 저장 디렉터리",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "reports",
        help="feature 문서화 artifact 저장 디렉터리",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="pandas read_csv chunk 크기",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help=(
            "빠른 재현 실행을 위해 원천 CSV 앞부분 N개 row만 사용한다. "
            "생략하면 전체 CSV를 streaming 방식으로 처리한다."
        ),
    )
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=30,
        help="purchase label 예측 window",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="시간 기준 train split 비율",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.15,
        help="시간 기준 validation split 비율",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = FeatureEngineeringPolicy(
        prediction_window_minutes=args.window_minutes,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
    )
    if args.max_rows is None:
        result = build_feature_dataset_from_csv_streaming(
            args.input,
            args.features_dir,
            args.reports_dir,
            policy=policy,
            chunksize=args.chunksize,
        )
        feature_sample_count = int(result["feature_sample_count"])
        split_rows = result["split_rows"]
    else:
        features = build_feature_dataset_from_csv(
            args.input,
            policy=policy,
            chunksize=args.chunksize,
            max_rows=args.max_rows,
        )
        build_feature_artifacts(
            features,
            args.features_dir,
            args.reports_dir,
            source_path=args.input,
            max_rows=args.max_rows,
        )
        feature_sample_count = len(features)
        split_rows = build_split_summary_rows(features)

    print(f"feature_sample_count={feature_sample_count}")
    for row in split_rows:
        print(
            f"{row['split']}_sample_count={row['sample_count']} "
            f"{row['split']}_positive_ratio={row['positive_ratio']}"
        )
    print(f"features_dir={args.features_dir}")
    print(f"reports_dir={args.reports_dir}")


if __name__ == "__main__":
    main()
