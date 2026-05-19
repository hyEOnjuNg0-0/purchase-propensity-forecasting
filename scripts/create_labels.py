"""Step 3 구매 예측 라벨링 실행 스크립트."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from purchase_time_forecasting.labeling import (  # noqa: E402
    LabelingPolicy,
    summarize_label_distribution_from_csv,
    write_label_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2019-Oct.csv Step 3 라벨 분포 artifact를 생성한다."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data" / "2019-Oct.csv",
        help="라벨링할 원천 CSV 경로",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "reports",
        help="라벨링 artifact 저장 디렉터리",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1_000_000,
        help="pandas read_csv chunk 크기",
    )
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=30,
        help="purchase label 예측 window",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = LabelingPolicy(prediction_window_minutes=args.window_minutes)
    result = summarize_label_distribution_from_csv(
        args.input,
        policy=policy,
        chunksize=args.chunksize,
    )
    write_label_artifacts(result, args.reports_dir, policy)
    print(f"candidate_event_count={result.candidate_event_count}")
    print(f"labeled_sample_count={result.labeled_sample_count}")
    print(f"positive_count={result.positive_count}")
    print(f"negative_count={result.negative_count}")
    print(f"positive_ratio={result.positive_ratio:.6f}")
    print(
        "excluded_at_or_after_first_purchase_count="
        f"{result.excluded_at_or_after_first_purchase_count}"
    )
    print(f"reports_dir={args.reports_dir}")


if __name__ == "__main__":
    main()
