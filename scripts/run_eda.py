"""Step 4 EDA 및 문제 타당성 검증 실행 스크립트."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from purchase_time_forecasting.exploratory_analysis import (  # noqa: E402
    analyze_problem_validity_from_csv,
    write_eda_artifacts,
)
from purchase_time_forecasting.labeling import LabelingPolicy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2019-Oct.csv Step 4 EDA 및 문제 타당성 artifact를 생성한다."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data" / "2019-Oct.csv",
        help="EDA 대상 원천 CSV 경로",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "reports",
        help="EDA artifact 저장 디렉터리",
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
    parser.add_argument(
        "--max-pattern-length",
        type=int,
        default=5,
        help="sequence pattern에 사용할 최대 초기 이벤트 수",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="pattern/category artifact에 저장할 상위 행 수",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="빠른 EDA를 위해 원천 CSV 앞부분 N개 row만 분석한다. 생략하면 전체를 분석한다.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = LabelingPolicy(prediction_window_minutes=args.window_minutes)
    result = analyze_problem_validity_from_csv(
        args.input,
        policy=policy,
        chunksize=args.chunksize,
        max_pattern_length=args.max_pattern_length,
        top_n=args.top_n,
        max_rows=args.max_rows,
    )
    write_eda_artifacts(result, args.reports_dir)

    overview = {row["metric"]: row["value"] for row in result.overview_rows}
    print(f"session_count={overview['session_count']}")
    print(f"labeled_sample_count={overview['labeled_sample_count']}")
    print(f"positive_sample_count={overview['positive_sample_count']}")
    print(f"positive_sample_ratio={overview['positive_sample_ratio']}")
    print(f"reports_dir={args.reports_dir}")


if __name__ == "__main__":
    main()
