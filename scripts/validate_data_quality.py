"""Step 2 데이터 신뢰성 검증 실행 스크립트."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from purchase_conversion_prediction.data_quality import (  # noqa: E402
    validate_data_quality,
    write_quality_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2019-Oct.csv Step 2 데이터 신뢰성 검증 artifact를 생성한다."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data" / "2019-Oct.csv",
        help="검증할 원천 CSV 경로",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "reports",
        help="검증 artifact 저장 디렉터리",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1_000_000,
        help="pandas read_csv chunk 크기",
    )
    parser.add_argument(
        "--extreme-session-min-length",
        type=int,
        default=100,
        help="극단 세션 최소 이벤트 수 기준",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_data_quality(
        args.input,
        chunksize=args.chunksize,
        extreme_session_min_length=args.extreme_session_min_length,
    )
    write_quality_artifacts(result, args.reports_dir)
    print(f"row_count={result.row_count}")
    print(f"duplicate_row_count={result.duplicate_row_count}")
    print(f"duplicate_row_ratio={result.duplicate_row_ratio:.6f}")
    print(f"invalid_price_count={result.invalid_price_count}")
    print(f"time_reversal_event_count={result.time_reversal_event_count}")
    print(f"extreme_session_threshold={result.extreme_session_threshold}")
    print(f"extreme_session_count={result.extreme_session_count}")
    print(f"reports_dir={args.reports_dir}")


if __name__ == "__main__":
    main()

