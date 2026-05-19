"""Step 1 데이터 프로파일링 실행 스크립트."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from purchase_time_forecasting.data_profiling import (  # noqa: E402
    profile_csv,
    write_profile_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2019-Oct.csv Step 1 데이터 프로파일링 artifact를 생성한다."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data" / "2019-Oct.csv",
        help="프로파일링할 원천 CSV 경로",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "reports",
        help="프로파일링 artifact 저장 디렉터리",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1_000_000,
        help="pandas read_csv chunk 크기",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = profile_csv(args.input, chunksize=args.chunksize)
    write_profile_artifacts(profile, args.reports_dir)
    print(f"row_count={profile.row_count}")
    print(f"event_time_min={profile.event_time_min}")
    print(f"event_time_max={profile.event_time_max}")
    print(f"purchase_event_ratio={profile.purchase_event_ratio:.6f}")
    print(f"purchase_session_ratio={profile.purchase_session_ratio:.6f}")
    print(f"reports_dir={args.reports_dir}")


if __name__ == "__main__":
    main()

