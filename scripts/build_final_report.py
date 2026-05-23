"""Step 10 최종 모델 비교 및 간단한 해석 artifact 생성 스크립트."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from purchase_conversion_prediction.streamlit_report import (  # noqa: E402
    build_final_model_comparison,
    build_model_interpretation_summary,
    write_final_report_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="baseline과 GRU metric을 통합해 최종 비교/해석 artifact를 생성한다."
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "reports",
        help="모델 metric과 최종 리포트 artifact 디렉터리",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    baseline_metrics = pd.read_csv(reports_dir / "model_metrics.csv")
    gru_metrics = pd.read_csv(reports_dir / "gru_model_metrics.csv")
    feature_importance = pd.read_csv(reports_dir / "baseline_feature_importance.csv")

    comparison = build_final_model_comparison(
        baseline_metrics=baseline_metrics,
        gru_metrics=gru_metrics,
    )
    interpretation = build_model_interpretation_summary(
        comparison=comparison,
        feature_importance=feature_importance,
    )
    write_final_report_artifacts(
        comparison=comparison,
        interpretation_markdown=interpretation,
        reports_dir=reports_dir,
    )

    print(f"final_model_comparison_rows={len(comparison)}")
    print(f"reports_dir={reports_dir}")


if __name__ == "__main__":
    main()
