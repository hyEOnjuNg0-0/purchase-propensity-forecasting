from __future__ import annotations

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


def test_analyze_problem_validity_builds_step4_aggregates(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "\n".join(
            [
                "event_time,event_type,product_id,category_id,category_code,brand,price,user_id,user_session",
                "2019-10-01 00:00:00 UTC,view,1,10,electronics.phone,brand_a,10.0,101,s1",
                "2019-10-01 00:05:00 UTC,cart,1,10,electronics.phone,brand_a,20.0,101,s1",
                "2019-10-01 00:20:00 UTC,purchase,1,10,electronics.phone,brand_a,20.0,101,s1",
                "2019-10-01 01:00:00 UTC,view,2,20,apparel.shoes,brand_b,600.0,102,s2",
                "2019-10-01 01:45:00 UTC,cart,2,20,apparel.shoes,brand_b,800.0,102,s2",
                "2019-10-01 01:00:00 UTC,view,3,10,electronics.phone,brand_a,120.0,103,s3",
                "2019-10-01 02:00:00 UTC,purchase,3,10,electronics.phone,brand_a,90.0,103,s3",
            ]
        ),
        encoding="utf-8",
    )

    result = analyze_problem_validity_from_csv(
        csv_path,
        chunksize=3,
        max_pattern_length=2,
        top_n=10,
    )

    overview = {row["metric"]: row["value"] for row in result.overview_rows}
    assert overview["session_count"] == "3"
    assert overview["labeled_sample_count"] == "5"
    assert overview["positive_sample_count"] == "2"

    pattern_rows = {
        row["sequence_pattern"]: row for row in result.sequence_pattern_rows
    }
    assert pattern_rows["view > cart"]["session_count"] == "2"
    assert pattern_rows["view > cart"]["purchase_rate"] == "0.500000"

    hour_rows = {row["hour"]: row for row in result.hourly_purchase_rate_rows}
    assert hour_rows["0"]["positive_rate"] == "1.000000"
    assert hour_rows["1"]["positive_rate"] == "0.000000"

    sample_rows = {row["label_segment"]: row for row in result.sample_comparison_rows}
    assert sample_rows["positive"]["sample_count"] == "2"
    assert sample_rows["negative"]["sample_count"] == "3"


def test_eda_artifacts_are_written(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "\n".join(
            [
                "event_time,event_type,product_id,category_id,category_code,brand,price,user_id,user_session",
                "2019-10-01 00:00:00 UTC,view,1,10,electronics.phone,brand_a,10.0,101,s1",
            ]
        ),
        encoding="utf-8",
    )
    result = analyze_problem_validity_from_csv(csv_path, chunksize=1)

    write_eda_artifacts(result, tmp_path)

    assert (tmp_path / "eda_problem_validity_summary.csv").exists()
    assert (tmp_path / "eda_session_length_purchase_rate.csv").exists()
    assert (tmp_path / "eda_sequence_pattern_purchase_rate.csv").exists()
    assert (tmp_path / "eda_price_band_purchase_rate.csv").exists()
    assert (tmp_path / "eda_category_conversion.csv").exists()
    assert (tmp_path / "eda_hourly_purchase_rate.csv").exists()
    assert (tmp_path / "eda_positive_negative_sample_comparison.csv").exists()
    assert (tmp_path / "eda_report.md").exists()
