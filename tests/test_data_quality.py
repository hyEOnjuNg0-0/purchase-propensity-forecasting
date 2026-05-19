from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from purchase_time_forecasting.data_quality import (  # noqa: E402
    build_quality_summary_rows,
    validate_data_quality,
    write_quality_artifacts,
)


def test_validate_data_quality_detects_step2_issues(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "\n".join(
            [
                "event_time,event_type,product_id,category_id,category_code,brand,price,user_id,user_session",
                "2019-10-01 00:10:00 UTC,view,1,10,electronics.phone,brand_a,100.0,101,s1",
                "2019-10-01 00:05:00 UTC,cart,1,10,electronics.phone,brand_a,0.0,101,s1",
                "2019-10-01 00:20:00 UTC,purchase,1,10,electronics.phone,brand_a,100.0,101,s1",
                "2019-10-01 00:20:00 UTC,purchase,1,10,electronics.phone,brand_a,100.0,101,s1",
                "bad-time,wishlist,2,20,,,-5.0,102,s2",
                "2019-10-01 00:30:00 UTC,view,3,30,appliances,brand_c,50.0,103,s3",
                "2019-10-01 00:31:00 UTC,view,3,30,appliances,brand_c,50.0,103,s3",
                "2019-10-01 00:32:00 UTC,view,3,30,appliances,brand_c,50.0,103,s3",
            ]
        ),
        encoding="utf-8",
    )

    result = validate_data_quality(
        csv_path,
        chunksize=3,
        extreme_session_min_length=3,
        top_extreme_sessions=5,
    )

    assert result.row_count == 8
    assert result.missing_columns == ()
    assert result.unexpected_event_types == ("wishlist",)
    assert result.event_time_parse_failures == 1
    assert result.price_zero_count == 1
    assert result.price_negative_count == 1
    assert result.invalid_price_count == 2
    assert result.duplicate_row_count == 1
    assert result.time_reversal_event_count == 1
    assert result.time_reversal_session_count == 1
    assert result.purchase_session_count == 1
    assert result.non_purchase_session_count == 2
    assert result.extreme_session_count == 2
    assert result.max_session_length == 4
    assert result.extreme_sessions[0]["user_session"] == "s1"


def test_quality_artifacts_are_written(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "\n".join(
            [
                "event_time,event_type,product_id,category_id,category_code,brand,price,user_id,user_session",
                "2019-10-01 00:00:00 UTC,view,1,10,electronics.phone,brand_a,100.0,101,s1",
            ]
        ),
        encoding="utf-8",
    )
    result = validate_data_quality(csv_path, chunksize=1, extreme_session_min_length=1)
    rows = build_quality_summary_rows(result)

    output_dir = tmp_path / "reports"
    write_quality_artifacts(result, output_dir)

    assert any(row["check_id"] == "schema_required_columns" for row in rows)
    assert (output_dir / "data_quality_summary.csv").exists()
    assert (output_dir / "data_quality_missing_values.csv").exists()
    assert (output_dir / "data_quality_session_comparison.csv").exists()
    assert (output_dir / "data_quality_extreme_sessions.csv").exists()
    assert (output_dir / "data_quality_report.md").exists()
    assert (output_dir / "data_quality_test_candidates.md").exists()

