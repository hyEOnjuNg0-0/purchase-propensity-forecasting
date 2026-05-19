from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from purchase_time_forecasting.data_profiling import (  # noqa: E402
    build_event_type_rows,
    build_quality_issue_rows,
    build_summary_rows,
    profile_csv,
    write_profile_artifacts,
)


class DataProfilingTest(unittest.TestCase):
    def test_profile_csv_counts_core_step1_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "sample.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "event_time,event_type,product_id,category_id,category_code,brand,price,user_id,user_session",
                        "2019-10-01 00:00:00 UTC,view,1,10,electronics.phone,brand_a,100.0,101,s1",
                        "2019-10-01 00:03:00 UTC,cart,1,10,electronics.phone,brand_a,100.0,101,s1",
                        "2019-10-01 00:05:00 UTC,purchase,1,10,electronics.phone,brand_a,100.0,101,s1",
                        "2019-10-01 00:07:00 UTC,view,2,20,,brand_b,50.0,102,s2",
                        "not-a-time,wishlist,3,30,appliances,,30.0,103,s3",
                    ]
                ),
                encoding="utf-8",
            )

            profile = profile_csv(csv_path, chunksize=2)

        self.assertEqual(profile.row_count, 5)
        self.assertEqual(profile.event_time_min, "2019-10-01T00:00:00+00:00")
        self.assertEqual(profile.event_time_max, "2019-10-01T00:07:00+00:00")
        self.assertEqual(profile.event_time_parse_failures, 1)
        self.assertEqual(profile.event_type_counts["view"], 2)
        self.assertEqual(profile.event_type_counts["purchase"], 1)
        self.assertEqual(profile.unique_counts["user_id"], 3)
        self.assertEqual(profile.unique_counts["user_session"], 3)
        self.assertEqual(profile.purchase_event_count, 1)
        self.assertEqual(profile.purchase_session_count, 1)
        self.assertAlmostEqual(profile.purchase_event_ratio, 0.2)
        self.assertAlmostEqual(profile.purchase_session_ratio, 1 / 3)

    def test_report_rows_include_quality_issue_draft(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "sample.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "event_time,event_type,product_id,category_id,category_code,brand,price,user_id,user_session",
                        "bad-time,wishlist,1,10,,brand_a,100.0,101,s1",
                    ]
                ),
                encoding="utf-8",
            )
            profile = profile_csv(csv_path, chunksize=1)

            summary_rows = build_summary_rows(profile)
            event_rows = build_event_type_rows(profile)
            issue_rows = build_quality_issue_rows(profile)

            output_dir = Path(temp_dir) / "reports"
            write_profile_artifacts(profile, output_dir)

            self.assertTrue((output_dir / "data_profile_summary.csv").exists())
            self.assertTrue((output_dir / "data_profile_report.md").exists())

        self.assertTrue(any(row["metric"] == "row count" for row in summary_rows))
        self.assertEqual(event_rows[0]["event_type"], "wishlist")
        self.assertTrue(any(row["issue"] == "허용 범위 밖 event_type" for row in issue_rows))
        self.assertTrue(any(row["issue"] == "event_time 파싱 실패" for row in issue_rows))


if __name__ == "__main__":
    unittest.main()

