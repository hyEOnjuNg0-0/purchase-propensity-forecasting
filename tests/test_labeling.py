from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402

from purchase_conversion_prediction.labeling import (  # noqa: E402
    LabelingPolicy,
    build_label_distribution_rows,
    create_prefix_labels,
    summarize_label_distribution,
    write_label_artifacts,
)


def test_create_prefix_labels_uses_only_events_at_or_before_cutoff() -> None:
    events = pd.DataFrame(
        [
            {
                "event_time": "2019-10-01 00:00:00 UTC",
                "event_type": "view",
                "user_session": "s1",
            },
            {
                "event_time": "2019-10-01 00:10:00 UTC",
                "event_type": "cart",
                "user_session": "s1",
            },
            {
                "event_time": "2019-10-01 00:20:00 UTC",
                "event_type": "purchase",
                "user_session": "s1",
            },
            {
                "event_time": "2019-10-01 00:25:00 UTC",
                "event_type": "view",
                "user_session": "s1",
            },
        ]
    )

    labels = create_prefix_labels(events)

    assert labels["label"].tolist() == [1, 1]
    assert labels["prefix_event_types"].tolist() == ["view", "view cart"]
    assert labels["prefix_event_types"].str.contains("purchase").sum() == 0
    assert labels["minutes_until_purchase"].tolist() == [20.0, 10.0]


def test_create_prefix_labels_respects_30_minute_window_and_session_boundary() -> None:
    events = pd.DataFrame(
        [
            {
                "event_time": "2019-10-01 00:00:00 UTC",
                "event_type": "view",
                "user_session": "s1",
            },
            {
                "event_time": "2019-10-01 00:31:00 UTC",
                "event_type": "purchase",
                "user_session": "s1",
            },
            {
                "event_time": "2019-10-01 00:02:00 UTC",
                "event_type": "view",
                "user_session": "s2",
            },
            {
                "event_time": "2019-10-01 00:03:00 UTC",
                "event_type": "purchase",
                "user_session": "s3",
            },
        ]
    )

    labels = create_prefix_labels(events)

    assert labels.loc[labels["user_session"].eq("s1"), "label"].tolist() == [0]
    assert labels.loc[labels["user_session"].eq("s2"), "label"].tolist() == [0]
    assert labels["user_session"].tolist() == ["s1", "s2"]


def test_label_distribution_counts_exclusions_and_ratio() -> None:
    events = pd.DataFrame(
        [
            {
                "event_time": "2019-10-01 00:00:00 UTC",
                "event_type": "view",
                "user_session": "s1",
            },
            {
                "event_time": "2019-10-01 00:05:00 UTC",
                "event_type": "purchase",
                "user_session": "s1",
            },
            {
                "event_time": "2019-10-01 00:06:00 UTC",
                "event_type": "view",
                "user_session": "s1",
            },
            {
                "event_time": "bad-time",
                "event_type": "view",
                "user_session": "s2",
            },
            {
                "event_time": "2019-10-01 00:07:00 UTC",
                "event_type": "view",
                "user_session": None,
            },
        ]
    )

    result = summarize_label_distribution(events)
    rows = build_label_distribution_rows(result)

    assert result.candidate_event_count == 5
    assert result.labeled_sample_count == 1
    assert result.positive_count == 1
    assert result.negative_count == 0
    assert result.excluded_at_or_after_first_purchase_count == 2
    assert result.excluded_invalid_time_count == 1
    assert result.excluded_missing_session_count == 1
    assert result.session_count == 1
    assert result.purchase_session_count == 1
    assert any(row["metric"] == "positive_ratio" and row["value"] == "1.000000" for row in rows)


def test_label_artifacts_are_written(tmp_path: Path) -> None:
    events = pd.DataFrame(
        [
            {
                "event_time": "2019-10-01 00:00:00 UTC",
                "event_type": "view",
                "user_session": "s1",
            }
        ]
    )
    policy = LabelingPolicy(prediction_window_minutes=30)
    result = summarize_label_distribution(events, policy)

    write_label_artifacts(result, tmp_path, policy)

    assert (tmp_path / "label_distribution.csv").exists()
    assert (tmp_path / "labeling_policy.csv").exists()
    assert (tmp_path / "labeling_report.md").exists()
