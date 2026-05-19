from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402

from purchase_time_forecasting.feature_engineering import (  # noqa: E402
    FeatureEngineeringPolicy,
    build_feature_artifacts,
    build_feature_dataset,
    build_feature_dictionary_rows,
    build_transformer_scope_rows,
)


def test_build_feature_dataset_keeps_prefix_features_before_future_purchase() -> None:
    events = pd.DataFrame(
        [
            {
                "event_time": "2019-10-01 00:00:00 UTC",
                "event_type": "view",
                "product_id": 1,
                "category_id": 10,
                "category_code": "electronics.phone",
                "brand": "brand_a",
                "price": 10.0,
                "user_id": 101,
                "user_session": "s1",
            },
            {
                "event_time": "2019-10-01 00:05:00 UTC",
                "event_type": "cart",
                "product_id": 2,
                "category_id": 10,
                "category_code": "electronics.phone",
                "brand": "brand_a",
                "price": 20.0,
                "user_id": 101,
                "user_session": "s1",
            },
            {
                "event_time": "2019-10-01 00:20:00 UTC",
                "event_type": "purchase",
                "product_id": 2,
                "category_id": 10,
                "category_code": "electronics.phone",
                "brand": "brand_a",
                "price": 20.0,
                "user_id": 101,
                "user_session": "s1",
            },
            {
                "event_time": "2019-10-01 01:00:00 UTC",
                "event_type": "view",
                "product_id": 3,
                "category_id": 20,
                "category_code": "apparel.shoes",
                "brand": "brand_b",
                "price": 600.0,
                "user_id": 101,
                "user_session": "s2",
            },
        ]
    )

    features = build_feature_dataset(events)

    assert features["label"].tolist() == [1, 1, 0]
    assert features["event_type_sequence"].tolist() == ["view", "view cart", "view"]
    assert features["event_type_sequence"].str.contains("purchase").sum() == 0
    assert features.loc[1, "time_gap_minutes_sequence"] == "0.000000 5.000000"
    assert features.loc[0, "user_past_event_count"] == 0
    assert features.loc[1, "user_past_event_count"] == 1
    assert features.loc[2, "user_past_purchase_count"] == 1


def test_feature_dictionary_excludes_raw_ids_and_raw_timestamp_from_model_input() -> None:
    dictionary_rows = build_feature_dictionary_rows()
    roles = {row["feature_name"]: row["model_role"] for row in dictionary_rows}

    assert roles["user_session"] == "key"
    assert roles["user_id"] == "audit_only"
    assert roles["cutoff_time"] == "audit_only"
    assert roles["label"] == "target"
    assert roles["event_type_sequence"] == "sequence_input"
    assert roles["session_elapsed_minutes"] == "tabular_input"


def test_transformer_scope_uses_train_split_only() -> None:
    events = pd.DataFrame(
        [
            {
                "event_time": "2019-10-01 00:00:00 UTC",
                "event_type": "view",
                "product_id": 1,
                "category_id": 10,
                "category_code": "electronics.phone",
                "brand": "brand_a",
                "price": 10.0,
                "user_id": 101,
                "user_session": "s1",
            },
            {
                "event_time": "2019-10-01 00:05:00 UTC",
                "event_type": "cart",
                "product_id": 2,
                "category_id": 10,
                "category_code": "electronics.phone",
                "brand": "brand_a",
                "price": 20.0,
                "user_id": 101,
                "user_session": "s1",
            },
            {
                "event_time": "2019-10-01 01:00:00 UTC",
                "event_type": "remove_from_cart",
                "product_id": 3,
                "category_id": 20,
                "category_code": "apparel.shoes",
                "brand": "brand_b",
                "price": 600.0,
                "user_id": 102,
                "user_session": "s2",
            },
            {
                "event_time": "2019-10-01 02:00:00 UTC",
                "event_type": "view",
                "product_id": 4,
                "category_id": 30,
                "category_code": "kids.toys",
                "brand": "brand_c",
                "price": 5.0,
                "user_id": 103,
                "user_session": "s3",
            },
        ]
    )
    policy = FeatureEngineeringPolicy(train_ratio=0.5, validation_ratio=0.25)
    features = build_feature_dataset(events, policy=policy)

    rows = build_transformer_scope_rows(features)
    vocab_by_feature = {
        row["feature_name"]: row["fitted_values"]
        for row in rows
        if row["transformer_type"] == "categorical_vocab"
    }

    assert features["split"].tolist() == ["train", "train", "validation", "test"]
    assert vocab_by_feature["last_event_type"] == "cart|view"
    assert "remove_from_cart" not in vocab_by_feature["last_event_type"]
    assert all(row["fit_split"] == "train" for row in rows)


def test_feature_artifacts_are_written(tmp_path: Path) -> None:
    events = pd.DataFrame(
        [
            {
                "event_time": "2019-10-01 00:00:00 UTC",
                "event_type": "view",
                "product_id": 1,
                "category_id": 10,
                "category_code": "electronics.phone",
                "brand": "brand_a",
                "price": 10.0,
                "user_id": 101,
                "user_session": "s1",
            }
        ]
    )
    features = build_feature_dataset(events)

    build_feature_artifacts(features, tmp_path / "features", tmp_path / "reports")

    assert (tmp_path / "features" / "feature_dataset.csv").exists()
    assert (tmp_path / "reports" / "feature_dictionary.csv").exists()
    assert (tmp_path / "reports" / "feature_leakage_checklist.csv").exists()
    assert (tmp_path / "reports" / "feature_transformer_scope.csv").exists()
    assert (tmp_path / "reports" / "feature_report.md").exists()
