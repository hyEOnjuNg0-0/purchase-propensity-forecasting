from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402

from purchase_conversion_prediction.feature_engineering import (  # noqa: E402
    FeatureEngineeringPolicy,
    build_feature_artifacts,
    build_feature_dataset,
    build_feature_dataset_from_csv,
    build_feature_dataset_from_csv_streaming,
    build_feature_dictionary_rows,
    build_sample_index,
    build_sequence_feature_dataset,
    build_tabular_feature_dataset,
    build_transformer_scope_rows,
    normalize_until_time,
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

    assert features["sample_id"].tolist() == [
        "sample_000000000000",
        "sample_000000000001",
        "sample_000000000003",
    ]
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

    assert roles["sample_id"] == "key"
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


def test_feature_artifacts_are_split_by_sample_contract(tmp_path: Path) -> None:
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
                "event_time": "2019-10-01 00:01:00 UTC",
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
                "event_time": "2019-10-01 00:02:00 UTC",
                "event_type": "view",
                "product_id": 3,
                "category_id": 11,
                "category_code": "electronics.tablet",
                "brand": "brand_b",
                "price": 30.0,
                "user_id": 101,
                "user_session": "s1",
            }
        ]
    )
    features = build_feature_dataset(events)

    build_feature_artifacts(
        features,
        tmp_path / "features",
        tmp_path / "reports",
        max_sequence_length=2,
    )

    sample_index = pd.read_csv(tmp_path / "features" / "sample_index.csv")
    tabular = pd.read_csv(tmp_path / "features" / "tabular_feature_dataset.csv")
    sequence = pd.read_parquet(tmp_path / "features" / "sequence_feature_dataset.parquet")

    assert sample_index.columns.tolist() == [
        "sample_id",
        "user_session",
        "user_id",
        "cutoff_time",
        "split",
        "label",
        "minutes_until_purchase",
    ]
    assert "label" not in tabular.columns
    assert "split" not in tabular.columns
    assert "event_type_sequence" not in tabular.columns
    assert sequence["sample_id"].tolist() == sample_index["sample_id"].tolist()
    assert tabular["sample_id"].tolist() == sample_index["sample_id"].tolist()
    assert sequence["event_type_sequence"].tolist()[-1] == "cart view"
    assert (tmp_path / "reports" / "feature_dictionary.csv").exists()
    assert (tmp_path / "reports" / "feature_leakage_checklist.csv").exists()
    assert (tmp_path / "reports" / "feature_transformer_scope.csv").exists()
    assert (tmp_path / "reports" / "feature_report.md").exists()


def test_feature_artifact_builders_share_sample_id_and_split_model_inputs() -> None:
    events = pd.DataFrame(
        [
            {
                "event_time": f"2019-10-01 00:0{minute}:00 UTC",
                "event_type": event_type,
                "product_id": minute + 1,
                "category_id": 10,
                "category_code": "electronics.phone",
                "brand": "brand_a",
                "price": 10.0 + minute,
                "user_id": 101,
                "user_session": "s1",
            }
            for minute, event_type in enumerate(["view", "cart", "view"])
        ]
    )
    features = build_feature_dataset(events)

    sample_index = build_sample_index(features)
    tabular = build_tabular_feature_dataset(features)
    sequence = build_sequence_feature_dataset(features, max_sequence_length=2)

    assert sample_index["sample_id"].tolist() == tabular["sample_id"].tolist()
    assert sample_index["sample_id"].tolist() == sequence["sample_id"].tolist()
    assert {"user_id", "user_session", "cutoff_time", "label", "split"}.isdisjoint(
        tabular.columns
    )
    assert tabular.columns.tolist() == [
        "sample_id",
        "prefix_length",
        "last_event_type",
        "session_elapsed_minutes",
        "time_since_previous_event_minutes",
        "hour",
        "event_count_view",
        "event_count_cart",
        "event_count_remove_from_cart",
        "unique_product_count",
        "unique_category_count",
        "last_price",
        "last_price_bin",
        "user_past_event_count",
        "user_past_purchase_count",
        "user_past_cart_count",
    ]
    assert sequence["event_type_sequence"].tolist() == ["view", "view cart", "cart view"]


def test_build_feature_dataset_from_csv_filters_until_date_inclusively(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "\n".join(
            [
                "event_time,event_type,product_id,category_id,category_code,brand,price,user_id,user_session",
                "2019-10-01 23:59:59 UTC,view,1,10,electronics.phone,brand_a,10.0,101,s1",
                "2019-10-02 00:00:00 UTC,view,2,20,apparel.shoes,brand_b,20.0,102,s2",
            ]
        ),
        encoding="utf-8",
    )

    features = build_feature_dataset_from_csv(
        csv_path,
        chunksize=1,
        until_time="2019-10-01",
    )

    assert len(features) == 1
    assert features["cutoff_time"].tolist() == ["2019-10-01T23:59:59+00:00"]
    assert normalize_until_time("2019-10-01").isoformat() == "2019-10-01T23:59:59.999999999+00:00"


def test_streaming_feature_dataset_filters_until_date(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "\n".join(
            [
                "event_time,event_type,product_id,category_id,category_code,brand,price,user_id,user_session",
                "2019-10-01 00:00:00 UTC,view,1,10,electronics.phone,brand_a,10.0,101,s1",
                "2019-10-01 00:10:00 UTC,purchase,1,10,electronics.phone,brand_a,10.0,101,s1",
                "2019-10-02 00:00:00 UTC,view,2,20,apparel.shoes,brand_b,20.0,102,s2",
            ]
        ),
        encoding="utf-8",
    )

    result = build_feature_dataset_from_csv_streaming(
        csv_path,
        tmp_path / "features",
        tmp_path / "reports",
        chunksize=2,
        until_time="2019-10-01",
    )
    sample_index = pd.read_csv(tmp_path / "features" / "sample_index.csv")
    tabular = pd.read_csv(tmp_path / "features" / "tabular_feature_dataset.csv")
    sequence = pd.read_parquet(tmp_path / "features" / "sequence_feature_dataset.parquet")

    assert result["feature_sample_count"] == 1
    assert sample_index["cutoff_time"].tolist() == ["2019-10-01T00:00:00+00:00"]
    assert sample_index["label"].tolist() == [1]
    assert tabular["sample_id"].tolist() == sample_index["sample_id"].tolist()
    assert sequence["sample_id"].tolist() == sample_index["sample_id"].tolist()
