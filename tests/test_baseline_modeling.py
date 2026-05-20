from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402

from purchase_time_forecasting.baseline_modeling import (  # noqa: E402
    BaselineTrainingPolicy,
    TabularPreprocessor,
    build_baseline_dataset,
    compute_binary_metrics,
    train_baseline_models,
    write_baseline_artifacts,
)


def _sample_index() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "user_session": "a",
                "user_id": "u1",
                "cutoff_time": "2019-10-01T00:00:00+00:00",
                "split": "train",
                "label": 0,
                "minutes_until_purchase": None,
            },
            {
                "sample_id": "s2",
                "user_session": "a",
                "user_id": "u1",
                "cutoff_time": "2019-10-01T00:05:00+00:00",
                "split": "train",
                "label": 1,
                "minutes_until_purchase": 5.0,
            },
            {
                "sample_id": "s3",
                "user_session": "b",
                "user_id": "u2",
                "cutoff_time": "2019-10-01T00:10:00+00:00",
                "split": "validation",
                "label": 1,
                "minutes_until_purchase": 10.0,
            },
            {
                "sample_id": "s4",
                "user_session": "c",
                "user_id": "u3",
                "cutoff_time": "2019-10-01T00:15:00+00:00",
                "split": "test",
                "label": 0,
                "minutes_until_purchase": None,
            },
        ]
    )


def _tabular_features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "prefix_length": 1,
                "last_event_type": "view",
                "session_elapsed_minutes": 0.0,
                "time_since_previous_event_minutes": None,
                "hour": 0,
                "event_count_view": 1,
                "event_count_cart": 0,
                "event_count_remove_from_cart": 0,
                "unique_product_count": 1,
                "unique_category_count": 1,
                "last_price": 10.0,
                "last_price_bin": "<=25",
                "user_past_event_count": 0,
                "user_past_purchase_count": 0,
                "user_past_cart_count": 0,
            },
            {
                "sample_id": "s2",
                "prefix_length": 2,
                "last_event_type": "cart",
                "session_elapsed_minutes": 5.0,
                "time_since_previous_event_minutes": 5.0,
                "hour": 0,
                "event_count_view": 1,
                "event_count_cart": 1,
                "event_count_remove_from_cart": 0,
                "unique_product_count": 2,
                "unique_category_count": 1,
                "last_price": 20.0,
                "last_price_bin": "<=25",
                "user_past_event_count": 1,
                "user_past_purchase_count": 0,
                "user_past_cart_count": 0,
            },
            {
                "sample_id": "s3",
                "prefix_length": 3,
                "last_event_type": "cart",
                "session_elapsed_minutes": 10.0,
                "time_since_previous_event_minutes": 5.0,
                "hour": 0,
                "event_count_view": 2,
                "event_count_cart": 1,
                "event_count_remove_from_cart": 0,
                "unique_product_count": 2,
                "unique_category_count": 1,
                "last_price": 25.0,
                "last_price_bin": "25-50",
                "user_past_event_count": 2,
                "user_past_purchase_count": 0,
                "user_past_cart_count": 1,
            },
            {
                "sample_id": "s4",
                "prefix_length": 1,
                "last_event_type": "view",
                "session_elapsed_minutes": 0.0,
                "time_since_previous_event_minutes": None,
                "hour": 0,
                "event_count_view": 1,
                "event_count_cart": 0,
                "event_count_remove_from_cart": 0,
                "unique_product_count": 1,
                "unique_category_count": 1,
                "last_price": 100.0,
                "last_price_bin": "50-100",
                "user_past_event_count": 0,
                "user_past_purchase_count": 0,
                "user_past_cart_count": 0,
            },
        ]
    )


def test_build_baseline_dataset_joins_labels_without_metadata_as_features() -> None:
    dataset = build_baseline_dataset(_sample_index(), _tabular_features())

    assert dataset["sample_id"].tolist() == ["s1", "s2", "s3", "s4"]
    assert dataset["split"].tolist() == ["train", "train", "validation", "test"]
    assert dataset["label"].tolist() == [0, 1, 1, 0]
    assert "user_id" not in dataset.columns
    assert "cutoff_time" not in dataset.columns
    assert "minutes_until_purchase" not in dataset.columns


def test_preprocessor_fits_categories_and_imputation_on_train_only() -> None:
    dataset = build_baseline_dataset(_sample_index(), _tabular_features())
    train = dataset.loc[dataset["split"].eq("train")]
    validation = dataset.loc[dataset["split"].eq("validation")]
    preprocessor = TabularPreprocessor.fit(train)

    transformed = preprocessor.transform(validation)

    assert "last_event_type__cart" in transformed.columns
    assert "last_price_bin__25-50" not in transformed.columns
    assert transformed.isna().sum().sum() == 0
    assert preprocessor.numeric_fill_values["time_since_previous_event_minutes"] == 5.0


def test_compute_binary_metrics_returns_pr_roc_f1_and_top_k_metrics() -> None:
    metrics = compute_binary_metrics(
        y_true=[0, 1, 1, 0],
        y_score=[0.1, 0.9, 0.8, 0.2],
        threshold=0.5,
        top_k_fraction=0.5,
    )

    assert metrics["pr_auc"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["recall_at_k"] == 1.0
    assert metrics["precision_at_k"] == 1.0


def test_train_baseline_models_writes_metrics_and_importance(tmp_path: Path) -> None:
    dataset = build_baseline_dataset(_sample_index(), _tabular_features())
    result = train_baseline_models(
        dataset,
        policy=BaselineTrainingPolicy(
            class_imbalance_strategies=("none", "balanced"),
            logistic_max_iter=20,
            logistic_learning_rate=0.2,
            train_lightgbm=False,
        ),
    )

    assert set(result.metrics["model_name"]) == {"logistic_regression"}
    assert set(result.metrics["class_imbalance_strategy"]) == {"none", "balanced"}
    assert set(result.metrics["split"]) == {"train", "validation", "test"}
    assert "pr_auc" in result.metrics.columns
    assert result.metrics["pr_auc"].dtype.kind == "f"
    assert not result.feature_importance.empty
    assert result.feature_importance["feature_name"].notna().all()

    write_baseline_artifacts(result, tmp_path)

    assert (tmp_path / "model_metrics.csv").exists()
    assert (tmp_path / "baseline_feature_importance.csv").exists()
    assert (tmp_path / "baseline_report.md").exists()
