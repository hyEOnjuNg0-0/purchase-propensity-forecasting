from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402

from purchase_time_forecasting.streamlit_report import (  # noqa: E402
    BASELINE_BUILD_COMMAND,
    ArtifactRequirement,
    best_metric_summary,
    build_artifact_status,
    missing_artifact_commands,
    prepare_baseline_test_metrics,
    prepare_training_feature_dictionary,
    select_best_strategy,
    step8_artifact_requirements,
    top_feature_importance,
)


def _metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_name": "logistic_regression",
                "class_imbalance_strategy": "none",
                "split": "test",
                "sample_count": 100,
                "positive_count": 10,
                "pr_auc": 0.20,
                "roc_auc": 0.60,
                "f1": 0.10,
                "recall_at_k": 0.30,
                "precision_at_k": 0.15,
                "status": "evaluated",
            },
            {
                "model_name": "lightgbm",
                "class_imbalance_strategy": "none",
                "split": "validation",
                "sample_count": 100,
                "positive_count": 10,
                "pr_auc": 0.30,
                "roc_auc": 0.70,
                "f1": 0.20,
                "recall_at_k": 0.40,
                "precision_at_k": 0.20,
                "status": "evaluated",
            },
            {
                "model_name": "lightgbm",
                "class_imbalance_strategy": "balanced",
                "split": "validation",
                "sample_count": 100,
                "positive_count": 10,
                "pr_auc": 0.35,
                "roc_auc": 0.72,
                "f1": 0.21,
                "recall_at_k": 0.42,
                "precision_at_k": 0.21,
                "status": "evaluated",
            },
            {
                "model_name": "lightgbm",
                "class_imbalance_strategy": "balanced",
                "split": "test",
                "sample_count": 100,
                "positive_count": 10,
                "pr_auc": 0.32,
                "roc_auc": 0.71,
                "f1": 0.19,
                "recall_at_k": 0.41,
                "precision_at_k": 0.20,
                "status": "evaluated",
            },
        ]
    )


def test_step8_artifact_status_lists_missing_files_and_build_command(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "model_metrics.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    status = build_artifact_status(step8_artifact_requirements(reports_dir))

    assert status["exists"].tolist() == [True, False, False]
    assert set(missing_artifact_commands(status)) == {BASELINE_BUILD_COMMAND}


def test_missing_artifact_commands_deduplicates_commands(tmp_path: Path) -> None:
    status = build_artifact_status(
        [
            ArtifactRequirement("a", tmp_path / "a.csv", "run"),
            ArtifactRequirement("b", tmp_path / "b.csv", "run"),
        ]
    )

    assert missing_artifact_commands(status) == ["run"]


def test_prepare_baseline_test_metrics_filters_test_rows_and_sorts_by_pr_auc() -> None:
    prepared = prepare_baseline_test_metrics(_metrics())

    assert prepared["model_display"].tolist() == [
        "LightGBM (balanced)",
        "Logistic Regression (none)",
    ]
    assert prepared["pr_auc"].tolist() == [0.32, 0.20]
    assert set(prepared["split"].tolist() if "split" in prepared.columns else []) == set()


def test_best_metric_summary_uses_test_pr_auc() -> None:
    best = best_metric_summary(_metrics(), split="test", metric="pr_auc")

    assert best is not None
    assert best["model_display"] == "LightGBM (balanced)"
    assert best["pr_auc"] == 0.32


def test_select_best_strategy_uses_validation_metric() -> None:
    assert select_best_strategy(_metrics(), "lightgbm") == "balanced"


def test_top_feature_importance_filters_lightgbm_strategy_and_rank() -> None:
    importance = pd.DataFrame(
        [
            {
                "model_name": "lightgbm",
                "class_imbalance_strategy": "none",
                "feature_name": "prefix_length",
                "importance": 20,
                "importance_type": "gain",
                "rank": 1,
                "status": "estimated",
            },
            {
                "model_name": "lightgbm",
                "class_imbalance_strategy": "balanced",
                "feature_name": "last_event_type__cart",
                "importance": 30,
                "importance_type": "gain",
                "rank": 2,
                "status": "estimated",
            },
            {
                "model_name": "lightgbm",
                "class_imbalance_strategy": "balanced",
                "feature_name": "event_count_cart",
                "importance": 40,
                "importance_type": "gain",
                "rank": 1,
                "status": "estimated",
            },
        ]
    )

    top = top_feature_importance(importance, strategy="balanced", top_n=2)

    assert top["feature_name"].tolist() == [
        "event_count_cart",
        "last_event_type__cart",
    ]


def test_prepare_training_feature_dictionary_keeps_feature_name_and_description() -> None:
    feature_dictionary = pd.DataFrame(
        [
            {
                "feature_name": "sample_id",
                "feature_group": "identifier",
                "dtype": "string",
                "model_role": "key",
                "leakage_policy": "모델별 dataset 연결 key",
            },
            {
                "feature_name": "prefix_length",
                "feature_group": "tabular",
                "dtype": "integer",
                "model_role": "tabular_input",
                "leakage_policy": "기준 시점까지 관측된 prefix만 사용",
            },
            {
                "feature_name": "event_type_sequence",
                "feature_group": "sequence",
                "dtype": "string_sequence",
                "model_role": "sequence_input",
                "leakage_policy": "기준 시점까지의 prefix sequence만 사용",
            },
        ]
    )

    prepared = prepare_training_feature_dictionary(feature_dictionary)

    assert prepared.columns.tolist() == ["feature_name", "feature_description"]
    assert prepared["feature_name"].tolist() == [
        "prefix_length",
        "event_type_sequence",
    ]
    assert prepared["feature_description"].tolist() == [
        "기준 시점까지 관측된 세션 내 이벤트 개수",
        "기준 시점까지의 최근 행동 유형 sequence",
    ]
