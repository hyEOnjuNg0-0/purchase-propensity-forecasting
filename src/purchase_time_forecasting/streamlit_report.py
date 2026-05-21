"""Streamlit 보고서 표시용 artifact 정리 로직."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


BASELINE_BUILD_COMMAND = (
    ".\\scripts\\run_ptf.ps1 python scripts\\train_baselines.py"
)
FEATURE_DESCRIPTIONS = {
    "prefix_length": "기준 시점까지 관측된 세션 내 이벤트 개수",
    "last_event_type": "기준 시점의 마지막 사용자 행동 유형",
    "session_elapsed_minutes": "세션 시작 이후 기준 시점까지 경과한 시간",
    "time_since_previous_event_minutes": "직전 이벤트 이후 기준 시점까지 경과한 시간",
    "hour": "기준 시점의 시간대",
    "event_count_view": "기준 시점까지 누적된 view 이벤트 수",
    "event_count_cart": "기준 시점까지 누적된 cart 이벤트 수",
    "event_count_remove_from_cart": "기준 시점까지 누적된 remove_from_cart 이벤트 수",
    "unique_product_count": "기준 시점까지 조회 또는 상호작용한 고유 상품 수",
    "unique_category_count": "기준 시점까지 조회 또는 상호작용한 고유 카테고리 수",
    "last_price": "기준 시점 마지막 이벤트의 상품 가격",
    "last_price_bin": "기준 시점 마지막 상품 가격의 구간화 값",
    "user_past_event_count": "기준 시점 이전 사용자의 과거 이벤트 수",
    "user_past_purchase_count": "기준 시점 이전 사용자의 과거 구매 이벤트 수",
    "user_past_cart_count": "기준 시점 이전 사용자의 과거 cart 이벤트 수",
    "event_type_sequence": "기준 시점까지의 최근 행동 유형 sequence",
    "product_id_sequence": "기준 시점까지의 최근 상품 ID sequence",
    "category_id_sequence": "기준 시점까지의 최근 카테고리 ID sequence",
    "price_bin_sequence": "기준 시점까지의 최근 가격 구간 sequence",
    "time_gap_minutes_sequence": "기준 시점까지의 최근 이벤트 간 시간 간격 sequence",
}


@dataclass(frozen=True)
class ArtifactRequirement:
    """Streamlit 보고서가 표시할 artifact 요구사항."""

    name: str
    path: Path
    build_command: str


def step8_artifact_requirements(reports_dir: Path) -> tuple[ArtifactRequirement, ...]:
    """Step 8 baseline 화면에 필요한 artifact 목록을 반환한다."""

    return (
        ArtifactRequirement(
            name="baseline metric table",
            path=reports_dir / "model_metrics.csv",
            build_command=BASELINE_BUILD_COMMAND,
        ),
        ArtifactRequirement(
            name="baseline feature importance",
            path=reports_dir / "baseline_feature_importance.csv",
            build_command=BASELINE_BUILD_COMMAND,
        ),
        ArtifactRequirement(
            name="baseline model status",
            path=reports_dir / "baseline_model_status.csv",
            build_command=BASELINE_BUILD_COMMAND,
        ),
    )


def build_artifact_status(
    requirements: Iterable[ArtifactRequirement],
) -> pd.DataFrame:
    """artifact 존재 여부를 Streamlit 표시용 표로 변환한다."""

    rows = []
    for requirement in requirements:
        rows.append(
            {
                "artifact": requirement.name,
                "path": str(requirement.path),
                "exists": requirement.path.exists(),
                "build_command": requirement.build_command,
            }
        )
    return pd.DataFrame(rows)


def missing_artifact_commands(status: pd.DataFrame) -> list[str]:
    """누락 artifact를 생성하기 위한 명령 목록을 중복 제거해 반환한다."""

    if status.empty or "exists" not in status.columns:
        return []
    missing = status.loc[~status["exists"].astype(bool), "build_command"].dropna()
    return list(dict.fromkeys(missing.astype(str).tolist()))


def prepare_baseline_test_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """baseline metric 중 test split 결과만 보고서 표시용으로 정리한다."""

    required = {
        "model_name",
        "class_imbalance_strategy",
        "split",
        "sample_count",
        "positive_count",
        "pr_auc",
        "roc_auc",
        "f1",
        "recall_at_k",
        "precision_at_k",
        "status",
    }
    if metrics.empty or not required.issubset(metrics.columns):
        return pd.DataFrame(columns=[*required, "model_display"])

    test_metrics = metrics.loc[
        metrics["split"].astype(str).eq("test")
        & metrics["status"].astype(str).eq("evaluated")
    ].copy()
    if test_metrics.empty:
        return test_metrics

    for column in [
        "sample_count",
        "positive_count",
        "pr_auc",
        "roc_auc",
        "f1",
        "recall_at_k",
        "precision_at_k",
    ]:
        test_metrics[column] = pd.to_numeric(test_metrics[column], errors="coerce")

    test_metrics["model_display"] = test_metrics.apply(
        lambda row: (
            f"{_display_model_name(row['model_name'])} "
            f"({row['class_imbalance_strategy']})"
        ),
        axis=1,
    )
    return (
        test_metrics.sort_values("pr_auc", ascending=False, kind="mergesort")
        .loc[
            :,
            [
                "model_display",
                "model_name",
                "class_imbalance_strategy",
                "sample_count",
                "positive_count",
                "pr_auc",
                "roc_auc",
                "f1",
                "recall_at_k",
                "precision_at_k",
                "status",
            ],
        ]
        .reset_index(drop=True)
    )


def best_metric_summary(
    metrics: pd.DataFrame,
    split: str = "test",
    metric: str = "pr_auc",
) -> dict[str, object] | None:
    """지정 split에서 가장 높은 metric을 가진 baseline 결과를 반환한다."""

    if metrics.empty or metric not in metrics.columns:
        return None
    candidates = metrics.loc[
        metrics["split"].astype(str).eq(split)
        & metrics.get("status", pd.Series(index=metrics.index, dtype=str))
        .astype(str)
        .eq("evaluated")
    ].copy()
    if candidates.empty:
        return None
    candidates[metric] = pd.to_numeric(candidates[metric], errors="coerce")
    candidates = candidates.dropna(subset=[metric])
    if candidates.empty:
        return None
    row = candidates.loc[candidates[metric].idxmax()].to_dict()
    row["model_display"] = (
        f"{_display_model_name(row['model_name'])} "
        f"({row['class_imbalance_strategy']})"
    )
    return row


def select_best_strategy(
    metrics: pd.DataFrame,
    model_name: str,
    split: str = "validation",
    metric: str = "pr_auc",
) -> str | None:
    """특정 모델의 validation 기준 최적 class imbalance 전략을 선택한다."""

    if metrics.empty or metric not in metrics.columns:
        return None
    required = {"model_name", "class_imbalance_strategy", "split"}
    if not required.issubset(metrics.columns):
        return None
    candidates = metrics.loc[
        metrics["model_name"].astype(str).eq(model_name)
        & metrics["split"].astype(str).eq(split)
    ].copy()
    if "status" in candidates.columns:
        candidates = candidates.loc[candidates["status"].astype(str).eq("evaluated")]
    candidates[metric] = pd.to_numeric(candidates[metric], errors="coerce")
    candidates = candidates.dropna(subset=[metric])
    if candidates.empty:
        return None
    return str(candidates.loc[candidates[metric].idxmax(), "class_imbalance_strategy"])


def top_feature_importance(
    feature_importance: pd.DataFrame,
    model_name: str = "lightgbm",
    strategy: str | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """모델별 feature importance 상위 항목을 정리한다."""

    required = {"model_name", "feature_name", "importance"}
    if feature_importance.empty or not required.issubset(feature_importance.columns):
        return pd.DataFrame(columns=["feature_name", "importance", "rank"])

    filtered = feature_importance.loc[
        feature_importance["model_name"].astype(str).eq(model_name)
    ].copy()
    if strategy is not None and "class_imbalance_strategy" in filtered.columns:
        filtered = filtered.loc[
            filtered["class_imbalance_strategy"].astype(str).eq(strategy)
        ]
    if "status" in filtered.columns:
        filtered = filtered.loc[filtered["status"].astype(str).eq("estimated")]
    if filtered.empty:
        return filtered

    filtered["importance"] = pd.to_numeric(filtered["importance"], errors="coerce")
    filtered = filtered.dropna(subset=["importance"])
    sort_columns = ["importance"]
    ascending = [False]
    if "rank" in filtered.columns:
        filtered["rank"] = pd.to_numeric(filtered["rank"], errors="coerce")
        sort_columns = ["rank", "importance"]
        ascending = [True, False]

    columns = [
        column
        for column in [
            "feature_name",
            "importance",
            "importance_type",
            "rank",
            "class_imbalance_strategy",
        ]
        if column in filtered.columns
    ]
    return (
        filtered.sort_values(sort_columns, ascending=ascending, kind="mergesort")
        .loc[:, columns]
        .head(top_n)
        .reset_index(drop=True)
    )


def prepare_training_feature_dictionary(feature_dictionary: pd.DataFrame) -> pd.DataFrame:
    """학습 입력 feature 이름과 설명만 표시용으로 정리한다."""

    required = {"feature_name", "model_role"}
    output_columns = ["feature_name", "feature_description"]
    if feature_dictionary.empty or not required.issubset(feature_dictionary.columns):
        return pd.DataFrame(columns=output_columns)

    training_roles = {"tabular_input", "sequence_input"}
    filtered = feature_dictionary.loc[
        feature_dictionary["model_role"].astype(str).isin(training_roles),
        ["feature_name", "model_role"],
    ].copy()
    if filtered.empty:
        return pd.DataFrame(columns=output_columns)

    filtered["feature_description"] = filtered["feature_name"].map(
        FEATURE_DESCRIPTIONS
    )
    missing_description = filtered["feature_description"].isna()
    filtered.loc[missing_description, "feature_description"] = filtered.loc[
        missing_description, "feature_name"
    ].map(lambda name: f"`{name}` 학습 입력 feature")

    role_order = {"tabular_input": 0, "sequence_input": 1}
    filtered["_role_order"] = filtered["model_role"].map(role_order).fillna(99)
    filtered["_original_order"] = range(len(filtered))
    return (
        filtered.sort_values(["_role_order", "_original_order"], kind="mergesort")
        .loc[:, output_columns]
        .reset_index(drop=True)
    )


def _display_model_name(model_name: object) -> str:
    mapping = {
        "logistic_regression": "Logistic Regression",
        "lightgbm": "LightGBM",
        "gru": "GRU",
    }
    return mapping.get(str(model_name), str(model_name))
