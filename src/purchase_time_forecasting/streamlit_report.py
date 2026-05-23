"""Streamlit 보고서 표시용 artifact 정리 로직."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


BASELINE_BUILD_COMMAND = (
    ".\\scripts\\run_ptf.ps1 python scripts\\train_baselines.py"
)
FINAL_REPORT_BUILD_COMMAND = (
    ".\\scripts\\run_ptf.ps1 python scripts\\build_final_report.py"
)
PYTEST_COMMAND = ".\\scripts\\run_ptf.ps1 python -m pytest"
STREAMLIT_RUN_COMMAND = ".\\scripts\\run_ptf.ps1 streamlit run app/streamlit_app.py"
DOCKER_BUILD_COMMAND = "docker build -t purchase-propensity-report ."
DOCKER_RUN_COMMAND = (
    "docker run --rm -p 8501:8501 purchase-propensity-report"
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
METRIC_COLUMNS = (
    "sample_count",
    "positive_count",
    "pr_auc",
    "roc_auc",
    "f1",
    "recall_at_k",
    "precision_at_k",
)
MODEL_ORDER = {
    "logistic_regression": 0,
    "lightgbm": 1,
    "gru": 2,
}
SPLIT_ORDER = {
    "train": 0,
    "validation": 1,
    "test": 2,
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


def final_report_artifact_requirements(reports_dir: Path) -> tuple[ArtifactRequirement, ...]:
    """Step 10 최종 비교 화면에 필요한 artifact 목록을 반환한다."""

    return (
        ArtifactRequirement(
            name="final model comparison",
            path=reports_dir / "final_model_comparison.csv",
            build_command=FINAL_REPORT_BUILD_COMMAND,
        ),
        ArtifactRequirement(
            name="model interpretation summary",
            path=reports_dir / "model_interpretation_summary.md",
            build_command=FINAL_REPORT_BUILD_COMMAND,
        ),
    )


def build_reproducibility_commands() -> list[str]:
    """Step 12 마감 점검용 전체 실행 순서를 반환한다."""

    return [
        ".\\scripts\\run_ptf.ps1 python scripts\\profile_data.py",
        ".\\scripts\\run_ptf.ps1 python scripts\\validate_data_quality.py",
        ".\\scripts\\run_ptf.ps1 python scripts\\create_labels.py",
        ".\\scripts\\run_ptf.ps1 python scripts\\run_eda.py",
        ".\\scripts\\run_ptf.ps1 python scripts\\build_features.py",
        BASELINE_BUILD_COMMAND,
        ".\\scripts\\run_ptf.ps1 python scripts\\train_gru.py",
        FINAL_REPORT_BUILD_COMMAND,
        PYTEST_COMMAND,
        STREAMLIT_RUN_COMMAND,
        DOCKER_BUILD_COMMAND,
        DOCKER_RUN_COMMAND,
    ]


def build_step12_follow_up_items() -> list[str]:
    """마감 범위 밖의 고비용 개선 항목을 보고서 표시용으로 반환한다."""

    return [
        "SASRec: GRU 이후 self-attention 기반 sequence model로 확장한다.",
        "TiSASRec: 이벤트 간 시간 간격을 attention 구조에 직접 반영하는 고도화 후보로 둔다.",
        "SHAP: LightGBM feature importance를 보완하는 국소 설명 분석으로 추가 검토한다.",
        "attention/embedding 분석: sequence 표현을 행동 패턴 관점에서 해석하는 후속 분석으로 분리한다.",
    ]


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


def build_final_model_comparison(
    baseline_metrics: pd.DataFrame,
    gru_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """baseline과 GRU metric을 대표 전략 기준의 최종 비교표로 통합한다."""

    baseline = _normalize_metric_frame(
        baseline_metrics,
        source_artifact="model_metrics.csv",
    )
    gru = _normalize_metric_frame(
        gru_metrics,
        source_artifact="gru_model_metrics.csv",
    )
    combined = pd.concat([baseline, gru], ignore_index=True)
    if combined.empty:
        return _empty_final_model_comparison()

    selected_rows = []
    for model_name in combined["model_name"].dropna().astype(str).unique():
        strategy = select_best_strategy(
            combined,
            model_name=model_name,
            split="validation",
            metric="pr_auc",
        )
        if strategy is None:
            strategy = _select_available_strategy(combined, model_name)
        if strategy is None:
            continue
        selected = combined.loc[
            combined["model_name"].astype(str).eq(model_name)
            & combined["class_imbalance_strategy"].astype(str).eq(strategy)
        ].copy()
        selected_rows.append(selected)

    if not selected_rows:
        return _empty_final_model_comparison()

    comparison = pd.concat(selected_rows, ignore_index=True)
    comparison["model_display"] = comparison.apply(
        lambda row: (
            f"{_display_model_name(row['model_name'])} "
            f"({row['class_imbalance_strategy']})"
        ),
        axis=1,
    )
    comparison["selection_metric"] = "validation_pr_auc"
    comparison["sample_contract_status"] = _sample_contract_status_by_split(comparison)
    comparison["_split_order"] = comparison["split"].map(SPLIT_ORDER).fillna(99)
    comparison["_model_order"] = comparison["model_name"].map(MODEL_ORDER).fillna(99)

    output_columns = [
        "model_display",
        "model_name",
        "class_imbalance_strategy",
        "split",
        *METRIC_COLUMNS,
        "threshold",
        "top_k_fraction",
        "status",
        "selection_metric",
        "sample_contract_status",
        "source_artifact",
    ]
    return (
        comparison.sort_values(
            ["_split_order", "_model_order"],
            kind="mergesort",
        )
        .loc[:, [column for column in output_columns if column in comparison.columns]]
        .reset_index(drop=True)
    )


def prepare_final_test_metrics(comparison: pd.DataFrame) -> pd.DataFrame:
    """최종 모델 비교표 중 test split만 PR-AUC 기준으로 정렬한다."""

    if comparison.empty or "split" not in comparison.columns:
        return pd.DataFrame()
    test_metrics = comparison.loc[
        comparison["split"].astype(str).eq("test")
        & comparison.get("status", pd.Series(index=comparison.index, dtype=str))
        .astype(str)
        .eq("evaluated")
    ].copy()
    if test_metrics.empty:
        return test_metrics
    for column in METRIC_COLUMNS:
        if column in test_metrics.columns:
            test_metrics[column] = pd.to_numeric(test_metrics[column], errors="coerce")
    return (
        test_metrics.sort_values("pr_auc", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )


def build_model_interpretation_summary(
    comparison: pd.DataFrame,
    feature_importance: pd.DataFrame,
    top_n: int = 5,
) -> str:
    """최종 모델 비교와 LightGBM feature importance를 짧은 해석 markdown으로 변환한다."""

    lines = [
        "# Step 10 최종 모델 비교 및 해석",
        "",
        "## 모델 비교 결론",
        "",
    ]
    test_metrics = prepare_final_test_metrics(comparison)
    if test_metrics.empty:
        lines.append("- 표시 가능한 test split 모델 metric이 없다.")
    else:
        best = test_metrics.iloc[0]
        lines.append(
            f"- test PR-AUC 기준 최상위 모델은 {best['model_display']}이며 "
            f"PR-AUC는 {_format_metric(best['pr_auc'])}이다."
        )
        contract_status = sorted(
            test_metrics.get("sample_contract_status", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        if contract_status:
            lines.append(
                "- 동일 sample 비교 점검: "
                + ", ".join(contract_status)
                + "."
            )
        gru_note = _build_gru_comparison_note(test_metrics)
        if gru_note:
            lines.append(f"- {gru_note}")

    lines.extend(["", "## LightGBM 주요 feature 해석", ""])
    lightgbm_strategy = _selected_strategy_from_comparison(comparison, "lightgbm")
    top_features = top_feature_importance(
        feature_importance,
        model_name="lightgbm",
        strategy=lightgbm_strategy,
        top_n=top_n,
    )
    if top_features.empty:
        lines.append("- 표시 가능한 LightGBM feature importance가 없다.")
    else:
        if lightgbm_strategy is not None:
            lines.append(f"- 해석 대상 전략: `{lightgbm_strategy}`")
        for row in top_features.to_dict("records"):
            feature_name = str(row["feature_name"])
            lines.append(
                f"- `{feature_name}`: {_feature_behavior_interpretation(feature_name)}"
            )

    lines.extend(
        [
            "",
            "## 해석 범위",
            "",
            "- Step 10에서는 복잡한 SHAP, attention map, embedding clustering을 수행하지 않는다.",
            "- 해석은 동일 split metric 비교와 LightGBM feature importance 기반의 짧은 행동 관점 요약으로 제한한다.",
            "",
        ]
    )
    return "\n".join(lines)


def write_final_report_artifacts(
    comparison: pd.DataFrame,
    interpretation_markdown: str,
    reports_dir: Path,
) -> None:
    """Step 10 최종 비교 artifact를 저장한다."""

    output_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(
        output_dir / "final_model_comparison.csv",
        index=False,
        encoding="utf-8",
    )
    (output_dir / "model_interpretation_summary.md").write_text(
        interpretation_markdown,
        encoding="utf-8",
    )


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


def _format_metric(value: object) -> str:
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        return "N/A"
    return f"{float(numeric_value):.6f}"


def _normalize_metric_frame(
    metrics: pd.DataFrame,
    source_artifact: str,
) -> pd.DataFrame:
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
        return _empty_final_model_comparison()
    normalized = metrics.copy()
    for column in METRIC_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    for column in ["threshold", "top_k_fraction"]:
        if column not in normalized.columns:
            normalized[column] = pd.NA
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.loc[normalized["status"].astype(str).eq("evaluated")].copy()
    normalized["source_artifact"] = source_artifact
    return normalized


def _empty_final_model_comparison() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "model_display",
            "model_name",
            "class_imbalance_strategy",
            "split",
            *METRIC_COLUMNS,
            "threshold",
            "top_k_fraction",
            "status",
            "selection_metric",
            "sample_contract_status",
            "source_artifact",
        ]
    )


def _select_available_strategy(
    metrics: pd.DataFrame,
    model_name: str,
) -> str | None:
    candidates = metrics.loc[metrics["model_name"].astype(str).eq(model_name)].copy()
    if candidates.empty:
        return None
    for split in ("test", "train"):
        split_candidates = candidates.loc[candidates["split"].astype(str).eq(split)].copy()
        if split_candidates.empty:
            continue
        split_candidates["pr_auc"] = pd.to_numeric(
            split_candidates["pr_auc"],
            errors="coerce",
        )
        split_candidates = split_candidates.dropna(subset=["pr_auc"])
        if not split_candidates.empty:
            return str(
                split_candidates.loc[
                    split_candidates["pr_auc"].idxmax(),
                    "class_imbalance_strategy",
                ]
            )
    return str(candidates.iloc[0]["class_imbalance_strategy"])


def _sample_contract_status_by_split(comparison: pd.DataFrame) -> pd.Series:
    statuses = pd.Series(index=comparison.index, dtype=object)
    for split, split_frame in comparison.groupby("split", sort=False):
        pairs = split_frame.loc[:, ["sample_count", "positive_count"]].drop_duplicates()
        if len(split_frame) <= 1:
            status = "single_model_only"
        elif len(pairs) == 1:
            status = "matched_by_split_counts"
        else:
            status = "mismatch_by_split_counts"
        statuses.loc[split_frame.index] = status
    return statuses


def _selected_strategy_from_comparison(
    comparison: pd.DataFrame,
    model_name: str,
) -> str | None:
    if comparison.empty or {"model_name", "class_imbalance_strategy"} - set(comparison.columns):
        return None
    rows = comparison.loc[comparison["model_name"].astype(str).eq(model_name)]
    if rows.empty:
        return None
    return str(rows.iloc[0]["class_imbalance_strategy"])


def _build_gru_comparison_note(test_metrics: pd.DataFrame) -> str | None:
    if test_metrics.empty or "model_name" not in test_metrics.columns:
        return None
    gru_rows = test_metrics.loc[test_metrics["model_name"].astype(str).eq("gru")]
    baseline_rows = test_metrics.loc[~test_metrics["model_name"].astype(str).eq("gru")]
    if gru_rows.empty or baseline_rows.empty:
        return None
    gru_pr_auc = float(pd.to_numeric(gru_rows.iloc[0]["pr_auc"], errors="coerce"))
    best_baseline_pr_auc = float(
        pd.to_numeric(baseline_rows.iloc[0]["pr_auc"], errors="coerce")
    )
    if pd.isna(gru_pr_auc) or pd.isna(best_baseline_pr_auc):
        return None
    difference = gru_pr_auc - best_baseline_pr_auc
    if difference >= 0:
        return (
            "GRU는 최고 baseline보다 test PR-AUC가 "
            f"{difference:.6f} 높아 sequence 입력의 추가 효용을 보였다."
        )
    return (
        "GRU는 최고 baseline보다 test PR-AUC가 "
        f"{abs(difference):.6f} 낮다. 현재 설정에서는 짧은 prefix 요약 feature와 "
        "class imbalance 대응만으로도 강한 baseline이 형성됐고, GRU는 더 긴 학습이나 "
        "sequence 표현 튜닝이 필요할 수 있다."
    )


def _feature_behavior_interpretation(feature_name: str) -> str:
    base_name = feature_name.split("__", maxsplit=1)[0]
    custom = {
        "event_count_cart": "cart 누적 횟수는 구매 의도에 가까운 행동 강도를 나타낸다.",
        "last_event_type": "마지막 행동 유형은 기준 시점의 구매 의도 강도를 요약한다.",
        "last_event_type__cart": "마지막 행동이 cart인지 여부는 즉시 구매 가능성과 연결된다.",
        "prefix_length": "prefix 길이는 세션 내 탐색 깊이와 구매 전환 가능성을 함께 반영한다.",
        "unique_product_count": "고유 상품 수는 비교 탐색 범위와 관심 강도를 나타낸다.",
        "unique_category_count": "고유 카테고리 수는 탐색 범위의 넓이를 나타낸다.",
        "last_price": "최근 상품 가격은 구매 장벽과 상품군 차이를 함께 반영한다.",
        "hour": "시간대는 쇼핑 맥락과 구매 전환 패턴의 일중 변동을 반영한다.",
        "session_elapsed_minutes": "세션 경과 시간은 사용자가 탐색에 투자한 누적 시간을 나타낸다.",
        "time_since_previous_event_minutes": "직전 이벤트 이후 경과 시간은 최근 관심도가 식었는지 여부를 나타낸다.",
        "user_past_event_count": "과거 이벤트 수는 사용자의 전체 활동성과 재방문 가능성을 나타낸다.",
        "user_past_purchase_count": "과거 구매 횟수는 사용자의 반복 구매 성향을 나타낸다.",
        "user_past_cart_count": "과거 cart 횟수는 사용자의 장바구니 활용 성향을 나타낸다.",
    }
    if feature_name in custom:
        return custom[feature_name]
    if base_name in custom:
        return custom[base_name]
    return FEATURE_DESCRIPTIONS.get(base_name, "구매 전환과 관련된 tabular 행동 요약 feature다.")
