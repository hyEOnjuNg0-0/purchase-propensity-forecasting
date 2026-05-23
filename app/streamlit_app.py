from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from purchase_conversion_prediction.streamlit_report import (  # noqa: E402
    BASELINE_BUILD_COMMAND,
    FINAL_REPORT_BUILD_COMMAND,
    best_metric_summary,
    build_step12_follow_up_items,
    prepare_baseline_test_metrics,
    prepare_final_test_metrics,
    prepare_training_feature_dictionary,
    select_best_strategy,
    top_feature_importance,
)


ARTIFACTS_DIR = REPO_ROOT / "artifacts"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
RAW_DATA_PREVIEW_PATH = REPORTS_DIR / "raw_data_preview.csv"

BASELINE_METRIC_COLUMNS = {
    "model_display": "모델",
    "sample_count": "test sample",
    "positive_count": "positive",
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "f1": "F1",
    "recall_at_k": "Recall@K",
    "precision_at_k": "Precision@K",
}
GRU_BUILD_COMMAND = ".\\scripts\\run_ptf.ps1 python scripts\\train_gru.py"
GRU_METRIC_COLUMNS = {
    "split": "split",
    "sample_count": "sample",
    "positive_count": "positive",
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "f1": "F1",
    "recall_at_k": "Recall@K",
    "precision_at_k": "Precision@K",
}
FINAL_METRIC_COLUMNS = {
    "model_display": "모델",
    "sample_count": "test sample",
    "positive_count": "positive",
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "f1": "F1",
    "recall_at_k": "Recall@K",
    "precision_at_k": "Precision@K",
    "sample_contract_status": "sample 점검",
}
@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame | None:
    artifact_path = Path(path)
    if not artifact_path.exists():
        return None
    return pd.read_csv(artifact_path)


@st.cache_data(show_spinner=False)
def read_markdown(path: str) -> str | None:
    artifact_path = Path(path)
    if not artifact_path.exists():
        return None
    return artifact_path.read_text(encoding="utf-8")


def render_missing_artifact(path: Path, command: str) -> None:
    st.warning(f"`{_relative_path(path)}` artifact 없음")
    st.code(command, language="powershell")


def render_navigation() -> str:
    st.sidebar.title("Contents")
    sections = [
        "1. Overview",
        "2. Data Quality",
        "3. Labeling",
        "4. EDA",
        "5. Features",
        "6. Baseline Results",
        "7. GRU Results",
        "8. Integrated Comparison",
        "9. Limitations",
    ]
    selected = st.sidebar.radio(
        "Report Sections",
        sections,
        label_visibility="collapsed",
    )
    return selected


def render_section_title(title: str, description: str | None = None) -> None:
    st.header(title)
    if description:
        st.caption(description)


def render_overview() -> None:
    render_section_title("Overview")

    st.subheader("목표")
    st.markdown(
        """
        `2019-Oct.csv`의 세션 내 행동 이력을 이용해 기준 시점 이후
        **30분 내 purchase 발생 확률**을 예측한다. 
        
        운영 서비스가 아니라,
        데이터 검증, 라벨링, feature 생성, baseline 모델링 과정을 한눈에 확인하는
        Streamlit 보고서 결과물을 목표로 한다.
        """
    )

    st.divider()

    st.subheader("사용 데이터")
    st.markdown(
        """
        - 데이터셋 출처 : Kaggle `eCommerce behavior data from multi category store`
        - 분석 데이터 : `2019-Oct.csv` 전체
        - 학습 데이터 : `2019-Oct.csv`의 2019-10-05까지
        """
    )
    raw_preview = read_csv(str(RAW_DATA_PREVIEW_PATH))
    st.subheader("원천 데이터 예시")
    if raw_preview is None:
        render_missing_artifact(
            RAW_DATA_PREVIEW_PATH,
            ".\\scripts\\run_ptf.ps1 python scripts\\profile_data.py",
        )
    else:
        st.caption("`2019-Oct.csv` 상위 15행")
        st.dataframe(raw_preview.head(15), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("핵심 흐름")
    flow = pd.DataFrame(
        [
            {   
                "": "1",
                "단계": "문제 정의",
                "핵심": "세션 prefix 기준 향후 30분 내 구매 발생 예측",
            },
            {
                "": "2",
                "단계": "데이터 검증",
                "핵심": "스키마, 결측, 가격 이상치, 세션 시간 정합성 확인",
            },
            {
                "": "3",
                "단계": "라벨링",
                "핵심": "첫 purchase 기준 30분 window label 생성",
            },
            {
                "": "4",
                "단계": "EDA",
                "핵심": "세션 길이, 행동 패턴, 가격대, 시간대별 구매율 확인",
            },
            {
                "": "5",
                "단계": "Feature Engineering",
                "핵심": "공통 sample index와 tabular/sequence feature artifact 생성",
            },
            {
                "": "6",
                "단계": "Baseline 모델 평가",
                "핵심": "Logistic Regression, LightGBM test 성능 비교",
            },
            {
                "": "7",
                "단계": "Sequence",
                "핵심": "GRU 학습 artifact로 행동 순서 정보의 추가 효용 확인",
            },
        ]
    )
    st.dataframe(flow, use_container_width=True, hide_index=True)


def render_data_quality() -> None:
    render_section_title(
        "Data Quality",
        "모델링 전에 원천 데이터가 예측 문제로 사용할 수 있는지 점검",
    )
    summary = read_csv(str(REPORTS_DIR / "data_quality_summary.csv"))
    if summary is None:
        render_missing_artifact(
            REPORTS_DIR / "data_quality_summary.csv",
            ".\\scripts\\run_ptf.ps1 python scripts\\validate_data_quality.py",
        )
        return

    status_counts = summary["status"].value_counts().rename_axis("status").reset_index(
        name="count"
    )
    st.subheader("검증 상태 요약")
    st.dataframe(status_counts, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("검증 상세")
    detail_summary = summary.drop(columns=["next_action"], errors="ignore")
    st.dataframe(detail_summary, use_container_width=True, hide_index=True)


def render_labeling() -> None:
    render_section_title(
        "Labeling",
        "30분 purchase label 정책과 실제 label 분포 확인",
    )
    distribution = read_csv(str(REPORTS_DIR / "label_distribution.csv"))
    policy = read_csv(str(REPORTS_DIR / "labeling_policy.csv"))

    st.subheader("라벨 분포")
    if distribution is None:
        render_missing_artifact(
            REPORTS_DIR / "label_distribution.csv",
            ".\\scripts\\run_ptf.ps1 python scripts\\create_labels.py",
        )
    else:
        st.dataframe(distribution, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("라벨링 정책")
    if policy is None:
        render_missing_artifact(
            REPORTS_DIR / "labeling_policy.csv",
            ".\\scripts\\run_ptf.ps1 python scripts\\create_labels.py",
        )
    else:
        st.dataframe(policy, use_container_width=True, hide_index=True)


def render_eda() -> None:
    render_section_title(
        "EDA",
        "구매율에 영향을 줄 수 있는 세션 길이, 행동 패턴, 가격대, 시간대 패턴 분석",
    )
    tabs = st.tabs(["세션 길이", "행동 패턴", "가격대", "시간대"])
    eda_specs = [
        (
            tabs[0],
            "세션 길이별 구매율",
            REPORTS_DIR / "eda_session_length_purchase_rate.csv",
            "session_length_band",
            "purchase_rate",
        ),
        (
            tabs[1],
            "초기 행동 패턴별 구매율",
            REPORTS_DIR / "eda_sequence_pattern_purchase_rate.csv",
            "sequence_pattern",
            "purchase_rate",
        ),
        (
            tabs[2],
            "가격대별 positive 비율",
            REPORTS_DIR / "eda_price_band_purchase_rate.csv",
            "price_band",
            "positive_rate",
        ),
        (
            tabs[3],
            "시간대별 구매율",
            REPORTS_DIR / "eda_hourly_purchase_rate.csv",
            "hour",
            "positive_rate",
        ),
    ]

    for tab, title, path, index_column, value_column in eda_specs:
        with tab:
            frame = read_csv(str(path))
            st.subheader(title)
            if frame is None:
                render_missing_artifact(
                    path,
                    ".\\scripts\\run_ptf.ps1 python scripts\\run_eda.py",
                )
                continue
            if index_column in frame.columns and value_column in frame.columns:
                chart_frame = frame.loc[:, [index_column, value_column]].copy()
                chart_frame[value_column] = pd.to_numeric(
                    chart_frame[value_column], errors="coerce"
                )
                st.bar_chart(chart_frame.set_index(index_column), height=260)
            st.dataframe(frame, use_container_width=True, hide_index=True)
            st.markdown(_build_eda_plain_interpretation(frame, index_column, value_column))


def render_features() -> None:
    render_section_title(
        "Features",
        "공통 sample index와 baseline/sequence 입력 artifact의 split 및 feature 구성 확인",
    )
    split_summary = read_csv(str(REPORTS_DIR / "feature_split_summary.csv"))
    feature_dictionary = read_csv(str(REPORTS_DIR / "feature_dictionary.csv"))

    st.subheader("Split 분포")
    if split_summary is None:
        render_missing_artifact(
            REPORTS_DIR / "feature_split_summary.csv",
            ".\\scripts\\run_ptf.ps1 python scripts\\build_features.py",
        )
    else:
        st.dataframe(split_summary, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Feature Dictionary")
    if feature_dictionary is None:
        render_missing_artifact(
            REPORTS_DIR / "feature_dictionary.csv",
            ".\\scripts\\run_ptf.ps1 python scripts\\build_features.py",
        )
    else:
        training_feature_dictionary = prepare_training_feature_dictionary(
            feature_dictionary
        )
        st.dataframe(
            training_feature_dictionary,
            use_container_width=True,
            hide_index=True,
        )


def render_baseline_results() -> None:
    render_section_title(
        "Baseline Results",
        "Logistic Regression과 LightGBM의 test 성능과 주요 feature 비교",
    )
    metrics = read_csv(str(REPORTS_DIR / "model_metrics.csv"))
    importance = read_csv(str(REPORTS_DIR / "baseline_feature_importance.csv"))
    status = read_csv(str(REPORTS_DIR / "baseline_model_status.csv"))

    if metrics is None:
        render_missing_artifact(REPORTS_DIR / "model_metrics.csv", BASELINE_BUILD_COMMAND)
        return

    test_metrics = prepare_baseline_test_metrics(metrics)
    best = best_metric_summary(metrics, split="test", metric="pr_auc")
    if best is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Best test model", str(best["model_display"]))
        col2.metric("Best test PR-AUC", f"{float(best['pr_auc']):.4f}")
        col3.metric("Best test ROC-AUC", f"{float(best['roc_auc']):.4f}")

    st.divider()

    st.subheader("Test 성능 비교")
    if test_metrics.empty:
        st.info("표시 가능한 test split baseline metric 없음")
    else:
        display_metrics = test_metrics.loc[:, list(BASELINE_METRIC_COLUMNS)].rename(
            columns=BASELINE_METRIC_COLUMNS
        )
        st.dataframe(display_metrics, use_container_width=True, hide_index=True)
        chart_data = test_metrics.loc[:, ["model_display", "pr_auc"]].set_index(
            "model_display"
        )
        st.bar_chart(chart_data, height=260)

    st.divider()

    st.subheader("모델 상태")
    if status is None:
        render_missing_artifact(
            REPORTS_DIR / "baseline_model_status.csv",
            BASELINE_BUILD_COMMAND,
        )
    else:
        st.dataframe(status, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("LightGBM 주요 feature")
    if importance is None:
        render_missing_artifact(
            REPORTS_DIR / "baseline_feature_importance.csv",
            BASELINE_BUILD_COMMAND,
        )
    else:
        strategy = select_best_strategy(metrics, model_name="lightgbm")
        top_features = top_feature_importance(
            importance,
            model_name="lightgbm",
            strategy=strategy,
            top_n=15,
        )
        if top_features.empty:
            st.info("표시 가능한 LightGBM feature importance 없음")
        else:
            if strategy is not None:
                st.caption(f"validation PR-AUC 기준 선택 전략: `{strategy}`")
            st.bar_chart(
                top_features.loc[:, ["feature_name", "importance"]].set_index(
                    "feature_name"
                ),
                height=320,
            )
            display_features = top_features.drop(
                columns=["importance_type", "class_imbalance_strategy"],
                errors="ignore",
            )
            ordered_columns = [
                column
                for column in ["rank", "feature_name", "importance"]
                if column in display_features.columns
            ]
            remaining_columns = [
                column
                for column in display_features.columns
                if column not in ordered_columns
            ]
            st.dataframe(
                display_features.loc[:, [*ordered_columns, *remaining_columns]],
                use_container_width=True,
                hide_index=True,
            )


def render_gru_results() -> None:
    render_section_title(
        "GRU Results",
        "행동 순서 정보를 입력으로 사용한 sequence model 단독 결과",
    )
    metrics = read_csv(str(REPORTS_DIR / "gru_model_metrics.csv"))
    history = read_csv(str(REPORTS_DIR / "gru_training_history.csv"))
    status = read_csv(str(REPORTS_DIR / "gru_model_status.csv"))

    if metrics is None:
        render_missing_artifact(REPORTS_DIR / "gru_model_metrics.csv", GRU_BUILD_COMMAND)
        return

    prepared_metrics = _prepare_gru_metrics(metrics)
    test_metrics = prepared_metrics.loc[prepared_metrics["split"].astype(str).eq("test")]
    if test_metrics.empty:
        st.info("표시 가능한 GRU test metric 없음")
    else:
        test_row = test_metrics.iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Test PR-AUC", _format_float(test_row["pr_auc"]))
        col2.metric("Test ROC-AUC", _format_float(test_row["roc_auc"]))
        col3.metric("Test Recall@K", _format_float(test_row["recall_at_k"]))

        st.markdown(
            _build_gru_plain_interpretation(
                pr_auc=test_row["pr_auc"],
                roc_auc=test_row["roc_auc"],
                recall_at_k=test_row["recall_at_k"],
                precision_at_k=test_row["precision_at_k"],
            )
        )

    st.divider()
    st.subheader("Split별 GRU 성능")
    if prepared_metrics.empty:
        st.info("표시 가능한 GRU metric 없음")
    else:
        display_columns = [
            column for column in GRU_METRIC_COLUMNS if column in prepared_metrics.columns
        ]
        st.dataframe(
            prepared_metrics.loc[:, display_columns].rename(columns=GRU_METRIC_COLUMNS),
            use_container_width=True,
            hide_index=True,
        )
        chart_columns = [
            column for column in ["split", "pr_auc", "roc_auc"] if column in prepared_metrics
        ]
        if len(chart_columns) == 3:
            st.bar_chart(
                prepared_metrics.loc[:, chart_columns].set_index("split"),
                height=260,
            )

    st.divider()
    st.subheader("학습 추이")
    if history is None:
        render_missing_artifact(
            REPORTS_DIR / "gru_training_history.csv",
            GRU_BUILD_COMMAND,
        )
    elif history.empty or "epoch" not in history.columns:
        st.info("표시 가능한 GRU 학습 이력 없음")
    else:
        prepared_history = history.copy()
        prepared_history["epoch"] = pd.to_numeric(
            prepared_history["epoch"],
            errors="coerce",
        )
        for column in ["train_loss", "validation_pr_auc", "validation_roc_auc"]:
            if column in prepared_history.columns:
                prepared_history[column] = pd.to_numeric(
                    prepared_history[column],
                    errors="coerce",
                )
        metric_columns = [
            column
            for column in ["validation_pr_auc", "validation_roc_auc"]
            if column in prepared_history.columns
        ]
        if metric_columns:
            chart_columns = st.columns(len(metric_columns))
            for chart_column, metric_column in zip(chart_columns, metric_columns):
                with chart_column:
                    _render_training_metric_chart(
                        prepared_history,
                        metric_column=metric_column,
                        metric_label=_metric_label(metric_column),
                    )
        st.markdown(_build_gru_training_interpretation(prepared_history))
        st.dataframe(prepared_history, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("학습 설정")
    if status is None:
        render_missing_artifact(
            REPORTS_DIR / "gru_model_status.csv",
            GRU_BUILD_COMMAND,
        )
    else:
        st.dataframe(status, use_container_width=True, hide_index=True)


def render_integrated_comparison() -> None:
    render_section_title(
        "Integrated Comparison",
        "Logistic Regression, LightGBM, GRU를 같은 test sample 기준으로 비교",
    )
    comparison = read_csv(str(REPORTS_DIR / "final_model_comparison.csv"))
    interpretation = read_markdown(
        str(REPORTS_DIR / "model_interpretation_summary.md")
    )
    if comparison is None:
        render_missing_artifact(
            REPORTS_DIR / "final_model_comparison.csv",
            FINAL_REPORT_BUILD_COMMAND,
        )
        return

    test_metrics = prepare_final_test_metrics(comparison)
    if test_metrics.empty:
        st.info("표시 가능한 test split 최종 모델 비교 metric 없음")
    else:
        best = test_metrics.iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("최종 선택 모델", str(best["model_display"]))
        col2.metric("Test PR-AUC", f"{float(best['pr_auc']):.4f}")
        col3.metric("Test ROC-AUC", f"{float(best['roc_auc']):.4f}")

        st.markdown(_build_final_plain_interpretation(test_metrics))

        st.divider()
        st.subheader("Test 성능 비교")
        display_columns = [
            column for column in FINAL_METRIC_COLUMNS if column in test_metrics.columns
        ]
        st.dataframe(
            test_metrics.loc[:, display_columns].rename(columns=FINAL_METRIC_COLUMNS),
            use_container_width=True,
            hide_index=True,
        )
        st.bar_chart(
            test_metrics.loc[:, ["model_display", "pr_auc"]].set_index("model_display"),
            height=260,
        )

    st.divider()
    st.subheader("Train/Validation/Test 전체 비교")
    st.dataframe(comparison, use_container_width=True, hide_index=True)


def render_limitations() -> None:
    render_section_title(
        "Limitations",
        "이번 마감 범위와 후속 개선안을 분리해 해석 범위를 명확히 표시",
    )
    st.subheader("현재 범위")
    st.markdown(
        """
        - 분석 대상은 `2019-Oct.csv` 단일 월이며, 운영 서비스나 실시간 inference는 포함하지 않는다.
        - 최종 비교는 Logistic Regression, LightGBM, GRU를 동일 split과 metric으로 비교하는 데 집중한다.
        - GRU가 baseline보다 낮거나 비슷하더라도 sequence 입력 실험 결과와 한계를 함께 제시한다.
        """
    )

    st.divider()
    st.subheader("후속 개선안")
    st.markdown("\n".join(f"- {item}" for item in build_step12_follow_up_items()))


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _format_float(value: object) -> str:
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        return "N/A"
    return f"{float(numeric_value):.4f}"


def _format_percent(value: object) -> str:
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        return "N/A"
    return f"{float(numeric_value):.2%}"


def _build_eda_plain_interpretation(
    frame: pd.DataFrame,
    index_column: str,
    value_column: str,
) -> str:
    if frame.empty or {index_column, value_column} - set(frame.columns):
        return "- 해석: 표시 가능한 EDA 요약 없음"

    working = frame.copy()
    working[value_column] = pd.to_numeric(working[value_column], errors="coerce")

    count_column = _first_existing_column(working, ["sample_count", "session_count"])
    positive_column = _first_existing_column(
        working,
        ["positive_count", "purchase_session_count"],
    )
    if count_column is not None:
        working[count_column] = pd.to_numeric(working[count_column], errors="coerce")
        working = working.loc[working[count_column].fillna(0).gt(0)]

    working = working.dropna(subset=[value_column])
    if working.empty:
        return "- 해석: 유효한 EDA rate 없음"

    highest = working.loc[working[value_column].idxmax()]
    lowest = working.loc[working[value_column].idxmin()]
    spread = float(highest[value_column]) - float(lowest[value_column])

    lines = [
        (
            f"- 최고 구간: `{highest[index_column]}` "
            f"({_format_percent(highest[value_column])})"
        ),
        (
            f"- 최저 구간: `{lowest[index_column]}` "
            f"({_format_percent(lowest[value_column])})"
        ),
        f"- 구간 차이: `{spread * 100:.2f}%p`",
    ]

    if count_column is not None and positive_column is not None:
        working[positive_column] = pd.to_numeric(
            working[positive_column],
            errors="coerce",
        )
        total_count = working[count_column].sum()
        total_positive = working[positive_column].sum()
        if total_count > 0:
            lines.append(
                f"- 전체 기준 rate: {_format_percent(total_positive / total_count)}"
            )

    lines.append(_eda_context_sentence(index_column, str(highest[index_column])))
    return "\n".join(lines)


def _first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _eda_context_sentence(index_column: str, highest_label: str) -> str:
    if index_column == "session_length_band":
        return "- 의미: 세션 길이에 따라 구매율이 달라져 탐색 깊이가 구매 예측 feature로 유효"
    if index_column == "sequence_pattern":
        if "cart" in highest_label:
            return "- 의미: `cart` 포함 행동 패턴이 강한 구매 의도 신호"
        return "- 의미: 초기 행동 순서만으로도 구매율 차이 발생"
    if index_column == "price_band":
        return "- 의미: 가격대별 positive 비율 차이 확인. 가격 feature의 보조 설명력 근거"
    if index_column == "hour":
        return "- 의미: 시간대별 positive 비율 변동 확인. hour feature 사용 근거"
    return "- 의미: 구간별 rate 차이를 통해 feature 후보의 설명력 확인"


def _prepare_gru_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {
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
        return pd.DataFrame(columns=list(required))

    prepared = metrics.loc[metrics["status"].astype(str).eq("evaluated")].copy()
    for column in [
        "sample_count",
        "positive_count",
        "pr_auc",
        "roc_auc",
        "f1",
        "recall_at_k",
        "precision_at_k",
    ]:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    split_order = {"train": 0, "validation": 1, "test": 2}
    prepared["_split_order"] = prepared["split"].map(split_order).fillna(99)
    return (
        prepared.sort_values("_split_order", kind="mergesort")
        .drop(columns=["_split_order"])
        .reset_index(drop=True)
    )


def _render_training_metric_chart(
    history: pd.DataFrame,
    metric_column: str,
    metric_label: str,
) -> None:
    if {"epoch", metric_column} - set(history.columns):
        return

    chart_frame = history.loc[:, ["epoch", metric_column]].dropna().copy()
    if chart_frame.empty:
        return

    chart_frame["epoch"] = pd.to_numeric(chart_frame["epoch"], errors="coerce")
    chart_frame[metric_column] = pd.to_numeric(
        chart_frame[metric_column],
        errors="coerce",
    )
    chart_frame = chart_frame.dropna()
    if chart_frame.empty:
        return

    lower_bound, upper_bound = _metric_axis_domain(chart_frame[metric_column])
    chart = (
        alt.Chart(chart_frame)
        .mark_line(point=True)
        .encode(
            x=alt.X("epoch:O", title="epoch"),
            y=alt.Y(
                f"{metric_column}:Q",
                title=metric_label,
                scale=alt.Scale(domain=[lower_bound, upper_bound], zero=False),
            ),
            tooltip=[
                alt.Tooltip("epoch:O", title="epoch"),
                alt.Tooltip(f"{metric_column}:Q", title=metric_label, format=".6f"),
            ],
        )
        .properties(height=240)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(f"y-axis: {_format_float(lower_bound)} - {_format_float(upper_bound)}")


def _metric_axis_domain(values: pd.Series) -> tuple[float, float]:
    numeric_values = pd.to_numeric(values, errors="coerce").dropna()
    if numeric_values.empty:
        return 0.0, 1.0

    minimum = float(numeric_values.min())
    maximum = float(numeric_values.max())
    if minimum == maximum:
        padding = max(abs(minimum) * 0.05, 0.01)
    else:
        padding = (maximum - minimum) * 0.2
    lower_bound = max(0.0, minimum - padding)
    upper_bound = min(1.0, maximum + padding)
    if lower_bound == upper_bound:
        upper_bound = min(1.0, lower_bound + 0.01)
    return lower_bound, upper_bound


def _metric_label(metric_column: str) -> str:
    labels = {
        "validation_pr_auc": "Validation PR-AUC",
        "validation_roc_auc": "Validation ROC-AUC",
    }
    return labels.get(metric_column, metric_column)


def _build_gru_plain_interpretation(
    pr_auc: object,
    roc_auc: object,
    recall_at_k: object,
    precision_at_k: object,
) -> str:
    return "\n".join(
        [
            "- GRU: `view -> cart -> view`처럼 기준 시점까지의 행동 순서를 직접 읽는 sequence model",
            f"- Test PR-AUC `{_format_float(pr_auc)}`: 구매 sample이 적은 조건에서 구매 가능성이 큰 sample을 앞순위에 배치하는 힘",
            f"- Test ROC-AUC `{_format_float(roc_auc)}`: 구매/비구매를 전반적으로 구분하는 힘",
            f"- 상위 10% 후보 기준: 실제 구매자의 `{_format_float(recall_at_k)}` 포착, 후보 내부 구매 비율 `{_format_float(precision_at_k)}`",
        ]
    )


def _build_gru_training_interpretation(history: pd.DataFrame) -> str:
    if history.empty or "epoch" not in history.columns:
        return "- Epoch 해석: 표시 가능한 학습 이력 없음"

    prepared = history.copy()
    prepared["epoch"] = pd.to_numeric(prepared["epoch"], errors="coerce")
    if "validation_pr_auc" in prepared.columns:
        prepared["validation_pr_auc"] = pd.to_numeric(
            prepared["validation_pr_auc"],
            errors="coerce",
        )
    if "train_loss" in prepared.columns:
        prepared["train_loss"] = pd.to_numeric(prepared["train_loss"], errors="coerce")

    valid_epochs = prepared.dropna(subset=["epoch"])
    if valid_epochs.empty:
        return "- Epoch 해석: epoch 값 확인 불가"

    epoch_count = int(valid_epochs["epoch"].nunique())
    lines = [
        f"- Epoch `{epoch_count}` 설정: 최종 하이퍼파라미터 탐색이 아니라 baseline 대비 sequence 입력 효용을 확인하는 1차 비교용 설정",
        "- 전체 sequence artifact 기반 학습 비용이 크므로, 짧은 epoch로 먼저 validation 경향과 과적합 신호 확인",
    ]

    valid_pr_auc = valid_epochs.dropna(subset=["validation_pr_auc"])
    if not valid_pr_auc.empty:
        best_row = valid_pr_auc.loc[valid_pr_auc["validation_pr_auc"].idxmax()]
        first_row = valid_pr_auc.sort_values("epoch", kind="mergesort").iloc[0]
        last_row = valid_pr_auc.sort_values("epoch", kind="mergesort").iloc[-1]
        lines.append(
            f"- Validation PR-AUC 최고점: epoch `{int(best_row['epoch'])}`, `{_format_float(best_row['validation_pr_auc'])}`"
        )
        if float(last_row["validation_pr_auc"]) < float(best_row["validation_pr_auc"]):
            lines.append(
                f"- 학습 경향: 마지막 epoch PR-AUC `{_format_float(last_row['validation_pr_auc'])}`로 최고점 대비 하락. 단순 epoch 증가보다 과적합 제어와 sequence 표현 튜닝 우선"
            )
        elif float(last_row["validation_pr_auc"]) > float(first_row["validation_pr_auc"]):
            lines.append(
                "- 학습 경향: validation PR-AUC 상승세. 추가 epoch 또는 early stopping 기반 재학습 검토 여지"
            )
        else:
            lines.append(
                "- 학습 경향: validation PR-AUC 정체. epoch 증가만으로 큰 개선을 기대하기 어려운 상태"
            )

    valid_loss = valid_epochs.dropna(subset=["train_loss"])
    if len(valid_loss) >= 2:
        ordered_loss = valid_loss.sort_values("epoch", kind="mergesort")
        first_loss = ordered_loss.iloc[0]["train_loss"]
        last_loss = ordered_loss.iloc[-1]["train_loss"]
        if float(last_loss) < float(first_loss):
            lines.append(
                f"- Train loss: `{_format_float(first_loss)}`에서 `{_format_float(last_loss)}`로 감소. 학습은 진행됐지만 validation 성능은 별도 판단 필요"
            )

    return "\n".join(lines)


def _build_final_plain_interpretation(test_metrics: pd.DataFrame) -> str:
    if test_metrics.empty:
        return "표시 가능한 test split 최종 비교 metric 없음"

    best = test_metrics.iloc[0]
    lines = [
        "#### 최종 결론",
        (
            f"- test PR-AUC 기준 최종 선택 모델은 `{best['model_display']}`이다. "
            f"PR-AUC `{_format_float(best['pr_auc'])}`, "
            f"ROC-AUC `{_format_float(best['roc_auc'])}`를 기록했다."
        ),
        (
            "- PR-AUC를 우선 지표로 둔 이유는 purchase label이 적은 불균형 문제에서 "
            "구매 가능성이 높은 sample을 앞순위에 배치하는 능력이 더 중요하기 때문이다."
        ),
    ]

    contract_status = (
        test_metrics.get("sample_contract_status", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if contract_status:
        if contract_status == ["matched_by_split_counts"]:
            lines.append(
                (
                    "- 비교 조건: 세 모델 모두 같은 test sample 수와 positive 수로 평가됐다. "
                    "따라서 성능 차이는 입력 표현과 모델 구조 차이로 해석할 수 있다."
                )
            )
        else:
            lines.append(
                "- 비교 조건 주의: sample 점검 결과가 불일치하므로 최종 결론 전 평가 조건 확인이 필요하다."
            )

    lines.extend(["", "#### 모델별 해석"])

    logistic_rows = test_metrics.loc[
        test_metrics["model_name"].astype(str).eq("logistic_regression")
    ]
    lightgbm_rows = test_metrics.loc[
        test_metrics["model_name"].astype(str).eq("lightgbm")
    ]
    gru_rows = test_metrics.loc[test_metrics["model_name"].astype(str).eq("gru")]

    if not logistic_rows.empty:
        logistic = logistic_rows.iloc[0]
        lines.append(
            (
                f"- Logistic Regression: PR-AUC `{_format_float(logistic['pr_auc'])}`. "
                "단순 선형 baseline으로 문제의 최소 기준 성능을 제공한다."
            )
        )
    if not lightgbm_rows.empty:
        lightgbm = lightgbm_rows.iloc[0]
        lines.append(
            (
                f"- LightGBM: PR-AUC `{_format_float(lightgbm['pr_auc'])}`. "
                "prefix 길이, cart 누적 횟수, 가격, 시간대 같은 tabular feature의 비선형 조합을 가장 잘 활용했다."
            )
        )
    if not gru_rows.empty:
        gru = gru_rows.iloc[0]
        lines.append(
            (
                f"- GRU: PR-AUC `{_format_float(gru['pr_auc'])}`. "
                "행동 순서를 직접 입력했지만 현재 학습 설정에서는 tabular 요약 feature보다 낮았다."
            )
        )

    baseline_rows = test_metrics.loc[
        ~test_metrics["model_name"].astype(str).eq("gru")
    ]
    if not gru_rows.empty and not baseline_rows.empty:
        gru = gru_rows.iloc[0]
        baseline = baseline_rows.iloc[0]
        difference = float(gru["pr_auc"]) - float(baseline["pr_auc"])
        lines.extend(["", "#### 핵심 비교"])
        if difference >= 0:
            lines.append(
                (
                    f"- GRU는 최고 baseline보다 PR-AUC가 `{difference:.4f}` 높다. "
                    "이 경우 행동 순서 정보가 tabular 요약 feature에 추가 이득을 준 것으로 해석할 수 있다."
                )
            )
        else:
            lines.append(
                (
                    f"- GRU는 최고 baseline보다 PR-AUC가 `{abs(difference):.4f}` 낮다. "
                    "현재 결과에서는 행동 순서 자체보다 가격, 시간, 탐색량, cart 이력 같은 요약 feature가 더 강한 신호였다."
                )
            )
            lines.append(
                "- 결론: 이번 포트폴리오 범위의 최종 선택은 LightGBM이며, GRU는 epoch, hidden size, sequence 표현 튜닝을 후속 개선안으로 둔다."
            )

    return "\n".join(lines)


def main() -> None:
    st.set_page_config(
        page_title="30분 내 구매 확률 예측",
        layout="wide",
    )
    st.title("30분 내 구매 확률 예측")
    st.caption("세션 행동 prefix 기반 PurchaseConversionPrediction 결과 보고서")

    selected_section = render_navigation()
    renderers = {
        "1. Overview": render_overview,
        "2. Data Quality": render_data_quality,
        "3. Labeling": render_labeling,
        "4. EDA": render_eda,
        "5. Features": render_features,
        "6. Baseline Results": render_baseline_results,
        "7. GRU Results": render_gru_results,
        "8. Integrated Comparison": render_integrated_comparison,
        "9. Limitations": render_limitations,
    }
    renderers[selected_section]()


if __name__ == "__main__":
    main()
