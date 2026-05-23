from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from purchase_time_forecasting.streamlit_report import (  # noqa: E402
    BASELINE_BUILD_COMMAND,
    FINAL_REPORT_BUILD_COMMAND,
    best_metric_summary,
    prepare_baseline_test_metrics,
    prepare_final_test_metrics,
    select_best_strategy,
    top_feature_importance,
)


ARTIFACTS_DIR = REPO_ROOT / "artifacts"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

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
FEATURE_DESCRIPTIONS = {
    "prefix_length": "기준 시점까지 관측된 세션 내 이벤트 개수",
    "last_event_type": "기준 시점의 마지막 사용자 행동 유형",
    "session_elapsed_minutes": "세션 시작 이후 기준 시점까지 경과한 시간",
    "time_since_previous_event_minutes": "직전 이벤트 이후 기준 시점까지 경과한 시간",
    "hour": "기준 시점의 시간대",
    "event_count_view": "기준 시점까지 누적된 view 이벤트 수",
    "event_count_cart": "기준 시점까지 누적된 cart 이벤트 수",
    "event_count_remove_from_cart": "기준 시점까지 누적된 remove_from_cart 이벤트 수",
    "unique_product_count": "기준 시점까지 상호작용한 고유 상품 수",
    "unique_category_count": "기준 시점까지 상호작용한 고유 카테고리 수",
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
    st.warning(f"`{_relative_path(path)}` artifact가 없습니다.")
    st.code(command, language="powershell")


def render_navigation() -> str:
    st.sidebar.title("목차")
    sections = [
        "1. Overview",
        "2. Data Quality",
        "3. Labeling",
        "4. EDA",
        "5. Features",
        "6. Baseline Results",
        "7. GRU 결과",
        "8. 통합비교 및 최종 해석",
        "9. Reproducibility",
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
        - 사용 데이터 : `2019-Oct.csv`에서 2019-10-10까지
        """
    )

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
            "purchase_rate",
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
        st.info("표시 가능한 test split baseline metric이 없습니다.")
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
            st.info("표시 가능한 LightGBM feature importance가 없습니다.")
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
        "GRU 결과",
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
        st.info("표시 가능한 GRU test metric이 없습니다.")
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
        st.info("표시 가능한 GRU metric이 없습니다.")
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
        st.info("표시 가능한 GRU 학습 이력이 없습니다.")
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
            st.line_chart(
                prepared_history.loc[:, ["epoch", *metric_columns]].set_index("epoch"),
                height=240,
            )
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
        "통합비교 및 최종 해석",
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
        st.info("표시 가능한 test split 최종 모델 비교 metric이 없습니다.")
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

    st.divider()
    st.subheader("원문 해석 artifact")
    if interpretation is None:
        render_missing_artifact(
            REPORTS_DIR / "model_interpretation_summary.md",
            FINAL_REPORT_BUILD_COMMAND,
        )
    else:
        with st.expander("model_interpretation_summary.md 원문 보기"):
            st.markdown(interpretation)


def render_reproducibility() -> None:
    render_section_title(
        "Reproducibility",
        "분석 artifact와 Streamlit 보고서를 재생성하는 실행 순서입니다.",
    )
    st.code(
        "\n".join(
            [
                ".\\scripts\\run_ptf.ps1 python scripts\\profile_data.py",
                ".\\scripts\\run_ptf.ps1 python scripts\\validate_data_quality.py",
                ".\\scripts\\run_ptf.ps1 python scripts\\create_labels.py",
                ".\\scripts\\run_ptf.ps1 python scripts\\run_eda.py",
                ".\\scripts\\run_ptf.ps1 python scripts\\build_features.py",
                ".\\scripts\\run_ptf.ps1 python scripts\\train_baselines.py",
                ".\\scripts\\run_ptf.ps1 python scripts\\train_gru.py",
                ".\\scripts\\run_ptf.ps1 python scripts\\build_final_report.py",
                ".\\scripts\\run_ptf.ps1 streamlit run app/streamlit_app.py",
            ]
        ),
        language="powershell",
    )


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _lookup_metric(frame: pd.DataFrame | None, metric_name: str) -> float | None:
    if frame is None or {"metric", "value"} - set(frame.columns):
        return None
    values = frame.loc[frame["metric"].astype(str).eq(metric_name), "value"]
    if values.empty:
        return None
    return pd.to_numeric(values.iloc[0], errors="coerce")


def _format_ratio(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "artifact 없음"
    return f"{float(value):.2%}"


def _format_float(value: object) -> str:
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        return "N/A"
    return f"{float(numeric_value):.4f}"


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


def _build_gru_plain_interpretation(
    pr_auc: object,
    roc_auc: object,
    recall_at_k: object,
    precision_at_k: object,
) -> str:
    return "\n".join(
        [
            "- GRU는 `view -> cart -> view`처럼 기준 시점까지의 행동 순서를 직접 읽는 모델입니다.",
            f"- Test PR-AUC는 `{_format_float(pr_auc)}`입니다. 구매 sample이 적은 문제에서 구매 가능성이 높은 사용자를 앞쪽에 세우는 힘을 봅니다.",
            f"- Test ROC-AUC는 `{_format_float(roc_auc)}`입니다. 구매/비구매를 전반적으로 구분하는 힘은 이 값으로 확인합니다.",
            f"- 상위 10% 후보만 본다면 실제 구매자의 `{_format_float(recall_at_k)}`를 잡고, 그 후보 안의 구매 비율은 `{_format_float(precision_at_k)}`입니다.",
        ]
    )


def _build_final_plain_interpretation(test_metrics: pd.DataFrame) -> str:
    if test_metrics.empty:
        return "표시 가능한 test split 최종 비교 metric이 없습니다."

    best = test_metrics.iloc[0]
    lines = [
        f"- 최종 비교에서는 `{best['model_display']}`가 test PR-AUC `{_format_float(best['pr_auc'])}`로 가장 높습니다.",
        "- 이 값이 높다는 뜻은 실제 구매할 가능성이 큰 sample을 더 앞순위에 잘 배치했다는 의미입니다.",
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
                "- 세 모델은 같은 test sample 수와 positive 수로 비교되어, 입력 표현 차이를 비교하는 조건이 맞습니다."
            )
        else:
            lines.append(
                "- sample 점검 결과가 완전히 일치하지 않습니다. 최종 결론을 읽을 때 비교 조건을 먼저 확인해야 합니다."
            )

    gru_rows = test_metrics.loc[test_metrics["model_name"].astype(str).eq("gru")]
    baseline_rows = test_metrics.loc[
        ~test_metrics["model_name"].astype(str).eq("gru")
    ]
    if not gru_rows.empty and not baseline_rows.empty:
        gru = gru_rows.iloc[0]
        baseline = baseline_rows.iloc[0]
        difference = float(gru["pr_auc"]) - float(baseline["pr_auc"])
        if difference >= 0:
            lines.append(
                f"- GRU는 최고 baseline보다 PR-AUC가 `{difference:.4f}` 높습니다. 이 경우 행동 순서 정보가 tabular 요약보다 추가 이득을 준 것으로 볼 수 있습니다."
            )
        else:
            lines.append(
                f"- GRU는 최고 baseline보다 PR-AUC가 `{abs(difference):.4f}` 낮습니다. 현재 결과에서는 최근 행동 순서 전체보다 가격, 시간, 탐색량 같은 요약 feature를 쓰는 LightGBM이 더 실용적입니다."
            )
            lines.append(
                "- GRU 성능을 올리려면 epoch, hidden size, sequence feature 표현을 다시 튜닝해야 하지만, 이번 최종 선택 근거는 LightGBM 쪽이 더 명확합니다."
            )

    return "\n".join(lines)


def prepare_training_feature_dictionary(feature_dictionary: pd.DataFrame) -> pd.DataFrame:
    required = {"feature_name", "model_role"}
    output_columns = ["feature_name", "feature_description"]
    if feature_dictionary.empty or not required.issubset(feature_dictionary.columns):
        return pd.DataFrame(columns=output_columns)

    filtered = feature_dictionary.loc[
        feature_dictionary["model_role"].astype(str).isin(
            {"tabular_input", "sequence_input"}
        ),
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


def main() -> None:
    st.set_page_config(
        page_title="30분 내 구매 확률 예측",
        layout="wide",
    )
    st.title("30분 내 구매 확률 예측")
    st.caption("세션 행동 prefix 기반 Purchase Propensity Forecasting 결과 보고서")

    selected_section = render_navigation()
    renderers = {
        "1. Overview": render_overview,
        "2. Data Quality": render_data_quality,
        "3. Labeling": render_labeling,
        "4. EDA": render_eda,
        "5. Features": render_features,
        "6. Baseline Results": render_baseline_results,
        "7. GRU 결과": render_gru_results,
        "8. 통합비교 및 최종 해석": render_integrated_comparison,
        "9. Reproducibility": render_reproducibility,
    }
    renderers[selected_section]()


if __name__ == "__main__":
    main()
