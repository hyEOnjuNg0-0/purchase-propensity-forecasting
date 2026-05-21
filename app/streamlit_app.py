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
    best_metric_summary,
    prepare_baseline_test_metrics,
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
        "7. Next Step",
        "8. Reproducibility",
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


def render_next_steps() -> None:
    render_section_title(
        "Next Step",
        "Step 10에서 수행할 최종 모델 비교 범위 정리",
    )
    st.markdown(
        """
        Step 10에서는 Logistic Regression, LightGBM, GRU 결과를 동일 metric schema로
        통합한다. 핵심 비교 기준은 PR-AUC이며, GRU가 baseline보다 높지 않더라도
        현재 sequence 표현의 한계를 간단히 해석한다.
        """
    )


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
        "7. Next Step": render_next_steps,
        "8. Reproducibility": render_reproducibility,
    }
    renderers[selected_section]()


if __name__ == "__main__":
    main()
