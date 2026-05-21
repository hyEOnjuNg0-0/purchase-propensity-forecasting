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
    build_artifact_status,
    missing_artifact_commands,
    prepare_baseline_test_metrics,
    select_best_strategy,
    step8_artifact_requirements,
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


def render_artifact_status() -> None:
    status = build_artifact_status(step8_artifact_requirements(REPORTS_DIR))
    st.sidebar.header("Artifact 상태")
    sidebar_status = status.copy()
    sidebar_status["path"] = sidebar_status["path"].map(
        lambda value: _relative_path(Path(value))
    )
    st.sidebar.dataframe(sidebar_status, use_container_width=True, hide_index=True)

    commands = missing_artifact_commands(status)
    if commands:
        st.sidebar.caption("누락 artifact 생성 명령")
        for command in commands:
            st.sidebar.code(command, language="powershell")


def render_overview() -> None:
    st.header("Overview")
    st.markdown(
        """
        이 보고서는 `2019-Oct.csv`의 세션 내 행동 이력을 이용해 기준 시점 이후
        **30분 내 purchase 발생 확률**을 예측하는 개인 과제 결과물이다.

        - 예측 단위: `user_session` 내부 prefix sequence
        - 라벨: 같은 세션에서 `(cutoff_time, cutoff_time + 30분]` 내 첫 purchase 발생 여부
        - 현재 단계: Logistic Regression과 LightGBM baseline 결과를 Streamlit에 연결
        - 다음 단계: 같은 `sample_id` 기준으로 GRU sequence 모델 학습 후 비교
        """
    )


def render_data_quality() -> None:
    st.header("Data Quality")
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
    left, right = st.columns([1, 2])
    with left:
        st.subheader("검증 상태 요약")
        st.dataframe(status_counts, use_container_width=True, hide_index=True)
    with right:
        st.subheader("검증 상세")
        st.dataframe(summary, use_container_width=True, hide_index=True)


def render_labeling() -> None:
    st.header("Labeling")
    distribution = read_csv(str(REPORTS_DIR / "label_distribution.csv"))
    policy = read_csv(str(REPORTS_DIR / "labeling_policy.csv"))

    left, right = st.columns(2)
    with left:
        st.subheader("라벨 분포")
        if distribution is None:
            render_missing_artifact(
                REPORTS_DIR / "label_distribution.csv",
                ".\\scripts\\run_ptf.ps1 python scripts\\create_labels.py",
            )
        else:
            st.dataframe(distribution, use_container_width=True, hide_index=True)
    with right:
        st.subheader("라벨링 정책")
        if policy is None:
            render_missing_artifact(
                REPORTS_DIR / "labeling_policy.csv",
                ".\\scripts\\run_ptf.ps1 python scripts\\create_labels.py",
            )
        else:
            st.dataframe(policy, use_container_width=True, hide_index=True)


def render_eda() -> None:
    st.header("EDA")
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
    st.header("Features")
    split_summary = read_csv(str(REPORTS_DIR / "feature_split_summary.csv"))
    feature_dictionary = read_csv(str(REPORTS_DIR / "feature_dictionary.csv"))

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Split 분포")
        if split_summary is None:
            render_missing_artifact(
                REPORTS_DIR / "feature_split_summary.csv",
                ".\\scripts\\run_ptf.ps1 python scripts\\build_features.py",
            )
        else:
            st.dataframe(split_summary, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Feature Dictionary")
        if feature_dictionary is None:
            render_missing_artifact(
                REPORTS_DIR / "feature_dictionary.csv",
                ".\\scripts\\run_ptf.ps1 python scripts\\build_features.py",
            )
        else:
            st.dataframe(feature_dictionary, use_container_width=True, hide_index=True)


def render_baseline_results() -> None:
    st.header("Baseline Results")
    metrics = read_csv(str(REPORTS_DIR / "model_metrics.csv"))
    importance = read_csv(str(REPORTS_DIR / "baseline_feature_importance.csv"))
    status = read_csv(str(REPORTS_DIR / "baseline_model_status.csv"))
    report = read_markdown(str(REPORTS_DIR / "baseline_report.md"))

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

    st.subheader("모델 상태")
    if status is None:
        render_missing_artifact(
            REPORTS_DIR / "baseline_model_status.csv",
            BASELINE_BUILD_COMMAND,
        )
    else:
        st.dataframe(status, use_container_width=True, hide_index=True)

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
            st.dataframe(top_features, use_container_width=True, hide_index=True)

    if report:
        with st.expander("Baseline report markdown"):
            st.markdown(report)


def render_next_steps() -> None:
    st.header("Next Step")
    st.markdown(
        """
        Step 9에서는 `sequence_feature_dataset.parquet`를 사용해 최소 GRU 모델을 학습한다.
        목표는 복잡한 튜닝이 아니라, 동일 `sample_id`와 동일 split 기준에서
        tabular baseline 대비 sequence 표현이 추가 효용을 주는지 확인하는 것이다.
        """
    )


def render_reproducibility() -> None:
    st.header("Reproducibility")
    st.code(
        "\n".join(
            [
                ".\\scripts\\run_ptf.ps1 python scripts\\profile_data.py",
                ".\\scripts\\run_ptf.ps1 python scripts\\validate_data_quality.py",
                ".\\scripts\\run_ptf.ps1 python scripts\\create_labels.py",
                ".\\scripts\\run_ptf.ps1 python scripts\\run_eda.py",
                ".\\scripts\\run_ptf.ps1 python scripts\\build_features.py",
                ".\\scripts\\run_ptf.ps1 python scripts\\train_baselines.py",
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


def main() -> None:
    st.set_page_config(
        page_title="30분 내 구매 확률 예측",
        layout="wide",
    )
    st.title("30분 내 구매 확률 예측")
    st.caption("세션 행동 prefix 기반 Purchase Propensity Forecasting 결과 보고서")

    render_artifact_status()
    render_overview()
    render_data_quality()
    render_labeling()
    render_eda()
    render_features()
    render_baseline_results()
    render_next_steps()
    render_reproducibility()


if __name__ == "__main__":
    main()
