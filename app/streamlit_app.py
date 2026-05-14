from pathlib import Path

import pandas as pd
import streamlit as st


ARTIFACTS_DIR = Path("artifacts")
REPORTS_DIR = ARTIFACTS_DIR / "reports"
FIGURES_DIR = ARTIFACTS_DIR / "figures"


def read_markdown(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def show_missing_artifacts() -> None:
    st.info(
        "아직 표시할 분석 artifact가 없습니다. 실제 분석 pipeline 실행 후 "
        "`artifacts/reports`와 `artifacts/figures`에 결과 파일을 저장하면 이 화면에 표시됩니다."
    )
    st.code(
        "\n".join(
            [
                "artifacts/",
                "|-- reports/",
                "|   |-- project_summary.md",
                "|   |-- data_quality_summary.csv",
                "|   `-- model_metrics.csv",
                "`-- figures/",
                "    |-- event_distribution.png",
                "    |-- model_comparison.png",
                "    `-- representation_umap.png",
            ]
        ),
        language="text",
    )


def render_summary() -> bool:
    summary = read_markdown(REPORTS_DIR / "project_summary.md")
    if summary is None:
        return False
    st.markdown(summary)
    return True


def render_table(title: str, path: Path) -> bool:
    data = read_csv(path)
    if data is None:
        return False
    st.subheader(title)
    st.dataframe(data, use_container_width=True, hide_index=True)
    return True


def render_figure(title: str, path: Path) -> bool:
    if not path.exists():
        return False
    st.subheader(title)
    st.image(str(path), use_container_width=True)
    return True


def main() -> None:
    st.set_page_config(
        page_title="Purchase Time Forecasting",
        layout="wide",
    )
    st.title("Purchase Time Forecasting")
    st.caption("사용자 행동 시퀀스 기반 향후 30분 내 구매 예측 포트폴리오")

    rendered = False

    with st.container():
        st.header("프로젝트 요약")
        rendered = render_summary() or rendered

    left, right = st.columns(2)
    with left:
        rendered = render_table(
            "데이터 신뢰성 검증",
            REPORTS_DIR / "data_quality_summary.csv",
        ) or rendered
    with right:
        rendered = render_table(
            "모델 성능 비교",
            REPORTS_DIR / "model_metrics.csv",
        ) or rendered

    st.header("핵심 시각화")
    col1, col2, col3 = st.columns(3)
    with col1:
        rendered = render_figure(
            "이벤트 분포",
            FIGURES_DIR / "event_distribution.png",
        ) or rendered
    with col2:
        rendered = render_figure(
            "모델 성능 비교",
            FIGURES_DIR / "model_comparison.png",
        ) or rendered
    with col3:
        rendered = render_figure(
            "행동 패턴 임베딩",
            FIGURES_DIR / "representation_umap.png",
        ) or rendered

    if not rendered:
        show_missing_artifacts()


if __name__ == "__main__":
    main()
