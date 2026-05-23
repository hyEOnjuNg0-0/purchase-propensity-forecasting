"""원천 CSV 데이터 프로파일링 유스케이스."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = (
    "event_time",
    "event_type",
    "product_id",
    "category_id",
    "category_code",
    "brand",
    "price",
    "user_id",
    "user_session",
)

ALLOWED_EVENT_TYPES = ("view", "cart", "remove_from_cart", "purchase")
EVENT_TIME_FORMAT = "%Y-%m-%d %H:%M:%S UTC"


@dataclass(frozen=True)
class DataProfile:
    """Step 1 데이터 프로파일링 결과."""

    source_path: Path
    file_size_bytes: int
    row_count: int
    columns: tuple[str, ...]
    dtypes: dict[str, str]
    estimated_memory_bytes: int
    event_time_min: str | None
    event_time_max: str | None
    event_time_parse_failures: int
    event_type_counts: dict[str, int]
    unique_counts: dict[str, int]
    missing_counts: dict[str, int]
    purchase_event_count: int
    purchase_session_count: int

    @property
    def purchase_event_ratio(self) -> float:
        if self.row_count == 0:
            return 0.0
        return self.purchase_event_count / self.row_count

    @property
    def purchase_session_ratio(self) -> float:
        session_count = self.unique_counts.get("user_session", 0)
        if session_count == 0:
            return 0.0
        return self.purchase_session_count / session_count


@dataclass
class _ProfileAccumulator:
    source_path: Path
    file_size_bytes: int
    row_count: int = 0
    columns: tuple[str, ...] = ()
    dtypes: dict[str, str] = field(default_factory=dict)
    observed_memory_bytes: int = 0
    event_time_min: pd.Timestamp | None = None
    event_time_max: pd.Timestamp | None = None
    event_time_parse_failures: int = 0
    event_type_counts: dict[str, int] = field(default_factory=dict)
    missing_counts: dict[str, int] = field(default_factory=dict)
    unique_values: dict[str, set] = field(
        default_factory=lambda: {
            "user_id": set(),
            "user_session": set(),
            "product_id": set(),
            "category_id": set(),
        }
    )
    purchase_sessions: set = field(default_factory=set)

    def update(self, chunk: pd.DataFrame) -> None:
        if not self.columns:
            self.columns = tuple(chunk.columns)
            self.dtypes = {column: str(dtype) for column, dtype in chunk.dtypes.items()}

        chunk_rows = len(chunk)
        self.row_count += chunk_rows
        self.observed_memory_bytes += int(chunk.memory_usage(deep=True).sum())

        for column in chunk.columns:
            self.missing_counts[column] = (
                self.missing_counts.get(column, 0) + int(chunk[column].isna().sum())
            )

        if "event_time" in chunk:
            event_times = pd.to_datetime(
                chunk["event_time"],
                format=EVENT_TIME_FORMAT,
                errors="coerce",
                utc=True,
            )
            self.event_time_parse_failures += int(event_times.isna().sum())
            valid_event_times = event_times.dropna()
            if not valid_event_times.empty:
                chunk_min = valid_event_times.min()
                chunk_max = valid_event_times.max()
                if self.event_time_min is None or chunk_min < self.event_time_min:
                    self.event_time_min = chunk_min
                if self.event_time_max is None or chunk_max > self.event_time_max:
                    self.event_time_max = chunk_max

        if "event_type" in chunk:
            counts = chunk["event_type"].value_counts(dropna=False)
            for event_type, count in counts.items():
                key = "<missing>" if pd.isna(event_type) else str(event_type)
                self.event_type_counts[key] = self.event_type_counts.get(key, 0) + int(count)

        for column, values in self.unique_values.items():
            if column in chunk:
                values.update(chunk[column].dropna().unique().tolist())

        if {"event_type", "user_session"}.issubset(chunk.columns):
            purchase_sessions = chunk.loc[
                chunk["event_type"].eq("purchase"), "user_session"
            ].dropna()
            self.purchase_sessions.update(purchase_sessions.unique().tolist())

    def to_profile(self) -> DataProfile:
        unique_counts = {
            column: len(values) for column, values in self.unique_values.items()
        }
        purchase_event_count = self.event_type_counts.get("purchase", 0)
        return DataProfile(
            source_path=self.source_path,
            file_size_bytes=self.file_size_bytes,
            row_count=self.row_count,
            columns=self.columns,
            dtypes=self.dtypes,
            estimated_memory_bytes=self.observed_memory_bytes,
            event_time_min=_timestamp_to_text(self.event_time_min),
            event_time_max=_timestamp_to_text(self.event_time_max),
            event_time_parse_failures=self.event_time_parse_failures,
            event_type_counts=dict(sorted(self.event_type_counts.items())),
            unique_counts=unique_counts,
            missing_counts=dict(sorted(self.missing_counts.items())),
            purchase_event_count=purchase_event_count,
            purchase_session_count=len(self.purchase_sessions),
        )


def profile_csv(csv_path: Path, chunksize: int = 1_000_000) -> DataProfile:
    """CSV를 chunk 단위로 읽어 Step 1 프로파일링 지표를 계산한다."""

    path = Path(csv_path)
    accumulator = _ProfileAccumulator(
        source_path=path,
        file_size_bytes=path.stat().st_size,
    )
    for chunk in pd.read_csv(path, chunksize=chunksize):
        accumulator.update(chunk)
    return accumulator.to_profile()


def build_summary_rows(profile: DataProfile) -> list[dict[str, str]]:
    """리포트용 요약표 행을 생성한다."""

    rows = [
        _row("데이터 크기", "파일 경로", str(profile.source_path), "프로파일링 대상 CSV"),
        _row("데이터 크기", "CSV 파일 크기", _format_bytes(profile.file_size_bytes), "디스크 기준"),
        _row("데이터 크기", "row count", f"{profile.row_count:,}", "전체 이벤트 수"),
        _row(
            "데이터 크기",
            "추정 pandas memory footprint",
            _format_bytes(profile.estimated_memory_bytes),
            "chunk deep memory 합산 기준",
        ),
        _row("스키마", "컬럼 수", f"{len(profile.columns):,}", "CSV header 기준"),
        _row(
            "시간 범위",
            "event_time min",
            profile.event_time_min or "",
            "UTC 파싱 가능 값 기준",
        ),
        _row(
            "시간 범위",
            "event_time max",
            profile.event_time_max or "",
            "UTC 파싱 가능 값 기준",
        ),
        _row(
            "시간 범위",
            "event_time parse failures",
            f"{profile.event_time_parse_failures:,}",
            "파싱 실패 또는 결측 event_time",
        ),
        _row(
            "행동 분포",
            "purchase event count",
            f"{profile.purchase_event_count:,}",
            "event_type == purchase",
        ),
        _row(
            "행동 분포",
            "purchase event ratio",
            _format_ratio(profile.purchase_event_ratio),
            "전체 이벤트 중 purchase 비율",
        ),
        _row(
            "행동 분포",
            "purchase session count",
            f"{profile.purchase_session_count:,}",
            "purchase가 1회 이상 포함된 세션 수",
        ),
        _row(
            "행동 분포",
            "purchase session ratio",
            _format_ratio(profile.purchase_session_ratio),
            "전체 세션 중 purchase 포함 세션 비율",
        ),
    ]

    for column, count in profile.unique_counts.items():
        rows.append(
            _row(
                "고유값",
                f"{column} unique count",
                f"{count:,}",
                "결측 제외 exact count",
            )
        )

    for column in profile.columns:
        dtype = profile.dtypes.get(column, "")
        rows.append(_row("스키마", f"{column} dtype", dtype, "pandas 추론 dtype"))

    return rows


def build_event_type_rows(profile: DataProfile) -> list[dict[str, str]]:
    total = profile.row_count
    return [
        {
            "event_type": event_type,
            "count": str(count),
            "ratio": _format_ratio(count / total if total else 0.0),
        }
        for event_type, count in sorted(
            profile.event_type_counts.items(), key=lambda item: item[1], reverse=True
        )
    ]


def build_missing_value_rows(profile: DataProfile) -> list[dict[str, str]]:
    total = profile.row_count
    return [
        {
            "column": column,
            "missing_count": str(profile.missing_counts.get(column, 0)),
            "missing_ratio": _format_ratio(
                profile.missing_counts.get(column, 0) / total if total else 0.0
            ),
        }
        for column in profile.columns
    ]


def build_quality_issue_rows(profile: DataProfile) -> list[dict[str, str]]:
    """Step 2 전에 검토할 데이터 품질 이슈 초안을 생성한다."""

    issues: list[dict[str, str]] = []
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in profile.columns]
    if missing_columns:
        issues.append(
            _issue(
                "critical",
                "필수 컬럼 누락",
                ", ".join(missing_columns),
                "원천 데이터 스키마 또는 입력 파일을 확인한다.",
            )
        )

    unexpected_events = sorted(
        set(profile.event_type_counts) - set(ALLOWED_EVENT_TYPES) - {"<missing>"}
    )
    if unexpected_events:
        issues.append(
            _issue(
                "warning",
                "허용 범위 밖 event_type",
                ", ".join(unexpected_events),
                "Step 2에서 event_type 허용 값 검증 로직으로 분리한다.",
            )
        )

    if profile.event_time_parse_failures:
        issues.append(
            _issue(
                "warning",
                "event_time 파싱 실패",
                f"{profile.event_time_parse_failures:,} rows",
                "원천 값 패턴과 timezone 표기를 확인한다.",
            )
        )

    for column, missing_count in profile.missing_counts.items():
        if missing_count == 0:
            continue
        issues.append(
            _issue(
                "info",
                f"{column} 결측 존재",
                f"{missing_count:,} rows ({_format_ratio(missing_count / profile.row_count)})",
                "모델 영향과 unknown 처리 여부를 Step 2/5에서 결정한다.",
            )
        )

    if not issues:
        issues.append(
            _issue(
                "info",
                "Step 1 기준 주요 품질 이슈 없음",
                "필수 컬럼, event_time parsing, event_type 분포 기준",
                "Step 2에서 중복, 가격 이상치, 세션 시간 역전 검증을 수행한다.",
            )
        )

    return issues


def write_profile_artifacts(profile: DataProfile, reports_dir: Path) -> None:
    """프로파일링 결과를 Streamlit/리포트에서 재사용 가능한 artifact로 저장한다."""

    output_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    build_raw_data_preview(profile.source_path).to_csv(
        output_dir / "raw_data_preview.csv",
        index=False,
        encoding="utf-8",
    )
    _write_csv(output_dir / "data_profile_summary.csv", build_summary_rows(profile))
    _write_csv(
        output_dir / "data_profile_event_type_distribution.csv",
        build_event_type_rows(profile),
    )
    _write_csv(
        output_dir / "data_profile_missing_values.csv",
        build_missing_value_rows(profile),
    )
    _write_csv(
        output_dir / "data_quality_issues_draft.csv",
        build_quality_issue_rows(profile),
    )
    (output_dir / "data_profile_report.md").write_text(
        build_markdown_report(profile),
        encoding="utf-8",
    )


def build_raw_data_preview(csv_path: Path, row_count: int = 15) -> pd.DataFrame:
    """Streamlit Overview에 표시할 원천 CSV 상위 행 preview를 생성한다."""

    return pd.read_csv(Path(csv_path), nrows=row_count)


def build_markdown_report(profile: DataProfile) -> str:
    """사람이 바로 읽을 수 있는 Step 1 리포트 초안을 생성한다."""

    lines = [
        "# Step 1 데이터 프로파일링",
        "",
        "## 요약",
        "",
        f"- 대상 파일: `{profile.source_path}`",
        f"- 전체 row count: {profile.row_count:,}",
        f"- CSV 파일 크기: {_format_bytes(profile.file_size_bytes)}",
        f"- 추정 pandas memory footprint: {_format_bytes(profile.estimated_memory_bytes)}",
        f"- event_time 범위: {profile.event_time_min} ~ {profile.event_time_max}",
        f"- purchase event 비율: {_format_ratio(profile.purchase_event_ratio)}",
        f"- purchase 포함 세션 비율: {_format_ratio(profile.purchase_session_ratio)}",
        "",
        "## 산출물",
        "",
        "- `data_profile_summary.csv`: Step 1 핵심 요약표",
        "- `data_profile_event_type_distribution.csv`: event_type 분포",
        "- `data_profile_missing_values.csv`: 컬럼별 결측 현황",
        "- `data_quality_issues_draft.csv`: 데이터 품질 이슈 초안",
        "",
        "## 다음 검증 후보",
        "",
        "- 필수 컬럼 존재 여부를 자동 검증한다.",
        "- `price <= 0` 이상치를 확인한다.",
        "- 완전 중복 row 비율과 세션 내 시간 역전 여부를 확인한다.",
        "- 극단적으로 긴 세션과 purchase/비purchase 세션의 차이를 비교한다.",
        "",
    ]
    return "\n".join(lines)


def _write_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    pd.DataFrame(list(rows)).to_csv(path, index=False, encoding="utf-8")


def _row(section: str, metric: str, value: str, description: str) -> dict[str, str]:
    return {
        "section": section,
        "metric": metric,
        "value": value,
        "description": description,
    }


def _issue(
    severity: str,
    issue: str,
    evidence: str,
    next_action: str,
) -> dict[str, str]:
    return {
        "severity": severity,
        "issue": issue,
        "evidence": evidence,
        "next_action": next_action,
    }


def _timestamp_to_text(value: pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _format_bytes(value: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:,.2f} {unit}"
        size /= 1024
    return f"{value:,} B"


def _format_ratio(value: float) -> str:
    return f"{value:.6f}"
