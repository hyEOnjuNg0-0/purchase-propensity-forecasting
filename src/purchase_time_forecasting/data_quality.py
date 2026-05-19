"""데이터 신뢰성 검증 유스케이스."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from purchase_time_forecasting.data_profiling import (
    ALLOWED_EVENT_TYPES,
    EVENT_TIME_FORMAT,
    REQUIRED_COLUMNS,
)


@dataclass
class SessionStats:
    """세션 단위 정합성 검증에 필요한 최소 통계."""

    event_count: int = 0
    purchase_count: int = 0
    min_time: pd.Timestamp | None = None
    max_time: pd.Timestamp | None = None
    last_seen_time: pd.Timestamp | None = None
    price_sum: float = 0.0
    valid_price_count: int = 0
    has_time_reversal: bool = False

    def update(
        self,
        event_count: int,
        purchase_count: int,
        min_time: pd.Timestamp | None,
        max_time: pd.Timestamp | None,
        first_seen_time: pd.Timestamp | None,
        last_seen_time: pd.Timestamp | None,
        price_sum: float,
        valid_price_count: int,
    ) -> bool:
        """chunk 집계 결과를 병합하고 세션 시간 역전 여부를 반환한다."""

        reversal_detected = False
        if (
            self.last_seen_time is not None
            and first_seen_time is not None
            and first_seen_time < self.last_seen_time
        ):
            reversal_detected = True

        self.event_count += int(event_count)
        self.purchase_count += int(purchase_count)
        self.price_sum += float(price_sum)
        self.valid_price_count += int(valid_price_count)

        if min_time is not None:
            if self.min_time is None or min_time < self.min_time:
                self.min_time = min_time
        if max_time is not None:
            if self.max_time is None or max_time > self.max_time:
                self.max_time = max_time
        if last_seen_time is not None:
            self.last_seen_time = last_seen_time

        if reversal_detected:
            self.has_time_reversal = True
        return reversal_detected

    @property
    def duration_minutes(self) -> float | None:
        if self.min_time is None or self.max_time is None:
            return None
        return (self.max_time - self.min_time).total_seconds() / 60

    @property
    def has_purchase(self) -> bool:
        return self.purchase_count > 0

    @property
    def avg_price(self) -> float | None:
        if self.valid_price_count == 0:
            return None
        return self.price_sum / self.valid_price_count


@dataclass(frozen=True)
class DataQualityResult:
    """Step 2 데이터 신뢰성 검증 결과."""

    source_path: Path
    row_count: int
    columns: tuple[str, ...]
    missing_columns: tuple[str, ...]
    unexpected_event_types: tuple[str, ...]
    event_time_parse_failures: int
    missing_counts: dict[str, int]
    price_missing_count: int
    price_zero_count: int
    price_negative_count: int
    price_min: float | None
    price_max: float | None
    duplicate_row_count: int
    duplicate_hash_collision_note: str
    session_count: int
    purchase_session_count: int
    non_purchase_session_count: int
    time_reversal_event_count: int
    time_reversal_session_count: int
    extreme_session_threshold: int
    extreme_session_count: int
    max_session_length: int
    max_session_duration_minutes: float | None
    session_comparison: dict[str, dict[str, float | int | str]]
    extreme_sessions: list[dict[str, str]]

    @property
    def duplicate_row_ratio(self) -> float:
        if self.row_count == 0:
            return 0.0
        return self.duplicate_row_count / self.row_count

    @property
    def invalid_price_count(self) -> int:
        return self.price_zero_count + self.price_negative_count

    @property
    def invalid_price_ratio(self) -> float:
        if self.row_count == 0:
            return 0.0
        return self.invalid_price_count / self.row_count

    @property
    def time_reversal_event_ratio(self) -> float:
        if self.row_count == 0:
            return 0.0
        return self.time_reversal_event_count / self.row_count


def validate_data_quality(
    csv_path: Path,
    chunksize: int = 1_000_000,
    extreme_session_min_length: int = 100,
    top_extreme_sessions: int = 20,
) -> DataQualityResult:
    """Step 2 데이터 신뢰성 검증을 수행한다.

    대용량 CSV 처리를 위해 row 단위 검증과 세션 단위 검증을 분리한다.
    완전 중복 row는 pandas row hash 기준으로 전체 파일 범위에서 계산한다.
    """

    path = Path(csv_path)
    row_result = _validate_rows(path, chunksize)
    session_result = _validate_sessions(
        path,
        chunksize,
        extreme_session_min_length,
        top_extreme_sessions,
    )

    return DataQualityResult(
        source_path=path,
        row_count=row_result["row_count"],
        columns=row_result["columns"],
        missing_columns=row_result["missing_columns"],
        unexpected_event_types=row_result["unexpected_event_types"],
        event_time_parse_failures=row_result["event_time_parse_failures"],
        missing_counts=row_result["missing_counts"],
        price_missing_count=row_result["price_missing_count"],
        price_zero_count=row_result["price_zero_count"],
        price_negative_count=row_result["price_negative_count"],
        price_min=row_result["price_min"],
        price_max=row_result["price_max"],
        duplicate_row_count=row_result["duplicate_row_count"],
        duplicate_hash_collision_note="pandas.util.hash_pandas_object 64-bit row hash 기준",
        session_count=session_result["session_count"],
        purchase_session_count=session_result["purchase_session_count"],
        non_purchase_session_count=session_result["non_purchase_session_count"],
        time_reversal_event_count=session_result["time_reversal_event_count"],
        time_reversal_session_count=session_result["time_reversal_session_count"],
        extreme_session_threshold=session_result["extreme_session_threshold"],
        extreme_session_count=session_result["extreme_session_count"],
        max_session_length=session_result["max_session_length"],
        max_session_duration_minutes=session_result["max_session_duration_minutes"],
        session_comparison=session_result["session_comparison"],
        extreme_sessions=session_result["extreme_sessions"],
    )


def _validate_rows(path: Path, chunksize: int) -> dict:
    row_count = 0
    columns: tuple[str, ...] = ()
    missing_counts: dict[str, int] = {}
    event_time_parse_failures = 0
    event_types: set[str] = set()
    price_missing_count = 0
    price_zero_count = 0
    price_negative_count = 0
    price_min: float | None = None
    price_max: float | None = None
    seen_hashes: set[int] = set()
    duplicate_row_count = 0

    for chunk in pd.read_csv(path, chunksize=chunksize):
        if not columns:
            columns = tuple(chunk.columns)

        row_count += len(chunk)
        for column in chunk.columns:
            missing_counts[column] = (
                missing_counts.get(column, 0) + int(chunk[column].isna().sum())
            )

        if "event_time" in chunk:
            parsed_time = pd.to_datetime(
                chunk["event_time"],
                format=EVENT_TIME_FORMAT,
                errors="coerce",
                utc=True,
            )
            event_time_parse_failures += int(parsed_time.isna().sum())

        if "event_type" in chunk:
            event_types.update(chunk["event_type"].dropna().astype(str).unique().tolist())

        if "price" in chunk:
            price = pd.to_numeric(chunk["price"], errors="coerce")
            price_missing_count += int(price.isna().sum())
            price_zero_count += int(price.eq(0).sum())
            price_negative_count += int(price.lt(0).sum())
            valid_price = price.dropna()
            if not valid_price.empty:
                chunk_min = float(valid_price.min())
                chunk_max = float(valid_price.max())
                price_min = chunk_min if price_min is None else min(price_min, chunk_min)
                price_max = chunk_max if price_max is None else max(price_max, chunk_max)

        row_hashes = pd.util.hash_pandas_object(chunk, index=False).to_numpy()
        for row_hash in row_hashes:
            hash_value = int(row_hash)
            if hash_value in seen_hashes:
                duplicate_row_count += 1
            else:
                seen_hashes.add(hash_value)

    missing_columns = tuple(column for column in REQUIRED_COLUMNS if column not in columns)
    unexpected_event_types = tuple(sorted(event_types - set(ALLOWED_EVENT_TYPES)))

    return {
        "row_count": row_count,
        "columns": columns,
        "missing_columns": missing_columns,
        "unexpected_event_types": unexpected_event_types,
        "event_time_parse_failures": event_time_parse_failures,
        "missing_counts": dict(sorted(missing_counts.items())),
        "price_missing_count": price_missing_count,
        "price_zero_count": price_zero_count,
        "price_negative_count": price_negative_count,
        "price_min": price_min,
        "price_max": price_max,
        "duplicate_row_count": duplicate_row_count,
    }


def _validate_sessions(
    path: Path,
    chunksize: int,
    extreme_session_min_length: int,
    top_extreme_sessions: int,
) -> dict:
    session_stats: dict[str, SessionStats] = {}
    time_reversal_event_count = 0

    for chunk in pd.read_csv(path, chunksize=chunksize):
        required = {"user_session", "event_time", "event_type", "price"}
        if not required.issubset(chunk.columns):
            continue

        working = chunk.loc[chunk["user_session"].notna()].copy()
        if working.empty:
            continue

        working["_event_time"] = pd.to_datetime(
            working["event_time"],
            format=EVENT_TIME_FORMAT,
            errors="coerce",
            utc=True,
        )
        working["_is_purchase"] = working["event_type"].eq("purchase").astype("int64")
        working["_price"] = pd.to_numeric(working["price"], errors="coerce")

        valid_time = working.loc[working["_event_time"].notna()]
        intra_reversal_sessions: set[str] = set()
        if not valid_time.empty:
            previous_time = valid_time.groupby("user_session")["_event_time"].shift()
            intra_reversal_mask = valid_time["_event_time"].lt(previous_time)
            time_reversal_event_count += int(intra_reversal_mask.sum())
            intra_reversal_sessions = set(
                valid_time.loc[intra_reversal_mask, "user_session"].astype(str).tolist()
            )
            first_seen = valid_time.groupby("user_session", sort=False)["_event_time"].first()
            last_seen = valid_time.groupby("user_session", sort=False)["_event_time"].last()
            min_time = valid_time.groupby("user_session", sort=False)["_event_time"].min()
            max_time = valid_time.groupby("user_session", sort=False)["_event_time"].max()
        else:
            first_seen = pd.Series(dtype="datetime64[ns, UTC]")
            last_seen = pd.Series(dtype="datetime64[ns, UTC]")
            min_time = pd.Series(dtype="datetime64[ns, UTC]")
            max_time = pd.Series(dtype="datetime64[ns, UTC]")

        grouped = working.groupby("user_session", sort=False).agg(
            event_count=("event_time", "size"),
            purchase_count=("_is_purchase", "sum"),
            price_sum=("_price", "sum"),
            valid_price_count=("_price", "count"),
        )

        for row in grouped.itertuples():
            session_id = str(row.Index)
            stats = session_stats.setdefault(session_id, SessionStats())
            if session_id in intra_reversal_sessions:
                stats.has_time_reversal = True
            cross_chunk_reversal = stats.update(
                event_count=int(row.event_count),
                purchase_count=int(row.purchase_count),
                min_time=_series_timestamp(min_time, row.Index),
                max_time=_series_timestamp(max_time, row.Index),
                first_seen_time=_series_timestamp(first_seen, row.Index),
                last_seen_time=_series_timestamp(last_seen, row.Index),
                price_sum=float(row.price_sum),
                valid_price_count=int(row.valid_price_count),
            )
            if cross_chunk_reversal:
                time_reversal_event_count += 1

    session_values = list(session_stats.values())
    session_lengths = [stats.event_count for stats in session_values]
    threshold = _extreme_session_threshold(session_lengths, extreme_session_min_length)
    extreme_sessions = _build_extreme_session_rows(
        session_stats,
        threshold,
        top_extreme_sessions,
    )

    purchase_sessions = [stats for stats in session_values if stats.has_purchase]
    non_purchase_sessions = [stats for stats in session_values if not stats.has_purchase]
    durations = [
        stats.duration_minutes
        for stats in session_values
        if stats.duration_minutes is not None
    ]

    return {
        "session_count": len(session_values),
        "purchase_session_count": len(purchase_sessions),
        "non_purchase_session_count": len(non_purchase_sessions),
        "time_reversal_event_count": time_reversal_event_count,
        "time_reversal_session_count": sum(
            1 for stats in session_values if stats.has_time_reversal
        ),
        "extreme_session_threshold": threshold,
        "extreme_session_count": sum(1 for length in session_lengths if length >= threshold),
        "max_session_length": max(session_lengths, default=0),
        "max_session_duration_minutes": max(durations, default=None),
        "session_comparison": {
            "purchase_session": _summarize_sessions(purchase_sessions),
            "non_purchase_session": _summarize_sessions(non_purchase_sessions),
        },
        "extreme_sessions": extreme_sessions,
    }


def build_quality_summary_rows(result: DataQualityResult) -> list[dict[str, str]]:
    """체크리스트형 데이터 신뢰성 검증 요약표를 생성한다."""

    checks = [
        _check_row(
            "schema_required_columns",
            "필수 컬럼 존재 여부",
            not result.missing_columns,
            ", ".join(result.missing_columns) if result.missing_columns else "누락 없음",
            "누락 시 원천 데이터 또는 컬럼 매핑을 수정한다.",
        ),
        _check_row(
            "event_time_parse",
            "event_time 파싱 가능 여부",
            result.event_time_parse_failures == 0,
            f"{result.event_time_parse_failures:,} rows",
            "파싱 실패 row를 원천 값 패턴별로 확인한다.",
        ),
        _check_row(
            "event_type_allowed_values",
            "event_type 허용 값 검증",
            not result.unexpected_event_types,
            ", ".join(result.unexpected_event_types)
            if result.unexpected_event_types
            else "허용 값만 존재",
            "`view`, `cart`, `remove_from_cart`, `purchase` 외 값은 정책을 결정한다.",
        ),
        _check_row(
            "price_positive",
            "price <= 0 이상치 검증",
            result.invalid_price_count == 0,
            f"{result.invalid_price_count:,} rows ({_format_ratio(result.invalid_price_ratio)})",
            "무료/오류 가격 여부를 확인하고 제외 또는 보정 정책을 정한다.",
        ),
        _check_row(
            "duplicate_rows",
            "완전 중복 row 비율",
            result.duplicate_row_count == 0,
            f"{result.duplicate_row_count:,} rows ({_format_ratio(result.duplicate_row_ratio)})",
            "중복 이벤트가 식별되면 학습 전 제거 여부를 결정한다.",
        ),
        _check_row(
            "session_time_order",
            "세션 내 시간 역전 여부",
            result.time_reversal_event_count == 0,
            f"{result.time_reversal_event_count:,} events, "
            f"{result.time_reversal_session_count:,} sessions",
            "라벨링 전 세션 내 event_time 정렬을 강제한다.",
        ),
        _check_row(
            "extreme_session_length",
            "극단적으로 긴 세션 탐색",
            result.extreme_session_count == 0,
            f">= {result.extreme_session_threshold:,} events: "
            f"{result.extreme_session_count:,} sessions",
            "상위 세션을 확인해 bot성 행동 또는 장기 세션 처리 정책을 정한다.",
        ),
    ]

    for column in result.columns:
        missing_count = result.missing_counts.get(column, 0)
        checks.append(
            _check_row(
                f"missing_{column}",
                f"{column} 결측률",
                missing_count == 0,
                f"{missing_count:,} rows "
                f"({_format_ratio(missing_count / result.row_count if result.row_count else 0.0)})",
                "결측률이 높은 feature는 unknown 처리 또는 제외 정책을 정한다.",
            )
        )

    return checks


def build_missing_value_rows(result: DataQualityResult) -> list[dict[str, str]]:
    return [
        {
            "column": column,
            "missing_count": str(result.missing_counts.get(column, 0)),
            "missing_ratio": _format_ratio(
                result.missing_counts.get(column, 0) / result.row_count
                if result.row_count
                else 0.0
            ),
        }
        for column in result.columns
    ]


def build_session_comparison_rows(result: DataQualityResult) -> list[dict[str, str]]:
    rows = []
    for segment, metrics in result.session_comparison.items():
        for metric, value in metrics.items():
            rows.append(
                {
                    "segment": segment,
                    "metric": metric,
                    "value": str(value),
                }
            )
    return rows


def write_quality_artifacts(result: DataQualityResult, reports_dir: Path) -> None:
    """검증 결과를 리포트/대시보드용 artifact로 저장한다."""

    output_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(output_dir / "data_quality_summary.csv", build_quality_summary_rows(result))
    _write_csv(
        output_dir / "data_quality_missing_values.csv",
        build_missing_value_rows(result),
    )
    _write_csv(
        output_dir / "data_quality_session_comparison.csv",
        build_session_comparison_rows(result),
    )
    _write_csv(output_dir / "data_quality_extreme_sessions.csv", result.extreme_sessions)
    (output_dir / "data_quality_report.md").write_text(
        build_quality_markdown_report(result),
        encoding="utf-8",
    )
    (output_dir / "data_quality_test_candidates.md").write_text(
        build_test_candidate_markdown(),
        encoding="utf-8",
    )


def build_quality_markdown_report(result: DataQualityResult) -> str:
    lines = [
        "# Step 2 데이터 신뢰성 검증",
        "",
        "## 핵심 결과",
        "",
        f"- 필수 컬럼 누락: {', '.join(result.missing_columns) if result.missing_columns else '없음'}",
        f"- event_time 파싱 실패: {result.event_time_parse_failures:,} rows",
        f"- 허용 범위 밖 event_type: {', '.join(result.unexpected_event_types) if result.unexpected_event_types else '없음'}",
        f"- `price <= 0`: {result.invalid_price_count:,} rows ({_format_ratio(result.invalid_price_ratio)})",
        f"- 완전 중복 row: {result.duplicate_row_count:,} rows ({_format_ratio(result.duplicate_row_ratio)})",
        f"- 세션 내 시간 역전: {result.time_reversal_event_count:,} events, {result.time_reversal_session_count:,} sessions",
        f"- 극단 세션 기준: {result.extreme_session_threshold:,} events 이상",
        f"- 극단 세션 수: {result.extreme_session_count:,}",
        f"- 최대 세션 길이: {result.max_session_length:,} events",
        "",
        "## 모델링 전 결정 필요 사항",
        "",
        "- `brand`, `category_code` 결측은 unknown category 처리 여부를 Step 5에서 확정한다.",
        "- `user_session` 결측 row는 세션 기반 라벨링 대상에서 제외하는 정책을 우선 검토한다.",
        "- 중복 row가 존재하면 라벨링 전 제거 여부와 제거 기준을 문서화한다.",
        "- 시간 역전 세션은 prefix 생성 전에 `event_time` 기준 정렬을 강제한다.",
        "- 극단적으로 긴 세션은 bot성 행동 또는 장기 세션 여부를 샘플링 검토한다.",
        "",
    ]
    return "\n".join(lines)


def build_test_candidate_markdown() -> str:
    return "\n".join(
        [
            "# 데이터 검증 로직 테스트 후보",
            "",
            "- 필수 컬럼이 누락되면 `schema_required_columns`가 실패해야 한다.",
            "- 허용되지 않은 `event_type`이 있으면 unexpected event로 기록해야 한다.",
            "- 파싱 불가능한 `event_time`은 실패 건수에 포함해야 한다.",
            "- `price <= 0`은 이상 가격 건수에 포함해야 한다.",
            "- 완전 동일 row가 반복되면 중복 row로 계산해야 한다.",
            "- 같은 `user_session` 내 관측 순서상 시간이 감소하면 시간 역전으로 기록해야 한다.",
            "- purchase 포함 세션과 비purchase 세션의 기본 통계는 분리 산출해야 한다.",
            "- 극단 세션 기준 이상인 세션은 `data_quality_extreme_sessions.csv`에 포함해야 한다.",
            "",
        ]
    )


def _summarize_sessions(
    sessions: list[SessionStats],
) -> dict[str, float | int | str]:
    event_counts = [stats.event_count for stats in sessions]
    durations = [
        stats.duration_minutes for stats in sessions if stats.duration_minutes is not None
    ]
    avg_prices = [stats.avg_price for stats in sessions if stats.avg_price is not None]
    return {
        "session_count": len(sessions),
        "avg_event_count": _mean(event_counts),
        "median_event_count": _median(event_counts),
        "avg_duration_minutes": _mean(durations),
        "median_duration_minutes": _median(durations),
        "avg_price_per_event": _mean(avg_prices),
    }


def _build_extreme_session_rows(
    session_stats: dict[str, SessionStats],
    threshold: int,
    top_n: int,
) -> list[dict[str, str]]:
    candidates = [
        (session_id, stats)
        for session_id, stats in session_stats.items()
        if stats.event_count >= threshold
    ]
    candidates.sort(key=lambda item: item[1].event_count, reverse=True)
    rows = []
    for session_id, stats in candidates[:top_n]:
        rows.append(
            {
                "user_session": session_id,
                "event_count": str(stats.event_count),
                "purchase_count": str(stats.purchase_count),
                "duration_minutes": _format_optional_float(stats.duration_minutes),
                "has_time_reversal": str(stats.has_time_reversal),
            }
        )
    return rows


def _extreme_session_threshold(
    session_lengths: list[int],
    extreme_session_min_length: int,
) -> int:
    return extreme_session_min_length


def _series_timestamp(series: pd.Series, key: object) -> pd.Timestamp | None:
    if key not in series.index:
        return None
    value = series.loc[key]
    if pd.isna(value):
        return None
    return value


def _check_row(
    check_id: str,
    check_name: str,
    passed: bool,
    evidence: str,
    next_action: str,
) -> dict[str, str]:
    return {
        "check_id": check_id,
        "check_name": check_name,
        "status": "pass" if passed else "review",
        "evidence": evidence,
        "next_action": next_action,
    }


def _write_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    pd.DataFrame(list(rows)).to_csv(path, index=False, encoding="utf-8")


def _mean(values: list[float | int]) -> float | str:
    if not values:
        return ""
    return round(float(pd.Series(values).mean()), 6)


def _median(values: list[float | int]) -> float | str:
    if not values:
        return ""
    return round(float(pd.Series(values).median()), 6)


def _format_ratio(value: float) -> str:
    return f"{value:.6f}"


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"
