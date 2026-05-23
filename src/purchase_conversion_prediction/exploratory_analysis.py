"""Step 4 EDA 및 문제 타당성 검증 유스케이스."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from purchase_conversion_prediction.data_profiling import EVENT_TIME_FORMAT
from purchase_conversion_prediction.labeling import LABEL_WINDOW_MINUTES, LabelingPolicy


PRICE_BINS = (-float("inf"), 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, float("inf"))
PRICE_BAND_LABELS = ("<=25", "25-50", "50-100", "100-250", "250-500", "500-1000", ">1000")
SESSION_LENGTH_BANDS = (
    (1, 1, "1"),
    (2, 2, "2"),
    (3, 5, "3-5"),
    (6, 10, "6-10"),
    (11, 20, "11-20"),
    (21, 50, "21-50"),
    (51, 100, "51-100"),
    (101, None, "101+"),
)


@dataclass
class _SessionAggregate:
    event_count: int = 0
    purchase_count: int = 0
    min_time: pd.Timestamp | None = None
    max_time: pd.Timestamp | None = None
    price_sum: float = 0.0
    valid_price_count: int = 0

    def update(
        self,
        event_count: int,
        purchase_count: int,
        min_time: pd.Timestamp | None,
        max_time: pd.Timestamp | None,
        price_sum: float,
        valid_price_count: int,
    ) -> None:
        self.event_count += int(event_count)
        self.purchase_count += int(purchase_count)
        self.price_sum += float(price_sum)
        self.valid_price_count += int(valid_price_count)
        if min_time is not None and (self.min_time is None or min_time < self.min_time):
            self.min_time = min_time
        if max_time is not None and (self.max_time is None or max_time > self.max_time):
            self.max_time = max_time

    @property
    def has_purchase(self) -> bool:
        return self.purchase_count > 0


@dataclass
class _LabelAggregate:
    sample_count: int = 0
    positive_count: int = 0
    price_sum: float = 0.0
    valid_price_count: int = 0

    def update(
        self,
        sample_count: int,
        positive_count: int,
        price_sum: float = 0.0,
        valid_price_count: int = 0,
    ) -> None:
        self.sample_count += int(sample_count)
        self.positive_count += int(positive_count)
        self.price_sum += float(price_sum)
        self.valid_price_count += int(valid_price_count)

    @property
    def positive_rate(self) -> float:
        if self.sample_count == 0:
            return 0.0
        return self.positive_count / self.sample_count

    @property
    def avg_price(self) -> float | None:
        if self.valid_price_count == 0:
            return None
        return self.price_sum / self.valid_price_count


@dataclass
class _SampleSegmentAggregate:
    sample_count: int = 0
    price_sum: float = 0.0
    valid_price_count: int = 0
    event_type_counts: dict[str, int] = field(default_factory=dict)

    def update(self, events: pd.DataFrame) -> None:
        self.sample_count += len(events)
        price = pd.to_numeric(events["price"], errors="coerce")
        valid_price = price.dropna()
        self.price_sum += float(valid_price.sum())
        self.valid_price_count += int(valid_price.count())
        for event_type, count in events["event_type"].astype(str).value_counts().items():
            self.event_type_counts[event_type] = self.event_type_counts.get(event_type, 0) + int(count)

    @property
    def avg_price(self) -> float | None:
        if self.valid_price_count == 0:
            return None
        return self.price_sum / self.valid_price_count


@dataclass(frozen=True)
class EDAResult:
    """Step 4 EDA artifact 생성을 위한 집계 결과."""

    source_path: Path
    prediction_window_minutes: int
    overview_rows: list[dict[str, str]]
    session_length_purchase_rate_rows: list[dict[str, str]]
    sequence_pattern_rows: list[dict[str, str]]
    price_band_purchase_rate_rows: list[dict[str, str]]
    category_conversion_rows: list[dict[str, str]]
    hourly_purchase_rate_rows: list[dict[str, str]]
    sample_comparison_rows: list[dict[str, str]]


def analyze_problem_validity_from_csv(
    csv_path: Path,
    policy: LabelingPolicy | None = None,
    chunksize: int = 1_000_000,
    max_pattern_length: int = 5,
    top_n: int = 20,
    max_rows: int | None = None,
) -> EDAResult:
    """라벨링된 구매 예측 문제의 EDA 집계를 생성한다.

    전체 prefix sample을 파일로 저장하지 않고, 첫 purchase 시각을 기준으로 라벨 집계를
    계산한다. sequence pattern은 기준 시점 이후 이벤트가 섞이지 않도록 첫 purchase 전
    이벤트 또는 비구매 세션의 관측 이벤트만 사용한다.
    """

    active_policy = policy or LabelingPolicy(prediction_window_minutes=LABEL_WINDOW_MINUTES)
    path = Path(csv_path)
    first_pass = _collect_session_and_category_stats(path, chunksize, max_rows)
    second_pass = _collect_label_based_stats(
        path=path,
        chunksize=chunksize,
        policy=active_policy,
        first_purchase_times=first_pass["first_purchase_times"],
        sessions=first_pass["sessions"],
        max_pattern_length=max_pattern_length,
        max_rows=max_rows,
    )

    overview_rows = _build_overview_rows(
        path,
        active_policy,
        first_pass["analyzed_row_count"],
        max_rows,
        first_pass["sessions"],
        second_pass["sample_segments"],
    )
    return EDAResult(
        source_path=path,
        prediction_window_minutes=active_policy.prediction_window_minutes,
        overview_rows=overview_rows,
        session_length_purchase_rate_rows=_build_session_length_rows(first_pass["sessions"]),
        sequence_pattern_rows=_build_sequence_pattern_rows(
            second_pass["session_patterns"],
            first_pass["sessions"],
            top_n,
        ),
        price_band_purchase_rate_rows=_build_label_aggregate_rows(
            second_pass["price_band_aggregates"],
            "price_band",
        ),
        category_conversion_rows=_build_category_conversion_rows(
            first_pass["category_event_counts"],
            top_n,
        ),
        hourly_purchase_rate_rows=_build_label_aggregate_rows(
            second_pass["hourly_aggregates"],
            "hour",
        ),
        sample_comparison_rows=_build_sample_comparison_rows(
            second_pass["sample_segments"],
        ),
    )


def write_eda_artifacts(result: EDAResult, reports_dir: Path) -> None:
    """EDA 결과를 리포트/대시보드용 artifact로 저장한다."""

    output_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(output_dir / "eda_problem_validity_summary.csv", result.overview_rows)
    _write_csv(
        output_dir / "eda_session_length_purchase_rate.csv",
        result.session_length_purchase_rate_rows,
    )
    _write_csv(
        output_dir / "eda_sequence_pattern_purchase_rate.csv",
        result.sequence_pattern_rows,
    )
    _write_csv(
        output_dir / "eda_price_band_purchase_rate.csv",
        result.price_band_purchase_rate_rows,
    )
    _write_csv(output_dir / "eda_category_conversion.csv", result.category_conversion_rows)
    _write_csv(
        output_dir / "eda_hourly_purchase_rate.csv",
        result.hourly_purchase_rate_rows,
    )
    _write_csv(
        output_dir / "eda_positive_negative_sample_comparison.csv",
        result.sample_comparison_rows,
    )
    (output_dir / "eda_report.md").write_text(
        build_eda_markdown_report(result),
        encoding="utf-8",
    )


def build_eda_markdown_report(result: EDAResult) -> str:
    """Step 4 리포트 초안을 생성한다."""

    overview = {row["metric"]: row["value"] for row in result.overview_rows}
    top_pattern = result.sequence_pattern_rows[0] if result.sequence_pattern_rows else {}
    lines = [
        "# Step 4 EDA 및 문제 타당성 검증",
        "",
        "## 핵심 요약",
        "",
        f"- 대상 파일: `{result.source_path}`",
        f"- 예측 window: {result.prediction_window_minutes}분",
        f"- 분석 row 수: {overview.get('analyzed_row_count', '0')}",
        f"- 세션 수: {overview.get('session_count', '0')}",
        f"- 라벨링 sample 수: {overview.get('labeled_sample_count', '0')}",
        f"- positive sample 비율: {overview.get('positive_sample_ratio', '0.000000')}",
        f"- 최상위 sequence pattern: {top_pattern.get('sequence_pattern', '')}",
        "",
        "## 리포트용 chart 후보",
        "",
        "- `eda_session_length_purchase_rate.csv`: 세션 길이별 구매율",
        "- `eda_sequence_pattern_purchase_rate.csv`: 초기 event sequence pattern별 구매율",
        "- `eda_price_band_purchase_rate.csv`: 가격대별 30분 내 구매율",
        "- `eda_category_conversion.csv`: category별 purchase/view 전환율",
        "- `eda_hourly_purchase_rate.csv`: 시간대별 30분 내 구매율",
        "- `eda_positive_negative_sample_comparison.csv`: positive/negative sample 차이",
        "",
        "## 문제 타당성 판단 포인트",
        "",
        "- positive 비율과 가격대/시간대/행동 pattern별 차이가 baseline 모델의 학습 신호 후보이다.",
        "- category 전환율은 descriptive EDA이며, 모델 feature에는 기준 시점 이후 정보가 들어가지 않도록 Step 5에서 별도 검증한다.",
        "- sequence pattern 집계는 첫 purchase 전 이벤트만 사용해 사후 구매 이벤트가 pattern에 섞이지 않도록 제한했다.",
        "",
    ]
    return "\n".join(lines)


def _collect_session_and_category_stats(
    path: Path,
    chunksize: int,
    max_rows: int | None,
) -> dict[str, object]:
    sessions: dict[str, _SessionAggregate] = {}
    first_purchase_times: dict[str, pd.Timestamp] = {}
    category_event_counts: dict[str, dict[str, int]] = {}
    analyzed_row_count = 0

    for chunk in _read_event_chunks(path, chunksize, max_rows):
        analyzed_row_count += len(chunk)
        working = _prepare_events(chunk)
        if working.empty:
            continue

        grouped = working.groupby("user_session", sort=False).agg(
            event_count=("event_type", "size"),
            purchase_count=("_is_purchase", "sum"),
            min_time=("_event_time", "min"),
            max_time=("_event_time", "max"),
            price_sum=("_price", "sum"),
            valid_price_count=("_price", "count"),
        )
        for row in grouped.itertuples():
            session_id = str(row.Index)
            stats = sessions.setdefault(session_id, _SessionAggregate())
            stats.update(
                event_count=int(row.event_count),
                purchase_count=int(row.purchase_count),
                min_time=_optional_timestamp(row.min_time),
                max_time=_optional_timestamp(row.max_time),
                price_sum=float(row.price_sum),
                valid_price_count=int(row.valid_price_count),
            )

        purchase_events = working.loc[working["_is_purchase"].eq(1)]
        if not purchase_events.empty:
            purchase_min = purchase_events.groupby("user_session")["_event_time"].min()
            for session_value, event_time in purchase_min.items():
                session_id = str(session_value)
                current = first_purchase_times.get(session_id)
                if current is None or event_time < current:
                    first_purchase_times[session_id] = event_time

        _update_category_counts(category_event_counts, working)

    return {
        "sessions": sessions,
        "first_purchase_times": first_purchase_times,
        "category_event_counts": category_event_counts,
        "analyzed_row_count": analyzed_row_count,
    }


def _collect_label_based_stats(
    path: Path,
    chunksize: int,
    policy: LabelingPolicy,
    first_purchase_times: dict[str, pd.Timestamp],
    sessions: dict[str, _SessionAggregate],
    max_pattern_length: int,
    max_rows: int | None,
) -> dict[str, object]:
    window = pd.Timedelta(minutes=policy.prediction_window_minutes)
    session_patterns: dict[str, list[str]] = {}
    price_band_aggregates: dict[str, _LabelAggregate] = {
        label: _LabelAggregate() for label in PRICE_BAND_LABELS
    }
    price_band_aggregates["<missing>"] = _LabelAggregate()
    hourly_aggregates: dict[str, _LabelAggregate] = {
        str(hour): _LabelAggregate() for hour in range(24)
    }
    sample_segments = {
        "positive": _SampleSegmentAggregate(),
        "negative": _SampleSegmentAggregate(),
    }

    for chunk in _read_event_chunks(path, chunksize, max_rows):
        working = _prepare_events(chunk)
        if working.empty:
            continue

        first_purchase_series = pd.to_datetime(
            working["user_session"].astype(str).map(first_purchase_times),
            errors="coerce",
            utc=True,
        )
        has_purchase = first_purchase_series.notna()
        before_first_purchase = ~has_purchase | working["_event_time"].lt(first_purchase_series)
        positive = (
            before_first_purchase
            & has_purchase
            & first_purchase_series.le(working["_event_time"] + window)
        )

        labeled = working.loc[before_first_purchase].copy()
        if labeled.empty:
            continue
        labeled["_label"] = positive.loc[labeled.index].astype("int64")
        labeled["_price_band"] = _price_bands(labeled["_price"])
        labeled["_hour"] = labeled["_event_time"].dt.hour.astype(str)

        _update_session_patterns(session_patterns, labeled, sessions, max_pattern_length)
        _update_label_aggregates(price_band_aggregates, labeled, "_price_band")
        _update_label_aggregates(hourly_aggregates, labeled, "_hour")

        for segment, value in (("positive", 1), ("negative", 0)):
            segment_events = labeled.loc[labeled["_label"].eq(value)]
            if not segment_events.empty:
                sample_segments[segment].update(segment_events)

    return {
        "session_patterns": session_patterns,
        "price_band_aggregates": price_band_aggregates,
        "hourly_aggregates": hourly_aggregates,
        "sample_segments": sample_segments,
    }


def _read_event_chunks(
    path: Path,
    chunksize: int,
    max_rows: int | None,
) -> Iterable[pd.DataFrame]:
    usecols = [
        "event_time",
        "event_type",
        "price",
        "user_session",
        "category_code",
    ]
    remaining = max_rows
    for chunk in pd.read_csv(path, usecols=lambda column: column in usecols, chunksize=chunksize):
        if remaining is not None:
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.head(remaining)
            remaining -= len(chunk)
        yield chunk


def _prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    required = {"event_time", "event_type", "user_session"}
    if not required.issubset(events.columns):
        missing = ", ".join(sorted(required - set(events.columns)))
        raise ValueError(f"EDA 필수 컬럼 누락: {missing}")

    working = events.copy()
    if "price" not in working:
        working["price"] = pd.NA
    if "category_code" not in working:
        working["category_code"] = pd.NA

    working["_event_time"] = pd.to_datetime(
        working["event_time"],
        format=EVENT_TIME_FORMAT,
        errors="coerce",
        utc=True,
    )
    working = working.loc[
        working["user_session"].notna() & working["_event_time"].notna()
    ].copy()
    if working.empty:
        return working

    working["user_session"] = working["user_session"].astype(str)
    working["event_type"] = working["event_type"].astype(str)
    working["_price"] = pd.to_numeric(working["price"], errors="coerce")
    working["_is_purchase"] = working["event_type"].eq("purchase").astype("int64")
    return working


def _update_category_counts(
    category_event_counts: dict[str, dict[str, int]],
    events: pd.DataFrame,
) -> None:
    category = _top_level_category(events["category_code"])
    grouped = events.assign(_category=category).groupby(["_category", "event_type"]).size()
    for (category_name, event_type), count in grouped.items():
        counts = category_event_counts.setdefault(str(category_name), {})
        counts[str(event_type)] = counts.get(str(event_type), 0) + int(count)


def _update_session_patterns(
    session_patterns: dict[str, list[str]],
    events: pd.DataFrame,
    sessions: dict[str, _SessionAggregate],
    max_pattern_length: int,
) -> None:
    if max_pattern_length <= 0:
        return

    for session_id, group in events.groupby("user_session", sort=False):
        if str(session_id) not in sessions:
            continue
        pattern = session_patterns.setdefault(str(session_id), [])
        if len(pattern) >= max_pattern_length:
            continue
        remaining = max_pattern_length - len(pattern)
        pattern.extend(group["event_type"].astype(str).head(remaining).tolist())


def _update_label_aggregates(
    aggregates: dict[str, _LabelAggregate],
    events: pd.DataFrame,
    group_column: str,
) -> None:
    grouped = events.groupby(group_column, sort=False).agg(
        sample_count=("event_type", "size"),
        positive_count=("_label", "sum"),
        price_sum=("_price", "sum"),
        valid_price_count=("_price", "count"),
    )
    for row in grouped.itertuples():
        key = str(row.Index)
        aggregate = aggregates.setdefault(key, _LabelAggregate())
        aggregate.update(
            sample_count=int(row.sample_count),
            positive_count=int(row.positive_count),
            price_sum=float(row.price_sum),
            valid_price_count=int(row.valid_price_count),
        )


def _build_overview_rows(
    path: Path,
    policy: LabelingPolicy,
    analyzed_row_count: int,
    max_rows: int | None,
    sessions: dict[str, _SessionAggregate],
    sample_segments: dict[str, _SampleSegmentAggregate],
) -> list[dict[str, str]]:
    positive_count = sample_segments["positive"].sample_count
    negative_count = sample_segments["negative"].sample_count
    sample_count = positive_count + negative_count
    purchase_session_count = sum(1 for stats in sessions.values() if stats.has_purchase)
    return [
        _metric_row("source_path", str(path)),
        _metric_row("prediction_window_minutes", str(policy.prediction_window_minutes)),
        _metric_row("analyzed_row_count", str(analyzed_row_count)),
        _metric_row("max_rows", str(max_rows) if max_rows is not None else ""),
        _metric_row("session_count", str(len(sessions))),
        _metric_row("purchase_session_count", str(purchase_session_count)),
        _metric_row("labeled_sample_count", str(sample_count)),
        _metric_row("positive_sample_count", str(positive_count)),
        _metric_row("negative_sample_count", str(negative_count)),
        _metric_row(
            "positive_sample_ratio",
            _format_ratio(positive_count / sample_count if sample_count else 0.0),
        ),
    ]


def _build_session_length_rows(
    sessions: dict[str, _SessionAggregate],
) -> list[dict[str, str]]:
    band_counts = {
        label: {"session_count": 0, "purchase_session_count": 0}
        for _, _, label in SESSION_LENGTH_BANDS
    }
    for stats in sessions.values():
        label = _session_length_band(stats.event_count)
        band_counts[label]["session_count"] += 1
        if stats.has_purchase:
            band_counts[label]["purchase_session_count"] += 1

    rows = []
    for _, _, label in SESSION_LENGTH_BANDS:
        counts = band_counts[label]
        session_count = counts["session_count"]
        purchase_count = counts["purchase_session_count"]
        rows.append(
            {
                "session_length_band": label,
                "session_count": str(session_count),
                "purchase_session_count": str(purchase_count),
                "purchase_rate": _format_ratio(
                    purchase_count / session_count if session_count else 0.0
                ),
            }
        )
    return rows


def _build_sequence_pattern_rows(
    session_patterns: dict[str, list[str]],
    sessions: dict[str, _SessionAggregate],
    top_n: int,
) -> list[dict[str, str]]:
    pattern_counts: dict[str, dict[str, int]] = {}
    for session_id, pattern_values in session_patterns.items():
        pattern = " > ".join(pattern_values) if pattern_values else "<empty>"
        counts = pattern_counts.setdefault(
            pattern,
            {"session_count": 0, "purchase_session_count": 0},
        )
        counts["session_count"] += 1
        if sessions.get(session_id, _SessionAggregate()).has_purchase:
            counts["purchase_session_count"] += 1

    rows = []
    for pattern, counts in sorted(
        pattern_counts.items(),
        key=lambda item: item[1]["session_count"],
        reverse=True,
    )[:top_n]:
        session_count = counts["session_count"]
        purchase_count = counts["purchase_session_count"]
        rows.append(
            {
                "sequence_pattern": pattern,
                "session_count": str(session_count),
                "purchase_session_count": str(purchase_count),
                "purchase_rate": _format_ratio(
                    purchase_count / session_count if session_count else 0.0
                ),
            }
        )
    return rows


def _build_label_aggregate_rows(
    aggregates: dict[str, _LabelAggregate],
    key_name: str,
) -> list[dict[str, str]]:
    rows = []
    for key, aggregate in aggregates.items():
        rows.append(
            {
                key_name: str(key),
                "sample_count": str(aggregate.sample_count),
                "positive_count": str(aggregate.positive_count),
                "positive_rate": _format_ratio(aggregate.positive_rate),
                "avg_price": _format_optional_float(aggregate.avg_price),
            }
        )

    return sorted(rows, key=lambda row: _sort_key(row[key_name]))


def _build_category_conversion_rows(
    category_event_counts: dict[str, dict[str, int]],
    top_n: int,
) -> list[dict[str, str]]:
    rows = []
    for category, counts in category_event_counts.items():
        total_events = sum(counts.values())
        view_events = counts.get("view", 0)
        purchase_events = counts.get("purchase", 0)
        rows.append(
            {
                "category": category,
                "total_events": str(total_events),
                "view_events": str(view_events),
                "purchase_events": str(purchase_events),
                "purchase_to_view_rate": _format_ratio(
                    purchase_events / view_events if view_events else 0.0
                ),
            }
        )
    rows.sort(key=lambda row: int(row["total_events"]), reverse=True)
    return rows[:top_n]


def _build_sample_comparison_rows(
    sample_segments: dict[str, _SampleSegmentAggregate],
) -> list[dict[str, str]]:
    rows = []
    for segment in ("positive", "negative"):
        aggregate = sample_segments[segment]
        row = {
            "label_segment": segment,
            "sample_count": str(aggregate.sample_count),
            "avg_price": _format_optional_float(aggregate.avg_price),
        }
        for event_type in ("view", "cart", "remove_from_cart", "purchase"):
            count = aggregate.event_type_counts.get(event_type, 0)
            row[f"{event_type}_ratio"] = _format_ratio(
                count / aggregate.sample_count if aggregate.sample_count else 0.0
            )
        rows.append(row)
    return rows


def _top_level_category(category_code: pd.Series) -> pd.Series:
    category = category_code.fillna("<missing>").astype(str)
    category = category.mask(category.eq("") | category.eq("nan"), "<missing>")
    return category.str.split(".", n=1).str[0]


def _price_bands(price: pd.Series) -> pd.Series:
    band = pd.Series("<missing>", index=price.index, dtype="object")
    valid_price = price.dropna()
    if valid_price.empty:
        return band
    band.loc[valid_price.index] = pd.cut(
        valid_price,
        bins=PRICE_BINS,
        labels=PRICE_BAND_LABELS,
        include_lowest=True,
        right=True,
    ).astype(str)
    return band


def _session_length_band(event_count: int) -> str:
    for lower, upper, label in SESSION_LENGTH_BANDS:
        if upper is None and event_count >= lower:
            return label
        if upper is not None and lower <= event_count <= upper:
            return label
    return "101+"


def _metric_row(metric: str, value: str) -> dict[str, str]:
    return {"metric": metric, "value": value}


def _write_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    pd.DataFrame(list(rows)).to_csv(path, index=False, encoding="utf-8")


def _optional_timestamp(value: object) -> pd.Timestamp | None:
    if pd.isna(value):
        return None
    return value


def _format_ratio(value: float) -> str:
    return f"{value:.6f}"


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _sort_key(value: str) -> tuple[int, object]:
    if value == "<missing>":
        return (1, value)
    try:
        return (0, int(value))
    except ValueError:
        return (0, value)
