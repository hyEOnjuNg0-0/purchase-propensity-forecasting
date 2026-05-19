"""Step 3 구매 예측 라벨링 유스케이스."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from purchase_time_forecasting.data_profiling import EVENT_TIME_FORMAT


LABEL_WINDOW_MINUTES = 30
LABEL_EXCLUDED_REASON_AT_OR_AFTER_FIRST_PURCHASE = "at_or_after_first_purchase"
LABEL_EXCLUDED_REASON_INVALID_TIME = "invalid_event_time"
LABEL_EXCLUDED_REASON_MISSING_SESSION = "missing_user_session"


@dataclass(frozen=True)
class LabelingPolicy:
    """prefix 라벨 생성 정책."""

    prediction_window_minutes: int = LABEL_WINDOW_MINUTES
    exclude_at_or_after_first_purchase: bool = True
    include_current_event_in_prefix: bool = True


@dataclass(frozen=True)
class LabelDistribution:
    """Step 3 라벨 분포 요약."""

    source_path: Path | None
    prediction_window_minutes: int
    candidate_event_count: int
    labeled_sample_count: int
    positive_count: int
    negative_count: int
    excluded_missing_session_count: int
    excluded_invalid_time_count: int
    excluded_at_or_after_first_purchase_count: int
    session_count: int
    purchase_session_count: int

    @property
    def positive_ratio(self) -> float:
        if self.labeled_sample_count == 0:
            return 0.0
        return self.positive_count / self.labeled_sample_count

    @property
    def negative_ratio(self) -> float:
        if self.labeled_sample_count == 0:
            return 0.0
        return self.negative_count / self.labeled_sample_count


def create_prefix_labels(
    events: pd.DataFrame,
    policy: LabelingPolicy | None = None,
) -> pd.DataFrame:
    """세션 내 prefix sequence와 향후 구매 라벨을 생성한다.

    feature 후보인 prefix 컬럼은 기준 시점까지의 이벤트만 사용한다. 라벨은 같은
    `user_session`에서 기준 시점 이후 window 이내에 발생하는 첫 purchase로 계산한다.
    """

    active_policy = policy or LabelingPolicy()
    required_columns = {"event_time", "event_type", "user_session"}
    missing_columns = required_columns - set(events.columns)
    if missing_columns:
        raise ValueError(f"라벨링 필수 컬럼 누락: {', '.join(sorted(missing_columns))}")

    if events.empty:
        return _empty_label_frame()

    working = events.copy()
    working["_source_order"] = range(len(working))
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
        return _empty_label_frame()

    working["user_session"] = working["user_session"].astype(str)
    working["event_type"] = working["event_type"].astype(str)
    working = working.sort_values(
        ["user_session", "_event_time", "_source_order"],
        kind="mergesort",
    )

    rows: list[dict[str, object]] = []
    window = pd.Timedelta(minutes=active_policy.prediction_window_minutes)
    for session_id, session_events in working.groupby("user_session", sort=False):
        event_times = session_events["_event_time"].tolist()
        event_types = session_events["event_type"].tolist()
        purchase_times = [
            event_time
            for event_time, event_type in zip(event_times, event_types)
            if event_type == "purchase"
        ]
        first_purchase_time = min(purchase_times) if purchase_times else None
        prefix: list[str] = []

        for position, (_, row) in enumerate(session_events.iterrows(), start=1):
            cutoff_time = row["_event_time"]
            event_type = str(row["event_type"])
            if active_policy.include_current_event_in_prefix:
                prefix.append(event_type)

            if (
                active_policy.exclude_at_or_after_first_purchase
                and first_purchase_time is not None
                and cutoff_time >= first_purchase_time
            ):
                continue

            future_purchases = [
                purchase_time
                for purchase_time in purchase_times
                if cutoff_time < purchase_time <= cutoff_time + window
            ]
            next_purchase_time = min(future_purchases) if future_purchases else None
            label = int(next_purchase_time is not None)
            rows.append(
                {
                    "user_session": session_id,
                    "cutoff_time": cutoff_time.isoformat(),
                    "session_position": position,
                    "prefix_length": len(prefix),
                    "prefix_event_types": " ".join(prefix),
                    "last_event_type": event_type,
                    "label": label,
                    "minutes_until_purchase": _minutes_between(
                        cutoff_time,
                        next_purchase_time,
                    ),
                }
            )

            if not active_policy.include_current_event_in_prefix:
                prefix.append(event_type)

    if not rows:
        return _empty_label_frame()
    return pd.DataFrame(rows)


def summarize_label_distribution(
    events: pd.DataFrame,
    policy: LabelingPolicy | None = None,
    source_path: Path | None = None,
) -> LabelDistribution:
    """DataFrame 기준 라벨 분포를 계산한다."""

    active_policy = policy or LabelingPolicy()
    stats = _summarize_frame(events, active_policy)
    return _stats_to_distribution(stats, active_policy, source_path)


def summarize_label_distribution_from_csv(
    csv_path: Path,
    policy: LabelingPolicy | None = None,
    chunksize: int = 1_000_000,
) -> LabelDistribution:
    """원천 CSV를 두 번 읽어 전체 라벨 분포를 계산한다.

    첫 번째 pass는 세션별 첫 purchase 시각을 수집하고, 두 번째 pass는 각 이벤트가
    라벨링 sample에 포함되는지와 positive 여부를 계산한다.
    """

    active_policy = policy or LabelingPolicy()
    path = Path(csv_path)
    first_purchase_times = _collect_first_purchase_times(path, chunksize)
    stats = _empty_stats()
    stats["purchase_session_ids"].update(first_purchase_times)

    usecols = ["event_time", "event_type", "user_session"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        _update_distribution_stats(chunk, active_policy, stats, first_purchase_times)

    return _stats_to_distribution(stats, active_policy, path)


def build_label_distribution_rows(result: LabelDistribution) -> list[dict[str, str]]:
    """라벨 분포표 artifact 행을 생성한다."""

    rows = [
        _metric_row("prediction_window_minutes", str(result.prediction_window_minutes)),
        _metric_row("candidate_event_count", str(result.candidate_event_count)),
        _metric_row("labeled_sample_count", str(result.labeled_sample_count)),
        _metric_row("positive_count", str(result.positive_count)),
        _metric_row("negative_count", str(result.negative_count)),
        _metric_row("positive_ratio", _format_ratio(result.positive_ratio)),
        _metric_row("negative_ratio", _format_ratio(result.negative_ratio)),
        _metric_row(
            "excluded_missing_session_count",
            str(result.excluded_missing_session_count),
        ),
        _metric_row("excluded_invalid_time_count", str(result.excluded_invalid_time_count)),
        _metric_row(
            "excluded_at_or_after_first_purchase_count",
            str(result.excluded_at_or_after_first_purchase_count),
        ),
        _metric_row("session_count", str(result.session_count)),
        _metric_row("purchase_session_count", str(result.purchase_session_count)),
    ]
    return rows


def build_label_policy_rows(policy: LabelingPolicy) -> list[dict[str, str]]:
    """라벨링 정책표 artifact 행을 생성한다."""

    return [
        {
            "policy": "prediction_window",
            "value": f"{policy.prediction_window_minutes} minutes",
            "description": "기준 시점 이후 window 이내 같은 세션 purchase를 positive로 정의한다.",
        },
        {
            "policy": "cutoff_interval",
            "value": "(cutoff_time, cutoff_time + window]",
            "description": "기준 시점과 같은 시각의 purchase는 미래 구매로 보지 않는다.",
        },
        {
            "policy": "prefix_feature_scope",
            "value": "events_at_or_before_cutoff",
            "description": "prefix feature는 기준 시점까지 관측된 이벤트만 포함한다.",
        },
        {
            "policy": "purchase_event_policy",
            "value": "exclude_at_or_after_first_purchase",
            "description": "첫 purchase 이벤트와 그 이후 이벤트는 학습 sample에서 제외한다.",
        },
    ]


def write_label_artifacts(
    result: LabelDistribution,
    reports_dir: Path,
    policy: LabelingPolicy | None = None,
) -> None:
    """라벨링 결과를 리포트/대시보드용 artifact로 저장한다."""

    active_policy = policy or LabelingPolicy(
        prediction_window_minutes=result.prediction_window_minutes
    )
    output_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(
        output_dir / "label_distribution.csv",
        build_label_distribution_rows(result),
    )
    _write_csv(output_dir / "labeling_policy.csv", build_label_policy_rows(active_policy))
    (output_dir / "labeling_report.md").write_text(
        build_label_markdown_report(result),
        encoding="utf-8",
    )


def build_label_markdown_report(result: LabelDistribution) -> str:
    """Step 3 라벨링 리포트 초안을 생성한다."""

    lines = [
        "# Step 3 라벨링 설계 및 TDD",
        "",
        "## 라벨 정의",
        "",
        f"- 예측 window: 기준 시점 이후 {result.prediction_window_minutes}분",
        "- positive: 같은 `user_session`에서 `(cutoff_time, cutoff_time + window]` 구간에 purchase 발생",
        "- sample 정책: 첫 purchase 이벤트와 그 이후 이벤트는 학습 sample에서 제외",
        "- feature 범위: prefix feature는 기준 시점까지의 이벤트만 사용",
        "",
        "## 라벨 분포",
        "",
        f"- 후보 이벤트 수: {result.candidate_event_count:,}",
        f"- 라벨링 sample 수: {result.labeled_sample_count:,}",
        f"- positive: {result.positive_count:,} ({_format_ratio(result.positive_ratio)})",
        f"- negative: {result.negative_count:,} ({_format_ratio(result.negative_ratio)})",
        f"- 첫 purchase 이후 제외: {result.excluded_at_or_after_first_purchase_count:,}",
        f"- 세션 수: {result.session_count:,}",
        f"- purchase 포함 세션 수: {result.purchase_session_count:,}",
        "",
        "## 누수 방지 확인",
        "",
        "- 라벨 계산은 기준 시점 이후 purchase 시각만 참조한다.",
        "- prefix feature 생성은 기준 시점 이후 이벤트를 포함하지 않는다.",
        "- raw `user_id`, `user_session`, 원문 `event_time`은 모델 입력 feature가 아니라 key와 검증 용도로만 사용한다.",
        "",
    ]
    return "\n".join(lines)


def _collect_first_purchase_times(path: Path, chunksize: int) -> dict[str, pd.Timestamp]:
    first_purchase_times: dict[str, pd.Timestamp] = {}
    usecols = ["event_time", "event_type", "user_session"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        purchase_events = chunk.loc[
            chunk["event_type"].eq("purchase") & chunk["user_session"].notna()
        ].copy()
        if purchase_events.empty:
            continue
        purchase_events["_event_time"] = pd.to_datetime(
            purchase_events["event_time"],
            format=EVENT_TIME_FORMAT,
            errors="coerce",
            utc=True,
        )
        purchase_events = purchase_events.loc[purchase_events["_event_time"].notna()]
        chunk_first_purchase_times = purchase_events.groupby("user_session")[
            "_event_time"
        ].min()
        for session_value, event_time in chunk_first_purchase_times.items():
            session_id = str(session_value)
            current = first_purchase_times.get(session_id)
            if current is None or event_time < current:
                first_purchase_times[session_id] = event_time
    return first_purchase_times


def _summarize_frame(events: pd.DataFrame, policy: LabelingPolicy) -> dict[str, object]:
    stats = _empty_stats()
    if events.empty:
        return stats

    working = events.copy()
    working["_event_time"] = pd.to_datetime(
        working["event_time"],
        format=EVENT_TIME_FORMAT,
        errors="coerce",
        utc=True,
    )
    valid_purchase_events = working.loc[
        working["event_type"].eq("purchase")
        & working["user_session"].notna()
        & working["_event_time"].notna()
    ]
    first_purchase_times = {
        str(session_id): event_time
        for session_id, event_time in valid_purchase_events.groupby("user_session")[
            "_event_time"
        ].min().items()
    }
    stats["purchase_session_ids"].update(first_purchase_times)
    _update_distribution_stats(working, policy, stats, first_purchase_times)
    return stats


def _update_distribution_stats(
    events: pd.DataFrame,
    policy: LabelingPolicy,
    stats: dict[str, object],
    first_purchase_times: dict[str, pd.Timestamp],
) -> None:
    stats["candidate_event_count"] += len(events)
    event_times = pd.to_datetime(
        events["event_time"],
        format=EVENT_TIME_FORMAT,
        errors="coerce",
        utc=True,
    )
    window = pd.Timedelta(minutes=policy.prediction_window_minutes)
    session_values = events["user_session"]
    missing_session_mask = session_values.isna()
    valid_time_mask = event_times.notna()
    valid_mask = session_values.notna() & valid_time_mask

    stats["excluded_missing_session_count"] += int(missing_session_mask.sum())
    stats["excluded_invalid_time_count"] += int(
        (session_values.notna() & ~valid_time_mask).sum()
    )
    if not valid_mask.any():
        return

    valid_sessions = session_values.loc[valid_mask].astype(str)
    stats["session_ids"].update(valid_sessions.unique().tolist())

    first_purchase_series = pd.to_datetime(
        valid_sessions.map(first_purchase_times),
        errors="coerce",
        utc=True,
    )
    valid_event_times = pd.Series(event_times.loc[valid_mask].to_numpy(), index=valid_sessions.index)
    has_purchase = first_purchase_series.notna()
    excluded_mask = (
        has_purchase & valid_event_times.ge(first_purchase_series)
        if policy.exclude_at_or_after_first_purchase
        else pd.Series(False, index=valid_sessions.index)
    )
    stats["excluded_at_or_after_first_purchase_count"] += int(excluded_mask.sum())

    labeled_mask = ~excluded_mask
    labeled_count = int(labeled_mask.sum())
    stats["labeled_sample_count"] += labeled_count
    if labeled_count == 0:
        return

    positive_mask = (
        labeled_mask
        & has_purchase
        & valid_event_times.lt(first_purchase_series)
        & first_purchase_series.le(valid_event_times + window)
    )
    positive_count = int(positive_mask.sum())
    stats["positive_count"] += positive_count
    stats["negative_count"] += labeled_count - positive_count


def _stats_to_distribution(
    stats: dict[str, object],
    policy: LabelingPolicy,
    source_path: Path | None,
) -> LabelDistribution:
    return LabelDistribution(
        source_path=source_path,
        prediction_window_minutes=policy.prediction_window_minutes,
        candidate_event_count=int(stats["candidate_event_count"]),
        labeled_sample_count=int(stats["labeled_sample_count"]),
        positive_count=int(stats["positive_count"]),
        negative_count=int(stats["negative_count"]),
        excluded_missing_session_count=int(stats["excluded_missing_session_count"]),
        excluded_invalid_time_count=int(stats["excluded_invalid_time_count"]),
        excluded_at_or_after_first_purchase_count=int(
            stats["excluded_at_or_after_first_purchase_count"]
        ),
        session_count=len(stats["session_ids"]),
        purchase_session_count=len(stats["purchase_session_ids"]),
    )


def _empty_stats() -> dict[str, object]:
    return {
        "candidate_event_count": 0,
        "labeled_sample_count": 0,
        "positive_count": 0,
        "negative_count": 0,
        "excluded_missing_session_count": 0,
        "excluded_invalid_time_count": 0,
        "excluded_at_or_after_first_purchase_count": 0,
        "session_ids": set(),
        "purchase_session_ids": set(),
    }


def _empty_label_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "user_session",
            "cutoff_time",
            "session_position",
            "prefix_length",
            "prefix_event_types",
            "last_event_type",
            "label",
            "minutes_until_purchase",
        ]
    )


def _minutes_between(
    start: pd.Timestamp,
    end: pd.Timestamp | None,
) -> float | None:
    if end is None:
        return None
    return (end - start).total_seconds() / 60


def _metric_row(metric: str, value: str) -> dict[str, str]:
    return {"metric": metric, "value": value}


def _write_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    pd.DataFrame(list(rows)).to_csv(path, index=False, encoding="utf-8")


def _format_ratio(value: float) -> str:
    return f"{value:.6f}"
