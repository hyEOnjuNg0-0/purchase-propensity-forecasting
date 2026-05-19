"""Step 5 Feature Engineering 유스케이스."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from purchase_time_forecasting.data_profiling import EVENT_TIME_FORMAT
from purchase_time_forecasting.exploratory_analysis import PRICE_BAND_LABELS, PRICE_BINS
from purchase_time_forecasting.labeling import LABEL_WINDOW_MINUTES


EVENT_TYPES = ("view", "cart", "remove_from_cart", "purchase")
SEQUENCE_DELIMITER = " "


@dataclass(frozen=True)
class FeatureEngineeringPolicy:
    """feature dataset 생성 정책."""

    prediction_window_minutes: int = LABEL_WINDOW_MINUTES
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    max_sequence_length: int = 50
    exclude_at_or_after_first_purchase: bool = True

    @property
    def test_ratio(self) -> float:
        return max(0.0, 1.0 - self.train_ratio - self.validation_ratio)


def build_feature_dataset(
    events: pd.DataFrame,
    policy: FeatureEngineeringPolicy | None = None,
) -> pd.DataFrame:
    """prefix 기반 tabular/sequence feature dataset을 생성한다.

    모델 입력 feature는 기준 시점까지 관측된 이벤트만 사용한다. `user_id`,
    `user_session`, `cutoff_time`은 feature dataset에 남기지만 model input이 아니라
    audit/key 컬럼으로만 사용하도록 feature dictionary에서 분리한다.
    """

    active_policy = policy or FeatureEngineeringPolicy()
    _validate_split_policy(active_policy)
    working = _prepare_events(events)
    if working.empty:
        return _empty_feature_frame()

    first_purchase_times = _first_purchase_times(working)
    user_history = _build_user_history(working)
    rows: list[dict[str, object]] = []
    window = pd.Timedelta(minutes=active_policy.prediction_window_minutes)

    for session_id, session_events in working.groupby("user_session", sort=False):
        session_key = str(session_id)
        first_purchase_time = first_purchase_times.get(session_key)
        prefix = _PrefixAccumulator()
        previous_event_time: pd.Timestamp | None = None
        session_start_time: pd.Timestamp | None = None

        for position, (_, row) in enumerate(
            session_events.iterrows(),
            start=1,
        ):
            source_order = int(row["_source_order"])
            cutoff_time = row["_event_time"]
            if session_start_time is None:
                session_start_time = cutoff_time

            if (
                active_policy.exclude_at_or_after_first_purchase
                and first_purchase_time is not None
                and cutoff_time >= first_purchase_time
            ):
                previous_event_time = cutoff_time
                prefix.add(row, previous_event_time, session_start_time)
                continue

            gap_minutes = _minutes_between(previous_event_time, cutoff_time)
            prefix.add(row, previous_event_time, session_start_time)
            next_purchase_time = (
                first_purchase_time
                if first_purchase_time is not None
                and cutoff_time < first_purchase_time <= cutoff_time + window
                else None
            )
            history = user_history.get(source_order, _empty_user_history())
            rows.append(
                {
                    "sample_id": _sample_id(source_order),
                    "user_session": session_key,
                    "user_id": _string_or_missing(row["user_id"]),
                    "cutoff_time": cutoff_time.isoformat(),
                    "split_order_time": cutoff_time,
                    "source_order": int(source_order),
                    "session_position": position,
                    "prefix_length": prefix.event_count,
                    "last_event_type": str(row["event_type"]),
                    "session_elapsed_minutes": _minutes_between(
                        session_start_time,
                        cutoff_time,
                    ),
                    "time_since_previous_event_minutes": gap_minutes,
                    "hour": int(cutoff_time.hour),
                    "day_of_week": int(cutoff_time.dayofweek),
                    "event_count_view": prefix.event_type_counts.get("view", 0),
                    "event_count_cart": prefix.event_type_counts.get("cart", 0),
                    "event_count_remove_from_cart": prefix.event_type_counts.get(
                        "remove_from_cart",
                        0,
                    ),
                    "event_count_purchase": prefix.event_type_counts.get("purchase", 0),
                    "unique_product_count": len(prefix.product_ids),
                    "unique_category_count": len(prefix.category_ids),
                    "unique_brand_count": len(prefix.brands),
                    "avg_price": prefix.avg_price,
                    "max_price": prefix.max_price,
                    "last_price": prefix.last_price,
                    "last_price_bin": prefix.last_price_bin,
                    "event_type_sequence": SEQUENCE_DELIMITER.join(prefix.event_types),
                    "product_id_sequence": SEQUENCE_DELIMITER.join(prefix.product_ids_sequence),
                    "category_id_sequence": SEQUENCE_DELIMITER.join(prefix.category_ids_sequence),
                    "price_bin_sequence": SEQUENCE_DELIMITER.join(prefix.price_bins_sequence),
                    "time_gap_minutes_sequence": SEQUENCE_DELIMITER.join(
                        prefix.time_gaps_sequence
                    ),
                    "user_past_event_count": history["user_past_event_count"],
                    "user_past_session_count": history["user_past_session_count"],
                    "user_past_purchase_count": history["user_past_purchase_count"],
                    "user_past_cart_count": history["user_past_cart_count"],
                    "user_minutes_since_last_event": history[
                        "user_minutes_since_last_event"
                    ],
                    "label": int(next_purchase_time is not None),
                    "minutes_until_purchase": _minutes_between(
                        cutoff_time,
                        next_purchase_time,
                    ),
                }
            )
            previous_event_time = cutoff_time

    if not rows:
        return _empty_feature_frame()

    features = pd.DataFrame(rows)
    features["split"] = _assign_time_splits(features, active_policy)
    features = features.drop(columns=["split_order_time", "source_order"])
    return features[_feature_column_order()]


def build_feature_dataset_from_csv(
    csv_path: Path,
    policy: FeatureEngineeringPolicy | None = None,
    chunksize: int = 1_000_000,
    max_rows: int | None = None,
    until_time: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """CSV에서 Step 5 feature dataset을 생성한다.

    전체 파일 feature dataset은 메모리 사용량이 크므로 `max_rows`로 재현 가능한 부분
    실행을 지원한다. 부분 실행도 원천 CSV에서 읽은 실제 row만 사용하며 모의 데이터는
    생성하지 않는다.
    """

    path = Path(csv_path)
    cutoff_time = normalize_until_time(until_time)
    frames: list[pd.DataFrame] = []
    remaining = max_rows
    usecols = [
        "event_time",
        "event_type",
        "product_id",
        "category_id",
        "category_code",
        "brand",
        "price",
        "user_id",
        "user_session",
    ]
    for chunk in _read_csv_chunks_until(path, usecols, chunksize, cutoff_time):
        if remaining is not None:
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.head(remaining)
            remaining -= len(chunk)
        if chunk.empty:
            continue
        frames.append(chunk)

    if not frames:
        return _empty_feature_frame()
    return build_feature_dataset(pd.concat(frames, ignore_index=True), policy=policy)


def build_feature_dataset_from_csv_streaming(
    csv_path: Path,
    features_dir: Path,
    reports_dir: Path,
    policy: FeatureEngineeringPolicy | None = None,
    chunksize: int = 1_000_000,
    until_time: str | pd.Timestamp | None = None,
) -> dict[str, object]:
    """전체 CSV 기준 feature dataset을 chunk 단위로 생성한다.

    대용량 원천 데이터를 메모리에 모두 올리지 않기 위해 세 번 읽는다. 첫 번째 pass는
    세션별 첫 purchase 시각, 두 번째 pass는 라벨 sample 기준 시간 split 경계, 세 번째
    pass는 feature row를 CSV에 append 저장한다. 입력 CSV가 event_time 기준으로 정렬되어
    있다는 Kaggle 원천 데이터 특성을 전제로 사용자 과거 행동 집계와 세션 prefix를
    순차 계산한다.
    """

    active_policy = policy or FeatureEngineeringPolicy()
    _validate_split_policy(active_policy)
    cutoff_time = normalize_until_time(until_time)
    path = Path(csv_path)
    feature_output_dir = Path(features_dir)
    report_output_dir = Path(reports_dir)
    feature_output_dir.mkdir(parents=True, exist_ok=True)
    report_output_dir.mkdir(parents=True, exist_ok=True)
    sample_index_path = feature_output_dir / "sample_index.csv"
    tabular_path = feature_output_dir / "tabular_feature_dataset.csv"
    sequence_path = feature_output_dir / "sequence_feature_dataset.parquet"
    for output_path in (sample_index_path, tabular_path, sequence_path):
        if output_path.exists():
            output_path.unlink()

    first_purchase_times = _collect_first_purchase_times_from_csv(
        path,
        chunksize,
        cutoff_time,
    )
    split_boundaries = _compute_split_boundaries_from_csv(
        path,
        first_purchase_times,
        active_policy,
        chunksize,
        cutoff_time,
    )
    stats = _StreamingFeatureStats()
    _write_streaming_feature_artifacts(
        path=path,
        sample_index_path=sample_index_path,
        tabular_path=tabular_path,
        sequence_path=sequence_path,
        first_purchase_times=first_purchase_times,
        split_boundaries=split_boundaries,
        policy=active_policy,
        chunksize=chunksize,
        stats=stats,
        until_time=cutoff_time,
    )

    _write_csv(
        report_output_dir / "feature_dictionary.csv",
        build_feature_dictionary_rows(),
    )
    _write_csv(
        report_output_dir / "feature_leakage_checklist.csv",
        build_leakage_checklist_rows(),
    )
    _write_csv(
        report_output_dir / "feature_transformer_scope.csv",
        stats.build_transformer_scope_rows(),
    )
    split_rows = stats.build_split_summary_rows()
    _write_csv(report_output_dir / "feature_split_summary.csv", split_rows)
    (report_output_dir / "feature_report.md").write_text(
        build_feature_markdown_report_from_rows(
            split_rows=split_rows,
            sample_count=stats.sample_count,
            source_path=path,
            max_rows=None,
            until_time=cutoff_time,
        ),
        encoding="utf-8",
    )
    return {
        "feature_sample_count": stats.sample_count,
        "split_rows": split_rows,
        "sample_index_path": sample_index_path,
        "tabular_path": tabular_path,
        "sequence_path": sequence_path,
    }


def normalize_until_time(
    until_time: str | pd.Timestamp | None,
) -> pd.Timestamp | None:
    """CLI/API 입력 cutoff를 UTC Timestamp로 정규화한다.

    `YYYY-MM-DD` 형식은 해당 날짜 전체를 포함하도록 23:59:59.999999999 UTC로
    변환한다. timestamp 형식은 지정한 시각까지 포함한다.
    """

    if until_time is None:
        return None
    if isinstance(until_time, str):
        text = until_time.strip()
        is_date_only = len(text) == 10 and text.count("-") == 2
        timestamp = pd.Timestamp(text)
        if is_date_only:
            timestamp = timestamp + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    else:
        timestamp = pd.Timestamp(until_time)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def build_feature_dictionary_rows() -> list[dict[str, str]]:
    """모델 입력 여부와 누수 정책을 명시한 feature dictionary를 생성한다."""

    rows = [
        _feature_row("sample_id", "identifier", "string", "key", "모델별 dataset 연결 key"),
        _feature_row("user_session", "identifier", "string", "key", "세션 grouping/key로만 사용"),
        _feature_row("user_id", "identifier", "string", "audit_only", "raw ID는 모델 입력 제외"),
        _feature_row(
            "cutoff_time",
            "time",
            "datetime",
            "audit_only",
            "원문 timestamp는 모델 입력 제외, 파생 시간 feature만 사용",
        ),
        _feature_row("split", "split", "string", "audit_only", "시간 기준 split 검증용"),
        _feature_row("label", "target", "integer", "target", "향후 window 내 purchase 여부"),
        _feature_row(
            "minutes_until_purchase",
            "target_diagnostic",
            "float",
            "audit_only",
            "평가/오류 분석용이며 학습 입력 제외",
        ),
    ]

    for feature_name in _tabular_input_columns():
        rows.append(
            _feature_row(
                feature_name,
                "tabular",
                _feature_dtype(feature_name),
                "tabular_input",
                "기준 시점까지 관측된 prefix 또는 기준 시점 이전 사용자 이력만 사용",
            )
        )

    for feature_name in _sequence_input_columns():
        rows.append(
            _feature_row(
                feature_name,
                "sequence",
                "string_sequence",
                "sequence_input",
                "기준 시점까지의 prefix sequence만 사용",
            )
        )
    return rows


def build_leakage_checklist_rows() -> list[dict[str, str]]:
    """Step 5 누수 방지 체크리스트를 생성한다."""

    return [
        _check_row(
            "raw_user_id_excluded",
            "raw `user_id` 모델 입력 제외",
            "pass",
            "`user_id`는 audit_only로만 정의",
        ),
        _check_row(
            "raw_user_session_key_only",
            "`user_session` key 용도 제한",
            "pass",
            "`user_session`은 key로만 정의",
        ),
        _check_row(
            "raw_event_time_excluded",
            "원문 `event_time` 모델 입력 제외",
            "pass",
            "`cutoff_time`은 audit_only이며 hour/day/elapsed 파생값만 입력",
        ),
        _check_row(
            "prefix_scope",
            "sequence feature 미래 이벤트 제외",
            "pass",
            "session별 누적 prefix accumulator로 기준 시점까지의 이벤트만 저장",
        ),
        _check_row(
            "post_purchase_excluded",
            "첫 purchase 이후 sample 제외",
            "pass",
            "첫 purchase 시점과 이후 이벤트는 학습 sample 생성에서 제외",
        ),
        _check_row(
            "user_history_scope",
            "사용자 과거 행동 집계 기준 시점 이전 제한",
            "pass",
            "user_past_* feature는 현재 이벤트 처리 전에 계산",
        ),
        _check_row(
            "transformer_fit_scope",
            "encoder/scaler fit 범위 train split 제한",
            "pass",
            "`feature_transformer_scope.csv`가 train split 기준 fit 값을 기록",
        ),
        _check_row(
            "common_sample_contract",
            "baseline/sequence dataset 공통 sample 기준",
            "pass",
            "`sample_index.csv`, `tabular_feature_dataset.csv`, `sequence_feature_dataset.parquet`가 동일 `sample_id`를 공유",
        ),
    ]


def build_transformer_scope_rows(features: pd.DataFrame) -> list[dict[str, str]]:
    """encoder/scaler fit 범위를 train split으로 제한한 artifact 행을 생성한다."""

    if features.empty or "split" not in features:
        return []

    train = features.loc[features["split"].eq("train")]
    rows: list[dict[str, str]] = []
    for feature_name in _categorical_input_columns():
        values = _sorted_unique_text(train[feature_name]) if feature_name in train else []
        rows.append(
            {
                "feature_name": feature_name,
                "transformer_type": "categorical_vocab",
                "fit_split": "train",
                "train_observation_count": str(len(train)),
                "fitted_values": "|".join(values),
            }
        )

    for feature_name in _numeric_input_columns():
        if feature_name not in train:
            continue
        values = pd.to_numeric(train[feature_name], errors="coerce").dropna()
        rows.append(
            {
                "feature_name": feature_name,
                "transformer_type": "standard_scaler_stats",
                "fit_split": "train",
                "train_observation_count": str(int(values.count())),
                "fitted_values": (
                    f"mean={values.mean():.6f}|std={values.std(ddof=0):.6f}"
                    if not values.empty
                    else "mean=|std="
                ),
            }
        )
    return rows


def build_feature_artifacts(
    features: pd.DataFrame,
    features_dir: Path,
    reports_dir: Path,
    source_path: Path | None = None,
    max_rows: int | None = None,
    until_time: str | pd.Timestamp | None = None,
    max_sequence_length: int = 50,
) -> None:
    """feature dataset과 Step 5 문서화 artifact를 저장한다."""

    feature_output_dir = Path(features_dir)
    report_output_dir = Path(reports_dir)
    feature_output_dir.mkdir(parents=True, exist_ok=True)
    report_output_dir.mkdir(parents=True, exist_ok=True)

    build_sample_index(features).to_csv(
        feature_output_dir / "sample_index.csv",
        index=False,
        encoding="utf-8",
    )
    build_tabular_feature_dataset(features).to_csv(
        feature_output_dir / "tabular_feature_dataset.csv",
        index=False,
        encoding="utf-8",
    )
    build_sequence_feature_dataset(
        features,
        max_sequence_length=max_sequence_length,
    ).to_parquet(
        feature_output_dir / "sequence_feature_dataset.parquet",
        index=False,
    )
    _write_csv(
        report_output_dir / "feature_dictionary.csv",
        build_feature_dictionary_rows(),
    )
    _write_csv(
        report_output_dir / "feature_leakage_checklist.csv",
        build_leakage_checklist_rows(),
    )
    _write_csv(
        report_output_dir / "feature_transformer_scope.csv",
        build_transformer_scope_rows(features),
    )
    _write_csv(
        report_output_dir / "feature_split_summary.csv",
        build_split_summary_rows(features),
    )
    (report_output_dir / "feature_report.md").write_text(
        build_feature_markdown_report(features, source_path, max_rows, until_time),
        encoding="utf-8",
    )


def build_sample_index(features: pd.DataFrame) -> pd.DataFrame:
    """모델 간 평가를 연결하는 공통 sample index를 생성한다."""

    return features.loc[:, _sample_index_columns()].copy()


def build_tabular_feature_dataset(features: pd.DataFrame) -> pd.DataFrame:
    """baseline 모델 입력용 tabular feature dataset을 생성한다."""

    return features.loc[:, ["sample_id", *_tabular_input_columns()]].copy()


def build_sequence_feature_dataset(
    features: pd.DataFrame,
    max_sequence_length: int = 50,
) -> pd.DataFrame:
    """sequence 모델 입력용 prefix sequence dataset을 생성한다."""

    _validate_max_sequence_length(max_sequence_length)
    sequence = features.loc[:, ["sample_id", *_sequence_input_columns()]].copy()
    for column in _sequence_input_columns():
        sequence[column] = sequence[column].map(
            lambda value: _tail_sequence(value, max_sequence_length)
        )
    return sequence


def build_split_summary_rows(features: pd.DataFrame) -> list[dict[str, str]]:
    """train/validation/test split별 sample 및 label 분포를 요약한다."""

    rows = []
    for split in ("train", "validation", "test"):
        segment = features.loc[features["split"].eq(split)] if "split" in features else features
        sample_count = len(segment)
        positive_count = int(segment["label"].sum()) if "label" in segment else 0
        rows.append(
            {
                "split": split,
                "sample_count": str(sample_count),
                "positive_count": str(positive_count),
                "negative_count": str(sample_count - positive_count),
                "positive_ratio": _format_ratio(
                    positive_count / sample_count if sample_count else 0.0
                ),
                "cutoff_time_min": str(segment["cutoff_time"].min()) if sample_count else "",
                "cutoff_time_max": str(segment["cutoff_time"].max()) if sample_count else "",
            }
        )
    return rows


def build_feature_markdown_report(
    features: pd.DataFrame,
    source_path: Path | None = None,
    max_rows: int | None = None,
    until_time: str | pd.Timestamp | None = None,
) -> str:
    """Step 5 리포트 초안을 생성한다."""

    split_rows = build_split_summary_rows(features)
    lines = [
        "# Step 5 Feature Engineering",
        "",
        "## 핵심 요약",
        "",
        f"- 대상 파일: `{source_path}`" if source_path else "- 대상 파일: DataFrame 입력",
        f"- 입력 row 제한: {max_rows:,}" if max_rows is not None else "- 입력 row 제한: 없음",
        f"- 종료 일시 필터: {normalize_until_time(until_time)}"
        if until_time is not None
        else "- 종료 일시 필터: 없음",
        f"- feature sample 수: {len(features):,}",
        "- raw `user_id`, `user_session`, `cutoff_time`은 모델 입력에서 제외하고 audit/key 용도로만 유지한다.",
        "- `sample_id` 기준으로 sample index, tabular feature, sequence feature를 연결한다.",
        "- sequence feature는 기준 시점까지의 prefix 중 최근 `max_sequence_length`개만 별도 parquet artifact에 저장한다.",
        "- encoder/scaler fit 범위는 train split으로 제한한다.",
        "",
        "## 산출물",
        "",
        "- `artifacts/features/sample_index.csv`: 모델 간 공통 평가 sample index",
        "- `artifacts/features/tabular_feature_dataset.csv`: baseline 모델 입력용 tabular feature",
        "- `artifacts/features/sequence_feature_dataset.parquet`: sequence 모델 입력용 prefix sequence feature",
        "- `feature_dictionary.csv`: feature별 모델 입력 역할과 누수 정책",
        "- `feature_leakage_checklist.csv`: Step 5 누수 방지 체크리스트",
        "- `feature_transformer_scope.csv`: train split 기준 encoder/scaler fit 범위",
        "- `feature_split_summary.csv`: split별 label 분포",
        "",
        "## Split 요약",
        "",
    ]
    for row in split_rows:
        lines.append(
            f"- {row['split']}: {int(row['sample_count']):,} samples, "
            f"positive ratio {row['positive_ratio']}"
        )
    lines.append("")
    return "\n".join(lines)


def build_feature_markdown_report_from_rows(
    split_rows: list[dict[str, str]],
    sample_count: int,
    source_path: Path | None = None,
    max_rows: int | None = None,
    until_time: str | pd.Timestamp | None = None,
) -> str:
    """streaming 실행 결과처럼 DataFrame이 없을 때 Step 5 리포트를 생성한다."""

    lines = [
        "# Step 5 Feature Engineering",
        "",
        "## 핵심 요약",
        "",
        f"- 대상 파일: `{source_path}`" if source_path else "- 대상 파일: DataFrame 입력",
        f"- 입력 row 제한: {max_rows:,}" if max_rows is not None else "- 입력 row 제한: 없음",
        f"- 종료 일시 필터: {normalize_until_time(until_time)}"
        if until_time is not None
        else "- 종료 일시 필터: 없음",
        f"- feature sample 수: {sample_count:,}",
        "- raw `user_id`, `user_session`, `cutoff_time`은 모델 입력에서 제외하고 audit/key 용도로만 유지한다.",
        "- `sample_id` 기준으로 sample index, tabular feature, sequence feature를 연결한다.",
        "- sequence feature는 기준 시점까지의 prefix 중 최근 `max_sequence_length`개만 별도 parquet artifact에 저장한다.",
        "- encoder/scaler fit 범위는 train split으로 제한한다.",
        "",
        "## 산출물",
        "",
        "- `artifacts/features/sample_index.csv`: 모델 간 공통 평가 sample index",
        "- `artifacts/features/tabular_feature_dataset.csv`: baseline 모델 입력용 tabular feature",
        "- `artifacts/features/sequence_feature_dataset.parquet`: sequence 모델 입력용 prefix sequence feature",
        "- `feature_dictionary.csv`: feature별 모델 입력 역할과 누수 정책",
        "- `feature_leakage_checklist.csv`: Step 5 누수 방지 체크리스트",
        "- `feature_transformer_scope.csv`: train split 기준 encoder/scaler fit 범위",
        "- `feature_split_summary.csv`: split별 label 분포",
        "",
        "## Split 요약",
        "",
    ]
    for row in split_rows:
        lines.append(
            f"- {row['split']}: {int(row['sample_count']):,} samples, "
            f"positive ratio {row['positive_ratio']}"
        )
    lines.append("")
    return "\n".join(lines)


@dataclass
class _NumericFitStats:
    count: int = 0
    total: float = 0.0
    squared_total: float = 0.0

    def update(self, value: object) -> None:
        if value is None or pd.isna(value):
            return
        numeric_value = float(value)
        self.count += 1
        self.total += numeric_value
        self.squared_total += numeric_value * numeric_value

    @property
    def mean(self) -> float | None:
        if self.count == 0:
            return None
        return self.total / self.count

    @property
    def std(self) -> float | None:
        if self.count == 0:
            return None
        mean = self.mean or 0.0
        variance = max(0.0, (self.squared_total / self.count) - (mean * mean))
        return variance ** 0.5


@dataclass
class _SplitStats:
    sample_count: int = 0
    positive_count: int = 0
    cutoff_time_min: str | None = None
    cutoff_time_max: str | None = None

    def update(self, row: dict[str, object]) -> None:
        cutoff_time = str(row["cutoff_time"])
        self.sample_count += 1
        self.positive_count += int(row["label"])
        if self.cutoff_time_min is None or cutoff_time < self.cutoff_time_min:
            self.cutoff_time_min = cutoff_time
        if self.cutoff_time_max is None or cutoff_time > self.cutoff_time_max:
            self.cutoff_time_max = cutoff_time


@dataclass
class _StreamingFeatureStats:
    sample_count: int = 0
    split_stats: dict[str, _SplitStats] = field(
        default_factory=lambda: {
            "train": _SplitStats(),
            "validation": _SplitStats(),
            "test": _SplitStats(),
        }
    )
    categorical_values: dict[str, set[str]] = field(
        default_factory=lambda: {column: set() for column in _categorical_input_columns()}
    )
    numeric_stats: dict[str, _NumericFitStats] = field(
        default_factory=lambda: {column: _NumericFitStats() for column in _numeric_input_columns()}
    )

    def update(self, row: dict[str, object]) -> None:
        split = str(row["split"])
        self.sample_count += 1
        self.split_stats[split].update(row)
        if split != "train":
            return
        for column in _categorical_input_columns():
            value = row.get(column)
            if value is not None and not pd.isna(value):
                self.categorical_values[column].add(str(value))
        for column in _numeric_input_columns():
            self.numeric_stats[column].update(row.get(column))

    def build_split_summary_rows(self) -> list[dict[str, str]]:
        rows = []
        for split in ("train", "validation", "test"):
            stats = self.split_stats[split]
            negative_count = stats.sample_count - stats.positive_count
            rows.append(
                {
                    "split": split,
                    "sample_count": str(stats.sample_count),
                    "positive_count": str(stats.positive_count),
                    "negative_count": str(negative_count),
                    "positive_ratio": _format_ratio(
                        stats.positive_count / stats.sample_count
                        if stats.sample_count
                        else 0.0
                    ),
                    "cutoff_time_min": stats.cutoff_time_min or "",
                    "cutoff_time_max": stats.cutoff_time_max or "",
                }
            )
        return rows

    def build_transformer_scope_rows(self) -> list[dict[str, str]]:
        train_count = self.split_stats["train"].sample_count
        rows = []
        for feature_name in _categorical_input_columns():
            rows.append(
                {
                    "feature_name": feature_name,
                    "transformer_type": "categorical_vocab",
                    "fit_split": "train",
                    "train_observation_count": str(train_count),
                    "fitted_values": "|".join(sorted(self.categorical_values[feature_name])),
                }
            )
        for feature_name in _numeric_input_columns():
            stats = self.numeric_stats[feature_name]
            rows.append(
                {
                    "feature_name": feature_name,
                    "transformer_type": "standard_scaler_stats",
                    "fit_split": "train",
                    "train_observation_count": str(stats.count),
                    "fitted_values": (
                        f"mean={stats.mean:.6f}|std={stats.std:.6f}"
                        if stats.mean is not None and stats.std is not None
                        else "mean=|std="
                    ),
                }
            )
        return rows


@dataclass
class _StreamingSessionState:
    prefix: _PrefixAccumulator = field(default_factory=lambda: _PrefixAccumulator())
    previous_event_time: pd.Timestamp | None = None
    session_start_time: pd.Timestamp | None = None
    position: int = 0


@dataclass
class _StreamingUserState:
    event_count: int = 0
    sessions: set[str] = field(default_factory=set)
    purchase_count: int = 0
    cart_count: int = 0
    last_event_time: pd.Timestamp | None = None

    def history_before(self, event_time: pd.Timestamp) -> dict[str, object]:
        return {
            "user_past_event_count": self.event_count,
            "user_past_session_count": len(self.sessions),
            "user_past_purchase_count": self.purchase_count,
            "user_past_cart_count": self.cart_count,
            "user_minutes_since_last_event": _minutes_between(
                self.last_event_time,
                event_time,
            ),
        }

    def update(self, row: pd.Series) -> None:
        self.event_count += 1
        self.sessions.add(str(row["user_session"]))
        if str(row["event_type"]) == "purchase":
            self.purchase_count += 1
        if str(row["event_type"]) == "cart":
            self.cart_count += 1
        self.last_event_time = row["_event_time"]


def _collect_first_purchase_times_from_csv(
    path: Path,
    chunksize: int,
    until_time: pd.Timestamp | None = None,
) -> dict[str, pd.Timestamp]:
    first_purchase_times: dict[str, pd.Timestamp] = {}
    usecols = ["event_time", "event_type", "user_session"]
    for chunk in _read_csv_chunks_until(path, usecols, chunksize, until_time):
        working = chunk.loc[
            chunk["event_type"].eq("purchase") & chunk["user_session"].notna()
        ].copy()
        if working.empty:
            continue
        working["_event_time"] = pd.to_datetime(
            working["event_time"],
            format=EVENT_TIME_FORMAT,
            errors="coerce",
            utc=True,
        )
        working = working.loc[working["_event_time"].notna()]
        grouped = working.groupby("user_session")["_event_time"].min()
        for session_value, event_time in grouped.items():
            session_id = str(session_value)
            current = first_purchase_times.get(session_id)
            if current is None or event_time < current:
                first_purchase_times[session_id] = event_time
    return first_purchase_times


def _compute_split_boundaries_from_csv(
    path: Path,
    first_purchase_times: dict[str, pd.Timestamp],
    policy: FeatureEngineeringPolicy,
    chunksize: int,
    until_time: pd.Timestamp | None = None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    time_counts: dict[pd.Timestamp, int] = {}
    usecols = ["event_time", "user_session"]
    for chunk in _read_csv_chunks_until(path, usecols, chunksize, until_time):
        working = chunk.loc[chunk["user_session"].notna()].copy()
        if working.empty:
            continue
        working["_event_time"] = pd.to_datetime(
            working["event_time"],
            format=EVENT_TIME_FORMAT,
            errors="coerce",
            utc=True,
        )
        working = working.loc[working["_event_time"].notna()]
        if working.empty:
            continue
        first_purchase_series = pd.to_datetime(
            working["user_session"].astype(str).map(first_purchase_times),
            errors="coerce",
            utc=True,
        )
        eligible = first_purchase_series.isna() | working["_event_time"].lt(first_purchase_series)
        counts = working.loc[eligible, "_event_time"].value_counts()
        for event_time, count in counts.items():
            time_counts[event_time] = time_counts.get(event_time, 0) + int(count)

    total = sum(time_counts.values())
    if total == 0:
        return (None, None)
    train_target = int(total * policy.train_ratio)
    validation_target = train_target + int(total * policy.validation_ratio)
    train_boundary = None
    validation_boundary = None
    cumulative = 0
    for event_time in sorted(time_counts):
        cumulative += time_counts[event_time]
        if train_boundary is None and cumulative >= train_target:
            train_boundary = event_time
        if validation_boundary is None and cumulative >= validation_target:
            validation_boundary = event_time
            break
    return (train_boundary, validation_boundary)


def _write_streaming_feature_artifacts(
    path: Path,
    sample_index_path: Path,
    tabular_path: Path,
    sequence_path: Path,
    first_purchase_times: dict[str, pd.Timestamp],
    split_boundaries: tuple[pd.Timestamp | None, pd.Timestamp | None],
    policy: FeatureEngineeringPolicy,
    chunksize: int,
    stats: _StreamingFeatureStats,
    until_time: pd.Timestamp | None = None,
) -> None:
    session_states: dict[str, _StreamingSessionState] = {}
    user_states: dict[str, _StreamingUserState] = {}
    sample_index_rows: list[dict[str, object]] = []
    tabular_rows: list[dict[str, object]] = []
    sequence_rows: list[dict[str, object]] = []
    sample_index_header = True
    tabular_header = True
    sequence_writer = None
    global_order = 0
    usecols = [
        "event_time",
        "event_type",
        "product_id",
        "category_id",
        "category_code",
        "brand",
        "price",
        "user_id",
        "user_session",
    ]
    for chunk in _read_csv_chunks_until(path, usecols, chunksize, until_time):
        chunk = chunk.copy()
        chunk["_source_order"] = range(global_order, global_order + len(chunk))
        global_order += len(chunk)
        working = _prepare_streaming_events(chunk)
        if working.empty:
            continue
        for _, row in working.iterrows():
            feature_row = _build_streaming_feature_row(
                row=row,
                session_states=session_states,
                user_states=user_states,
                first_purchase_times=first_purchase_times,
                split_boundaries=split_boundaries,
                policy=policy,
            )
            if feature_row is None:
                continue
            sample_index_rows.append(_sample_index_row(feature_row))
            tabular_rows.append(_tabular_feature_row(feature_row))
            sequence_rows.append(
                _sequence_feature_row(
                    feature_row,
                    max_sequence_length=policy.max_sequence_length,
                )
            )
            stats.update(feature_row)
            if len(sample_index_rows) >= 100_000:
                _append_rows_to_csv(
                    sample_index_path,
                    sample_index_rows,
                    _sample_index_columns(),
                    sample_index_header,
                )
                _append_rows_to_csv(
                    tabular_path,
                    tabular_rows,
                    _tabular_artifact_columns(),
                    tabular_header,
                )
                sequence_writer = _append_sequence_rows_to_parquet(
                    sequence_path,
                    sequence_rows,
                    _sequence_artifact_columns(),
                    sequence_writer,
                )
                sample_index_header = False
                tabular_header = False
                sample_index_rows = []
                tabular_rows = []
                sequence_rows = []
    if sample_index_rows:
        _append_rows_to_csv(
            sample_index_path,
            sample_index_rows,
            _sample_index_columns(),
            sample_index_header,
        )
        _append_rows_to_csv(
            tabular_path,
            tabular_rows,
            _tabular_artifact_columns(),
            tabular_header,
        )
        sequence_writer = _append_sequence_rows_to_parquet(
            sequence_path,
            sequence_rows,
            _sequence_artifact_columns(),
            sequence_writer,
        )
    if sequence_writer is not None:
        sequence_writer.close()
    elif not sequence_path.exists():
        pd.DataFrame(columns=_sequence_artifact_columns()).to_parquet(
            sequence_path,
            index=False,
        )


def _prepare_streaming_events(events: pd.DataFrame) -> pd.DataFrame:
    working = events.copy()
    for column in (
        "product_id",
        "category_id",
        "category_code",
        "brand",
        "price",
        "user_id",
    ):
        if column not in working:
            working[column] = pd.NA
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
    return working.sort_values(["_event_time", "_source_order"], kind="mergesort")


def _filter_raw_events_until_time(
    events: pd.DataFrame,
    until_time: pd.Timestamp | None,
) -> pd.DataFrame:
    if until_time is None or events.empty or "event_time" not in events:
        return events
    event_times = pd.to_datetime(
        events["event_time"],
        format=EVENT_TIME_FORMAT,
        errors="coerce",
        utc=True,
    )
    return events.loc[event_times.notna() & event_times.le(until_time)].copy()


def _read_csv_chunks_until(
    path: Path,
    usecols: list[str],
    chunksize: int,
    until_time: pd.Timestamp | None,
) -> Iterable[pd.DataFrame]:
    for chunk in pd.read_csv(
        path,
        usecols=lambda column: column in usecols,
        chunksize=chunksize,
    ):
        if until_time is None:
            yield chunk
            continue

        event_times = pd.to_datetime(
            chunk["event_time"],
            format=EVENT_TIME_FORMAT,
            errors="coerce",
            utc=True,
        )
        valid_event_times = event_times.dropna()
        if valid_event_times.empty:
            continue
        if valid_event_times.min() > until_time:
            break

        filtered = chunk.loc[event_times.le(until_time)].copy()
        if not filtered.empty:
            yield filtered


def _build_streaming_feature_row(
    row: pd.Series,
    session_states: dict[str, _StreamingSessionState],
    user_states: dict[str, _StreamingUserState],
    first_purchase_times: dict[str, pd.Timestamp],
    split_boundaries: tuple[pd.Timestamp | None, pd.Timestamp | None],
    policy: FeatureEngineeringPolicy,
) -> dict[str, object] | None:
    session_id = str(row["user_session"])
    user_id = _string_or_missing(row["user_id"])
    cutoff_time = row["_event_time"]
    first_purchase_time = first_purchase_times.get(session_id)
    session_state = session_states.setdefault(session_id, _StreamingSessionState())
    user_state = user_states.setdefault(user_id, _StreamingUserState())
    history = user_state.history_before(cutoff_time)

    should_exclude = (
        policy.exclude_at_or_after_first_purchase
        and first_purchase_time is not None
        and cutoff_time >= first_purchase_time
    )
    if session_state.session_start_time is None:
        session_state.session_start_time = cutoff_time
    session_state.position += 1

    if should_exclude:
        user_state.update(row)
        session_states.pop(session_id, None)
        return None

    gap_minutes = _minutes_between(session_state.previous_event_time, cutoff_time)
    session_state.prefix.add(
        row,
        session_state.previous_event_time,
        session_state.session_start_time,
    )
    next_purchase_time = (
        first_purchase_time
        if first_purchase_time is not None
        and cutoff_time < first_purchase_time <= cutoff_time + pd.Timedelta(
            minutes=policy.prediction_window_minutes
        )
        else None
    )
    prefix = session_state.prefix
    feature_row = {
        "sample_id": _sample_id(int(row["_source_order"])),
        "user_session": session_id,
        "user_id": user_id,
        "cutoff_time": cutoff_time.isoformat(),
        "split": _split_for_time(cutoff_time, split_boundaries),
        "session_position": session_state.position,
        "prefix_length": prefix.event_count,
        "last_event_type": str(row["event_type"]),
        "session_elapsed_minutes": _minutes_between(
            session_state.session_start_time,
            cutoff_time,
        ),
        "time_since_previous_event_minutes": gap_minutes,
        "hour": int(cutoff_time.hour),
        "day_of_week": int(cutoff_time.dayofweek),
        "event_count_view": prefix.event_type_counts.get("view", 0),
        "event_count_cart": prefix.event_type_counts.get("cart", 0),
        "event_count_remove_from_cart": prefix.event_type_counts.get(
            "remove_from_cart",
            0,
        ),
        "event_count_purchase": prefix.event_type_counts.get("purchase", 0),
        "unique_product_count": len(prefix.product_ids),
        "unique_category_count": len(prefix.category_ids),
        "unique_brand_count": len(prefix.brands),
        "avg_price": prefix.avg_price,
        "max_price": prefix.max_price,
        "last_price": prefix.last_price,
        "last_price_bin": prefix.last_price_bin,
        "event_type_sequence": SEQUENCE_DELIMITER.join(prefix.event_types),
        "product_id_sequence": SEQUENCE_DELIMITER.join(prefix.product_ids_sequence),
        "category_id_sequence": SEQUENCE_DELIMITER.join(prefix.category_ids_sequence),
        "price_bin_sequence": SEQUENCE_DELIMITER.join(prefix.price_bins_sequence),
        "time_gap_minutes_sequence": SEQUENCE_DELIMITER.join(prefix.time_gaps_sequence),
        "user_past_event_count": history["user_past_event_count"],
        "user_past_session_count": history["user_past_session_count"],
        "user_past_purchase_count": history["user_past_purchase_count"],
        "user_past_cart_count": history["user_past_cart_count"],
        "user_minutes_since_last_event": history["user_minutes_since_last_event"],
        "label": int(next_purchase_time is not None),
        "minutes_until_purchase": _minutes_between(cutoff_time, next_purchase_time),
    }
    session_state.previous_event_time = cutoff_time
    user_state.update(row)
    return feature_row


def _append_feature_rows(
    output_path: Path,
    rows: list[dict[str, object]],
    header: bool,
) -> None:
    pd.DataFrame(rows, columns=_feature_column_order()).to_csv(
        output_path,
        mode="w" if header else "a",
        header=header,
        index=False,
        encoding="utf-8",
    )


def _append_rows_to_csv(
    output_path: Path,
    rows: list[dict[str, object]],
    columns: list[str],
    header: bool,
) -> None:
    pd.DataFrame(rows, columns=columns).to_csv(
        output_path,
        mode="w" if header else "a",
        header=header,
        index=False,
        encoding="utf-8",
    )


def _append_sequence_rows_to_parquet(
    output_path: Path,
    rows: list[dict[str, object]],
    columns: list[str],
    writer: object,
) -> object:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(
        pd.DataFrame(rows, columns=columns),
        preserve_index=False,
    )
    if writer is None:
        writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
    writer.write_table(table)
    return writer


def _split_for_time(
    event_time: pd.Timestamp,
    split_boundaries: tuple[pd.Timestamp | None, pd.Timestamp | None],
) -> str:
    train_boundary, validation_boundary = split_boundaries
    if train_boundary is not None and event_time <= train_boundary:
        return "train"
    if validation_boundary is not None and event_time <= validation_boundary:
        return "validation"
    return "test"


@dataclass
class _PrefixAccumulator:
    event_count: int = 0
    event_types: list[str] | None = None
    product_ids_sequence: list[str] | None = None
    category_ids_sequence: list[str] | None = None
    price_bins_sequence: list[str] | None = None
    time_gaps_sequence: list[str] | None = None
    event_type_counts: dict[str, int] | None = None
    product_ids: set[str] | None = None
    category_ids: set[str] | None = None
    brands: set[str] | None = None
    price_sum: float = 0.0
    price_count: int = 0
    max_price: float | None = None
    last_price: float | None = None
    last_price_bin: str = "<missing>"

    def __post_init__(self) -> None:
        self.event_types = []
        self.product_ids_sequence = []
        self.category_ids_sequence = []
        self.price_bins_sequence = []
        self.time_gaps_sequence = []
        self.event_type_counts = {}
        self.product_ids = set()
        self.category_ids = set()
        self.brands = set()

    @property
    def avg_price(self) -> float | None:
        if self.price_count == 0:
            return None
        return self.price_sum / self.price_count

    def add(
        self,
        row: pd.Series,
        previous_event_time: pd.Timestamp | None,
        session_start_time: pd.Timestamp,
    ) -> None:
        event_type = str(row["event_type"])
        product_id = _string_or_missing(row["product_id"])
        category_id = _string_or_missing(row["category_id"])
        brand = _string_or_missing(row["brand"])
        price = row["_price"]
        price_bin = _price_bin(price)
        gap_minutes = _minutes_between(previous_event_time, row["_event_time"])

        self.event_count += 1
        self.event_types.append(event_type)
        self.product_ids_sequence.append(product_id)
        self.category_ids_sequence.append(category_id)
        self.price_bins_sequence.append(price_bin)
        self.time_gaps_sequence.append(_format_float(gap_minutes))
        self.event_type_counts[event_type] = self.event_type_counts.get(event_type, 0) + 1
        if product_id != "<missing>":
            self.product_ids.add(product_id)
        if category_id != "<missing>":
            self.category_ids.add(category_id)
        if brand != "<missing>":
            self.brands.add(brand)
        if pd.notna(price):
            price_value = float(price)
            self.price_sum += price_value
            self.price_count += 1
            self.max_price = price_value if self.max_price is None else max(self.max_price, price_value)
            self.last_price = price_value
        self.last_price_bin = price_bin


def _prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"event_time", "event_type", "user_session"}
    missing_columns = required_columns - set(events.columns)
    if missing_columns:
        raise ValueError(f"Feature Engineering 필수 컬럼 누락: {', '.join(sorted(missing_columns))}")

    working = events.copy()
    for column in (
        "product_id",
        "category_id",
        "category_code",
        "brand",
        "price",
        "user_id",
    ):
        if column not in working:
            working[column] = pd.NA

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
        return working

    working["user_session"] = working["user_session"].astype(str)
    working["event_type"] = working["event_type"].astype(str)
    working["_price"] = pd.to_numeric(working["price"], errors="coerce")
    return working.sort_values(
        ["user_session", "_event_time", "_source_order"],
        kind="mergesort",
    )


def _first_purchase_times(events: pd.DataFrame) -> dict[str, pd.Timestamp]:
    purchase_events = events.loc[events["event_type"].eq("purchase")]
    if purchase_events.empty:
        return {}
    return {
        str(session_id): event_time
        for session_id, event_time in purchase_events.groupby("user_session")[
            "_event_time"
        ].min().items()
    }


def _build_user_history(events: pd.DataFrame) -> dict[int, dict[str, object]]:
    history_by_source_order: dict[int, dict[str, object]] = {}
    user_state: dict[str, dict[str, object]] = {}
    chronological = events.sort_values(["_event_time", "_source_order"], kind="mergesort")

    for _, row in chronological.iterrows():
        source_order = int(row["_source_order"])
        user_id = _string_or_missing(row["user_id"])
        state = user_state.setdefault(
            user_id,
            {
                "event_count": 0,
                "sessions": set(),
                "purchase_count": 0,
                "cart_count": 0,
                "last_event_time": None,
            },
        )
        last_event_time = state["last_event_time"]
        history_by_source_order[source_order] = {
            "user_past_event_count": int(state["event_count"]),
            "user_past_session_count": len(state["sessions"]),
            "user_past_purchase_count": int(state["purchase_count"]),
            "user_past_cart_count": int(state["cart_count"]),
            "user_minutes_since_last_event": _minutes_between(
                last_event_time,
                row["_event_time"],
            ),
        }

        state["event_count"] = int(state["event_count"]) + 1
        state["sessions"].add(str(row["user_session"]))
        if str(row["event_type"]) == "purchase":
            state["purchase_count"] = int(state["purchase_count"]) + 1
        if str(row["event_type"]) == "cart":
            state["cart_count"] = int(state["cart_count"]) + 1
        state["last_event_time"] = row["_event_time"]
    return history_by_source_order


def _assign_time_splits(
    features: pd.DataFrame,
    policy: FeatureEngineeringPolicy,
) -> pd.Series:
    split = pd.Series("test", index=features.index, dtype="object")
    if features.empty:
        return split

    ordered_times = (
        features["split_order_time"]
        .drop_duplicates()
        .sort_values(kind="mergesort")
        .tolist()
    )
    total = len(ordered_times)
    train_end = int(total * policy.train_ratio)
    validation_end = train_end + int(total * policy.validation_ratio)

    if total and train_end == 0 and policy.train_ratio > 0:
        train_end = 1
    if total - train_end > 1 and validation_end == train_end and policy.validation_ratio > 0:
        validation_end += 1
    validation_end = min(validation_end, total)

    train_times = set(ordered_times[:train_end])
    validation_times = set(ordered_times[train_end:validation_end])
    split.loc[features["split_order_time"].isin(train_times)] = "train"
    split.loc[features["split_order_time"].isin(validation_times)] = "validation"
    return split


def _validate_split_policy(policy: FeatureEngineeringPolicy) -> None:
    if policy.train_ratio < 0 or policy.validation_ratio < 0:
        raise ValueError("train_ratio와 validation_ratio는 0 이상이어야 한다.")
    if policy.train_ratio + policy.validation_ratio > 1:
        raise ValueError("train_ratio + validation_ratio는 1 이하여야 한다.")
    _validate_max_sequence_length(policy.max_sequence_length)


def _validate_max_sequence_length(max_sequence_length: int) -> None:
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length는 1 이상이어야 한다.")


def _feature_column_order() -> list[str]:
    return [
        "sample_id",
        "user_session",
        "user_id",
        "cutoff_time",
        "split",
        *_tabular_input_columns(),
        *_sequence_input_columns(),
        "label",
        "minutes_until_purchase",
    ]


def _sample_index_columns() -> list[str]:
    return [
        "sample_id",
        "user_session",
        "user_id",
        "cutoff_time",
        "split",
        "label",
        "minutes_until_purchase",
    ]


def _tabular_artifact_columns() -> list[str]:
    return ["sample_id", *_tabular_input_columns()]


def _sequence_artifact_columns() -> list[str]:
    return ["sample_id", *_sequence_input_columns()]


def _tabular_input_columns() -> list[str]:
    return [
        "session_position",
        "prefix_length",
        "last_event_type",
        "session_elapsed_minutes",
        "time_since_previous_event_minutes",
        "hour",
        "day_of_week",
        "event_count_view",
        "event_count_cart",
        "event_count_remove_from_cart",
        "event_count_purchase",
        "unique_product_count",
        "unique_category_count",
        "unique_brand_count",
        "avg_price",
        "max_price",
        "last_price",
        "last_price_bin",
        "user_past_event_count",
        "user_past_session_count",
        "user_past_purchase_count",
        "user_past_cart_count",
        "user_minutes_since_last_event",
    ]


def _sequence_input_columns() -> list[str]:
    return [
        "event_type_sequence",
        "product_id_sequence",
        "category_id_sequence",
        "price_bin_sequence",
        "time_gap_minutes_sequence",
    ]


def _categorical_input_columns() -> list[str]:
    return ["last_event_type", "last_price_bin"]


def _numeric_input_columns() -> list[str]:
    return [
        column
        for column in _tabular_input_columns()
        if column not in set(_categorical_input_columns())
    ]


def _empty_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_feature_column_order())


def _feature_row(
    feature_name: str,
    feature_group: str,
    dtype: str,
    model_role: str,
    leakage_policy: str,
) -> dict[str, str]:
    return {
        "feature_name": feature_name,
        "feature_group": feature_group,
        "dtype": dtype,
        "model_role": model_role,
        "leakage_policy": leakage_policy,
    }


def _check_row(
    check_id: str,
    check_name: str,
    status: str,
    evidence: str,
) -> dict[str, str]:
    return {
        "check_id": check_id,
        "check_name": check_name,
        "status": status,
        "evidence": evidence,
    }


def _write_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    pd.DataFrame(list(rows)).to_csv(path, index=False, encoding="utf-8")


def _sample_index_row(row: dict[str, object]) -> dict[str, object]:
    return {column: row.get(column) for column in _sample_index_columns()}


def _tabular_feature_row(row: dict[str, object]) -> dict[str, object]:
    return {column: row.get(column) for column in _tabular_artifact_columns()}


def _sequence_feature_row(
    row: dict[str, object],
    max_sequence_length: int,
) -> dict[str, object]:
    sequence_row = {"sample_id": row.get("sample_id")}
    for column in _sequence_input_columns():
        sequence_row[column] = _tail_sequence(row.get(column), max_sequence_length)
    return sequence_row


def _tail_sequence(value: object, max_sequence_length: int) -> str:
    if value is None or pd.isna(value):
        return ""
    tokens = str(value).split(SEQUENCE_DELIMITER)
    return SEQUENCE_DELIMITER.join(tokens[-max_sequence_length:])


def _sample_id(source_order: int) -> str:
    return f"sample_{source_order:012d}"


def _price_bin(price: object) -> str:
    if pd.isna(price):
        return "<missing>"
    price_value = float(price)
    for upper_bound, label in zip(PRICE_BINS[1:], PRICE_BAND_LABELS):
        if price_value <= upper_bound:
            return str(label)
    return str(PRICE_BAND_LABELS[-1])


def _minutes_between(
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> float | None:
    if start is None or end is None:
        return None
    return (end - start).total_seconds() / 60


def _string_or_missing(value: object) -> str:
    if pd.isna(value):
        return "<missing>"
    return str(value)


def _empty_user_history() -> dict[str, object]:
    return {
        "user_past_event_count": 0,
        "user_past_session_count": 0,
        "user_past_purchase_count": 0,
        "user_past_cart_count": 0,
        "user_minutes_since_last_event": None,
    }


def _feature_dtype(feature_name: str) -> str:
    if feature_name in set(_categorical_input_columns()):
        return "category"
    return "float" if "minutes" in feature_name or "price" in feature_name else "integer"


def _sorted_unique_text(values: pd.Series) -> list[str]:
    return sorted(str(value) for value in values.dropna().unique().tolist())


def _format_ratio(value: float) -> str:
    return f"{value:.6f}"


def _format_float(value: float | None) -> str:
    if value is None:
        return "0.000000"
    return f"{value:.6f}"
