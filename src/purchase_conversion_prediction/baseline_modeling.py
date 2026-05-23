"""Step 7 baseline 모델 학습과 평가 유스케이스."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_FEATURES = ("last_event_type", "last_price_bin")
METADATA_COLUMNS = {
    "sample_id",
    "user_session",
    "user_id",
    "cutoff_time",
    "split",
    "label",
    "minutes_until_purchase",
}
SPLITS = ("train", "validation", "test")
ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class BaselineTrainingPolicy:
    """baseline 학습 정책."""

    threshold: float = 0.5
    top_k_fraction: float = 0.1
    class_imbalance_strategies: tuple[str, ...] = ("none", "balanced")
    logistic_max_iter: int = 300
    train_lightgbm: bool = True
    lightgbm_n_estimators: int = 200
    random_state: int = 42


@dataclass(frozen=True)
class BaselineTrainingResult:
    """baseline 학습 산출물."""

    metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    model_status: pd.DataFrame
    report_markdown: str


@dataclass(frozen=True)
class TabularPreprocessor:
    """train split 기준으로 fit한 sklearn tabular 전처리기."""

    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    transformer: ColumnTransformer

    @classmethod
    def fit(cls, train: pd.DataFrame) -> "TabularPreprocessor":
        feature_columns = _feature_columns(train)
        numeric_features = tuple(
            column for column in feature_columns if column not in CATEGORICAL_FEATURES
        )
        categorical_features = tuple(
            column for column in CATEGORICAL_FEATURES if column in train.columns
        )
        transformer = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline(
                        steps=[
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="median",
                                    keep_empty_features=True,
                                ),
                            ),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    list(numeric_features),
                ),
                (
                    "categorical",
                    Pipeline(
                        steps=[
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="constant",
                                    fill_value="<missing>",
                                    keep_empty_features=True,
                                ),
                            ),
                            (
                                "encoder",
                                OneHotEncoder(
                                    handle_unknown="ignore",
                                    sparse_output=False,
                                ),
                            ),
                        ]
                    ),
                    list(categorical_features),
                ),
            ],
            remainder="drop",
        )
        transformer.fit(train.loc[:, [*numeric_features, *categorical_features]])
        return cls(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            transformer=transformer,
        )

    @property
    def feature_names(self) -> list[str]:
        names = list(self.numeric_features)
        for column, values in self.categorical_values.items():
            for value in values:
                names.append(f"{column}__{value}")
        return names

    @property
    def categorical_values(self) -> dict[str, tuple[str, ...]]:
        if not self.categorical_features:
            return {}
        encoder = self.transformer.named_transformers_["categorical"].named_steps[
            "encoder"
        ]
        return {
            column: tuple(str(value) for value in values)
            for column, values in zip(self.categorical_features, encoder.categories_)
        }

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        feature_frame = frame.loc[
            :, [*self.numeric_features, *self.categorical_features]
        ]
        transformed = self.transformer.transform(feature_frame)

        return pd.DataFrame(
            np.asarray(transformed, dtype=float),
            index=frame.index,
            columns=self.feature_names,
        )


def build_baseline_dataset(
    sample_index: pd.DataFrame,
    tabular_features: pd.DataFrame,
) -> pd.DataFrame:
    """`sample_id` 기준으로 baseline 학습 dataset을 생성한다."""

    _validate_baseline_inputs(sample_index, tabular_features)
    joined = tabular_features.merge(
        sample_index.loc[:, ["sample_id", "split", "label"]],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(sample_index) or len(joined) != len(tabular_features):
        raise ValueError("sample_index와 tabular_feature_dataset의 sample_id 집합이 일치하지 않는다.")
    joined["label"] = pd.to_numeric(joined["label"], errors="raise").astype(int)
    return joined.loc[:, ["sample_id", *_feature_columns(joined), "split", "label"]]


def load_baseline_dataset(
    features_dir: Path,
    max_samples_per_split: int | None = None,
    chunksize: int = 200_000,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """feature artifact에서 baseline 학습 dataset을 로드한다."""

    feature_dir = Path(features_dir)
    sample_index_path = feature_dir / "sample_index.csv"
    tabular_path = feature_dir / "tabular_feature_dataset.csv"
    if max_samples_per_split is None:
        _progress(progress_callback, "sample_index.csv 전체 로드 시작")
        sample_index = pd.read_csv(sample_index_path)
        _progress(
            progress_callback,
            f"sample_index.csv 로드 완료: {len(sample_index):,} rows",
        )
        _progress(progress_callback, "tabular_feature_dataset.csv 전체 로드 시작")
        tabular = pd.read_csv(tabular_path)
        _progress(
            progress_callback,
            f"tabular_feature_dataset.csv 로드 완료: {len(tabular):,} rows",
        )
        _progress(progress_callback, "baseline dataset join 시작")
        dataset = build_baseline_dataset(sample_index, tabular)
        _progress(
            progress_callback,
            f"baseline dataset join 완료: {len(dataset):,} rows",
        )
        return dataset

    _progress(
        progress_callback,
        f"sample_index.csv 제한 로드 시작: split별 최대 {max_samples_per_split:,} rows",
    )
    sample_index = _read_sample_index_limited(
        sample_index_path,
        max_samples_per_split=max_samples_per_split,
        chunksize=chunksize,
    )
    _progress(
        progress_callback,
        f"sample_index.csv 제한 로드 완료: {len(sample_index):,} rows",
    )
    _progress(progress_callback, "tabular_feature_dataset.csv 제한 로드 시작")
    tabular = _read_tabular_for_sample_ids(
        tabular_path,
        set(sample_index["sample_id"].astype(str)),
        chunksize=chunksize,
    )
    _progress(
        progress_callback,
        f"tabular_feature_dataset.csv 제한 로드 완료: {len(tabular):,} rows",
    )
    _progress(progress_callback, "baseline dataset join 시작")
    dataset = build_baseline_dataset(sample_index, tabular)
    _progress(
        progress_callback,
        f"baseline dataset join 완료: {len(dataset):,} rows",
    )
    return dataset


def train_baseline_models(
    dataset: pd.DataFrame,
    policy: BaselineTrainingPolicy | None = None,
    progress_callback: ProgressCallback | None = None,
) -> BaselineTrainingResult:
    """Logistic Regression과 LightGBM baseline을 학습하고 평가한다."""

    active_policy = policy or BaselineTrainingPolicy()
    _progress(progress_callback, "baseline 학습 입력 검증 시작")
    _validate_policy(active_policy)
    _validate_training_dataset(dataset)
    _progress(
        progress_callback,
        f"baseline 학습 입력 검증 완료: {len(dataset):,} rows",
    )

    train = dataset.loc[dataset["split"].eq("train")].copy()
    if train.empty:
        raise ValueError("baseline 학습에는 train split sample이 필요하다.")
    _progress(progress_callback, _split_summary_message(dataset))

    _progress(progress_callback, "tabular 전처리기 fit 시작")
    preprocessor = TabularPreprocessor.fit(train)
    _progress(
        progress_callback,
        f"tabular 전처리기 fit 완료: {len(preprocessor.feature_names):,} features",
    )
    transformed_by_split = {}
    for split in SPLITS:
        _progress(progress_callback, f"{split} split 전처리 transform 시작")
        transformed = preprocessor.transform(dataset.loc[dataset["split"].eq(split)])
        transformed_by_split[split] = transformed
        _progress(
            progress_callback,
            f"{split} split 전처리 transform 완료: {len(transformed):,} rows",
        )
    labels_by_split = {
        split: dataset.loc[dataset["split"].eq(split), "label"].astype(int).to_numpy()
        for split in SPLITS
    }

    metric_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []

    for strategy in active_policy.class_imbalance_strategies:
        _progress(
            progress_callback,
            f"Logistic Regression 학습 시작: strategy={strategy}",
        )
        logistic_model, detail = _fit_logistic_regression(
            transformed_by_split["train"],
            labels_by_split["train"],
            strategy=strategy,
            policy=active_policy,
        )
        if logistic_model is None:
            skip_status = "skipped_invalid_training_data"
            _progress(progress_callback, f"Logistic Regression 학습 생략: {detail}")
            status_rows.append(
                _status_row("logistic_regression", strategy, skip_status, detail)
            )
            metric_rows.extend(
                _skipped_metric_rows(
                    "logistic_regression",
                    strategy,
                    labels_by_split,
                    active_policy,
                    skip_status,
                )
            )
            continue

        _progress(progress_callback, f"Logistic Regression 학습 완료: {detail}")
        status_rows.append(
            _status_row("logistic_regression", strategy, "trained", detail)
        )
        _progress(
            progress_callback,
            f"Logistic Regression 평가 시작: strategy={strategy}",
        )
        metric_rows.extend(
            _evaluate_model(
                model_name="logistic_regression",
                strategy=strategy,
                transformed_by_split=transformed_by_split,
                labels_by_split=labels_by_split,
                predict_scores=lambda frame, model=logistic_model: model.predict_proba(
                    frame
                )[:, 1],
                policy=active_policy,
            )
        )
        _progress(
            progress_callback,
            f"Logistic Regression 평가 완료: strategy={strategy}",
        )
        importance_rows.extend(
            _coefficient_importance_rows(
                "logistic_regression",
                strategy,
                preprocessor.feature_names,
                np.asarray(logistic_model.coef_[0], dtype=float),
            )
        )

        if not active_policy.train_lightgbm:
            continue

        _progress(progress_callback, f"LightGBM 학습 시작: strategy={strategy}")
        lightgbm_model, detail = _fit_lightgbm(
            transformed_by_split["train"],
            labels_by_split["train"],
            strategy,
            active_policy,
        )
        if lightgbm_model is None:
            skip_status = (
                "skipped_missing_dependency"
                if "패키지" in detail
                else "skipped_invalid_training_data"
            )
            _progress(progress_callback, f"LightGBM 학습 생략: {detail}")
            status_rows.append(
                _status_row("lightgbm", strategy, skip_status, detail)
            )
            metric_rows.extend(
                _skipped_metric_rows(
                    "lightgbm",
                    strategy,
                    labels_by_split,
                    active_policy,
                    skip_status,
                )
            )
            continue

        _progress(progress_callback, f"LightGBM 학습 완료: {detail}")
        status_rows.append(_status_row("lightgbm", strategy, "trained", detail))
        _progress(progress_callback, f"LightGBM 평가 시작: strategy={strategy}")
        metric_rows.extend(
            _evaluate_model(
                model_name="lightgbm",
                strategy=strategy,
                transformed_by_split=transformed_by_split,
                labels_by_split=labels_by_split,
                predict_scores=lambda frame, model=lightgbm_model: model.predict_proba(
                    frame
                )[:, 1],
                policy=active_policy,
            )
        )
        _progress(progress_callback, f"LightGBM 평가 완료: strategy={strategy}")
        importance_rows.extend(
            _lightgbm_importance_rows(
                "lightgbm",
                strategy,
                preprocessor.feature_names,
                lightgbm_model,
            )
        )

    _progress(progress_callback, "baseline 학습 결과 정리 시작")
    metrics = pd.DataFrame(metric_rows)
    feature_importance = pd.DataFrame(importance_rows)
    model_status = pd.DataFrame(status_rows)
    result = BaselineTrainingResult(
        metrics=metrics,
        feature_importance=feature_importance,
        model_status=model_status,
        report_markdown=build_baseline_report(metrics, model_status),
    )
    _progress(progress_callback, "baseline 학습 결과 정리 완료")
    return result


def compute_binary_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: float = 0.5,
    top_k_fraction: float = 0.1,
) -> dict[str, float]:
    """binary classification metric을 계산한다."""

    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    if len(y) != len(scores):
        raise ValueError("y_true와 y_score 길이가 다르다.")
    if len(y) == 0:
        return _empty_metric_values()

    predicted = scores >= threshold

    top_k_count = max(1, int(np.ceil(len(y) * top_k_fraction)))
    top_indices = np.argsort(-scores, kind="mergesort")[:top_k_count]
    positives_in_top_k = int(y[top_indices].sum())
    recall_at_k = _safe_divide(positives_in_top_k, int(y.sum()))
    precision_at_k = _safe_divide(positives_in_top_k, top_k_count)

    return {
        "pr_auc": _average_precision(y, scores),
        "roc_auc": _roc_auc(y, scores),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "recall_at_k": recall_at_k,
        "precision_at_k": precision_at_k,
    }


def write_baseline_artifacts(
    result: BaselineTrainingResult,
    reports_dir: Path,
) -> None:
    """Step 7 baseline artifact를 저장한다."""

    output_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _format_metrics_for_output(result.metrics).to_csv(
        output_dir / "model_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    result.feature_importance.to_csv(
        output_dir / "baseline_feature_importance.csv",
        index=False,
        encoding="utf-8",
    )
    result.model_status.to_csv(
        output_dir / "baseline_model_status.csv",
        index=False,
        encoding="utf-8",
    )
    (output_dir / "baseline_report.md").write_text(
        result.report_markdown,
        encoding="utf-8",
    )


def build_baseline_report(
    metrics: pd.DataFrame,
    model_status: pd.DataFrame,
) -> str:
    """baseline 결과 markdown 리포트를 생성한다."""

    lines = [
        "# Step 7 Baseline Models",
        "",
        "## 핵심 요약",
        "",
        "- 입력 artifact: `sample_index.csv`, `tabular_feature_dataset.csv`",
        "- `sample_id`로 label/split을 join하고 audit metadata는 학습 입력에서 제외한다.",
        "- categorical feature는 train split에서 관측한 값만 one-hot 인코딩한다.",
        "- numeric 결측 대체와 표준화 통계는 train split에서만 계산한다.",
        "",
        "## 모델 상태",
        "",
    ]
    if model_status.empty:
        lines.append("- 학습된 모델 없음")
    else:
        for row in model_status.to_dict("records"):
            lines.append(
                f"- {row['model_name']} / {row['class_imbalance_strategy']}: "
                f"{row['status']} ({row['detail']})"
            )

    lines.extend(["", "## Validation PR-AUC", ""])
    validation = metrics.loc[metrics["split"].eq("validation")] if not metrics.empty else metrics
    if validation.empty:
        lines.append("- validation metric 없음")
    else:
        for row in validation.to_dict("records"):
            lines.append(
                f"- {row['model_name']} / {row['class_imbalance_strategy']}: "
                f"PR-AUC {_format_metric(row['pr_auc'])}, "
                f"ROC-AUC {_format_metric(row['roc_auc'])}, "
                f"F1 {_format_metric(row['f1'])}"
            )
    lines.append("")
    return "\n".join(lines)


def _validate_baseline_inputs(
    sample_index: pd.DataFrame,
    tabular_features: pd.DataFrame,
) -> None:
    sample_required = {"sample_id", "split", "label"}
    tabular_required = {"sample_id"}
    missing_sample = sample_required - set(sample_index.columns)
    missing_tabular = tabular_required - set(tabular_features.columns)
    if missing_sample:
        raise ValueError(f"sample_index 필수 컬럼 누락: {', '.join(sorted(missing_sample))}")
    if missing_tabular:
        raise ValueError(
            f"tabular_feature_dataset 필수 컬럼 누락: {', '.join(sorted(missing_tabular))}"
        )
    if sample_index["sample_id"].duplicated().any():
        raise ValueError("sample_index sample_id는 고유해야 한다.")
    if tabular_features["sample_id"].duplicated().any():
        raise ValueError("tabular_feature_dataset sample_id는 고유해야 한다.")


def _validate_training_dataset(dataset: pd.DataFrame) -> None:
    required = {"sample_id", "split", "label"}
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"baseline dataset 필수 컬럼 누락: {', '.join(sorted(missing))}")
    if dataset.empty:
        raise ValueError("baseline dataset이 비어 있다.")
    unknown_splits = set(dataset["split"].dropna().astype(str).unique()) - set(SPLITS)
    if unknown_splits:
        raise ValueError(f"지원하지 않는 split 값: {', '.join(sorted(unknown_splits))}")


def _validate_policy(policy: BaselineTrainingPolicy) -> None:
    if not 0 < policy.top_k_fraction <= 1:
        raise ValueError("top_k_fraction은 0보다 크고 1 이하여야 한다.")
    if not 0 <= policy.threshold <= 1:
        raise ValueError("threshold는 0 이상 1 이하여야 한다.")
    allowed = {"none", "balanced"}
    unknown = set(policy.class_imbalance_strategies) - allowed
    if unknown:
        raise ValueError(f"지원하지 않는 class imbalance 전략: {', '.join(sorted(unknown))}")


def _progress(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _split_summary_message(dataset: pd.DataFrame) -> str:
    parts = []
    for split in SPLITS:
        labels = dataset.loc[dataset["split"].eq(split), "label"].astype(int)
        parts.append(
            f"{split}={len(labels):,} rows, positive={int(labels.sum()):,}"
        )
    return "split별 학습 데이터 분포: " + " / ".join(parts)


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in METADATA_COLUMNS]


def _read_sample_index_limited(
    path: Path,
    max_samples_per_split: int,
    chunksize: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    positive_target = max(1, max_samples_per_split // 2)
    negative_target = max_samples_per_split - positive_target
    counts = {
        split: {0: 0, 1: 0}
        for split in SPLITS
    }
    targets = {0: negative_target, 1: positive_target}
    for chunk in pd.read_csv(path, chunksize=chunksize):
        selected_parts = []
        for split in SPLITS:
            for label in (0, 1):
                remaining = targets[label] - counts[split][label]
                if remaining <= 0:
                    continue
                selected = chunk.loc[
                    chunk["split"].eq(split) & chunk["label"].eq(label)
                ].head(remaining)
                counts[split][label] += len(selected)
                selected_parts.append(selected)
        if selected_parts:
            frames.append(pd.concat(selected_parts, ignore_index=True))
        if all(
            counts[split][label] >= targets[label]
            for split in SPLITS
            for label in (0, 1)
        ):
            break
    if not frames:
        return pd.DataFrame(columns=["sample_id", "split", "label"])
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["split", "label", "sample_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def _read_tabular_for_sample_ids(
    path: Path,
    sample_ids: set[str],
    chunksize: int,
) -> pd.DataFrame:
    frames = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        selected = chunk.loc[chunk["sample_id"].astype(str).isin(sample_ids)]
        if not selected.empty:
            frames.append(selected)
    if not frames:
        return pd.DataFrame(columns=["sample_id"])
    return pd.concat(frames, ignore_index=True)


def _evaluate_model(
    model_name: str,
    strategy: str,
    transformed_by_split: dict[str, pd.DataFrame],
    labels_by_split: dict[str, np.ndarray],
    predict_scores,
    policy: BaselineTrainingPolicy,
) -> list[dict[str, object]]:
    rows = []
    for split in SPLITS:
        labels = labels_by_split[split]
        scores = predict_scores(transformed_by_split[split]) if len(labels) else []
        metric_values = compute_binary_metrics(
            labels,
            scores,
            threshold=policy.threshold,
            top_k_fraction=policy.top_k_fraction,
        )
        rows.append(
            {
                "model_name": model_name,
                "class_imbalance_strategy": strategy,
                "split": split,
                "sample_count": str(len(labels)),
                "positive_count": str(int(labels.sum()) if len(labels) else 0),
                **metric_values,
                "threshold": f"{policy.threshold:.6f}",
                "top_k_fraction": f"{policy.top_k_fraction:.6f}",
                "status": "evaluated",
            }
        )
    return rows


def _skipped_metric_rows(
    model_name: str,
    strategy: str,
    labels_by_split: dict[str, np.ndarray],
    policy: BaselineTrainingPolicy,
    status: str,
) -> list[dict[str, object]]:
    rows = []
    for split in SPLITS:
        labels = labels_by_split[split]
        rows.append(
            {
                "model_name": model_name,
                "class_imbalance_strategy": strategy,
                "split": split,
                "sample_count": str(len(labels)),
                "positive_count": str(int(labels.sum()) if len(labels) else 0),
                **_empty_metric_values(),
                "threshold": f"{policy.threshold:.6f}",
                "top_k_fraction": f"{policy.top_k_fraction:.6f}",
                "status": status,
            }
        )
    return rows


def _fit_lightgbm(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    strategy: str,
    policy: BaselineTrainingPolicy,
) -> tuple[object | None, str]:
    if np.unique(y_train).size < 2:
        return None, "train split에 단일 class만 있어 LightGBM 학습을 생략한다."
    try:
        from lightgbm import LGBMClassifier
    except ModuleNotFoundError:
        return None, "lightgbm 패키지가 설치되어 있지 않다."

    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)
    scale_pos_weight = (
        negative_count / positive_count
        if strategy == "balanced" and positive_count
        else 1.0
    )
    model = LGBMClassifier(
        objective="binary",
        n_estimators=policy.lightgbm_n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=policy.random_state,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_train, y_train)
    return model, f"lightgbm scale_pos_weight={scale_pos_weight:.6f}"


def _fit_logistic_regression(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    strategy: str,
    policy: BaselineTrainingPolicy,
) -> tuple[LogisticRegression | None, str]:
    if np.unique(y_train).size < 2:
        return None, "train split에 단일 class만 있어 Logistic Regression 학습을 생략한다."

    class_weight = "balanced" if strategy == "balanced" else None
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=policy.logistic_max_iter,
        solver="lbfgs",
        random_state=policy.random_state,
    )
    model.fit(x_train, y_train)
    return model, f"sklearn class_weight={class_weight or 'none'}"


def _coefficient_importance_rows(
    model_name: str,
    strategy: str,
    feature_names: list[str],
    coefficients: np.ndarray,
) -> list[dict[str, object]]:
    importance = np.abs(coefficients)
    order = np.argsort(-importance, kind="mergesort")
    rows = []
    for rank, index in enumerate(order, start=1):
        rows.append(
            {
                "model_name": model_name,
                "class_imbalance_strategy": strategy,
                "feature_name": feature_names[index],
                "importance": f"{importance[index]:.10f}",
                "importance_type": "abs_coefficient",
                "rank": str(rank),
                "status": "estimated",
            }
        )
    return rows


def _lightgbm_importance_rows(
    model_name: str,
    strategy: str,
    feature_names: list[str],
    model: object,
) -> list[dict[str, object]]:
    values = np.asarray(model.feature_importances_, dtype=float)
    order = np.argsort(-values, kind="mergesort")
    return [
        {
            "model_name": model_name,
            "class_imbalance_strategy": strategy,
            "feature_name": feature_names[index],
            "importance": f"{values[index]:.10f}",
            "importance_type": "split_count",
            "rank": str(rank),
            "status": "estimated",
        }
        for rank, index in enumerate(order, start=1)
    ]


def _status_row(
    model_name: str,
    strategy: str,
    status: str,
    detail: str,
) -> dict[str, str]:
    return {
        "model_name": model_name,
        "class_imbalance_strategy": strategy,
        "status": status,
        "detail": detail,
    }


def _average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    if int(y_true.sum()) == 0:
        return np.nan
    return float(average_precision_score(y_true, scores))


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return np.nan
    return float(roc_auc_score(y_true, scores))


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _format_metric(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.6f}"


def _format_metrics_for_output(metrics: pd.DataFrame) -> pd.DataFrame:
    formatted = metrics.copy()
    for column in ("pr_auc", "roc_auc", "f1", "recall_at_k", "precision_at_k"):
        if column in formatted:
            formatted[column] = formatted[column].map(_format_metric)
    return formatted


def _empty_metric_values() -> dict[str, float]:
    return {
        "pr_auc": np.nan,
        "roc_auc": np.nan,
        "f1": np.nan,
        "recall_at_k": np.nan,
        "precision_at_k": np.nan,
    }
