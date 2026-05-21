"""Step 9 최소 GRU sequence 모델 학습 유스케이스."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import data

from purchase_time_forecasting.baseline_modeling import compute_binary_metrics
from purchase_time_forecasting.feature_engineering import (
    EVENT_TYPES,
    SEQUENCE_DELIMITER,
)


PAD_TOKEN = "<pad>"
UNKNOWN_TOKEN = "<unknown>"
SPLITS = ("train", "validation", "test")
ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class SequenceDatasetPolicy:
    """GRU 입력 sequence dataset 생성 정책."""

    max_sequence_length: int = 50
    fit_split: str = "train"


@dataclass(frozen=True)
class GruTrainingPolicy:
    """개인 과제 범위의 최소 GRU 학습 정책."""

    max_sequence_length: int = 50
    embedding_dim: int = 8
    hidden_dim: int = 16
    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 0.001
    threshold: float = 0.5
    top_k_fraction: float = 0.1
    use_pos_weight: bool = True
    random_state: int = 42
    device: str = "auto"

    @property
    def class_imbalance_strategy(self) -> str:
        return "pos_weight" if self.use_pos_weight else "none"


@dataclass(frozen=True)
class EventTypeVocabulary:
    """event type token을 embedding index로 변환하는 vocabulary."""

    token_to_id: dict[str, int]
    pad_token: str = PAD_TOKEN
    unknown_token: str = UNKNOWN_TOKEN

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def unknown_id(self) -> int:
        return self.token_to_id[self.unknown_token]

    def encode(self, tokens: Iterable[str]) -> list[int]:
        """token sequence를 정수 ID sequence로 변환한다."""

        return [
            self.token_to_id.get(str(token), self.unknown_id)
            for token in tokens
            if str(token)
        ]


@dataclass(frozen=True)
class GruEventTypeDataset:
    """GRU classifier 학습과 평가에 필요한 event type 입력 묶음."""

    sample_ids: np.ndarray
    splits: np.ndarray
    labels: np.ndarray
    event_type_token_ids: np.ndarray
    sequence_lengths: np.ndarray
    vocabulary: EventTypeVocabulary

    @property
    def sample_count(self) -> int:
        return int(len(self.sample_ids))

    @property
    def max_sequence_length(self) -> int:
        if self.event_type_token_ids.ndim != 2:
            return 0
        return int(self.event_type_token_ids.shape[1])

    def split_mask(self, split: str) -> np.ndarray:
        """특정 split sample을 선택하는 boolean mask를 반환한다."""

        return self.splits.astype(str) == split


@dataclass(frozen=True)
class GruTrainingResult:
    """GRU 학습 산출물."""

    metrics: pd.DataFrame
    model_status: pd.DataFrame
    training_history: pd.DataFrame
    report_markdown: str


def build_event_type_vocabulary(
    sample_index: pd.DataFrame,
    sequence_features: pd.DataFrame,
    policy: SequenceDatasetPolicy | None = None,
) -> EventTypeVocabulary:
    """train split의 event type sequence만 사용해 vocabulary를 fit한다."""

    active_policy = policy or SequenceDatasetPolicy()
    _validate_policy(active_policy)
    joined = _join_sequence_inputs(sample_index, sequence_features)
    train = joined.loc[joined["split"].astype(str).eq(active_policy.fit_split)]
    observed_tokens: set[str] = set()
    for value in train["event_type_sequence"]:
        observed_tokens.update(_parse_sequence(value))

    ordered_tokens = [
        token
        for token in EVENT_TYPES
        if token in observed_tokens and token not in {PAD_TOKEN, UNKNOWN_TOKEN}
    ]
    extra_tokens = sorted(
        observed_tokens - set(ordered_tokens) - {PAD_TOKEN, UNKNOWN_TOKEN}
    )
    token_to_id = {
        PAD_TOKEN: 0,
        UNKNOWN_TOKEN: 1,
        **{
            token: index
            for index, token in enumerate([*ordered_tokens, *extra_tokens], start=2)
        },
    }
    return EventTypeVocabulary(token_to_id=token_to_id)


def build_gru_event_type_dataset(
    sample_index: pd.DataFrame,
    sequence_features: pd.DataFrame,
    policy: SequenceDatasetPolicy | None = None,
    vocabulary: EventTypeVocabulary | None = None,
) -> GruEventTypeDataset:
    """event type sequence 중심의 GRU 입력 dataset을 생성한다."""

    active_policy = policy or SequenceDatasetPolicy()
    _validate_policy(active_policy)
    joined = _join_sequence_inputs(sample_index, sequence_features)
    active_vocabulary = vocabulary or build_event_type_vocabulary(
        sample_index,
        sequence_features,
        policy=active_policy,
    )

    encoded_rows = [
        _encode_and_pad(
            value,
            vocabulary=active_vocabulary,
            max_sequence_length=active_policy.max_sequence_length,
        )
        for value in joined["event_type_sequence"]
    ]
    token_ids = np.asarray([row[0] for row in encoded_rows], dtype=np.int64)
    lengths = np.asarray([row[1] for row in encoded_rows], dtype=np.int64)

    return GruEventTypeDataset(
        sample_ids=joined["sample_id"].astype(str).to_numpy(),
        splits=joined["split"].astype(str).to_numpy(),
        labels=joined["label"].astype(int).to_numpy(dtype=np.int64),
        event_type_token_ids=token_ids,
        sequence_lengths=lengths,
        vocabulary=active_vocabulary,
    )


def load_gru_event_type_dataset(
    features_dir: Path,
    policy: SequenceDatasetPolicy | None = None,
    max_samples_per_split: int | None = None,
) -> GruEventTypeDataset:
    """Step 5 feature artifact에서 GRU event type dataset을 로드한다."""

    feature_dir = Path(features_dir)
    sample_index = pd.read_csv(feature_dir / "sample_index.csv")
    if max_samples_per_split is not None:
        sample_index = _limit_sample_index(
            sample_index,
            max_samples_per_split=max_samples_per_split,
        )
    sequence_features = pd.read_parquet(
        feature_dir / "sequence_feature_dataset.parquet"
    )
    if max_samples_per_split is not None:
        sequence_features = sequence_features.loc[
            sequence_features["sample_id"].astype(str).isin(
                set(sample_index["sample_id"].astype(str))
            )
        ].copy()
    return build_gru_event_type_dataset(
        sample_index=sample_index,
        sequence_features=sequence_features,
        policy=policy,
    )


def train_gru_classifier(
    dataset: GruEventTypeDataset,
    policy: GruTrainingPolicy | None = None,
    progress_callback: ProgressCallback | None = None,
) -> GruTrainingResult:
    """event type sequence 기반 GRU classifier를 학습하고 평가한다."""

    active_policy = policy or GruTrainingPolicy(
        max_sequence_length=dataset.max_sequence_length
    )
    _progress(progress_callback, "GRU 학습 입력 검증 시작")
    _validate_training_policy(active_policy)
    _validate_gru_dataset(dataset)
    _progress(
        progress_callback,
        f"GRU 학습 입력 검증 완료: {dataset.sample_count:,} rows",
    )
    _progress(progress_callback, _split_summary_message(dataset))

    torch.manual_seed(active_policy.random_state)
    device = _resolve_torch_device(active_policy.device)
    _progress(progress_callback, f"GRU 학습 device 설정: {device}")

    model = _GruEventTypeClassifier(
        vocabulary_size=len(dataset.vocabulary.token_to_id),
        embedding_dim=active_policy.embedding_dim,
        hidden_dim=active_policy.hidden_dim,
        pad_id=dataset.vocabulary.pad_id,
    ).to(device)
    train_labels = dataset.labels[dataset.split_mask("train")]
    loss = nn.BCEWithLogitsLoss(
        pos_weight=_pos_weight_tensor(
            train_labels,
            device=device,
            enabled=active_policy.use_pos_weight,
        )
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=active_policy.learning_rate)
    train_loader = _build_torch_loader(
        dataset,
        split="train",
            batch_size=active_policy.batch_size,
            shuffle=True,
        )
    eval_loaders = {
        split: _build_torch_loader(
            dataset,
            split=split,
            batch_size=active_policy.batch_size,
            shuffle=False,
        )
        for split in SPLITS
    }

    history_rows: list[dict[str, object]] = []
    for epoch in range(1, active_policy.epochs + 1):
        _progress(progress_callback, f"GRU epoch {epoch}/{active_policy.epochs} 학습 시작")
        train_loss = _train_one_epoch(
            model=model,
            loader=train_loader,
            loss_function=loss,
            optimizer=optimizer,
            device=device,
        )
        validation_scores = _predict_scores(
            model=model,
            loader=eval_loaders["validation"],
            device=device,
        )
        validation_labels = dataset.labels[dataset.split_mask("validation")]
        validation_metrics = compute_binary_metrics(
            validation_labels,
            validation_scores,
            threshold=active_policy.threshold,
            top_k_fraction=active_policy.top_k_fraction,
        )
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_pr_auc": validation_metrics["pr_auc"],
                "validation_roc_auc": validation_metrics["roc_auc"],
            }
        )
        _progress(
            progress_callback,
            (
                f"GRU epoch {epoch}/{active_policy.epochs} 완료: "
                f"train_loss={train_loss:.6f}, "
                f"validation_pr_auc={_format_metric(validation_metrics['pr_auc'])}"
            ),
        )

    metric_rows = []
    _progress(progress_callback, "GRU split별 평가 시작")
    for split in SPLITS:
        split_scores = _predict_scores(
            model=model,
            loader=eval_loaders[split],
            device=device,
        )
        split_labels = dataset.labels[dataset.split_mask(split)]
        metric_values = compute_binary_metrics(
            split_labels,
            split_scores,
            threshold=active_policy.threshold,
            top_k_fraction=active_policy.top_k_fraction,
        )
        metric_rows.append(
            {
                "model_name": "gru",
                "class_imbalance_strategy": active_policy.class_imbalance_strategy,
                "split": split,
                "sample_count": str(len(split_labels)),
                "positive_count": str(int(split_labels.sum()) if len(split_labels) else 0),
                **metric_values,
                "threshold": f"{active_policy.threshold:.6f}",
                "top_k_fraction": f"{active_policy.top_k_fraction:.6f}",
                "status": "evaluated",
            }
        )
        _progress(progress_callback, f"GRU {split} split 평가 완료")

    metrics = pd.DataFrame(metric_rows)
    model_status = pd.DataFrame(
        [
            {
                "model_name": "gru",
                "class_imbalance_strategy": active_policy.class_imbalance_strategy,
                "status": "trained",
                "detail": _training_detail(dataset, active_policy),
            }
        ]
    )
    training_history = pd.DataFrame(history_rows)
    result = GruTrainingResult(
        metrics=metrics,
        model_status=model_status,
        training_history=training_history,
        report_markdown=build_gru_report(metrics, model_status, training_history),
    )
    _progress(progress_callback, "GRU 학습 결과 정리 완료")
    return result


def write_gru_artifacts(
    result: GruTrainingResult,
    reports_dir: Path,
) -> None:
    """Step 9 GRU artifact를 저장한다."""

    output_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _format_metrics_for_output(result.metrics).to_csv(
        output_dir / "gru_model_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    result.model_status.to_csv(
        output_dir / "gru_model_status.csv",
        index=False,
        encoding="utf-8",
    )
    _format_history_for_output(result.training_history).to_csv(
        output_dir / "gru_training_history.csv",
        index=False,
        encoding="utf-8",
    )
    (output_dir / "gru_report.md").write_text(
        result.report_markdown,
        encoding="utf-8",
    )


def build_gru_report(
    metrics: pd.DataFrame,
    model_status: pd.DataFrame,
    training_history: pd.DataFrame,
) -> str:
    """GRU 결과 markdown 리포트를 생성한다."""

    lines = [
        "# Step 9 Minimal GRU Model",
        "",
        "## 핵심 요약",
        "",
        "- 입력 artifact: `sample_index.csv`, `sequence_feature_dataset.parquet`",
        "- 첫 구현은 event type sequence만 사용한다.",
        "- `sample_id`, `label`, `split`은 baseline과 동일 계약을 따른다.",
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

    lines.extend(["", "## Validation History", ""])
    if training_history.empty:
        lines.append("- 학습 이력 없음")
    else:
        for row in training_history.to_dict("records"):
            lines.append(
                f"- epoch {row['epoch']}: train_loss "
                f"{_format_metric(row['train_loss'])}, validation PR-AUC "
                f"{_format_metric(row['validation_pr_auc'])}"
            )

    lines.extend(["", "## Test Metrics", ""])
    test_metrics = (
        metrics.loc[metrics["split"].eq("test")]
        if not metrics.empty and "split" in metrics
        else metrics
    )
    if test_metrics.empty:
        lines.append("- test metric 없음")
    else:
        for row in test_metrics.to_dict("records"):
            lines.append(
                f"- PR-AUC {_format_metric(row['pr_auc'])}, "
                f"ROC-AUC {_format_metric(row['roc_auc'])}, "
                f"F1 {_format_metric(row['f1'])}, "
                f"Recall@K {_format_metric(row['recall_at_k'])}, "
                f"Precision@K {_format_metric(row['precision_at_k'])}"
            )
    lines.append("")
    return "\n".join(lines)


def _join_sequence_inputs(
    sample_index: pd.DataFrame,
    sequence_features: pd.DataFrame,
) -> pd.DataFrame:
    sample_required = {"sample_id", "split", "label"}
    sequence_required = {"sample_id", "event_type_sequence"}
    missing_sample = sample_required - set(sample_index.columns)
    missing_sequence = sequence_required - set(sequence_features.columns)
    if missing_sample:
        raise ValueError(f"sample_index 필수 컬럼 누락: {', '.join(sorted(missing_sample))}")
    if missing_sequence:
        raise ValueError(
            "sequence_feature_dataset 필수 컬럼 누락: "
            f"{', '.join(sorted(missing_sequence))}"
        )
    if sample_index["sample_id"].duplicated().any():
        raise ValueError("sample_index sample_id는 고유해야 한다.")
    if sequence_features["sample_id"].duplicated().any():
        raise ValueError("sequence_feature_dataset sample_id는 고유해야 한다.")

    joined = sequence_features.loc[:, ["sample_id", "event_type_sequence"]].merge(
        sample_index.loc[:, ["sample_id", "split", "label"]],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(sample_index) or len(joined) != len(sequence_features):
        raise ValueError("sample_index와 sequence_feature_dataset의 sample_id 집합이 일치하지 않는다.")

    unknown_splits = set(joined["split"].dropna().astype(str).unique()) - set(SPLITS)
    if unknown_splits:
        raise ValueError(f"지원하지 않는 split 값: {', '.join(sorted(unknown_splits))}")
    joined["label"] = pd.to_numeric(joined["label"], errors="raise").astype(int)
    return joined


def _limit_sample_index(
    sample_index: pd.DataFrame,
    max_samples_per_split: int,
) -> pd.DataFrame:
    if max_samples_per_split <= 0:
        raise ValueError("max_samples_per_split은 1 이상이어야 한다.")
    parts = []
    for split in SPLITS:
        split_frame = sample_index.loc[sample_index["split"].astype(str).eq(split)]
        parts.append(split_frame.head(max_samples_per_split))
    if not parts:
        return sample_index.head(0).copy()
    return pd.concat(parts, ignore_index=True)


def _encode_and_pad(
    value: object,
    vocabulary: EventTypeVocabulary,
    max_sequence_length: int,
) -> tuple[list[int], int]:
    tokens = _parse_sequence(value)[-max_sequence_length:]
    encoded = vocabulary.encode(tokens)
    length = len(encoded)
    padded = encoded + [vocabulary.pad_id] * (max_sequence_length - length)
    return padded, length


def _parse_sequence(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [
        token
        for token in str(value).split(SEQUENCE_DELIMITER)
        if token
    ]


def _validate_policy(policy: SequenceDatasetPolicy) -> None:
    if policy.max_sequence_length <= 0:
        raise ValueError("max_sequence_length는 1 이상이어야 한다.")
    if policy.fit_split not in SPLITS:
        raise ValueError(f"지원하지 않는 fit_split 값: {policy.fit_split}")


def _validate_training_policy(policy: GruTrainingPolicy) -> None:
    if policy.max_sequence_length <= 0:
        raise ValueError("max_sequence_length는 1 이상이어야 한다.")
    if policy.embedding_dim <= 0:
        raise ValueError("embedding_dim은 1 이상이어야 한다.")
    if policy.hidden_dim <= 0:
        raise ValueError("hidden_dim은 1 이상이어야 한다.")
    if policy.batch_size <= 0:
        raise ValueError("batch_size는 1 이상이어야 한다.")
    if policy.epochs <= 0:
        raise ValueError("epochs는 1 이상이어야 한다.")
    if policy.learning_rate <= 0:
        raise ValueError("learning_rate는 0보다 커야 한다.")
    if not 0 <= policy.threshold <= 1:
        raise ValueError("threshold는 0 이상 1 이하여야 한다.")
    if not 0 < policy.top_k_fraction <= 1:
        raise ValueError("top_k_fraction은 0보다 크고 1 이하여야 한다.")


def _resolve_torch_device(device_name: str) -> torch.device:
    requested = str(device_name).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(
            "CUDA device를 요청했지만 현재 PyTorch가 CUDA를 사용할 수 없다. "
            "CUDA 지원 PyTorch를 설치한 뒤 다시 실행해야 한다."
        )
    return torch.device(requested)


def _validate_gru_dataset(dataset: GruEventTypeDataset) -> None:
    if dataset.sample_count == 0:
        raise ValueError("GRU dataset이 비어 있다.")
    if dataset.event_type_token_ids.ndim != 2:
        raise ValueError("event_type_token_ids는 2차원 배열이어야 한다.")
    if len(dataset.labels) != dataset.sample_count:
        raise ValueError("labels 길이가 sample 수와 다르다.")
    if len(dataset.sequence_lengths) != dataset.sample_count:
        raise ValueError("sequence_lengths 길이가 sample 수와 다르다.")
    if not dataset.split_mask("train").any():
        raise ValueError("GRU 학습에는 train split sample이 필요하다.")
    train_labels = dataset.labels[dataset.split_mask("train")]
    if np.unique(train_labels).size < 2:
        raise ValueError("train split에 단일 class만 있어 GRU 학습을 진행할 수 없다.")


def _build_torch_loader(
    dataset: GruEventTypeDataset,
    split: str,
    batch_size: int,
    shuffle: bool,
):
    mask = dataset.split_mask(split)
    token_ids = torch.as_tensor(dataset.event_type_token_ids[mask], dtype=torch.long)
    lengths = torch.as_tensor(dataset.sequence_lengths[mask], dtype=torch.long)
    labels = torch.as_tensor(dataset.labels[mask], dtype=torch.float32)
    tensor_dataset = data.TensorDataset(token_ids, lengths, labels)
    return data.DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)


class _GruEventTypeClassifier(nn.Module):
    """event type sequence만 사용하는 최소 GRU classifier."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int,
        hidden_dim: int,
        pad_id: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            vocabulary_size,
            embedding_dim,
            padding_idx=pad_id,
        )
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, token_ids, lengths):
        embedded = self.embedding(token_ids)
        clamped_lengths = torch.clamp(lengths.cpu(), min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            clamped_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return self.classifier(hidden[-1]).squeeze(1)


def _pos_weight_tensor(labels: np.ndarray, device, enabled: bool):
    if not enabled:
        return None
    positive_count = int(labels.sum())
    negative_count = int(len(labels) - positive_count)
    if positive_count == 0:
        return None
    return torch.tensor([negative_count / positive_count], dtype=torch.float32).to(device)


def _train_one_epoch(
    model,
    loader,
    loss_function,
    optimizer,
    device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for token_ids, lengths, labels in loader:
        token_ids = token_ids.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(token_ids, lengths)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = int(labels.numel())
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
    return total_loss / total_count if total_count else float("nan")


def _predict_scores(
    model,
    loader,
    device,
) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for token_ids, lengths, _ in loader:
            token_ids = token_ids.to(device)
            lengths = lengths.to(device)
            logits = model(token_ids, lengths)
            batch_scores = torch.sigmoid(logits).detach().cpu().numpy()
            scores.append(batch_scores)
    if not scores:
        return np.asarray([], dtype=float)
    return np.concatenate(scores).astype(float)


def _split_summary_message(dataset: GruEventTypeDataset) -> str:
    parts = []
    for split in SPLITS:
        labels = dataset.labels[dataset.split_mask(split)]
        parts.append(
            f"{split}={len(labels):,} rows, positive={int(labels.sum()) if len(labels) else 0:,}"
        )
    return "split별 GRU 학습 데이터 분포: " + " / ".join(parts)


def _training_detail(dataset: GruEventTypeDataset, policy: GruTrainingPolicy) -> str:
    return (
        f"event_type_only, vocab_size={len(dataset.vocabulary.token_to_id)}, "
        f"max_sequence_length={dataset.max_sequence_length}, "
        f"embedding_dim={policy.embedding_dim}, hidden_dim={policy.hidden_dim}, "
        f"epochs={policy.epochs}, batch_size={policy.batch_size}, "
        f"learning_rate={policy.learning_rate:.6f}, device={policy.device}"
    )


def _progress(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _format_metric(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.6f}"


def _format_metrics_for_output(metrics: pd.DataFrame) -> pd.DataFrame:
    formatted = metrics.copy()
    for column in ("pr_auc", "roc_auc", "f1", "recall_at_k", "precision_at_k"):
        if column in formatted:
            formatted[column] = formatted[column].map(_format_metric)
    return formatted


def _format_history_for_output(history: pd.DataFrame) -> pd.DataFrame:
    formatted = history.copy()
    for column in ("train_loss", "validation_pr_auc", "validation_roc_auc"):
        if column in formatted:
            formatted[column] = formatted[column].map(_format_metric)
    return formatted
