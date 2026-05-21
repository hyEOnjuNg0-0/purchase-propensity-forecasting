"""Step 9 GRU 학습 전 sequence dataset 구성 로직."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from purchase_time_forecasting.feature_engineering import (
    EVENT_TYPES,
    SEQUENCE_DELIMITER,
)


PAD_TOKEN = "<pad>"
UNKNOWN_TOKEN = "<unknown>"
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class SequenceDatasetPolicy:
    """GRU 입력 sequence dataset 생성 정책."""

    max_sequence_length: int = 50
    fit_split: str = "train"


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
    """GRU classifier 학습 직전까지 필요한 event type 입력 묶음."""

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
) -> GruEventTypeDataset:
    """Step 5 feature artifact에서 GRU event type dataset을 로드한다."""

    feature_dir = Path(features_dir)
    sample_index = pd.read_csv(feature_dir / "sample_index.csv")
    sequence_features = pd.read_parquet(
        feature_dir / "sequence_feature_dataset.parquet"
    )
    return build_gru_event_type_dataset(
        sample_index=sample_index,
        sequence_features=sequence_features,
        policy=policy,
    )


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
