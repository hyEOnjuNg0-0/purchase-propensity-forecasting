from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402

from purchase_time_forecasting.sequence_modeling import (  # noqa: E402
    PAD_TOKEN,
    UNKNOWN_TOKEN,
    EventTypeVocabulary,
    GruTrainingPolicy,
    SequenceDatasetPolicy,
    build_event_type_vocabulary,
    build_gru_event_type_dataset,
    load_gru_event_type_dataset,
    train_gru_classifier,
    write_gru_artifacts,
)


def _sample_index() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"sample_id": "s1", "split": "train", "label": 0},
            {"sample_id": "s2", "split": "train", "label": 1},
            {"sample_id": "s3", "split": "validation", "label": 0},
            {"sample_id": "s4", "split": "test", "label": 1},
        ]
    )


def _sequence_features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"sample_id": "s1", "event_type_sequence": "view"},
            {"sample_id": "s2", "event_type_sequence": "view cart remove_from_cart"},
            {"sample_id": "s3", "event_type_sequence": "view purchase"},
            {"sample_id": "s4", "event_type_sequence": "view cart purchase"},
        ]
    )


def test_event_type_vocabulary_uses_train_split_only() -> None:
    vocabulary = build_event_type_vocabulary(_sample_index(), _sequence_features())

    assert vocabulary.token_to_id[PAD_TOKEN] == 0
    assert vocabulary.token_to_id[UNKNOWN_TOKEN] == 1
    assert vocabulary.token_to_id["view"] == 2
    assert vocabulary.token_to_id["cart"] == 3
    assert vocabulary.token_to_id["remove_from_cart"] == 4
    assert "purchase" not in vocabulary.token_to_id


def test_gru_event_type_dataset_pads_sequences_and_links_labels() -> None:
    dataset = build_gru_event_type_dataset(
        _sample_index(),
        _sequence_features(),
        policy=SequenceDatasetPolicy(max_sequence_length=2),
    )

    assert dataset.sample_count == 4
    assert dataset.max_sequence_length == 2
    assert dataset.sample_ids.tolist() == ["s1", "s2", "s3", "s4"]
    assert dataset.labels.tolist() == [0, 1, 0, 1]
    assert dataset.sequence_lengths.tolist() == [1, 2, 2, 2]
    assert dataset.event_type_token_ids[0].tolist() == [
        dataset.vocabulary.token_to_id["view"],
        dataset.vocabulary.pad_id,
    ]
    assert dataset.event_type_token_ids[1].tolist() == [
        dataset.vocabulary.token_to_id["cart"],
        dataset.vocabulary.token_to_id["remove_from_cart"],
    ]
    assert dataset.event_type_token_ids[2].tolist() == [
        dataset.vocabulary.token_to_id["view"],
        dataset.vocabulary.unknown_id,
    ]


def test_gru_event_type_dataset_accepts_explicit_vocabulary() -> None:
    vocabulary = EventTypeVocabulary(
        token_to_id={
            PAD_TOKEN: 0,
            UNKNOWN_TOKEN: 1,
            "view": 2,
            "cart": 3,
            "purchase": 4,
        }
    )

    dataset = build_gru_event_type_dataset(
        _sample_index(),
        _sequence_features(),
        policy=SequenceDatasetPolicy(max_sequence_length=3),
        vocabulary=vocabulary,
    )

    assert dataset.event_type_token_ids[3].tolist() == [2, 3, 4]
    assert dataset.vocabulary is vocabulary


def test_load_gru_event_type_dataset_reads_feature_artifacts(tmp_path: Path) -> None:
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    _sample_index().to_csv(
        features_dir / "sample_index.csv",
        index=False,
        encoding="utf-8",
    )
    _sequence_features().to_parquet(
        features_dir / "sequence_feature_dataset.parquet",
        index=False,
    )

    dataset = load_gru_event_type_dataset(
        features_dir,
        policy=SequenceDatasetPolicy(max_sequence_length=2),
    )

    assert dataset.sample_ids.tolist() == ["s1", "s2", "s3", "s4"]
    assert dataset.event_type_token_ids.shape == (4, 2)


def test_gru_event_type_dataset_requires_matching_sample_ids() -> None:
    sequence_features = _sequence_features().replace({"sample_id": {"s4": "missing"}})

    try:
        build_gru_event_type_dataset(_sample_index(), sequence_features)
    except ValueError as error:
        assert "sample_id 집합이 일치하지 않는다" in str(error)
    else:
        raise AssertionError("sample_id 불일치가 ValueError를 발생시켜야 한다.")


def test_train_gru_classifier_logs_metrics_and_history(tmp_path: Path) -> None:
    sample_index = pd.DataFrame(
        [
            {"sample_id": "s1", "split": "train", "label": 0},
            {"sample_id": "s2", "split": "train", "label": 1},
            {"sample_id": "s3", "split": "train", "label": 0},
            {"sample_id": "s4", "split": "train", "label": 1},
            {"sample_id": "s5", "split": "validation", "label": 0},
            {"sample_id": "s6", "split": "validation", "label": 1},
            {"sample_id": "s7", "split": "test", "label": 0},
            {"sample_id": "s8", "split": "test", "label": 1},
        ]
    )
    sequence_features = pd.DataFrame(
        [
            {"sample_id": "s1", "event_type_sequence": "view"},
            {"sample_id": "s2", "event_type_sequence": "view cart"},
            {"sample_id": "s3", "event_type_sequence": "view view"},
            {"sample_id": "s4", "event_type_sequence": "view cart cart"},
            {"sample_id": "s5", "event_type_sequence": "view"},
            {"sample_id": "s6", "event_type_sequence": "view cart"},
            {"sample_id": "s7", "event_type_sequence": "view view"},
            {"sample_id": "s8", "event_type_sequence": "cart cart"},
        ]
    )
    dataset = build_gru_event_type_dataset(
        sample_index,
        sequence_features,
        policy=SequenceDatasetPolicy(max_sequence_length=3),
    )
    progress_messages: list[str] = []

    result = train_gru_classifier(
        dataset,
        policy=GruTrainingPolicy(
            max_sequence_length=3,
            embedding_dim=4,
            hidden_dim=4,
            batch_size=2,
            epochs=2,
            learning_rate=0.01,
            random_state=7,
            device="cpu",
        ),
        progress_callback=progress_messages.append,
    )

    assert result.metrics["model_name"].tolist() == ["gru", "gru", "gru"]
    assert result.metrics["split"].tolist() == ["train", "validation", "test"]
    assert result.metrics["status"].tolist() == ["evaluated", "evaluated", "evaluated"]
    assert result.model_status.loc[0, "status"] == "trained"
    assert result.training_history["epoch"].tolist() == [1, 2]
    assert "GRU epoch 1/2 학습 시작" in progress_messages
    assert "GRU 학습 결과 정리 완료" in progress_messages

    write_gru_artifacts(result, tmp_path)

    assert (tmp_path / "gru_model_metrics.csv").exists()
    assert (tmp_path / "gru_model_status.csv").exists()
    assert (tmp_path / "gru_training_history.csv").exists()
    assert (tmp_path / "gru_report.md").exists()
