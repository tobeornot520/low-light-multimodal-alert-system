from pathlib import Path

import numpy as np
import pytest

from lowlight_alert.quality import FaceQuality, QualityIssue
from lowlight_alert.template_store import TemplateStore, TemplateStoreError


def accepted_quality() -> FaceQuality:
    return FaceQuality(
        issues=(),
        face_size=120,
        brightness=128.0,
        sharpness=100.0,
        yaw_ratio=0.1,
        nose_position=0.5,
    )


def test_store_adds_and_loads_multiple_templates(tmp_path: Path) -> None:
    store = TemplateStore(tmp_path / "templates")

    first = store.add_template(
        "person-a",
        "授权对象A",
        np.array([3.0, 4.0], dtype=np.float32),
        accepted_quality(),
        0.96,
    )
    second = store.add_template(
        "person-a",
        None,
        np.array([4.0, 3.0], dtype=np.float32),
        accepted_quality(),
        0.97,
    )
    loaded = store.load("person-a")

    assert first.template_count == 1
    assert second.template_count == 2
    assert loaded is not None
    assert loaded.display_name == "授权对象A"
    assert loaded.features.shape == (2, 2)
    assert np.linalg.norm(loaded.features[0]) == pytest.approx(1.0)
    assert len(loaded.records) == 2
    assert list((tmp_path / "templates").glob("*.tmp")) == []


def test_list_subjects_returns_counts(tmp_path: Path) -> None:
    store = TemplateStore(tmp_path / "templates")
    store.add_template(
        "person-b",
        None,
        np.array([1.0, 0.0], dtype=np.float32),
        accepted_quality(),
        0.95,
    )

    summaries = store.list_subjects()

    assert [(item.subject_id, item.template_count) for item in summaries] == [("person-b", 1)]


@pytest.mark.parametrize("subject_id", ["../escape", "has space", "", "a" * 65])
def test_store_rejects_unsafe_subject_ids(tmp_path: Path, subject_id: str) -> None:
    store = TemplateStore(tmp_path / "templates")

    with pytest.raises(TemplateStoreError, match="subject ID"):
        store.add_template(
            subject_id,
            None,
            np.array([1.0, 0.0], dtype=np.float32),
            accepted_quality(),
            0.95,
        )


def test_store_rejects_feature_dimension_change(tmp_path: Path) -> None:
    store = TemplateStore(tmp_path / "templates")
    store.add_template(
        "person-a",
        None,
        np.array([1.0, 0.0], dtype=np.float32),
        accepted_quality(),
        0.95,
    )

    with pytest.raises(TemplateStoreError, match="feature dimension"):
        store.add_template(
            "person-a",
            None,
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            accepted_quality(),
            0.95,
        )


def test_store_refuses_quality_rejected_feature(tmp_path: Path) -> None:
    rejected = FaceQuality(
        issues=(QualityIssue.BLURRY,),
        face_size=120,
        brightness=128.0,
        sharpness=10.0,
        yaw_ratio=0.1,
        nose_position=0.5,
    )

    with pytest.raises(TemplateStoreError, match="quality-rejected"):
        TemplateStore(tmp_path / "templates").add_template(
            "person-a",
            None,
            np.array([1.0, 0.0], dtype=np.float32),
            rejected,
            0.95,
        )

    assert not (tmp_path / "templates").exists()
