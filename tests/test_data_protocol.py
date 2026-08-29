from pathlib import Path

import pytest
import yaml

from lowlight_alert.data_protocol import ManifestError, load_manifest, validate_manifest


def _manifest(**overrides):
    value = {
        "schema_version": 1,
        "experiment": {
            "experiment_id": "exp-1",
            "run_id": "run-1",
            "dataset_split": "dev",
            "purpose": "threshold_development",
            "lighting_condition": "dim",
            "modalities": ["rgb", "nir"],
        },
        "source": {
            "path": "source.mp4",
            "synchronized_streams": [
                {"modality": "nir", "path": "nir.mp4", "clock_offset_ms": 0}
            ],
        },
        "camera": {},
        "scene": {},
        "annotations": {},
        "software": {},
        "retention": {},
        "participants": {
            "registered_subject_ids": ["person-a"],
            "unknown_subject_ids": ["unknown-a"],
            "enrollment_session_ids": ["enroll-1"],
            "test_session_ids": ["test-1"],
        },
    }
    value["experiment"].update(overrides)
    return value


def test_validate_manifest_returns_safe_summary() -> None:
    result = validate_manifest(_manifest())

    assert result.modalities == ("rgb", "nir")
    assert result.registered_subject_ids == ("person-a",)
    assert result.path == Path("<memory>")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"dataset_split": "train"}, "dataset_split"),
        ({"lighting_condition": "sunny"}, "lighting_condition"),
        ({"modalities": ["lidar"]}, "unsupported"),
    ],
)
def test_validate_manifest_rejects_invalid_protocol(change, message: str) -> None:
    with pytest.raises(ManifestError, match=message):
        validate_manifest(_manifest(**change))


def test_validate_manifest_rejects_session_overlap() -> None:
    value = _manifest()
    value["participants"]["test_session_ids"] = ["enroll-1"]

    with pytest.raises(ManifestError, match="overlap"):
        validate_manifest(value)


def test_validate_manifest_requires_a_stream_for_each_non_rgb_modality() -> None:
    value = _manifest()
    value["source"]["synchronized_streams"] = []

    with pytest.raises(ManifestError, match="missing synchronized streams"):
        validate_manifest(value)


def test_frozen_test_requires_test_split() -> None:
    with pytest.raises(ManifestError, match="requires dataset_split test"):
        validate_manifest(_manifest(purpose="frozen_test"))


def test_load_manifest_reads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(_manifest()), encoding="utf-8")
    assert load_manifest(path).run_id == "run-1"
