"""Validation helpers for experiment manifests.

The manifest is deliberately small and human-readable.  It describes a run,
its sensor modalities, and the session split without containing names or raw
biometric data.  Media files may be collected later, so validation checks the
protocol rather than requiring every referenced path to exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ManifestError(ValueError):
    """Raised when an experiment manifest violates the data protocol."""


_CONDITIONS = {"normal", "dim", "backlight", "near_black"}
_SPLITS = {"dev", "test"}
_PURPOSES = {"pipeline_smoke", "threshold_development", "frozen_test"}
_MODALITIES = {"rgb", "nir", "thermal", "tof", "audio"}


@dataclass(frozen=True, slots=True)
class DataManifest:
    """Validated high-level fields used by collection and evaluation tools."""

    path: Path
    experiment_id: str
    run_id: str
    dataset_split: str
    purpose: str
    lighting_condition: str
    modalities: tuple[str, ...]
    registered_subject_ids: tuple[str, ...]
    unknown_subject_ids: tuple[str, ...]
    enrollment_session_ids: tuple[str, ...]
    test_session_ids: tuple[str, ...]
    raw: dict[str, Any]


def _required_mapping(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ManifestError(f"manifest.{name} must be a mapping")
    return value


def _text(mapping: dict[str, Any], key: str, section: str, *, required: bool = True) -> str:
    value = mapping.get(key)
    if value is None and not required:
        return ""
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"manifest.{section}.{key} must be a non-empty string")
    return value.strip()


def _ids(mapping: dict[str, Any], key: str, section: str) -> tuple[str, ...]:
    value = mapping.get(key, [])
    if not isinstance(value, (list, tuple)):
        raise ManifestError(f"manifest.{section}.{key} must be a list")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ManifestError(f"manifest.{section}.{key}[{index}] must be a non-empty string")
        result.append(item.strip())
    if len(result) != len(set(result)):
        raise ManifestError(f"manifest.{section}.{key} must not contain duplicates")
    return tuple(result)


def _modalities(experiment: dict[str, Any], source: dict[str, Any]) -> tuple[str, ...]:
    value = experiment.get("modalities", source.get("modalities"))
    if value is None:
        variant = str(experiment.get("input_variant", "raw_rgb")).lower()
        value = ["rgb"] if "rgb" in variant else []
    if not isinstance(value, (list, tuple)) or not value:
        raise ManifestError("manifest.experiment.modalities must be a non-empty list")
    result = tuple(str(item).strip().lower() for item in value)
    invalid = sorted(set(result) - _MODALITIES)
    if invalid:
        raise ManifestError(f"unsupported manifest modalities: {', '.join(invalid)}")
    if len(result) != len(set(result)):
        raise ManifestError("manifest.experiment.modalities must not contain duplicates")
    return result


def _validate_synchronized_streams(source: dict[str, Any], modalities: tuple[str, ...]) -> None:
    expected = set(modalities) - {"rgb"}
    streams = source.get("synchronized_streams", [])
    if not isinstance(streams, (list, tuple)):
        raise ManifestError("manifest.source.synchronized_streams must be a list")
    recorded: set[str] = set()
    for index, stream in enumerate(streams):
        if not isinstance(stream, dict):
            raise ManifestError(f"manifest.source.synchronized_streams[{index}] must be a mapping")
        modality = stream.get("modality")
        if not isinstance(modality, str) or modality.strip().lower() not in _MODALITIES - {"rgb"}:
            raise ManifestError(
                f"manifest.source.synchronized_streams[{index}].modality is invalid"
            )
        normalized = modality.strip().lower()
        if normalized in recorded:
            raise ManifestError("manifest.source.synchronized_streams must not repeat a modality")
        _text(stream, "path", f"source.synchronized_streams[{index}]")
        try:
            float(stream.get("clock_offset_ms"))
        except (TypeError, ValueError) as exc:
            raise ManifestError(
                f"manifest.source.synchronized_streams[{index}].clock_offset_ms must be numeric"
            ) from exc
        recorded.add(normalized)
    missing = sorted(expected - recorded)
    unexpected = sorted(recorded - expected)
    if missing:
        raise ManifestError("missing synchronized streams for: " + ", ".join(missing))
    if unexpected:
        raise ManifestError(
            "synchronized streams not declared as modalities: " + ", ".join(unexpected)
        )


def validate_manifest(raw: Any, *, path: Path | None = None) -> DataManifest:
    """Validate a loaded YAML mapping and return its safe summary."""
    if not isinstance(raw, dict):
        raise ManifestError("manifest root must be a mapping")
    if raw.get("schema_version") != 1:
        raise ManifestError("manifest.schema_version must be 1")
    experiment = _required_mapping(raw, "experiment")
    source = _required_mapping(raw, "source")
    participants = _required_mapping(raw, "participants")
    _required_mapping(raw, "camera")
    _required_mapping(raw, "scene")
    _required_mapping(raw, "annotations")
    _required_mapping(raw, "software")
    _required_mapping(raw, "retention")

    experiment_id = _text(experiment, "experiment_id", "experiment")
    run_id = _text(experiment, "run_id", "experiment")
    dataset_split = _text(experiment, "dataset_split", "experiment")
    if dataset_split not in _SPLITS:
        raise ManifestError("manifest.experiment.dataset_split must be dev or test")
    purpose = _text(experiment, "purpose", "experiment")
    if purpose not in _PURPOSES:
        raise ManifestError("manifest.experiment.purpose is not supported")
    lighting_condition = _text(experiment, "lighting_condition", "experiment")
    if lighting_condition not in _CONDITIONS:
        raise ManifestError("manifest.experiment.lighting_condition is not supported")
    _text(source, "path", "source")

    registered = _ids(participants, "registered_subject_ids", "participants")
    unknown = _ids(participants, "unknown_subject_ids", "participants")
    overlap = sorted(set(registered) & set(unknown))
    if overlap:
        raise ManifestError(
            "subject IDs cannot be both registered and unknown: " + ", ".join(overlap)
        )
    enrollment = _ids(participants, "enrollment_session_ids", "participants")
    testing = _ids(participants, "test_session_ids", "participants")
    shared_sessions = sorted(set(enrollment) & set(testing))
    if shared_sessions:
        raise ManifestError(
            "enrollment_session_ids and test_session_ids overlap: " + ", ".join(shared_sessions)
        )
    modalities = _modalities(experiment, source)
    _validate_synchronized_streams(source, modalities)
    if purpose == "frozen_test" and dataset_split != "test":
        raise ManifestError("manifest.experiment.frozen_test requires dataset_split test")
    return DataManifest(
        path=path or Path("<memory>"),
        experiment_id=experiment_id,
        run_id=run_id,
        dataset_split=dataset_split,
        purpose=purpose,
        lighting_condition=lighting_condition,
        modalities=modalities,
        registered_subject_ids=registered,
        unknown_subject_ids=unknown,
        enrollment_session_ids=enrollment,
        test_session_ids=testing,
        raw=raw,
    )


def load_manifest(path: Path | str) -> DataManifest:
    """Load and validate a YAML manifest from disk."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ManifestError(f"manifest does not exist: {manifest_path}")
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ManifestError(f"cannot read manifest {manifest_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ManifestError(f"cannot parse manifest {manifest_path}: {exc}") from exc
    return validate_manifest(raw, path=manifest_path)
