from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from lowlight_alert.quality import FaceQuality


class TemplateStoreError(RuntimeError):
    """Raised when a biometric template store is invalid or cannot be updated."""


_SUBJECT_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


def validate_subject_id(subject_id: str) -> str:
    if not _SUBJECT_ID_PATTERN.fullmatch(subject_id):
        raise TemplateStoreError(
            "subject ID must be 1-64 ASCII letters, digits, hyphens or underscores"
        )
    return subject_id


@dataclass(frozen=True)
class SubjectTemplates:
    subject_id: str
    display_name: str
    features: np.ndarray
    records: tuple[dict[str, object], ...]
    created_at: str
    updated_at: str

    @property
    def template_count(self) -> int:
        return self.features.shape[0]


@dataclass(frozen=True)
class SubjectSummary:
    subject_id: str
    display_name: str
    template_count: int
    updated_at: str


class TemplateStore:
    SCHEMA_VERSION = 1

    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self, subject_id: str) -> SubjectTemplates | None:
        path = self._path(subject_id)
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as archive:
                features = np.asarray(archive["features"], dtype=np.float32)
                manifest = json.loads(str(archive["manifest"].item()))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise TemplateStoreError(f"cannot read template file {path}: {exc}") from exc

        return self._validate_loaded(subject_id, features, manifest, path)

    def add_template(
        self,
        subject_id: str,
        display_name: str | None,
        feature: np.ndarray,
        quality: FaceQuality,
        detection_score: float,
    ) -> SubjectTemplates:
        subject_id = validate_subject_id(subject_id)
        if not quality.passed:
            raise TemplateStoreError("quality-rejected feature cannot be stored")
        if not np.isfinite(detection_score) or not 0 <= detection_score <= 1:
            raise TemplateStoreError("detection score must be finite and between 0 and 1")
        normalized = self._normalize_feature(feature)
        existing = self.load(subject_id)
        now = datetime.now(UTC).isoformat()

        if existing is None:
            created_at = now
            name = self._display_name(display_name, subject_id)
            features = normalized.reshape(1, -1)
            records: list[dict[str, object]] = []
        else:
            if normalized.shape[0] != existing.features.shape[1]:
                raise TemplateStoreError(
                    f"feature dimension does not match existing templates for {subject_id}"
                )
            created_at = existing.created_at
            name = (
                self._display_name(display_name, subject_id)
                if display_name is not None
                else existing.display_name
            )
            features = np.vstack((existing.features, normalized))
            records = list(existing.records)

        records.append(
            {
                "template_id": uuid4().hex,
                "captured_at": now,
                "detection_score": float(detection_score),
                "quality": {
                    "face_size": quality.face_size,
                    "brightness": quality.brightness,
                    "sharpness": quality.sharpness,
                    "yaw_ratio": quality.yaw_ratio,
                    "nose_position": quality.nose_position,
                },
            }
        )
        manifest = {
            "schema_version": self.SCHEMA_VERSION,
            "subject_id": subject_id,
            "display_name": name,
            "created_at": created_at,
            "updated_at": now,
            "records": records,
        }
        self._write_atomic(self._path(subject_id), features, manifest)
        return SubjectTemplates(
            subject_id=subject_id,
            display_name=name,
            features=features,
            records=tuple(records),
            created_at=created_at,
            updated_at=now,
        )

    def list_subjects(self) -> list[SubjectSummary]:
        if not self.root.is_dir():
            return []
        summaries: list[SubjectSummary] = []
        for path in sorted(self.root.glob("*.npz")):
            loaded = self.load(path.stem)
            if loaded is None:
                continue
            summaries.append(
                SubjectSummary(
                    subject_id=loaded.subject_id,
                    display_name=loaded.display_name,
                    template_count=loaded.template_count,
                    updated_at=loaded.updated_at,
                )
            )
        return summaries

    def _path(self, subject_id: str) -> Path:
        return self.root / f"{validate_subject_id(subject_id)}.npz"

    @staticmethod
    def _display_name(display_name: str | None, subject_id: str) -> str:
        if display_name is None:
            return subject_id
        stripped = display_name.strip()
        if not stripped or len(stripped) > 100:
            raise TemplateStoreError("display name must contain 1-100 characters")
        return stripped

    @staticmethod
    def _normalize_feature(feature: np.ndarray) -> np.ndarray:
        vector = np.asarray(feature, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
            raise TemplateStoreError("feature must be a finite one-dimensional vector")
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise TemplateStoreError("feature cannot have zero length")
        return vector / norm

    def _validate_loaded(
        self,
        subject_id: str,
        features: np.ndarray,
        manifest: object,
        path: Path,
    ) -> SubjectTemplates:
        if not isinstance(manifest, dict):
            raise TemplateStoreError(f"invalid manifest in {path}")
        if manifest.get("schema_version") != self.SCHEMA_VERSION:
            raise TemplateStoreError(f"unsupported template schema in {path}")
        if manifest.get("subject_id") != subject_id:
            raise TemplateStoreError(f"subject ID mismatch in {path}")
        records = manifest.get("records")
        if features.ndim != 2 or features.shape[0] == 0:
            raise TemplateStoreError(f"invalid feature matrix in {path}")
        if not np.all(np.isfinite(features)):
            raise TemplateStoreError(f"non-finite feature found in {path}")
        if not isinstance(records, list) or len(records) != features.shape[0]:
            raise TemplateStoreError(f"template metadata count mismatch in {path}")
        try:
            return SubjectTemplates(
                subject_id=subject_id,
                display_name=str(manifest["display_name"]),
                features=features,
                records=tuple(records),
                created_at=str(manifest["created_at"]),
                updated_at=str(manifest["updated_at"]),
            )
        except KeyError as exc:
            raise TemplateStoreError(f"missing manifest field in {path}: {exc}") from exc

    def _write_atomic(self, path: Path, features: np.ndarray, manifest: dict[str, object]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{path.stem}.", suffix=".tmp", dir=self.root
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                np.savez_compressed(
                    output,
                    features=np.asarray(features, dtype=np.float32),
                    manifest=np.asarray(
                        json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))
                    ),
                )
            os.replace(temp_path, path)
        except OSError as exc:
            raise TemplateStoreError(f"cannot write template file {path}: {exc}") from exc
        finally:
            temp_path.unlink(missing_ok=True)
