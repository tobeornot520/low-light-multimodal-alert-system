from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from lowlight_alert.detector import FaceDetection

if TYPE_CHECKING:
    from lowlight_alert.template_store import SubjectTemplates, TemplateStore


class FaceRecognitionError(RuntimeError):
    """Raised when SFace cannot load or extract a valid feature."""


class MatchState(StrEnum):
    REGISTERED = "registered"
    UNKNOWN = "unknown"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class IdentityMatch:
    state: MatchState
    subject_id: str | None = None
    display_name: str | None = None
    similarity: float | None = None
    runner_up_similarity: float | None = None
    reason: str | None = None


class TemplateMatcher:
    """Match a normalized feature against an open set of enrolled subjects."""

    def __init__(
        self,
        subjects: Iterable[SubjectTemplates],
        accept_threshold: float,
        reject_threshold: float,
        min_margin: float,
    ) -> None:
        if not 0.0 <= reject_threshold < accept_threshold <= 1.0:
            raise FaceRecognitionError(
                "recognition thresholds must satisfy 0 <= reject < accept <= 1"
            )
        if not 0.0 <= min_margin <= 1.0:
            raise FaceRecognitionError("recognition min_margin must be between 0 and 1")
        self._subjects = tuple(subjects)
        self._accept_threshold = accept_threshold
        self._reject_threshold = reject_threshold
        self._min_margin = min_margin

    @classmethod
    def from_store(
        cls,
        store: TemplateStore,
        accept_threshold: float,
        reject_threshold: float,
        min_margin: float,
    ) -> TemplateMatcher:
        subjects = []
        for summary in store.list_subjects():
            loaded = store.load(summary.subject_id)
            if loaded is not None:
                subjects.append(loaded)
        return cls(subjects, accept_threshold, reject_threshold, min_margin)

    @property
    def subject_count(self) -> int:
        return len(self._subjects)

    def match(self, feature: np.ndarray) -> IdentityMatch:
        query = self._normalize(feature)
        if not self._subjects:
            return IdentityMatch(MatchState.UNKNOWN, reason="no_enrolled_subjects")

        candidates: list[tuple[float, SubjectTemplates]] = []
        for subject in self._subjects:
            if subject.features.ndim != 2 or subject.features.shape[0] == 0:
                raise FaceRecognitionError(f"invalid templates for {subject.subject_id}")
            if subject.features.shape[1] != query.shape[0]:
                raise FaceRecognitionError(
                    f"feature dimension does not match templates for {subject.subject_id}"
                )
            template_norms = np.linalg.norm(subject.features, axis=1)
            if not np.all(np.isfinite(template_norms)) or np.any(template_norms <= 0):
                raise FaceRecognitionError(f"invalid template feature for {subject.subject_id}")
            normalized_templates = subject.features / template_norms[:, np.newaxis]
            scores = normalized_templates @ query
            if not np.all(np.isfinite(scores)):
                raise FaceRecognitionError(f"non-finite template score for {subject.subject_id}")
            candidates.append((float(np.clip(np.max(scores), -1.0, 1.0)), subject))

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_subject = candidates[0]
        runner_up_score = candidates[1][0] if len(candidates) > 1 else None
        margin_ok = runner_up_score is None or best_score - runner_up_score >= self._min_margin
        if best_score >= self._accept_threshold and margin_ok:
            state = MatchState.REGISTERED
            reason = None
        elif best_score < self._reject_threshold:
            state = MatchState.UNKNOWN
            reason = "below_reject_threshold"
        else:
            state = MatchState.UNCERTAIN
            reason = "score_or_margin_in_gray_zone"

        return IdentityMatch(
            state=state,
            subject_id=best_subject.subject_id,
            display_name=best_subject.display_name,
            similarity=best_score,
            runner_up_similarity=runner_up_score,
            reason=reason,
        )

    @staticmethod
    def _normalize(feature: np.ndarray) -> np.ndarray:
        vector = np.asarray(feature, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
            raise FaceRecognitionError("query feature must be a finite one-dimensional vector")
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise FaceRecognitionError("query feature cannot have zero length")
        return vector / norm


def _create_sface(model_path: Path):
    return cv2.FaceRecognizerSF.create(str(model_path), "")


def _detection_row(detection: FaceDetection) -> np.ndarray:
    values: list[float] = [float(value) for value in detection.box]
    for x, y in detection.landmarks:
        values.extend((float(x), float(y)))
    values.append(float(detection.score))
    return np.asarray(values, dtype=np.float32)


class SFaceFeatureExtractor:
    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise FaceRecognitionError(f"SFace model does not exist: {model_path}")
        if model_path.stat().st_size == 0:
            raise FaceRecognitionError(f"SFace model is empty: {model_path}")
        try:
            self._recognizer = _create_sface(model_path)
        except cv2.error as exc:
            raise FaceRecognitionError(f"cannot load SFace model {model_path}: {exc}") from exc

    def extract(self, frame: np.ndarray, detection: FaceDetection) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            raise FaceRecognitionError("SFace expects a non-empty BGR image")
        try:
            aligned = self._recognizer.alignCrop(frame, _detection_row(detection))
            feature = self._recognizer.feature(aligned)
        except cv2.error as exc:
            raise FaceRecognitionError(f"SFace feature extraction failed: {exc}") from exc

        vector = np.asarray(feature, dtype=np.float32).reshape(-1)
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            raise FaceRecognitionError("SFace returned an invalid feature")
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise FaceRecognitionError("SFace returned a zero-length feature")
        return vector / norm

    @staticmethod
    def similarity(first: np.ndarray, second: np.ndarray) -> float:
        if first.shape != second.shape or first.ndim != 1:
            raise FaceRecognitionError("features must be one-dimensional with equal shape")
        if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
            raise FaceRecognitionError("features contain non-finite values")
        return float(np.clip(np.dot(first, second), -1.0, 1.0))
