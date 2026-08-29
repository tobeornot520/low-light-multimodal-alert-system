from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import monotonic

import numpy as np

from lowlight_alert.config import (
    CameraSettings,
    DetectionSettings,
    EnrollmentSettings,
    PreviewSettings,
    QualitySettings,
)
from lowlight_alert.detector import (
    YuNetFaceDetector,
    draw_detection,
    draw_detections,
)
from lowlight_alert.preview import PreviewFrameResult, run_preview
from lowlight_alert.quality import FaceQualityEvaluator, QualityIssue
from lowlight_alert.recognizer import SFaceFeatureExtractor
from lowlight_alert.template_store import TemplateStore

_ISSUE_LABELS = {
    QualityIssue.INVALID_CROP: "crop",
    QualityIssue.FACE_TOO_SMALL: "small",
    QualityIssue.TOO_DARK: "dark",
    QualityIssue.TOO_BRIGHT: "bright",
    QualityIssue.BLURRY: "blur",
    QualityIssue.EXTREME_POSE: "pose",
}


class EnrollmentSession:
    def __init__(
        self,
        detector: YuNetFaceDetector,
        evaluator: FaceQualityEvaluator,
        extractor: SFaceFeatureExtractor,
        store: TemplateStore,
        settings: EnrollmentSettings,
        subject_id: str,
        display_name: str | None,
        target_count: int,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if target_count <= 0:
            raise ValueError("target template count must be positive")
        self.detector = detector
        self.evaluator = evaluator
        self.extractor = extractor
        self.store = store
        self.settings = settings
        self.subject_id = subject_id
        self.display_name = display_name
        self.target_count = target_count
        self.clock = clock
        self.accepted_count = 0
        self._last_attempt_at = float("-inf")
        existing = store.load(subject_id)
        self._known_features = (
            [] if existing is None else [feature.copy() for feature in existing.features]
        )

    def process_frame(self, frame: np.ndarray) -> PreviewFrameResult:
        detections = self.detector.detect(frame)
        progress = f"Enrollment: {self.accepted_count}/{self.target_count}"
        if len(detections) != 1:
            draw_detections(frame, detections)
            return PreviewFrameResult((progress, "Require exactly one face"))

        detection = detections[0]
        quality = self.evaluator.evaluate(frame, detection)
        if not quality.passed:
            reasons = ",".join(_ISSUE_LABELS[issue] for issue in quality.issues)
            draw_detection(frame, detection, f"Reject: {reasons}", (0, 180, 255))
            return PreviewFrameResult((progress, "Quality rejected"))

        now = self.clock()
        if now - self._last_attempt_at < self.settings.capture_interval_seconds:
            draw_detection(frame, detection, "Hold", (0, 210, 255))
            return PreviewFrameResult((progress, "Waiting for capture interval"))

        self._last_attempt_at = now
        feature = self.extractor.extract(frame, detection)
        if self._is_duplicate(feature):
            draw_detection(frame, detection, "Change pose", (0, 210, 255))
            return PreviewFrameResult((progress, "Near-duplicate template"))

        self.store.add_template(
            self.subject_id,
            self.display_name,
            feature,
            quality,
            detection.score,
        )
        self._known_features.append(feature.copy())
        self.accepted_count += 1
        complete = self.accepted_count >= self.target_count
        draw_detection(frame, detection, "Saved", (40, 220, 80))
        return PreviewFrameResult(
            (f"Enrollment: {self.accepted_count}/{self.target_count}",),
            stop=complete,
        )

    def _is_duplicate(self, feature: np.ndarray) -> bool:
        threshold = self.settings.duplicate_similarity_threshold
        return any(
            self.extractor.similarity(feature, known) >= threshold for known in self._known_features
        )


def run_enrollment_preview(
    camera_settings: CameraSettings,
    preview_settings: PreviewSettings,
    detection_settings: DetectionSettings,
    quality_settings: QualitySettings,
    enrollment_settings: EnrollmentSettings,
    yunet_model_path: Path,
    sface_model_path: Path,
    subject_id: str,
    display_name: str | None,
    target_count: int,
) -> None:
    session = EnrollmentSession(
        detector=YuNetFaceDetector(yunet_model_path, detection_settings),
        evaluator=FaceQualityEvaluator(quality_settings),
        extractor=SFaceFeatureExtractor(sface_model_path),
        store=TemplateStore(enrollment_settings.templates_dir),
        settings=enrollment_settings,
        subject_id=subject_id,
        display_name=display_name,
        target_count=target_count,
    )
    run_preview(camera_settings, preview_settings, session.process_frame)
    print(
        f"Enrollment ended for {subject_id}: "
        f"{session.accepted_count}/{target_count} new templates saved."
    )
