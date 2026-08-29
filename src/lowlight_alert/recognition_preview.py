from __future__ import annotations

from pathlib import Path

import numpy as np

from lowlight_alert.config import (
    CameraSettings,
    DetectionSettings,
    PreviewSettings,
    QualitySettings,
    RecognitionSettings,
)
from lowlight_alert.detector import YuNetFaceDetector, draw_detection
from lowlight_alert.preview import run_preview
from lowlight_alert.quality import FaceQuality, FaceQualityEvaluator, QualityIssue
from lowlight_alert.recognizer import MatchState, SFaceFeatureExtractor, TemplateMatcher
from lowlight_alert.template_store import TemplateStore

_ISSUE_LABELS = {
    QualityIssue.INVALID_CROP: "crop",
    QualityIssue.FACE_TOO_SMALL: "small",
    QualityIssue.TOO_DARK: "dark",
    QualityIssue.TOO_BRIGHT: "bright",
    QualityIssue.BLURRY: "blur",
    QualityIssue.EXTREME_POSE: "pose",
}


def _load_matcher(store: TemplateStore, settings: RecognitionSettings) -> TemplateMatcher:
    subjects = []
    for summary in store.list_subjects():
        loaded = store.load(summary.subject_id)
        if loaded is not None:
            subjects.append(loaded)
    return TemplateMatcher(
        subjects,
        settings.accept_threshold,
        settings.reject_threshold,
        settings.min_margin,
    )


def _quality_label(quality: FaceQuality) -> str:
    reasons = ",".join(_ISSUE_LABELS[issue] for issue in quality.issues)
    return f"Unusable: {reasons}"


def _identity_label(state: MatchState, display_name: str | None, score: float | None) -> str:
    score_label = "n/a" if score is None else f"{score:.2f}"
    if state is MatchState.REGISTERED:
        return f"{display_name or 'Registered'} {score_label}"
    if state is MatchState.UNKNOWN:
        return f"Unknown {score_label}"
    return f"Uncertain {score_label}"


def run_recognition_preview(
    camera_settings: CameraSettings,
    preview_settings: PreviewSettings,
    detection_settings: DetectionSettings,
    quality_settings: QualitySettings,
    recognition_settings: RecognitionSettings,
    templates_dir: Path,
    yunet_model_path: Path,
    sface_model_path: Path,
) -> None:
    detector = YuNetFaceDetector(yunet_model_path, detection_settings)
    evaluator = FaceQualityEvaluator(quality_settings)
    extractor = SFaceFeatureExtractor(sface_model_path)
    matcher = _load_matcher(TemplateStore(templates_dir), recognition_settings)

    def process_frame(frame: np.ndarray) -> tuple[str, ...]:
        detections = detector.detect(frame)
        counts = {state: 0 for state in MatchState}
        for detection in detections:
            quality = evaluator.evaluate(frame, detection)
            if not quality.passed:
                draw_detection(frame, detection, _quality_label(quality), (0, 180, 255))
                continue

            feature = extractor.extract(frame, detection)
            result = matcher.match(feature)
            counts[result.state] += 1
            color = {
                MatchState.REGISTERED: (40, 220, 80),
                MatchState.UNKNOWN: (40, 80, 235),
                MatchState.UNCERTAIN: (0, 210, 255),
            }[result.state]
            draw_detection(
                frame,
                detection,
                _identity_label(result.state, result.display_name, result.similarity),
                color,
            )

        return (
            f"Faces: {len(detections)} | Registered: {counts[MatchState.REGISTERED]} "
            f"| Unknown: {counts[MatchState.UNKNOWN]} "
            f"| Uncertain: {counts[MatchState.UNCERTAIN]} "
            f"| Templates: {matcher.subject_count}",
        )

    run_preview(camera_settings, preview_settings, process_frame)
