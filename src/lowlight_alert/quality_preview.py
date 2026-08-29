from pathlib import Path

import numpy as np

from lowlight_alert.config import (
    CameraSettings,
    DetectionSettings,
    PreviewSettings,
    QualitySettings,
)
from lowlight_alert.detector import YuNetFaceDetector, draw_detection
from lowlight_alert.preview import run_preview
from lowlight_alert.quality import FaceQuality, FaceQualityEvaluator, QualityIssue

_ISSUE_LABELS = {
    QualityIssue.INVALID_CROP: "crop",
    QualityIssue.FACE_TOO_SMALL: "small",
    QualityIssue.TOO_DARK: "dark",
    QualityIssue.TOO_BRIGHT: "bright",
    QualityIssue.BLURRY: "blur",
    QualityIssue.EXTREME_POSE: "pose",
}


def _quality_label(quality: FaceQuality) -> str:
    if quality.passed:
        return "Quality OK"
    reasons = ",".join(_ISSUE_LABELS[issue] for issue in quality.issues)
    return f"Reject: {reasons}"


def run_quality_preview(
    camera_settings: CameraSettings,
    preview_settings: PreviewSettings,
    detection_settings: DetectionSettings,
    quality_settings: QualitySettings,
    model_path: Path,
) -> None:
    detector = YuNetFaceDetector(model_path, detection_settings)
    evaluator = FaceQualityEvaluator(quality_settings)

    def process_frame(frame: np.ndarray) -> tuple[str]:
        detections = detector.detect(frame)
        evaluations = [
            (detection, evaluator.evaluate(frame, detection)) for detection in detections
        ]
        passed = sum(quality.passed for _, quality in evaluations)
        for detection, quality in evaluations:
            color = (40, 220, 80) if quality.passed else (0, 180, 255)
            draw_detection(frame, detection, _quality_label(quality), color)
        return (f"Faces: {len(detections)} | Quality OK: {passed}",)

    run_preview(camera_settings, preview_settings, process_frame)
