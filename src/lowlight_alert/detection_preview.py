from pathlib import Path

import numpy as np

from lowlight_alert.config import CameraSettings, DetectionSettings, PreviewSettings
from lowlight_alert.detector import YuNetFaceDetector, draw_detections
from lowlight_alert.preview import run_preview


def run_detection_preview(
    camera_settings: CameraSettings,
    preview_settings: PreviewSettings,
    detection_settings: DetectionSettings,
    model_path: Path,
) -> None:
    detector = YuNetFaceDetector(model_path, detection_settings)

    def process_frame(frame: np.ndarray) -> tuple[str]:
        detections = detector.detect(frame)
        draw_detections(frame, detections)
        return (f"Faces: {len(detections)}",)

    run_preview(camera_settings, preview_settings, process_frame)
