from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np

from lowlight_alert.config import DetectionSettings

Point: TypeAlias = tuple[int, int]
BoundingBox: TypeAlias = tuple[int, int, int, int]


class FaceDetectorError(RuntimeError):
    """Raised when YuNet cannot load or process a frame."""


@dataclass(frozen=True)
class FaceDetection:
    box: BoundingBox
    landmarks: tuple[Point, Point, Point, Point, Point]
    score: float


def _create_yunet(model_path: Path, settings: DetectionSettings, input_size: tuple[int, int]):
    return cv2.FaceDetectorYN.create(
        str(model_path),
        "",
        input_size,
        settings.score_threshold,
        settings.nms_threshold,
        settings.top_k,
    )


class YuNetFaceDetector:
    """Convert YuNet's raw output into stable project-level detections."""

    def __init__(self, model_path: Path, settings: DetectionSettings) -> None:
        if not model_path.is_file():
            raise FaceDetectorError(f"YuNet model does not exist: {model_path}")
        if model_path.stat().st_size == 0:
            raise FaceDetectorError(f"YuNet model is empty: {model_path}")

        self._input_size = (320, 320)
        try:
            self._detector = _create_yunet(model_path, settings, self._input_size)
        except cv2.error as exc:
            raise FaceDetectorError(f"cannot load YuNet model {model_path}: {exc}") from exc

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            raise FaceDetectorError("YuNet expects a non-empty BGR image")

        height, width = frame.shape[:2]
        input_size = (width, height)
        try:
            if input_size != self._input_size:
                self._detector.setInputSize(input_size)
                self._input_size = input_size
            _, faces = self._detector.detect(frame)
        except cv2.error as exc:
            raise FaceDetectorError(f"YuNet detection failed: {exc}") from exc

        if faces is None:
            return []
        rows = np.asarray(faces)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.ndim != 2 or rows.shape[1] < 15:
            raise FaceDetectorError(f"unexpected YuNet output shape: {rows.shape}")
        return [self._parse_row(row) for row in rows]

    @staticmethod
    def _parse_row(row: np.ndarray) -> FaceDetection:
        values = [float(value) for value in row[:15]]
        box = tuple(round(value) for value in values[:4])
        landmarks = tuple(
            (round(values[index]), round(values[index + 1])) for index in range(4, 14, 2)
        )
        return FaceDetection(
            box=box,
            landmarks=landmarks,
            score=values[14],
        )


_LANDMARK_COLORS = (
    (255, 0, 0),
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
)


def draw_detection(
    frame: np.ndarray,
    detection: FaceDetection,
    label: str,
    color: tuple[int, int, int] = (40, 220, 80),
) -> None:
    x, y, width, height = detection.box
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
    for point, landmark_color in zip(detection.landmarks, _LANDMARK_COLORS, strict=True):
        cv2.circle(frame, point, 2, landmark_color, 2)

    text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
    label_x = min(max(0, x), max(0, frame.shape[1] - text_width - 4))
    label_y = y - 8 if y >= 24 else max(18, y + 22)
    position = (label_x, label_y)
    cv2.putText(frame, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
    cv2.putText(frame, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)


def draw_detections(frame: np.ndarray, detections: list[FaceDetection]) -> None:
    for detection in detections:
        draw_detection(frame, detection, f"Face {detection.score:.2f}")
