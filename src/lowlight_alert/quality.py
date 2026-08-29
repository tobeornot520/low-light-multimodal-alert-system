from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import hypot

import cv2
import numpy as np

from lowlight_alert.config import QualitySettings
from lowlight_alert.detector import FaceDetection


class QualityEvaluationError(RuntimeError):
    """Raised when a frame cannot be evaluated."""


class QualityIssue(StrEnum):
    INVALID_CROP = "invalid_crop"
    FACE_TOO_SMALL = "face_too_small"
    TOO_DARK = "too_dark"
    TOO_BRIGHT = "too_bright"
    BLURRY = "blurry"
    EXTREME_POSE = "extreme_pose"


@dataclass(frozen=True)
class FaceQuality:
    issues: tuple[QualityIssue, ...]
    face_size: int
    brightness: float | None
    sharpness: float | None
    yaw_ratio: float | None
    nose_position: float | None

    @property
    def passed(self) -> bool:
        return not self.issues


class FaceQualityEvaluator:
    def __init__(self, settings: QualitySettings) -> None:
        self.settings = settings

    def evaluate(self, frame: np.ndarray, detection: FaceDetection) -> FaceQuality:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            raise QualityEvaluationError("quality evaluation expects a non-empty BGR image")

        x, y, width, height = detection.box
        face_size = max(0, min(width, height))
        issues: list[QualityIssue] = []
        if face_size < self.settings.min_face_size:
            issues.append(QualityIssue.FACE_TOO_SMALL)

        frame_height, frame_width = frame.shape[:2]
        left = max(0, x)
        top = max(0, y)
        right = min(frame_width, x + width)
        bottom = min(frame_height, y + height)

        brightness: float | None = None
        sharpness: float | None = None
        if right <= left or bottom <= top:
            issues.insert(0, QualityIssue.INVALID_CROP)
        else:
            crop = frame[top:bottom, left:right]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if brightness < self.settings.min_brightness:
                issues.append(QualityIssue.TOO_DARK)
            elif brightness > self.settings.max_brightness:
                issues.append(QualityIssue.TOO_BRIGHT)
            if sharpness < self.settings.min_sharpness:
                issues.append(QualityIssue.BLURRY)

        yaw_ratio, nose_position = self._pose_metrics(detection)
        nose_position_valid = (
            nose_position is not None
            and self.settings.min_nose_position <= nose_position <= self.settings.max_nose_position
        )
        if yaw_ratio is None or yaw_ratio > self.settings.max_yaw_ratio or not nose_position_valid:
            issues.append(QualityIssue.EXTREME_POSE)

        return FaceQuality(
            issues=tuple(issues),
            face_size=face_size,
            brightness=brightness,
            sharpness=sharpness,
            yaw_ratio=yaw_ratio,
            nose_position=nose_position,
        )

    @staticmethod
    def _pose_metrics(detection: FaceDetection) -> tuple[float | None, float | None]:
        right_eye, left_eye, nose, right_mouth, left_mouth = detection.landmarks
        eye_mid_x = (right_eye[0] + left_eye[0]) / 2
        eye_mid_y = (right_eye[1] + left_eye[1]) / 2
        mouth_mid_y = (right_mouth[1] + left_mouth[1]) / 2

        eye_distance = hypot(left_eye[0] - right_eye[0], left_eye[1] - right_eye[1])
        eye_to_mouth = mouth_mid_y - eye_mid_y
        if eye_distance < 1 or eye_to_mouth <= 1:
            return None, None

        yaw_ratio = abs(nose[0] - eye_mid_x) / eye_distance
        nose_position = (nose[1] - eye_mid_y) / eye_to_mouth
        return yaw_ratio, nose_position
