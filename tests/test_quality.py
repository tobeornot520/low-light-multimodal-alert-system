import numpy as np
import pytest

from lowlight_alert.config import QualitySettings
from lowlight_alert.detector import FaceDetection
from lowlight_alert.quality import (
    FaceQualityEvaluator,
    QualityEvaluationError,
    QualityIssue,
)


def frontal_detection(box: tuple[int, int, int, int] = (20, 20, 120, 120)) -> FaceDetection:
    return FaceDetection(
        box=box,
        landmarks=((50, 60), (110, 60), (80, 90), (60, 120), (100, 120)),
        score=0.98,
    )


def checkerboard(size: int = 200) -> np.ndarray:
    rows, columns = np.indices((size, size))
    gray = ((rows + columns) % 2 * 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def test_sharp_well_lit_frontal_face_passes() -> None:
    result = FaceQualityEvaluator(QualitySettings()).evaluate(checkerboard(), frontal_detection())

    assert result.passed is True
    assert result.issues == ()
    assert result.brightness == pytest.approx(127.5)
    assert result.sharpness is not None
    assert result.sharpness > 60
    assert result.yaw_ratio == pytest.approx(0)
    assert result.nose_position == pytest.approx(0.5)


def test_small_face_is_rejected() -> None:
    result = FaceQualityEvaluator(QualitySettings()).evaluate(
        checkerboard(), frontal_detection((20, 20, 50, 60))
    )

    assert QualityIssue.FACE_TOO_SMALL in result.issues


@pytest.mark.parametrize(
    ("value", "issue"),
    [
        (0, QualityIssue.TOO_DARK),
        (255, QualityIssue.TOO_BRIGHT),
        (127, QualityIssue.BLURRY),
    ],
)
def test_exposure_and_blur_failures(value: int, issue: QualityIssue) -> None:
    frame = np.full((200, 200, 3), value, dtype=np.uint8)

    result = FaceQualityEvaluator(QualitySettings()).evaluate(frame, frontal_detection())

    assert issue in result.issues
    assert result.passed is False


def test_extreme_pose_is_rejected() -> None:
    detection = FaceDetection(
        box=(20, 20, 120, 120),
        landmarks=((50, 60), (110, 60), (120, 90), (60, 120), (100, 120)),
        score=0.98,
    )

    result = FaceQualityEvaluator(QualitySettings()).evaluate(checkerboard(), detection)

    assert QualityIssue.EXTREME_POSE in result.issues
    assert result.yaw_ratio is not None
    assert result.yaw_ratio > 0.35


def test_invalid_crop_is_reported() -> None:
    result = FaceQualityEvaluator(QualitySettings()).evaluate(
        checkerboard(), frontal_detection((250, 250, 120, 120))
    )

    assert QualityIssue.INVALID_CROP in result.issues
    assert result.brightness is None
    assert result.sharpness is None


def test_invalid_frame_raises_clear_error() -> None:
    with pytest.raises(QualityEvaluationError, match="non-empty BGR"):
        FaceQualityEvaluator(QualitySettings()).evaluate(
            np.zeros((200, 200), dtype=np.uint8), frontal_detection()
        )
