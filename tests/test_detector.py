from pathlib import Path

import numpy as np
import pytest

import lowlight_alert.detector as detector_module
from lowlight_alert.config import DetectionSettings
from lowlight_alert.detector import FaceDetectorError, YuNetFaceDetector, draw_detections


class FakeYuNet:
    def __init__(self, faces: np.ndarray | None) -> None:
        self.faces = faces
        self.input_sizes: list[tuple[int, int]] = []

    def setInputSize(self, size: tuple[int, int]) -> None:
        self.input_sizes.append(size)

    def detect(self, frame: np.ndarray):
        del frame
        return 1, self.faces


def model_file(tmp_path: Path) -> Path:
    path = tmp_path / "yunet.onnx"
    path.write_bytes(b"model")
    return path


def test_detector_parses_box_landmarks_and_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    faces = np.array(
        [
            [
                10.4,
                20.6,
                100.2,
                120.9,
                30.1,
                40.2,
                70.3,
                40.4,
                50.5,
                65.6,
                35.7,
                90.8,
                65.9,
                90.1,
                0.96,
            ]
        ],
        dtype=np.float32,
    )
    fake = FakeYuNet(faces)
    monkeypatch.setattr(detector_module, "_create_yunet", lambda *args: fake)
    detector = YuNetFaceDetector(model_file(tmp_path), DetectionSettings())

    detections = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    assert fake.input_sizes == [(640, 480)]
    assert len(detections) == 1
    assert detections[0].box == (10, 21, 100, 121)
    assert detections[0].landmarks[0] == (30, 40)
    assert detections[0].landmarks[-1] == (66, 90)
    assert detections[0].score == pytest.approx(0.96)


def test_detector_returns_empty_list_for_no_faces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeYuNet(None)
    monkeypatch.setattr(detector_module, "_create_yunet", lambda *args: fake)
    detector = YuNetFaceDetector(model_file(tmp_path), DetectionSettings())

    assert detector.detect(np.zeros((320, 320, 3), dtype=np.uint8)) == []
    assert fake.input_sizes == []


def test_detector_rejects_invalid_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detector_module, "_create_yunet", lambda *args: FakeYuNet(None))
    detector = YuNetFaceDetector(model_file(tmp_path), DetectionSettings())

    with pytest.raises(FaceDetectorError, match="non-empty BGR"):
        detector.detect(np.zeros((320, 320), dtype=np.uint8))


def test_detector_requires_model_file(tmp_path: Path) -> None:
    with pytest.raises(FaceDetectorError, match="does not exist"):
        YuNetFaceDetector(tmp_path / "missing.onnx", DetectionSettings())


def test_draw_detections_changes_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    faces = np.array(
        [[10, 20, 100, 120, 30, 40, 70, 40, 50, 65, 35, 90, 65, 90, 0.96]],
        dtype=np.float32,
    )
    monkeypatch.setattr(
        detector_module,
        "_create_yunet",
        lambda *args: FakeYuNet(faces),
    )
    detector = YuNetFaceDetector(model_file(tmp_path), DetectionSettings())
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    draw_detections(frame, detector.detect(frame))

    assert np.any(frame)
