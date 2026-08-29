from collections.abc import Callable

import cv2
import numpy as np
import pytest

import lowlight_alert.camera as camera_module
from lowlight_alert.camera import (
    CameraCapture,
    CameraOpenError,
    CameraReadError,
    probe_cameras,
)
from lowlight_alert.config import CameraSettings


class FakeCapture:
    def __init__(self, opened: bool = True, readable: bool = True) -> None:
        self.opened = opened
        self.readable = readable
        self.released = False
        self.properties: dict[int, float] = {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 25,
        }

    def isOpened(self) -> bool:
        return self.opened

    def set(self, property_id: int, value: float) -> bool:
        self.properties[property_id] = value
        return True

    def get(self, property_id: int) -> float:
        return self.properties.get(property_id, 0)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.readable:
            return False, None
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def capture_factory(captures: list[FakeCapture]) -> Callable[[int, int], FakeCapture]:
    def factory(index: int, backend: int) -> FakeCapture:
        del index, backend
        return captures.pop(0)

    return factory


def test_camera_context_configures_reads_and_releases(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeCapture()
    monkeypatch.setattr(camera_module.cv2, "VideoCapture", capture_factory([fake]))
    settings = CameraSettings(width=1280, height=720, fps=30)

    with CameraCapture(settings) as camera:
        assert camera.info.width == 1280
        assert camera.info.height == 720
        assert camera.info.fps == 30
        frame = camera.read()
        assert frame is not None
        assert frame.shape == (480, 640, 3)

    assert fake.released is True


def test_camera_open_failure_releases_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeCapture(opened=False)
    monkeypatch.setattr(camera_module.cv2, "VideoCapture", capture_factory([fake]))

    with pytest.raises(CameraOpenError, match="cannot open camera"):
        CameraCapture(CameraSettings()).open()

    assert fake.released is True


def test_camera_raises_after_repeated_read_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeCapture(readable=False)
    monkeypatch.setattr(camera_module.cv2, "VideoCapture", capture_factory([fake]))
    camera = CameraCapture(CameraSettings(read_failure_limit=2))
    camera.open()

    assert camera.read() is None
    with pytest.raises(CameraReadError, match="2 consecutive frames"):
        camera.read()
    camera.release()


def test_probe_returns_only_readable_cameras(monkeypatch: pytest.MonkeyPatch) -> None:
    captures = [
        FakeCapture(opened=False),
        FakeCapture(readable=False),
        FakeCapture(),
    ]
    all_captures = captures.copy()
    monkeypatch.setattr(camera_module.cv2, "VideoCapture", capture_factory(captures))

    results = probe_cameras(max_index=3, backend="auto")

    assert [camera.index for camera in results] == [2]
    assert all(capture.released for capture in all_captures)
