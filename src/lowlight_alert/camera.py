from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from lowlight_alert.config import CameraSettings


class CameraError(RuntimeError):
    """Base error for camera operations."""


class CameraOpenError(CameraError):
    """Raised when a camera cannot be opened."""


class CameraReadError(CameraError):
    """Raised after repeated frame read failures."""


@dataclass(frozen=True)
class CameraInfo:
    index: int
    backend: str
    width: int
    height: int
    fps: float


def _backend_id(name: str) -> int:
    return {
        "auto": cv2.CAP_ANY,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
    }[name]


def _create_capture(index: int, backend: str) -> cv2.VideoCapture:
    return cv2.VideoCapture(index, _backend_id(backend))


def _capture_info(capture: cv2.VideoCapture, index: int, backend: str) -> CameraInfo:
    return CameraInfo(
        index=index,
        backend=backend,
        width=round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(capture.get(cv2.CAP_PROP_FPS)),
    )


class CameraCapture:
    """Own an OpenCV camera and release it reliably."""

    def __init__(self, settings: CameraSettings) -> None:
        self.settings = settings
        self._capture: cv2.VideoCapture | None = None
        self._info: CameraInfo | None = None
        self._consecutive_failures = 0

    def open(self) -> CameraInfo:
        if self._capture is not None:
            raise CameraOpenError("camera is already open")

        capture = _create_capture(self.settings.index, self.settings.backend)
        if not capture.isOpened():
            capture.release()
            raise CameraOpenError(
                f"cannot open camera {self.settings.index} with backend {self.settings.backend}"
            )

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.height)
        capture.set(cv2.CAP_PROP_FPS, self.settings.fps)
        self._capture = capture
        self._info = _capture_info(capture, self.settings.index, self.settings.backend)
        return self._info

    @property
    def info(self) -> CameraInfo:
        if self._info is None:
            raise CameraError("camera is not open")
        return self._info

    def read(self) -> np.ndarray | None:
        if self._capture is None:
            raise CameraReadError("camera is not open")

        ok, frame = self._capture.read()
        if ok and frame is not None:
            self._consecutive_failures = 0
            return frame

        self._consecutive_failures += 1
        if self._consecutive_failures >= self.settings.read_failure_limit:
            raise CameraReadError(
                f"camera {self.settings.index} failed to provide "
                f"{self._consecutive_failures} consecutive frames"
            )
        return None

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._info = None
        self._consecutive_failures = 0

    def __enter__(self) -> CameraCapture:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


def probe_cameras(max_index: int, backend: str) -> list[CameraInfo]:
    """Return cameras that can open and provide at least one frame."""
    cameras: list[CameraInfo] = []
    for index in range(max_index):
        capture = _create_capture(index, backend)
        try:
            if not capture.isOpened():
                continue
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            cameras.append(_capture_info(capture, index, backend))
        finally:
            capture.release()
    return cameras
