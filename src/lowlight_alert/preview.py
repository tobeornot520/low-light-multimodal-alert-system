from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from time import monotonic

import cv2
import numpy as np

from lowlight_alert.camera import CameraCapture, CameraError, CameraInfo
from lowlight_alert.config import CameraSettings, PreviewSettings


@dataclass(frozen=True)
class PreviewFrameResult:
    lines: tuple[str, ...] = ()
    stop: bool = False


FrameProcessor = Callable[[np.ndarray], Sequence[str] | PreviewFrameResult]


def _average_fps(timestamps: deque[float]) -> float:
    if len(timestamps) < 2:
        return 0.0
    elapsed = timestamps[-1] - timestamps[0]
    return (len(timestamps) - 1) / elapsed if elapsed > 0 else 0.0


def _draw_status(
    frame: np.ndarray,
    info: CameraInfo,
    fps: float,
    show_fps: bool,
    extra_lines: Sequence[str],
) -> None:
    lines = [f"Camera {info.index} | {frame.shape[1]}x{frame.shape[0]}"]
    if show_fps:
        lines[0] += f" | {fps:.1f} FPS"
    lines.extend(extra_lines)
    lines.append("Press Q or Esc to exit")

    for row, message in enumerate(lines):
        position = (16, 30 + row * 28)
        cv2.putText(frame, message, position, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(
            frame,
            message,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
        )


def run_preview(
    camera_settings: CameraSettings,
    preview_settings: PreviewSettings,
    frame_processor: FrameProcessor | None = None,
) -> None:
    """Open a camera preview until the user presses Q or Escape."""
    timestamps: deque[float] = deque(maxlen=30)
    window_created = False

    try:
        with CameraCapture(camera_settings) as camera:
            info = camera.info
            print(
                f"Opened camera {info.index}: {info.width}x{info.height}, "
                f"{info.fps:.1f} FPS, backend={info.backend}"
            )

            cv2.namedWindow(preview_settings.window_name, cv2.WINDOW_NORMAL)
            window_created = True
            while True:
                frame = camera.read()
                if frame is None:
                    continue
                if preview_settings.mirror:
                    frame = cv2.flip(frame, 1)

                processed = frame_processor(frame) if frame_processor else ()
                result = (
                    processed
                    if isinstance(processed, PreviewFrameResult)
                    else PreviewFrameResult(tuple(processed))
                )
                timestamps.append(monotonic())
                _draw_status(
                    frame,
                    info,
                    _average_fps(timestamps),
                    preview_settings.show_fps,
                    result.lines,
                )
                cv2.imshow(preview_settings.window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in {27, ord("q"), ord("Q")}:
                    break
                if cv2.getWindowProperty(preview_settings.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if result.stop:
                    break
    except cv2.error as exc:
        raise CameraError(f"OpenCV preview failed: {exc}") from exc
    finally:
        if window_created:
            with suppress(cv2.error):
                cv2.destroyWindow(preview_settings.window_name)
