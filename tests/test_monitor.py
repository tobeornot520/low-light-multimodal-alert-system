from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest

from lowlight_alert.detector import FaceDetection
from lowlight_alert.event_log import EventLog
from lowlight_alert.events import EventType, Polygon, Zone, ZoneStateMachine
from lowlight_alert.monitor_preview import MonitorError, MonitorSession
from lowlight_alert.quality import FaceQuality
from lowlight_alert.recognizer import IdentityMatch, MatchState
from lowlight_alert.tracking import SimpleTrackManager


def _detection(box: tuple[int, int, int, int] = (20, 20, 40, 40)) -> FaceDetection:
    x, y, width, height = box
    return FaceDetection(
        box=box,
        landmarks=(
            (x + width // 3, y + height // 3),
            (x + 2 * width // 3, y + height // 3),
            (x + width // 2, y + height // 2),
            (x + width // 3, y + 2 * height // 3),
            (x + 2 * width // 3, y + 2 * height // 3),
        ),
        score=0.95,
    )


class _SequenceDetector:
    def __init__(self, frames: Iterable[list[FaceDetection]]) -> None:
        self._frames = iter(frames)

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        del frame
        return next(self._frames)


class _PassingEvaluator:
    def evaluate(self, frame: np.ndarray, detection: FaceDetection) -> FaceQuality:
        del frame, detection
        return FaceQuality(
            issues=(),
            face_size=40,
            brightness=120.0,
            sharpness=100.0,
            yaw_ratio=0.0,
            nose_position=0.5,
        )


class _Extractor:
    def extract(self, frame: np.ndarray, detection: FaceDetection) -> np.ndarray:
        del frame, detection
        return np.ones(4, dtype=np.float32)


class _Matcher:
    subject_count = 1

    def match(self, feature: np.ndarray) -> IdentityMatch:
        del feature
        return IdentityMatch(
            state=MatchState.REGISTERED,
            subject_id="person-a",
            display_name="Person A",
            similarity=0.91,
        )


def _session(
    tmp_path: Path,
    frames: Iterable[list[FaceDetection]],
    times: Iterable[float],
    *,
    max_missing_frames: int = 5,
    lost_tolerance_seconds: float = 1.0,
) -> MonitorSession:
    return MonitorSession(
        detector=_SequenceDetector(frames),
        evaluator=_PassingEvaluator(),
        extractor=_Extractor(),
        matcher=_Matcher(),
        tracker=SimpleTrackManager(
            confirm_frames=1,
            max_missing_frames=max_missing_frames,
        ),
        # The whole normalized frame is the monitored region for the synthetic
        # replay, so the detection centroid is always inside.
        zones=(Zone("room", Polygon([(0, 0), (1, 0), (1, 1)])),),
        event_machine=ZoneStateMachine(
            {"room": Polygon([(0, 0), (1, 0), (1, 1)])},
            dwell_seconds=99.0,
            lost_tolerance_seconds=lost_tolerance_seconds,
            confirm_frames=1,
        ),
        event_log=EventLog(tmp_path / "events.jsonl"),
        clock=iter(times).__next__,
    )


def _process(session: MonitorSession) -> tuple[EventType, ...]:
    result = session.process_frame(np.zeros((100, 100, 3), dtype=np.uint8))
    del result
    return tuple(event.event_type for event in session.last_events)


def test_short_occlusion_does_not_emit_leave_or_duplicate_enter(tmp_path: Path) -> None:
    face = _detection()
    session = _session(
        tmp_path,
        [[face], [], [face]],
        [0.0, 0.1, 0.2],
    )

    assert _process(session) == (EventType.ENTER,)
    assert _process(session) == ()
    assert _process(session) == ()
    assert [row["event_type"] for row in session.event_log.read_events()] == ["enter"]


def test_missing_track_expires_after_real_tolerance(tmp_path: Path) -> None:
    face = _detection()
    session = _session(
        tmp_path,
        [[face], [], [], []],
        [0.0, 0.1, 0.5, 1.2],
    )

    assert _process(session) == (EventType.ENTER,)
    assert _process(session) == ()
    assert _process(session) == ()
    assert _process(session) == (EventType.LEAVE,)
    rows = session.event_log.read_events()
    assert [row["event_type"] for row in rows] == ["enter", "leave"]
    assert rows[-1]["reason"] == "track_missing_timeout"


def test_tracker_end_emits_one_immediate_leave(tmp_path: Path) -> None:
    face = _detection()
    session = _session(
        tmp_path,
        [[face], []],
        [0.0, 0.1],
        max_missing_frames=0,
        lost_tolerance_seconds=10.0,
    )

    assert _process(session) == (EventType.ENTER,)
    assert _process(session) == (EventType.LEAVE,)
    assert session.event_machine.tracks == {}
    assert [row["event_type"] for row in session.event_log.read_events()] == [
        "enter",
        "leave",
    ]
    final_event = session.event_log.read_events()[-1]
    assert final_event["reason"] == "tracker_ended"
    assert "track_lost" in final_event["evidence_flags"]


def test_explicit_timestamp_and_headless_stats_are_deterministic(tmp_path: Path) -> None:
    face = _detection()
    session = _session(tmp_path, [[face]], [])
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    original = frame.copy()

    session.process_frame_at(frame, 2.5, render=False)

    assert np.array_equal(frame, original)
    assert session.last_events[0].observed_at == 2.5
    assert session.last_frame_stats.detections == 1
    assert session.last_frame_stats.quality_passed == 1
    assert session.last_frame_stats.registered == 1
    assert session.last_frame_stats.confirmed_tracks == 1

    with pytest.raises(MonitorError, match="moved backwards"):
        session.process_frame_at(frame, 2.4, render=False)


def test_finish_at_uses_media_time_and_resets_tracker(tmp_path: Path) -> None:
    face = _detection()
    session = _session(tmp_path, [[face]], [])
    session.process_frame_at(np.zeros((100, 100, 3), dtype=np.uint8), 0.0, render=False)

    events = session.finish_at(1.0)

    assert events[0].event_type is EventType.LEAVE
    assert events[0].observed_at == 1.0
    assert events[0].duration == 1.0
    assert session.tracker.reset(1.0).ended == ()
