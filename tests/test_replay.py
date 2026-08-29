from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import lowlight_alert.replay as replay_module
from lowlight_alert.event_log import EventLog
from lowlight_alert.events import EventType
from lowlight_alert.monitor_preview import MonitorFrameStats
from lowlight_alert.replay import ReplayError, run_replay


@dataclass(frozen=True)
class _Event:
    event_id: str
    event_type: EventType


class _Capture:
    def __init__(
        self,
        frames: list[np.ndarray],
        *,
        fps: float = 2.0,
        opened: bool = True,
        declared_frame_count: int | None = None,
    ) -> None:
        self.frames = [frame.copy() for frame in frames]
        self.fps = fps
        self.opened = opened
        self.declared_frame_count = declared_frame_count
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def get(self, property_id: int) -> float:
        if property_id == cv2.CAP_PROP_FPS:
            return self.fps
        if property_id == cv2.CAP_PROP_FRAME_WIDTH:
            return 4.0
        if property_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return 2.0
        if property_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(
                len(self.frames)
                if self.declared_frame_count is None
                else self.declared_frame_count
            )
        return 0.0

    def release(self) -> None:
        self.released = True


class _Session:
    def __init__(
        self,
        *,
        events_by_frame: dict[int, tuple[_Event, ...]] | None = None,
        finish_events: tuple[_Event, ...] = (),
        fail_at: int | None = None,
        event_log_path: Path | None = None,
        write_event_log: bool = False,
    ) -> None:
        self.frames: list[np.ndarray] = []
        self.timestamps: list[float] = []
        self.render_values: list[bool] = []
        self.finish_timestamps: list[float] = []
        self.last_frame_stats = MonitorFrameStats()
        self.last_events: tuple[_Event, ...] = ()
        self.events_by_frame = events_by_frame or {}
        self.finish_events = finish_events
        self.fail_at = fail_at
        if event_log_path is None:
            event_log_path = Path(tempfile.gettempdir()) / f"unused-events-{id(self)}.jsonl"
        self.event_log = (
            EventLog(event_log_path)
            if write_event_log
            else SimpleNamespace(path=event_log_path)
        )

    def process_frame_at(self, frame: np.ndarray, timestamp: float, *, render: bool) -> None:
        index = len(self.frames)
        if self.fail_at == index:
            raise RuntimeError("synthetic processing failure")
        self.frames.append(frame.copy())
        self.timestamps.append(timestamp)
        self.render_values.append(render)
        self.last_events = self.events_by_frame.get(index, ())
        if hasattr(self.event_log, "append_many"):
            self.event_log.append_many(self.last_events)
        self.last_frame_stats = MonitorFrameStats(
            detections=index + 1,
            quality_passed=1,
            quality_rejected=index,
            registered=1,
            unknown=index % 2,
            uncertain=0,
            active_tracks=index + 1,
            confirmed_tracks=index,
            events=tuple(event.event_type for event in self.last_events),
        )

    def finish_at(self, timestamp: float) -> tuple[_Event, ...]:
        self.finish_timestamps.append(timestamp)
        self.last_events = self.finish_events
        if hasattr(self.event_log, "append_many"):
            self.event_log.append_many(self.finish_events)
        return self.finish_events


def _frame(left: int = 0, right: int = 0) -> np.ndarray:
    frame = np.zeros((2, 4, 3), dtype=np.uint8)
    frame[:, :2] = left
    frame[:, 2:] = right
    return frame


def _paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"constant-frame-rate-video")
    return source, tmp_path / "telemetry.csv", tmp_path / "report.json"


def _run(
    tmp_path: Path,
    capture: _Capture,
    session: _Session,
    **overrides,
):
    source, telemetry, report = _paths(tmp_path)
    options = {
        "telemetry_path": telemetry,
        "report_path": report,
        "condition": "dark",
        "experiment_id": "exp-01",
        "capture_factory": lambda path: capture,
        "timer": iter((10.0, 12.0)).__next__,
        "utc_clock": lambda: datetime(2026, 8, 21, 12, 0, tzinfo=UTC),
    }
    session.event_log.path = tmp_path / "events.jsonl"
    options.update(overrides)
    result = run_replay(source, session, **options)
    return result, telemetry, report


def test_missing_source_is_rejected_before_capture(tmp_path: Path) -> None:
    called = False

    def factory(path: Path):
        nonlocal called
        called = True
        return _Capture([])

    with pytest.raises(ReplayError, match="does not exist"):
        run_replay(
            tmp_path / "missing.mp4",
            _Session(),
            telemetry_path=tmp_path / "telemetry.csv",
            report_path=tmp_path / "report.json",
            condition="dark",
            experiment_id="exp-01",
            capture_factory=factory,
        )

    assert called is False


@pytest.mark.parametrize("fps", [0.0, -1.0, float("nan")])
def test_invalid_source_fps_is_rejected_and_capture_released(
    tmp_path: Path, fps: float
) -> None:
    source, telemetry, report = _paths(tmp_path)
    capture = _Capture([_frame()], fps=fps)

    with pytest.raises(ReplayError, match="source video FPS"):
        run_replay(
            source,
            _Session(),
            telemetry_path=telemetry,
            report_path=report,
            condition="dark",
            experiment_id="exp-01",
            capture_factory=lambda path: capture,
        )

    assert capture.released is True
    assert not telemetry.exists()
    assert not report.exists()


@pytest.mark.parametrize("fps_override", [0.0, -1.0, float("inf"), True])
def test_invalid_fps_override_is_rejected(tmp_path: Path, fps_override) -> None:
    source, telemetry, report = _paths(tmp_path)

    with pytest.raises(ReplayError, match="fps_override"):
        run_replay(
            source,
            _Session(),
            telemetry_path=telemetry,
            report_path=report,
            condition="dark",
            experiment_id="exp-01",
            fps_override=fps_override,
        )


def test_zero_decoded_frames_is_an_error_and_releases_capture(tmp_path: Path) -> None:
    source, telemetry, report = _paths(tmp_path)
    capture = _Capture([], fps=25.0)
    session = _Session()

    with pytest.raises(ReplayError, match="no readable frames"):
        run_replay(
            source,
            session,
            telemetry_path=telemetry,
            report_path=report,
            condition="dark",
            experiment_id="exp-01",
            capture_factory=lambda path: capture,
        )

    assert capture.released is True
    assert session.finish_timestamps == []
    assert not telemetry.exists()
    assert not report.exists()


def test_uses_frame_index_over_fps_and_honors_max_frames(tmp_path: Path) -> None:
    capture = _Capture([_frame(), _frame(), _frame(), _frame()], fps=99.0)
    session = _Session()

    result, telemetry, _ = _run(
        tmp_path,
        capture,
        session,
        fps_override=4.0,
        max_frames=3,
    )

    assert result.frames_processed == 3
    assert session.timestamps == [0.0, 0.25, 0.5]
    assert session.finish_timestamps == [0.75]
    assert session.render_values == [False, False, False]
    with telemetry.open(newline="", encoding="utf-8") as source:
        assert [float(row["source_time_s"]) for row in csv.DictReader(source)] == [
            0.0,
            0.25,
            0.5,
        ]
    assert result.events_path.is_file()
    assert result.events_path.read_bytes() == b""


def test_mirror_is_applied_before_session_processing(tmp_path: Path) -> None:
    capture = _Capture([_frame(left=10, right=200)])
    session = _Session()

    _run(tmp_path, capture, session, mirror=True)

    assert np.all(session.frames[0][:, :2] == 200)
    assert np.all(session.frames[0][:, 2:] == 10)


def test_processing_failure_releases_capture_without_outputs(tmp_path: Path) -> None:
    source, telemetry, report = _paths(tmp_path)
    capture = _Capture([_frame(), _frame()])

    with pytest.raises(ReplayError, match="synthetic processing failure"):
        run_replay(
            source,
            _Session(fail_at=1),
            telemetry_path=telemetry,
            report_path=report,
            condition="dark",
            experiment_id="exp-01",
            capture_factory=lambda path: capture,
            timer=lambda: 1.0,
        )

    assert capture.released is True
    assert not telemetry.exists()
    assert not report.exists()


def test_report_and_telemetry_contain_stats_events_and_source_metadata(tmp_path: Path) -> None:
    enter = _Event("event-enter", EventType.ENTER)
    leave = _Event("event-leave", EventType.LEAVE)
    capture = _Capture([_frame(), _frame()], fps=2.0)
    session = _Session(events_by_frame={0: (enter,)}, finish_events=(leave,))

    result, telemetry, report = _run(tmp_path, capture, session)

    assert result.processing_seconds == 2.0
    assert result.processing_fps == 1.0
    assert result.event_ids == ("event-enter", "event-leave")
    with telemetry.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 2
    assert rows[0]["experiment_id"] == "exp-01"
    assert rows[0]["condition"] == "dark"
    assert rows[0]["detections"] == "1"
    assert rows[0]["event_ids"] == "event-enter"
    assert rows[1]["quality_rejected"] == "1"

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["status"] == "complete"
    assert payload["generated_at"] == "2026-08-21T12:00:00+00:00"
    assert payload["experiment_id"] == "exp-01"
    assert payload["condition"] == "dark"
    assert payload["source"]["sha256"] == hashlib.sha256(
        b"constant-frame-rate-video"
    ).hexdigest()
    assert payload["source"]["effective_fps"] == 2.0
    assert payload["counts"]["frames"] == 2
    assert payload["counts"]["detections"] == 3
    assert payload["counts"]["events_by_type"] == {"enter": 1}
    assert payload["counts"]["finalization_events_by_type"] == {"leave": 1}
    assert payload["replay"]["termination_reason"] == "eof"
    assert payload["replay"]["source_complete"] is True
    assert payload["units"]["registered"] == "matcher face-frame observations"
    assert payload["events"]["ids"] == ["event-enter", "event-leave"]
    assert payload["replay"]["headless"] is True
    assert payload["replay"]["timestamp_policy"] == "frame_index / fps"
    assert payload["replay"]["last_frame_timestamp_seconds"] == 0.5
    assert payload["replay"]["eof_timestamp_seconds"] == 1.0
    assert payload["outputs"]["events_jsonl"].endswith("events.jsonl")
    assert payload["limitations"]
    assert capture.released is True


def test_max_frames_is_explicitly_truncated_and_flush_events_are_separate(
    tmp_path: Path,
) -> None:
    capture = _Capture([_frame(), _frame(), _frame()], fps=2.0)
    leave = _Event("event-leave", EventType.LEAVE)
    session = _Session(finish_events=(leave,))

    result, _, report = _run(tmp_path, capture, session, max_frames=2)

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert result.termination_reason == "max_frames"
    assert result.source_complete is False
    assert session.finish_timestamps == [1.0]
    assert payload["replay"]["termination_reason"] == "max_frames"
    assert payload["replay"]["source_complete"] is False
    assert payload["counts"]["frame_events"] == 0
    assert payload["counts"]["finalization_events"] == 1
    assert payload["events"]["finalization"]["ids"] == ["event-leave"]


def test_missing_declared_frame_count_leaves_completeness_unknown(tmp_path: Path) -> None:
    capture = _Capture([_frame()], declared_frame_count=0)

    result, _, report = _run(tmp_path, capture, _Session())

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert result.termination_reason == "eof"
    assert result.source_complete is None
    assert payload["replay"]["source_complete"] is None


def test_declared_frame_count_mismatch_is_rejected_without_final_outputs(
    tmp_path: Path,
) -> None:
    source, telemetry, report = _paths(tmp_path)
    capture = _Capture([_frame(), _frame()], declared_frame_count=3)
    session = _Session()

    with pytest.raises(ReplayError, match="metadata declares 3 frames"):
        run_replay(
            source,
            session,
            telemetry_path=telemetry,
            report_path=report,
            condition="dim",
            experiment_id="exp-truncated",
            capture_factory=lambda path: capture,
        )

    assert capture.released is True
    assert not telemetry.exists()
    assert not report.exists()
    assert not (tmp_path / "events.jsonl").exists()


def test_frame_size_change_is_rejected_without_final_outputs(tmp_path: Path) -> None:
    source, telemetry, report = _paths(tmp_path)
    capture = _Capture([_frame(), np.zeros((3, 4, 3), dtype=np.uint8)])
    session = _Session()

    with pytest.raises(ReplayError, match="changes size"):
        run_replay(
            source,
            session,
            telemetry_path=telemetry,
            report_path=report,
            condition="normal",
            experiment_id="exp-size-change",
            capture_factory=lambda path: capture,
        )

    assert not telemetry.exists()
    assert not report.exists()


def test_real_event_log_is_verified_and_empty_runs_are_materialized(tmp_path: Path) -> None:
    capture = _Capture([_frame()])
    session = _Session(write_event_log=True)

    result, _, report = _run(tmp_path, capture, session)

    payload = json.loads(report.read_text(encoding="utf-8"))
    metadata = payload["outputs"]["events_metadata"]
    assert result.events_path.is_file()
    assert metadata["verified_against_session"] is True
    assert metadata["record_count"] == 0
    assert metadata["size_bytes"] == 0


@pytest.mark.parametrize("existing", ["telemetry", "report"])
def test_existing_output_is_rejected(tmp_path: Path, existing: str) -> None:
    source, telemetry, report = _paths(tmp_path)
    target = telemetry if existing == "telemetry" else report
    target.write_text("keep", encoding="utf-8")

    with pytest.raises(ReplayError, match="already exists"):
        run_replay(
            source,
            _Session(),
            telemetry_path=telemetry,
            report_path=report,
            condition="dark",
            experiment_id="exp-01",
        )

    assert target.read_text(encoding="utf-8") == "keep"


def test_existing_event_log_and_output_path_conflicts_are_rejected(tmp_path: Path) -> None:
    source, telemetry, report = _paths(tmp_path)
    events = tmp_path / "events.jsonl"
    events.write_text("existing\n", encoding="utf-8")
    session = _Session(event_log_path=events)

    with pytest.raises(ReplayError, match="events output already exists"):
        run_replay(
            source,
            session,
            telemetry_path=telemetry,
            report_path=report,
            condition="dark",
            experiment_id="exp-01",
        )

    events.unlink()
    session.event_log.path = telemetry
    with pytest.raises(ReplayError, match="must be different"):
        run_replay(
            source,
            session,
            telemetry_path=telemetry,
            report_path=report,
            condition="dark",
            experiment_id="exp-01",
        )


def test_report_hashes_named_provenance_artifacts(tmp_path: Path) -> None:
    config = tmp_path / "experiment.yaml"
    yunet = tmp_path / "yunet.onnx"
    sface = tmp_path / "sface.onnx"
    config.write_bytes(b"config")
    yunet.write_bytes(b"yunet")
    sface.write_bytes(b"sface")

    _, _, report = _run(
        tmp_path,
        _Capture([_frame()]),
        _Session(),
        artifact_paths={"config": config, "yunet": yunet, "sface": sface},
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert list(payload["artifacts"]) == ["config", "sface", "yunet"]
    assert payload["artifacts"]["config"]["sha256"] == hashlib.sha256(b"config").hexdigest()
    assert payload["artifacts"]["yunet"]["size_bytes"] == len(b"yunet")


def test_report_hashes_template_directory_as_one_artifact(tmp_path: Path) -> None:
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "person-a.npz").write_bytes(b"template-a")
    (templates / "person-b.npz").write_bytes(b"template-b")

    _, _, report = _run(
        tmp_path,
        _Capture([_frame()]),
        _Session(),
        artifact_paths={"templates": templates},
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    metadata = payload["artifacts"]["templates"]
    assert metadata["type"] == "directory"
    assert metadata["file_count"] == 2
    assert metadata["size_bytes"] == len(b"template-a") + len(b"template-b")
    assert len(metadata["sha256"]) == 64


def test_missing_provenance_artifact_is_rejected(tmp_path: Path) -> None:
    source, telemetry, report = _paths(tmp_path)

    with pytest.raises(ReplayError, match="artifact does not exist"):
        run_replay(
            source,
            _Session(event_log_path=tmp_path / "events.jsonl"),
            telemetry_path=telemetry,
            report_path=report,
            condition="dark",
            experiment_id="exp-01",
            artifact_paths={"config": tmp_path / "missing.yaml"},
        )


def test_module_capture_factory_and_timer_can_be_monkeypatched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, telemetry, report = _paths(tmp_path)
    capture = _Capture([_frame()])
    times = iter((4.0, 5.0))
    monkeypatch.setattr(replay_module, "_create_capture", lambda path: capture)
    monkeypatch.setattr(replay_module, "_timer", times.__next__)

    result = run_replay(
        source,
        _Session(),
        telemetry_path=telemetry,
        report_path=report,
        condition="normal",
        experiment_id="exp-monkeypatch",
    )

    assert result.processing_seconds == 1.0
    assert capture.released is True
