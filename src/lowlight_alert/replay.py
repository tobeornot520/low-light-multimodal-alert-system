"""Deterministic, headless replay of constant-frame-rate video files."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import cv2
import numpy as np

from lowlight_alert.events import AlertEvent
from lowlight_alert.monitor_preview import MonitorFrameStats, MonitorSession


class ReplayError(RuntimeError):
    """Raised when a video replay cannot produce a complete experiment result."""


class VideoCaptureLike(Protocol):
    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def get(self, property_id: int) -> float: ...

    def release(self) -> None: ...


CaptureFactory = Callable[[Path], VideoCaptureLike]
Timer = Callable[[], float]
UtcClock = Callable[[], datetime]


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Paths and headline measurements for one completed replay."""

    source_path: Path
    events_path: Path
    telemetry_path: Path
    report_path: Path
    generated_at: str
    frames_processed: int
    media_duration_seconds: float
    processing_seconds: float
    processing_fps: float | None
    event_ids: tuple[str, ...]
    event_types: tuple[str, ...]
    frame_event_ids: tuple[str, ...] = ()
    finalization_event_ids: tuple[str, ...] = ()
    termination_reason: str = "eof"
    source_complete: bool | None = None


_TELEMETRY_FIELDS = (
    "experiment_id",
    "condition",
    "frame_index",
    "source_time_s",
    "detections",
    "quality_passed",
    "quality_rejected",
    "registered",
    "unknown",
    "uncertain",
    "active_tracks",
    "confirmed_tracks",
    "event_count",
    "event_ids",
    "event_types",
)

_COUNT_FIELDS = (
    "detections",
    "quality_passed",
    "quality_rejected",
    "registered",
    "unknown",
    "uncertain",
)

REPLAY_LIMITATIONS = (
    "Timestamps use frame_index / fps and do not preserve variable-frame-rate PTS.",
    "Replay is headless and does not retain decoded frames or evidence images.",
    "Observed counts are pipeline outputs, not accuracy metrics without ground truth.",
    "Event IDs are run-specific, so separate output files are required for repeated runs.",
    "Finalization events (including machine_flush) describe pipeline cleanup, "
    "not a ground-truth scene exit.",
    "A max_frames run is intentionally truncated and cannot establish source-video completeness.",
)


def _create_capture(path: Path) -> VideoCaptureLike:
    return cv2.VideoCapture(str(path))


def _timer() -> float:
    return perf_counter()


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _file_hash(path: Path, label: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise ReplayError(f"cannot hash {label} {path}: {exc}") from exc
    return digest.hexdigest()


def _path(value: Path | str, label: str) -> Path:
    try:
        return Path(value).expanduser().resolve(strict=False)
    except (OSError, TypeError, ValueError) as exc:
        raise ReplayError(f"{label} must be a valid path") from exc


def _label(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReplayError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_fps(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ReplayError(f"{label} must be a finite positive number")
    try:
        fps = float(value)
    except (TypeError, ValueError) as exc:
        raise ReplayError(f"{label} must be a finite positive number") from exc
    if not isfinite(fps) or fps <= 0:
        raise ReplayError(f"{label} must be a finite positive number")
    return fps


def _optional_capture_number(capture: VideoCaptureLike, property_id: int) -> float | None:
    try:
        value = float(capture.get(property_id))
    except (TypeError, ValueError, cv2.error):
        return None
    return value if isfinite(value) else None


def _positive_integer_metadata(value: float | None) -> int | None:
    if value is None or value <= 0:
        return None
    return round(value)


def _validate_max_frames(max_frames: int | None) -> int | None:
    if max_frames is None:
        return None
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0:
        raise ReplayError("max_frames must be a positive integer")
    return max_frames


def _event_log_path(session: MonitorSession) -> Path:
    event_log = getattr(session, "event_log", None)
    if event_log is None or not hasattr(event_log, "path"):
        raise ReplayError("session must expose event_log.path")
    return _path(event_log.path, "session.event_log.path")


def _artifact_metadata(
    artifact_paths: Mapping[str, Path | str] | None,
) -> dict[str, dict[str, Any]]:
    if artifact_paths is None:
        return {}
    if not isinstance(artifact_paths, Mapping):
        raise ReplayError("artifact_paths must be a mapping of names to files")

    artifacts: dict[str, dict[str, Any]] = {}
    for raw_name, raw_path in artifact_paths.items():
        name = _label(raw_name, "artifact name")
        if name in artifacts:
            raise ReplayError(f"duplicate artifact name after normalization: {name}")
        path = _path(raw_path, f"artifact_paths[{name!r}]")
        if not path.exists():
            raise ReplayError(f"artifact does not exist: {name}={path}")
        try:
            stat = path.stat()
        except OSError as exc:
            raise ReplayError(f"cannot inspect artifact {name} at {path}: {exc}") from exc
        if path.is_file():
            artifacts[name] = {
                "type": "file",
                "path": str(path),
                "sha256": _file_hash(path, f"artifact {name}"),
                "size_bytes": stat.st_size,
                "modified_time_ns": stat.st_mtime_ns,
            }
            continue
        if not path.is_dir():
            raise ReplayError(f"artifact must be a file or directory: {name}={path}")

        digest = hashlib.sha256()
        size_bytes = 0
        file_count = 0
        try:
            entries = sorted(item for item in path.rglob("*") if item.is_file())
        except OSError as exc:
            raise ReplayError(f"cannot enumerate artifact directory {name} at {path}") from exc
        for entry in entries:
            relative = entry.relative_to(path).as_posix()
            entry_hash = _file_hash(entry, f"artifact {name}/{relative}")
            try:
                size_bytes += entry.stat().st_size
            except OSError as exc:
                raise ReplayError(f"cannot inspect artifact {name}/{relative}") from exc
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(entry_hash.encode("ascii"))
            digest.update(b"\n")
            file_count += 1
        artifacts[name] = {
            "type": "directory",
            "path": str(path),
            "sha256": digest.hexdigest(),
            "size_bytes": size_bytes,
            "file_count": file_count,
            "modified_time_ns": stat.st_mtime_ns,
        }
    return dict(sorted(artifacts.items()))


def _generated_at(clock: UtcClock) -> str:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ReplayError("utc_clock must return a timezone-aware datetime")
    return value.astimezone(UTC).isoformat()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def _new_staging_path(path: Path) -> Path:
    """Reserve a unique sibling path and return it without an open handle."""
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".staging",
            dir=path.parent,
        )
        os.close(descriptor)
        staging = Path(name)
        staging.unlink()
        return staging
    except OSError as exc:
        raise ReplayError(f"cannot create staging path beside {path}: {exc}") from exc


def _event_mapping(event: Any) -> dict[str, Any]:
    if isinstance(event, Mapping):
        return dict(event)
    converter = getattr(event, "as_dict", None)
    if not callable(converter):
        converter = getattr(event, "to_dict", None)
    if callable(converter):
        value = converter()
        return dict(value) if isinstance(value, Mapping) else {}
    return {
        "event_id": getattr(event, "event_id", None),
        "event_type": getattr(event, "event_type", None),
        "reason": getattr(event, "reason", None),
    }


def _event_reason(event: Any) -> str | None:
    value = _event_mapping(event).get("reason")
    if value is None or value == "":
        return None
    return str(getattr(value, "value", value))


def _validate_paths(source: Path, events: Path, telemetry: Path, report: Path) -> None:
    if not source.is_file():
        raise ReplayError(f"source video does not exist or is not a file: {source}")
    outputs = {"events": events, "telemetry": telemetry, "report": report}
    if len(set(outputs.values())) != len(outputs):
        raise ReplayError("events, telemetry, and report outputs must be different files")
    if source in set(outputs.values()):
        raise ReplayError("output paths must not overwrite the source video")
    for label, path in outputs.items():
        if path.exists():
            raise ReplayError(f"{label} output already exists: {path}")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ReplayError(
                f"cannot create {label} output directory {path.parent}: {exc}"
            ) from exc
        if not path.parent.is_dir():
            raise ReplayError(f"{label} output parent is not a directory: {path.parent}")


def _publish_staged(staged: Path, final: Path) -> None:
    """Publish one run artifact after all validation has completed."""
    if not staged.is_file():
        raise ReplayError(f"staged output is missing: {staged}")
    try:
        # A hard link is an atomic no-clobber publication on the same volume.
        # It avoids the check-then-replace race that could overwrite another
        # process's artifact between an existence check and rename.
        os.link(staged, final)
    except FileExistsError as exc:
        raise ReplayError(f"output appeared while replay was running: {final}") from exc
    except OSError as exc:
        raise ReplayError(f"cannot publish replay output {final}: {exc}") from exc
    try:
        staged.unlink()
    except OSError as exc:
        with suppress(OSError):
            final.unlink()
        raise ReplayError(f"cannot finish publishing replay output {final}: {exc}") from exc


@contextmanager
def _atomic_text_output(path: Path, *, newline: str | None = None) -> Iterator[Any]:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    output = None
    try:
        output = os.fdopen(descriptor, "w", encoding="utf-8", newline=newline)
        descriptor = -1
        yield output
        output.flush()
        os.fsync(output.fileno())
        output.close()
        output = None
        if path.exists():
            raise ReplayError(f"output appeared while replay was running: {path}")
        os.replace(temporary, path)
    finally:
        if output is not None:
            output.close()
        if descriptor >= 0:
            os.close(descriptor)
        with suppress(OSError):
            temporary.unlink()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    try:
        with _atomic_text_output(path) as output:
            json.dump(payload, output, ensure_ascii=True, allow_nan=False, indent=2)
            output.write("\n")
    except ReplayError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise ReplayError(f"cannot write replay report {path}: {exc}") from exc


def _event_values(events: tuple[AlertEvent, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    event_ids: list[str] = []
    event_types: list[str] = []
    for event in events:
        event_id = str(event.event_id)
        raw_type = event.event_type
        event_type = str(getattr(raw_type, "value", raw_type))
        event_ids.append(event_id)
        event_types.append(event_type)
    return tuple(event_ids), tuple(event_types)


def _event_reasons(events: tuple[AlertEvent, ...]) -> tuple[str | None, ...]:
    return tuple(_event_reason(event) for event in events)


def _read_event_rows(path: Path) -> list[dict[str, Any]]:
    """Read a staged JSONL log strictly enough for run-level auditing."""
    try:
        raw_lines = path.read_bytes().splitlines()
    except OSError as exc:
        raise ReplayError(f"cannot read staged event log {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(raw_lines, start=1):
        if not raw.strip():
            continue
        try:
            value = json.loads(
                raw.decode("utf-8"),
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ReplayError(
                f"staged event log has invalid JSON at line {line_number}: {path}"
            ) from exc
        if not isinstance(value, dict) or not isinstance(value.get("event_id"), str):
            raise ReplayError(f"staged event log has invalid event at line {line_number}: {path}")
        rows.append(value)
    return rows


def _event_log_metadata(
    path: Path,
    expected_ids: tuple[str, ...],
    expected_types: tuple[str, ...],
    event_log: Any,
) -> dict[str, Any]:
    rows = _read_event_rows(path)
    actual_ids = tuple(str(row["event_id"]) for row in rows)
    actual_types = tuple(str(row.get("event_type", "")) for row in rows)
    # A real EventLog can perform its own strict parser check.  Lightweight
    # test doubles may only expose ``path``; their report remains explicitly
    # unverified instead of pretending the file contains the in-memory events.
    verified = callable(getattr(event_log, "read_events", None))
    if verified and (actual_ids != expected_ids or actual_types != expected_types):
        raise ReplayError(
            "event log contents do not match events emitted by the monitoring session"
        )
    try:
        stat = path.stat()
    except OSError as exc:
        raise ReplayError(f"cannot inspect staged event log {path}: {exc}") from exc
    return {
        "path": str(path),
        "sha256": _file_hash(path, "event log"),
        "size_bytes": stat.st_size,
        "record_count": len(rows),
        "verified_against_session": verified,
    }


def _file_metadata(path: Path, label: str, *, row_count: int | None = None) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError as exc:
        raise ReplayError(f"cannot inspect {label} {path}: {exc}") from exc
    metadata: dict[str, Any] = {
        "path": str(path),
        "sha256": _file_hash(path, label),
        "size_bytes": stat.st_size,
    }
    if row_count is not None:
        metadata["row_count"] = row_count
    return metadata


def _telemetry_row(
    experiment_id: str,
    condition: str,
    frame_index: int,
    timestamp: float,
    stats: MonitorFrameStats,
    events: tuple[AlertEvent, ...],
) -> dict[str, Any]:
    event_ids, event_types = _event_values(events)
    return {
        "experiment_id": experiment_id,
        "condition": condition,
        "frame_index": frame_index,
        "source_time_s": f"{timestamp:.9f}",
        "detections": stats.detections,
        "quality_passed": stats.quality_passed,
        "quality_rejected": stats.quality_rejected,
        "registered": stats.registered,
        "unknown": stats.unknown,
        "uncertain": stats.uncertain,
        "active_tracks": stats.active_tracks,
        "confirmed_tracks": stats.confirmed_tracks,
        "event_count": len(events),
        "event_ids": ";".join(event_ids),
        "event_types": ";".join(event_types),
    }


def run_replay(
    source_path: Path | str,
    session: MonitorSession,
    *,
    telemetry_path: Path | str,
    report_path: Path | str,
    condition: str,
    experiment_id: str,
    fps_override: float | None = None,
    mirror: bool = False,
    max_frames: int | None = None,
    artifact_paths: Mapping[str, Path | str] | None = None,
    capture_factory: CaptureFactory | None = None,
    timer: Timer | None = None,
    utc_clock: UtcClock | None = None,
) -> ReplayResult:
    """Replay one CFR video through ``session`` and persist experiment outputs.

    The video is processed as fast as the host permits.  Pipeline time always
    comes from ``frame_index / fps`` and is therefore independent of processing
    speed.  A new ``MonitorSession`` and new output paths must be used for each
    invocation.
    """

    source = _path(source_path, "source_path")
    events = _event_log_path(session)
    telemetry = _path(telemetry_path, "telemetry_path")
    report = _path(report_path, "report_path")
    normalized_condition = _label(condition, "condition")
    normalized_experiment = _label(experiment_id, "experiment_id")
    frame_limit = _validate_max_frames(max_frames)
    if fps_override is not None:
        fps_override = _positive_fps(fps_override, "fps_override")
    _validate_paths(source, events, telemetry, report)
    artifacts = _artifact_metadata(artifact_paths)

    try:
        source_stat = source.stat()
    except OSError as exc:
        raise ReplayError(f"cannot inspect source video {source}: {exc}") from exc
    source_sha256 = _file_hash(source, "source video")
    make_capture = _create_capture if capture_factory is None else capture_factory
    now = _timer if timer is None else timer
    wall_now = _utc_now if utc_clock is None else utc_clock
    event_log = getattr(session, "event_log", None)
    if event_log is None or not hasattr(event_log, "path"):
        raise ReplayError("session must expose event_log.path")
    original_event_log_path = event_log.path
    event_stage = _new_staging_path(events)
    telemetry_stage = _new_staging_path(telemetry)
    report_stage = _new_staging_path(report)
    published: list[Path] = []
    capture: VideoCaptureLike | None = None
    all_event_ids: list[str] = []
    all_event_types: list[str] = []
    frame_event_ids: list[str] = []
    frame_event_types: list[str] = []
    finalization_event_ids: list[str] = []
    finalization_event_types: list[str] = []
    finalization_event_reasons: list[str | None] = []
    totals = {name: 0 for name in _COUNT_FIELDS}
    active_track_observations = 0
    confirmed_track_observations = 0
    peak_active_tracks = 0
    peak_confirmed_tracks = 0
    first_frame_width: int | None = None
    first_frame_height: int | None = None
    frames_processed = 0
    last_timestamp = 0.0
    eof_timestamp = 0.0
    termination_reason = "eof"
    source_complete: bool | None = None
    completed = False

    try:
        # Keep all session-generated JSONL records private until the complete
        # replay has passed decoding, telemetry, and report validation.
        event_log.path = event_stage
        capture = make_capture(source)
        if capture is None or not capture.isOpened():
            raise ReplayError(f"cannot open source video: {source}")

        metadata_fps = _optional_capture_number(capture, cv2.CAP_PROP_FPS)
        if fps_override is None:
            fps = _positive_fps(metadata_fps, "source video FPS")
            fps_source = "video_metadata"
        else:
            fps = fps_override
            fps_source = "override"
        declared_width = _positive_integer_metadata(
            _optional_capture_number(capture, cv2.CAP_PROP_FRAME_WIDTH)
        )
        declared_height = _positive_integer_metadata(
            _optional_capture_number(capture, cv2.CAP_PROP_FRAME_HEIGHT)
        )
        declared_frame_count = _positive_integer_metadata(
            _optional_capture_number(capture, cv2.CAP_PROP_FRAME_COUNT)
        )

        started_at = float(now())
        if not isfinite(started_at):
            raise ReplayError("timer returned a non-finite value")
        with _atomic_text_output(telemetry_stage, newline="") as telemetry_output:
            writer = csv.DictWriter(telemetry_output, fieldnames=_TELEMETRY_FIELDS)
            writer.writeheader()
            while frame_limit is None or frames_processed < frame_limit:
                readable, frame = capture.read()
                if not readable:
                    termination_reason = "eof"
                    break
                if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                    raise ReplayError(
                        f"decoder returned an invalid frame at index {frames_processed}"
                    )
                if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                    raise ReplayError(
                        f"decoded frame {frames_processed} is not a uint8 three-channel BGR image"
                    )
                if first_frame_width is not None and (
                    frame.shape[1] != first_frame_width or frame.shape[0] != first_frame_height
                ):
                    raise ReplayError(
                        f"decoded frame {frames_processed} changes size from "
                        f"{first_frame_width}x{first_frame_height} to "
                        f"{frame.shape[1]}x{frame.shape[0]}"
                    )

                timestamp = frames_processed / fps
                if mirror:
                    frame = cv2.flip(frame, 1)
                session.process_frame_at(frame, timestamp, render=False)
                stats = session.last_frame_stats
                frame_events = tuple(session.last_events)
                writer.writerow(
                    _telemetry_row(
                        normalized_experiment,
                        normalized_condition,
                        frames_processed,
                        timestamp,
                        stats,
                        frame_events,
                    )
                )

                if first_frame_width is None:
                    first_frame_height, first_frame_width = frame.shape[:2]
                for name in _COUNT_FIELDS:
                    totals[name] += int(getattr(stats, name))
                active_track_observations += int(stats.active_tracks)
                confirmed_track_observations += int(stats.confirmed_tracks)
                peak_active_tracks = max(peak_active_tracks, int(stats.active_tracks))
                peak_confirmed_tracks = max(
                    peak_confirmed_tracks, int(stats.confirmed_tracks)
                )
                current_event_ids, current_event_types = _event_values(frame_events)
                frame_event_ids.extend(current_event_ids)
                frame_event_types.extend(current_event_types)
                all_event_ids.extend(current_event_ids)
                all_event_types.extend(current_event_types)
                last_timestamp = timestamp
                frames_processed += 1

            if frames_processed == 0:
                raise ReplayError(f"source video contains no readable frames: {source}")

            if frame_limit is not None and frames_processed >= frame_limit:
                termination_reason = "max_frames"
                source_complete = False
            elif declared_frame_count is not None and frames_processed < declared_frame_count:
                raise ReplayError(
                    f"video ended after {frames_processed} frames, but metadata declares "
                    f"{declared_frame_count} frames"
                )
            elif declared_frame_count is not None:
                source_complete = True

            eof_timestamp = frames_processed / fps
            finish_events = tuple(session.finish_at(eof_timestamp))
            finish_event_ids, finish_event_types = _event_values(finish_events)
            finalization_event_ids.extend(finish_event_ids)
            finalization_event_types.extend(finish_event_types)
            finalization_event_reasons.extend(_event_reasons(finish_events))
            all_event_ids.extend(finish_event_ids)
            all_event_types.extend(finish_event_types)
            if not event_stage.exists():
                event_stage.touch()

        finished_at = float(now())
        if not isfinite(finished_at) or finished_at < started_at:
            raise ReplayError("timer must return finite, non-decreasing values")
        processing_seconds = finished_at - started_at
        processing_fps = (
            frames_processed / processing_seconds if processing_seconds > 0 else None
        )
        media_duration_seconds = eof_timestamp
        generated_at = _generated_at(wall_now)
        frame_event_type_counts = dict(sorted(Counter(frame_event_types).items()))
        finalization_event_type_counts = dict(
            sorted(Counter(finalization_event_types).items())
        )
        all_event_type_counts = dict(sorted(Counter(all_event_types).items()))
        event_log_metadata = _event_log_metadata(
            event_stage,
            tuple(all_event_ids),
            tuple(all_event_types),
            event_log,
        )
        event_log_metadata["path"] = str(events)
        telemetry_metadata = _file_metadata(
            telemetry_stage,
            "telemetry CSV",
            row_count=frames_processed,
        )
        telemetry_metadata["path"] = str(telemetry)
        report_payload: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "generated_at": generated_at,
            "experiment_id": normalized_experiment,
            "condition": normalized_condition,
            "source": {
                "path": str(source),
                "sha256": source_sha256,
                "size_bytes": source_stat.st_size,
                "modified_time_ns": source_stat.st_mtime_ns,
                "declared_width": declared_width,
                "declared_height": declared_height,
                "declared_frame_count": declared_frame_count,
                "metadata_fps": metadata_fps,
                "effective_fps": fps,
                "fps_source": fps_source,
                "decoded_width": first_frame_width,
                "decoded_height": first_frame_height,
            },
            "replay": {
                "headless": True,
                "mirror": bool(mirror),
                "max_frames": frame_limit,
                "timestamp_policy": "frame_index / fps",
                "frames_processed": frames_processed,
                "last_frame_timestamp_seconds": last_timestamp,
                "eof_timestamp_seconds": eof_timestamp,
                "media_duration_seconds": media_duration_seconds,
                "termination_reason": termination_reason,
                "source_complete": source_complete,
                "processing_seconds": processing_seconds,
                "processing_fps": processing_fps,
                "processing_measurement_scope": "decode + session + telemetry write",
            },
            "counts": {
                "frames": frames_processed,
                **totals,
                "active_track_frame_observations": active_track_observations,
                "confirmed_track_frame_observations": confirmed_track_observations,
                "peak_active_tracks": peak_active_tracks,
                "peak_confirmed_tracks": peak_confirmed_tracks,
                "events": len(all_event_ids),
                "frame_events": len(frame_event_ids),
                "finalization_events": len(finalization_event_ids),
                "events_by_type": frame_event_type_counts,
                "all_events_by_type": all_event_type_counts,
                "finalization_events_by_type": finalization_event_type_counts,
            },
            "units": {
                "detections": "face-frame observations",
                "quality_passed": "face-frame observations",
                "quality_rejected": "face-frame observations",
                "registered": "matcher face-frame observations",
                "unknown": "matcher face-frame observations",
                "uncertain": "matcher face-frame observations",
                "processing_fps": "decoded frames per processing second",
            },
            "events": {
                "ids": all_event_ids,
                "types": all_event_types,
                "frame": {
                    "ids": frame_event_ids,
                    "types": frame_event_types,
                },
                "finalization": {
                    "ids": finalization_event_ids,
                    "types": finalization_event_types,
                    "reasons": finalization_event_reasons,
                },
            },
            "artifacts": artifacts,
            "outputs": {
                "events_jsonl": str(events),
                "telemetry_csv": str(telemetry),
                "report_json": str(report),
                "events_metadata": event_log_metadata,
                "telemetry_metadata": telemetry_metadata,
            },
            "limitations": list(REPLAY_LIMITATIONS),
        }
        _write_json_atomic(report_stage, report_payload)
        _publish_staged(event_stage, events)
        published.append(events)
        _publish_staged(telemetry_stage, telemetry)
        published.append(telemetry)
        _publish_staged(report_stage, report)
        published.append(report)
        completed = True
        return ReplayResult(
            source_path=source,
            events_path=events,
            telemetry_path=telemetry,
            report_path=report,
            generated_at=generated_at,
            frames_processed=frames_processed,
            media_duration_seconds=media_duration_seconds,
            processing_seconds=processing_seconds,
            processing_fps=processing_fps,
            event_ids=tuple(all_event_ids),
            event_types=tuple(all_event_types),
            frame_event_ids=tuple(frame_event_ids),
            finalization_event_ids=tuple(finalization_event_ids),
            termination_reason=termination_reason,
            source_complete=source_complete,
        )
    except ReplayError:
        raise
    except Exception as exc:
        raise ReplayError(f"video replay failed for {source}: {exc}") from exc
    finally:
        if capture is not None:
            with suppress(Exception):
                capture.release()
        with suppress(Exception):
            event_log.path = original_event_log_path
        for staging in (event_stage, telemetry_stage, report_stage):
            with suppress(OSError):
                staging.unlink()
        if not completed:
            for output in published:
                with suppress(OSError):
                    output.unlink()


run_video_replay = run_replay
