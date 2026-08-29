"""Durable, local JSONL storage for alert events.

The log deliberately stores dictionaries instead of trying to reconstruct a
particular event class when reading.  This keeps the file format forward
compatible: producers may add fields, while consumers can still inspect old
records.  Appends are serialized, flushed, and fsynced.  If a process dies in
the middle of an append, the next append removes an incomplete final line (or
finishes a complete JSON value that only missed its newline).
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, time
from enum import Enum
from pathlib import Path
from typing import Any, Literal

try:  # ``fcntl`` is unavailable on Windows; O_APPEND still gives line writes.
    import fcntl
except ImportError:  # pragma: no cover - exercised only on Windows
    fcntl = None  # type: ignore[assignment]


class EventLogError(RuntimeError):
    """Raised when an event cannot be serialized or the log cannot be used."""


class EventLogReadError(EventLogError):
    """Raised for malformed input when strict reading is requested."""


class EventLogWriteError(EventLogError):
    """Raised when an append or recovery operation fails."""


@dataclass(frozen=True, slots=True)
class BadEventLine:
    """Description of a malformed JSONL record.

    The object is useful to a caller that wants to count or audit skipped
    records without making the normal read path fail.
    """

    path: Path
    line_number: int
    reason: str
    raw: bytes


class BadEventLineError(EventLogReadError):
    """A malformed event line encountered while reading in strict mode."""

    def __init__(self, problem: BadEventLine) -> None:
        self.problem = problem
        self.path = problem.path
        self.line_number = problem.line_number
        self.reason = problem.reason
        self.raw = problem.raw
        super().__init__(
            f"invalid event log line {problem.path}, line {problem.line_number}: {problem.reason}"
        )


# This fallback mirrors the event object used by the event state-machine
# module.  At runtime event_log can also consume that module's AlertEvent (via
# ``as_dict``), so there is no import cycle or hard dependency here.
@dataclass(frozen=True, slots=True)
class AlertEvent:
    """Small serializable event value object for callers that need one.

    The logger accepts richer event classes and mappings as well.  Numeric
    timestamps are intentionally supported because the state machine may use
    monotonic/epoch values before a presentation layer formats them.
    """

    event_id: str
    event_type: str
    track_id: str | None = None
    zone: str | None = None
    observed_at: float | str | None = None
    first_seen: float | str | None = None
    duration: float = 0.0
    identity_state: str = "unknown"
    subject_id: str | None = None
    confidence: float | None = None
    quality_state: Any = None
    severity: int = 0
    evidence_flags: tuple[str, ...] = ()
    reason: str | None = None
    camera_id: str | None = None
    delivery_state: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly shallow representation of this event."""
        value = asdict(self)
        value["evidence_flags"] = list(self.evidence_flags)
        return value


BadLinePolicy = Literal["skip", "raise"]
BadLineCallback = Callable[[BadEventLine], None]


_PATH_LOCKS: dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _path_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[key] = lock
        return lock


def _jsonable(value: Any) -> Any:
    """Convert common value objects before strict JSON encoding."""
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    # NumPy scalars and similar numeric wrappers expose ``item``.  Avoid
    # importing NumPy merely for event logging.
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (str, bytes, bytearray)):
        try:
            converted = item()
        except (TypeError, ValueError):
            pass
        else:
            if converted is not value:
                return _jsonable(converted)
    return value


def event_mapping(event: AlertEvent | Mapping[str, Any] | Any) -> dict[str, Any]:
    """Normalize an event value to a validated, JSON-serializable mapping.

    ``as_dict`` and ``to_dict`` are preferred so domain objects control their
    wire representation.  Dataclasses and ordinary objects with ``__dict__``
    are supported as a convenience for tests and small integrations.
    """
    if isinstance(event, Mapping):
        raw = dict(event)
    else:
        converter = getattr(event, "as_dict", None)
        if not callable(converter):
            converter = getattr(event, "to_dict", None)
        if callable(converter):
            raw = converter()
        elif is_dataclass(event) and not isinstance(event, type):
            raw = asdict(event)
        elif hasattr(event, "__dict__"):
            raw = dict(vars(event))
        else:
            raise EventLogError("event must be a mapping or expose as_dict()/to_dict()")
        if not isinstance(raw, Mapping):
            raise EventLogError("event serializer must return a mapping")
        raw = dict(raw)

    if "event_id" not in raw:
        raise EventLogError("event must contain event_id")
    event_id = raw["event_id"]
    if not isinstance(event_id, str) or not event_id.strip():
        raise EventLogError("event_id must be a non-empty string")
    if "\n" in event_id or "\r" in event_id:
        raise EventLogError("event_id cannot contain line breaks")

    normalized = _jsonable(raw)
    if not isinstance(normalized, dict):  # defensive; _jsonable preserves maps
        raise EventLogError("event serializer must return a mapping")
    try:
        # Validate now so an append never writes a partially serializable row.
        json.dumps(normalized, ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise EventLogError(f"event is not JSON serializable: {exc}") from exc
    return normalized


def serialize_event(event: AlertEvent | Mapping[str, Any] | Any) -> bytes:
    """Serialize an event as one UTF-8 JSONL record, including its newline."""
    normalized = event_mapping(event)
    try:
        encoded = json.dumps(
            normalized,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:  # validation above should catch this
        raise EventLogError(f"event is not JSON serializable: {exc}") from exc
    return encoded + b"\n"


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


class EventLog:
    """Append-only, recoverable JSONL event log.

    ``append`` returns ``True`` when a new event was written and ``False`` for
    an already present ``event_id``.  Existing malformed lines are ignored by
    default; pass ``bad_line_policy="raise"`` (or ``strict=True``) to fail
    reads and duplicate scans instead.
    """

    def __init__(
        self,
        path: Path,
        *,
        bad_line_policy: BadLinePolicy = "skip",
        strict: bool | None = None,
        skip_bad_lines: bool | None = None,
    ) -> None:
        self.path = Path(path)
        if strict is not None:
            bad_line_policy = "raise" if strict else "skip"
        if skip_bad_lines is not None:
            bad_line_policy = "skip" if skip_bad_lines else "raise"
        if bad_line_policy not in {"skip", "raise"}:
            raise EventLogError("bad_line_policy must be 'skip' or 'raise'")
        self.bad_line_policy: BadLinePolicy = bad_line_policy
        self.last_bad_lines: tuple[BadEventLine, ...] = ()

    @property
    def exists(self) -> bool:
        return self.path.is_file()

    def append(self, event: AlertEvent | Mapping[str, Any] | Any) -> bool:
        """Append ``event`` unless its ``event_id`` is already present."""
        line = serialize_event(event)
        event_id = json.loads(line[:-1].decode("utf-8"))["event_id"]
        try:
            with self._locked_file(exclusive=True) as output:
                self._repair_tail(output)
                existing = self._event_ids(output)
                if event_id in existing:
                    return False
                self._write_bytes(output, line)
                return True
        except EventLogError:
            raise
        except OSError as exc:
            raise EventLogWriteError(f"cannot append event to {self.path}: {exc}") from exc

    append_event = append

    def append_many(self, events: Iterable[AlertEvent | Mapping[str, Any] | Any]) -> int:
        """Append a batch and return the number of newly written events."""
        lines: list[bytes] = []
        ids: set[str] = set()
        for event in events:
            line = serialize_event(event)
            event_id = json.loads(line[:-1].decode("utf-8"))["event_id"]
            if event_id not in ids:
                lines.append(line)
                ids.add(event_id)
        if not lines:
            return 0
        try:
            with self._locked_file(exclusive=True) as output:
                self._repair_tail(output)
                existing = self._event_ids(output)
                pending = [
                    line for line in lines if json.loads(line[:-1])["event_id"] not in existing
                ]
                if not pending:
                    return 0
                self._write_bytes(output, b"".join(pending))
                return len(pending)
        except EventLogError:
            raise
        except OSError as exc:
            raise EventLogWriteError(f"cannot append events to {self.path}: {exc}") from exc

    def iter_events(
        self,
        *,
        bad_line_policy: BadLinePolicy | None = None,
        strict: bool | None = None,
        skip_bad_lines: bool | None = None,
        on_bad_line: BadLineCallback | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield decoded event mappings in file order.

        A missing log is equivalent to an empty log.  Blank lines are ignored.
        Malformed non-blank lines are reported to ``on_bad_line`` and either
        skipped or raised according to the selected policy.
        """
        policy = self._resolve_policy(bad_line_policy, strict, skip_bad_lines)
        if not self.path.is_file():
            self.last_bad_lines = ()
            return
        problems: list[BadEventLine] = []
        try:
            with self._locked_file(exclusive=False) as source:
                source.seek(0)
                data = source.read()
        except OSError as exc:
            raise EventLogReadError(f"cannot read event log {self.path}: {exc}") from exc

        for line_number, raw in enumerate(data.splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                value = self._decode_line(raw, line_number)
            except BadEventLineError as exc:
                problem = exc.problem
                problems.append(problem)
                if on_bad_line is not None:
                    on_bad_line(problem)
                if policy == "raise":
                    self.last_bad_lines = tuple(problems)
                    raise
                continue
            yield value
        self.last_bad_lines = tuple(problems)

    def read_events(
        self,
        *,
        bad_line_policy: BadLinePolicy | None = None,
        strict: bool | None = None,
        skip_bad_lines: bool | None = None,
        on_bad_line: BadLineCallback | None = None,
    ) -> list[dict[str, Any]]:
        """Read all valid events, preserving their JSON object fields."""
        return list(
            self.iter_events(
                bad_line_policy=bad_line_policy,
                strict=strict,
                skip_bad_lines=skip_bad_lines,
                on_bad_line=on_bad_line,
            )
        )

    read = read_events

    def contains(self, event_id: str) -> bool:
        """Return whether a valid record with ``event_id`` exists."""
        if not isinstance(event_id, str) or not event_id.strip():
            raise EventLogError("event_id must be a non-empty string")
        return any(item.get("event_id") == event_id for item in self.iter_events())

    def repair(self) -> bool:
        """Repair an incomplete final line; return whether the file changed."""
        if not self.path.is_file():
            return False
        try:
            with self._locked_file(exclusive=True) as output:
                return self._repair_tail(output)
        except OSError as exc:
            raise EventLogWriteError(f"cannot repair event log {self.path}: {exc}") from exc

    def _resolve_policy(
        self,
        bad_line_policy: BadLinePolicy | None,
        strict: bool | None,
        skip_bad_lines: bool | None,
    ) -> BadLinePolicy:
        policy: str = bad_line_policy or self.bad_line_policy
        if strict is not None:
            policy = "raise" if strict else "skip"
        if skip_bad_lines is not None:
            policy = "skip" if skip_bad_lines else "raise"
        if policy not in {"skip", "raise"}:
            raise EventLogReadError("bad_line_policy must be 'skip' or 'raise'")
        return policy  # type: ignore[return-value]

    @contextmanager
    def _locked_file(self, *, exclusive: bool) -> Iterator[Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = _path_lock(self.path)
        # Keep open/read/write errors visible to the public caller so it can
        # classify them as read or write failures.  Wrapping the whole context
        # here would incorrectly turn a read error into a write error.
        with lock, open(self.path, "a+b", buffering=0) as output:
            try:
                if fcntl is not None:
                    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
                    fcntl.flock(output.fileno(), operation)
                yield output
            finally:
                if fcntl is not None:
                    with suppress(OSError):
                        fcntl.flock(output.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _write_bytes(output: Any, payload: bytes) -> None:
        # FileIO.write may legally perform a short write.  Looping keeps a
        # record whole under normal operation; an interrupted process is
        # repaired on the next append.
        view = memoryview(payload)
        while view:
            try:
                written = output.write(view)
            except OSError as exc:
                raise EventLogWriteError(f"cannot write event log: {exc}") from exc
            if not written:
                raise EventLogWriteError("event log write made no progress")
            view = view[written:]
        output.flush()
        try:
            os.fsync(output.fileno())
        except OSError as exc:
            raise EventLogWriteError(f"cannot flush event log: {exc}") from exc

    def _repair_tail(self, output: Any) -> bool:
        output.seek(0)
        data = output.read()
        if not data or data.endswith(b"\n"):
            return False
        last_newline = data.rfind(b"\n")
        tail = data[last_newline + 1 :]
        try:
            value = self._decode_line(tail, data[: last_newline + 1].count(b"\n") + 1)
        except BadEventLineError:
            if self.bad_line_policy == "raise":
                raise
            output.seek(last_newline + 1)
            output.truncate()
            output.flush()
            os.fsync(output.fileno())
            return True
        # The JSON value was complete but the process died before writing its
        # newline.  Preserve it and make the file valid JSONL.
        del value
        output.seek(0, os.SEEK_END)
        self._write_bytes(output, b"\n")
        return True

    def _event_ids(self, output: Any) -> set[str]:
        output.seek(0)
        data = output.read()
        ids: set[str] = set()
        for line_number, raw in enumerate(data.splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                value = self._decode_line(raw, line_number)
            except BadEventLineError as exc:
                if self.bad_line_policy == "raise":
                    raise exc
                continue
            ids.add(value["event_id"])
        return ids

    def _decode_line(self, raw: bytes, line_number: int) -> dict[str, Any]:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BadEventLineError(
                BadEventLine(self.path, line_number, f"invalid UTF-8: {exc}", raw)
            ) from exc
        try:
            value = json.loads(text, parse_constant=_reject_json_constant)
        except ValueError as exc:
            raise BadEventLineError(
                BadEventLine(
                    self.path,
                    line_number,
                    f"invalid JSON: {getattr(exc, 'msg', str(exc))}",
                    raw,
                )
            ) from exc
        if not isinstance(value, dict):
            raise BadEventLineError(
                BadEventLine(self.path, line_number, "event must be a JSON object", raw)
            )
        event_id = value.get("event_id")
        if not isinstance(event_id, str) or not event_id.strip():
            raise BadEventLineError(
                BadEventLine(self.path, line_number, "event_id must be a non-empty string", raw)
            )
        return value


JsonlEventLog = EventLog
EventLogger = EventLog


__all__ = [
    "AlertEvent",
    "BadEventLine",
    "BadEventLineError",
    "EventLog",
    "EventLogError",
    "EventLogReadError",
    "EventLogWriteError",
    "EventLogger",
    "JsonlEventLog",
    "event_mapping",
    "serialize_event",
]
