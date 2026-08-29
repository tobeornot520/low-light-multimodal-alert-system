from datetime import UTC, datetime
from pathlib import Path

import pytest

from lowlight_alert.event_log import (
    AlertEvent,
    BadEventLineError,
    EventLog,
    EventLogError,
)


def test_append_accepts_alert_event_and_mapping_and_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "events.jsonl"
    log = EventLog(path)

    event = AlertEvent(
        event_id="evt-1",
        event_type="zone_enter",
        track_id="track-1",
        observed_at=1720000000.0,
        severity=1,
        evidence_flags=("quality_pass", "multi_frame_confirmed"),
    )
    assert log.append(event) is True
    assert log.append(event) is False
    assert log.append({"event_id": "evt-2", "event_type": "zone_leave"}) is True

    rows = log.read_events()
    assert [row["event_id"] for row in rows] == ["evt-1", "evt-2"]
    assert rows[0]["evidence_flags"] == ["quality_pass", "multi_frame_confirmed"]
    assert path.read_bytes().count(b"\n") == 2


def test_idempotency_survives_new_log_instance(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    first = EventLog(path)
    second = EventLog(path)
    assert first.append({"event_id": "same", "value": 1})
    assert second.append({"event_id": "same", "value": 2}) is False
    assert second.read_events() == [{"event_id": "same", "value": 1}]


def test_serializer_handles_datetime_enum_and_rejects_bad_values(tmp_path: Path) -> None:
    from enum import StrEnum

    class Kind(StrEnum):
        ENTER = "zone_enter"

    log = EventLog(tmp_path / "events.jsonl")
    assert log.append(
        {
            "event_id": "typed",
            "event_type": Kind.ENTER,
            "observed_at": datetime(2026, 8, 21, tzinfo=UTC),
        }
    )
    row = log.read_events()[0]
    assert row["event_type"] == "zone_enter"
    assert row["observed_at"] == "2026-08-21T00:00:00+00:00"

    with pytest.raises(EventLogError, match="event_id"):
        log.append({"event_type": "zone_enter"})
    with pytest.raises(EventLogError, match="JSON serializable"):
        log.append({"event_id": "bad", "value": object()})


def test_read_skips_bad_lines_and_reports_them(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(
        b'{"event_id":"ok-1","event_type":"enter"}\n'
        b"not-json\n"
        b"[]\n"
        b'{"event_id":"non-standard","score":NaN}\n'
        b'{"event_id":"ok-2","event_type":"leave"}\n'
    )
    log = EventLog(path)
    seen = []
    rows = log.read_events(on_bad_line=seen.append)

    assert [row["event_id"] for row in rows] == ["ok-1", "ok-2"]
    assert [item.line_number for item in seen] == [2, 3, 4]
    assert len(log.last_bad_lines) == 3

    with pytest.raises(BadEventLineError, match=r"line 2"):
        log.read_events(strict=True)


def test_append_repairs_incomplete_tail_before_writing(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"event_id":"complete","event_type":"enter"}')
    log = EventLog(path)

    # A complete JSON object missing only its newline is preserved.
    assert log.append({"event_id": "next", "event_type": "leave"})
    assert [row["event_id"] for row in log.read_events()] == ["complete", "next"]
    assert path.read_bytes().endswith(b"\n")

    # A genuinely partial final record is discarded on the following append.
    with path.open("ab") as output:
        output.write(b'{"event_id":"crashed"')
    assert log.append({"event_id": "after-crash", "event_type": "enter"})
    assert [row["event_id"] for row in log.read_events()] == [
        "complete",
        "next",
        "after-crash",
    ]


def test_strict_append_does_not_silently_discard_bad_tail(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"event_id":"ok"}\n{"event_id":"partial"')
    log = EventLog(path, strict=True)

    with pytest.raises(BadEventLineError):
        log.append({"event_id": "new"})
    assert path.read_bytes().endswith(b'{"event_id":"partial"')


def test_append_many_deduplicates_batch_and_existing_rows(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.jsonl")
    assert log.append_many(
        [
            {"event_id": "one", "event_type": "enter"},
            {"event_id": "one", "event_type": "duplicate"},
            {"event_id": "two", "event_type": "leave"},
        ]
    ) == 2
    assert log.append_many([{"event_id": "one"}, {"event_id": "three"}]) == 1
    assert [row["event_id"] for row in log.read_events()] == ["one", "two", "three"]


def test_read_missing_file_is_empty_and_contains_validates_id(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "missing.jsonl")
    assert log.read_events() == []
    assert log.contains("not-there") is False
    with pytest.raises(EventLogError, match="event_id"):
        log.contains("")
