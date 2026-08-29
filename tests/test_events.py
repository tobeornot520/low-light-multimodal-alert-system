import pytest

from lowlight_alert.events import (
    EventError,
    EventType,
    Point,
    Polygon,
    TrackLifecycle,
    TrackObservation,
    ZoneEvaluator,
    ZoneStateMachine,
    point_in_polygon,
)


def square() -> Polygon:
    return Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])


def test_polygon_includes_edges_and_vertices() -> None:
    shape = square()

    assert shape.contains((5, 5))
    assert shape.contains((0, 5))
    assert shape.contains((10, 10))
    assert not shape.contains((-0.01, 5))
    assert not point_in_polygon((11, 5), shape)


def test_polygon_accepts_points_keyword_and_concave_shape() -> None:
    shape = Polygon(
        points=[(0, 0), (4, 0), (4, 4), (2, 2), (0, 4)],
    )

    assert shape.points == shape.vertices
    assert shape.contains((1, 1))
    assert not shape.contains((2, 3))


def test_invalid_geometry_is_rejected() -> None:
    with pytest.raises(EventError, match="at least three"):
        Polygon([(0, 0), (1, 1)])
    with pytest.raises(EventError, match="non-zero area"):
        Polygon([(0, 0), (1, 0), (2, 0)])
    with pytest.raises(EventError, match="finite"):
        Point(float("nan"), 0)


def test_zone_state_machine_emits_enter_stay_and_leave_once() -> None:
    machine = ZoneStateMachine(
        {"restricted": square()},
        dwell_seconds=2,
        lost_tolerance_seconds=1,
    )

    assert [event.event_type for event in machine.update("track-1", (5, 5), 0)] == [
        EventType.ENTER
    ]
    assert machine.update("track-1", (5, 5), 1) == []
    stay = machine.update("track-1", (5, 5), 2)
    assert [event.event_type for event in stay] == [EventType.STAY]
    assert stay[0].duration == pytest.approx(2)
    assert machine.update("track-1", (5, 5), 3) == []

    leave = machine.update("track-1", (20, 5), 4)
    assert [event.event_type for event in leave] == [EventType.LEAVE]
    assert leave[0].duration == pytest.approx(4)
    assert machine.update("track-1", (20, 5), 5) == []


def test_short_missing_observation_does_not_duplicate_enter() -> None:
    machine = ZoneStateMachine(square(), dwell_seconds=10, lost_tolerance_seconds=1)
    machine.update("track-1", (5, 5), 0)

    assert machine.mark_missing("track-1", 0.5) == []
    state = machine.get_track("track-1")
    assert state is not None
    assert state.lifecycle is TrackLifecycle.LOST

    assert machine.update("track-1", (5, 5), 0.9) == []
    state = machine.get_track("track-1")
    assert state is not None
    assert state.lifecycle is TrackLifecycle.CONFIRMED
    assert [event.event_type for event in machine.events] == [EventType.ENTER]


def test_missing_timeout_emits_leave_and_new_episode_can_reenter() -> None:
    machine = ZoneStateMachine(square(), dwell_seconds=10, lost_tolerance_seconds=1)
    machine.update("track-1", (5, 5), 0)
    assert machine.mark_missing("track-1", 0.5) == []

    leave = machine.advance(1.5)
    assert [event.event_type for event in leave] == [EventType.LEAVE]
    assert leave[0].reason == "track_missing_timeout"
    assert machine.get_track("track-1") is None

    reentry = machine.update("track-1", (5, 5), 2)
    assert [event.event_type for event in reentry] == [EventType.ENTER]
    assert [event.event_type for event in machine.events] == [
        EventType.ENTER,
        EventType.LEAVE,
        EventType.ENTER,
    ]


def test_confirmation_frames_delay_first_enter() -> None:
    machine = ZoneStateMachine(square(), confirm_frames=2, dwell_seconds=10)

    assert machine.update("track-1", (5, 5), 0) == []
    state = machine.get_track("track-1")
    assert state is not None
    assert state.lifecycle is TrackLifecycle.TENTATIVE
    assert [event.event_type for event in machine.update("track-1", (5, 5), 1)] == [
        EventType.ENTER
    ]
    assert machine.get_track("track-1").lifecycle is TrackLifecycle.CONFIRMED


def test_track_observation_and_event_serialization() -> None:
    machine = ZoneStateMachine(
        ZoneEvaluator([("restricted", square())]), camera_id="cam-a", dwell_seconds=10
    )
    events = machine.observe(
        TrackObservation(
            track_id="track-1",
            timestamp=4,
            point=Point(5, 5),
            identity_state="registered",
            subject_id="person-a",
            confidence=0.9,
        )
    )

    payload = events[0].as_dict()
    assert payload["event_type"] == "enter"
    assert payload["camera_id"] == "cam-a"
    assert payload["identity_state"] == "registered"
    assert payload["confidence"] == pytest.approx(0.9)


def test_timestamp_must_not_move_backwards() -> None:
    machine = ZoneStateMachine(square())
    machine.update("track-1", (1, 1), 2)

    with pytest.raises(EventError, match="moved backwards"):
        machine.update("track-1", (1, 1), 1)


def test_boundary_between_ordered_zones_is_deterministic() -> None:
    machine = ZoneStateMachine(
        [
            ("first", Polygon([(0, 0), (5, 0), (5, 10), (0, 10)])),
            ("second", Polygon([(5, 0), (10, 0), (10, 10), (5, 10)])),
        ],
        dwell_seconds=10,
    )

    events = machine.update("track-1", (5, 5), 0)
    assert events[0].zone == "first"


def test_event_ids_are_unique_across_state_machine_instances() -> None:
    first = ZoneStateMachine(square())
    second = ZoneStateMachine(square())

    first_event = first.update("track-1", (5, 5), 0)[0]
    second_event = second.update("track-1", (5, 5), 0)[0]

    assert first_event.event_id != second_event.event_id
    assert first_event.event_id.startswith("event-")
    assert second_event.event_id.startswith("event-")


def test_empty_zone_evaluator_classifies_everything_as_outside() -> None:
    evaluator = ZoneEvaluator([])

    assert evaluator.names == ()
    assert evaluator.zone_at((0, 0)) is None


def test_reset_allows_track_id_reuse_without_cooldown_suppression() -> None:
    machine = ZoneStateMachine(square(), event_cooldown_seconds=60)

    assert machine.update("track-1", (1, 1), 0)[0].event_type is EventType.ENTER
    machine.reset("track-1")
    assert machine.update("track-1", (1, 1), 1)[0].event_type is EventType.ENTER
