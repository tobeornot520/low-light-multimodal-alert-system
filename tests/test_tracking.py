import pytest

from lowlight_alert.detector import FaceDetection
from lowlight_alert.recognizer import IdentityMatch, MatchState
from lowlight_alert.tracking import (
    SimpleTrackManager,
    TrackingError,
    TrackObservation,
    intersection_over_union,
)


def detection(box: tuple[int, int, int, int]) -> FaceDetection:
    x, y, _, _ = box
    return FaceDetection(
        box=box,
        landmarks=(
            (x + 10, y + 10),
            (x + 30, y + 10),
            (x + 20, y + 20),
            (x + 14, y + 32),
            (x + 26, y + 32),
        ),
        score=0.95,
    )


def identity(state: MatchState, subject_id: str | None = None) -> IdentityMatch:
    return IdentityMatch(state=state, subject_id=subject_id, display_name=subject_id)


def test_iou_handles_overlap_and_invalid_boxes() -> None:
    assert intersection_over_union((0, 0, 10, 10), (5, 5, 10, 10)) == pytest.approx(1 / 7)
    assert intersection_over_union((0, 0, 0, 10), (0, 0, 10, 10)) == 0


def test_manager_confirms_track_and_stabilizes_identity() -> None:
    manager = SimpleTrackManager(confirm_frames=2, max_missing_frames=1)
    first = manager.update(
        [TrackObservation(detection((0, 0, 40, 40)), identity(MatchState.REGISTERED, "a"))],
        0.0,
    )
    second = manager.update(
        [TrackObservation(detection((2, 0, 40, 40)), identity(MatchState.REGISTERED, "a"))],
        0.1,
    )

    assert first.active[0].track_id == "track-1"
    assert first.active[0].confirmed is False
    assert second.active[0].confirmed is True
    assert second.active[0].identity is not None
    assert second.active[0].identity.subject_id == "a"


def test_manager_tolerates_missing_then_emits_end() -> None:
    manager = SimpleTrackManager(confirm_frames=1, max_missing_frames=1)
    manager.update([TrackObservation(detection((0, 0, 40, 40)))], 0.0)
    missing = manager.update([], 0.1)
    ended = manager.update([], 0.2)

    assert missing.active[0].missing_frames == 1
    assert missing.ended == ()
    assert ended.active == ()
    assert [item.track_id for item in ended.ended] == ["track-1"]


def test_manager_does_not_switch_on_single_identity_outlier() -> None:
    manager = SimpleTrackManager(confirm_frames=1, max_missing_frames=1)
    manager.update(
        [TrackObservation(detection((0, 0, 40, 40)), identity(MatchState.REGISTERED, "a"))],
        0.0,
    )
    manager.update(
        [TrackObservation(detection((1, 0, 40, 40)), identity(MatchState.REGISTERED, "b"))],
        0.1,
    )
    current = manager.update(
        [TrackObservation(detection((2, 0, 40, 40)), identity(MatchState.REGISTERED, "a"))],
        0.2,
    )

    assert current.active[0].identity is not None
    assert current.active[0].identity.subject_id == "a"


def test_manager_requires_consecutive_quality_passes_for_confirmation() -> None:
    manager = SimpleTrackManager(confirm_frames=2)
    first = manager.update(
        [
            TrackObservation(
                detection((0, 0, 40, 40)),
                identity(MatchState.REGISTERED, "a"),
                quality_passed=False,
            )
        ],
        0.0,
    )
    second = manager.update(
        [TrackObservation(detection((1, 0, 40, 40)), identity(MatchState.REGISTERED, "a"))],
        0.1,
    )
    third = manager.update(
        [TrackObservation(detection((2, 0, 40, 40)), identity(MatchState.REGISTERED, "a"))],
        0.2,
    )

    assert first.active[0].confirmed is False
    assert second.active[0].confirmed is False
    assert third.active[0].confirmed is True


def test_snapshot_reports_current_quality_without_erasing_stable_identity() -> None:
    manager = SimpleTrackManager(confirm_frames=1)
    manager.update(
        [TrackObservation(detection((0, 0, 40, 40)), identity(MatchState.REGISTERED, "a"))],
        0.0,
    )

    current = manager.update(
        [TrackObservation(detection((1, 0, 40, 40)), quality_passed=False)],
        0.1,
    ).active[0]

    assert current.quality_passed is False
    assert current.identity is not None
    assert current.identity.subject_id == "a"


def test_manager_validates_settings() -> None:
    with pytest.raises(TrackingError, match="confirm_frames"):
        SimpleTrackManager(confirm_frames=0)
