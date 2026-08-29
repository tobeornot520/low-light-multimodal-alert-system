from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from itertools import count
from math import isfinite

from lowlight_alert.detector import BoundingBox, FaceDetection
from lowlight_alert.recognizer import IdentityMatch, MatchState


class TrackingError(ValueError):
    """Raised when tracking input or configuration is invalid."""


@dataclass(frozen=True)
class TrackObservation:
    detection: FaceDetection
    identity: IdentityMatch | None = None
    quality_passed: bool = True


@dataclass(frozen=True)
class TrackSnapshot:
    track_id: str
    detection: FaceDetection
    identity: IdentityMatch | None
    quality_passed: bool
    confirmed: bool
    missing_frames: int
    first_seen: float
    last_seen: float


@dataclass(frozen=True)
class TrackUpdate:
    active: tuple[TrackSnapshot, ...]
    ended: tuple[TrackSnapshot, ...]


@dataclass
class _Track:
    track_id: str
    detection: FaceDetection
    first_seen: float
    last_seen: float
    hits: int = 1
    quality_hits: int = 0
    missing_frames: int = 0
    confirmed: bool = False
    history: deque[IdentityMatch] = None  # type: ignore[assignment]
    current_identity: IdentityMatch | None = None
    quality_passed: bool = True

    def snapshot(self) -> TrackSnapshot:
        return TrackSnapshot(
            track_id=self.track_id,
            detection=self.detection,
            identity=self.current_identity,
            quality_passed=self.quality_passed,
            confirmed=self.confirmed,
            missing_frames=self.missing_frames,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
        )


def intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    first_right = first_x + max(0, first_width)
    first_bottom = first_y + max(0, first_height)
    second_right = second_x + max(0, second_width)
    second_bottom = second_y + max(0, second_height)
    left = max(first_x, second_x)
    top = max(first_y, second_y)
    right = min(first_right, second_right)
    bottom = min(first_bottom, second_bottom)
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = max(0, first_right - first_x) * max(0, first_bottom - first_y)
    second_area = max(0, second_right - second_x) * max(0, second_bottom - second_y)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


class SimpleTrackManager:
    """Associate one-camera detections and stabilize identity labels."""

    def __init__(
        self,
        iou_threshold: float = 0.2,
        confirm_frames: int = 3,
        max_missing_frames: int = 5,
    ) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise TrackingError("iou_threshold must be between 0 and 1")
        if confirm_frames <= 0:
            raise TrackingError("confirm_frames must be positive")
        if max_missing_frames < 0:
            raise TrackingError("max_missing_frames cannot be negative")
        self.iou_threshold = iou_threshold
        self.confirm_frames = confirm_frames
        self.max_missing_frames = max_missing_frames
        self._tracks: dict[str, _Track] = {}
        self._next_id = count(1)

    def update(self, observations: list[TrackObservation], timestamp: float) -> TrackUpdate:
        if not isinstance(timestamp, (int, float)) or not isfinite(float(timestamp)):
            raise TrackingError("timestamp must be numeric")
        if any(not isinstance(item, TrackObservation) for item in observations):
            raise TrackingError("observations must contain TrackObservation values")

        unmatched_tracks = set(self._tracks)
        unmatched_observations = set(range(len(observations)))
        pairs: list[tuple[float, str, int]] = []
        for track_id, track in self._tracks.items():
            for index, observation in enumerate(observations):
                score = intersection_over_union(track.detection.box, observation.detection.box)
                if score >= self.iou_threshold:
                    pairs.append((score, track_id, index))
        pairs.sort(reverse=True)
        for _, track_id, index in pairs:
            if track_id not in unmatched_tracks or index not in unmatched_observations:
                continue
            self._apply_observation(self._tracks[track_id], observations[index], timestamp)
            unmatched_tracks.remove(track_id)
            unmatched_observations.remove(index)

        ended: list[TrackSnapshot] = []
        for track_id in list(unmatched_tracks):
            track = self._tracks[track_id]
            track.missing_frames += 1
            if not track.confirmed:
                track.quality_hits = 0
            if track.missing_frames > self.max_missing_frames:
                ended.append(track.snapshot())
                del self._tracks[track_id]

        for index in sorted(unmatched_observations):
            observation = observations[index]
            track_id = f"track-{next(self._next_id)}"
            track = _Track(
                track_id=track_id,
                detection=observation.detection,
                first_seen=float(timestamp),
                last_seen=float(timestamp),
                history=deque(maxlen=self.confirm_frames),
            )
            self._tracks[track_id] = track
            self._apply_observation(track, observation, timestamp, new=True)

        active = tuple(self._tracks[track_id].snapshot() for track_id in sorted(self._tracks))
        ordered_ended = tuple(sorted(ended, key=lambda item: item.track_id))
        return TrackUpdate(active=active, ended=ordered_ended)

    def reset(self, timestamp: float) -> TrackUpdate:
        ended = tuple(
            sorted(
                (track.snapshot() for track in self._tracks.values()),
                key=lambda item: item.track_id,
            )
        )
        self._tracks.clear()
        return TrackUpdate(active=(), ended=ended)

    def _apply_observation(
        self,
        track: _Track,
        observation: TrackObservation,
        timestamp: float,
        new: bool = False,
    ) -> None:
        track.detection = observation.detection
        track.quality_passed = observation.quality_passed
        track.last_seen = float(timestamp)
        track.missing_frames = 0
        if not new:
            track.hits += 1
        if observation.quality_passed:
            track.quality_hits += 1
        else:
            track.quality_hits = 0
        if observation.quality_passed and observation.identity is not None:
            track.history.append(observation.identity)
            track.current_identity = self._stable_identity(track.history)
        if not track.confirmed and track.quality_hits >= self.confirm_frames:
            track.confirmed = True

    @staticmethod
    def _stable_identity(history: deque[IdentityMatch]) -> IdentityMatch | None:
        if not history:
            return None
        keys = [
            (item.state, item.subject_id if item.state is MatchState.REGISTERED else None)
            for item in history
        ]
        (state, subject_id), count_value = Counter(keys).most_common(1)[0]
        if count_value * 2 <= len(history):
            return history[-1]
        for item in reversed(history):
            if item.state is state and (
                state is not MatchState.REGISTERED or item.subject_id == subject_id
            ):
                return item
        return history[-1]
