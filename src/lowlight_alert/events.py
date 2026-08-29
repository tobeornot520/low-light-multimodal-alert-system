"""Geometry and zone-event primitives.

The event layer deliberately knows nothing about cameras or object association.  A
caller supplies a stable ``track_id`` and the observed centroid for each frame;
this module turns those observations into zone membership transitions.  Keeping
that boundary small makes the state machine useful for replay tests as well as
for a live preview.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from math import hypot, isfinite
from time import monotonic
from types import MappingProxyType
from typing import Any, TypeAlias
from uuid import uuid4


class EventError(ValueError):
    """Raised when geometry, observations, or state-machine settings are invalid."""


@dataclass(frozen=True, slots=True)
class Point:
    """A two-dimensional point used for zone membership tests.

    Coordinates are stored as floats so callers can pass either pixels or a
    normalized coordinate system.  The object is iterable, which keeps it
    convenient to pass to code expecting an ``(x, y)`` pair.
    """

    x: float
    y: float

    def __post_init__(self) -> None:
        try:
            x = float(self.x)
            y = float(self.y)
        except (TypeError, ValueError) as exc:
            raise EventError("point coordinates must be numeric") from exc
        if not isfinite(x) or not isfinite(y):
            raise EventError("point coordinates must be finite")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)

    @classmethod
    def from_value(cls, value: PointLike) -> Point:
        """Convert a ``Point`` or a two-item coordinate sequence."""
        if isinstance(value, cls):
            return value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != 2:
                raise EventError("point must contain exactly two coordinates")
            return cls(value[0], value[1])
        # Accept simple iterable pairs (for example a generator) without
        # accidentally accepting arbitrary strings.
        try:
            values = tuple(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise EventError("point must be a Point or a two-item coordinate pair") from exc
        if len(values) != 2:
            raise EventError("point must contain exactly two coordinates")
        return cls(values[0], values[1])

    def __iter__(self):
        yield self.x
        yield self.y

    def distance_to(self, other: PointLike) -> float:
        target = Point.from_value(other)
        return hypot(self.x - target.x, self.y - target.y)

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    to_tuple = as_tuple


PointLike: TypeAlias = Point | Sequence[float] | Iterable[float]


@dataclass(frozen=True, slots=True, init=False)
class Polygon:
    """A simple polygon.

    ``Polygon([(x1, y1), ...])`` and ``Polygon(vertices=(Point(...), ...))``
    are both accepted.  At least three non-collinear vertices are required.
    The polygon may be clockwise or counter-clockwise and may be concave, but
    self-intersecting rings are intentionally outside this small primitive's
    scope.
    """

    vertices: tuple[Point, ...]

    def __init__(
        self,
        vertices: Iterable[PointLike] | None = None,
        *,
        points: Iterable[PointLike] | None = None,
    ) -> None:
        if vertices is not None and points is not None:
            raise EventError("provide either vertices or points, not both")
        raw = points if points is not None else vertices
        if raw is None:
            raise EventError("polygon requires vertices")
        try:
            converted = tuple(Point.from_value(point) for point in raw)
        except TypeError as exc:
            raise EventError("polygon vertices must be iterable") from exc
        if len(converted) < 3:
            raise EventError("polygon requires at least three vertices")
        area2 = sum(
            first.x * second.y - second.x * first.y
            for first, second in zip(converted, converted[1:] + converted[:1], strict=True)
        )
        if abs(area2) <= 1e-12:
            raise EventError("polygon vertices must enclose a non-zero area")
        object.__setattr__(self, "vertices", converted)

    @property
    def points(self) -> tuple[Point, ...]:
        """Alias for ``vertices`` used by callers that prefer point terminology."""
        return self.vertices

    def __iter__(self):
        return iter(self.vertices)

    def __len__(self) -> int:
        return len(self.vertices)

    def contains(self, point: PointLike) -> bool:
        return point_in_polygon(point, self)

    def contains_point(self, point: PointLike) -> bool:
        return self.contains(point)

    @property
    def area(self) -> float:
        area2 = sum(
            first.x * second.y - second.x * first.y
            for first, second in zip(
                self.vertices,
                self.vertices[1:] + self.vertices[:1],
                strict=True,
            )
        )
        return abs(area2) / 2.0


def _coerce_polygon(polygon: Polygon | Iterable[PointLike]) -> Polygon:
    return polygon if isinstance(polygon, Polygon) else Polygon(polygon)


def _on_segment(point: Point, start: Point, end: Point) -> bool:
    cross = (end.x - start.x) * (point.y - start.y) - (end.y - start.y) * (
        point.x - start.x
    )
    # Scale the tolerance with the coordinates.  This keeps a point exactly on
    # a pixel edge inside while avoiding false positives for distant segments.
    scale = max(1.0, abs(end.x - start.x), abs(end.y - start.y), abs(point.x), abs(point.y))
    if abs(cross) > 1e-9 * scale:
        return False
    return (
        min(start.x, end.x) - 1e-9 <= point.x <= max(start.x, end.x) + 1e-9
        and min(start.y, end.y) - 1e-9 <= point.y <= max(start.y, end.y) + 1e-9
    )


def point_in_polygon(point: PointLike, polygon: Polygon | Iterable[PointLike]) -> bool:
    """Return whether *point* lies in *polygon*.

    The ray-casting test is preceded by an explicit edge test, so points on an
    edge or vertex are considered inside.  This behavior is important for a
    camera region: a detection whose centroid lands exactly on the configured
    boundary should not flicker between two states.
    """
    candidate = Point.from_value(point)
    shape = _coerce_polygon(polygon)
    inside = False
    for start, end in zip(shape.vertices, shape.vertices[1:] + shape.vertices[:1], strict=True):
        if _on_segment(candidate, start, end):
            return True
        crosses = (start.y > candidate.y) != (end.y > candidate.y)
        if not crosses:
            continue
        intersection_x = (end.x - start.x) * (candidate.y - start.y) / (end.y - start.y) + start.x
        if candidate.x < intersection_x:
            inside = not inside
    return inside


# Readable aliases used by a few geometry callers.  Keeping one implementation
# avoids subtle differences in boundary treatment.
is_point_in_polygon = point_in_polygon
point_inside_polygon = point_in_polygon


class TrackLifecycle(StrEnum):
    """Lifecycle of an externally-associated track."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    ENDED = "ended"

    # ``ACTIVE`` is a readable compatibility alias for integrations that do
    # not distinguish tentative and confirmed tracks.
    ACTIVE = "confirmed"


class EventType(StrEnum):
    """Zone transition types emitted by :class:`ZoneStateMachine`."""

    ENTER = "enter"
    STAY = "stay"
    LEAVE = "leave"

    # Common vocabulary aliases.  Enum aliases retain the canonical wire value.
    ENTRY = "enter"
    ENTERED = "enter"
    DWELL = "stay"
    STAYING = "stay"
    EXIT = "leave"
    EXITED = "leave"
    LEFT = "leave"


ZoneEventType = EventType
AlertEventType = EventType


@dataclass(frozen=True, slots=True)
class Zone:
    """A named polygon used by :class:`ZoneEvaluator`."""

    name: str
    polygon: Polygon

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise EventError("zone name cannot be empty")
        object.__setattr__(self, "name", name)
        if not isinstance(self.polygon, Polygon):
            object.__setattr__(self, "polygon", Polygon(self.polygon))  # type: ignore[arg-type]

    def contains(self, point: PointLike) -> bool:
        return self.polygon.contains(point)


class ZoneEvaluator:
    """Resolve a point to the first matching named zone.

    ``zones`` can be a mapping of names to polygons, an iterable of ``Zone``
    objects, or a single polygon (named ``default``).  Ordering is preserved;
    this gives overlapping zones deterministic behavior.
    """

    def __init__(
        self,
        zones: Mapping[str, Polygon | Iterable[PointLike]] | Iterable[Zone] | Polygon,
        *,
        default_name: str = "default",
    ) -> None:
        if isinstance(zones, Polygon):
            prepared = (Zone(default_name, zones),)
        elif isinstance(zones, Mapping):
            prepared = tuple(
                Zone(name, _coerce_polygon(polygon)) for name, polygon in zones.items()
            )
        else:
            try:
                values = tuple(zones)
            except TypeError as exc:
                raise EventError("zones must be a mapping, iterable of Zone, or Polygon") from exc
            if values and all(self._looks_like_point(value) for value in values):
                prepared = (Zone(default_name, Polygon(values)),)
            else:
                prepared = tuple(
                    value if isinstance(value, Zone) else self._zone_from_pair(value)
                    for value in values
                )
        names = [zone.name for zone in prepared]
        if len(set(names)) != len(names):
            raise EventError("zone names must be unique")
        self._zones = prepared
        self._by_name = MappingProxyType({zone.name: zone for zone in prepared})

    @staticmethod
    def _zone_from_pair(value: Any) -> Zone:
        try:
            pair = tuple(value)
        except TypeError:
            pair = ()
        if len(pair) == 2:
            return Zone(str(pair[0]), _coerce_polygon(pair[1]))
        raise EventError("zone iterable values must be Zone objects or (name, polygon) pairs")

    @staticmethod
    def _looks_like_point(value: Any) -> bool:
        try:
            pair = tuple(value)
        except TypeError:
            return False
        if len(pair) != 2:
            return False
        try:
            float(pair[0])
            float(pair[1])
        except (TypeError, ValueError):
            return False
        return True

    @property
    def zones(self) -> tuple[Zone, ...]:
        return self._zones

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(zone.name for zone in self._zones)

    def zone_at(self, point: PointLike | None) -> str | None:
        if point is None:
            return None
        candidate = Point.from_value(point)
        for zone in self._zones:
            if zone.contains(candidate):
                return zone.name
        return None

    def evaluate(self, point: PointLike | None) -> str | None:
        """Alias for :meth:`zone_at`."""
        return self.zone_at(point)

    zone_for = zone_at
    classify = zone_at

    def is_inside(self, point: PointLike, zone: str | None = None) -> bool:
        if zone is None:
            return self.zone_at(point) is not None
        try:
            selected = self._by_name[zone]
        except KeyError as exc:
            raise EventError(f"unknown zone: {zone}") from exc
        return selected.contains(point)

    def contains(self, zone: str, point: PointLike) -> bool:
        return self.is_inside(point, zone)


@dataclass(frozen=True, slots=True)
class TrackObservation:
    """One frame's observation for an already-associated track.

    ``point=None`` or ``detected=False`` represents a temporary missing
    observation.  The state machine uses ``identity_state`` and ``confidence``
    as event evidence but does not interpret identity labels itself.
    """

    track_id: str
    timestamp: float
    point: Point | None = None
    detected: bool = True
    identity_state: Any = None
    subject_id: str | None = None
    confidence: float | None = None
    quality_state: Any = None

    def __post_init__(self) -> None:
        track_id = str(self.track_id).strip()
        if not track_id:
            raise EventError("track_id cannot be empty")
        object.__setattr__(self, "track_id", track_id)
        timestamp = _timestamp(self.timestamp)
        object.__setattr__(self, "timestamp", timestamp)
        if self.point is not None and not isinstance(self.point, Point):
            object.__setattr__(self, "point", Point.from_value(self.point))  # type: ignore[arg-type]
        if not self.detected:
            object.__setattr__(self, "point", None)
        if self.confidence is not None:
            object.__setattr__(self, "confidence", _confidence(self.confidence))

    @property
    def centroid(self) -> Point | None:
        return self.point


@dataclass(slots=True)
class TrackState:
    """Mutable state retained for one externally-associated track."""

    track_id: str
    first_seen: float = 0.0
    last_seen: float = 0.0
    centroid: Point | None = None
    lifecycle: TrackLifecycle = TrackLifecycle.TENTATIVE
    zone: str | None = None
    zone_entered_at: float | None = None
    missing_since: float | None = None
    observation_count: int = 0
    consecutive_observations: int = 0
    identity_state: Any = None
    subject_id: str | None = None
    confidence: float | None = None
    quality_state: Any = None
    identity_history: list[Any] = field(default_factory=list)
    stay_emitted: bool = False
    last_stay_at: float | None = None
    generation: int = 0

    def __post_init__(self) -> None:
        self.track_id = str(self.track_id).strip()
        if not self.track_id:
            raise EventError("track_id cannot be empty")
        self.first_seen = _timestamp(self.first_seen)
        self.last_seen = _timestamp(self.last_seen)
        if self.centroid is not None and not isinstance(self.centroid, Point):
            self.centroid = Point.from_value(self.centroid)  # type: ignore[arg-type]

    @property
    def state(self) -> TrackLifecycle:
        """Alias for ``lifecycle``."""
        return self.lifecycle

    @state.setter
    def state(self, value: TrackLifecycle) -> None:
        self.lifecycle = TrackLifecycle(value)

    @property
    def zone_name(self) -> str | None:
        return self.zone

    @property
    def inside(self) -> bool:
        return self.zone is not None

    @property
    def current_zone(self) -> str | None:
        return self.zone

    @property
    def lost(self) -> bool:
        return self.lifecycle is TrackLifecycle.LOST

    @property
    def first_seen_at(self) -> float:
        return self.first_seen

    @property
    def last_seen_at(self) -> float:
        return self.last_seen


@dataclass(frozen=True, slots=True)
class AlertEvent:
    """Auditable result of a zone transition."""

    event_id: str = ""
    event_type: EventType = EventType.ENTER
    track_id: str = ""
    zone: str | None = None
    observed_at: float = 0.0
    first_seen: float = 0.0
    duration: float = 0.0
    identity_state: Any = None
    subject_id: str | None = None
    confidence: float | None = None
    quality_state: Any = None
    severity: int = 0
    evidence_flags: tuple[str, ...] = ()
    reason: str | None = None
    camera_id: str | None = None
    delivery_state: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", EventType(self.event_type))
        track_id = str(self.track_id).strip()
        if not track_id:
            raise EventError("event track_id cannot be empty")
        object.__setattr__(self, "track_id", track_id)
        object.__setattr__(self, "observed_at", _timestamp(self.observed_at))
        object.__setattr__(self, "first_seen", _timestamp(self.first_seen))
        try:
            duration = float(self.duration)
        except (TypeError, ValueError) as exc:
            raise EventError("event duration must be numeric") from exc
        if duration < 0 or not isfinite(duration):
            raise EventError("event duration must be finite and non-negative")
        object.__setattr__(self, "duration", duration)
        if self.confidence is not None:
            object.__setattr__(self, "confidence", _confidence(self.confidence))
        if (
            not isinstance(self.severity, int)
            or isinstance(self.severity, bool)
            or self.severity < 0
            or self.severity > 2
        ):
            raise EventError("event severity must be between 0 and 2")
        object.__setattr__(self, "evidence_flags", tuple(str(flag) for flag in self.evidence_flags))
        if self.event_id:
            object.__setattr__(self, "event_id", str(self.event_id))
        else:
            object.__setattr__(self, "event_id", str(uuid4()))

    @property
    def type(self) -> EventType:
        return self.event_type

    @property
    def event_kind(self) -> EventType:
        return self.event_type

    @property
    def timestamp(self) -> float:
        return self.observed_at

    @property
    def zone_name(self) -> str | None:
        return self.zone

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-friendly event fields without mutating the event."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "track_id": self.track_id,
            "zone": self.zone,
            "observed_at": self.observed_at,
            "first_seen": self.first_seen,
            "duration": self.duration,
            "identity_state": _enum_value(self.identity_state),
            "subject_id": self.subject_id,
            "confidence": self.confidence,
            "quality_state": _enum_value(self.quality_state),
            "severity": self.severity,
            "evidence_flags": list(self.evidence_flags),
            "reason": self.reason,
            "camera_id": self.camera_id,
            "delivery_state": self.delivery_state,
        }

    to_dict = as_dict


def _timestamp(value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EventError("timestamp must be numeric") from exc
    if not isfinite(result):
        raise EventError("timestamp must be finite")
    return result


def _confidence(value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EventError("confidence must be numeric") from exc
    if not 0.0 <= result <= 1.0 or not isfinite(result):
        raise EventError("confidence must be between 0 and 1")
    return result


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


SeverityResolver: TypeAlias = Callable[[EventType, TrackState], int]


class ZoneStateMachine:
    """Convert observations into de-duplicated enter/stay/leave events.

    The machine does not associate detections into tracks.  Callers should feed
    one stable ID per person/object and can either call :meth:`update` directly
    or construct :class:`TrackObservation` values.  A missing observation is
    tolerated for ``lost_tolerance_seconds``; if the track returns in that
    window its original zone episode remains active and no duplicate ``enter``
    event is emitted.
    """

    def __init__(
        self,
        zones: (
            ZoneEvaluator
            | Mapping[str, Polygon | Iterable[PointLike]]
            | Iterable[Zone]
            | Polygon
        ),
        *,
        dwell_seconds: float = 5.0,
        lost_tolerance_seconds: float = 1.0,
        confirm_frames: int = 1,
        stay_interval_seconds: float | None = None,
        event_cooldown_seconds: float = 0.0,
        camera_id: str | None = None,
        severity_resolver: SeverityResolver | None = None,
        clock: Callable[[], float] = monotonic,
        dwell_threshold_seconds: float | None = None,
        dwell_threshold: float | None = None,
        lost_tolerance: float | None = None,
        missing_tolerance_seconds: float | None = None,
        missing_grace_seconds: float | None = None,
        dedup_seconds: float | None = None,
        dedup_window_seconds: float | None = None,
        cooldown_seconds: float | None = None,
        confirmation_frames: int | None = None,
        min_confirmed_observations: int | None = None,
    ) -> None:
        self._evaluator = zones if isinstance(zones, ZoneEvaluator) else ZoneEvaluator(zones)
        if dwell_threshold_seconds is not None:
            dwell_seconds = dwell_threshold_seconds
        if dwell_threshold is not None:
            dwell_seconds = dwell_threshold
        if lost_tolerance is not None:
            lost_tolerance_seconds = lost_tolerance
        if missing_tolerance_seconds is not None:
            lost_tolerance_seconds = missing_tolerance_seconds
        if missing_grace_seconds is not None:
            lost_tolerance_seconds = missing_grace_seconds
        if dedup_seconds is not None:
            event_cooldown_seconds = dedup_seconds
        if dedup_window_seconds is not None:
            event_cooldown_seconds = dedup_window_seconds
        if cooldown_seconds is not None:
            event_cooldown_seconds = cooldown_seconds
        if confirmation_frames is not None:
            confirm_frames = confirmation_frames
        if min_confirmed_observations is not None:
            confirm_frames = min_confirmed_observations
        self._dwell_seconds = _nonnegative(dwell_seconds, "dwell_seconds")
        self._lost_tolerance_seconds = _nonnegative(
            lost_tolerance_seconds, "lost_tolerance_seconds"
        )
        if (
            not isinstance(confirm_frames, int)
            or isinstance(confirm_frames, bool)
            or confirm_frames <= 0
        ):
            raise EventError("confirm_frames must be a positive integer")
        self._confirm_frames = confirm_frames
        if stay_interval_seconds is not None:
            stay_interval_seconds = _nonnegative(stay_interval_seconds, "stay_interval_seconds")
            if stay_interval_seconds == 0:
                raise EventError("stay_interval_seconds must be positive when provided")
        self._stay_interval_seconds = stay_interval_seconds
        self._event_cooldown_seconds = _nonnegative(
            event_cooldown_seconds, "event_cooldown_seconds"
        )
        self._camera_id = camera_id
        self._severity_resolver = severity_resolver
        self._clock = clock
        self._tracks: dict[str, TrackState] = {}
        # The counter is useful for local ordering, while the session prefix
        # prevents a fresh process from reusing IDs already persisted in JSONL.
        self._event_session = uuid4().hex[:12]
        self._event_counter = 0
        self._event_history: list[AlertEvent] = []
        self._last_event_at: dict[tuple[str, EventType, str | None], float] = {}

    @property
    def evaluator(self) -> ZoneEvaluator:
        return self._evaluator

    @property
    def tracks(self) -> Mapping[str, TrackState]:
        return MappingProxyType(self._tracks)

    @property
    def events(self) -> tuple[AlertEvent, ...]:
        return tuple(self._event_history)

    @property
    def dwell_seconds(self) -> float:
        return self._dwell_seconds

    @property
    def lost_tolerance_seconds(self) -> float:
        return self._lost_tolerance_seconds

    def get_track(self, track_id: str) -> TrackState | None:
        return self._tracks.get(str(track_id).strip())

    state_for = get_track

    @property
    def active_tracks(self) -> tuple[TrackState, ...]:
        return tuple(self._tracks.values())

    def update(
        self,
        track_or_observation: str | TrackObservation,
        point: PointLike | None = None,
        timestamp: float | None = None,
        *,
        centroid: PointLike | None = None,
        now: float | None = None,
        detected: bool = True,
        identity_state: Any = None,
        subject_id: str | None = None,
        confidence: float | None = None,
        quality_state: Any = None,
    ) -> list[AlertEvent]:
        """Apply one observation and return events emitted at this timestamp.

        For convenience, the first argument may be a pre-built
        :class:`TrackObservation`; otherwise it is the track ID and the other
        keyword arguments describe the observation.
        """
        if isinstance(track_or_observation, TrackObservation):
            if (
                point is not None
                or timestamp is not None
                or centroid is not None
                or now is not None
            ):
                raise EventError("point/timestamp cannot accompany TrackObservation")
            observation = track_or_observation
        else:
            if point is not None and centroid is not None:
                raise EventError("provide either point or centroid, not both")
            if timestamp is not None and now is not None:
                raise EventError("provide either timestamp or now, not both")
            track_id = str(track_or_observation).strip()
            if not track_id:
                raise EventError("track_id cannot be empty")
            observation = TrackObservation(
                track_id=track_id,
                timestamp=self._clock() if timestamp is None and now is None else (
                    timestamp if timestamp is not None else now
                ),
                point=point if point is not None else centroid,
                detected=detected,
                identity_state=identity_state,
                subject_id=subject_id,
                confidence=confidence,
                quality_state=quality_state,
            )
        return self._apply(observation)

    observe = update
    process = update
    on_observation = update

    def update_many(
        self,
        observations: Iterable[TrackObservation],
        *,
        timestamp: float | None = None,
    ) -> list[AlertEvent]:
        """Apply a batch of observations in input order.

        ``timestamp`` fills in a common timestamp only for observations whose
        timestamp is already set to that value by the caller; it is accepted as
        a convenience for frame loops and does not mutate frozen observations.
        """
        rows = list(observations)
        if timestamp is not None:
            frame_time = _timestamp(timestamp)
            rows = [
                TrackObservation(
                    track_id=row.track_id,
                    timestamp=frame_time,
                    point=row.point,
                    detected=row.detected,
                    identity_state=row.identity_state,
                    subject_id=row.subject_id,
                    confidence=row.confidence,
                    quality_state=row.quality_state,
                )
                for row in rows
            ]
        emitted: list[AlertEvent] = []
        for row in rows:
            emitted.extend(self._apply(row))
        return emitted

    process_frame = update_many

    def mark_missing(self, track_id: str, timestamp: float | None = None) -> list[AlertEvent]:
        """Record a missing detection for a track (without changing its centroid)."""
        return self.update(
            str(track_id),
            point=None,
            timestamp=self._clock() if timestamp is None else timestamp,
            detected=False,
        )

    def end_track(
        self,
        track_id: str,
        timestamp: float | None = None,
        *,
        reason: str = "track_ended",
    ) -> list[AlertEvent]:
        """End one track immediately when the association layer retires it.

        ``mark_missing`` intentionally honors the grace period configured for
        temporary occlusion.  A tracker can nevertheless know that a track is
        gone (for example after its maximum missing-frame budget is exceeded),
        so the monitor needs a targeted termination operation that does not
        advance or expire unrelated tracks.
        """
        normalized_id = str(track_id).strip()
        if not normalized_id:
            raise EventError("track_id cannot be empty")
        state = self._tracks.get(normalized_id)
        if state is None:
            return []
        now = _timestamp(self._clock() if timestamp is None else timestamp)
        if now < state.last_seen:
            raise EventError(
                f"timestamp for track {normalized_id} moved backwards "
                f"from {state.last_seen} to {now}"
            )
        emitted: list[AlertEvent] = []
        if state.zone is not None:
            emitted.extend(self._emit_leave(state, now, reason))
        state.zone = None
        state.zone_entered_at = None
        state.lifecycle = TrackLifecycle.ENDED
        state.last_seen = max(state.last_seen, now)
        self._tracks.pop(normalized_id, None)
        return emitted

    terminate_track = end_track

    def advance(self, timestamp: float | None = None) -> list[AlertEvent]:
        """Expire tracks whose missing grace period has elapsed.

        Live loops should call this when a frame contains no detection for a
        previously known track.  It is also useful for deterministic replay.
        """
        now = _timestamp(self._clock() if timestamp is None else timestamp)
        emitted: list[AlertEvent] = []
        for track_id in tuple(self._tracks):
            state = self._tracks.get(track_id)
            if state is None or state.missing_since is None:
                continue
            if now - state.missing_since >= self._lost_tolerance_seconds:
                emitted.extend(self._end_missing(state, now))
        return emitted

    tick = advance

    def flush(self, timestamp: float | None = None) -> list[AlertEvent]:
        """End all active tracks and emit pending leave events immediately."""
        now = _timestamp(self._clock() if timestamp is None else timestamp)
        emitted: list[AlertEvent] = []
        for state in tuple(self._tracks.values()):
            if state.zone is not None:
                emitted.extend(self._emit_leave(state, now, "machine_flush"))
            state.lifecycle = TrackLifecycle.ENDED
        self._tracks.clear()
        return emitted

    def reset(self, track_id: str | None = None) -> None:
        """Forget one track or all tracks without generating events."""
        if track_id is None:
            self._tracks.clear()
            self._last_event_at.clear()
        else:
            normalized_id = str(track_id).strip()
            self._tracks.pop(normalized_id, None)
            for key in tuple(self._last_event_at):
                if key[0] == normalized_id:
                    del self._last_event_at[key]

    def drain_events(self) -> list[AlertEvent]:
        """Return and clear the accumulated event history."""
        result = list(self._event_history)
        self._event_history.clear()
        return result

    def _apply(self, observation: TrackObservation) -> list[AlertEvent]:
        state = self._tracks.get(observation.track_id)
        if state is not None and observation.timestamp < state.last_seen:
            raise EventError(
                f"timestamp for track {observation.track_id} moved backwards "
                f"from {state.last_seen} to {observation.timestamp}"
            )

        emitted: list[AlertEvent] = []
        if state is None:
            if not observation.detected:
                return []
            state = TrackState(
                track_id=observation.track_id,
                first_seen=observation.timestamp,
                last_seen=observation.timestamp,
                centroid=observation.point,
                lifecycle=(
                    TrackLifecycle.CONFIRMED
                    if self._confirm_frames == 1
                    else TrackLifecycle.TENTATIVE
                ),
                observation_count=1,
                consecutive_observations=1,
                identity_state=observation.identity_state,
                subject_id=observation.subject_id,
                confidence=observation.confidence,
                quality_state=observation.quality_state,
                identity_history=(
                    [observation.identity_state]
                    if observation.identity_state is not None
                    else []
                ),
            )
            self._tracks[observation.track_id] = state
            if state.lifecycle is TrackLifecycle.CONFIRMED:
                state.zone = self._evaluator.zone_at(observation.point)
                if state.zone is not None:
                    state.zone_entered_at = observation.timestamp
                    emitted.extend(self._emit_enter(state, observation.timestamp))
            return emitted

        if not observation.detected or observation.point is None:
            return self._apply_missing(state, observation.timestamp)

        # A track that exceeded the grace period is a new episode, even if the
        # detector reuses its numeric ID.  Emit the old leave first so event
        # consumers see a well-ordered lifecycle.
        if state.missing_since is not None:
            if observation.timestamp - state.missing_since >= self._lost_tolerance_seconds:
                next_generation = state.generation + 1
                emitted.extend(self._end_missing(state, observation.timestamp))
                self._tracks.pop(state.track_id, None)
                fresh = TrackObservation(
                    track_id=observation.track_id,
                    timestamp=observation.timestamp,
                    point=observation.point,
                    detected=True,
                    identity_state=observation.identity_state,
                    subject_id=observation.subject_id,
                    confidence=observation.confidence,
                    quality_state=observation.quality_state,
                )
                emitted.extend(self._apply(fresh))
                replacement = self._tracks.get(observation.track_id)
                if replacement is not None:
                    replacement.generation = next_generation
                return emitted
            state.missing_since = None
            state.lifecycle = (
                TrackLifecycle.CONFIRMED
                if state.observation_count >= self._confirm_frames
                else TrackLifecycle.TENTATIVE
            )
            state.consecutive_observations = 0

        state.last_seen = observation.timestamp
        state.centroid = observation.point
        state.observation_count += 1
        state.consecutive_observations += 1
        # A frame can be trackable while its face quality or identity result is
        # temporarily unavailable. Preserve the last usable evidence in that
        # case instead of erasing it from a pending event.
        if observation.identity_state is not None:
            state.identity_state = observation.identity_state
        if observation.subject_id is not None:
            state.subject_id = observation.subject_id
        if observation.confidence is not None:
            state.confidence = observation.confidence
        if observation.quality_state is not None:
            state.quality_state = observation.quality_state
        if observation.identity_state is not None:
            state.identity_history.append(observation.identity_state)

        if (
            state.lifecycle is TrackLifecycle.TENTATIVE
            and state.consecutive_observations >= self._confirm_frames
        ):
            state.lifecycle = TrackLifecycle.CONFIRMED
            state.zone = self._evaluator.zone_at(observation.point)
            if state.zone is not None:
                state.zone_entered_at = observation.timestamp
                emitted.extend(self._emit_enter(state, observation.timestamp))
            return emitted
        if state.lifecycle is not TrackLifecycle.CONFIRMED:
            return emitted

        current_zone = self._evaluator.zone_at(observation.point)
        if current_zone != state.zone:
            if state.zone is not None:
                emitted.extend(self._emit_leave(state, observation.timestamp, "left_zone"))
            state.zone = current_zone
            state.stay_emitted = False
            state.last_stay_at = None
            if current_zone is not None:
                state.zone_entered_at = observation.timestamp
                emitted.extend(self._emit_enter(state, observation.timestamp))
            else:
                state.zone_entered_at = None
            return emitted

        if current_zone is not None:
            emitted.extend(self._maybe_emit_stay(state, observation.timestamp))
        return emitted

    def _apply_missing(self, state: TrackState, timestamp: float) -> list[AlertEvent]:
        if state.missing_since is None:
            state.missing_since = timestamp
            state.lifecycle = TrackLifecycle.LOST
            state.consecutive_observations = 0
        if timestamp - state.missing_since >= self._lost_tolerance_seconds:
            return self._end_missing(state, timestamp)
        return []

    def _end_missing(self, state: TrackState, timestamp: float) -> list[AlertEvent]:
        emitted: list[AlertEvent] = []
        if state.zone is not None:
            emitted.extend(self._emit_leave(state, timestamp, "track_missing_timeout"))
        state.zone = None
        state.zone_entered_at = None
        state.lifecycle = TrackLifecycle.ENDED
        state.last_seen = max(state.last_seen, timestamp)
        self._tracks.pop(state.track_id, None)
        return emitted

    def _maybe_emit_stay(self, state: TrackState, timestamp: float) -> list[AlertEvent]:
        if state.zone_entered_at is None:
            return []
        elapsed = timestamp - state.zone_entered_at
        if elapsed < self._dwell_seconds:
            return []
        if not state.stay_emitted:
            state.stay_emitted = True
            state.last_stay_at = timestamp
            return self._emit(state, EventType.STAY, timestamp, "dwell_threshold_reached", elapsed)
        if self._stay_interval_seconds is None or state.last_stay_at is None:
            return []
        if timestamp - state.last_stay_at < self._stay_interval_seconds:
            return []
        state.last_stay_at = timestamp
        return self._emit(state, EventType.STAY, timestamp, "dwell_interval_reached", elapsed)

    def _emit_enter(self, state: TrackState, timestamp: float) -> list[AlertEvent]:
        return self._emit(state, EventType.ENTER, timestamp, "entered_zone", 0.0)

    def _emit_leave(self, state: TrackState, timestamp: float, reason: str) -> list[AlertEvent]:
        duration = (
            max(0.0, timestamp - state.zone_entered_at)
            if state.zone_entered_at is not None
            else 0.0
        )
        return self._emit(state, EventType.LEAVE, timestamp, reason, duration)

    def _emit(
        self,
        state: TrackState,
        event_type: EventType,
        timestamp: float,
        reason: str,
        duration: float,
    ) -> list[AlertEvent]:
        zone = state.zone
        key = (state.track_id, event_type, zone)
        previous = self._last_event_at.get(key)
        if previous is not None and timestamp - previous < self._event_cooldown_seconds:
            return []
        self._last_event_at[key] = timestamp
        self._event_counter += 1
        severity = self._severity(event_type, state)
        flags = ["inside_zone"] if zone is not None else []
        if event_type is EventType.STAY:
            flags.append("dwell_threshold")
        if reason in {"track_missing_timeout", "tracker_ended"}:
            flags.append("track_lost")
        event = AlertEvent(
            event_id=f"event-{self._event_session}-{self._event_counter:06d}",
            event_type=event_type,
            track_id=state.track_id,
            zone=zone,
            observed_at=timestamp,
            first_seen=state.first_seen,
            duration=duration,
            identity_state=state.identity_state,
            subject_id=state.subject_id,
            confidence=state.confidence,
            quality_state=state.quality_state,
            severity=severity,
            evidence_flags=tuple(flags),
            reason=reason,
            camera_id=self._camera_id,
        )
        self._event_history.append(event)
        return [event]

    def _severity(self, event_type: EventType, state: TrackState) -> int:
        if self._severity_resolver is not None:
            value = self._severity_resolver(event_type, state)
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 2:
                raise EventError("severity_resolver must return an integer between 0 and 2")
            return value
        if event_type is EventType.STAY:
            return 2
        if event_type is EventType.ENTER:
            return 1
        return 0


def _nonnegative(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EventError(f"{name} must be numeric") from exc
    if result < 0 or not isfinite(result):
        raise EventError(f"{name} must be finite and non-negative")
    return result


# Domain vocabulary aliases keep integrations decoupled from whether an event
# is consumed as a generic alert or as a zone-specific transition.
ZoneEvent = AlertEvent
TrackEvent = AlertEvent


__all__ = [
    "AlertEvent",
    "AlertEventType",
    "EventError",
    "EventType",
    "Point",
    "PointLike",
    "Polygon",
    "TrackLifecycle",
    "TrackObservation",
    "TrackState",
    "Zone",
    "ZoneEvent",
    "ZoneEvaluator",
    "ZoneEventType",
    "ZoneStateMachine",
    "TrackEvent",
    "is_point_in_polygon",
    "point_inside_polygon",
    "point_in_polygon",
]
