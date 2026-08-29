from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from time import monotonic

import cv2
import numpy as np

from lowlight_alert.config import (
    CameraSettings,
    DetectionSettings,
    EventSettings,
    PreviewSettings,
    QualitySettings,
    RecognitionSettings,
    ZoneSettings,
)
from lowlight_alert.detector import YuNetFaceDetector, draw_detection
from lowlight_alert.event_log import EventLog
from lowlight_alert.events import (
    AlertEvent,
    EventType,
    Point,
    Polygon,
    Zone,
    ZoneStateMachine,
)
from lowlight_alert.events import (
    TrackObservation as EventObservation,
)
from lowlight_alert.preview import PreviewFrameResult, run_preview
from lowlight_alert.quality import FaceQuality, FaceQualityEvaluator, QualityIssue
from lowlight_alert.recognizer import (
    IdentityMatch,
    MatchState,
    SFaceFeatureExtractor,
    TemplateMatcher,
)
from lowlight_alert.template_store import TemplateStore
from lowlight_alert.tracking import SimpleTrackManager, TrackObservation, TrackSnapshot

_ISSUE_LABELS = {
    QualityIssue.INVALID_CROP: "crop",
    QualityIssue.FACE_TOO_SMALL: "small",
    QualityIssue.TOO_DARK: "dark",
    QualityIssue.TOO_BRIGHT: "bright",
    QualityIssue.BLURRY: "blur",
    QualityIssue.EXTREME_POSE: "pose",
}


class MonitorError(RuntimeError):
    """Raised when the local monitoring pipeline cannot be configured."""


@dataclass(frozen=True)
class MonitorFrameStats:
    """Structured observations produced for one processed frame."""

    detections: int = 0
    quality_passed: int = 0
    quality_rejected: int = 0
    registered: int = 0
    unknown: int = 0
    uncertain: int = 0
    active_tracks: int = 0
    confirmed_tracks: int = 0
    events: tuple[EventType, ...] = ()


def _build_matcher(templates_dir: Path, settings: RecognitionSettings) -> TemplateMatcher:
    return TemplateMatcher.from_store(
        TemplateStore(templates_dir),
        settings.accept_threshold,
        settings.reject_threshold,
        settings.min_margin,
    )


def _build_zones(settings: tuple[ZoneSettings, ...]) -> tuple[Zone, ...]:
    return tuple(
        Zone(
            item.name,
            Polygon(item.polygon),
        )
        for item in settings
    )


def _centroid(snapshot: TrackSnapshot, frame: np.ndarray) -> Point:
    x, y, width, height = snapshot.detection.box
    frame_height, frame_width = frame.shape[:2]
    return Point(
        (x + width / 2) / max(1, frame_width),
        (y + height / 2) / max(1, frame_height),
    )


def _identity_text(identity: IdentityMatch | None) -> str:
    if identity is None:
        return "Unusable"
    score = "n/a" if identity.similarity is None else f"{identity.similarity:.2f}"
    if identity.state is MatchState.REGISTERED:
        return f"{identity.display_name or identity.subject_id or 'Registered'} {score}"
    return f"{identity.state.value.title()} {score}"


def _quality_text(quality: FaceQuality) -> str:
    reasons = ",".join(_ISSUE_LABELS[issue] for issue in quality.issues)
    return f"Unusable: {reasons}"


def _draw_zones(frame: np.ndarray, zones: tuple[Zone, ...]) -> None:
    height, width = frame.shape[:2]
    for zone in zones:
        points = np.asarray(
            [(round(point.x * width), round(point.y * height)) for point in zone.polygon],
            dtype=np.int32,
        )
        cv2.polylines(frame, [points], True, (255, 180, 0), 2)
        if len(points):
            cv2.putText(
                frame,
                zone.name,
                tuple(points[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 180, 0),
                1,
            )


class MonitorSession:
    """Glue camera observations to tracking, zone events, and local logging."""

    def __init__(
        self,
        detector: YuNetFaceDetector,
        evaluator: FaceQualityEvaluator,
        extractor: SFaceFeatureExtractor,
        matcher: TemplateMatcher,
        tracker: SimpleTrackManager,
        zones: tuple[Zone, ...],
        event_machine: ZoneStateMachine,
        event_log: EventLog,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.detector = detector
        self.evaluator = evaluator
        self.extractor = extractor
        self.matcher = matcher
        self.tracker = tracker
        self.zones = zones
        self.event_machine = event_machine
        self.event_log = event_log
        self.clock = clock
        self.last_events: tuple[AlertEvent, ...] = ()
        self.last_frame_stats = MonitorFrameStats()
        self._last_timestamp: float | None = None

    def process_frame(self, frame: np.ndarray) -> PreviewFrameResult:
        """Process a live frame using the session clock."""
        return self.process_frame_at(frame, float(self.clock()))

    def process_frame_at(
        self,
        frame: np.ndarray,
        timestamp: float,
        *,
        render: bool = True,
    ) -> PreviewFrameResult:
        """Process a frame at an explicit media timestamp.

        Offline replay must use source-video time rather than processing wall
        time, otherwise dwell and missing-track thresholds depend on machine
        speed.  ``render=False`` skips preview overlays when benchmarking.
        """
        now = self._validate_timestamp(timestamp)
        detections = self.detector.detect(frame)
        observations: list[TrackObservation] = []
        labels: dict[tuple[int, int, int, int], tuple[str, tuple[int, int, int]]] = {}
        quality_passed = 0
        identity_counts = {state: 0 for state in MatchState}
        for detection in detections:
            quality = self.evaluator.evaluate(frame, detection)
            identity: IdentityMatch | None = None
            if quality.passed:
                quality_passed += 1
                feature = self.extractor.extract(frame, detection)
                identity = self.matcher.match(feature)
                identity_counts[identity.state] += 1
                label = _identity_text(identity)
                color = {
                    MatchState.REGISTERED: (40, 220, 80),
                    MatchState.UNKNOWN: (40, 80, 235),
                    MatchState.UNCERTAIN: (0, 210, 255),
                }[identity.state]
            else:
                label = _quality_text(quality)
                color = (0, 180, 255)
            labels[detection.box] = (label, color)
            observations.append(
                TrackObservation(
                    detection=detection,
                    identity=identity,
                    quality_passed=quality.passed,
                )
            )

        update = self.tracker.update(observations, now)
        emitted: list[AlertEvent] = []
        for snapshot in update.active:
            if not snapshot.confirmed:
                continue
            identity = snapshot.identity
            event_observation = EventObservation(
                track_id=snapshot.track_id,
                timestamp=now,
                point=_centroid(snapshot, frame),
                detected=snapshot.missing_frames == 0,
                identity_state=None if identity is None else identity.state,
                subject_id=None if identity is None else identity.subject_id,
                confidence=None if identity is None else identity.similarity,
                quality_state="pass" if snapshot.quality_passed else "unusable",
            )
            if snapshot.missing_frames:
                emitted.extend(self.event_machine.mark_missing(snapshot.track_id, now))
            else:
                emitted.extend(self.event_machine.update(event_observation))

        for snapshot in update.ended:
            emitted.extend(
                self.event_machine.end_track(
                    snapshot.track_id,
                    now,
                    reason="tracker_ended",
                )
            )
        # Active tracks that are temporarily missing are expired against the
        # real frame timestamp.  Advancing by ``now + tolerance`` here would
        # turn the first dropped frame into an immediate leave event.
        emitted.extend(self.event_machine.advance(now))
        if emitted:
            self.event_log.append_many(emitted)
        self.last_events = tuple(emitted)
        self._last_timestamp = now
        self.last_frame_stats = MonitorFrameStats(
            detections=len(detections),
            quality_passed=quality_passed,
            quality_rejected=len(detections) - quality_passed,
            registered=identity_counts[MatchState.REGISTERED],
            unknown=identity_counts[MatchState.UNKNOWN],
            uncertain=identity_counts[MatchState.UNCERTAIN],
            active_tracks=len(update.active),
            confirmed_tracks=sum(snapshot.confirmed for snapshot in update.active),
            events=tuple(event.event_type for event in emitted),
        )

        if render:
            _draw_zones(frame, self.zones)
            for snapshot in update.active:
                if snapshot.missing_frames:
                    continue
                label, color = labels.get(
                    snapshot.detection.box,
                    (_identity_text(snapshot.identity), (255, 255, 255)),
                )
                if snapshot.confirmed and snapshot.track_id in self.event_machine.tracks:
                    state = self.event_machine.tracks[snapshot.track_id]
                    if state.zone is not None:
                        label = f"{label} | {state.zone}"
                draw_detection(
                    frame,
                    snapshot.detection,
                    f"{snapshot.track_id}: {label}",
                    color,
                )

        event_text = (
            "none"
            if not emitted
            else ",".join(event.event_type.value for event in emitted)
        )
        return PreviewFrameResult(
            (
                f"Faces: {len(detections)} | Tracks: {len(update.active)} "
                f"| Templates: {self.matcher.subject_count}",
                f"Events: {event_text}",
            )
        )

    def finish(self) -> tuple[AlertEvent, ...]:
        """Finish a live session using the session clock."""
        return self.finish_at(float(self.clock()))

    def finish_at(self, timestamp: float) -> tuple[AlertEvent, ...]:
        """Finish a session at an explicit media timestamp."""
        now = self._validate_timestamp(timestamp)
        self.tracker.reset(now)
        emitted = self.event_machine.flush(now)
        if emitted:
            self.event_log.append_many(emitted)
        self.last_events = tuple(emitted)
        self._last_timestamp = now
        return tuple(emitted)

    def _validate_timestamp(self, timestamp: float) -> float:
        try:
            value = float(timestamp)
        except (TypeError, ValueError) as exc:
            raise MonitorError("monitor timestamp must be numeric") from exc
        if not isfinite(value):
            raise MonitorError("monitor timestamp must be finite")
        if self._last_timestamp is not None and value < self._last_timestamp:
            raise MonitorError(
                f"monitor timestamp moved backwards from {self._last_timestamp} to {value}"
            )
        return value


def create_monitor_session(
    templates_dir: Path,
    event_settings: EventSettings,
    detection_settings: DetectionSettings,
    quality_settings: QualitySettings,
    recognition_settings: RecognitionSettings,
    yunet_model_path: Path,
    sface_model_path: Path,
) -> MonitorSession:
    if not event_settings.zones:
        raise MonitorError("at least one events zone is required for monitor")
    zones = _build_zones(event_settings.zones)
    zone_severity = {item.name: item.severity for item in event_settings.zones}

    def severity(event_type: EventType, state) -> int:
        if event_type is EventType.LEAVE:
            return 0
        configured = zone_severity.get(state.zone or "", 1)
        return max(configured, 2 if event_type is EventType.STAY else 0)

    machine = ZoneStateMachine(
        zones,
        dwell_seconds=event_settings.dwell_seconds,
        lost_tolerance_seconds=event_settings.lost_tolerance_seconds,
        confirm_frames=1,
        event_cooldown_seconds=event_settings.cooldown_seconds,
        severity_resolver=severity,
    )
    return MonitorSession(
        detector=YuNetFaceDetector(yunet_model_path, detection_settings),
        evaluator=FaceQualityEvaluator(quality_settings),
        extractor=SFaceFeatureExtractor(sface_model_path),
        matcher=_build_matcher(templates_dir, recognition_settings),
        tracker=SimpleTrackManager(
            iou_threshold=event_settings.association_iou_threshold,
            confirm_frames=event_settings.confirm_frames,
            max_missing_frames=event_settings.max_missing_frames,
        ),
        zones=zones,
        event_machine=machine,
        event_log=EventLog(event_settings.log_path),
    )


def run_monitor_preview(
    camera_settings: CameraSettings,
    preview_settings: PreviewSettings,
    detection_settings: DetectionSettings,
    quality_settings: QualitySettings,
    recognition_settings: RecognitionSettings,
    event_settings: EventSettings,
    templates_dir: Path,
    yunet_model_path: Path,
    sface_model_path: Path,
) -> None:
    session = create_monitor_session(
        templates_dir,
        event_settings,
        detection_settings,
        quality_settings,
        recognition_settings,
        yunet_model_path,
        sface_model_path,
    )
    try:
        run_preview(camera_settings, preview_settings, session.process_frame)
    finally:
        session.finish()
