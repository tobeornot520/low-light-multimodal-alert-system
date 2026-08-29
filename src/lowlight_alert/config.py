from __future__ import annotations

from dataclasses import dataclass, fields
from math import isfinite
from pathlib import Path
from typing import Any, TypeVar

import yaml


class ConfigError(ValueError):
    """Raised when an application configuration is invalid."""


@dataclass(frozen=True)
class CameraSettings:
    index: int = 0
    backend: str = "auto"
    width: int = 1280
    height: int = 720
    fps: int = 30
    probe_max_index: int = 5
    read_failure_limit: int = 10

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ConfigError("camera.index must be zero or greater")
        if self.backend not in {"auto", "dshow", "msmf"}:
            raise ConfigError("camera.backend must be one of: auto, dshow, msmf")
        if self.width <= 0 or self.height <= 0:
            raise ConfigError("camera width and height must be positive")
        if self.fps <= 0:
            raise ConfigError("camera.fps must be positive")
        if self.probe_max_index <= 0:
            raise ConfigError("camera.probe_max_index must be positive")
        if self.read_failure_limit <= 0:
            raise ConfigError("camera.read_failure_limit must be positive")


@dataclass(frozen=True)
class PreviewSettings:
    window_name: str = "Low-light Alert Camera"
    mirror: bool = True
    show_fps: bool = True

    def __post_init__(self) -> None:
        if not self.window_name.strip():
            raise ConfigError("preview.window_name cannot be empty")


@dataclass(frozen=True)
class DetectionSettings:
    score_threshold: float = 0.9
    nms_threshold: float = 0.3
    top_k: int = 5000

    def __post_init__(self) -> None:
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ConfigError("detection.score_threshold must be between 0 and 1")
        if not 0.0 <= self.nms_threshold <= 1.0:
            raise ConfigError("detection.nms_threshold must be between 0 and 1")
        if self.top_k <= 0:
            raise ConfigError("detection.top_k must be positive")


@dataclass(frozen=True)
class QualitySettings:
    min_face_size: int = 80
    min_sharpness: float = 60.0
    min_brightness: float = 35.0
    max_brightness: float = 225.0
    max_yaw_ratio: float = 0.35
    min_nose_position: float = 0.25
    max_nose_position: float = 0.80

    def __post_init__(self) -> None:
        if self.min_face_size <= 0:
            raise ConfigError("quality.min_face_size must be positive")
        if self.min_sharpness < 0:
            raise ConfigError("quality.min_sharpness cannot be negative")
        if not 0 <= self.min_brightness < self.max_brightness <= 255:
            raise ConfigError("quality brightness limits must satisfy 0 <= min < max <= 255")
        if self.max_yaw_ratio < 0:
            raise ConfigError("quality.max_yaw_ratio cannot be negative")
        if not 0 <= self.min_nose_position < self.max_nose_position <= 1:
            raise ConfigError("quality nose position limits must satisfy 0 <= min < max <= 1")


@dataclass(frozen=True)
class EnrollmentSettings:
    templates_dir: Path = Path("data/templates")
    target_count: int = 20
    capture_interval_seconds: float = 0.6
    duplicate_similarity_threshold: float = 0.995

    def __post_init__(self) -> None:
        if self.target_count <= 0:
            raise ConfigError("enrollment.target_count must be positive")
        if self.capture_interval_seconds < 0:
            raise ConfigError("enrollment.capture_interval_seconds cannot be negative")
        if not 0 <= self.duplicate_similarity_threshold <= 1:
            raise ConfigError("enrollment.duplicate_similarity_threshold must be between 0 and 1")


@dataclass(frozen=True)
class RecognitionSettings:
    accept_threshold: float = 0.363
    reject_threshold: float = 0.300
    min_margin: float = 0.020

    def __post_init__(self) -> None:
        for name, value in (
            ("accept_threshold", self.accept_threshold),
            ("reject_threshold", self.reject_threshold),
            ("min_margin", self.min_margin),
        ):
            if not 0.0 <= value <= 1.0:
                raise ConfigError(f"recognition.{name} must be between 0 and 1")
        if self.reject_threshold >= self.accept_threshold:
            raise ConfigError(
                "recognition.reject_threshold must be lower than accept_threshold"
            )


@dataclass(frozen=True)
class ZoneSettings:
    name: str
    polygon: tuple[tuple[float, float], ...]
    severity: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise ConfigError("events zone name must be a string")
        name = self.name.strip()
        if not name or len(name) > 64:
            raise ConfigError("events zone name must contain 1-64 characters")
        object.__setattr__(self, "name", name)

        try:
            raw_points = tuple(self.polygon)
        except TypeError as exc:
            raise ConfigError("events zone polygon must be iterable") from exc
        if len(raw_points) < 3:
            raise ConfigError("events zone polygon must contain at least 3 points")

        points: list[tuple[float, float]] = []
        for point in raw_points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ConfigError("events zone polygon points must be finite x,y pairs")
            try:
                x, y = float(point[0]), float(point[1])
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    "events zone polygon points must be finite x,y pairs"
                ) from exc
            if not isfinite(x) or not isfinite(y):
                raise ConfigError("events zone polygon points must be finite x,y pairs")
            if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
                raise ConfigError("events zone polygon coordinates must be between 0 and 1")
            points.append((x, y))

        area2 = sum(
            first[0] * second[1] - second[0] * first[1]
            for first, second in zip(points, points[1:] + points[:1], strict=True)
        )
        if abs(area2) <= 1e-12:
            raise ConfigError("events zone polygon must enclose a non-zero area")
        object.__setattr__(self, "polygon", tuple(points))

        if isinstance(self.severity, bool) or not isinstance(self.severity, int):
            raise ConfigError("events zone severity must be an integer between 0 and 2")
        if not 0 <= self.severity <= 2:
            raise ConfigError("events zone severity must be between 0 and 2")


@dataclass(frozen=True)
class EventSettings:
    log_path: Path = Path("logs/events.jsonl")
    confirm_frames: int = 3
    max_missing_frames: int = 5
    lost_tolerance_seconds: float = 1.0
    dwell_seconds: float = 2.0
    cooldown_seconds: float = 10.0
    association_iou_threshold: float = 0.20
    zones: tuple[ZoneSettings, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.log_path, str) and not self.log_path.strip():
            raise ConfigError("events.log_path cannot be empty")
        try:
            log_path = Path(self.log_path)
        except TypeError as exc:
            raise ConfigError("events.log_path must be a path") from exc
        if log_path.exists() and log_path.is_dir():
            raise ConfigError("events.log_path must point to a file")
        object.__setattr__(self, "log_path", log_path)

        for name in ("confirm_frames", "max_missing_frames"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"events.{name} must be an integer")
        if self.confirm_frames <= 0:
            raise ConfigError("events.confirm_frames must be positive")
        if self.max_missing_frames < 0:
            raise ConfigError("events.max_missing_frames cannot be negative")

        for name in (
            "lost_tolerance_seconds",
            "dwell_seconds",
            "cooldown_seconds",
            "association_iou_threshold",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"events.{name} must be numeric")
            normalized = float(value)
            if not isfinite(normalized):
                raise ConfigError(f"events.{name} must be finite")
            object.__setattr__(self, name, normalized)
        if self.lost_tolerance_seconds < 0:
            raise ConfigError("events.lost_tolerance_seconds cannot be negative")
        if self.dwell_seconds < 0:
            raise ConfigError("events.dwell_seconds cannot be negative")
        if self.cooldown_seconds < 0:
            raise ConfigError("events.cooldown_seconds cannot be negative")
        if not 0.0 <= self.association_iou_threshold <= 1.0:
            raise ConfigError("events.association_iou_threshold must be between 0 and 1")

        try:
            zones = tuple(self.zones)
        except TypeError as exc:
            raise ConfigError("events.zones must be iterable") from exc
        if any(not isinstance(zone, ZoneSettings) for zone in zones):
            raise ConfigError("events.zones must contain ZoneSettings values")
        object.__setattr__(self, "zones", zones)
        names = [zone.name for zone in zones]
        if len(names) != len(set(names)):
            raise ConfigError("events zone names must be unique")


@dataclass(frozen=True)
class ModelSettings:
    yunet: Path = Path("models/face_detection_yunet_2023mar.onnx")
    sface: Path = Path("models/face_recognition_sface_2021dec.onnx")


@dataclass(frozen=True)
class AppSettings:
    camera: CameraSettings = CameraSettings()
    preview: PreviewSettings = PreviewSettings()
    detection: DetectionSettings = DetectionSettings()
    quality: QualitySettings = QualitySettings()
    enrollment: EnrollmentSettings = EnrollmentSettings()
    recognition: RecognitionSettings = RecognitionSettings()
    events: EventSettings = EventSettings()
    models: ModelSettings = ModelSettings()


SettingsType = TypeVar(
    "SettingsType",
    CameraSettings,
    PreviewSettings,
    DetectionSettings,
    QualitySettings,
    EnrollmentSettings,
    RecognitionSettings,
    EventSettings,
    ModelSettings,
)


def _mapping(value: Any, section: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{section} must be a mapping")
    return value


def _build_settings(settings_type: type[SettingsType], raw: Any, section: str) -> SettingsType:
    values = _mapping(raw, section)
    allowed = {field.name for field in fields(settings_type)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise ConfigError(f"unknown {section} setting(s): {joined}")
    try:
        return settings_type(**values)
    except ConfigError:
        raise
    except (AttributeError, TypeError, ValueError) as exc:
        raise ConfigError(f"invalid {section} settings: {exc}") from exc


def _build_zones(raw: Any) -> tuple[ZoneSettings, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise ConfigError("events.zones must be a list")
    zones: list[ZoneSettings] = []
    allowed = {field.name for field in fields(ZoneSettings)}
    for index, value in enumerate(raw):
        if not isinstance(value, dict):
            raise ConfigError(f"events.zones[{index}] must be a mapping")
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ConfigError(
                f"unknown events.zones[{index}] setting(s): {', '.join(unknown)}"
            )
        if "name" not in value or "polygon" not in value:
            raise ConfigError(f"events.zones[{index}] requires name and polygon")
        polygon_raw = value["polygon"]
        if not isinstance(polygon_raw, (list, tuple)):
            raise ConfigError(f"events.zones[{index}].polygon must be a list")
        if not isinstance(value["name"], str):
            raise ConfigError(f"events.zones[{index}].name must be a string")
        severity = value.get("severity", 1)
        if isinstance(severity, bool) or not isinstance(severity, int):
            raise ConfigError(
                f"events.zones[{index}].severity must be an integer between 0 and 2"
            )
        polygon: list[tuple[float, float]] = []
        for point_index, point in enumerate(polygon_raw):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ConfigError(
                    f"events.zones[{index}].polygon[{point_index}] must be an x,y pair"
                )
            try:
                polygon.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"invalid events.zones[{index}].polygon[{point_index}]"
                ) from exc
        zones.append(ZoneSettings(value["name"], tuple(polygon), severity))
    return tuple(zones)


def load_settings(path: Path | str | None = None) -> AppSettings:
    """Load settings from YAML, falling back to built-in defaults."""
    if path is None:
        return AppSettings()
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"configuration file does not exist: {path}")

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read configuration file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"cannot parse configuration file {path}: {exc}") from exc

    raw = _mapping(loaded, "root")
    allowed_sections = {
        "camera",
        "preview",
        "detection",
        "quality",
        "enrollment",
        "recognition",
        "events",
        "models",
    }
    unknown_sections = sorted(set(raw) - allowed_sections)
    if unknown_sections:
        joined = ", ".join(unknown_sections)
        raise ConfigError(f"unknown configuration section(s): {joined}")

    model_values = _mapping(raw.get("models"), "models")
    model_values = {key: Path(value) for key, value in model_values.items()}
    enrollment_values = _mapping(raw.get("enrollment"), "enrollment")
    if "templates_dir" in enrollment_values:
        enrollment_values = {
            **enrollment_values,
            "templates_dir": Path(enrollment_values["templates_dir"]),
        }
    event_values = _mapping(raw.get("events"), "events")
    if "log_path" in event_values:
        event_values = {**event_values, "log_path": Path(event_values["log_path"])}
    if "zones" in event_values:
        event_values = {**event_values, "zones": _build_zones(event_values["zones"])}
    return AppSettings(
        camera=_build_settings(CameraSettings, raw.get("camera"), "camera"),
        preview=_build_settings(PreviewSettings, raw.get("preview"), "preview"),
        detection=_build_settings(DetectionSettings, raw.get("detection"), "detection"),
        quality=_build_settings(QualitySettings, raw.get("quality"), "quality"),
        enrollment=_build_settings(EnrollmentSettings, enrollment_values, "enrollment"),
        recognition=_build_settings(RecognitionSettings, raw.get("recognition"), "recognition"),
        events=_build_settings(EventSettings, event_values, "events"),
        models=_build_settings(ModelSettings, model_values, "models"),
    )
