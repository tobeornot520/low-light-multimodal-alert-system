from pathlib import Path

import pytest

from lowlight_alert.config import ConfigError, load_settings


def test_load_settings_uses_defaults() -> None:
    settings = load_settings()

    assert settings.camera.index == 0
    assert settings.camera.backend == "auto"
    assert settings.preview.mirror is True
    assert settings.detection.score_threshold == 0.9
    assert settings.quality.min_face_size == 80
    assert settings.enrollment.target_count == 20
    assert settings.recognition.accept_threshold == 0.363
    assert settings.recognition.reject_threshold == 0.300
    assert settings.events.confirm_frames == 3
    assert settings.events.zones == ()
    assert settings.models.yunet.name == "face_detection_yunet_2023mar.onnx"


def test_load_settings_reads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text(
        """
camera:
  index: 2
  backend: dshow
  width: 640
  height: 480
preview:
  mirror: false
detection:
  score_threshold: 0.75
quality:
  min_face_size: 96
enrollment:
  templates_dir: local/templates
  target_count: 12
recognition:
  accept_threshold: 0.8
  reject_threshold: 0.4
  min_margin: 0.05
events:
  log_path: local/events.jsonl
  confirm_frames: 4
  max_missing_frames: 2
  lost_tolerance_seconds: 0.8
  dwell_seconds: 1.5
  cooldown_seconds: 4
  association_iou_threshold: 0.25
  zones:
    - name: door
      severity: 2
      polygon:
        - [0.1, 0.1]
        - [0.9, 0.1]
        - [0.9, 0.9]
        - [0.1, 0.9]
models:
  yunet: custom/yunet.onnx
""",
        encoding="utf-8",
    )

    settings = load_settings(path)

    assert settings.camera.index == 2
    assert settings.camera.backend == "dshow"
    assert settings.camera.width == 640
    assert settings.preview.mirror is False
    assert settings.detection.score_threshold == 0.75
    assert settings.quality.min_face_size == 96
    assert settings.enrollment.templates_dir == Path("local/templates")
    assert settings.enrollment.target_count == 12
    assert settings.recognition.accept_threshold == 0.8
    assert settings.recognition.reject_threshold == 0.4
    assert settings.recognition.min_margin == 0.05
    assert settings.events.log_path == Path("local/events.jsonl")
    assert settings.events.confirm_frames == 4
    assert settings.events.lost_tolerance_seconds == 0.8
    assert settings.events.zones[0].name == "door"
    assert settings.events.zones[0].severity == 2
    assert settings.events.zones[0].polygon[2] == (0.9, 0.9)
    assert settings.models.yunet == Path("custom/yunet.onnx")


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("camera:\n  indx: 1\n", "unknown camera setting"),
        ("camera:\n  index: -1\n", "camera.index"),
        ("detection:\n  score_threshold: 1.1\n", "detection.score_threshold"),
        (
            "quality:\n  min_brightness: 200\n  max_brightness: 100\n",
            "quality brightness limits",
        ),
        ("enrollment:\n  target_count: 0\n", "enrollment.target_count"),
        (
            "recognition:\n  reject_threshold: 0.9\n  accept_threshold: 0.8\n",
            "recognition.reject_threshold",
        ),
        ("events:\n  confirm_frames: 0\n", "events.confirm_frames"),
        (
            "events:\n  zones:\n    - name: bad\n      polygon: [[0, 0], [1, 0]]\n",
            "at least 3 points",
        ),
        ("unexpected: true\n", "unknown configuration section"),
    ],
)
def test_load_settings_rejects_invalid_config(tmp_path: Path, contents: str, message: str) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_settings(path)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (
            "events:\n  zones:\n    - name: 42\n      polygon: [[0, 0], [1, 0], [1, 1]]\n",
            "name must be a string",
        ),
        (
            "events:\n  zones:\n    - name: room\n"
            "      severity: 1.5\n"
            "      polygon: [[0, 0], [1, 0], [1, 1]]\n",
            "severity must be an integer",
        ),
        (
            "events:\n  zones:\n    - name: room\n"
            "      severity: true\n"
            "      polygon: [[0, 0], [1, 0], [1, 1]]\n",
            "severity must be an integer",
        ),
        (
            "events:\n  zones:\n    - name: room\n      polygon: [[0, 0], [0.5, 0], [1, 0]]\n",
            "non-zero area",
        ),
    ],
)
def test_load_settings_rejects_malformed_zone_values(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    path = tmp_path / "invalid-zone.yaml"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_settings(path)


def test_load_settings_normalizes_zone_name_and_accepts_string_path(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text(
        "events:\n"
        "  zones:\n"
        "    - name: '  room  '\n"
        "      polygon: [[0, 0], [1, 0], [1, 1]]\n",
        encoding="utf-8",
    )

    settings = load_settings(str(path))

    assert settings.events.zones[0].name == "room"
    assert settings.events.zones[0].polygon == ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))
