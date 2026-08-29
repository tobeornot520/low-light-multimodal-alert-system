from pathlib import Path
from types import SimpleNamespace

import pytest

import lowlight_alert.cli as cli_module
from lowlight_alert.replay import ReplayError


def _write_replay_config(tmp_path: Path) -> Path:
    path = tmp_path / "settings.yaml"
    path.write_text(
        "preview:\n"
        "  mirror: true\n"
        "enrollment:\n"
        f"  templates_dir: {tmp_path / 'templates'}\n"
        "events:\n"
        "  zones:\n"
        "    - name: room\n"
        "      polygon: [[0, 0], [1, 0], [1, 1]]\n"
        "models:\n"
        f"  yunet: {tmp_path / 'yunet.onnx'}\n"
        f"  sface: {tmp_path / 'sface.onnx'}\n",
        encoding="utf-8",
    )
    (tmp_path / "yunet.onnx").write_bytes(b"yunet")
    (tmp_path / "sface.onnx").write_bytes(b"sface")
    return path


def test_probe_rejects_non_positive_max_index(capsys) -> None:
    exit_code = cli_module.main(["probe", "--max-index", "0"])

    assert exit_code == 2
    assert "--max-index must be positive" in capsys.readouterr().err


def test_detect_dispatches_settings(monkeypatch) -> None:
    received = {}

    def fake_detection_preview(camera, preview, detection, model_path) -> None:
        received.update(
            camera=camera,
            preview=preview,
            detection=detection,
            model_path=model_path,
        )

    monkeypatch.setattr(cli_module, "run_detection_preview", fake_detection_preview)

    exit_code = cli_module.main(["detect", "--index", "2", "--no-mirror"])

    assert exit_code == 0
    assert received["camera"].index == 2
    assert received["preview"].mirror is False
    assert received["detection"].score_threshold == 0.9
    assert received["model_path"].name == "face_detection_yunet_2023mar.onnx"


def test_quality_dispatches_settings(monkeypatch) -> None:
    received = {}

    def fake_quality_preview(camera, preview, detection, quality, model_path) -> None:
        received.update(
            camera=camera,
            preview=preview,
            detection=detection,
            quality=quality,
            model_path=model_path,
        )

    monkeypatch.setattr(cli_module, "run_quality_preview", fake_quality_preview)

    exit_code = cli_module.main(["quality", "--width", "640"])

    assert exit_code == 0
    assert received["camera"].width == 640
    assert received["quality"].min_face_size == 80
    assert received["model_path"].name == "face_detection_yunet_2023mar.onnx"


def test_recognize_dispatches_settings(monkeypatch) -> None:
    received = {}

    def fake_recognition_preview(*args) -> None:
        received["args"] = args

    monkeypatch.setattr(cli_module, "run_recognition_preview", fake_recognition_preview)

    exit_code = cli_module.main(["recognize", "--index", "1"])

    assert exit_code == 0
    assert received["args"][0].index == 1
    assert received["args"][4].accept_threshold == 0.363
    assert received["args"][5].name == "templates"
    assert received["args"][6].name == "face_detection_yunet_2023mar.onnx"


def test_monitor_dispatches_event_and_recognition_settings(monkeypatch) -> None:
    received = {}

    def fake_monitor_preview(*args) -> None:
        received["args"] = args

    monkeypatch.setattr(cli_module, "run_monitor_preview", fake_monitor_preview)

    exit_code = cli_module.main(["monitor", "--index", "2", "--no-mirror"])

    assert exit_code == 0
    assert received["args"][0].index == 2
    assert received["args"][1].mirror is False
    assert received["args"][4].accept_threshold == 0.363
    assert received["args"][5].confirm_frames == 3
    assert received["args"][6].name == "templates"


def test_monitor_rejects_default_empty_zones(capsys) -> None:
    exit_code = cli_module.main(["monitor"])

    assert exit_code == 2
    assert "at least one events zone" in capsys.readouterr().err


def test_evaluate_dispatches_csv_and_threshold(tmp_path, capsys) -> None:
    path = tmp_path / "scores.csv"
    path.write_text(
        "genuine,score,condition\ntrue,0.9,normal\nfalse,0.1,normal\n",
        encoding="utf-8",
    )

    exit_code = cli_module.main(["evaluate", str(path), "--threshold", "0.5"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "CONDITION" in output
    assert "normal" in output
    assert "0.500" in output


def test_replay_dispatches_headless_session_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_replay_config(tmp_path)
    video_path = tmp_path / "source.mp4"
    video_path.write_bytes(b"video")
    received: dict[str, object] = {}
    session = SimpleNamespace(event_log=SimpleNamespace(path=None))

    def fake_create_monitor_session(*args):
        received["session_args"] = args
        session.event_log.path = args[1].log_path
        return session

    def fake_run_replay(*args, **kwargs):
        received["replay_args"] = args
        received["replay_kwargs"] = kwargs
        return SimpleNamespace(
            frames_processed=7,
            media_duration_seconds=0.56,
            processing_fps=25.0,
            event_ids=("event-1",),
        )

    monkeypatch.setattr(cli_module, "create_monitor_session", fake_create_monitor_session)
    monkeypatch.setattr(cli_module, "run_replay", fake_run_replay)

    exit_code = cli_module.main(
        [
            "--config",
            str(config_path),
            "replay",
            str(video_path),
            "--condition",
            "dim",
            "--experiment-id",
            "exp-01",
            "--output-dir",
            str(tmp_path / "replays"),
            "--source-fps",
            "12.5",
            "--max-frames",
            "7",
            "--no-mirror",
        ]
    )

    assert exit_code == 0
    run_directory = tmp_path / "replays" / "exp-01"
    assert run_directory.is_dir()
    session_args = received["session_args"]
    assert session_args[1].log_path == run_directory / "events.jsonl"
    replay_args = received["replay_args"]
    replay_kwargs = received["replay_kwargs"]
    assert replay_args == (video_path, session)
    assert replay_kwargs["telemetry_path"] == run_directory / "frames.csv"
    assert replay_kwargs["report_path"] == run_directory / "report.json"
    assert replay_kwargs["condition"] == "dim"
    assert replay_kwargs["fps_override"] == 12.5
    assert replay_kwargs["max_frames"] == 7
    assert replay_kwargs["mirror"] is False
    assert replay_kwargs["artifact_paths"] == {
        "config": config_path,
        "yunet": tmp_path / "yunet.onnx",
        "sface": tmp_path / "sface.onnx",
    }
    output = capsys.readouterr().out
    assert "Replay complete: 7 frames" in output
    assert f"Output: {run_directory}" in output


def test_replay_includes_templates_directory_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_replay_config(tmp_path)
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "person-a.npz").write_bytes(b"template")
    received: dict[str, object] = {}
    session = SimpleNamespace(event_log=SimpleNamespace(path=None))

    def fake_create_monitor_session(*args):
        session.event_log.path = args[1].log_path
        return session

    def fake_run_replay(*args, **kwargs):
        received["artifacts"] = kwargs["artifact_paths"]
        return SimpleNamespace(
            frames_processed=1,
            media_duration_seconds=0.1,
            processing_fps=None,
            event_ids=(),
        )

    monkeypatch.setattr(cli_module, "create_monitor_session", fake_create_monitor_session)
    monkeypatch.setattr(cli_module, "run_replay", fake_run_replay)
    exit_code = cli_module.main(
        [
            "--config",
            str(config_path),
            "replay",
            "source.mp4",
            "--condition",
            "normal",
            "--experiment-id",
            "exp-template-audit",
            "--output-dir",
            str(tmp_path / "replays"),
        ]
    )

    assert exit_code == 0
    assert received["artifacts"]["templates"] == templates


def test_replay_requires_explicit_config(capsys) -> None:
    exit_code = cli_module.main(
        [
            "replay",
            "source.mp4",
            "--condition",
            "normal",
            "--experiment-id",
            "exp-01",
        ]
    )

    assert exit_code == 2
    assert "replay requires --config" in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments",
    [
        ("--source-fps", "0"),
        ("--source-fps", "nan"),
        ("--max-frames", "0"),
    ],
)
def test_replay_rejects_invalid_numeric_options(
    tmp_path: Path,
    arguments: tuple[str, str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_replay_config(tmp_path)

    exit_code = cli_module.main(
        [
            "--config",
            str(config_path),
            "replay",
            "source.mp4",
            "--condition",
            "normal",
            "--experiment-id",
            "exp-01",
            *arguments,
        ]
    )

    assert exit_code == 2
    assert arguments[0] in capsys.readouterr().err


def test_replay_rejects_unsafe_or_existing_run_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_replay_config(tmp_path)
    output_root = tmp_path / "replays"
    existing = output_root / "exp-existing"
    existing.mkdir(parents=True)
    monkeypatch.setattr(
        cli_module,
        "create_monitor_session",
        lambda *args: pytest.fail("session must not be built for an existing run"),
    )

    common = [
        "--config",
        str(config_path),
        "replay",
        "source.mp4",
        "--condition",
        "normal",
        "--output-dir",
        str(output_root),
    ]
    unsafe_exit = cli_module.main([*common, "--experiment-id", "../escape"])
    unsafe_error = capsys.readouterr().err
    existing_exit = cli_module.main([*common, "--experiment-id", "exp-existing"])
    existing_error = capsys.readouterr().err

    assert unsafe_exit == 2
    assert "path separators" in unsafe_error
    assert existing_exit == 2
    assert "already exists" in existing_error


@pytest.mark.parametrize("experiment_id", ["exp:name", "exp*name", "exp\x7fname"])
def test_replay_rejects_cross_platform_unsafe_experiment_id(
    tmp_path: Path,
    experiment_id: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_replay_config(tmp_path)
    exit_code = cli_module.main(
        [
            "--config",
            str(config_path),
            "replay",
            "source.mp4",
            "--condition",
            "normal",
            "--experiment-id",
            experiment_id,
            "--output-dir",
            str(tmp_path / "replays"),
        ]
    )

    assert exit_code == 2
    assert "path separators" in capsys.readouterr().err


def test_replay_surfaces_video_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_replay_config(tmp_path)
    session = SimpleNamespace(event_log=SimpleNamespace(path=None))

    def fake_create_monitor_session(*args):
        session.event_log.path = args[1].log_path
        return session

    monkeypatch.setattr(cli_module, "create_monitor_session", fake_create_monitor_session)
    monkeypatch.setattr(
        cli_module,
        "run_replay",
        lambda *args, **kwargs: (_ for _ in ()).throw(ReplayError("cannot open source video")),
    )

    exit_code = cli_module.main(
        [
            "--config",
            str(config_path),
            "replay",
            "bad.mp4",
            "--condition",
            "near_black",
            "--experiment-id",
            "exp-bad-video",
            "--output-dir",
            str(tmp_path / "replays"),
        ]
    )

    assert exit_code == 2
    assert "cannot open source video" in capsys.readouterr().err
    assert not (tmp_path / "replays" / "exp-bad-video").exists()


def test_enroll_dispatches_subject_and_count(monkeypatch) -> None:
    received = {}

    def fake_enrollment_preview(*args) -> None:
        received["args"] = args

    monkeypatch.setattr(cli_module, "run_enrollment_preview", fake_enrollment_preview)

    exit_code = cli_module.main(
        [
            "enroll",
            "--subject-id",
            "person-a",
            "--display-name",
            "Person A",
            "--count",
            "3",
        ]
    )

    assert exit_code == 0
    assert received["args"][-3:] == ("person-a", "Person A", 3)


def test_enroll_rejects_non_positive_count(capsys) -> None:
    exit_code = cli_module.main(["enroll", "--subject-id", "person-a", "--count", "0"])

    assert exit_code == 2
    assert "--count must be positive" in capsys.readouterr().err
