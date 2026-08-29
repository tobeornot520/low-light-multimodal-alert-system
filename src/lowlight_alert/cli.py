from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import replace
from math import isfinite
from pathlib import Path

from lowlight_alert.camera import CameraError, probe_cameras
from lowlight_alert.config import AppSettings, ConfigError, load_settings
from lowlight_alert.data_protocol import ManifestError, load_manifest
from lowlight_alert.detection_preview import run_detection_preview
from lowlight_alert.detector import FaceDetectorError
from lowlight_alert.enrollment import run_enrollment_preview
from lowlight_alert.evaluation import (
    ComparisonScore,
    EvaluationError,
    evaluate_threshold,
    load_comparisons_csv,
)
from lowlight_alert.event_log import EventLogError
from lowlight_alert.events import EventError
from lowlight_alert.monitor_preview import (
    MonitorError,
    create_monitor_session,
    run_monitor_preview,
)
from lowlight_alert.preview import run_preview
from lowlight_alert.quality import QualityEvaluationError
from lowlight_alert.quality_preview import run_quality_preview
from lowlight_alert.recognition_preview import run_recognition_preview
from lowlight_alert.recognizer import FaceRecognitionError
from lowlight_alert.replay import ReplayError, run_replay
from lowlight_alert.template_store import TemplateStore, TemplateStoreError

_REPLAY_CONDITIONS = ("normal", "dim", "backlight", "near_black")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lowlight-alert",
        description="Low-light recognition and alert prototype",
    )
    parser.add_argument("--config", type=Path, help="YAML configuration file")
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser("probe", help="find readable cameras")
    probe.add_argument("--max-index", type=int, help="number of camera indexes to probe")
    probe.add_argument("--backend", choices=("auto", "dshow", "msmf"))

    preview = subparsers.add_parser("preview", help="show a live camera preview")
    detect = subparsers.add_parser("detect", help="show live YuNet face detections")
    quality = subparsers.add_parser("quality", help="show live face quality results")
    recognize = subparsers.add_parser(
        "recognize", help="show live open-set identity matching results"
    )
    monitor = subparsers.add_parser(
        "monitor", help="run local multi-frame zone monitoring and event logging"
    )
    evaluate = subparsers.add_parser(
        "evaluate", help="evaluate saved identity comparison scores"
    )
    replay = subparsers.add_parser(
        "replay", help="run deterministic headless monitoring on a saved video"
    )
    validate_manifest = subparsers.add_parser(
        "validate-manifest", help="validate a non-sensitive experiment manifest"
    )
    enroll = subparsers.add_parser("enroll", help="capture authorized face templates")
    subparsers.add_parser("subjects", help="list enrolled subjects")
    for command in (preview, detect, quality, recognize, monitor, enroll):
        command.add_argument("--index", type=int, help="camera index")
        command.add_argument("--backend", choices=("auto", "dshow", "msmf"))
        command.add_argument("--width", type=int, help="requested frame width")
        command.add_argument("--height", type=int, help="requested frame height")
        command.add_argument("--fps", type=int, help="requested frames per second")
        command.add_argument("--no-mirror", action="store_true", help="disable mirroring")
    enroll.add_argument("--subject-id", required=True, help="stable non-sensitive subject ID")
    enroll.add_argument("--display-name", help="optional local display name")
    enroll.add_argument("--count", type=int, help="number of new templates to capture")
    evaluate.add_argument(
        "scores", type=Path, help="CSV with genuine, score, and optional condition"
    )
    evaluate.add_argument("--threshold", type=float, help="matching threshold override")
    replay.add_argument("video", type=Path, help="saved constant-frame-rate video")
    replay.add_argument(
        "--condition",
        choices=_REPLAY_CONDITIONS,
        required=True,
        help="lighting condition recorded for this run",
    )
    replay.add_argument(
        "--experiment-id",
        required=True,
        help="unique run identifier used as the output directory name",
    )
    replay.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/replays"),
        help="output root; the experiment ID is appended",
    )
    replay.add_argument(
        "--source-fps",
        type=float,
        help="FPS override when the video metadata is missing or invalid",
    )
    replay.add_argument("--max-frames", type=int, help="optional positive frame limit")
    replay.add_argument("--no-mirror", action="store_true", help="disable mirroring")
    validate_manifest.add_argument("manifest", type=Path, help="experiment manifest YAML")
    return parser


def _camera_overrides(settings: AppSettings, args: argparse.Namespace) -> AppSettings:
    values = {
        name: getattr(args, name)
        for name in ("index", "backend", "width", "height", "fps")
        if hasattr(args, name) and getattr(args, name) is not None
    }
    camera = replace(settings.camera, **values)
    preview = settings.preview
    if getattr(args, "no_mirror", False):
        preview = replace(preview, mirror=False)
    return replace(settings, camera=camera, preview=preview)


def _run_probe(settings: AppSettings, args: argparse.Namespace) -> int:
    max_index = settings.camera.probe_max_index if args.max_index is None else args.max_index
    if max_index <= 0:
        raise ConfigError("--max-index must be positive")
    backend = args.backend or settings.camera.backend
    cameras = probe_cameras(max_index, backend)
    if not cameras:
        print(f"No readable cameras found in indexes 0..{max_index - 1}.")
        return 1

    print("INDEX  RESOLUTION  FPS    BACKEND")
    for camera in cameras:
        print(
            f"{camera.index:<5}  {camera.width}x{camera.height:<7}  "
            f"{camera.fps:<5.1f}  {camera.backend}"
        )
    return 0


def _run_subjects(settings: AppSettings) -> int:
    subjects = TemplateStore(settings.enrollment.templates_dir).list_subjects()
    if not subjects:
        print("No enrolled subjects.")
        return 0
    print("SUBJECT ID  TEMPLATES  DISPLAY NAME")
    for subject in subjects:
        print(f"{subject.subject_id:<10}  {subject.template_count:<9}  {subject.display_name}")
    return 0


def _run_evaluate(settings: AppSettings, args: argparse.Namespace) -> int:
    threshold = settings.recognition.accept_threshold if args.threshold is None else args.threshold
    rows = load_comparisons_csv(args.scores)
    groups: dict[str, list[ComparisonScore]] = {"all": rows}
    for condition in sorted({row.condition for row in rows}):
        groups[condition] = [row for row in rows if row.condition == condition]

    print("CONDITION  THRESHOLD  GENUINE  IMPOSTOR  TAR     FMR     FNMR")
    for condition, comparisons in groups.items():
        metrics = evaluate_threshold(comparisons, threshold)
        values = (
            "n/a" if metrics.tar is None else f"{metrics.tar:.4f}",
            "n/a" if metrics.fmr is None else f"{metrics.fmr:.4f}",
            "n/a" if metrics.fnmr is None else f"{metrics.fnmr:.4f}",
        )
        print(
            f"{condition:<9}  {metrics.threshold:<9.3f}  {metrics.genuine_count:<7}  "
            f"{metrics.impostor_count:<8}  {values[0]:<6}  {values[1]:<6}  {values[2]}"
        )
    return 0


def _run_validate_manifest(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    print(f"Manifest valid: {manifest.experiment_id}/{manifest.run_id}")
    print(f"Split: {manifest.dataset_split}; purpose: {manifest.purpose}")
    print(f"Condition: {manifest.lighting_condition}; modalities: {', '.join(manifest.modalities)}")
    print(
        "Participants: "
        f"{len(manifest.registered_subject_ids)} registered, "
        f"{len(manifest.unknown_subject_ids)} unknown"
    )
    return 0


def _replay_run_directory(output_root: Path, experiment_id: str) -> Path:
    normalized = experiment_id.strip()
    has_control_character = any(
        ord(character) < 32 or ord(character) == 127 for character in normalized
    )
    has_windows_forbidden_character = any(
        character in '<>:"/\\|?*' for character in normalized
    )
    if (
        normalized != experiment_id
        or normalized in {"", ".", ".."}
        or len(normalized) > 128
        or has_control_character
        or has_windows_forbidden_character
    ):
        raise ConfigError(
            "--experiment-id must be 1..128 characters without surrounding "
            "whitespace, control characters, or path separators"
        )
    run_directory = output_root / normalized
    try:
        run_directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ReplayError(f"replay output directory already exists: {run_directory}") from exc
    except OSError as exc:
        raise ReplayError(f"cannot create replay output directory {run_directory}: {exc}") from exc
    return run_directory


def _run_replay(settings: AppSettings, args: argparse.Namespace) -> int:
    if args.config is None:
        raise ConfigError("replay requires --config so the report can hash its configuration")
    if args.source_fps is not None and (
        not isfinite(args.source_fps) or args.source_fps <= 0
    ):
        raise ConfigError("--source-fps must be a finite positive number")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ConfigError("--max-frames must be positive")

    run_directory = _replay_run_directory(args.output_dir, args.experiment_id)
    try:
        event_settings = replace(settings.events, log_path=run_directory / "events.jsonl")
        session = create_monitor_session(
            settings.enrollment.templates_dir,
            event_settings,
            settings.detection,
            settings.quality,
            settings.recognition,
            settings.models.yunet,
            settings.models.sface,
        )
        artifacts = {
            "config": args.config,
            "yunet": settings.models.yunet,
            "sface": settings.models.sface,
        }
        if settings.enrollment.templates_dir.is_dir():
            artifacts["templates"] = settings.enrollment.templates_dir
        result = run_replay(
            args.video,
            session,
            telemetry_path=run_directory / "frames.csv",
            report_path=run_directory / "report.json",
            condition=args.condition,
            experiment_id=args.experiment_id,
            fps_override=args.source_fps,
            mirror=settings.preview.mirror,
            max_frames=args.max_frames,
            artifact_paths=artifacts,
        )
    except Exception:
        # The directory was reserved by this invocation.  Remove it only when
        # it is empty, so a failed run can be retried without risking artifacts
        # that may have appeared independently.
        with suppress(OSError):
            run_directory.rmdir()
        raise
    processing_fps = "n/a" if result.processing_fps is None else f"{result.processing_fps:.2f}"
    print(
        f"Replay complete: {result.frames_processed} frames, "
        f"{result.media_duration_seconds:.3f}s media, {processing_fps} processing FPS."
    )
    print(f"Events: {len(result.event_ids)}")
    print(f"Output: {run_directory}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        settings = _camera_overrides(load_settings(args.config), args)
        if args.command == "probe":
            return _run_probe(settings, args)
        if args.command == "subjects":
            return _run_subjects(settings)
        if args.command == "evaluate":
            return _run_evaluate(settings, args)
        if args.command == "validate-manifest":
            return _run_validate_manifest(args)
        if args.command == "replay":
            return _run_replay(settings, args)
        if args.command == "preview":
            run_preview(settings.camera, settings.preview)
        elif args.command == "detect":
            run_detection_preview(
                settings.camera,
                settings.preview,
                settings.detection,
                settings.models.yunet,
            )
        elif args.command == "quality":
            run_quality_preview(
                settings.camera,
                settings.preview,
                settings.detection,
                settings.quality,
                settings.models.yunet,
            )
        elif args.command == "recognize":
            run_recognition_preview(
                settings.camera,
                settings.preview,
                settings.detection,
                settings.quality,
                settings.recognition,
                settings.enrollment.templates_dir,
                settings.models.yunet,
                settings.models.sface,
            )
        elif args.command == "monitor":
            run_monitor_preview(
                settings.camera,
                settings.preview,
                settings.detection,
                settings.quality,
                settings.recognition,
                settings.events,
                settings.enrollment.templates_dir,
                settings.models.yunet,
                settings.models.sface,
            )
        else:
            target_count = settings.enrollment.target_count if args.count is None else args.count
            if target_count <= 0:
                raise ConfigError("--count must be positive")
            run_enrollment_preview(
                settings.camera,
                settings.preview,
                settings.detection,
                settings.quality,
                settings.enrollment,
                settings.models.yunet,
                settings.models.sface,
                args.subject_id,
                args.display_name,
                target_count,
            )
        return 0
    except (
        CameraError,
        ConfigError,
        ManifestError,
        EventLogError,
        EventError,
        EvaluationError,
        FaceDetectorError,
        FaceRecognitionError,
        MonitorError,
        QualityEvaluationError,
        ReplayError,
        TemplateStoreError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
