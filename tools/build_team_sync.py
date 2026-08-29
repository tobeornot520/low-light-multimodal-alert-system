"""Build the reviewed, source-only team synchronization archive."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

SNAPSHOT_DATE = "2026-08-29"
SNAPSHOT_NAME = f"lowlight-alert-team-sync-{SNAPSHOT_DATE.replace('-', '')}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip() or "unavailable"


def selected_files(root: Path) -> list[tuple[Path, Path]]:
    explicit = [
        (
            Path(".github/pull_request_template.md"),
            Path("project/.github/pull_request_template.md"),
        ),
        (Path(".github/workflows/ci.yml"), Path("project/.github/workflows/ci.yml")),
        (Path(".gitignore"), Path("project/.gitignore")),
        (Path("CONTRIBUTING.md"), Path("project/CONTRIBUTING.md")),
        (Path("README.md"), Path("project/README.md")),
        (Path("pyproject.toml"), Path("project/pyproject.toml")),
        (Path("requirements.txt"), Path("project/requirements.txt")),
        (Path("requirements-dev.txt"), Path("project/requirements-dev.txt")),
        (Path("run.py"), Path("project/run.py")),
        (Path("config/default.yaml"), Path("project/config/default.yaml")),
        (
            Path("config/experiment.example.yaml"),
            Path("project/config/experiment.example.yaml"),
        ),
        (
            Path("config/multimodal-manifest.example.yaml"),
            Path("project/config/multimodal-manifest.example.yaml"),
        ),
        (Path("models/README.md"), Path("project/models/README.md")),
        (Path("tools/build_team_sync.py"), Path("project/tools/build_team_sync.py")),
        (Path("参考资料/README.md"), Path("project/参考资料/README.md")),
        (
            Path("参考资料/组员同步与上手说明.md"),
            Path("project/参考资料/组员同步与上手说明.md"),
        ),
        (Path("参考资料/组员同步与上手说明.md"), Path("README-同步说明.md")),
        (
            Path("参考资料/项目申报书准备稿.md"),
            Path("project/参考资料/项目申报书准备稿.md"),
        ),
        (Path("参考资料/项目申报书准备稿.md"), Path("申报材料/项目申报书准备稿.md")),
        (
            Path("参考资料/项目会后共识与下一阶段路线.md"),
            Path("project/参考资料/项目会后共识与下一阶段路线.md"),
        ),
        (
            Path("参考资料/弱光实验执行与标注规范.md"),
            Path("project/参考资料/弱光实验执行与标注规范.md"),
        ),
        (
            Path("参考资料/弱光特定对象识别与分级预警知识手册.md"),
            Path("project/参考资料/弱光特定对象识别与分级预警知识手册.md"),
        ),
    ]
    result = [(root / source, destination) for source, destination in explicit]
    for source in sorted((root / "src/lowlight_alert").glob("*.py")):
        result.append((source, Path("project/src/lowlight_alert") / source.name))
    for source in sorted((root / "tests").glob("test_*.py")):
        result.append((source, Path("project/tests") / source.name))
    return result


def package_manifest(files: list[tuple[Path, Path]], base_commit: str) -> str:
    included = "\n".join(f"- {destination.as_posix()}" for _, destination in files)
    return f"""# Package manifest

snapshot_name: {SNAPSHOT_NAME}
snapshot_date: {SNAPSHOT_DATE}
base_commit: {base_commit}
source: reviewed working-tree snapshot

## Included files

{included}

## Deliberately excluded

- Git metadata and personal Git configuration
- Virtual environments, caches, build output, and IDE metadata
- ONNX binaries; project/models/README.md contains recovery hashes
- data/, captures/, logs/, templates, media, labels, event logs, and databases
- credentials, environment files, consent records, and identifying participant data
- pre-meeting drafts, raw ideation notes, and outdated task-allocation documents
- existing archives and the accidental files `h origin master` and
  `tore --staged README.md`
"""


def build(root: Path, output_directory: Path) -> tuple[Path, Path]:
    files = selected_files(root)
    base_commit = git_revision(root)
    missing = [source for source, _ in files if not source.is_file()]
    if missing:
        raise RuntimeError("missing selected file(s): " + ", ".join(map(str, missing)))
    destinations = [destination for _, destination in files]
    if len(destinations) != len(set(destinations)):
        raise RuntimeError("duplicate package destination")

    output_directory.mkdir(parents=True, exist_ok=True)
    archive = output_directory / f"{SNAPSHOT_NAME}.zip"
    checksum_file = output_directory / f"{SNAPSHOT_NAME}.zip.sha256"
    with tempfile.TemporaryDirectory(prefix="lowlight-alert-sync-") as temporary:
        package_root = Path(temporary) / SNAPSHOT_NAME
        for source, destination in files:
            target = package_root / destination
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            target.chmod(0o644)

        manifest_path = package_root / "PACKAGE-MANIFEST.md"
        manifest_path.write_text(
            package_manifest(files, base_commit), encoding="utf-8"
        )
        payload_files = sorted(path for path in package_root.rglob("*") if path.is_file())
        checksum_lines = [
            f"{sha256(path)}  {path.relative_to(package_root).as_posix()}"
            for path in payload_files
        ]
        (package_root / "FILES.sha256").write_text(
            "\n".join(checksum_lines) + "\n", encoding="utf-8"
        )

        temporary_archive = archive.with_suffix(".zip.tmp")
        temporary_archive.unlink(missing_ok=True)
        with zipfile.ZipFile(
            temporary_archive,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as destination_zip:
            for path in sorted(package_root.rglob("*")):
                if path.is_file():
                    archive_name = Path(SNAPSHOT_NAME) / path.relative_to(package_root)
                    destination_zip.write(path, archive_name.as_posix())
        temporary_archive.replace(archive)

    checksum_file.write_text(
        f"{sha256(archive)}  {archive.name}\n", encoding="ascii"
    )
    return archive, checksum_file


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[1]
    built_archive, built_checksum = build(repository, repository / "dist")
    print(built_archive)
    print(built_checksum)
