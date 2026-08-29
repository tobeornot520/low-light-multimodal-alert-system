from pathlib import Path

import numpy as np
import pytest

import lowlight_alert.recognizer as recognizer_module
from lowlight_alert.detector import FaceDetection
from lowlight_alert.recognizer import (
    FaceRecognitionError,
    IdentityMatch,
    MatchState,
    SFaceFeatureExtractor,
    TemplateMatcher,
)
from lowlight_alert.template_store import SubjectTemplates


class FakeSFace:
    def __init__(self, feature: np.ndarray) -> None:
        self.output = feature
        self.face_row: np.ndarray | None = None

    def alignCrop(self, frame: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        self.face_row = face_row
        return frame[:112, :112]

    def feature(self, aligned: np.ndarray) -> np.ndarray:
        del aligned
        return self.output


def model_file(tmp_path: Path) -> Path:
    path = tmp_path / "sface.onnx"
    path.write_bytes(b"model")
    return path


def detection() -> FaceDetection:
    return FaceDetection(
        box=(10, 20, 100, 120),
        landmarks=((30, 40), (70, 40), (50, 65), (35, 90), (65, 90)),
        score=0.96,
    )


def test_extract_aligns_and_normalizes_feature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeSFace(np.array([[3.0, 4.0]], dtype=np.float32))
    monkeypatch.setattr(recognizer_module, "_create_sface", lambda *args: fake)
    extractor = SFaceFeatureExtractor(model_file(tmp_path))

    feature = extractor.extract(np.zeros((200, 200, 3), dtype=np.uint8), detection())

    assert feature == pytest.approx(np.array([0.6, 0.8], dtype=np.float32))
    assert fake.face_row is not None
    assert fake.face_row.shape == (15,)
    assert fake.face_row[-1] == pytest.approx(0.96)


def test_extract_rejects_zero_feature(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        recognizer_module,
        "_create_sface",
        lambda *args: FakeSFace(np.zeros((1, 4), dtype=np.float32)),
    )
    extractor = SFaceFeatureExtractor(model_file(tmp_path))

    with pytest.raises(FaceRecognitionError, match="zero-length"):
        extractor.extract(np.zeros((200, 200, 3), dtype=np.uint8), detection())


def test_similarity_uses_normalized_dot_product() -> None:
    first = np.array([1.0, 0.0], dtype=np.float32)
    second = np.array([0.5, 0.5], dtype=np.float32)

    assert SFaceFeatureExtractor.similarity(first, second) == pytest.approx(0.5)


def test_extractor_requires_model_file(tmp_path: Path) -> None:
    with pytest.raises(FaceRecognitionError, match="does not exist"):
        SFaceFeatureExtractor(tmp_path / "missing.onnx")


def enrolled(subject_id: str, feature: tuple[float, ...]) -> SubjectTemplates:
    return SubjectTemplates(
        subject_id=subject_id,
        display_name=subject_id.upper(),
        features=np.asarray([feature], dtype=np.float32),
        records=({},),
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )


def test_matcher_returns_registered_for_clear_best_match() -> None:
    matcher = TemplateMatcher(
        [enrolled("person-a", (1.0, 0.0)), enrolled("person-b", (0.0, 1.0))],
        accept_threshold=0.8,
        reject_threshold=0.3,
        min_margin=0.05,
    )

    result = matcher.match(np.array([3.0, 0.0], dtype=np.float32))

    assert result == IdentityMatch(
        MatchState.REGISTERED,
        "person-a",
        "PERSON-A",
        pytest.approx(1.0),
        pytest.approx(0.0),
        None,
    )


def test_matcher_normalizes_loaded_template_rows() -> None:
    matcher = TemplateMatcher(
        [enrolled("person-a", (10.0, 0.0))],
        accept_threshold=0.8,
        reject_threshold=0.3,
        min_margin=0.05,
    )

    result = matcher.match(np.array([1.0, 0.0], dtype=np.float32))

    assert result.state is MatchState.REGISTERED
    assert result.similarity == pytest.approx(1.0)


def test_matcher_distinguishes_unknown_and_uncertain() -> None:
    matcher = TemplateMatcher(
        [enrolled("person-a", (1.0, 0.0))],
        accept_threshold=0.8,
        reject_threshold=0.3,
        min_margin=0.05,
    )

    unknown = matcher.match(np.array([0.0, 1.0], dtype=np.float32))
    uncertain = matcher.match(np.array([0.7, 0.71414286], dtype=np.float32))

    assert unknown.state is MatchState.UNKNOWN
    assert unknown.reason == "below_reject_threshold"
    assert uncertain.state is MatchState.UNCERTAIN
    assert uncertain.reason == "score_or_margin_in_gray_zone"


def test_matcher_uses_margin_to_reject_ambiguous_match() -> None:
    matcher = TemplateMatcher(
        [enrolled("person-a", (1.0, 0.0)), enrolled("person-b", (0.99, 0.14106736))],
        accept_threshold=0.8,
        reject_threshold=0.3,
        min_margin=0.02,
    )

    result = matcher.match(np.array([1.0, 0.0], dtype=np.float32))

    assert result.state is MatchState.UNCERTAIN
    assert result.subject_id == "person-a"


def test_matcher_reports_empty_store_as_unknown() -> None:
    result = TemplateMatcher([], 0.8, 0.3, 0.05).match(np.array([1.0, 0.0]))

    assert result.state is MatchState.UNKNOWN
    assert result.reason == "no_enrolled_subjects"
