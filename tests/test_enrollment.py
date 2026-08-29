import numpy as np

from lowlight_alert.config import EnrollmentSettings
from lowlight_alert.detector import FaceDetection
from lowlight_alert.enrollment import EnrollmentSession
from lowlight_alert.quality import FaceQuality, QualityIssue


def detection() -> FaceDetection:
    return FaceDetection(
        box=(20, 20, 120, 120),
        landmarks=((50, 60), (110, 60), (80, 90), (60, 120), (100, 120)),
        score=0.98,
    )


def quality(*issues: QualityIssue) -> FaceQuality:
    return FaceQuality(
        issues=issues,
        face_size=120,
        brightness=128.0,
        sharpness=100.0,
        yaw_ratio=0.0,
        nose_position=0.5,
    )


class FakeDetector:
    def __init__(self, detections: list[FaceDetection]) -> None:
        self.detections = detections

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        del frame
        return self.detections


class FakeEvaluator:
    def __init__(self, result: FaceQuality) -> None:
        self.result = result

    def evaluate(self, frame: np.ndarray, face: FaceDetection) -> FaceQuality:
        del frame, face
        return self.result


class FakeExtractor:
    def __init__(self, feature: np.ndarray) -> None:
        self.feature = feature
        self.calls = 0

    def extract(self, frame: np.ndarray, face: FaceDetection) -> np.ndarray:
        del frame, face
        self.calls += 1
        return self.feature

    @staticmethod
    def similarity(first: np.ndarray, second: np.ndarray) -> float:
        return float(np.dot(first, second))


class FakeStore:
    def __init__(self) -> None:
        self.saved: list[np.ndarray] = []

    def load(self, subject_id: str):
        del subject_id
        return None

    def add_template(self, subject_id, display_name, feature, result, score) -> None:
        del subject_id, display_name, result, score
        self.saved.append(feature.copy())


def session(detector, evaluator, extractor, store, target_count: int = 1):
    return EnrollmentSession(
        detector=detector,
        evaluator=evaluator,
        extractor=extractor,
        store=store,
        settings=EnrollmentSettings(capture_interval_seconds=0),
        subject_id="person-a",
        display_name=None,
        target_count=target_count,
        clock=lambda: 1.0,
    )


def test_quality_rejection_is_not_stored() -> None:
    store = FakeStore()
    extractor = FakeExtractor(np.array([1.0, 0.0], dtype=np.float32))
    enrollment = session(
        FakeDetector([detection()]),
        FakeEvaluator(quality(QualityIssue.BLURRY)),
        extractor,
        store,
    )

    result = enrollment.process_frame(np.zeros((200, 200, 3), dtype=np.uint8))

    assert result.stop is False
    assert extractor.calls == 0
    assert store.saved == []


def test_exactly_one_good_face_is_stored_and_completes() -> None:
    store = FakeStore()
    enrollment = session(
        FakeDetector([detection()]),
        FakeEvaluator(quality()),
        FakeExtractor(np.array([1.0, 0.0], dtype=np.float32)),
        store,
    )

    result = enrollment.process_frame(np.zeros((200, 200, 3), dtype=np.uint8))

    assert result.stop is True
    assert enrollment.accepted_count == 1
    assert len(store.saved) == 1


def test_multiple_faces_are_not_stored() -> None:
    store = FakeStore()
    extractor = FakeExtractor(np.array([1.0, 0.0], dtype=np.float32))
    enrollment = session(
        FakeDetector([detection(), detection()]),
        FakeEvaluator(quality()),
        extractor,
        store,
    )

    result = enrollment.process_frame(np.zeros((200, 200, 3), dtype=np.uint8))

    assert result.stop is False
    assert extractor.calls == 0
    assert store.saved == []


def test_near_duplicate_feature_is_not_added_twice() -> None:
    store = FakeStore()
    enrollment = session(
        FakeDetector([detection()]),
        FakeEvaluator(quality()),
        FakeExtractor(np.array([1.0, 0.0], dtype=np.float32)),
        store,
        target_count=2,
    )
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    first = enrollment.process_frame(frame)
    second = enrollment.process_frame(frame)

    assert first.stop is False
    assert second.stop is False
    assert enrollment.accepted_count == 1
    assert len(store.saved) == 1
