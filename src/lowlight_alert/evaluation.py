from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isnan
from pathlib import Path
from typing import Any


class EvaluationError(ValueError):
    """Raised when an offline comparison score file is invalid."""


@dataclass(frozen=True)
class ComparisonScore:
    genuine: bool
    score: float
    condition: str = "unspecified"


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    genuine_count: int
    impostor_count: int
    true_accepts: int
    false_rejects: int
    false_matches: int
    tar: float | None
    fmr: float | None
    fnmr: float | None


@dataclass(frozen=True)
class OpenSetObservation:
    """One runtime observation labelled as registered or unknown."""

    expected_registered: bool
    score: float | None = None
    predicted_state: str | None = None
    condition: str = "unspecified"


@dataclass(frozen=True)
class OpenSetMetrics:
    registered_count: int
    unknown_count: int
    accepted_registered: int
    rejected_registered: int
    uncertain_registered: int
    accepted_unknown: int
    rejected_unknown: int
    uncertain_unknown: int
    tar: float | None
    unknown_false_accept_rate: float | None
    fnmr: float | None
    uncertain_rate: float | None


@dataclass(frozen=True)
class EventMetrics:
    truth_count: int
    prediction_count: int
    matched_count: int
    duplicate_count: int
    false_positive_count: int
    false_negative_count: int
    precision: float | None
    recall: float | None
    f1: float | None
    latency_p50: float | None
    latency_p95: float | None
    severity_matches: int
    severity_accuracy: float | None


def evaluate_threshold(
    comparisons: Iterable[ComparisonScore], threshold: float
) -> ThresholdMetrics:
    if not -1.0 <= threshold <= 1.0:
        raise EvaluationError("threshold must be between -1 and 1")

    rows = list(comparisons)
    if not rows:
        raise EvaluationError("comparison score file is empty")
    for row in rows:
        if not -1.0 <= row.score <= 1.0:
            raise EvaluationError("comparison scores must be between -1 and 1")

    genuine = [row for row in rows if row.genuine]
    impostor = [row for row in rows if not row.genuine]
    true_accepts = sum(row.score >= threshold for row in genuine)
    false_matches = sum(row.score >= threshold for row in impostor)
    false_rejects = len(genuine) - true_accepts
    return ThresholdMetrics(
        threshold=threshold,
        genuine_count=len(genuine),
        impostor_count=len(impostor),
        true_accepts=true_accepts,
        false_rejects=false_rejects,
        false_matches=false_matches,
        tar=true_accepts / len(genuine) if genuine else None,
        fmr=false_matches / len(impostor) if impostor else None,
        fnmr=false_rejects / len(genuine) if genuine else None,
    )


def evaluate_open_set(observations: Iterable[OpenSetObservation]) -> OpenSetMetrics:
    """Evaluate runtime open-set decisions, including unknown false accepts.

    ``predicted_state`` takes precedence when supplied.  Otherwise a score is
    interpreted with the supplied thresholding system upstream; this function
    deliberately avoids silently treating an absent decision as a rejection.
    """
    rows = list(observations)
    if not rows:
        raise EvaluationError("open-set observations are empty")
    for row in rows:
        if row.score is not None and not -1.0 <= row.score <= 1.0:
            raise EvaluationError("open-set scores must be between -1 and 1")
        if row.predicted_state not in {"registered", "unknown", "uncertain"}:
            raise EvaluationError(
                "open-set predicted_state must be registered, unknown, or uncertain"
            )
    registered = [row for row in rows if row.expected_registered]
    unknown = [row for row in rows if not row.expected_registered]
    accepted_registered = sum(row.predicted_state == "registered" for row in registered)
    uncertain_registered = sum(row.predicted_state == "uncertain" for row in registered)
    accepted_unknown = sum(row.predicted_state == "registered" for row in unknown)
    uncertain_unknown = sum(row.predicted_state == "uncertain" for row in unknown)
    rejected_registered = len(registered) - accepted_registered
    rejected_unknown = len(unknown) - accepted_unknown
    uncertain_total = uncertain_registered + uncertain_unknown
    return OpenSetMetrics(
        registered_count=len(registered),
        unknown_count=len(unknown),
        accepted_registered=accepted_registered,
        rejected_registered=rejected_registered,
        uncertain_registered=uncertain_registered,
        accepted_unknown=accepted_unknown,
        rejected_unknown=rejected_unknown,
        uncertain_unknown=uncertain_unknown,
        tar=accepted_registered / len(registered) if registered else None,
        unknown_false_accept_rate=accepted_unknown / len(unknown) if unknown else None,
        fnmr=rejected_registered / len(registered) if registered else None,
        uncertain_rate=uncertain_total / len(rows),
    )


def evaluate_events(
    truth: Iterable[Mapping[str, Any]],
    predictions: Iterable[Mapping[str, Any]],
) -> EventMetrics:
    """One-to-one match zone events using the truth event's time window.

    Both inputs use the fields from ``event_truth.csv`` plus ``observed_at``
    for predictions. Matching requires equal zone and event type.  If truth
    includes ``gt_subject_id`` and a prediction includes ``subject_id``, those
    values must also agree.  Extra predictions in a matched truth window are
    counted as duplicates rather than hidden inside a single true positive.
    """
    truth_rows = [dict(row) for row in truth]
    prediction_rows = [dict(row) for row in predictions]
    _validate_event_rows(truth_rows, truth=True)
    _validate_event_rows(prediction_rows, truth=False)
    used_predictions: set[int] = set()
    matched = 0
    duplicates = 0
    latencies: list[float] = []
    severity_matches = 0
    severity_compared = 0
    for expected in truth_rows:
        candidates = [
            index
            for index, predicted in enumerate(prediction_rows)
            if index not in used_predictions and _event_matches(expected, predicted)
        ]
        if not candidates:
            continue
        selected = min(
            candidates,
            key=lambda index: abs(_event_time(prediction_rows[index]) - _truth_time(expected)),
        )
        used_predictions.add(selected)
        matched += 1
        predicted = prediction_rows[selected]
        latencies.append(_event_time(predicted) - _truth_time(expected))
        expected_severity = expected.get("expected_severity")
        if expected_severity not in (None, ""):
            severity_compared += 1
            severity_matches += int(int(expected_severity) == int(predicted.get("severity", 0)))
        # Remaining matching predictions describe the same incident and are duplicates.
        for index in candidates:
            if index != selected:
                used_predictions.add(index)
                duplicates += 1
    false_negative = len(truth_rows) - matched
    false_positive = len(prediction_rows) - matched - duplicates
    precision = matched / (matched + false_positive + duplicates) if prediction_rows else None
    recall = matched / len(truth_rows) if truth_rows else None
    if precision and recall:
        f1 = 2 * precision * recall / (precision + recall)
    elif precision == 0 or recall == 0:
        f1 = 0.0
    else:
        f1 = None
    return EventMetrics(
        truth_count=len(truth_rows), prediction_count=len(prediction_rows), matched_count=matched,
        duplicate_count=duplicates, false_positive_count=false_positive,
        false_negative_count=false_negative, precision=precision, recall=recall, f1=f1,
        latency_p50=_percentile(latencies, 50), latency_p95=_percentile(latencies, 95),
        severity_matches=severity_matches,
        severity_accuracy=severity_matches / severity_compared if severity_compared else None,
    )


def _truth_time(row: Mapping[str, Any]) -> float:
    return _number(row, "expected_time_s")


def _event_time(row: Mapping[str, Any]) -> float:
    return _number(row, "observed_at")


def _number(row: Mapping[str, Any], name: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationError(f"event row requires numeric {name}") from exc
    if isnan(value):
        raise EvaluationError(f"event row {name} cannot be NaN")
    return value


def _validate_event_rows(rows: list[dict[str, Any]], *, truth: bool) -> None:
    required = {"zone", "event_type", "expected_time_s" if truth else "observed_at"}
    for row in rows:
        if not required.issubset(row):
            raise EvaluationError("event rows are missing required matching fields")
        _truth_time(row) if truth else _event_time(row)


def _event_matches(expected: Mapping[str, Any], predicted: Mapping[str, Any]) -> bool:
    if expected["zone"] != predicted["zone"] or expected["event_type"] != predicted["event_type"]:
        return False
    truth_subject = expected.get("gt_subject_id")
    predicted_subject = predicted.get("subject_id")
    if (
        truth_subject not in (None, "")
        and predicted_subject not in (None, "")
        and truth_subject != predicted_subject
    ):
        return False
    before = float(expected.get("tolerance_before_s", 0.0) or 0.0)
    after = float(expected.get("tolerance_after_s", 0.0) or 0.0)
    return _truth_time(expected) - before <= _event_time(predicted) <= _truth_time(expected) + after


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


_TRUE_VALUES = {"1", "true", "yes", "genuine", "same"}
_FALSE_VALUES = {"0", "false", "no", "impostor", "different"}


def load_comparisons_csv(path: Path) -> list[ComparisonScore]:
    if not path.is_file():
        raise EvaluationError(f"score file does not exist: {path}")
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            required = {"genuine", "score"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise EvaluationError("score CSV must contain genuine and score columns")
            rows: list[ComparisonScore] = []
            for line_number, row in enumerate(reader, start=2):
                raw_genuine = (row.get("genuine") or "").strip().lower()
                if raw_genuine in _TRUE_VALUES:
                    genuine = True
                elif raw_genuine in _FALSE_VALUES:
                    genuine = False
                else:
                    raise EvaluationError(f"invalid genuine value on CSV line {line_number}")
                try:
                    score = float(row["score"])
                except (TypeError, ValueError) as exc:
                    raise EvaluationError(f"invalid score on CSV line {line_number}") from exc
                rows.append(
                    ComparisonScore(
                        genuine=genuine,
                        score=score,
                        condition=(row.get("condition") or "unspecified").strip() or "unspecified",
                    )
                )
    except OSError as exc:
        raise EvaluationError(f"cannot read score file {path}: {exc}") from exc
    return rows
