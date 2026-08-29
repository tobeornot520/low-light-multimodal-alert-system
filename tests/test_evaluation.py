from pathlib import Path

import pytest

from lowlight_alert.evaluation import (
    ComparisonScore,
    EvaluationError,
    OpenSetObservation,
    evaluate_events,
    evaluate_open_set,
    evaluate_threshold,
    load_comparisons_csv,
)


def test_evaluate_threshold_reports_open_set_rates() -> None:
    metrics = evaluate_threshold(
        [
            ComparisonScore(True, 0.9),
            ComparisonScore(True, 0.2),
            ComparisonScore(False, 0.8),
            ComparisonScore(False, 0.1),
        ],
        threshold=0.5,
    )

    assert metrics.genuine_count == 2
    assert metrics.impostor_count == 2
    assert metrics.true_accepts == 1
    assert metrics.false_rejects == 1
    assert metrics.false_matches == 1
    assert metrics.tar == pytest.approx(0.5)
    assert metrics.fmr == pytest.approx(0.5)
    assert metrics.fnmr == pytest.approx(0.5)


def test_evaluate_open_set_reports_unknown_false_accepts_and_uncertain() -> None:
    metrics = evaluate_open_set(
        [
            OpenSetObservation(True, predicted_state="registered"),
            OpenSetObservation(True, predicted_state="uncertain"),
            OpenSetObservation(False, predicted_state="registered"),
            OpenSetObservation(False, predicted_state="unknown"),
        ]
    )

    assert metrics.tar == pytest.approx(0.5)
    assert metrics.unknown_false_accept_rate == pytest.approx(0.5)
    assert metrics.uncertain_rate == pytest.approx(0.25)


def test_evaluate_events_matches_with_window_and_counts_duplicate() -> None:
    metrics = evaluate_events(
        [
            {
                "zone": "door",
                "event_type": "enter",
                "expected_time_s": 10,
                "tolerance_before_s": 1,
                "tolerance_after_s": 1,
                "expected_severity": 2,
            }
        ],
        [
            {"zone": "door", "event_type": "enter", "observed_at": 10.5, "severity": 2},
            {"zone": "door", "event_type": "enter", "observed_at": 10.8, "severity": 2},
            {"zone": "door", "event_type": "leave", "observed_at": 20, "severity": 0},
        ],
    )

    assert metrics.matched_count == 1
    assert metrics.duplicate_count == 1
    assert metrics.false_positive_count == 1
    assert metrics.f1 == pytest.approx(0.5)
    assert metrics.latency_p50 == pytest.approx(0.5)
    assert metrics.severity_accuracy == 1.0


def test_load_comparisons_csv_accepts_aliases_and_condition(tmp_path: Path) -> None:
    path = tmp_path / "scores.csv"
    path.write_text(
        "genuine,score,condition\ntrue,0.9,normal\nimpostor,0.1,dark\n",
        encoding="utf-8",
    )

    rows = load_comparisons_csv(path)

    assert [(row.genuine, row.score, row.condition) for row in rows] == [
        (True, 0.9, "normal"),
        (False, 0.1, "dark"),
    ]


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("genuine,score\nmaybe,0.5\n", "invalid genuine"),
        ("genuine,score\ntrue,bad\n", "invalid score"),
        ("score\n0.5\n", "must contain genuine and score"),
    ],
)
def test_load_comparisons_csv_rejects_invalid_data(
    tmp_path: Path, contents: str, message: str
) -> None:
    path = tmp_path / "invalid.csv"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(EvaluationError, match=message):
        load_comparisons_csv(path)


def test_evaluate_threshold_rejects_empty_or_invalid_scores() -> None:
    with pytest.raises(EvaluationError, match="empty"):
        evaluate_threshold([], 0.5)
    with pytest.raises(EvaluationError, match="between -1 and 1"):
        evaluate_threshold([ComparisonScore(True, 1.1)], 0.5)
