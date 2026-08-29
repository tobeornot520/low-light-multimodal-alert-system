import pytest

from lowlight_alert.fusion import FusionError, ModalityEvidence, fuse_evidence


def test_fusion_accepts_only_agreeing_registered_modalities() -> None:
    decision = fuse_evidence(
        [
            ModalityEvidence(
                "rgb", person_detected=True, identity_state="registered", subject_id="a"
            ),
            ModalityEvidence(
                "nir", person_detected=True, identity_state="registered", subject_id="a"
            ),
        ]
    )

    assert decision.state == "authorized"
    assert decision.subject_id == "a"


def test_fusion_keeps_conflicting_modalities_uncertain() -> None:
    decision = fuse_evidence(
        [
            ModalityEvidence(
                "rgb", person_detected=True, identity_state="registered", subject_id="a"
            ),
            ModalityEvidence("thermal", person_detected=True, identity_state="unknown"),
        ],
        zone_event=True,
    )

    assert decision.state == "uncertain"
    assert decision.severity == 2
    assert "modality_conflict" in decision.evidence_flags


def test_fusion_treats_presence_only_sensor_as_uncertain() -> None:
    decision = fuse_evidence([ModalityEvidence("thermal", person_detected=True)], zone_event=True)

    assert decision.state == "uncertain"
    assert decision.severity == 1


def test_fusion_reports_all_unavailable() -> None:
    decision = fuse_evidence([ModalityEvidence("rgb", available=False)])

    assert decision.state == "sensor_fault"
    assert decision.severity == 2


def test_fusion_rejects_invalid_identity_state() -> None:
    with pytest.raises(FusionError):
        ModalityEvidence("rgb", identity_state="maybe")
