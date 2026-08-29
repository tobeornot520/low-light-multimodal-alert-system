"""Modality-agnostic evidence fusion for tiered alerts.

This module consumes observations produced by RGB/NIR/thermal/other adapters.
It intentionally contains no device code, allowing the decision rules to be
tested before additional sensors are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class FusionError(ValueError):
    """Raised when modality evidence is malformed."""


@dataclass(frozen=True, slots=True)
class ModalityEvidence:
    modality: str
    available: bool = True
    person_detected: bool = False
    identity_state: str | None = None
    subject_id: str | None = None
    confidence: float | None = None
    quality_passed: bool | None = None
    timestamp: float | None = None

    def __post_init__(self) -> None:
        modality = str(self.modality).strip().lower()
        if not modality:
            raise FusionError("modality cannot be empty")
        object.__setattr__(self, "modality", modality)
        if self.confidence is not None and not 0.0 <= float(self.confidence) <= 1.0:
            raise FusionError("modality confidence must be between 0 and 1")
        if self.identity_state is not None:
            state = str(self.identity_state).strip().lower()
            if state not in {"registered", "unknown", "uncertain", "none"}:
                raise FusionError("identity_state must be registered, unknown, uncertain, or none")
            object.__setattr__(self, "identity_state", state)


@dataclass(frozen=True, slots=True)
class FusionDecision:
    state: str
    severity: int
    reason: str
    evidence_flags: tuple[str, ...]
    modalities: tuple[str, ...]
    subject_id: str | None = None

    def __post_init__(self) -> None:
        if self.state not in {"authorized", "unknown", "uncertain", "no_person", "sensor_fault"}:
            raise FusionError("unsupported fusion decision state")
        if not 0 <= self.severity <= 2:
            raise FusionError("fusion severity must be between 0 and 2")


def fuse_evidence(
    evidence: list[ModalityEvidence] | tuple[ModalityEvidence, ...],
    *,
    zone_event: bool = False,
) -> FusionDecision:
    """Fuse available modality results into one conservative decision.

    An identity is accepted only when all available identity-bearing evidence
    agrees on the same registered subject.  Unknown or conflicting evidence is
    never upgraded to an authorized identity.
    """
    if not evidence:
        raise FusionError("at least one modality evidence item is required")
    available = tuple(item for item in evidence if item.available)
    names = tuple(item.modality for item in available or evidence)
    flags: list[str] = [f"modalities:{','.join(names)}"]
    if not available:
        return FusionDecision(
            "sensor_fault",
            2,
            "no modality is available",
            tuple(flags + ["all_unavailable"]),
            names,
        )
    detected = tuple(item for item in available if item.person_detected)
    if not detected:
        return FusionDecision("no_person", 0, "no person detected", tuple(flags), names)

    registered = tuple(item for item in detected if item.identity_state == "registered")
    unknown = tuple(item for item in detected if item.identity_state == "unknown")
    uncertain = tuple(
        item for item in detected if item.identity_state in {"uncertain", None, "none"}
    )
    subjects = {item.subject_id for item in registered if item.subject_id}
    if len(subjects) == 1 and not unknown and not uncertain:
        flags.extend(("person_detected", "identity_agreement"))
        return FusionDecision(
            "authorized",
            0 if not zone_event else 1,
            "registered identity agreed",
            tuple(flags),
            names,
            next(iter(subjects)),
        )
    if registered and (unknown or uncertain or len(subjects) > 1):
        return FusionDecision(
            "uncertain",
            2 if zone_event else 1,
            "modalities disagree on identity",
            tuple(flags + ["modality_conflict"]),
            names,
        )
    if unknown:
        return FusionDecision(
            "unknown",
            2 if zone_event else 1,
            "person detected without an authorized identity",
            tuple(flags + ["unknown_identity"]),
            names,
        )
    return FusionDecision(
        "uncertain",
        1,
        "identity evidence is insufficient",
        tuple(flags + ["insufficient_identity_evidence"]),
        names,
    )


def decision_as_dict(decision: FusionDecision) -> dict[str, Any]:
    """Serialize a decision for event logs and telemetry."""
    return {
        "state": decision.state,
        "severity": decision.severity,
        "reason": decision.reason,
        "evidence_flags": list(decision.evidence_flags),
        "modalities": list(decision.modalities),
        "subject_id": decision.subject_id,
    }
