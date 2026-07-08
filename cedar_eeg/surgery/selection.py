"""CEDAR surgery selection and abstention logic."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

from cedar_eeg.config import P0Thresholds


class SurgeryDecision(str, Enum):
    ACCEPT = "ACCEPT"
    REPORT_ONLY = "REPORT_ONLY"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class SurgeryCandidate:
    name: str
    dropped_units: tuple[int, ...]
    leakage_before: float
    leakage_after: float
    source_bacc_before: float
    source_bacc_after: float
    target_bacc_before: float | None = None
    target_bacc_after: float | None = None
    r3_before: float = 0.0
    r3_after: float = 0.0
    stability: float = 1.0
    random_control_drop_frac: float = 0.0

    @property
    def leakage_drop(self) -> float:
        return float(self.leakage_before - self.leakage_after)

    @property
    def leakage_drop_frac(self) -> float:
        denom = max(abs(self.leakage_before), 1e-8)
        return float(self.leakage_drop / denom)

    @property
    def source_bacc_drop(self) -> float:
        return float(self.source_bacc_before - self.source_bacc_after)

    @property
    def target_bacc_drop(self) -> float | None:
        if self.target_bacc_before is None or self.target_bacc_after is None:
            return None
        return float(self.target_bacc_before - self.target_bacc_after)

    @property
    def r3_delta(self) -> float:
        return float(self.r3_after - self.r3_before)

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        out.update(
            leakage_drop=self.leakage_drop,
            leakage_drop_frac=self.leakage_drop_frac,
            source_bacc_drop=self.source_bacc_drop,
            target_bacc_drop=self.target_bacc_drop,
            r3_delta=self.r3_delta,
        )
        return out


def score_candidate(candidate: SurgeryCandidate, eps: float = 1e-8) -> float:
    """CEDAR utility: leakage drop per source-risk cost, stabilized by stability."""

    risk = max(0.0, candidate.source_bacc_drop)
    return float(candidate.leakage_drop * candidate.stability / (eps + risk))


def target_eval_warnings(candidate: SurgeryCandidate, thresholds: P0Thresholds) -> list[str]:
    """Diagnostic-only target-label warnings.

    These warnings must not affect candidate selection. They exist to support
    post-hoc scientific continuation decisions when held-out target labels are
    available for evaluation.
    """

    if candidate.target_bacc_drop is None:
        return []
    if candidate.target_bacc_drop > thresholds.max_target_bacc_drop:
        return [
            f"target_bacc_drop {candidate.target_bacc_drop:.3f} > "
            f"{thresholds.max_target_bacc_drop:.3f} (evaluation-only)"
        ]
    return []


def decide_p0(
    candidate: SurgeryCandidate,
    thresholds: P0Thresholds,
) -> tuple[SurgeryDecision, list[str]]:
    """Apply the frozen P0 source-side gate.

    Target labels, when present in the JSON, are diagnostic-only and are not
    permitted to change ACCEPT/ABSTAIN/REPORT_ONLY.
    """

    reasons: list[str] = []
    if candidate.leakage_drop_frac < thresholds.min_leakage_drop_frac:
        reasons.append(
            f"leakage_drop_frac {candidate.leakage_drop_frac:.3f} < "
            f"{thresholds.min_leakage_drop_frac:.3f}"
        )
    if candidate.source_bacc_drop > thresholds.max_source_bacc_drop:
        reasons.append(
            f"source_bacc_drop {candidate.source_bacc_drop:.3f} > "
            f"{thresholds.max_source_bacc_drop:.3f}"
        )
    if candidate.r3_delta > thresholds.max_r3_delta:
        reasons.append(f"r3_delta {candidate.r3_delta:.3f} > {thresholds.max_r3_delta:.3f}")
    if abs(candidate.random_control_drop_frac) > thresholds.max_random_control_drop_frac:
        reasons.append(
            f"random_control_drop_frac {candidate.random_control_drop_frac:.3f} exceeds "
            f"+/-{thresholds.max_random_control_drop_frac:.3f}"
        )
    if candidate.stability < thresholds.min_stability:
        reasons.append(f"stability {candidate.stability:.3f} < {thresholds.min_stability:.3f}")

    if not reasons:
        return SurgeryDecision.ACCEPT, []
    if candidate.leakage_drop_frac >= thresholds.min_leakage_drop_frac:
        return SurgeryDecision.ABSTAIN, reasons
    return SurgeryDecision.REPORT_ONLY, reasons
