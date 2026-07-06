"""Project A — observability audit schema.

Types for machine-checking the claim ledger of
`notes/project_A_observability/08_experimental_protocol.md` against the OACI identifiability
rules (`06_…`) and the contract taxonomy (`02_…`). Pure-Python, no heavy deps.

A *claim* declares (regime, estimand, observed coordinates, contracts, estimator, flags); the
audit engine (`audit.py`) returns a *verdict* (allowed / rejected + why), and a
*report* (`report.py`) serialises the ledger with the 08 §5 required fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Tuple


class Regime(str, Enum):
    """Observation regimes; ordered R0 ⊑ R1 ⊑ R2 (refinement, 06 §2)."""
    R0 = "R0"   # source-only
    R1 = "R1"   # target-unlabeled
    R2 = "R2"   # minimal-paired

    @property
    def level(self) -> int:
        return {"R0": 0, "R1": 1, "R2": 2}[self.value]

    def refines(self, other: "Regime") -> bool:
        """True iff self observes at least as much as `other` (self ⊒ other)."""
        return self.level >= other.level


class ContractID(str, Enum):
    C1 = "C1"; C2 = "C2"; C3 = "C3"; C4 = "C4"; C5 = "C5"; C6 = "C6"
    C7 = "C7"; C8 = "C8"; C9 = "C9"; C10 = "C10"; C11 = "C11"; C12 = "C12"


class Estimand(str, Enum):
    """Canonical estimand kinds an experiment may claim."""
    # source-side (R0-identifiable)
    SOURCE_RISK = "source_risk"
    SOURCE_LOSO = "source_loso"
    SOURCE_LEAKAGE_DIAGNOSTIC = "source_leakage_diagnostic"
    LEAKAGE = "leakage"                      # generic leakage diagnostic (I(Z;D|Y), I(Y;D|Z))
    # target-side (need R1/R2 + contracts)
    TARGET_RISK = "target_risk"
    TARGET_GAIN = "target_gain"
    TARGET_PRIOR = "target_prior"
    TARGET_CONCEPT = "target_concept"
    TARGET_TRANSPORT = "target_transport"
    # special reporting rule
    BALANCED_ACCURACY = "balanced_accuracy"


@dataclass(frozen=True)
class Claim:
    """A single thing an experiment asserts, with its declared observability context."""
    name: str
    regime: Regime
    estimand: Estimand
    observed: Tuple[str, ...] = ()                       # observed coordinates used
    contracts: FrozenSet[ContractID] = field(default_factory=frozenset)
    estimator: str = ""
    target_law_axiom: bool = False                       # declared external target-law axiom (R0 escape)
    oracle: bool = False                                 # metric is oracle / evaluation-only (Tier-1 / R2 held-out)
    has_target_labels: bool = False                      # R2 labeled slice available
    has_anchors: bool = False                            # R2 paired/transport/calibration anchors available
    conclusion: bool = True                              # is this a finalised conclusion? (guarded)
    # evaluation EVIDENCE (metric values). Does NOT change allowed/identifiable/reportable — the
    # audit verdict is keyed off (regime, estimand, contracts, flags), never the numbers.
    metric_payload: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # normalise a plain set / list of contracts to a frozenset of ContractID
        if not isinstance(self.contracts, frozenset):
            object.__setattr__(self, "contracts", frozenset(self.contracts))
        if isinstance(self.observed, list):
            object.__setattr__(self, "observed", tuple(self.observed))


@dataclass(frozen=True)
class Verdict:
    """The audit decision for one claim."""
    allowed: bool
    reason: str
    theorem: Optional[str] = None                        # licensing theorem, if allowed under contract
    failure_certificate: Optional[str] = None            # the CE/P0 that fires if a contract breaks
    missing_contracts: FrozenSet[ContractID] = field(default_factory=frozenset)
    checkable: FrozenSet[ContractID] = field(default_factory=frozenset)
    uncheckable: FrozenSet[ContractID] = field(default_factory=frozenset)
    is_diagnostic: bool = False                          # True for leakage-style diagnostics
    licenses_target_risk: bool = False                   # True only when an allowed target risk/gain
    # A metric can be REPORTABLE (an oracle/evaluation-only benchmark number) without being
    # IDENTIFIABLE (a target functional pinned down by the regime's observation under OA-0).
    reportable: bool = True                              # may appear in a results table
    identifiable: bool = False                           # pinned down by (regime, contracts) under OA-0

    @property
    def rejected(self) -> bool:
        return not self.allowed


@dataclass
class ObservabilityReport:
    """A ledger of (claim, verdict) pairs plus the 08 §5 report fields."""
    title: str
    entries: List[Tuple[Claim, Verdict]] = field(default_factory=list)

    def add(self, claim: Claim, verdict: Verdict) -> None:
        self.entries.append((claim, verdict))

    def allowed(self) -> List[Tuple[Claim, Verdict]]:
        return [(c, v) for c, v in self.entries if v.allowed]

    def rejected(self) -> List[Tuple[Claim, Verdict]]:
        return [(c, v) for c, v in self.entries if v.rejected]


class ForbiddenClaimViolation(Exception):
    """Raised when a rejected claim is finalised as a conclusion (an overclaim slipped through)."""
