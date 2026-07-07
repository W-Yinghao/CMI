"""Project A — canonical contract & theorem registry (machine form of `02_…` and `06_…`).

Encodes:
  * CONTRACTS  — C1..C12: one-line statement + per-regime checkability + failure certificate,
                 matching `notes/project_A_observability/02_contract_taxonomy.md`.
  * THEOREMS   — TOS-1 / TU-1 / TU-2 / PD-1 / MP-1 / MONO-1: required contracts + certificate.

Checkability status per (contract, regime) is one of "no" / "partial" / "yes". It is
MONOTONE NON-DECREASING along R0→R1→R2 (MONO-1): a coordinate checkable in a coarser regime
stays checkable in a finer one. This invariant is asserted in the smoke/self-test below.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Optional

from .schema import ContractID as C
from .schema import Regime

_YES, _PARTIAL, _NO = "yes", "partial", "no"


class ContractSpec:
    __slots__ = ("cid", "name", "statement", "checkable", "failure_certificate")

    def __init__(self, cid: C, name: str, statement: str,
                 checkable: Dict[Regime, str], failure_certificate: str):
        self.cid = cid
        self.name = name
        self.statement = statement
        self.checkable = checkable                    # Regime -> {"yes","partial","no"}
        self.failure_certificate = failure_certificate

    def is_checkable(self, regime: Regime) -> bool:
        """A contract is checkable in a regime iff its coordinate is observed there (yes/partial)."""
        return self.checkable[regime] != _NO


def _ck(r0: str, r1: str, r2: str) -> Dict[Regime, str]:
    return {Regime.R0: r0, Regime.R1: r1, Regime.R2: r2}


CONTRACTS: Dict[C, ContractSpec] = {
    C.C1: ContractSpec(C.C1, "class support overlap",
                       "supp p_T(z|y) ⊆ supp p_S(z|y) ∀y",
                       _ck(_NO, _PARTIAL, _YES), "CE-C1-1"),
    C.C2: ContractSpec(C.C2, "shared class-conditional geometry",
                       "p_T(z|y) = p_S(z|y) = p_ref(z|y) ∀y",
                       _ck(_NO, _NO, _YES), "CE-R1-1"),
    C.C3: ContractSpec(C.C3, "mixture / full-rank identifiability",
                       "B_{z,y}=p_ref(z|y) full column rank; confusion C invertible",
                       _ck(_YES, _YES, _YES), "CE-R1-2"),
    C.C4: ContractSpec(C.C4, "stable label mechanism / no target concept shift",
                       "p_T(Y|Y*,D)=p_S(Y|Y*,D) and p_T(Y*|X)=p_S(Y*|X)",
                       _ck(_NO, _NO, _PARTIAL), "CE-R1-1"),
    C.C5: ContractSpec(C.C5, "critic / estimator sufficiency",
                       "q_ψ → p_θ(D|Z,Y) (Step-A converged) — estimator layer, not observability",
                       _ck(_PARTIAL, _PARTIAL, _PARTIAL), "P0-2"),
    C.C6: ContractSpec(C.C6, "representation sufficiency / predictive span",
                       "I(Y;X)=I(Y;Z); precond. each domain spans ≥2 classes",
                       _ck(_PARTIAL, _PARTIAL, _PARTIAL), "P0-4"),
    C.C7: ContractSpec(C.C7, "reference-prior / GLS reweighting availability",
                       "per-domain π_d(y) known; w_d(y)=π*(y)/π_d(y) ⇒ Ĩ(Y;D)=0 (source-side)",
                       _ck(_PARTIAL, _PARTIAL, _YES), "CE-R1-2"),
    C.C8: ContractSpec(C.C8, "low-dimensional invertible transport",
                       "near-identity affine (‖A−I‖ small), full-rank, overlap",
                       _ck(_NO, _PARTIAL, _YES), "CE-MP-1"),
    C.C9: ContractSpec(C.C9, "source-to-target safety transfer",
                       "inner-LOSO source gain distribution transfers to unseen targets",
                       _ck(_PARTIAL, _PARTIAL, _YES), "CE-R0-2"),   # CE-R0-2 is the TOS-1 gain-sign-flip cert (02 §5)
    C.C10: ContractSpec(C.C10, "zero-Bayes / D⊥Y|Z escape",
                        "H(Y|Z)=0 sufficient not necessary; correct condition D⊥Y|Z",
                        _ck(_PARTIAL, _PARTIAL, _PARTIAL), "P0-3"),
    C.C11: ContractSpec(C.C11, "anchor validity",
                        "anchors pair samples from the same latent mechanism (no fake pairing)",
                        _ck(_NO, _NO, _PARTIAL), "CE-C11-1"),
    C.C12: ContractSpec(C.C12, "domain-factor separability",
                        "each D_j validly typed acquisition vs label-mechanism; determines_label correct",
                        _ck(_YES, _YES, _YES), "P0-4"),
    C.C14: ContractSpec(C.C14, "declared deployment prior / utility weighting",
                        "deployer DECLARES the class prior π* / utility weights under which target "
                        "risk/gain is evaluated (external operating condition, not source-only estimated)",
                        _ck(_NO, _PARTIAL, _YES), "CE-R1-2"),
    C.C15: ContractSpec(C.C15, "declared prior-uncertainty set / robustness criterion",
                        "deployer DECLARES a set of admissible operating priors (e.g. an L1 ball around a "
                        "reference prior) and asks for worst-/best-case gain over that set",
                        _ck(_NO, _PARTIAL, _YES), "CE-R1-2"),
}
# NOTE: C13 (class-balanced calibration DESIGN, Step 17) is intentionally documentation-only and is NOT
# registered here — it is a data-acquisition design contract, not a machine claim-gating contract. C14
# (Step 18) and C15 (Step 19) ARE registered because prior-weighted / robust-prior-weighted gain claims
# are gated on them (see audit.PRIOR_WEIGHTED_GAIN / ROBUST_PRIOR_WEIGHTED_GAIN).


class TheoremSpec:
    __slots__ = ("name", "required", "certificate", "regime_min", "note")

    def __init__(self, name: str, required: FrozenSet[C], certificate: Optional[str],
                 regime_min: Regime, note: str):
        self.name = name
        self.required = required
        self.certificate = certificate
        self.regime_min = regime_min
        self.note = note


THEOREMS: Dict[str, TheoremSpec] = {
    "TOS-1": TheoremSpec("TOS-1", frozenset(), "CE-R0-2", Regime.R0,
                         "source-only ceiling: target functionals non-identifiable under R0"),
    "TU-1": TheoremSpec("TU-1", frozenset({C.C1, C.C2, C.C3}), "CE-R1-2", Regime.R1,
                        "target prior identifiable under R1 + C1∧C2∧C3"),
    "TU-2": TheoremSpec("TU-2", frozenset(), "CE-R1-1", Regime.R1,
                        "concept non-identifiable from unlabeled target"),
    "PD-1": TheoremSpec("PD-1", frozenset({C.C7}), None, Regime.R0,  # positive result: no CE (verify_resolution.py)
                        "prior-decoupled additive relation (source-side; ID-1 + reweighting)"),
    "MP-1": TheoremSpec("MP-1", frozenset({C.C8, C.C11}), "CE-MP-1", Regime.R2,
                        "minimal-paired transport identifiable under C8∧C11"),
    "MONO-1": TheoremSpec("MONO-1", frozenset(), "CE-MONO-1", Regime.R0,
                          "information monotonicity; source breadth ≠ target observation"),
}

# The canonical forbidden overclaims (05 §6) — the report asserts none is made as a conclusion.
FORBIDDEN_CLAIMS = (
    "target concept shift detected from unlabeled target",
    "source-only target safety certified",
    "CMI/leakage guarantees accuracy",
    "GLS gives the target prior source-only",
    "R0 source metric reported as a target risk/gain/concept guarantee",
    "R1 unlabeled-target balanced accuracy reported as identifiable target metric",
    "declared deployment prior identifies the actual target prior",   # C14 must not become TU-1
    "prior-weighted gain reported as holding under all target priors",  # sign is prior-dependent
    "gain reported as prior-robust without a declared prior-uncertainty set",  # C15 must be declared
    "declared prior-uncertainty set identifies the actual target prior",  # C15 must not become TU-1
    # Step-20 closeout: the three terminal headline overclaims the project must never make
    "unlabeled offline-TTA is safe to deploy",                        # Step 15/16/19 refute this
    "the target prior is identified from R0/R1",                      # TU-1 boundary
    "prior-robust adaptation benefit exists under honest prior uncertainty",  # Step 19: none at margin
)


def check_monotone_checkability() -> bool:
    """MONO-1 invariant: checkability is non-decreasing R0→R1→R2 for every contract."""
    rank = {_NO: 0, _PARTIAL: 1, _YES: 2}
    for spec in CONTRACTS.values():
        r0, r1, r2 = (spec.checkable[Regime.R0], spec.checkable[Regime.R1],
                      spec.checkable[Regime.R2])
        if not (rank[r0] <= rank[r1] <= rank[r2]):
            return False
    return True
