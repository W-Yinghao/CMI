"""Project A — observability audit engine.

`check_claim_allowed(claim)` applies the OACI hard rules (08 §4, 06, 02) and returns a
`Verdict`. The rules, by estimand × regime:

  source_* / leakage            : identifiable in R0+ (leakage stays a DIAGNOSTIC, never a
                                  risk/accuracy guarantee).
  target_prior                  : R0 → rejected (TOS-1/CE-R0-3) unless a target-law axiom is
                                  declared; R1/R2 → allowed iff C1∧C2∧C3 (TU-1), else CE-R1-2.
  target_concept                : R0/R1 → rejected (TU-2/CE-R1-1); R2 → allowed only as a
                                  bounded/tested residual under C4(tested)∧C5∧C6 + anchors/labels.
  target_risk / target_gain     : R0 → rejected (TOS-1/CE-R0-2) unless axiom; R1 → rejected
                                  (needs labels); R2 → allowed on the labeled slice / bounded.
  target_transport              : R2 only, iff C8∧C11 (MP-1), else CE-MP-1 / CE-C11-1.
  balanced_accuracy             : R0 source metric; R1 rejected unless oracle-marked; R2 needs
                                  target labels or oracle mark.

Design note: the engine trusts the caller's DECLARED `regime` as the observation operator; it
does not cross-check that `claim.observed` only names coordinates that regime actually contains
(06 §2). A mis-declared regime is therefore not caught here — a structured regime↔observed
integrity check is left as a future hardening (it does not affect the anti-overclaim guarantees,
which are keyed off the declared regime + estimand + contracts).
"""
from __future__ import annotations

from typing import FrozenSet, List, Tuple

from .registry import CONTRACTS, THEOREMS, FORBIDDEN_CLAIMS
from .schema import (Claim, ContractID, Estimand, ForbiddenClaimViolation,
                     ObservabilityReport, Regime, Verdict)

_SOURCE_ESTIMANDS = {Estimand.SOURCE_RISK, Estimand.SOURCE_LOSO,
                     Estimand.SOURCE_LEAKAGE_DIAGNOSTIC}


def split_checkable_uncheckable_contracts(
        regime: Regime, contracts: FrozenSet[ContractID]
) -> Tuple[FrozenSet[ContractID], FrozenSet[ContractID]]:
    """Partition invoked contracts into those data-supported in `regime` vs merely assumed."""
    checkable, uncheckable = set(), set()
    for c in contracts:
        (checkable if CONTRACTS[c].is_checkable(regime) else uncheckable).add(c)
    return frozenset(checkable), frozenset(uncheckable)


def attach_failure_certificates(claim: Claim) -> List[str]:
    """The certificate(s) that fire if this claim's declared contracts break."""
    certs: List[str] = []
    for c in sorted(claim.contracts, key=lambda x: x.value):
        cert = CONTRACTS[c].failure_certificate
        if cert not in certs:
            certs.append(cert)
    return certs


def _verdict(allowed, reason, regime, contracts, *, theorem=None, cert=None,
             missing=frozenset(), diagnostic=False, licenses_risk=False) -> Verdict:
    checkable, uncheckable = split_checkable_uncheckable_contracts(regime, contracts)
    return Verdict(allowed=allowed, reason=reason, theorem=theorem, failure_certificate=cert,
                   missing_contracts=frozenset(missing), checkable=checkable,
                   uncheckable=uncheckable, is_diagnostic=diagnostic,
                   licenses_target_risk=licenses_risk)


def check_claim_allowed(claim: Claim) -> Verdict:
    reg, est, cs = claim.regime, claim.estimand, claim.contracts

    # --- source-side quantities: identifiable in every regime (they are about the source) ----
    if est in _SOURCE_ESTIMANDS:
        return _verdict(True, "source-side quantity; identifiable in R0+ (source law observed)",
                        reg, cs)

    # --- leakage: a diagnostic, NEVER a risk / accuracy guarantee (CSC-R2, C5) ---------------
    if est == Estimand.LEAKAGE:
        return _verdict(True, "leakage is a diagnostic, not a risk/accuracy guarantee "
                              "(fidelity governed by C5)", reg, cs,
                        cert=CONTRACTS[ContractID.C5].failure_certificate, diagnostic=True)

    # --- target prior: declared axiom (all regimes, monotone) OR R1 + C1∧C2∧C3 (TU-1) -------
    if est == Estimand.TARGET_PRIOR:
        # A declared target-law axiom fixes P_T; by MONO-1 it licenses the claim in EVERY regime
        # (R2 ⊒ R0), so honour it before the regime split — but it is an axiom, not evidence.
        if claim.target_law_axiom:
            return _verdict(True, "declared target-law axiom (not source-only identification)",
                            reg, cs, theorem=None)
        if reg == Regime.R0:
            return _verdict(False, "target prior non-identifiable under R0 (TOS-1)", reg, cs,
                            cert="CE-R0-3")
        need = THEOREMS["TU-1"].required
        missing = need - cs
        if missing:
            return _verdict(False, "target prior needs C1∧C2∧C3 (TU-1)", reg, cs,
                            cert="CE-R1-2", missing=missing)
        return _verdict(True, "target prior identifiable under TU-1 (C1∧C2∧C3)", reg, cs,
                        theorem="TU-1", cert="CE-R1-2")

    # --- target concept: never from R0/R1 (TU-2); R2 only as a bounded/tested residual -------
    if est == Estimand.TARGET_CONCEPT:
        if reg in (Regime.R0, Regime.R1):
            return _verdict(False, "concept non-identifiable from unlabeled target (TU-2)",
                            reg, cs, cert="CE-R1-1")
        need = {ContractID.C4, ContractID.C5, ContractID.C6}
        missing = need - cs
        has_anchor = claim.has_target_labels or claim.has_anchors
        if missing or not has_anchor:
            return _verdict(False, "concept residual needs C4(tested)∧C5∧C6 + anchors/labels (R2)",
                            reg, cs, cert="CE-R1-1",
                            missing=frozenset(missing) if missing else frozenset())
        return _verdict(True, "bounded/tested C4-violation residual under C5∧C6 + anchors "
                              "(NOT a point value, NOT 'concept detected')", reg, cs, cert="CE-R1-1")

    # --- target risk / gain: declared axiom (monotone) OR R1 forbidden / R2 labeled slice ----
    if est in (Estimand.TARGET_RISK, Estimand.TARGET_GAIN):
        if claim.target_law_axiom:
            return _verdict(True, "declared target-law axiom (not source-only identification)",
                            reg, cs, licenses_risk=True)
        if reg == Regime.R0:
            return _verdict(False, "target risk/gain non-identifiable under R0 (TOS-1)", reg, cs,
                            cert="CE-R0-2")
        if reg == Regime.R1:
            return _verdict(False, "target risk/gain needs target labels (R2), not R1 unlabeled",
                            reg, cs, cert="CE-R0-2")
        if not (claim.has_target_labels or claim.oracle):
            return _verdict(False, "R2 target risk/gain needs a held-out labeled slice / oracle mark",
                            reg, cs, cert="CE-R0-2")
        return _verdict(True, "target risk/gain on the labeled slice / bounded under declared "
                              "contracts", reg, cs, licenses_risk=True)

    # --- transport: R2 + C8∧C11 (MP-1) -------------------------------------------------------
    if est == Estimand.TARGET_TRANSPORT:
        if reg != Regime.R2:
            return _verdict(False, "transport needs paired anchors (R2)", reg, cs, cert="CE-MP-1")
        need = THEOREMS["MP-1"].required
        missing = need - cs
        if missing:
            return _verdict(False, "transport needs C8∧C11 (MP-1)", reg, cs,
                            cert="CE-MP-1" if ContractID.C8 in missing else "CE-C11-1",
                            missing=missing)
        return _verdict(True, "transport identifiable (or bounded) under MP-1 (C8∧C11)", reg, cs,
                        theorem="MP-1", cert="CE-MP-1")

    # --- balanced accuracy: R0 source metric; R1 not identifiable; R2 needs labels/oracle ----
    if est == Estimand.BALANCED_ACCURACY:
        if reg == Regime.R0:
            return _verdict(True, "R0 source validation metric (not a target metric)", reg, cs)
        if reg == Regime.R1:
            if claim.oracle:
                return _verdict(True, "oracle / evaluation-only, explicitly marked (NOT adaptation "
                                      "evidence)", reg, cs)
            return _verdict(False, "balanced accuracy not identifiable from R1 unlabeled target",
                            reg, cs, cert="CE-R0-2")
        # R2
        if claim.has_target_labels or claim.oracle:
            return _verdict(True, "R2 held-out target-label / oracle metric (evaluation-only)",
                            reg, cs)
        return _verdict(False, "R2 balanced accuracy needs held-out target labels / oracle mark",
                        reg, cs, cert="CE-R0-2")

    # --- default deny (unknown estimand) -----------------------------------------------------
    return _verdict(False, "unknown estimand; default-deny", reg, cs)


def build_report(title: str, claims: List[Claim]) -> ObservabilityReport:
    report = ObservabilityReport(title=title)
    for claim in claims:
        report.add(claim, check_claim_allowed(claim))
    return report


def assert_forbidden_claims_not_made(report: ObservabilityReport) -> None:
    """Raise if any REJECTED claim is finalised as a conclusion (an overclaim slipped through).

    Claims flagged `conclusion=False` (demonstrations of what the audit rejects) are exempt.
    """
    offenders = [c for c, v in report.entries if v.rejected and c.conclusion]
    if offenders:
        names = ", ".join(f"{c.name} ({c.regime.value}:{c.estimand.value})" for c in offenders)
        raise ForbiddenClaimViolation(
            "rejected claim(s) finalised as conclusions: " + names)
