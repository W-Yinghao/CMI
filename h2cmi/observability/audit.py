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

Design note: `validate_observed_coordinates` (Patch A) hard-rejects a claim whose declared
`regime` conflicts with its observed coordinates. It is DENY-BY-DEFAULT over a registered
coordinate vocabulary: an unregistered/mis-spelled token is itself a mismatch (fail-loud), so a
disguised target coordinate cannot slip past by renaming. Target LABELS may appear at R0/R1 only
when the claim is oracle/evaluation-only; anchors are R2-only regardless of the oracle mark.
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
             missing=frozenset(), diagnostic=False, licenses_risk=False,
             identifiable=None, reportable=None) -> Verdict:
    checkable, uncheckable = split_checkable_uncheckable_contracts(regime, contracts)
    if reportable is None:
        reportable = allowed                     # a rejected claim is not reportable
    if identifiable is None:
        identifiable = False                     # FAIL-CLOSED: every licensed path must opt in explicitly
    return Verdict(allowed=allowed, reason=reason, theorem=theorem, failure_certificate=cert,
                   missing_contracts=frozenset(missing), checkable=checkable,
                   uncheckable=uncheckable, is_diagnostic=diagnostic,
                   licenses_target_risk=licenses_risk, identifiable=identifiable,
                   reportable=reportable)


# Registered observed-coordinate vocabulary (Patch A): each token -> the minimum regime LEVEL
# whose operator (06 §2) contains it. 0 = source (R0+); 1 = target-UNLABELED (R1+); 2 = target
# labels / anchors (R2+). DENY-BY-DEFAULT: an unregistered token is itself a mismatch (fail-loud),
# so a disguised / mis-spelled target coordinate cannot slip through by renaming.
_COORDINATE_LEVEL = {
    # level 0 — source coordinates
    "X_s": 0, "Y_s": 0, "D_s": 0, "Z_s": 0, "source_metrics": 0, "source_leakage": 0,
    "source_loso": 0, "source_gate_features": 0,
    # level 1 — target-unlabeled coordinates
    "X_T": 1, "Z_T": 1, "D_T": 1, "target_X": 1, "target_marginal": 1, "target_support": 1,
    "pseudo_labels": 1,
    # level 2 — target labels (oracle-exemptible for EVALUATION only)
    "Y_T": 2, "target_y": 2, "target_labels": 2, "heldout_target_labels": 2,
    # level 2 — anchors (NOT oracle-exemptible: genuinely R2 observation, not eval labels)
    "anchors": 2, "paired_sessions": 2, "calibration_pairs": 2,
}
# only target-LABEL tokens may appear at R0/R1 as oracle/evaluation-only; anchors may not.
_ORACLE_EXEMPTIBLE = {"Y_T", "target_y", "target_labels", "heldout_target_labels"}


def validate_observed_coordinates(claim: Claim) -> List[str]:
    """Patch A: DENY-BY-DEFAULT integrity check that `claim.observed` fits the DECLARED regime.

    Each observed token must be a REGISTERED coordinate (unknown/mis-spelled token -> mismatch,
    fail-loud). A coordinate whose regime LEVEL exceeds the declared regime is a mismatch, UNLESS
    it is a target-LABEL token and the claim is marked oracle/evaluation-only. Anchors (R2-only)
    are never oracle-exempted. Returns a list of problem strings (empty = consistent).
    """
    problems: List[str] = []
    for tok in claim.observed:
        lvl = _COORDINATE_LEVEL.get(tok)
        if lvl is None:
            problems.append(f"unknown observed coordinate '{tok}' "
                            f"(not a registered R0/R1/R2 coordinate)")
            continue
        if lvl <= claim.regime.level:
            continue                                   # within the declared regime
        if tok in _ORACLE_EXEMPTIBLE and claim.oracle:
            continue                                   # eval-only target labels permitted at R0/R1
        problems.append(f"{claim.regime.value} claim observes '{tok}' "
                        f"(a level-{lvl} coordinate; regime level {claim.regime.level})")
    return problems


def check_claim_allowed(claim: Claim) -> Verdict:
    reg, est, cs = claim.regime, claim.estimand, claim.contracts

    # --- Patch A: regime <-> observed-coordinate integrity (hard reject on mismatch) ---------
    problems = validate_observed_coordinates(claim)
    if problems:
        return _verdict(False, "regime-observed mismatch: " + "; ".join(problems), reg, cs)

    # --- source-side quantities: identifiable in every regime (they are about the source) ----
    if est in _SOURCE_ESTIMANDS:
        return _verdict(True, "source-side quantity; identifiable in R0+ (source law observed)",
                        reg, cs, identifiable=True)

    # --- leakage: a diagnostic, NEVER a risk / accuracy guarantee (CSC-R2, C5) ---------------
    if est == Estimand.LEAKAGE:
        return _verdict(True, "leakage is a diagnostic, not a risk/accuracy guarantee "
                              "(fidelity governed by C5)", reg, cs,
                        cert=CONTRACTS[ContractID.C5].failure_certificate, diagnostic=True,
                        identifiable=True)   # identifiable as a diagnostic value (up to C5)

    # --- target prior: declared axiom (all regimes, monotone) OR R1 + C1∧C2∧C3 (TU-1) -------
    if est == Estimand.TARGET_PRIOR:
        # A declared target-law axiom fixes P_T; by MONO-1 it licenses the claim in EVERY regime
        # (R2 ⊒ R0), so honour it before the regime split — but it is an AXIOM, not evidence.
        if claim.target_law_axiom:
            return _verdict(True, "declared target-law axiom (not source-only identification)",
                            reg, cs, theorem=None, identifiable=False, reportable=True)
        if reg == Regime.R0:
            return _verdict(False, "target prior non-identifiable under R0 (TOS-1)", reg, cs,
                            cert="CE-R0-3")
        need = THEOREMS["TU-1"].required
        missing = need - cs
        if missing:
            return _verdict(False, "target prior needs C1∧C2∧C3 (TU-1)", reg, cs,
                            cert="CE-R1-2", missing=missing)
        return _verdict(True, "target prior identifiable under TU-1 (C1∧C2∧C3)", reg, cs,
                        theorem="TU-1", cert="CE-R1-2", identifiable=True)

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
        # a bounded/tested residual: reportable, but NOT a point-identified concept
        return _verdict(True, "bounded/tested C4-violation residual under C5∧C6 + anchors "
                              "(NOT a point value, NOT 'concept detected')", reg, cs, cert="CE-R1-1",
                        identifiable=False, reportable=True)

    # --- target risk / gain: axiom OR oracle eval-only OR R2 labeled slice; else rejected -----
    if est in (Estimand.TARGET_RISK, Estimand.TARGET_GAIN):
        if claim.target_law_axiom:
            return _verdict(True, "declared target-law axiom (not source-only identification)",
                            reg, cs, licenses_risk=True, identifiable=False, reportable=True)
        # an oracle / evaluation-only benchmark number is REPORTABLE but NOT identified from R0/R1
        if claim.oracle and reg in (Regime.R0, Regime.R1):
            return _verdict(True, "oracle/evaluation-only; not adaptation evidence; not "
                                  "identifiable from R0/R1", reg, cs, identifiable=False,
                            reportable=True, licenses_risk=False)
        if reg == Regime.R0:
            return _verdict(False, "target risk/gain non-identifiable under R0 (TOS-1)", reg, cs,
                            cert="CE-R0-2")
        if reg == Regime.R1:
            return _verdict(False, "target risk/gain needs target labels (R2) or an oracle/eval "
                                  "mark, not R1 unlabeled", reg, cs, cert="CE-R0-2")
        # R2
        if claim.has_target_labels:
            return _verdict(True, "target risk/gain on the R2 held-out labeled slice", reg, cs,
                            licenses_risk=True, identifiable=True, reportable=True)
        if claim.oracle:
            return _verdict(True, "oracle/evaluation-only R2 metric; not identified beyond the "
                                  "labeled slice", reg, cs, identifiable=False, reportable=True)
        return _verdict(False, "R2 target risk/gain needs a held-out labeled slice / oracle mark",
                        reg, cs, cert="CE-R0-2")

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
                        theorem="MP-1", cert="CE-MP-1", identifiable=True)

    # --- balanced accuracy: oracle eval-only; R0 source metric; R1 non-id; R2 labeled ---------
    if est == Estimand.BALANCED_ACCURACY:
        # an oracle target bAcc (strict-DG / TTA eval) is REPORTABLE but not identified from R0/R1
        if claim.oracle and reg in (Regime.R0, Regime.R1):
            return _verdict(True, "oracle/evaluation-only target bAcc; not adaptation evidence; "
                                  "not identifiable from R0/R1", reg, cs, identifiable=False,
                            reportable=True)
        if reg == Regime.R0:
            return _verdict(True, "R0 source validation metric (not a target metric)", reg, cs,
                            identifiable=True)
        if reg == Regime.R1:
            return _verdict(False, "balanced accuracy not identifiable from R1 unlabeled target "
                                  "(mark oracle/evaluation-only to report)", reg, cs, cert="CE-R0-2")
        # R2
        if claim.has_target_labels:
            return _verdict(True, "R2 held-out target-label metric (labeled slice)", reg, cs,
                            identifiable=True, reportable=True)
        if claim.oracle:
            return _verdict(True, "R2 oracle/evaluation-only metric", reg, cs,
                            identifiable=False, reportable=True)
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
