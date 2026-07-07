"""Fork 1 Tier-1 --- target-information split machinery + HARD leak gates (implementation stage; NO runs).

This module holds the leak-proof primitives the Tier-1 smoke driver would use IF a run were ever authorized:
  * stratified calibration/audit splits (disjoint by construction),
  * k-per-class selection from the CALIBRATION pool only (UNAVAILABLE if short; never touches audit),
  * `target_leak_structural_check` --- the real pre-run gate emitting TARGET_LEAK_STRUCTURAL_PASS or raising,
  * `DecisionContext` (has NO audit field) vs `AuditView` (audit only), enforcing "labels never both decide
    and evaluate" at the type level,
  * `budget_action` --- pure decision logic; B1 can never accept, B4 is DIAGNOSTIC only.

Nothing here reads real EEG. The heavy eraser/benefit computation lives behind the driver's execution lock.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

# Leak gates below enforce their invariants with unconditional `raise` (NOT `assert`), and this module refuses
# to import under optimized bytecode (`python -O` / PYTHONOPTIMIZE) where `assert` would be stripped. Belt and
# suspenders: even if a future edit reintroduced an assert, -O import would already have failed here.
if not __debug__:
    raise RuntimeError("target_info_splits: leak gates require assertions enabled; refuse to run under "
                       "-O / PYTHONOPTIMIZE (optimized bytecode strips asserts).")

TARGET_LEAK_TOKEN = "TARGET_LEAK_STRUCTURAL_PASS"
B1_ACTIONS = ("reject", "abstain", "request_labels")            # B1 can NEVER accept
DEPLOYABLE_ACCEPT = "accept"
BUDGET_FAMILIES = ("B0", "B1", "B2", "B3", "B4")
ACCEPT_SYNONYMS = ("accept", "deployable_accept", "accepted", "acc")
# positive ALLOWLIST of legal decision inputs (blocks any audit-label alias, not just two literals)
PERMITTED_DECISION_INPUTS = ("source_safety", "source_benefit_lcb", "unlabeled_target_mismatch",
                             "target_calibration_benefit_lcb", "same_k_random", "oracle_target_labels")


def _norm(s):
    return str(s).strip().lower()


def _family(budget):
    """Canonical budget family (B0..B4) from a budget key; raises on anything not in BUDGET_FAMILIES."""
    fam = str(budget).split("_", 1)[0]
    if fam not in BUDGET_FAMILIES:
        raise ValueError("unknown budget family %r (from %r)" % (fam, budget))
    return fam


# ------------------------------- calibration / audit splitting -------------------------------
def make_calibration_audit_splits(y, R, seed, calib_fraction=0.5):
    """Return R stratified (calibration_idx, audit_idx) pairs, disjoint by construction. Stratified by class so
    both splits keep every class. `y` = target-subject labels (audit split is NEVER used for any decision)."""
    y = np.asarray(y).astype(int)
    classes = np.unique(y)
    splits = []
    for r in range(R):
        rng = np.random.default_rng((seed + 1) * 100003 + r)
        cal, aud = [], []
        for c in classes:
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            n_cal = int(round(calib_fraction * len(idx)))
            # guarantee both sides non-empty when the class has >=2 trials
            n_cal = min(max(n_cal, 1), len(idx) - 1) if len(idx) >= 2 else len(idx)
            cal.extend(idx[:n_cal].tolist())
            aud.extend(idx[n_cal:].tolist())
        splits.append((np.array(sorted(cal), int), np.array(sorted(aud), int)))
    return splits


def effective_n_per_class(idx, y):
    y = np.asarray(y).astype(int)
    idx = np.asarray(idx, int)
    return {int(c): int((y[idx] == c).sum()) for c in np.unique(y)}


def select_k_per_class(calibration_idx, y, k, seed):
    """Pick k labeled trials PER CLASS from the CALIBRATION pool only. If any class has < k calibration trials,
    return (None, 'UNAVAILABLE', eff_n) WITHOUT ever drawing from audit. Returns (selected_idx, status, eff_n)."""
    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError("k must be an integer >= 1, got %r" % (k,))
    y = np.asarray(y).astype(int)
    calibration_idx = np.asarray(calibration_idx, int)
    eff = effective_n_per_class(calibration_idx, y)
    if any(eff[c] < k for c in eff):
        return None, "UNAVAILABLE", eff                         # never reuse audit labels to backfill
    rng = np.random.default_rng((seed + 1) * 7919 + k)
    sel = []
    for c in eff:
        pool = calibration_idx[y[calibration_idx] == c]
        rng.shuffle(pool)
        sel.extend(pool[:k].tolist())
    return np.array(sorted(sel), int), "OK", eff


# ------------------------------- decision / audit isolation (type level) -------------------------------
@dataclass(frozen=True)
class DecisionContext:
    """Everything a gate is allowed to see. Deliberately has NO audit field --- audit labels are structurally
    unavailable to any decision. Populated from source + target-CALIBRATION only."""
    budget: str
    source_safety_pass: bool
    benefit_thr: float = 0.01
    source_benefit_lcb: Optional[float] = None                  # B0
    cal_benefit_lcb: Optional[float] = None                     # B2/B3 (calibration split only)
    beats_random: bool = False                                  # same-k specificity control
    unlabeled_triage: Optional[str] = None                      # B1 (in B1_ACTIONS)


@dataclass(frozen=True)
class SourceContext:
    """Source-only data (labels allowed). Feeds eraser/head fit, source safety, source benefit."""
    Zs: np.ndarray
    ys: np.ndarray
    z_src: np.ndarray                                           # injected nuisance D_nuis (source-side)
    n_cls: int


@dataclass(frozen=True)
class CalibrationContext:
    """Target CALIBRATION data --- labels allowed for B2/B3 ONLY. Never contains audit indices/labels."""
    Zt_cal: np.ndarray
    yt_cal: np.ndarray
    n_per_class: dict


@dataclass(frozen=True)
class UnlabeledTargetContext:
    """Target features WITHOUT labels --- the only target signal B1 may read (triage)."""
    Zt: np.ndarray


@dataclass(frozen=True)
class AuditView:
    """The ONLY carrier of audit labels; passed exclusively to the final evaluation, never to a gate."""
    audit_idx: np.ndarray
    audit_y: np.ndarray


def b1_triage_action(mismatch_score, request_hi=0.5, reject_lo=0.02):
    """Unlabeled-target TRIAGE: map a source<->target mismatch score to reject/abstain/request_labels.
    Structurally cannot return 'accept' --- unlabeled evidence never licenses acceptance."""
    if mismatch_score >= request_hi:
        return "request_labels"
    if mismatch_score <= reject_lo:
        return "reject"
    return "abstain"


def budget_action(ctx: DecisionContext):
    """Pure decision logic. Never reads audit labels (DecisionContext has none). Invariants enforced with
    unconditional `raise` (not assert) so they survive optimized bytecode."""
    fam = _family(ctx.budget)                                   # raises on unknown family
    if fam == "B4":
        return "DIAGNOSTIC"                                     # oracle upper bound; NEVER a deployable accept
    if not ctx.source_safety_pass:
        return "reject"
    if fam == "B0":
        return DEPLOYABLE_ACCEPT if (ctx.source_benefit_lcb is not None
                                     and ctx.source_benefit_lcb > ctx.benefit_thr) else "abstain"
    if fam == "B1":
        act = ctx.unlabeled_triage
        if act not in B1_ACTIONS:                               # accept is forbidden for unlabeled target
            raise ValueError("B1 illegal action %r (accept is forbidden)" % (act,))
        return act
    if fam in ("B2", "B3"):
        # accept on the calibration benefit LCB; same-k specificity is a FLAG (accepted_specific vs
        # accepted_non_specific), computed by the caller -- it does not block the accept.
        if ctx.cal_benefit_lcb is not None and ctx.cal_benefit_lcb > ctx.benefit_thr:
            return DEPLOYABLE_ACCEPT
        return "abstain"
    raise ValueError("unhandled budget %r" % (ctx.budget,))


# ------------------------------- the HARD pre-run structural gate -------------------------------
def target_leak_structural_check(splits, budget_specs):
    """Enforce the leak-proofing invariants with unconditional `raise`; return TARGET_LEAK_STRUCTURAL_PASS or
    raise AssertionError. `splits` = iterable of (calibration_idx, audit_idx); `budget_specs` = {budget:
    {allowed_actions, decision_inputs, diagnostic_only?}} (from the driver config)."""
    def _fail(msg):
        raise AssertionError(msg)
    for r, (cal, aud) in enumerate(splits):
        cal_s, aud_s = set(int(i) for i in cal), set(int(j) for j in aud)
        if not cal_s.isdisjoint(aud_s):
            _fail("split %d: calibration ∩ audit != empty (%d overlap)" % (r, len(cal_s & aud_s)))
    permitted = set(_norm(x) for x in PERMITTED_DECISION_INPUTS)
    accept_syn = set(_norm(x) for x in ACCEPT_SYNONYMS)
    for b, spec in budget_specs.items():
        fam = _family(b)                                        # raises on unknown family
        allowed = set(_norm(x) for x in spec.get("allowed_actions", []))
        inputs = set(_norm(x) for x in spec.get("decision_inputs", []))
        # POSITIVE allowlist: every decision input must be explicitly permitted (blocks any audit-label alias),
        # plus a belt-and-suspenders substring check on 'audit'.
        illegal = inputs - permitted
        if illegal:
            _fail("%s: illegal decision input(s) %s (not in allowlist)" % (b, sorted(illegal)))
        if any("audit" in x for x in inputs):
            _fail("%s: audit label may never be a decision input" % b)
        if fam == "B1":
            if allowed & accept_syn:
                _fail("B1 must not allow any accept synonym (%s)" % sorted(allowed & accept_syn))
            if not allowed <= set(_norm(x) for x in B1_ACTIONS):
                _fail("B1 actions must be a subset of %s" % (B1_ACTIONS,))
        if fam == "B4":
            if spec.get("diagnostic_only") is not True:
                _fail("B4 must be diagnostic_only")
            if allowed != {"diagnostic"}:
                _fail("B4 allowed_actions must be exactly {DIAGNOSTIC}, got %s" % sorted(allowed))
    return TARGET_LEAK_TOKEN


# ------------------------------- deployable-accept accounting (B4 excluded) -------------------------------
def is_deployable_budget(budget):
    """B4 oracle is diagnostic; it must be excluded from true/false deployable accept accounting.
    Keyed on the canonical family (raises on unknown family) so a mis-prefixed rename cannot silently pass."""
    return _family(budget) != "B4"
