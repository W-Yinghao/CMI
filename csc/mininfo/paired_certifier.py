"""
CSC Route B3 — paired minimal-information certifier (information ladder; target-internal).

States (no COVARIATE_COMPATIBLE — B3's positive claim is concept confirmation, not covariate stability):
  CONCEPT_CONFIRMED                  paired conditional-change test rejects (p <= alpha) with valid pairs
  NO_CONCEPT_EVIDENCE_AFTER_PAIR_AUDIT  enough paired labels audited, test did NOT reject
  NEED_MORE_LABELS                   pairs valid but label budget too small to decide a non-rejection
  INVALID_PAIR_STRUCTURE             too few paired subjects / conditions to run a within-subject contrast
  UNIDENTIFIABLE                     m == 0 (Z-only triage cannot confirm; theory requires abstention)

`m` = number of paired target subjects whose labels are queried (the minimal information). `decide_n`
is the pre-registered budget at/above which a non-rejection is reported as NO_CONCEPT_EVIDENCE rather
than NEED_MORE_LABELS.
"""
from __future__ import annotations

import numpy as np

from .paired_conditional_test import paired_conditional_change_test, paired_validity

CONCEPT_CONFIRMED = "CONCEPT_CONFIRMED"
NO_CONCEPT_EVIDENCE = "NO_CONCEPT_EVIDENCE_AFTER_PAIR_AUDIT"
NEED_MORE_LABELS = "NEED_MORE_LABELS"
INVALID_PAIR = "INVALID_PAIR_STRUCTURE"
UNIDENTIFIABLE = "UNIDENTIFIABLE"


def certify_paired(Z, Y, D, G, m, alpha=0.05, decide_n=20, min_pairs=4, min_confirm_pairs=20,
                   h1_basis="pc", condition_coding="centered", rank=3, C=None, n_boot=200, seed=0):
    # ADOPTED METHOD = "pc_centered": h1_basis="pc" + condition_coding="centered" (+-0.5). B3-P2.2:
    # full_z+"01" was a PRE-DECLARED NEGATIVE (clean ~0.96/covariate ~1.00 @ m=20 = type-I catastrophe);
    # the centered coding fixed that AND unlocked pure_conditional on pc (0->0.75 @ m30), but the full_z
    # BASIS was NOT promoted (looser controls). full_z retained DIAGNOSTIC only. No finite-sample control
    # is claimed -- development-clean only. See notes/CSC_B_P22_R1c_RESULT.md.
    """Query `m` paired target subjects' labels and run the within-subject conditional-change test.
    `min_confirm_pairs` (B3-P2.2) forbids CONCEPT_CONFIRMED below a minimum audit size (small-m audits
    are unstable) -> NEED_MORE_LABELS, logging `would_confirm_without_min_pairs`. Returns a dict with the
    certificate state + per-cluster log fields."""
    Z = np.asarray(Z, float); Y = np.asarray(Y); D = np.asarray(D); G = np.asarray(G)
    paired_subs = [s for s in np.unique(G) if len(np.unique(D[G == s])) >= 2]
    log = dict(m=int(m), n_paired_available=len(paired_subs), n_queried=0, valid=False,
               p_value=float("nan"), T=float("nan"), reason="")

    # Z-only triage: no labels -> cannot confirm a conditional change (theory: must abstain)
    if m <= 0:
        log.update(state=UNIDENTIFIABLE, reason="m=0: Z-only triage cannot confirm")
        return log
    # pair structure must exist at all
    if len(paired_subs) < min_pairs:
        log.update(state=INVALID_PAIR, reason=f"{len(paired_subs)} paired subjects < {min_pairs}")
        return log

    rng = np.random.default_rng(seed)
    pick = rng.choice(paired_subs, size=min(int(m), len(paired_subs)), replace=False)
    mask = np.isin(G, pick)
    Zq, Yq, Dq, Gq = Z[mask], Y[mask], D[mask], G[mask]
    # explicit label-budget accounting (B3-P2.1): m = queried paired SUBJECTS; the audit uses ALL their
    # (subject,condition) cell labels -> record both the subject count and the labelled-cell/epoch counts.
    n_cells = len({(int(s), int(c)) for s, c in zip(Gq, Dq)})
    log.update(n_queried=int(len(pick)), n_queried_subjects=int(len(pick)),
               n_labeled_subject_conditions=int(n_cells), n_labeled_epochs=int(len(Yq)))
    ok, reason = paired_validity(Yq, Dq, Gq, min_subjects=min(min_pairs, len(pick)))
    if not ok:
        log.update(state=NEED_MORE_LABELS, reason=f"audit not valid: {reason}")
        return log

    t = paired_conditional_change_test(Zq, Yq, Dq, Gq, h1_basis=h1_basis,
                                       condition_coding=condition_coding, rank=rank, C=C,
                                       n_boot=n_boot, seed=seed)
    would_confirm = bool(t["valid"] and t["p_value"] <= alpha)   # raw significance (pre-guard)
    log.update(valid=bool(t["valid"]), p_value=float(t["p_value"]), T=float(t["T"]), reason=t["reason"],
               n_pairs=t["n_pairs"], classes_by_condition=t["classes_by_condition"],
               n_boot_invalid=t["n_boot_invalid"], h1_basis=t["h1_basis"],
               condition_coding=t["condition_coding"], condition_code_values=t["condition_code_values"],
               weighted_condition_mean_check=t["weighted_condition_mean_check"], C_used=t["C_used"],
               n_features_interaction=t["n_features_interaction"], observed_T=float(t["T"]),
               null_mean=t["null_mean"], null_sd=t["null_sd"], min_confirm_pairs=int(min_confirm_pairs),
               would_confirm_without_min_pairs=would_confirm, would_confirm_without_guard=would_confirm)
    if not t["valid"]:
        log["state"] = NEED_MORE_LABELS             # null not estimable on this audit -> need cleaner/more
    elif would_confirm and len(pick) >= min_confirm_pairs:
        log["state"] = CONCEPT_CONFIRMED            # significant AND audit large enough to claim it
    elif would_confirm:
        log["state"] = NEED_MORE_LABELS             # GUARD: significant but audit too small to confirm
    elif len(pick) >= decide_n:
        log["state"] = NO_CONCEPT_EVIDENCE          # enough labels, no conditional change found
    else:
        log["state"] = NEED_MORE_LABELS             # non-rejection on a small budget is not a decision
    return log
