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


def certify_paired(Z, Y, D, G, m, alpha=0.05, decide_n=20, min_pairs=4,
                   rank=3, C=0.5, n_boot=200, seed=0):
    """Query `m` paired target subjects' labels and run the within-subject conditional-change test.
    Returns a dict with the certificate state + per-cluster log fields."""
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
    log["n_queried"] = int(len(pick))
    ok, reason = paired_validity(Yq, Dq, Gq, min_subjects=min(min_pairs, len(pick)))
    if not ok:
        log.update(state=NEED_MORE_LABELS, reason=f"audit not valid: {reason}")
        return log

    t = paired_conditional_change_test(Zq, Yq, Dq, Gq, rank=rank, C=C, n_boot=n_boot, seed=seed)
    log.update(valid=bool(t["valid"]), p_value=float(t["p_value"]), T=float(t["T"]), reason=t["reason"])
    if not t["valid"]:
        log["state"] = NEED_MORE_LABELS
    elif t["p_value"] <= alpha:
        log["state"] = CONCEPT_CONFIRMED
    elif len(pick) >= decide_n:
        log["state"] = NO_CONCEPT_EVIDENCE          # enough labels, no conditional change found
    else:
        log["state"] = NEED_MORE_LABELS             # non-rejection on a small budget is not a decision
    return log
