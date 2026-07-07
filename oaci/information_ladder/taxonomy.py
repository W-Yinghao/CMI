"""C24 — information-boundary taxonomy (I1-I7), deterministic. In Stage-1 (R3/R4 = REQUIRES_REINFERENCE) it
reports the read-only-ESTABLISHED cases (I1 witnesses, I4/I5 few-label, I6 oracle) and marks I2/I3 (unlabeled-
target) UNRESOLVED, with final=False -- C24 is NOT finalized until the P0-gated re-inference supplies R3/R4.
When R3/R4 are supplied, it resolves I2 vs I3 (identity-leakage-gated -> I7) and finalizes."""
from __future__ import annotations

from . import schema


def _r5_case(few):
    if few.get("few_labels_recover"):
        return schema.I4
    # label-hungry: recovers only at the largest budget, or not within tested budgets
    curve = few.get("curve", [])
    if curve:
        last = curve[-1]
        if last.get("gap_closed") is not None and last["gap_closed"] >= schema.SUCCESS_GAP_CLOSED:
            return schema.I5
    return None


def gauge_taxonomy(witnesses, few, oracle, identity, r1_gap, r3r4=None) -> dict:
    identity_laden = bool(identity.get("source_features_identity_separable"))
    source_nonid = bool(witnesses.get("source_nonidentifying") and (r1_gap is None or r1_gap < schema.SUCCESS_GAP_CLOSED))
    oracle_recovers = bool(oracle.get("oracle_gap_over_raw") and oracle["oracle_gap_over_raw"] > 0)
    # KEY disclosure: k=0 is the 0-label transductive target-mean centering (== target-centered oracle). If it
    # already recovers, the offset is a target-GROUPING quantity, NOT primarily a label quantity.
    zero_gap = few.get("zero_label_transductive_gap")
    grouping_recovers = bool(zero_gap is not None and zero_gap >= schema.SUCCESS_GAP_CLOSED)
    label_gain = few.get("label_gain_over_grouping")
    labels_refine = bool(label_gain is not None and label_gain >= schema.SUCCESS_GAP_CLOSED)
    r5_case = None if grouping_recovers else _r5_case(few)

    established = []
    if source_nonid:
        established.append(schema.I1)
    if grouping_recovers:
        established.append(schema.I6)                        # target-grouping (0-label transductive == oracle) recovers
        if labels_refine:
            established.append(schema.I4)                    # competence labels REFINE beyond grouping (secondary)
    elif r5_case:
        established.append(r5_case)
    elif oracle_recovers:
        established.append(schema.I6)

    # R3/R4 resolution
    r3r4_status = (r3r4 or {}).get("status", schema.STATUS_REQUIRES_REINFERENCE)
    final = (r3r4_status == schema.STATUS_OK)
    unresolved = [] if final else [schema.I2, schema.I3]
    unlabeled_case = None
    if final:
        gap = r3r4.get("gap_closed")
        loto_gen = r3r4.get("loto_generalizes")
        if identity_laden and not loto_gen and (r3r4.get("auc_improve") or 0) > 0:
            unlabeled_case = schema.I7                      # apparent unlabeled recovery is identity leakage
        elif gap is not None and gap >= schema.SUCCESS_GAP_CLOSED and loto_gen:
            unlabeled_case = schema.I2
        else:
            unlabeled_case = schema.I3
        established = [c for c in established if c != schema.I6] + [unlabeled_case] + \
                      ([schema.I6] if (oracle_recovers and unlabeled_case == schema.I3) else [])

    primary = (unlabeled_case if final else
               (schema.I6 if grouping_recovers else (r5_case or (schema.I1 if source_nonid else (schema.I6 if oracle_recovers else None)))))
    grouping_note = (" (via 0-label TRANSDUCTIVE target grouping == the target-centered oracle; competence labels "
                     "only refine it -> the missing ingredient is target GROUPING, not source observability and not "
                     "primarily labels)" if grouping_recovers else "")
    interp = {
        schema.I1: "source-only summaries are non-identifying for the per-target offset (cross-target source distance does not predict offset distance).",
        schema.I2: "UNLABELED target-marginal information recovers the offset (target-unlabeled gauge closes the oracle gap, generalizes LOTO, no identity leakage).",
        schema.I3: "UNLABELED target-marginal information is insufficient; the offset is not in the target distribution's confidence geometry.",
        schema.I4: "a SMALL labeled target-calibration budget (<=%d/class) sharply recovers the offset -> the missing quantity is target-specific scalar calibration." % schema.FEW_LABEL_RECOVERS_MAX_K,
        schema.I5: "the offset is label-hungry: many target labels are needed to recover it.",
        schema.I6: "target-centered/rank grouping recovers pooled transport" + grouping_note + "; source-only summaries do not.",
        schema.I7: "apparent target-unlabeled recovery depends on target-identity leakage, not valid marginal information.",
        None: "no read-only rung establishes offset recovery yet.",
    }[primary]
    return {"r3r4_status": r3r4_status, "final": final, "established_readonly": established,
            "unresolved_pending_reinference": unresolved, "primary_provisional": primary,
            "unlabeled_case": unlabeled_case, "source_nonidentifiable": source_nonid,
            "few_label_case": r5_case, "oracle_recovers": oracle_recovers, "identity_laden": identity_laden,
            "offset_recovered_by_transductive_grouping": grouping_recovers, "labels_refine_beyond_grouping": labels_refine,
            "zero_label_transductive_gap": zero_gap, "interpretation": interp,
            "next_science": ("C24 read-only rungs establish: " + (", ".join(established) if established else "none")
                             + ". I2/I3 (unlabeled-target) require the P0-gated target-audit re-inference before C24 finalizes."
                             if not final else "C24 finalized: unlabeled-target rung = " + str(unlabeled_case)),
            "diagnostic_only_non_deployable": True}
