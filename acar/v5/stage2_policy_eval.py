"""ACAR V5 Stage-2 POLICY evaluation (numpy lazy). Applies a candidate (with its FIT-only thresholds) to a split's per-subject
batches and produces the subject-clustered {adapted, harmful} records the certifier consumes, plus the label-based utility
quantities red and red_upper. This is the ONLY Stage-2 module that consumes a label VALUE — via the evaluation label view — to
form ΔR_a(B) = R_B(f_a) − R_B(f_0) (NLL) and the per-batch harm outcome. Routing/scalarization/thresholds never see a label.

One subject's windows are split (in window order) into fixed-size batches; a batch below MIN_BATCH is forced to identity.
"""
from __future__ import annotations
from acar.v5 import protocol as P

STAGE2_BATCH_SIZE = 32                 # recording-ordered batch size (ACAR B); real value re-confirmed for the real Stage-2B run
STAGE2_MIN_BATCH = 8                   # a sub-MIN_BATCH remainder is forced to identity (never adapts)


class Stage2PolicyEvalError(RuntimeError):
    pass


def _nll(p, y):
    """Mean over a batch's windows of the per-window negative log-likelihood of the subject's label y."""
    import numpy as np
    p = np.clip(np.asarray(p, float), 1e-12, 1.0)
    return float(-np.log(p[:, int(y)]).mean())


def window_batches(Z, batch_size=STAGE2_BATCH_SIZE, min_batch=STAGE2_MIN_BATCH):
    """Split a subject's [n_win, dim] embeddings (assumed window-ordered) into consecutive batches; flag sub-min_batch tails
    (which are forced to identity and never routed). Shared by threshold FITTING and CAL/EVAL routing so both use the SAME
    32-window ACAR-B decision unit."""
    import numpy as np
    Z = np.asarray(Z, float)
    n = Z.shape[0]
    out = []
    for s in range(0, n, batch_size):
        zb = Z[s:s + batch_size]
        out.append((zb, zb.shape[0] < min_batch))
    return out


def evaluate_candidate_disease(candidate, thresholds, subject_keys, by_subject, source_lda, label_view, *,
                               action_provider, batch_size=STAGE2_BATCH_SIZE, min_batch=STAGE2_MIN_BATCH):
    """Evaluate one candidate on one disease's split subjects. Returns subject_records (for ltt.gate_disease) + red + red_upper.
    Labels are read ONLY here (label_view.resolve_label), ONLY to compute ΔR / harm — never for routing."""
    import numpy as np
    from acar.v5 import scalarization as SCAL
    from acar.v5 import stage2_action_records as AR
    subject_records, red_terms, upper_terms = [], [], []
    for sk in subject_keys:
        rec = by_subject.get(sk)
        if rec is None:
            raise Stage2PolicyEvalError(f"subject {sk!r} missing from the loaded feature map")
        Z = np.asarray(rec["embedding"], float)
        y = int(label_view.resolve_label(sk))                    # LABEL enters ONLY here (evaluation view)
        batch_meta, chosen_drs, upper_drs = [], [], []
        for zb, forced_id in window_batches(Z, batch_size, min_batch):
            outputs = AR.subject_action_outputs(zb, source_lda, action_provider=action_provider)   # label-free
            r0 = _nll(outputs["identity"][0], y)
            dr = {a: (_nll(outputs[a][0], y) - r0) for a in P.ACTIONS}          # ΔR_a
            if forced_id:
                chosen = P.IDENTITY
            else:
                chosen = SCAL.decide(candidate, AR.batch_from_outputs(sk, outputs), thresholds)
            dr_chosen = 0.0 if chosen == P.IDENTITY else float(dr[chosen])
            adapted = chosen != P.IDENTITY
            batch_meta.append({"adapted": bool(adapted), "harmful": bool(adapted and dr_chosen > 0.0)})
            chosen_drs.append(dr_chosen)
            upper_drs.append(min(0.0, min(dr.values())))                        # harmful-oracle clip to no-op
        subject_records.append({"subject": sk, "batches": batch_meta})
        red_terms.append(float(np.mean(chosen_drs)) if chosen_drs else 0.0)
        upper_terms.append(float(np.mean(upper_drs)) if upper_drs else 0.0)
    red = -float(np.mean(red_terms)) if red_terms else 0.0                       # red = −mean_subject(chosen ΔR)
    red_upper = -float(np.mean(upper_terms)) if upper_terms else 0.0             # red_upper = −mean_subject[min(0, min_a ΔR_a)]
    # per-subject terms let the OOF engine aggregate red/red_upper across folds (each subject is EVAL in exactly one fold)
    return {"subject_records": subject_records, "red": red, "red_upper": red_upper, "n_subjects": len(subject_keys),
            "red_terms": red_terms, "upper_terms": upper_terms}
