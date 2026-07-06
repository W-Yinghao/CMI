"""ACAR V6-A0 — action-viability audit: oracle envelope + beneficial coverage (Q1) on ELIGIBLE EVAL batches.

DIAGNOSTIC-ONLY / EXPLORATORY. CODE + SYNTHETIC/FIXTURE TESTS ONLY — NOT executed on real DEV labels (V6-A0b is a separate
authorization). Numpy + acar.features imported lazily; NO torch / sklearn here.

Pinned semantics (notes/ACAR_V6_A0_DIAGNOSTIC_RUNNER_IMPLEMENTATION.md):
  * Primary action set = V5 final admitted Stage-2B: identity + {matched_coral, spdim, t3a}; the matched_coral IMPLEMENTATION is
    stable_matched_coral_v1 (via `stage2_real_action_provider.real_action_provider`) — the old unsafe pmct path is NEVER used.
  * LABEL SEAM: labels enter ONLY `batch_action_delta_r` (via `label_view.resolve_label`) to form ΔR_a = NLL(f_a) − NLL(f_0); the
    beneficial target 1[ΔR_a<0] derives from ΔR. All features (paired φ_a + source confidence/entropy/size) are label-free.
  * Forced sub-MIN_BATCH tails are adaptation-INELIGIBLE (Stage-2B3): excluded from the action envelope and the beneficial-coverage
    denominator; counted separately in `accounting`.
  * EVAL split only (the continuation gate). FIT/CAL are descriptive elsewhere and cannot set the primary gate.
"""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5 import stage2_policy_eval as PE
from acar.v5 import stage2_action_records as AR

PRIMARY_ACTIONS = P.ACTIONS               # ("matched_coral", "spdim", "t3a"); matched_coral impl = stable_matched_coral_v1
MIN_BATCH = PE.STAGE2_MIN_BATCH           # 8
_PAIRED = ("d_entropy", "d_margin", "flip_rate", "JS", "Bures", "post_sep", "n_eff")


def _nll(p, y):
    import numpy as np
    p = np.clip(np.asarray(p, float), 1e-12, 1.0)
    return float(-np.log(p[:, int(y)]).mean())


def batch_action_delta_r(outputs, y):
    """LABEL SEAM. ΔR_a = NLL(p_a, y) − NLL(p_0, y) for each non-identity action. The label `y` is read ONLY here."""
    r0 = _nll(outputs["identity"][0], y)
    return {a: (_nll(outputs[a][0], y) - r0) for a in PRIMARY_ACTIONS}


def batch_label_free_features(outputs, n):
    """Label-free per-action paired features (φ_a) + batch source-confidence / entropy / size. No label read."""
    import numpy as np
    import acar.features as AF
    p0, z0 = outputs["identity"]
    p0 = np.asarray(p0, float)
    src_conf = float(p0.max(axis=1).mean())
    clp = np.clip(p0, 1e-12, 1.0)
    batch_entropy = float((-(clp * np.log(clp)).sum(axis=1)).mean())
    per_action = {a: AR._to_protocol_features(AF.paired_features(p0, outputs[a][0], z0, outputs[a][1]))
                  for a in PRIMARY_ACTIONS}
    return {"per_action": per_action, "source_confidence": src_conf, "batch_entropy": batch_entropy, "batch_size": int(n)}


def collect_eval_records(disease_folds, action_provider, *, provenance_by_subject=None):
    """Collect ELIGIBLE EVAL batch records for ONE disease (forced tails excluded). Each record carries label-free features +
    ΔR_a (label seam) + subject_key/batch_id/provenance. Returns (records, accounting). EVAL split ONLY."""
    import numpy as np
    from acar.v5 import stage2_feature_loader as FL
    prov_map = provenance_by_subject or {}
    records = []
    acct = {"n_eval_subjects": 0, "n_eval_batches_total": 0, "n_eval_forced_tails": 0, "n_eval_eligible_batches": 0}
    for fold in disease_folds:
        by_subject, source_lda, lv = fold["by_subject"], fold["source_lda"], fold["label_view"]
        for sk in FL.subjects_in_group(by_subject, "eval"):                     # EVAL subjects only
            acct["n_eval_subjects"] += 1
            Z = np.asarray(by_subject[sk]["embedding"], float)
            y = int(lv.resolve_label(sk))                                       # LABEL — used ONLY for ΔR below
            prov = str(prov_map.get(sk, "native"))
            for bi, (zb, forced) in enumerate(PE.window_batches(Z)):
                acct["n_eval_batches_total"] += 1
                if forced:                                                      # sub-MIN_BATCH tail: ineligible, no action, oracle 0
                    acct["n_eval_forced_tails"] += 1
                    continue
                acct["n_eval_eligible_batches"] += 1
                outputs = AR.subject_action_outputs(zb, source_lda, action_provider=action_provider)   # label-free
                records.append({"subject_key": sk, "batch_id": int(bi), "provenance": prov, "n": int(zb.shape[0]),
                                "features": batch_label_free_features(outputs, zb.shape[0]),
                                "delta_r": batch_action_delta_r(outputs, y)})   # ΔR (label seam)
    return records, acct


def oracle_envelope(records):
    """Q1 on ELIGIBLE EVAL batches, SUBJECT-MACRO (each subject weight 1 — batch-rich subjects do NOT dominate the gate).
    `oracle_red_upper` = −mean_subject mean_batch min(0, min_a ΔR_a) (best action OR no-op per batch; benefit only).
    `beneficial_coverage_subject_macro` = mean_subject (eligible batches with min_a ΔR_a < 0 / eligible batches) — the PRIMARY
    coverage that feeds the continuation gate. `beneficial_coverage_batch_weighted` = the global batch fraction (descriptive only).
    `oracle_conditional_harm` = 0 by construction (oracle adapts only when best<0). `no_safe_action_rate_subject_macro` = mean_
    subject fraction of eligible batches where EVERY action is harmful (descriptive). If NO subject has an eligible EVAL batch the
    primary metrics are NaN (the gate then fails)."""
    import numpy as np
    by_subj = {}
    for r in records:
        by_subj.setdefault(r["subject_key"], []).append(r)
    upper_terms, benef_frac, nosafe_frac = [], [], []
    n_batches = n_benef = n_no_safe = 0
    for _sk, rs in by_subj.items():
        ut, b_s, ns_s = [], 0, 0
        for r in rs:
            best = min(r["delta_r"].values())                                  # label-aware oracle: best of the 3 actions
            ut.append(min(0.0, best))
            n_batches += 1
            if best < 0.0:
                b_s += 1
                n_benef += 1
            else:
                ns_s += 1
                n_no_safe += 1                                                  # no beneficial action -> oracle no-ops
        n_s = len(rs)
        upper_terms.append(float(np.mean(ut)))
        benef_frac.append(b_s / n_s)                                           # per-subject beneficial fraction
        nosafe_frac.append(ns_s / n_s)
    nan = float("nan")
    have = len(by_subj) > 0
    return {"oracle_red_upper": (-float(np.mean(upper_terms)) if have else nan),
            "beneficial_coverage_subject_macro": (float(np.mean(benef_frac)) if have else nan),      # PRIMARY (gate)
            "beneficial_coverage_batch_weighted": ((n_benef / n_batches) if n_batches else nan),      # descriptive
            "no_safe_action_rate_subject_macro": (float(np.mean(nosafe_frac)) if have else nan),      # descriptive
            "oracle_conditional_harm": 0.0,
            "n_eligible_batches": n_batches, "n_subjects_with_eligible": len(by_subj)}
