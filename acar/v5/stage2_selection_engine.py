"""ACAR V5 Stage-2 selection ENGINE (orchestrator; numpy lazy). Runs the frozen selection under a VALID Stage-2B authorization:

  FIT → per-fold thresholds ; CAL → H1–H3 raw p-values → Holm over (candidate×disease×{H1,H2,H3}) → G1/G3/G4 certification ;
  EVAL → red / red_upper / v2_replay → G2 / G5 + the final OOF G1–G5 report. Joint objective = maximize min_disease(red −
  v2_replay_red) among G1–G5-eligible candidates, with the deterministic tie-break (lower harm_among_adapted → higher coverage →
  more conservative family P3 ≺ P1/P2/P4 ≺ P5). A CAL-certified winner that FAILS the final EVAL G1/G3/G4 report ⇒ DEV_STOP (never
  a silent reselection). No eligible candidate ⇒ DEV_STOP.

Fails closed without an authorization bound to the admitted package. Stage-2B0 exercises this on SYNTHETIC fixtures (injected
action + v2-replay providers, synthetic labels); NO real DEV selection is run.
"""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5 import ltt as LTT
from acar.v5 import stage2b_authorization as AUTH
from acar.v5 import stage2_selection_manifest as MANIFEST
from acar.v5 import stage2_feature_loader as FL
from acar.v5 import stage2_thresholds as TH
from acar.v5 import stage2_policy_eval as PE
from acar.v5 import stage2_action_records as AR
from acar.v5 import stage2_gates as GATES
from acar.v5 import stage2_selection_report as RPT

DISEASES = ("PD", "SCZ")
_FAMILY_RANK = {"P3": 0, "P1": 1, "P2": 1, "P4": 1, "P5": 2}   # conservatism: P3 ≺ P1/P2/P4 ≺ P5
_TIE = 1e-9


class Stage2EngineError(RuntimeError):
    pass


def holm_family_keys_pvalues(per, manifest):
    """Build the FIXED Holm family over ALL candidate × disease × {H1,H2,H3} cells (22×2×3=132). A NON-EVALUABLE cell (per is
    None) contributes raw p = 1.0 for each hypothesis so the family size NEVER shrinks — skipping non-evaluable cells would
    reduce the multiplicity correction and bias certification of the other candidates (Stage-2B0b correction)."""
    keys, pvals = [], []
    for cand in manifest:
        cid = cand["id"]
        for d in DISEASES:
            ev = per.get((cid, d))
            for h in ("H1", "H2", "H3"):
                keys.append((cid, d, h))
                pvals.append(ev["cal_raw"][h] if ev is not None else 1.0)
    return keys, pvals


def holm_adjusted_map(per, manifest):
    """Holm-adjust the FIXED family (132 cells) together and return {(cid, disease, hypothesis): adjusted_p}."""
    keys, pvals = holm_family_keys_pvalues(per, manifest)
    adj = GATES.holm_adjust(pvals)
    return {keys[i]: float(adj[i]) for i in range(len(keys))}


def _fit_batches(by_subject, fit_keys, source_lda, action_provider):
    """Build FIT action-record batches at the SAME 32-window ACAR-B granularity that routing uses at CAL/EVAL — one record per
    32-window chunk over the FIT subjects. Sub-MIN_BATCH tails are forced to identity at CAL/EVAL (never routed), so they are
    EXCLUDED from the FIT quantile universe. This keeps the fitted operating point calibrated to the units it is applied to."""
    import numpy as np
    out = []
    for sk in fit_keys:
        Z = np.asarray(by_subject[sk]["embedding"], float)
        for zb, forced_id in PE.window_batches(Z):
            if forced_id:
                continue
            out.append(AR.batch_from_outputs(sk, AR.subject_action_outputs(zb, source_lda, action_provider=action_provider)))
    return out


def _evaluate_candidate_on_disease(candidate, folds, action_provider):
    """OOF over a disease's folds: aggregate CAL records (certification) + EVAL records / red terms (utility). Returns an eval
    dict, or None if the candidate is NON-EVALUABLE (zero FIT proposed-action records in any fold)."""
    import numpy as np
    cal_records, eval_records, red_terms, upper_terms = [], [], [], []
    for fold in folds:
        by_subject, source_lda, lv = fold["by_subject"], fold["source_lda"], fold["label_view"]
        fit_keys = FL.subjects_in_group(by_subject, "fit")
        fit_batches = _fit_batches(by_subject, fit_keys, source_lda, action_provider)
        try:
            thr = TH.fit_thresholds(candidate, fit_batches)
        except TH.NonEvaluableCandidate:
            return None
        cal = PE.evaluate_candidate_disease(candidate, thr, FL.subjects_in_group(by_subject, "cal"),
                                            by_subject, source_lda, lv, action_provider=action_provider)
        ev = PE.evaluate_candidate_disease(candidate, thr, FL.subjects_in_group(by_subject, "eval"),
                                           by_subject, source_lda, lv, action_provider=action_provider)
        cal_records += cal["subject_records"]
        eval_records += ev["subject_records"]
        red_terms += ev["red_terms"]
        upper_terms += ev["upper_terms"]
    if not eval_records:
        return None
    return {"cal_raw": GATES.cal_raw_pvalues(cal_records),
            "red": -float(np.mean(red_terms)) if red_terms else 0.0,
            "red_upper": -float(np.mean(upper_terms)) if upper_terms else 0.0,
            "eval_gate": LTT.gate_disease(eval_records)}


def _row(ev, v2_d, cert, g2_d, g5_d, g5_comp, adj_h1, adj_h2, adj_h3):
    b = ev["cal_raw"]["bounds"]
    return RPT.candidate_disease_result(
        coverage_lcb=b["coverage_lcb"], l_harm_all_ucb=b["l_harm_all_ucb"],
        harm_among_adapted_ucb=b["harm_among_adapted_ucb"], h2_evaluable=b["h2_evaluable"],
        g1=b["G1_coverage"], g3=b["G3_l_harm_all"], g4=b["G4_harm_among_adapted"],
        holm_p_h1=adj_h1, holm_p_h2=adj_h2, holm_p_h3=adj_h3, cert_pass=cert,
        red=ev["red"], red_upper=ev["red_upper"], v2_replay_red=v2_d, g2_margin=ev["red"] - v2_d,
        g2=g2_d, g5=g5_d, g5_comparator=g5_comp)


def _select_winner(eligible, per, manifest):
    """eligible = [(cid, min_margin)]. Max min_margin; ties → lower macro harm_among_adapted_ucb → higher macro coverage_lcb →
    more conservative family (P3≺P1/P2/P4≺P5) → lexicographic id."""
    fam = {c["id"]: c["family"] for c in manifest}

    def macro(cid, key, sign):
        vals = [per[(cid, d)]["eval_gate"][key] for d in DISEASES]
        vals = [v for v in vals if v is not None]
        return sign * (sum(vals) / len(vals)) if vals else sign * 0.0

    best_margin = max(m for _, m in eligible)
    tied = [cid for cid, m in eligible if m >= best_margin - _TIE]
    tied.sort(key=lambda cid: (macro(cid, "harm_among_adapted_ucb", +1.0),   # lower harm first
                               macro(cid, "coverage_lcb", -1.0),             # higher coverage first (negated)
                               _FAMILY_RANK[fam[cid]], cid))
    return tied[0]


def _dev_stop(reason, per_candidate=None, per_disease=None, macro=None, extra_notes=None):
    notes = {"dev_stop_reason": reason, "cal_eval": "H1-H3 Holm on CAL; G2/G5 + final G1-G5 report on EVAL"}
    if extra_notes:
        notes.update(extra_notes)
    return RPT.build_selection_report(
        outcome=RPT.OUTCOME_DEV_STOP, selected_candidate_id=None, per_candidate=per_candidate or {},
        per_disease=per_disease or {}, macro=macro or {}, holm_family_alpha=P.ALPHA,
        objective="maximize min_disease(red - v2_replay_red)", notes=notes)


def run_selection(authorization, *, stage1b_run_id, stage1b_registry_sha256, disease_inputs,
                  action_provider=AR.production_action_provider, v2_replay_provider=GATES.real_v2_replay_provider):
    """Run Stage-2 candidate selection under a valid Stage-2B authorization. `disease_inputs` = {disease: {"folds": [{"by_subject",
    "source_lda", "label_view"}, ...]}}. Returns a Stage-2 selection report (SELECTED or DEV_STOP). Fail-closed."""
    AUTH.require_stage2b_ready(authorization, stage1b_run_id=stage1b_run_id, stage1b_registry_sha256=stage1b_registry_sha256)
    if set(disease_inputs) != set(DISEASES):
        raise Stage2EngineError(f"disease_inputs must cover exactly {sorted(DISEASES)}")
    manifest = MANIFEST.selection_manifest()                                       # the 22 joint candidates

    # Pass 1 — evaluate every candidate on both diseases (OOF)
    per = {}
    for cand in manifest:
        for d in DISEASES:
            per[(cand["id"], d)] = _evaluate_candidate_on_disease(cand, disease_inputs[d]["folds"], action_provider)

    # Holm over the FIXED certification family: ALL candidate × disease × {H1,H2,H3} cells (22×2×3=132). Non-evaluable cells
    # enter as raw p=1 (cert_pass=False) so the family size never shrinks (Stage-2B0b correction).
    adj_map = holm_adjusted_map(per, manifest)
    holm_family_size = len(adj_map)
    n_nonevaluable = sum(1 for cand in manifest for d in DISEASES if per[(cand["id"], d)] is None)
    holm_notes = {"holm_family_size": holm_family_size, "holm_nonevaluable_cells": n_nonevaluable}
    cert_pass = {}
    for cand in manifest:
        cid = cand["id"]
        for d in DISEASES:
            ev = per[(cid, d)]
            cert_pass[(cid, d)] = (ev is not None and GATES.cert_pass_from_adjusted(
                adj_map[(cid, d, "H1")], adj_map[(cid, d, "H2")], adj_map[(cid, d, "H3")]))

    # v2_replay_red per disease (fail-closed: missing ⇒ selection cannot run)
    v2 = {}
    try:
        for d in DISEASES:
            v2[d] = float(v2_replay_provider(d, {"disease_inputs": disease_inputs, "per": per}))
    except GATES.V2ReplayNotEvaluable as e:
        return _dev_stop(f"v2_replay not evaluable: {e}", extra_notes=holm_notes)
    macro_v2 = float(sum(v2[d] for d in DISEASES) / len(DISEASES))

    # best-eligible P3 comparator (EVAL red among CAL-certified P3 candidates), per disease
    p3_ids = [c["id"] for c in manifest if c["family"] == "P3"]
    red_p3_best = {}
    for d in DISEASES:
        vals = [per[(cid, d)]["red"] for cid in p3_ids if per[(cid, d)] is not None and cert_pass[(cid, d)]]
        red_p3_best[d] = max(vals) if vals else None

    # Pass 2 — eligibility (G1–G5) + objective margins
    per_candidate, eligible = {}, []
    for cand in manifest:
        cid = cand["id"]
        macro_red = sum((per[(cid, d)]["red"] if per[(cid, d)] else 0.0) for d in DISEASES) / len(DISEASES)
        g2m = GATES.g2_macro(macro_red, macro_v2)
        rows, ok, margins = {}, True, []
        for d in DISEASES:
            ev = per[(cid, d)]
            if ev is None:
                rows[d], ok = None, False
                continue
            cp = cert_pass[(cid, d)]
            g2d = GATES.g2_per_disease(ev["red"], v2[d])
            g5d = GATES.g5_pass(ev["red"], ev["red_upper"], red_p3_best[d])
            margins.append(ev["red"] - v2[d])
            rows[d] = _row(ev, v2[d], cp, g2d, g5d, red_p3_best[d],
                           adj_map[(cid, d, "H1")], adj_map[(cid, d, "H2")], adj_map[(cid, d, "H3")])
            if not (cp and g2d and g5d):
                ok = False
        per_candidate[cid] = rows
        if ok and g2m:
            eligible.append((cid, min(margins)))

    per_disease = {d: {"v2_replay_red": v2[d], "red_p3_best": red_p3_best[d]} for d in DISEASES}
    macro = {"v2_replay_red": macro_v2}
    if not eligible:
        return _dev_stop("no candidate passed G1-G5 (both diseases + macro)", per_candidate, per_disease, macro,
                         extra_notes=holm_notes)

    winner = _select_winner(eligible, per, manifest)
    for d in DISEASES:                                                             # final OOF EVAL G1/G3/G4 gate
        if not per[(winner, d)]["eval_gate"]["certification_pass"]:
            return _dev_stop(f"selected {winner} failed the final EVAL G1/G3/G4 report on {d}",
                             per_candidate, per_disease, macro, extra_notes=holm_notes)
    return RPT.build_selection_report(
        outcome=RPT.OUTCOME_SELECTED, selected_candidate_id=winner, per_candidate=per_candidate,
        per_disease=per_disease, macro=macro, holm_family_alpha=P.ALPHA,
        objective="maximize min_disease(red - v2_replay_red)",
        notes={"cal_eval": "H1-H3 Holm on CAL; G2/G5 + final G1-G5 report on EVAL", **holm_notes})
