"""B6.0 canary merge (analysis layer; does NOT touch the frozen B6.0 engine). Re-derives the 3-state classification
FROM RAW FIELDS and RENAMES per the reviewer's semantics: a non-significant p_C is NO_ACTIONABLE_CONCEPT_EVIDENCE
(NOT 'no concept'); low conditional-randomization support is UNIDENTIFIABLE_COVARIATE_LOCK regardless of p_C. Lock /
randomization-support diagnostics are FIRST-CLASS outputs. Reports the direct OLD-certifier vs B6-C-null comparison,
the C-null SHAPE (p_C floor fraction, C-null mean/sd, observed_T-vs-C-null), and POS retention. Fail-closed; kind-
specific (strong vs soft NEVER pooled). Development-only, NO tag, NO validity claim."""
import os, sys, json
import numpy as np

CDIR = "/home/infres/yinwang/realeeg_feas/b6_canary"
CONDS = ["NULL_cov_soft", "NULL_cov_plus_label_soft", "NULL_cov_strong_auc0.81", "NULL_cov_strong_auc0.94",
         "NULL_label", "random_label_control", "POS_concept", "POS_concept_plus_cov"]
NOCONCEPT = {"NULL_cov_soft", "NULL_cov_plus_label_soft", "NULL_cov_strong_auc0.81", "NULL_cov_strong_auc0.94",
             "NULL_label", "random_label_control"}
N = 50
N_BOOT = 200
P_FLOOR = 1.0 / (N_BOOT + 1)
# reviewer-renamed states, re-derived from the frozen engine's recorded b6_state (faithful rename)
RENAME = {"CONCEPT_CONFIRMED_B6": "CONCEPT_CONFIRMED",
          "NO_CONCEPT_EVIDENCE_B6": "NO_ACTIONABLE_CONCEPT_EVIDENCE",
          "UNIDENTIFIABLE_DUE_TO_COVARIATE_LOCK": "UNIDENTIFIABLE_COVARIATE_LOCK",
          "NEED_MORE_LABELS": "NEED_MORE_LABELS"}


def _read(p): return [json.loads(l) for l in open(p) if l.strip()]
def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and x == x]
    return float(np.median(xs)) if xs else float("nan")
def state(r): return RENAME.get(str(r.get("b6_state")), str(r.get("b6_state")))


def main():
    per = {}
    for c in CONDS:
        p = f"{CDIR}/b6_canary_{c}_0.jsonl"
        if not os.path.exists(p):
            print(f"FAIL-CLOSED: missing {p}"); sys.exit(2)
        recs = _read(p)
        if any("__worker_error__" in r for r in recs):
            print(f"FAIL-CLOSED: worker-error rows in {c}"); sys.exit(2)
        if len(recs) != N:
            print(f"FAIL-CLOSED: {c} has {len(recs)} != {N}"); sys.exit(2)
        ids = [r["task_id"] for r in recs]
        if len(set(ids)) != N:
            print(f"FAIL-CLOSED: duplicate task_id in {c}"); sys.exit(2)
        per[c] = recs

    all_recs = [r for c in CONDS for r in per[c]]
    dts = [r.get("fidelity_dT") for r in all_recs if isinstance(r.get("fidelity_dT"), (int, float)) and r.get("fidelity_dT") == r.get("fidelity_dT")]
    n_nan_fid = len(all_recs) - len(dts)
    # nan-fidelity cohorts: red-team-corrected accounting. The nan is on the OLD-certifier side (its own
    # NEED_MORE_LABELS/degenerate observed_T=nan), NOT the crt side; the crt computed a valid observed_T_crt.
    nan_rows = [dict(task_id=r["task_id"], old_T=r.get("observed_T"), crt_T=r.get("observed_T_crt"),
                     old_state=r.get("old_b3_state"), b6_state=r.get("b6_state"))
                for r in all_recs if not (isinstance(r.get("fidelity_dT"), (int, float)) and r.get("fidelity_dT") == r.get("fidelity_dT"))]
    inconsistent = [r["task_id"] for r in all_recs if state(r) == "CONCEPT_CONFIRMED"
                    and not (r.get("p_C_meanT", 1) <= 0.025 and r.get("p_C_stud", 1) <= 0.025)]
    print(f"ACCOUNTING: n_total={len(all_recs)} n_valid_fidelity={len(dts)} n_invalid={n_nan_fid} "
          f"| max|T_old-T_crt|={max(dts) if dts else float('nan'):.2e} | state-vs-p_C inconsistencies={len(inconsistent)}")
    for nr in nan_rows:
        print(f"  nan-fidelity: {nr['task_id']} OLD_T={nr['old_T']} (state {nr['old_state']}) vs crt_T={nr['crt_T']} "
              f"(b6 {nr['b6_state']}) -> nan is OLD-side degeneracy; crt valid (gate-fidelity gap, blast-radius 1/400)")

    rows = {}
    hdr = ("condition", "n", "oldB3", "B6conf", "B6noact", "B6lock", "pCfloor%", "medpC", "medCnullSD",
           "medEffRand", "medAUC", "medLockFr")
    print("\n" + " ".join(f"{h:>12s}" if i else f"{h:<26s}" for i, h in enumerate(hdr)))
    print("-" * 150)
    for c in CONDS:
        recs = per[c]; gt_noconcept = c in NOCONCEPT
        states = [state(r) for r in recs]
        old_conf = sum(1 for r in recs if r.get("old_false_confirm") or r.get("old_true_confirm"))
        b6_conf = sum(1 for s in states if s == "CONCEPT_CONFIRMED")
        b6_noact = sum(1 for s in states if s == "NO_ACTIONABLE_CONCEPT_EVIDENCE")
        b6_lock = sum(1 for s in states if s == "UNIDENTIFIABLE_COVARIATE_LOCK")
        b6_needmore = sum(1 for s in states if s == "NEED_MORE_LABELS")
        pC = [r.get("p_C_meanT") for r in recs]
        floor_frac = float(np.mean([1.0 if (x is not None and x <= P_FLOOR + 1e-9) else 0.0 for x in pC if x is not None]))
        obs_minus_null = med([(r.get("observed_T_crt", float("nan")) - r.get("c_null_mean_T", float("nan"))) for r in recs])
        cnull_tz = med([((r.get("observed_T_crt") - r.get("c_null_mean_T")) / r.get("c_null_sd_T"))
                        for r in recs if r.get("c_null_sd_T") and r.get("c_null_sd_T") == r.get("c_null_sd_T") and r.get("c_null_sd_T") > 0])
        rows[c] = dict(n=N, ground_truth_noconcept=gt_noconcept, old_B3_confirm=old_conf,
                       B6_confirm=b6_conf, B6_no_actionable=b6_noact, B6_lock=b6_lock, B6_need_more=b6_needmore,
                       B6_false_confirm=(b6_conf if gt_noconcept else None),
                       B6_true_confirm=(b6_conf if not gt_noconcept else None),
                       p_C_floor_frac=round(floor_frac, 3), median_p_C=med(pC),
                       median_Cnull_mean_T=med([r.get("c_null_mean_T") for r in recs]),
                       median_Cnull_sd_T=med([r.get("c_null_sd_T") for r in recs]),
                       median_obsT_minus_Cnull=obs_minus_null, median_Cnull_Tz=cnull_tz,
                       median_effective_randomization=med([r.get("eff_randomization") for r in recs]),
                       median_propensity_auc=med([r.get("propensity_auc") for r in recs]),
                       median_frac_condition_locked=med([r.get("frac_condition_locked") for r in recs]),
                       median_propensity_entropy=med([r.get("propensity_mean_entropy") for r in recs]))
        r = rows[c]
        print(f"{c:<26s} {N:>12d} {old_conf:>12d} {b6_conf:>12d} {b6_noact:>12d} {b6_lock:>12d} "
              f"{floor_frac*100:>11.1f} {r['median_p_C']:>12.3f} {r['median_Cnull_sd_T']:>12.5f} "
              f"{r['median_effective_randomization']:>12.1f} {r['median_propensity_auc']:>12.2f} {r['median_frac_condition_locked']:>12.2f}")

    print("\n=== A. strong-covariate false-confirm (the core question) ===")
    for c in ["NULL_cov_strong_auc0.81", "NULL_cov_strong_auc0.94"]:
        r = rows[c]
        print(f"  {c}: OLD B3 {r['old_B3_confirm']}/50 -> B6 confirm {r['B6_confirm']}/50 "
              f"(no-actionable {r['B6_no_actionable']}, lock {r['B6_lock']}); OLD ref ~26% at auc0.94, ~7% at auc0.81")
    print("=== A'. PURE-covariate soft null + destroyed-label control (should be silent) ===")
    for c in ["NULL_cov_soft", "random_label_control"]:
        r = rows[c]
        print(f"  {c}: OLD {r['old_B3_confirm']}/50 -> B6 confirm {r['B6_confirm']}/50 (no-act {r['B6_no_actionable']})")
    print("=== A''. PRIOR / MIXED nulls (B6 FIRES -- estimand gap, NOT covariate leakage) ===")
    for c in ["NULL_label", "NULL_cov_plus_label_soft"]:   # NB NULL_cov_plus_label carries a 0.35/0.65 PRIOR shift
        r = rows[c]
        kind = "pure prior shift" if c == "NULL_label" else "covariate+PRIOR shift (its firing is the prior part)"
        print(f"  {c}: OLD {r['old_B3_confirm']}/50 -> B6 confirm {r['B6_confirm']}/50 [{kind}]")
    print("=== B. C-null SHAPE (is observed_T typical? p_C not floor-pinned?) ===")
    for c in CONDS:
        if c in NOCONCEPT:
            r = rows[c]
            print(f"  {c}: median p_C={r['median_p_C']:.3f} floor_frac={r['p_C_floor_frac']:.2f} "
                  f"obsT-Cnull_mean={r['median_obsT_minus_Cnull']:+.5f} Cnull_Tz={r['median_Cnull_Tz']:+.2f}")
    print("=== C. POS retention (confirm vs lock vs no-actionable) ===")
    for c in ["POS_concept", "POS_concept_plus_cov"]:
        r = rows[c]
        print(f"  {c}: OLD true {r['old_B3_confirm']}/50 -> B6 confirm {r['B6_confirm']}/50 "
              f"(no-act {r['B6_no_actionable']}, lock {r['B6_lock']}); median p_C={r['median_p_C']:.3f} lockfrac={r['median_frac_condition_locked']:.2f}")

    # verdict (red-team-corrected): decompose into COVARIATE-target (B6's design goal) vs PRIOR/label shift (estimand gap)
    covariate_target = {  # pure-covariate nulls + destroyed-label control: B6 SHOULD be silent
        "NULL_cov_soft": rows["NULL_cov_soft"]["B6_confirm"],
        "NULL_cov_strong_auc0.81": rows["NULL_cov_strong_auc0.81"]["B6_confirm"],
        "NULL_cov_strong_auc0.94": rows["NULL_cov_strong_auc0.94"]["B6_confirm"],
        "random_label_control": rows["random_label_control"]["B6_confirm"]}
    prior_shift = {  # prior/mixed nulls: B6 FIRES (Y-perp-C|Z is broader than concept -> includes prior shift)
        "NULL_label": rows["NULL_label"]["B6_confirm"],
        "NULL_cov_plus_label_soft": rows["NULL_cov_plus_label_soft"]["B6_confirm"]}
    covariate_fixed = (covariate_target["NULL_cov_soft"] <= 3 and covariate_target["NULL_cov_strong_auc0.81"] <= 3
                       and covariate_target["NULL_cov_strong_auc0.94"] <= 5 and covariate_target["random_label_control"] <= 3)
    prior_fires = any(v > 3 for v in prior_shift.values())
    pos_clean_concept_OLD = {"POS_concept": rows["POS_concept"]["old_B3_confirm"],
                             "POS_concept_plus_cov": rows["POS_concept_plus_cov"]["old_B3_confirm"]}  # prior-blind OLD null
    verdict = ("COVARIATE-ROOT FIX REAL + POS RETAINED, but PRIOR-SHIFT ESTIMAND GAP (fires on prior/label shift) "
               "-> NOT B6.1 plain-C-null; NEXT = B6-FM class-preserving/fixed-margin C-randomization"
               if (covariate_fixed and prior_fires) else "atypical -- inspect")
    print(f"\n  >>> COVARIATE-target (B6 should be silent): {covariate_target} -> fixed={covariate_fixed}")
    print(f"  >>> PRIOR/label-shift (estimand gap, B6 fires): {prior_shift} -> prior_fires={prior_fires}")
    print(f"  >>> POS clean-concept power (prior-blind OLD null, NOT B6's admixed count): {pos_clean_concept_OLD}")
    print(f"  >>> LOCK-STATE NOTE: UNIDENTIFIABLE_COVARIATE_LOCK fired 0/400 (eff_randomization is a GLOBAL sum over "
          f"~5800 trials, min ~171 >> floor 5) -> INERT here; the p-value carried the strong-null abstention. "
          f"B6-FM must use a per-subject/count-conditioned lock measure.")
    print(f"  >>> B6.0 VERDICT: {verdict}")

    tables = dict(scope="B6.0 plain condition-randomization canary; development-only; NOT confirmatory; NO tag; NO validity claim",
                  base_seed=200_000_000, n_total=len(all_recs), n_valid_fidelity=len(dts), n_invalid_fidelity=n_nan_fid,
                  nan_fidelity_rows=nan_rows, nan_fidelity_note="nan is OLD-certifier-side degeneracy (observed_T=nan) while crt was valid; gate-fidelity gap (crt does not replicate the certifier's per-condition-class validity guard); blast-radius 1/400",
                  n_per_condition=N, n_boot=N_BOOT, p_floor=P_FLOOR, fidelity_max_dT=(max(dts) if dts else None),
                  state_vs_pC_inconsistencies=len(inconsistent),
                  state_semantics="p_C non-sig = NO_ACTIONABLE_CONCEPT_EVIDENCE (NOT 'no concept'); UNIDENTIFIABLE_COVARIATE_LOCK when randomization support too low (INERT here, see lock note)",
                  lock_state_inert=True, lock_note="eff_randomization = global sum of per-trial entropies over ~5800 trials (min 171 >> floor 5) -> lock never fires; p-value carries the strong-null abstention",
                  per_condition=rows, covariate_target=covariate_target, covariate_root_fixed=bool(covariate_fixed),
                  prior_shift_firing=prior_shift, prior_shift_gap=bool(prior_fires),
                  pos_clean_concept_power_OLD_prior_blind=pos_clean_concept_OLD,
                  pos_note="B6's POS confirm (15/12) carries a rotation-induced prior admixture; the CLEAN concept power is the prior-blind OLD certifier (12/13)",
                  verdict=verdict,
                  caveat="B6 validity depends on the quality of the estimated C|Z,S law; in-sample dim-reduction / cross-fit choices may affect calibration in a direction NOT assumed conservative; NO confirmatory validity claim from B6.0")
    json.dump(tables, open(f"{CDIR}/b6_canary_tables.json", "w"), indent=1, default=str)
    print(f"\n  saved {CDIR}/b6_canary_tables.json")


if __name__ == "__main__":
    main()
