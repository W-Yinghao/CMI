"""B6-FM canary merge. Direct OLD-certifier vs B6.0-plain vs B6-FM comparison + margin diagnostics (first-class).
Decisive question: does the fixed-margin (class-preserving) C-null DROP the NULL_label false-confirm that B6.0
plain-C-null re-opened (B6.0: 25/50), toward the OLD certifier's 0/50 -- while KEEPING the strong-cov fix and POS
power? Fail-closed; kind-specific (never pooled). Development-only, NO tag, NO validity claim."""
import os, sys, json
import numpy as np

CDIR = "/home/infres/yinwang/realeeg_feas/b6fm_canary"
B60 = "/home/infres/yinwang/realeeg_feas/b6_canary/b6_canary_tables.json"
CONDS = ["NULL_cov_soft", "NULL_cov_plus_label_soft", "NULL_cov_strong_auc0.81", "NULL_cov_strong_auc0.94",
         "NULL_label", "random_label_control", "POS_concept", "POS_concept_plus_cov"]
NOCONCEPT = {"NULL_cov_soft", "NULL_cov_plus_label_soft", "NULL_cov_strong_auc0.81", "NULL_cov_strong_auc0.94",
             "NULL_label", "random_label_control"}
N = 50


def _read(p): return [json.loads(l) for l in open(p) if l.strip()]
def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and x == x]
    return float(np.median(xs)) if xs else float("nan")


def main():
    per = {}
    for c in CONDS:
        p = f"{CDIR}/b6fm_canary_{c}_0.jsonl"
        if not os.path.exists(p):
            print(f"FAIL-CLOSED: missing {p}"); sys.exit(2)
        recs = _read(p)
        if any("__worker_error__" in r for r in recs):
            print(f"FAIL-CLOSED: worker-error rows in {c}"); sys.exit(2)
        if len(recs) != N or len({r["task_id"] for r in recs}) != N:
            print(f"FAIL-CLOSED: {c} count/dup"); sys.exit(2)
        per[c] = recs
    b60 = json.load(open(B60))["per_condition"] if os.path.exists(B60) else {}

    all_recs = [r for c in CONDS for r in per[c]]
    dts = [r.get("fidelity_dT") for r in all_recs if isinstance(r.get("fidelity_dT"), (int, float)) and r.get("fidelity_dT") == r.get("fidelity_dT")]
    max_margin_err = max([r.get("margin_fidelity_max_err", 0) or 0 for r in all_recs] + [0])
    max_count_err = max([r.get("max_subject_count_err", 0) or 0 for r in all_recs] + [0])
    n_sampler_invalid = sum(1 for r in all_recs if r.get("fm_sampler_invalid"))
    print(f"ACCOUNTING: n_total={len(all_recs)} n_valid_fidelity={len(dts)} | max|T_old-T_crt|={max(dts) if dts else float('nan'):.2e} "
          f"| MARGIN_FIDELITY max_err={max_margin_err:.0f} max_count_err={max_count_err:.0f} | sampler_invalid={n_sampler_invalid}")

    rows = {}
    print(f"\n{'condition':26s} {'GT':>8} {'OLD':>4} {'B6.0':>5} {'B6FM_conf':>9} {'FMnoact':>8} {'FMlock':>7} {'FMinv':>6} {'medpC_FM':>9} {'feas':>7} {'uniqC*':>7} {'AUC':>5}")
    print("-" * 128)
    for c in CONDS:
        recs = per[c]; gt_noconcept = c in NOCONCEPT
        old_conf = sum(1 for r in recs if r.get("old_false_confirm") or r.get("old_true_confirm"))
        fm_conf = sum(1 for r in recs if r.get("fm_state") == "CONCEPT_CONFIRMED")
        fm_noact = sum(1 for r in recs if r.get("fm_state") == "NO_ACTIONABLE_CONCEPT_EVIDENCE")
        fm_lock = sum(1 for r in recs if r.get("fm_state") == "UNIDENTIFIABLE_MARGIN_LOCK")
        fm_inv = sum(1 for r in recs if r.get("fm_state") == "SAMPLER_INVALID")
        b60_conf = (b60.get(c, {}).get("B6_confirm") if b60 else None)
        rows[c] = dict(n=N, ground_truth_noconcept=gt_noconcept, old_confirm=old_conf, b6_0_confirm=b60_conf,
                       fm_confirm=fm_conf, fm_false_confirm=(fm_conf if gt_noconcept else None),
                       fm_true_confirm=(fm_conf if not gt_noconcept else None),
                       fm_no_actionable=fm_noact, fm_margin_lock=fm_lock, fm_sampler_invalid=fm_inv,
                       median_p_C_FM=med([r.get("p_C_FM_meanT") for r in recs]),
                       median_feasible_swaps=med([r.get("margin_feasible_swaps") for r in recs]),
                       median_unique_Cstar=med([r.get("unique_Cstar") for r in recs]),
                       median_propensity_auc=med([r.get("propensity_auc") for r in recs]),
                       median_frac_strata_single=med([r.get("frac_strata_single_condition") for r in recs]),
                       median_covariate_auc_gap=med([r.get("covariate_auc_gap") for r in recs]),
                       median_Cnull_sd_T=med([r.get("c_null_sd_T") for r in recs]))
        r = rows[c]
        print(f"{c:26s} {'NOCON' if gt_noconcept else 'CONC':>8} {old_conf:>4} {str(b60_conf):>5} {fm_conf:>9} {fm_noact:>8} "
              f"{fm_lock:>7} {fm_inv:>6} {r['median_p_C_FM']:>9.3f} {r['median_feasible_swaps']:>7.0f} {r['median_unique_Cstar']:>7.0f} {r['median_propensity_auc']:>5.2f}")

    print("\n=== DECISIVE: does fixed-margin drop the prior-shift false-confirm? (OLD / B6.0-plain / B6-FM) ===")
    for c in ["NULL_label", "NULL_cov_plus_label_soft"]:
        r = rows[c]
        print(f"  {c}: OLD {r['old_confirm']}/50 | B6.0 {r['b6_0_confirm']}/50 | B6-FM {r['fm_confirm']}/50")
    print("=== strong-cov must HOLD (B6.0 was 0/0) ===")
    for c in ["NULL_cov_strong_auc0.81", "NULL_cov_strong_auc0.94"]:
        r = rows[c]
        print(f"  {c}: OLD {r['old_confirm']}/50 | B6.0 {r['b6_0_confirm']}/50 | B6-FM {r['fm_confirm']}/50 (lock {r['fm_margin_lock']})")
    print("=== POS retention (B6.0 was 15/12; clean concept = OLD 12/13) ===")
    for c in ["POS_concept", "POS_concept_plus_cov"]:
        r = rows[c]
        print(f"  {c}: OLD {r['old_confirm']}/50 | B6.0 {r['b6_0_confirm']}/50 | B6-FM {r['fm_confirm']}/50 (lock {r['fm_margin_lock']}, med p_C {r['median_p_C_FM']:.3f})")

    # hard screen
    screen = dict(NULL_cov_soft=rows["NULL_cov_soft"]["fm_confirm"] <= 3,
                  NULL_cov_plus_label=rows["NULL_cov_plus_label_soft"]["fm_confirm"] <= 3,
                  strong81=rows["NULL_cov_strong_auc0.81"]["fm_confirm"] <= 3,
                  strong94=rows["NULL_cov_strong_auc0.94"]["fm_confirm"] <= 5,
                  NULL_label=rows["NULL_label"]["fm_confirm"] <= 3,
                  random=rows["random_label_control"]["fm_confirm"] <= 3,
                  POS_concept_pos=rows["POS_concept"]["fm_confirm"] > 0)
    passed = all(screen.values())
    nl = rows["NULL_label"]["fm_confirm"]
    verdict = ("B6-FM PASSES hard screen -> first real-EEG redesign worth scaling" if passed
               else f"B6-FM FAILS hard screen (NULL_label {nl}/50) -> fixed-margin+MARGINAL-propensity does NOT separate "
                    f"within-class covariate from concept (red-team-predicted); NOT a param-tune fix")
    print(f"\n  >>> hard screen: {screen}")
    print(f"  >>> B6-FM VERDICT: {verdict}")

    tables = dict(scope="B6-FM fixed-margin/class-preserving C-randomization canary; dev-only; NOT confirmatory; NO tag; NO validity claim",
                  base_seed=200_000_000, n_total=len(all_recs), n_valid_fidelity=len(dts),
                  margin_fidelity_max_err=max_margin_err, max_subject_count_err=max_count_err,
                  n_sampler_invalid=n_sampler_invalid, fidelity_max_dT=(max(dts) if dts else None),
                  per_condition=rows, hard_screen=screen, hard_screen_passed=bool(passed), verdict=verdict,
                  caveat="propensity is MARGINAL P(C|Z,S) (Y-free); covariate_auc_gap is marginal-only (blind to within-class covariate); B6 validity depends on the estimated law; NO confirmatory validity claim")
    json.dump(tables, open(f"{CDIR}/b6fm_canary_tables.json", "w"), indent=1, default=str)
    print(f"\n  saved {CDIR}/b6fm_canary_tables.json")


if __name__ == "__main__":
    main()
