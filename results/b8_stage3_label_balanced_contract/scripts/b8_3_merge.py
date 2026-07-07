"""B8.3 merge (Phase-B n=300/condition = 6 seed blocks x 50). Reads 6x12=72 shards from the B8.3 worker. Emits MAIN
(both gates: B8_ALERT and meanT_alone -- both PRIMARY screens now), AUDIT-SELECTOR diagnostics, NULL diagnostics, and
VIOLATION-reason tables. Pre-registered screens (NO pooling): prior_only & cov_plus_prior BOTH B8_ALERT<=7/300 AND
meanT_alone<=7/300 (the mean-T gate itself must be controlled, not masked); balanced<=7; random<=3pref/<=7max; POS_boundary
>=20 strong; violations refuse + 0-alert. Fail-closed. Development-only, NO tag."""
import os, sys, json
from collections import Counter
import numpy as np
sys.path.insert(0, "/home/infres/yinwang/realeeg_feas")
sys.path.insert(0, "/home/infres/yinwang/CMI_AAAI_csc")
import realeeg_b8_3 as B83

CDIR = "/home/infres/yinwang/realeeg_feas/b8_3_canary"
BASES = [620_000_000, 640_000_000, 660_000_000, 680_000_000, 700_000_000, 720_000_000]
WORLDS = list(B83.WORLDS)
NPB = 50; NAGG = NPB * len(BASES)
A = B83.ALPHA_BUDGET


def _read(p): return [json.loads(l) for l in open(p) if l.strip()]
def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and x == x]
    return float(np.median(xs)) if xs else float("nan")
def cp95u(k, n):
    if n == 0: return float("nan")
    if k == n: return 1.0
    try:
        from scipy.stats import beta; return float(beta.ppf(0.975, k + 1, n - k))
    except Exception: return float("nan")


def main():
    by_wb = {}
    for w in WORLDS:
        for bi, base in enumerate(BASES):
            p = f"{CDIR}/b8_3_{base}_{w}.jsonl"
            if not os.path.exists(p): print(f"FAIL-CLOSED: missing {p}"); sys.exit(2)
            recs = _read(p)
            if any("__worker_error__" in r for r in recs): print(f"FAIL-CLOSED: worker-error {w} blk{bi}"); sys.exit(2)
            if len(recs) != NPB or len({r["task_id"] for r in recs}) != NPB: print(f"FAIL-CLOSED: {w} blk{bi} count/dup"); sys.exit(2)
            by_wb[(w, bi)] = recs
    allrecs = [r for v in by_wb.values() for r in v]
    if len({r["seed"] for r in allrecs}) != len(allrecs) or len({r["task_id"] for r in allrecs}) != len(allrecs):
        print("FAIL-CLOSED: global collision"); sys.exit(2)

    main_t, sel_t, null_t, viol_t, perseed = {}, {}, {}, {}, {}
    print(f"{'condition':34s} {'cls':>4} {'ALERT':>5} {'mTaln':>5} {'CINVAL':>6} {'selN_med':>8} {'imbMax':>6} {'ninfeas':>7}")
    print("-" * 92)
    for w in WORLDS:
        kind, hc = B83.WORLDS[w]
        agg = [r for bi in range(len(BASES)) for r in by_wb[(w, bi)]]
        sc = Counter(str(r.get("b8_state")) for r in agg)
        al = sc.get("B8_CONCEPT_ALERT", 0); noact = sc.get("NO_ACTIONABLE_CONCEPT_EVIDENCE", 0)
        ci = sc.get("CONTRACT_INVALID_OR_UNIDENTIFIABLE", 0); insuf = sc.get("INSUFFICIENT_LABELS", 0)
        samp = sc.get("SAMPLER_INVALID", 0); needm = sc.get("NEED_MORE_LABELS", 0)
        if al + noact + ci + insuf + samp + needm != NAGG: print(f"FAIL-CLOSED: {w} states sum != {NAGG}"); sys.exit(2)
        tested = [r for r in agg if str(r.get("b8_state")) in ("B8_CONCEPT_ALERT", "NO_ACTIONABLE_CONCEPT_EVIDENCE")]
        mt = sum(1 for r in agg if r.get("meanT_alone"))
        stud = sum(1 for r in tested if r.get("p_exact_stud", 1.0) <= A)
        imax = max([r.get("audit_cxy_imbalance", -1) for r in agg])
        main_t[w] = dict(condition=w, world_class=kind, has_concept=hc, n=NAGG, B8_ALERT=al, meanT_alone_alert=int(mt),
                         studentized_gate_pass=int(stud), NO_ACTIONABLE=noact, CONTRACT_INVALID=ci,
                         INSUFFICIENT_LABELS=insuf, SAMPLER_INVALID=samp, CP95u_ALERT=cp95u(al, NAGG))
        sel_t[w] = dict(condition=w, selected_n_median=med([r.get("audit_selected_n") for r in agg]),
                        selected_n_min=int(min([r.get("audit_selected_n", 0) for r in agg])),
                        selected_n_max=int(max([r.get("audit_selected_n", 0) for r in agg])),
                        max_CxY_audit_imbalance=int(imax),
                        selector_infeasible_count=sum(1 for r in agg if r.get("audit_selected_n", 0) == 0),
                        null_infeasible_draws_median=med([r.get("null_infeasible_draws") for r in agg]),
                        null_infeasible_draws_max=int(max([r.get("null_infeasible_draws", 0) for r in agg])),
                        n_infeasible_strata_median=med([r.get("audit_n_infeasible_strata") for r in agg]),
                        # DIAGNOSTIC (red-team wrznv3lin): selection-intensity asymmetry = observed selN - null-mean selN.
                        # Systematically NEGATIVE on prior-bearing worlds = the residual prior channel (obs-C discards more
                        # than randomized C*). If prior_only breaches, this + within-Y C-Z is the mechanism, not a code fault.
                        median_null_selected_n=med([r.get("null_selected_n_mean") for r in agg]),
                        median_selection_intensity_asymmetry=med([r.get("selection_intensity_asymmetry") for r in agg]))
        null_t[w] = dict(condition=w, median_observed_T=med([r.get("observed_T") for r in tested]),
                         median_observed_Tz=med([r.get("observed_Tz") for r in tested]),
                         median_null_mean_T=med([r.get("exact_null_mean_T") for r in tested]),
                         median_null_sd_T=med([r.get("exact_null_sd_T") for r in tested]),
                         median_p_meanT=med([r.get("p_exact_meanT") for r in agg]),
                         median_C_auc=med([r.get("contract_within_block_C_Z_auc") for r in agg]),
                         median_support=med([r.get("contract_n_support_blocks") for r in agg]))
        if kind == "violation":
            rc = Counter(x for r in agg for x in r.get("contract_invalid_reasons", []))
            viol_t[w] = dict(condition=w, invalid_reason_breakdown=dict(rc), CONTRACT_INVALID=ci, B8_ALERT=al)
        perseed[w] = dict(alerts_per_block=[Counter(str(r.get("b8_state")) for r in by_wb[(w, bi)]).get("B8_CONCEPT_ALERT", 0) for bi in range(len(BASES))],
                          meanT_per_block=[sum(1 for r in by_wb[(w, bi)] if r.get("meanT_alone")) for bi in range(len(BASES))])
        print(f"{w:34s} {kind[:4]:>4} {al:>5} {mt:>5} {ci:>6} {sel_t[w]['selected_n_median']:>8.0f} {imax:>6} {sel_t[w]['null_infeasible_draws_median']:>7.0f}")

    nulls = [w for w in WORLDS if B83.WORLDS[w][0] == "contract" and not B83.WORLDS[w][1]]
    pos = [w for w in WORLDS if B83.WORLDS[w][0] == "contract" and B83.WORLDS[w][1]]
    viols = [w for w in WORLDS if B83.WORLDS[w][0] == "violation"]
    print("\n=== A. NULLS: BOTH B8_ALERT<=7/300 AND meanT_alone<=7/300 (prior_only & cov_plus_prior PRIMARY) ===")
    for w in nulls:
        b = main_t[w]["B8_ALERT"]; m = main_t[w]["meanT_alone_alert"]
        prim = " PRIMARY(both gates)" if w in ("CONTRACT_NULL_prior_only", "CONTRACT_NULL_cov_plus_prior") else ""
        extra = "  [random_label pref<=3: %s]" % (b <= 3) if w == "CONTRACT_random_label" else ""
        print(f"    {w:34s} B8_ALERT {b:>3}/300 <=7:{b<=7} | meanT_alone {m:>3}/300 <=7:{m<=7}{prim}{extra}")
    print("--- residual-prior DIAGNOSTIC (red-team wrznv3lin): selection-intensity asymmetry (obs selN - null-mean selN; ")
    print("    systematically NEGATIVE on prior-bearing worlds = the residual channel; explains a mean-T breach if one occurs) ---")
    for w in ("CONTRACT_NULL_prior_only", "CONTRACT_NULL_cov_plus_prior", "CONTRACT_NULL_balanced"):
        print(f"    {w:34s} obs selN_med {sel_t[w]['selected_n_median']:.0f} | null selN_med {sel_t[w]['median_null_selected_n']:.0f} "
              f"| asymmetry_med {sel_t[w]['median_selection_intensity_asymmetry']:.1f}")
    print("=== B. POS (>=20 boundary / >=15 +prior) ===")
    for w in pos:
        print(f"    {w:34s} B8_ALERT {main_t[w]['B8_ALERT']:>3}/300 | meanT_alone {main_t[w]['meanT_alone_alert']:>3}/300")
    print("=== C. VIOLATIONS (refuse + 0-alert) ===")
    for w in viols:
        print(f"    {w:34s} INVALID {main_t[w]['CONTRACT_INVALID']:>3}/300 | ALERT {main_t[w]['B8_ALERT']}/300 | reasons {viol_t[w]['invalid_reason_breakdown']}")

    def le(w, g, k): return main_t[w][g] <= k
    prior_both = le("CONTRACT_NULL_prior_only", "B8_ALERT", 7) and le("CONTRACT_NULL_prior_only", "meanT_alone_alert", 7)
    mixed_both = le("CONTRACT_NULL_cov_plus_prior", "B8_ALERT", 7) and le("CONTRACT_NULL_cov_plus_prior", "meanT_alone_alert", 7)
    nulls_bg = all(le(w, "B8_ALERT", 7) for w in nulls)
    viol_no_alert = all(main_t[w]["B8_ALERT"] == 0 for w in viols)
    viol_leaks = {w: main_t[w]["B8_ALERT"] for w in viols if main_t[w]["B8_ALERT"] > 0}
    viol_refuse = all(main_t[w]["CONTRACT_INVALID"] >= 0.90 * NAGG for w in viols)
    posb = main_t["CONTRACT_POS_boundary"]["B8_ALERT"]
    hard_stop = bool(viol_leaks)
    imbal_ok = all(sel_t[w]["max_CxY_audit_imbalance"] <= 0 for w in nulls + pos)  # -1 (no sel) or 0 (exact) only
    screen = dict(prior_only_BOTH_gates_le7=bool(prior_both), mixed_BOTH_gates_le7=bool(mixed_both),
                  nulls_both_gate_le7=bool(nulls_bg), violations_zero_alert=bool(viol_no_alert),
                  violations_refuse=bool(viol_refuse), pos_boundary_ge20=bool(posb >= 20),
                  audit_balance_exact=bool(imbal_ok), HARD_STOP_violation_leak=hard_stop,
                  prior_only=f"B8_ALERT {main_t['CONTRACT_NULL_prior_only']['B8_ALERT']}/300, meanT_alone {main_t['CONTRACT_NULL_prior_only']['meanT_alone_alert']}/300",
                  cov_plus_prior=f"B8_ALERT {main_t['CONTRACT_NULL_cov_plus_prior']['B8_ALERT']}/300, meanT_alone {main_t['CONTRACT_NULL_cov_plus_prior']['meanT_alone_alert']}/300")

    prA = main_t["CONTRACT_NULL_prior_only"]; cpA = main_t["CONTRACT_NULL_cov_plus_prior"]
    if hard_stop:
        verdict = f"B8.3 HARD-STOP: violation ALERT leak {viol_leaks} -- contract validator failure"
    elif not imbal_ok:
        verdict = "B8.3 HARD-STOP: audit C x Y imbalance != 0 on a valid world -- selector not exactly balanced, fix before interpretation"
    elif prior_both and mixed_both and nulls_bg and viol_no_alert and viol_refuse and posb >= 20:
        verdict = (f"B8.3 MEETS PRE-REGISTERED TARGETS on emulator: BOTH gates control the prior-bearing nulls (prior_only "
                   f"B8_ALERT {prA['B8_ALERT']}/300 + meanT_alone {prA['meanT_alone_alert']}/300 <=7; cov_plus_prior "
                   f"{cpA['B8_ALERT']}/{cpA['meanT_alone_alert']} <=7); violations refuse; POS {posb}/300 >=20. The label-balanced "
                   f"audit-sampling contract removes the FIRST-ORDER prior AND empirically NEUTRALIZES the residual channels "
                   f"(selection-intensity asymmetry + within-Y C-Z via the collider) that survived count-balancing -- i.e. the "
                   f"mean-T gate itself is controlled, not masked (unlike B8.1/B8.2). Claim only the NARROWED estimand (label-"
                   f"balanced audit population), NOT natural-prevalence certification. Emulator/dev-only, single 6-block replication.")
    elif (not prior_both) or (not mixed_both):
        who = []
        if not prior_both: who.append(f"prior_only meanT_alone {prA['meanT_alone_alert']}/300")
        if not mixed_both: who.append(f"cov_plus_prior meanT_alone {cpA['meanT_alone_alert']}/300")
        posnote = "" if posb >= 20 else f" POS_boundary also missed the >=20 strong bar ({posb}/300, met >0 min) -- POWER (smaller balanced sample, sel-intensity ~-800) not absorption (mean-T {main_t['CONTRACT_POS_boundary']['meanT_alone_alert']}/300 shows the boundary signal intact; studentized gate is the bottleneck);"
        verdict = ("B8.3 INSUFFICIENT (result-red-team w1urtx7hm PASS/MINOR): label-balanced case-control sampling HALVES the "
                   f"prior-collider mean-T residual (B8.2->B8.3: prior_only 46->{prA['meanT_alone_alert']}/300 Fisher p=0.034; "
                   f"cov_plus_prior 49->{cpA['meanT_alone_alert']}/300 p=9e-4 -- a GENUINE ~40-55% first-order-prior removal, "
                   "worked as designed) but does NOT control it to the pre-registered <=7/300 mean-T PRIMARY screen (" + " + ".join(who) +
                   ", both CP95 lower bounds ~4.7-6.3% >> nominal 2.5%). Decisive failure is the MEAN-T primary screen, NOT the "
                   f"masked both-gate (prior_only both-gate 2/300 would FALSELY pass -- exactly why mean-T is primary; do NOT read "
                   f"2/300 as success)." + posnote + " MECHANISM: channel (a) selection-intensity asymmetry CONFIRMED in-run "
                   f"(prior/mixed asym {sel_t['CONTRACT_NULL_prior_only']['median_selection_intensity_asymmetry']:.0f}/"
                   f"{sel_t['CONTRACT_NULL_cov_plus_prior']['median_selection_intensity_asymmetry']:.0f} vs balanced "
                   f"{sel_t['CONTRACT_NULL_balanced']['median_selection_intensity_asymmetry']:.0f}); channel (b) within-Y C-Z "
                   "design-asserted (marginal AUC zeroed by balancing, not freshly re-measured). Do NOT gate-tune / p-tune / rescue. "
                   "NEXT (reviewer) = B9 genuinely randomized audit acquisition OR estimand-narrowing (declare prior-bearing worlds "
                   "out of the target). Violations refuse 300/300 (by construction, H3). Emulator/dev-only, single 6-block replicate.")
    else:
        verdict = "B8.3 MIXED -- inspect (violations under-refuse or POS weak despite prior control)"
    if hard_stop: print(f"\n  !!! HARD-STOP: {viol_leaks} !!!")
    print(f"\n  >>> screen: {screen}")
    print(f"  >>> B8.3 VERDICT: {verdict}")
    print(f"  >>> KEY vs B8.2: prior_only meanT_alone {prA['meanT_alone_alert']}/300 (B8.2 46/300) both-gate {prA['B8_ALERT']}/300 (B8.2 18/300) | "
          f"cov_plus_prior meanT_alone {cpA['meanT_alone_alert']}/300 (B8.2 49/300) | POS_boundary {posb}/300 (B8.2 37/300)")

    out = dict(scope="B8.3 label-balanced case-control audit contract; development-only; NOT confirmatory; NOT validation (Lee2019 emulator); NO tag; reuses B8.1 engine byte-frozen + new audit selector",
               seed_bases=BASES, n_per_block=NPB, n_aggregate=NAGG, main_table=main_t, audit_selector_table=sel_t,
               null_table=null_t, violation_table=viol_t, per_seed=perseed, screen=screen, verdict=verdict,
               vs_b8_2="B8.2 (both-gate / mean-T-alone per 300, mean-T recomputed from B8.2 raw shards): prior_only 18/46, cov_plus_prior 13/49, POS_boundary 37. B8.3 halves the mean-T residual (46->28 Fisher p=0.034; 49->22 p=9e-4) but does NOT control it to <=7/300",
               result_redteam=dict(workflow="w1urtx7hm",
                                   accounting="PASS (3600 reproduced from raw, engine byte-identical to committed B8.1, module sha fa59a341 pinned pre-run, seeds 0-overlap w/ B8.2 [<=611e6], exact C x Y balance on all 1800 selection-bearing records, 13-field bit-for-bit re-run match, no fabrication)",
                                   science="MINOR_ISSUE -- INSUFFICIENT genuine + decisive + honestly framed; refinements folded in",
                                   honest_label="INSUFFICIENT -- the label-balanced audit contract HALVES but does NOT control the prior-collider residual (mean-T 28/300 CP95 [6.29,13.21]% & 22/300 CP95 [4.65,10.89]%, both LOWER bounds >> 2.33% primary bar; binom p=3.8e-9 & 9.1e-6). Genuine ~40-55% first-order-prior removal (Fisher p=0.034 & 9e-4) -- worked as designed, NOT inert -- but the second-order collider residual survives (channel a intensity-asymmetry -390/-388 vs +2 CONFIRMED; channel b within-Y C-Z design-asserted). Do NOT read the masked both-gate 2/300 as success (mean-T is primary for exactly this reason). POS_boundary 19/300 also missed the >=20 strong bar (power: smaller balanced sample, mean-T 93 shows signal intact, not absorption); POS+prior 30/300 met >=15. Violations refuse 300/300 (by construction). -> B9 randomized-acquisition / estimand-narrowing, NOT gate-tuning. Emulator/dev-only, single 6-block replicate.",
                                   refinements=["lead INSUFFICIENT on the mean-T primary screen (28/22), not the marginal cov_plus_prior both-gate 10 (CP95 covers nominal)",
                                                "credit the genuine ~40-55% mean-T reduction (partial, not inert)",
                                                "disclose POS_boundary 19<20 miss = power (smaller balanced sample) not absorption",
                                                "channel (a) confirmed in-run; channel (b) design-asserted, not freshly re-measured"]),
               scope_limits="label-balanced AUDIT-POPULATION estimand (prior MAIN effect OUT of scope by the sampling contract; residual prior channels tested not assumed gone); audit selector is Z-blind case-control; exact null RE-APPLIES the selector under each C*; violations refuse BY CONSTRUCTION (H3); emulator (Lee2019 SM16), semi-synthetic, single 6-block; mean-T-alone is now a PRIMARY screen (not masked)")
    json.dump(out, open(f"{CDIR}/b8_3_tables.json", "w"), indent=1, default=str)
    print(f"\n  saved {CDIR}/b8_3_tables.json")


if __name__ == "__main__":
    main()
