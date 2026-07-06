"""B8.1 canary merge. Per-world B8 state distribution + provenance diagnostics + the PRE-REGISTERED screen:
contract-valid nulls B8_ALERT<=3/50 (prior_only=PRIMARY target vs B8.0 5/50; cov_plus_prior must STAY<=3/50);
violation worlds CONTRACT_INVALID high AND B8_ALERT<=3/50 (correct REFUSAL); POS_boundary B8_ALERT>0 (>=5 strong).
Reports the invalid-reason breakdown for violations + the AUC<=tau-yet-refused fraction for the quiet cells (provenance
closing the B8.0 stress gap). Fail-closed; contract-valid vs violating reported separately (NO pooling). Dev-only, NO tag."""
import os, sys, json
from collections import Counter
import numpy as np
sys.path.insert(0, "/home/infres/yinwang/realeeg_feas")
sys.path.insert(0, "/home/infres/yinwang/CMI_AAAI_csc")   # csc.mininfo for realeeg_b8_1 import
import realeeg_b8_1 as B8

CDIR = "/home/infres/yinwang/realeeg_feas/b8_1_canary"
WORLDS = list(B8.WORLDS)
N = 50
STATES = ["B8_CONCEPT_ALERT", "NO_ACTIONABLE_CONCEPT_EVIDENCE", "CONTRACT_INVALID_OR_UNIDENTIFIABLE",
          "INSUFFICIENT_LABELS", "SAMPLER_INVALID", "NEED_MORE_LABELS"]


def _read(p): return [json.loads(l) for l in open(p) if l.strip()]
def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and x == x]
    return float(np.median(xs)) if xs else float("nan")


def main():
    per = {}
    for w in WORLDS:
        p = f"{CDIR}/b8_1_canary_{w}_0.jsonl"
        if not os.path.exists(p):
            print(f"FAIL-CLOSED: missing {p}"); sys.exit(2)
        recs = _read(p)
        if any("__worker_error__" in r for r in recs):
            print(f"FAIL-CLOSED: worker-error rows in {w}"); sys.exit(2)
        if len(recs) != N or len({r["task_id"] for r in recs}) != N:
            print(f"FAIL-CLOSED: {w} count/dup"); sys.exit(2)
        per[w] = recs

    rows = {}
    print(f"{'world':34s} {'cls':>4} {'conc':>4} {'ALERT':>5} {'NOACT':>5} {'CINVAL':>6} {'valid':>5} {'medAUC':>6} {'medMatch':>8}")
    print("-" * 96)
    for w in WORLDS:
        recs = per[w]; kind, has_concept = B8.WORLDS[w]
        sc = Counter(str(r.get("b8_state")) for r in recs)
        alert = sc.get("B8_CONCEPT_ALERT", 0); noact = sc.get("NO_ACTIONABLE_CONCEPT_EVIDENCE", 0)
        cinval = sc.get("CONTRACT_INVALID_OR_UNIDENTIFIABLE", 0); insuf = sc.get("INSUFFICIENT_LABELS", 0)
        sampinv = sc.get("SAMPLER_INVALID", 0); needm = sc.get("NEED_MORE_LABELS", 0)
        valid = sum(1 for r in recs if r.get("contract_valid"))
        reason_ct = Counter(x for r in recs for x in r.get("contract_invalid_reasons", []))
        auc_le_tau = sum(1 for r in recs if r.get("auc_le_tau"))
        # quiet-cell key metric: refused-by-provenance WHILE AUC<=tau (AUC-only would have passed)
        prov_only_refused = sum(1 for r in recs if r.get("auc_le_tau")
                                and "assignment_not_following_schedule" in r.get("contract_invalid_reasons", [])
                                and "block_confounding_auc" not in r.get("contract_invalid_reasons", []))
        # DESIGN-RED-TEAM (Lens1): separate the two gates. mean-T-alone vs both-gate co-fire, among cohorts that
        # actually computed T (state ALERT/NO_ACTIONABLE). The 2nd-order mean-T residual leans safety on the studentized
        # AND-gate -> a mean-T-only drift toward 0.025 must be VISIBLE before it converts to a both-gate alert.
        tested = [r for r in recs if str(r.get("b8_state")) in ("B8_CONCEPT_ALERT", "NO_ACTIONABLE_CONCEPT_EVIDENCE")]
        a = B8.ALPHA_BUDGET
        meanT_alone = sum(1 for r in tested if r.get("p_exact_meanT", 1.0) <= a)
        stud_alone = sum(1 for r in tested if r.get("p_exact_stud", 1.0) <= a)
        both_gate = sum(1 for r in tested if r.get("p_exact_meanT", 1.0) <= a and r.get("p_exact_stud", 1.0) <= a)
        rows[w] = dict(n=N, world_class=kind, has_concept=has_concept, alert=alert, no_actionable=noact,
                       contract_invalid=cinval, insufficient=insuf, sampler_invalid=sampinv, need_more=needm,
                       contract_valid=valid, invalid_reason_breakdown=dict(reason_ct),
                       auc_le_tau=int(auc_le_tau), provenance_only_refused=int(prov_only_refused),
                       n_tested=len(tested), meanT_alone_alert=int(meanT_alone), stud_alone_alert=int(stud_alone),
                       both_gate_alert=int(both_gate),
                       median_auc=med([r.get("contract_within_block_C_Z_auc") for r in recs]),
                       median_match=med([r.get("provenance_match") for r in recs]),
                       median_support=med([r.get("contract_n_support_blocks") for r in recs]),
                       median_p_meanT=med([r.get("p_exact_meanT") for r in recs]),
                       median_p_stud=med([r.get("p_exact_stud") for r in recs]),
                       states=dict(sc))
        print(f"{w:34s} {kind[:4]:>4} {str(has_concept)[:4]:>4} {alert:>5} {noact:>5} {cinval:>6} {valid:>5} "
              f"{rows[w]['median_auc']:>6.3f} {rows[w]['median_match']:>8.3f}")
        # disjoint state accounting (fail-closed)
        tot = alert + noact + cinval + insuf + sampinv + needm
        if tot != N:
            print(f"FAIL-CLOSED: {w} states sum {tot} != {N}"); sys.exit(2)

    # PRE-REGISTERED screen (kind-specific, NO pooling)
    contract_nulls = [w for w in WORLDS if B8.WORLDS[w][0] == "contract" and not B8.WORLDS[w][1]]
    contract_pos = [w for w in WORLDS if B8.WORLDS[w][0] == "contract" and B8.WORLDS[w][1]]
    violations = [w for w in WORLDS if B8.WORLDS[w][0] == "violation"]
    print("\n=== A. CONTRACT-valid NULLS: B8_ALERT <= 3/50 (prior_only=PRIMARY target vs B8.0 5/50) ===")
    for w in contract_nulls:
        tag = "  <-- PRIMARY (B8.0=5/50)" if w == "CONTRACT_NULL_prior_only" else ("  <-- must STAY controlled (B8.0=1/50)" if w == "CONTRACT_NULL_cov_plus_prior" else "")
        print(f"    {w:34s} ALERT {rows[w]['alert']}/50 <=3:{rows[w]['alert'] <= 3}{tag}")
    print("=== B. CONTRACT-valid POS: B8_ALERT power (>0, >=5 strong) ===")
    for w in contract_pos:
        print(f"    {w:34s} ALERT {rows[w]['alert']}/50 (>0 power; >=5 strong)")
    print("=== C. CONTRACT-VIOLATING: CONTRACT_INVALID high AND B8_ALERT <= 3/50 (correct REFUSAL) ===")
    for w in violations:
        print(f"    {w:34s} INVALID {rows[w]['contract_invalid']}/50 | ALERT {rows[w]['alert']}/50 <=3:{rows[w]['alert'] <= 3} "
              f"| medAUC {rows[w]['median_auc']:.3f} | reasons {rows[w]['invalid_reason_breakdown']}")
    print("=== D. STRESS-GAP closure (quiet cells): refused-by-provenance WHILE AUC<=tau (AUC-only would PASS) ===")
    for w in ("VIOLATION_quiet_cov_no_concept", "VIOLATION_quiet_cov_plus_concept"):
        print(f"    {w:34s} AUC<=tau {rows[w]['auc_le_tau']}/50 | provenance-only-refused {rows[w]['provenance_only_refused']}/50 "
              f"| ALERT {rows[w]['alert']}/50 (has_concept={B8.WORLDS[w][1]})")
    print("=== E. TWO-GATE separation (mean-T-alone vs studentized-alone vs both-gate=ALERT; among cohorts that computed T) ===")
    print("        the 2nd-order mean-T residual (permuting C with Y fixed) leans prior_only safety on the studentized AND-gate")
    for w in contract_nulls + contract_pos:
        r = rows[w]
        print(f"    {w:34s} tested {r['n_tested']:>2}/50 | meanT-alone {r['meanT_alone_alert']:>2} | stud-alone {r['stud_alone_alert']:>2} | BOTH(=ALERT) {r['both_gate_alert']:>2}")

    null_ok = all(rows[w]["alert"] <= 3 for w in contract_nulls)
    random_ok = rows["CONTRACT_random_label"]["alert"] <= 1
    prior_ok = rows["CONTRACT_NULL_prior_only"]["alert"] <= 3
    mixed_ok = rows["CONTRACT_NULL_cov_plus_prior"]["alert"] <= 3
    # DESIGN-RED-TEAM (Lens3): contract-FIRST GUARANTEES violations -> CONTRACT_INVALID with ZERO alerts. Any violation
    # alert is a provenance LEAK = the pre-registered hard-stop "contract-invalid world alerts before provenance refusal".
    viol_no_alert = all(rows[w]["alert"] == 0 for w in violations)
    viol_leaks = {w: rows[w]["alert"] for w in violations if rows[w]["alert"] > 0}
    viol_refuse = all(rows[w]["contract_invalid"] >= 45 for w in violations)   # violations should overwhelmingly refuse
    pos = rows["CONTRACT_POS_boundary"]["alert"]
    quiet_stress = all(rows[w]["provenance_only_refused"] > 0 for w in ("VIOLATION_quiet_cov_no_concept", "VIOLATION_quiet_cov_plus_concept"))
    hard_stop_hit = bool(viol_leaks)
    screen = dict(contract_nulls_le3=bool(null_ok), random_label_le1=bool(random_ok),
                  prior_only_le3_PRIMARY=bool(prior_ok), mixed_cov_plus_prior_le3=bool(mixed_ok),
                  violations_zero_alert=bool(viol_no_alert), violations_refused=bool(viol_refuse),
                  pos_boundary_power=bool(pos > 0), pos_boundary_strong=bool(pos >= 5),
                  quiet_stress_gap_closed=bool(quiet_stress), HARD_STOP_violation_leak=hard_stop_hit)

    # RESULT-RED-TEAM (wvfdv8j1c) HONESTY CORRECTION: the decision-level SCREEN can be MET while "STRONG"/"FIXED"
    # OVER-CLAIMS. The prior-collider is CONTROLLED by the conservative studentized AND-gate, NOT fixed at the statistic
    # level (mean-T-alone residual is intact). Report the pooled null rate + mean-T floor so the redistribution/masking is
    # visible; never say FIXED/STRONG. Decision-level screen-met => "MEETS-PRE-REGISTERED-TARGETS-ON-EMULATOR" (caveated).
    pooled_null_alerts = sum(rows[w]["alert"] for w in contract_nulls); pooled_null_n = 50 * len(contract_nulls)
    prior_meanT_alone = rows["CONTRACT_NULL_prior_only"]["meanT_alone_alert"]   # the WORST cell (= B8.0's total)
    screen["pooled_null_alerts"] = f"{pooled_null_alerts}/{pooled_null_n}"
    screen["prior_only_meanT_alone_residual"] = f"{prior_meanT_alone}/50 (B8.0 both-gate was 5/50; the collider's mean-T signature is INTACT -- decision-level control leans on the studentized AND-gate)"
    ok_core = null_ok and prior_ok and mixed_ok and viol_no_alert and viol_refuse
    if hard_stop_hit:
        verdict = f"B8.1 HARD-STOP: violation ALERT leak {viol_leaks} -- provenance gate failed (contract-invalid world alerted before refusal). Fix before any science claim."
    elif ok_core and pos >= 5:
        verdict = ("B8.1 MEETS-PRE-REGISTERED-TARGETS-ON-EMULATOR (decision-level), NOT 'strong'/'fixed': all nulls controlled "
                   "+ mixed retained + violations refused (0 alerts) + POS>=5. CAVEATS (result-red-team): prior-collider is "
                   f"CONTROLLED by the studentized AND-gate NOT fixed -- mean-T-alone residual INTACT {prior_meanT_alone}/50; "
                   f"5->2 both-gate is CI-overlapping (Fisher p~0.22); pooled null FLAT {pooled_null_alerts}/{pooled_null_n} "
                   "(B8.0 7/200) = redistribution not tightening; POS modest (POS/50) & statistically UNCHANGED vs B8.0 4/50 "
                   "(p~0.36), separates from the POOLED floor not the mean-T floor; violations refused BY CONSTRUCTION (H3). "
                   "Emulator, single seed, n=50. Budget frontier LATER, NOT now.")
    elif ok_core and pos > 0:
        verdict = ("B8.1 MEETS-TARGETS-BUT-POS-WEAK (decision-level): nulls+mixed controlled + violations refused, POS>0 but <5. "
                   "Same CAVEATS: collider CONTROLLED-not-fixed (mean-T residual intact), pooled null flat, emulator/single-seed.")
    elif ok_core and pos == 0:
        verdict = "B8.1 SAFE-POWERLESS: nulls+violations controlled but POS=0 (contract too strong / label budget too small)"
    elif not prior_ok:
        verdict = f"B8.1 prior_only STILL HIGH ({rows['CONTRACT_NULL_prior_only']['alert']}/50): class-balanced contract did NOT even decision-level-control the collider -- inspect, do NOT retune p"
    elif not mixed_ok:
        verdict = f"B8.1 MIXED-CELL REGRESSION ({rows['CONTRACT_NULL_cov_plus_prior']['alert']}/50): B8.0's mixed control lost -- STOP + diagnose"
    elif not null_ok:
        verdict = "B8.1 NULL OVER-ALERT: a clean contract null (balanced/random_label) fired >3 -- inspect"
    else:
        verdict = "B8.1 MIXED -- inspect (violations under-refused or a validator/provenance issue)"
    if hard_stop_hit:
        print(f"\n  !!! HARD-STOP: violation ALERT leak {viol_leaks} (contract-first must give 0 violation alerts) !!!")
    print(f"\n  >>> screen: {screen}")
    print(f"  >>> B8.1 VERDICT: {verdict}")
    print(f"  >>> KEY vs B8.0 (n=50, CIs wide): prior_only both-gate {rows['CONTRACT_NULL_prior_only']['alert']}/50 (B8.0 5/50; "
          f"CI-overlapping) | prior_only mean-T-ALONE {prior_meanT_alone}/50 (residual INTACT vs B8.0 5/50) | "
          f"cov_plus_prior {rows['CONTRACT_NULL_cov_plus_prior']['alert']}/50 (B8.0 1/50) | pooled null {pooled_null_alerts}/{pooled_null_n} (B8.0 7/200, FLAT) | "
          f"POS_boundary {pos}/50 (B8.0 4/50; p~0.36 UNCHANGED)")

    tables = dict(scope="B8.1 class-balanced randomized-audit contract canary; development-only; NOT confirmatory; NOT validation (Lee2019 emulator); NO tag",
                  base_seed=420_000_000, n_per_world=N, per_world=rows, screen=screen, verdict=verdict,
                  vs_b8_0=dict(prior_only_both_gate=dict(b8_1=rows["CONTRACT_NULL_prior_only"]["alert"], b8_0="5/50", note="CI-overlapping; Fisher 5->2 p~0.22 not decisive at n=50"),
                               prior_only_meanT_alone=dict(b8_1=prior_meanT_alone, b8_0="5/50", note="RESIDUAL INTACT -- the collider's mean-T signature is unchanged; decision-level control is via the studentized AND-gate (masking), NOT a statistic-level fix"),
                               cov_plus_prior=dict(b8_1=rows["CONTRACT_NULL_cov_plus_prior"]["alert"], b8_0="1/50", note="mixed control genuinely retained"),
                               pooled_null=dict(b8_1=f"{pooled_null_alerts}/{pooled_null_n}", b8_0="7/200", note="FLAT -- aggregate null control unchanged; only prior_only moved (redistribution within noise; balanced 0->1, random_label 1->2 got worse)"),
                               pos_boundary=dict(b8_1=pos, b8_0="4/50", note="statistically UNCHANGED (Fisher 4->8 p~0.36); modest 16% (84% miss); separates from the POOLED null floor (p~0.002) but NOT from prior_only's mean-T floor 5/50 (p~0.28)")),
                  result_redteam=dict(workflow="wvfdv8j1c", accounting="PASS (clean, 0 mismatches, disjoint seeds, isolation clean)",
                                      science="MINOR_ISSUE -- 'STRONG'/'FIXED' OVER-CLAIMED; decision-level screen met but relabel to MEETS-PRE-REGISTERED-TARGETS-ON-EMULATOR with caveats",
                                      honest_label="MEETS-PRE-REGISTERED-TARGETS-ON-EMULATOR (decision-level); prior-collider CONTROLLED by studentized AND-gate NOT fixed (mean-T residual intact); pooled null flat; POS modest+unchanged vs B8.0; violations refused by construction (H3); emulator/single-seed/n=50"),
                  core_change="null stratifier: observed post-treatment Y (B8.0 collider) -> pre-assignment design_class Dc (=cued class y0); class-balanced randomization within (block,Dc); hard provenance gate",
                  provenance_gates_exercised="Of the 5 hard gates, H1(table exists)/H2(hash integrity)/H5(schedule balance) are NON-DISCRIMINATIVE BY CONSTRUCTION in this emulator (H2/H5 always pass; no world corrupts the table). The canary exercises only H3(schedule-adherence, the sole novel discriminator carrying every violation refusal) + H4(support, fires on condition_lock) + D1(AUC, secondary backstop). Read as an H3/H4 test, NOT a 5-gate validation.",
                  two_gate_note="2nd-order mean-T residual: permuting C with Y held fixed breaks the C->Y prior main-effect link, so the null is exact only for the sharp interaction-null (h0 absorbs the main effect). prior_only safety leans on the studentized AND-gate; section E reports meanT-alone vs both-gate so a mean-T drift toward 0.025 is visible.",
                  scope_limits="emulator (Lee2019 SM16 geometry), semi-synthetic, n=50/world, single base seed; provenance H3 = pre-registration schedule-adherence with ZERO tolerance (verifies a known randomization; does NOT discover confounding from observational data; robustness to noisy/graded real-world adherence out of scope); _schedule_hash pins per-(block,Dc) COUNTS not per-trial assignment (H3 is the per-trial gate); AUC validator linear top-8-PC (secondary)")
    json.dump(tables, open(f"{CDIR}/b8_1_canary_tables.json", "w"), indent=1, default=str)
    print(f"\n  saved {CDIR}/b8_1_canary_tables.json")


if __name__ == "__main__":
    main()
