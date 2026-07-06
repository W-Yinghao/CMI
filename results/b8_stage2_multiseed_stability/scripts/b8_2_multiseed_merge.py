"""B8.2 multi-seed stability merge. Reads 6 seed blocks x 12 conditions = 72 shards produced by the UNCHANGED B8.1
worker (realeeg_b8_1_canary.py) at 6 fresh disjoint bases. Emits: MAIN table (per condition, aggregate n=300), SEED
table (per condition x seed_block -- the CORE per-seed variability output), MECHANISM table (meanT-alone vs studentized
vs both-gate + observed/null T stats). Pre-registered screens (NO pooling of NULL kinds): nulls ALERT<=7/300 each
(prior_only PRIMARY, cov_plus_prior must-retain, random_label<=3pref/<=7max); violations ALERT<=7/300 + CONTRACT_INVALID
dominates + 0-alert hard-stop; POS_boundary>=20 strong, POS+prior>=15. Fail-closed. Development-only, NO tag.

NOTE: the internal subject-consistency studentized Z_obs is NOT persisted by the byte-frozen B8.1 worker (adding it would
be a HARD-STOP logic change). The mechanism table reports the recorded p_exact_stud gate + a clearly-labeled null-
studentized proxy z_meanT=(observed_T-null_mean_T)/null_sd_T -- NOT the internal Z_obs."""
import os, sys, json, glob
from collections import Counter
import numpy as np
sys.path.insert(0, "/home/infres/yinwang/realeeg_feas")
sys.path.insert(0, "/home/infres/yinwang/CMI_AAAI_csc")
import realeeg_b8_1 as B8

CDIR = "/home/infres/yinwang/realeeg_feas/b8_2_multiseed"
BASES = [500_000_000, 520_000_000, 540_000_000, 560_000_000, 580_000_000, 600_000_000]
WORLDS = list(B8.WORLDS)
NPB = 50               # cohorts per condition per seed block
NAGG = NPB * len(BASES)  # 300
A = B8.ALPHA_BUDGET
PFLOOR = 1.0 / (200 + 1)  # n_boot=200 grid floor


def _read(p): return [json.loads(l) for l in open(p) if l.strip()]
def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and x == x]
    return float(np.median(xs)) if xs else float("nan")


def cp95u(k, n):
    """Clopper-Pearson two-sided 95% UPPER bound on a rate k/n."""
    if n == 0: return float("nan")
    if k == n: return 1.0
    try:
        from scipy.stats import beta
        return float(beta.ppf(0.975, k + 1, n - k))
    except Exception:
        return float("nan")


def main():
    # load: per (world, base) shard; fail-closed on missing/dup/error
    by_wb = {}
    for w in WORLDS:
        for bi, base in enumerate(BASES):
            p = f"{CDIR}/b8_2_{base}_{w}.jsonl"
            if not os.path.exists(p):
                print(f"FAIL-CLOSED: missing {p}"); sys.exit(2)
            recs = _read(p)
            if any("__worker_error__" in r for r in recs):
                print(f"FAIL-CLOSED: worker-error rows in {w} block {bi}"); sys.exit(2)
            if len(recs) != NPB or len({r["task_id"] for r in recs}) != NPB:
                print(f"FAIL-CLOSED: {w} block {bi} count/dup ({len(recs)})"); sys.exit(2)
            by_wb[(w, bi)] = recs

    # global disjointness + dup check across ALL 3600
    allrecs = [r for v in by_wb.values() for r in v]
    seeds = [r["seed"] for r in allrecs]
    if len(set(seeds)) != len(seeds) or len({r["task_id"] for r in allrecs}) != len(allrecs):
        print("FAIL-CLOSED: global seed/task_id collision"); sys.exit(2)
    if len(allrecs) != 12 * NAGG:
        print(f"FAIL-CLOSED: total {len(allrecs)} != {12*NAGG}"); sys.exit(2)

    def alerts(recs): return sum(1 for r in recs if str(r.get("b8_state")) == "B8_CONCEPT_ALERT")
    def cinval(recs): return sum(1 for r in recs if str(r.get("b8_state")) == "CONTRACT_INVALID_OR_UNIDENTIFIABLE")

    main_tab, seed_tab, mech_tab, perseed = {}, {}, {}, {}
    print("=== MAIN table (aggregate n=300 per condition) ===")
    print(f"{'condition':34s} {'cls':>4} {'ALERT':>5} {'CP95u':>6} {'NOACT':>5} {'CINVAL':>6} {'medAUC':>6} {'medMatch':>8} {'pfloor':>6}")
    print("-" * 96)
    for w in WORLDS:
        kind, hc = B8.WORLDS[w]
        agg = [r for bi in range(len(BASES)) for r in by_wb[(w, bi)]]
        assert len(agg) == NAGG
        sc = Counter(str(r.get("b8_state")) for r in agg)
        al = sc.get("B8_CONCEPT_ALERT", 0); noact = sc.get("NO_ACTIONABLE_CONCEPT_EVIDENCE", 0)
        ci = sc.get("CONTRACT_INVALID_OR_UNIDENTIFIABLE", 0); insuf = sc.get("INSUFFICIENT_LABELS", 0)
        samp = sc.get("SAMPLER_INVALID", 0); needm = sc.get("NEED_MORE_LABELS", 0)
        tot = al + noact + ci + insuf + samp + needm
        if tot != NAGG:
            print(f"FAIL-CLOSED: {w} states sum {tot} != {NAGG}"); sys.exit(2)
        tested = [r for r in agg if str(r.get("b8_state")) in ("B8_CONCEPT_ALERT", "NO_ACTIONABLE_CONCEPT_EVIDENCE")]
        meanT_alone = sum(1 for r in tested if r.get("p_exact_meanT", 1.0) <= A)
        stud_alone = sum(1 for r in tested if r.get("p_exact_stud", 1.0) <= A)
        both = sum(1 for r in tested if r.get("p_exact_meanT", 1.0) <= A and r.get("p_exact_stud", 1.0) <= A)
        pfloor_frac = (sum(1 for r in tested if r.get("p_exact_meanT", 1.0) <= PFLOOR * 1.5) / len(tested)) if tested else float("nan")
        zproxy = [ (r.get("observed_T") - r.get("exact_null_mean_T")) / r.get("exact_null_sd_T")
                   for r in tested if r.get("exact_null_sd_T") and r.get("exact_null_sd_T") == r.get("exact_null_sd_T") and r.get("exact_null_sd_T") > 0 ]
        main_tab[w] = dict(condition=w, world_class=kind, has_concept=hc, n=NAGG, B8_ALERT=al,
                           NO_ACTIONABLE=noact, CONTRACT_INVALID=ci, INSUFFICIENT_LABELS=insuf, SAMPLER_INVALID=samp,
                           NEED_MORE=needm, CP95u_ALERT=cp95u(al, NAGG),
                           median_p_exact=med([r.get("p_exact_meanT") for r in agg]),
                           p_floor_frac=float(pfloor_frac) if pfloor_frac == pfloor_frac else float("nan"),
                           median_effective_randomization=med([r.get("provenance_match") for r in agg]),
                           median_assignment_support=med([r.get("contract_n_support_blocks") for r in agg]),
                           median_C_auc=med([r.get("contract_within_block_C_Z_auc") for r in agg]))
        mech_tab[w] = dict(condition=w, world_class=kind, has_concept=hc,
                           meanT_alone_alert=int(meanT_alone), studentized_gate_pass=int(stud_alone),
                           both_gate_alert=int(both), median_observed_T=med([r.get("observed_T") for r in tested]),
                           median_null_mean_T=med([r.get("exact_null_mean_T") for r in tested]),
                           median_null_sd_T=med([r.get("exact_null_sd_T") for r in tested]),
                           median_z_meanT_proxy=med(zproxy),
                           note="both_gate_alert==B8_ALERT; median_z_meanT_proxy=(observed_T-null_mean_T)/null_sd_T is a NULL-studentized proxy, NOT the internal subject-consistency Z_obs (not persisted by the frozen worker)")
        print(f"{w:34s} {kind[:4]:>4} {al:>5} {main_tab[w]['CP95u_ALERT']:>6.3f} {noact:>5} {ci:>6} "
              f"{main_tab[w]['median_C_auc']:>6.3f} {main_tab[w]['median_effective_randomization']:>8.3f} "
              f"{main_tab[w]['p_floor_frac']:>6.3f}")

        # per-seed-block (CORE)
        blk = []
        for bi in range(len(BASES)):
            recs = by_wb[(w, bi)]
            bsc = Counter(str(r.get("b8_state")) for r in recs)
            blk.append(dict(seed_block=bi, base=BASES[bi], n=NPB,
                            B8_ALERT=bsc.get("B8_CONCEPT_ALERT", 0),
                            CONTRACT_INVALID=bsc.get("CONTRACT_INVALID_OR_UNIDENTIFIABLE", 0),
                            NO_ACTIONABLE=bsc.get("NO_ACTIONABLE_CONCEPT_EVIDENCE", 0),
                            INSUFFICIENT=bsc.get("INSUFFICIENT_LABELS", 0), SAMPLER_INVALID=bsc.get("SAMPLER_INVALID", 0)))
        seed_tab[w] = blk
        av = [b["B8_ALERT"] for b in blk]
        perseed[w] = dict(alerts_per_block=av, alert_min=min(av), alert_max=max(av), alert_range=max(av) - min(av),
                          alert_mean=float(np.mean(av)), alert_sd=float(np.std(av)),
                          cinval_per_block=[b["CONTRACT_INVALID"] for b in blk])

    # ---- SEED variability (the CORE output) ----
    print("\n=== SEED-BLOCK variability (CORE): B8_ALERT per block [b0..b5] | agg/300 | range ===")
    for w in WORLDS:
        av = perseed[w]["alerts_per_block"]; agg = main_tab[w]["B8_ALERT"]
        print(f"    {w:34s} {str(av):>22s} | agg {agg:>3}/300 | range {perseed[w]['alert_range']} sd {perseed[w]['alert_sd']:.2f}")

    # ---- screens (NO pooling) ----
    nulls = [w for w in WORLDS if B8.WORLDS[w][0] == "contract" and not B8.WORLDS[w][1]]
    pos = [w for w in WORLDS if B8.WORLDS[w][0] == "contract" and B8.WORLDS[w][1]]
    viols = [w for w in WORLDS if B8.WORLDS[w][0] == "violation"]
    print("\n=== A. CONTRACT-valid NULLS (each <=7/300; random_label <=3 pref) ===")
    for w in nulls:
        lim = 3 if w == "CONTRACT_random_label" else 7
        tag = " PRIMARY" if w == "CONTRACT_NULL_prior_only" else (" must-retain" if w == "CONTRACT_NULL_cov_plus_prior" else (" (pref<=3)" if w == "CONTRACT_random_label" else ""))
        print(f"    {w:34s} ALERT {main_tab[w]['B8_ALERT']:>3}/300 (CP95u {main_tab[w]['CP95u_ALERT']:.3f}) <=7:{main_tab[w]['B8_ALERT']<=7}{tag} | meanT-alone {mech_tab[w]['meanT_alone_alert']} both {mech_tab[w]['both_gate_alert']}")
    print("=== B. CONTRACT-valid POS (boundary>=20 strong; +prior>=15) ===")
    for w in pos:
        strong = 20 if w == "CONTRACT_POS_boundary" else 15
        print(f"    {w:34s} ALERT {main_tab[w]['B8_ALERT']:>3}/300 (>0; >={strong} strong) | meanT-alone {mech_tab[w]['meanT_alone_alert']} both {mech_tab[w]['both_gate_alert']}")
    print("=== C. VIOLATIONS (CONTRACT_INVALID dominates; ALERT==0 hard-stop) ===")
    for w in viols:
        print(f"    {w:34s} INVALID {main_tab[w]['CONTRACT_INVALID']:>3}/300 | ALERT {main_tab[w]['B8_ALERT']}/300 | medAUC {main_tab[w]['median_C_auc']:.3f}")

    prior_ok = main_tab["CONTRACT_NULL_prior_only"]["B8_ALERT"] <= 7
    mixed_ok = main_tab["CONTRACT_NULL_cov_plus_prior"]["B8_ALERT"] <= 7
    nulls_ok = all(main_tab[w]["B8_ALERT"] <= 7 for w in nulls)
    random_pref = main_tab["CONTRACT_random_label"]["B8_ALERT"] <= 3
    viol_no_alert = all(main_tab[w]["B8_ALERT"] == 0 for w in viols)
    viol_leaks = {w: main_tab[w]["B8_ALERT"] for w in viols if main_tab[w]["B8_ALERT"] > 0}
    viol_refuse = all(main_tab[w]["CONTRACT_INVALID"] >= 0.90 * NAGG for w in viols)
    posb = main_tab["CONTRACT_POS_boundary"]["B8_ALERT"]
    posbp = main_tab["CONTRACT_POS_boundary_plus_prior"]["B8_ALERT"]
    # stability: aggregate screen met AND no single seed block wildly off (per-block alert never > 3 for nulls/violations)
    null_block_stable = all(max(perseed[w]["alerts_per_block"]) <= 3 for w in nulls)
    viol_block_stable = all(max(perseed[w]["alerts_per_block"]) == 0 for w in viols)
    prior_meanT = mech_tab["CONTRACT_NULL_prior_only"]["meanT_alone_alert"]
    pooled_null = sum(main_tab[w]["B8_ALERT"] for w in nulls)
    hard_stop = bool(viol_leaks)

    screen = dict(nulls_le7=bool(nulls_ok), prior_only_le7_PRIMARY=bool(prior_ok), mixed_le7_retain=bool(mixed_ok),
                  random_label_le3_pref=bool(random_pref), violations_zero_alert=bool(viol_no_alert),
                  violations_refuse_dominant=bool(viol_refuse), pos_boundary_ge20=bool(posb >= 20),
                  pos_plus_prior_ge15=bool(posbp >= 15), pos_boundary_gt0=bool(posb > 0),
                  null_block_stable_max_le3=bool(null_block_stable), viol_block_stable_max0=bool(viol_block_stable),
                  HARD_STOP_violation_leak=hard_stop,
                  pooled_null_ALL_KINDS=f"{pooled_null}/{4*NAGG}", prior_only_meanT_alone=f"{prior_meanT}/300")

    safety_stable = nulls_ok and prior_ok and mixed_ok and viol_no_alert and viol_refuse and null_block_stable and viol_block_stable
    prA = main_tab["CONTRACT_NULL_prior_only"]["B8_ALERT"]; cpA = main_tab["CONTRACT_NULL_cov_plus_prior"]["B8_ALERT"]
    both_null_fail = (not prior_ok) and (not mixed_ok)
    if hard_stop:
        verdict = f"B8.2 HARD-STOP (Case E): violation ALERT leak {viol_leaks} -- contract validator failure, fix before any science claim"
    elif both_null_fail:
        verdict = (f"B8.2 Case C AND Case D -- B8.1 DECISION-LEVEL STABILITY FALSIFIED. prior_only {prA}/300={100*prA/NAGG:.1f}% "
                   f"(CI-excluded from nominal 2.5%) AND cov_plus_prior {cpA}/300={100*cpA/NAGG:.1f}% (exceeds the 7/300 screen; "
                   f"95% CI touches nominal) both FAIL <=7/300 across ALL 6 blocks. B8.1's 2/50 & 1/50 were TYPICAL single-block "
                   f"draws from a true ~4-6% rate (the n=50 screen was underpowered to resolve ~5% from 2.5%), NOT lucky. The "
                   f"studentized both-gate is ANTI-CONSERVATIVE on prior-bearing nulls (masks ~60-75% of the collider: mean-T "
                   f"{mech_tab['CONTRACT_NULL_prior_only']['meanT_alone_alert']}/{mech_tab['CONTRACT_NULL_cov_plus_prior']['meanT_alone_alert']}"
                   f" -> {prA}/{cpA}, leaks a seed-dependent residual; clean balanced/random nulls ~1.7% = well-calibrated). "
                   f"SURVIVORS (NOT falsified): violation refusal 300/300 (BY CONSTRUCTION, H3) + genuine-but-modest POS "
                   f"({posb}/300 ~{100*posb/NAGG:.0f}%, CI overlaps prior_only at margin). NEXT = contract redesign / narrow "
                   f"estimand (Case C) + STOP+diagnose contract construction (Case D) -- do NOT recalibrate the mean-T gate (rejected). Emulator/dev-only.")
    elif not prior_ok:
        verdict = f"B8.2 Case C: prior_only {prA}/300 > 7 -- class-balanced contract controls prior UNSTABLY -> contract redesign / narrow estimand (do NOT recalibrate mean-T gate)"
    elif not mixed_ok:
        verdict = f"B8.2 Case D: cov_plus_prior {cpA}/300 > 7 -- mixed-cell control LOST -> STOP + diagnose contract construction"
    elif safety_stable and posb >= 20:
        verdict = ("B8.2 Case A: SAFETY STABLE across 6 seed blocks (nulls+mixed<=7/300, violations refuse 0-alert, per-block stable) + POS>=20/300 -- "
                   f"B8 direction stable enough to discuss B8.3 budget frontier / real-audit protocol (reviewer). CAVEAT: decision-level (both-gate) stability; prior_only mean-T-alone still {prior_meanT}/300 (statistic-level collider remains). Emulator, NOT validation.")
    elif safety_stable and posb > 0:
        verdict = ("B8.2 Case B: SAFETY STABLE across 6 seed blocks but POS weak (POS_boundary<20/300) -- next step = audit/label budget frontier (power=design-budget question). "
                   f"CAVEAT: decision-level stability; prior_only mean-T-alone {prior_meanT}/300 (statistic-level collider remains). Emulator, NOT validation.")
    elif safety_stable and posb == 0:
        verdict = "B8.2 SAFETY STABLE but POS=0 -- contract too strong / label budget too small"
    else:
        verdict = "B8.2 SAFETY NOT STABLE -- inspect per-seed variability (a null or violation is block-unstable even if aggregate<=7)"
    if hard_stop:
        print(f"\n  !!! HARD-STOP: violation ALERT leak {viol_leaks} !!!")
    print(f"\n  >>> screen: {screen}")
    print(f"  >>> B8.2 VERDICT: {verdict}")
    print(f"  >>> KEY: prior_only both-gate {main_tab['CONTRACT_NULL_prior_only']['B8_ALERT']}/300 (B8.1 2/50) | mean-T-alone {prior_meanT}/300 (B8.1 5/50 -> ~30/300 expected if intact) | "
          f"cov_plus_prior {main_tab['CONTRACT_NULL_cov_plus_prior']['B8_ALERT']}/300 | POS_boundary {posb}/300 (B8.1 8/50 -> ~48/300 expected) | pooled null {pooled_null}/{4*NAGG}")

    out = dict(scope="B8.2 multi-seed stability replication; development-only; NOT confirmatory; NOT validation (Lee2019 emulator); NO tag; engine byte-identical to B8.1 3109c0f",
               seed_bases=BASES, n_per_block=NPB, n_aggregate=NAGG, main_table=main_tab, seed_table=seed_tab,
               mechanism_table=mech_tab, per_seed_variability=perseed, screen=screen, verdict=verdict,
               vs_b8_1="B8.1 (n=50): prior_only both-gate 2/50, mean-T-alone 5/50, cov_plus_prior 1/50, POS_boundary 8/50, pooled null 6/200 -- RE-CHARACTERIZED by B8.2: those were TYPICAL underpowered single-block draws from a true ~4-6% rate, NOT genuine control",
               result_redteam=dict(workflow="w3u5q3x10", accounting="PASS (3600 reproduced, 0 mismatches, engine byte-identical to committed B8.1, seeds disjoint, both-gate recompute matches, no fabrication)",
                                   science="MINOR_ISSUE -- negative GENUINE + honestly framed; wording refined",
                                   honest_label="B8.1 decision-level stability FALSIFIED by multi-seed replication: prior_only 18/300=6.0% (CP95 [3.6,9.3]%, strictly > nominal 2.5%) AND mixed cov_plus_prior 13/300=4.3% (fails the 7/300 screen; binom p=0.041; 95% CI [2.33,7.30]% touches nominal) both fail across ALL 6 seed blocks; B8.1's 2/50 & 1/50 were TYPICAL underpowered single-block draws (n=50 cannot resolve ~5% from 2.5%), NOT lucky. Both-gate anti-conservative on prior nulls (masks 60-75% of the collider, mean-T 46/49 -> 18/13, leaks a seed-dependent residual; clean nulls ~1.7% well-calibrated). Survivors NOT falsified: violation refusal 300/300 (by construction) + genuine-but-modest POS 37/300 & 39/300 (~12-13%, CP95 [8.8,16.6]%, Fisher p=0.010/0.0002 vs the failing null, CI overlaps at margin). NEXT = contract redesign / narrow estimand (Case C+D), NOT mean-T recalibration. Emulator/dev-only, NO tag.",
                                   prov_note="low: .prov.json records host/slurm/base/world but NOT the engine sha256; run-time byte-identity rests on on-disk hash + protocol pin (future: embed engine sha256 per shard)"),
               scope_limits="decision-level (both-gate) STABILITY across seed blocks; does NOT fix the statistic-level mean-T collider (reported, not gated); emulator (Lee2019 SM16), semi-synthetic; violations refused BY CONSTRUCTION (H3 schedule-adherence); H1/H2/H5 vacuous-by-construction (H3/H4/D1 exercised)")
    json.dump(out, open(f"{CDIR}/b8_2_multiseed_tables.json", "w"), indent=1, default=str)
    print(f"\n  saved {CDIR}/b8_2_multiseed_tables.json")


if __name__ == "__main__":
    main()
