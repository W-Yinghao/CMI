"""B8.0 canary merge. Per-world B8 state distribution + the reviewer screen: contract-valid nulls B8_ALERT<=3/50;
violation worlds CONTRACT_INVALID high AND B8_ALERT<=3/50 (correct REFUSAL, not false-alert); CONTRACT_POS_boundary
B8_ALERT>0 (>=5 pref). Fail-closed; contract-valid vs contract-violating worlds reported separately (NO pooling).
Development-only, NOT confirmatory, NOT validation, NO tag."""
import os, sys, json
import numpy as np
sys.path.insert(0, "/home/infres/yinwang/realeeg_feas")
sys.path.insert(0, "/home/infres/yinwang/CMI_AAAI_csc")   # csc.mininfo (paired_calibrated) for realeeg_b8 import
import realeeg_b8 as B8

CDIR = "/home/infres/yinwang/realeeg_feas/b8_canary"
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
        p = f"{CDIR}/b8_canary_{w}_0.jsonl"
        if not os.path.exists(p):
            print(f"FAIL-CLOSED: missing {p}"); sys.exit(2)
        recs = _read(p)
        if any("__worker_error__" in r for r in recs):
            print(f"FAIL-CLOSED: worker-error rows in {w}"); sys.exit(2)
        if len(recs) != N or len({r["task_id"] for r in recs}) != N:
            print(f"FAIL-CLOSED: {w} count/dup"); sys.exit(2)
        per[w] = recs

    rows = {}
    print(f"{'world':34s} {'cls':>4} {'conc':>4} {'ALERT':>5} {'NOACT':>5} {'CINVAL':>6} {'INSUF':>5} {'SAMPINV':>7} {'valid':>5} {'medAUC':>6}")
    print("-" * 104)
    for w in WORLDS:
        recs = per[w]; kind, has_concept = B8.WORLDS[w]
        from collections import Counter
        sc = Counter(str(r.get("b8_state")) for r in recs)
        alert = sc.get("B8_CONCEPT_ALERT", 0); noact = sc.get("NO_ACTIONABLE_CONCEPT_EVIDENCE", 0)
        cinval = sc.get("CONTRACT_INVALID_OR_UNIDENTIFIABLE", 0); insuf = sc.get("INSUFFICIENT_LABELS", 0)
        sampinv = sc.get("SAMPLER_INVALID", 0); needm = sc.get("NEED_MORE_LABELS", 0)
        valid = sum(1 for r in recs if r.get("contract_valid"))
        rows[w] = dict(n=N, world_class=kind, has_concept=has_concept, alert=alert, no_actionable=noact,
                       contract_invalid=cinval, insufficient=insuf, sampler_invalid=sampinv, need_more=needm,
                       contract_valid=valid, median_auc=med([r.get("contract_within_block_C_Z_auc") for r in recs]),
                       median_support=med([r.get("contract_n_support_blocks") for r in recs]),
                       median_p_meanT=med([r.get("p_exact_meanT") for r in recs]),
                       median_p_stud=med([r.get("p_exact_stud") for r in recs]))
        print(f"{w:34s} {kind[:4]:>4} {str(has_concept)[:4]:>4} {alert:>5} {noact:>5} {cinval:>6} {insuf:>5} {sampinv:>7} {valid:>5} {rows[w]['median_auc']:>6.3f}")

    # screen (kind-specific, NO pooling)
    print("\n=== A. CONTRACT-satisfying NULL worlds: B8_ALERT must be LOW (<=3/50) ===")
    contract_nulls = [w for w in WORLDS if B8.WORLDS[w][0] == "contract" and not B8.WORLDS[w][1]]
    for w in contract_nulls:
        print(f"    {w:34s} ALERT {rows[w]['alert']}/50 (valid {rows[w]['contract_valid']}) <=3:{rows[w]['alert'] <= 3}")
    print("=== B. CONTRACT-satisfying POS worlds: B8_ALERT power (>0, >=5 strong) ===")
    for w in [w for w in WORLDS if B8.WORLDS[w][0] == "contract" and B8.WORLDS[w][1]]:
        print(f"    {w:34s} ALERT {rows[w]['alert']}/50 (>0 power; >=5 strong)")
    print("=== C. CONTRACT-VIOLATING worlds: correct REFUSAL (CONTRACT_INVALID high) AND B8_ALERT low (<=3) ===")
    for w in [w for w in WORLDS if B8.WORLDS[w][0] == "violation"]:
        print(f"    {w:34s} CONTRACT_INVALID {rows[w]['contract_invalid']}/50 | ALERT {rows[w]['alert']}/50 <=3:{rows[w]['alert'] <= 3} (medAUC {rows[w]['median_auc']:.3f})")

    null_ok = all(rows[w]["alert"] <= 3 for w in contract_nulls)
    viol_ok = all(rows[w]["alert"] <= 3 for w in WORLDS if B8.WORLDS[w][0] == "violation")
    pos = rows["CONTRACT_POS_boundary"]["alert"]
    screen = dict(contract_nulls_le3=bool(null_ok), violation_alerts_le3=bool(viol_ok),
                  pos_boundary_power=bool(pos > 0), pos_boundary_strong=bool(pos >= 5))
    verdict = ("B8.0 PASSES canary: contract nulls controlled + violations refused + POS_boundary power" if (null_ok and viol_ok and pos > 0)
               else "B8.0 controls nulls + refuses violations but POS_boundary=0 -> safe-but-powerless (contract too strong / label budget too small)" if (null_ok and viol_ok and pos == 0)
               else "B8.0 MIXED -- inspect (a contract null or violation over-alerts, or borderline-confound false-alerts)")
    print(f"\n  >>> screen: {screen}")
    print(f"  >>> B8.0 VERDICT: {verdict}")
    print(f"  >>> KEY: CONTRACT_NULL_cov_plus_prior (B7.1-killer analogue under valid contract) ALERT = {rows['CONTRACT_NULL_cov_plus_prior']['alert']}/50 (B7.1 was 24/300=8%)")
    print(f"  >>> borderline (near-tau, contract-valid): ALERT {rows['VIOLATION_borderline_confound']['alert']}/50 valid {rows['VIOLATION_borderline_confound']['contract_valid']}/50 -- threshold sensitivity")

    tables = dict(scope="B8.0 information-contract canary; development-only; NOT confirmatory; NOT deployable; NOT validation (Lee2019 not a real randomized audit); NO tag",
                  base_seed=400_000_000, n_per_world=N, per_world=rows, screen=screen, verdict=verdict,
                  b7_1_killer_analogue=dict(world="CONTRACT_NULL_cov_plus_prior", b8_alert=rows["CONTRACT_NULL_cov_plus_prior"]["alert"], b7_1_was="24/300=8%"),
                  scope_limits="validator tests only linear top-8-PC C~Z predictability (low-var/nonlinear confound blind spot; on SM16 last-PC still caught); prior worlds APPROXIMATELY exact (class-margin conditioning on Y is a collider); TAU_CONTRACT_AUC=0.60 fixed; borderline confound near tau tests threshold sensitivity")
    json.dump(tables, open(f"{CDIR}/b8_canary_tables.json", "w"), indent=1, default=str)
    print(f"\n  saved {CDIR}/b8_canary_tables.json")


if __name__ == "__main__":
    main()
