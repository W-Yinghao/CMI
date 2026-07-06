"""B7.1 exposed full dev replay analysis. Reads the 8 fresh b7s1 shards (n=300 each, base 300e6), each row carrying
BOTH witnesses (old_b3_state + b6_state, same cohort). Computes the PINNED primary B7_confirm = old_B3_confirm AND
B6_plain_confirm (gated states), the DISJOINT partition (sums to 300), CP95 upper per null kind (NO pooling), the p
distributions, and POS retention. Fail-closed. Development-only, NOT confirmatory, NOT a universal type-I guarantee."""
import os, sys, json, math, hashlib
import numpy as np
try:
    from scipy.stats import beta
    def cp_upper(k, n, a=0.05): return 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
except Exception:
    def cp_upper(k, n, a=0.05): return min(1.0, (k + 1.645 * math.sqrt(k + 1)) / n)

RD = "/home/infres/yinwang/realeeg_feas/b7_stage1_rows"
CONDS = ["NULL_cov_soft", "NULL_cov_plus_label_soft", "NULL_cov_strong_auc0.81", "NULL_cov_strong_auc0.94",
         "NULL_label", "random_label_control", "POS_concept", "POS_concept_plus_cov"]
NOCONCEPT = {"NULL_cov_soft", "NULL_cov_plus_label_soft", "NULL_cov_strong_auc0.81", "NULL_cov_strong_auc0.94",
             "NULL_label", "random_label_control"}
N = 300
N_BOOT = 200
P_FLOOR = 1.0 / (N_BOOT + 1)
OLD_DECIDED = {"CONCEPT_CONFIRMED", "NO_CONCEPT_EVIDENCE_AFTER_PAIR_AUDIT"}


def _read(p): return [json.loads(l) for l in open(p) if l.strip()]
def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and x == x]
    return float(np.median(xs)) if xs else float("nan")
def floorfrac(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and x == x]
    return float(np.mean([1.0 if x <= P_FLOOR + 1e-9 else 0.0 for x in xs])) if xs else float("nan")


def main():
    tables, rows_out = {}, []
    for c in CONDS:
        p = f"{RD}/b7s1_{c}_0.jsonl"
        if not os.path.exists(p):
            print(f"FAIL-CLOSED: missing {p}"); sys.exit(2)
        recs = _read(p)
        if any("__worker_error__" in r for r in recs):
            print(f"FAIL-CLOSED: worker-error rows in {c}"); sys.exit(2)
        if len(recs) != N or len({r["task_id"] for r in recs}) != N:
            print(f"FAIL-CLOSED: {c} count/dup ({len(recs)})"); sys.exit(2)
        gt_noconcept = c in NOCONCEPT
        part = dict(both=0, old_only=0, b6_plain_only=0, neither_valid=0, invalid=0)
        p_olds, p_cs, p_duals, obsTs, cnullsd, effr, aucs = [], [], [], [], [], [], []
        pos_ret_old = pos_ret_b6 = 0
        for r in sorted(recs, key=lambda z: z["task_id"]):
            old_state = str(r.get("old_b3_state")); old_valid = old_state in OLD_DECIDED
            old_confirm = (old_state == "CONCEPT_CONFIRMED")
            b6_state = str(r.get("b6_state")); b6_valid = bool(r.get("b6_valid")) and not bool(r.get("b6_abstain_lock"))
            b6_confirm = (b6_state == "CONCEPT_CONFIRMED_B6")
            valid = old_valid and b6_valid
            if not valid:
                st = "UNIDENTIFIABLE_OR_INVALID"; part["invalid"] += 1
            elif old_confirm and b6_confirm:
                st = "DUAL_CONCEPT_ALERT"; part["both"] += 1
            elif old_confirm:
                st = "NO_ACTIONABLE_CONCEPT_EVIDENCE"; part["old_only"] += 1
            elif b6_confirm:
                st = "NO_ACTIONABLE_CONCEPT_EVIDENCE"; part["b6_plain_only"] += 1
            else:
                st = "NO_ACTIONABLE_CONCEPT_EVIDENCE"; part["neither_valid"] += 1
            b7_confirm = (st == "DUAL_CONCEPT_ALERT")
            po, pc = r.get("old_fixed_margin_p"), r.get("p_C_meanT")
            pd = max(po, pc) if isinstance(po, (int, float)) and isinstance(pc, (int, float)) and po == po and pc == pc else None
            if isinstance(po, (int, float)) and po == po: p_olds.append(po)
            if isinstance(pc, (int, float)) and pc == pc: p_cs.append(pc)
            if pd is not None: p_duals.append(pd)
            obsTs.append(r.get("observed_T")); cnullsd.append(r.get("c_null_sd_T"))
            effr.append(r.get("eff_randomization")); aucs.append(r.get("propensity_auc"))
            if not gt_noconcept:                       # POS retention
                pos_ret_old += old_confirm; pos_ret_b6 += b6_confirm
            rows_out.append(dict(task_id=r["task_id"], condition=c, ground_truth_noconcept=gt_noconcept,
                                 old_b3_state=old_state, old_confirm=bool(old_confirm),
                                 b6_plain_state=b6_state, b6_plain_confirm=bool(b6_confirm), b6_valid=bool(b6_valid),
                                 b7_state=st, b7_primary_confirm=bool(b7_confirm),
                                 b7_false_confirm=bool(b7_confirm and gt_noconcept),
                                 b7_true_confirm=bool(b7_confirm and not gt_noconcept),
                                 p_old_fixed_margin=po, p_C_plain=pc, p_dual=pd, fidelity_dT=r.get("fidelity_dT")))
        ps = part["both"] + part["old_only"] + part["b6_plain_only"] + part["neither_valid"] + part["invalid"]
        if ps != N:
            print(f"FAIL-CLOSED: {c} disjoint partition {ps} != {N}"); sys.exit(2)
        b7c = part["both"]
        tables[c] = dict(n=N, ground_truth_noconcept=gt_noconcept, old_confirm=None, b6_plain_confirm=None,
                         b7_primary_confirm=b7c, b7_false_confirm=(b7c if gt_noconcept else None),
                         b7_true_confirm=(b7c if not gt_noconcept else None),
                         partition_both=part["both"], partition_old_only=part["old_only"],
                         partition_b6_plain_only=part["b6_plain_only"], partition_neither_valid=part["neither_valid"],
                         partition_invalid=part["invalid"], partition_sum=ps,
                         b7_cp95_upper=(round(cp_upper(b7c, N), 4)),
                         median_p_old=med(p_olds), median_p_C_plain=med(p_cs), median_p_dual=med(p_duals),
                         p_old_floor_frac=round(floorfrac(p_olds), 3), p_C_floor_frac=round(floorfrac(p_cs), 3),
                         median_observed_T=med(obsTs), median_Cnull_sd=med(cnullsd),
                         median_eff_randomization=med(effr), median_propensity_auc=med(aucs),
                         pos_retained_among_old=(pos_ret_old if not gt_noconcept else None),
                         pos_retained_among_b6_plain=(pos_ret_b6 if not gt_noconcept else None))
        # fill old/b6 confirm counts
        tables[c]["old_confirm"] = sum(1 for r in recs if str(r.get("old_b3_state")) == "CONCEPT_CONFIRMED")
        tables[c]["b6_plain_confirm"] = sum(1 for r in recs if str(r.get("b6_state")) == "CONCEPT_CONFIRMED_B6")

    # print
    print(f"{'condition':26s} {'GT':>6} {'old':>4} {'B6pl':>5} {'B7':>4} {'CP95u':>7} | {'both':>4} {'oOnly':>5} {'B6Only':>6} {'neithV':>6} {'inval':>5} {'Σ':>4} | {'medpDual':>8} {'AUC':>5}")
    print("-" * 132)
    for c in CONDS:
        t = tables[c]
        print(f"{c:26s} {'NOCON' if t['ground_truth_noconcept'] else 'CONC':>6} {t['old_confirm']:>4} {t['b6_plain_confirm']:>5} "
              f"{t['b7_primary_confirm']:>4} {t['b7_cp95_upper']:>7.4f} | {t['partition_both']:>4} {t['partition_old_only']:>5} "
              f"{t['partition_b6_plain_only']:>6} {t['partition_neither_valid']:>6} {t['partition_invalid']:>5} {t['partition_sum']:>4} | "
              f"{t['median_p_dual']:>8.3f} {t['median_propensity_auc']:>5.2f}")

    # endpoints (kind-specific, NO pooling)
    nulls = [c for c in CONDS if c in NOCONCEPT]
    safety = {c: dict(b7=tables[c]["b7_primary_confirm"], cp95u=tables[c]["b7_cp95_upper"],
                      le7=tables[c]["b7_primary_confirm"] <= 7) for c in nulls}
    safety_pass = all(v["le7"] for v in safety.values())
    posc, poscc = tables["POS_concept"]["b7_primary_confirm"], tables["POS_concept_plus_cov"]["b7_primary_confirm"]
    util = ("STRONG" if (posc >= 20 and poscc >= 15) else "WEAK" if (safety_pass and posc > 0) else "FAIL" if posc == 0 else "MIXED")
    verdict = ("SAFETY-PASS + STRONG utility -> eligible for fresh-validation-protocol DISCUSSION" if (safety_pass and util == "STRONG")
               else "SAFETY-PASS + WEAK utility (safe-but-weak)" if (safety_pass and util == "WEAK")
               else "SAFETY-PASS + ZERO utility (abstention-only boundary)" if (safety_pass and posc == 0)
               else "SAFETY-FAIL -> no retuning; analyze failure -> likely B8 information-contract")
    print(f"\n  === PRIMARY SAFETY (kind-specific, NO pooling; each null B7<=7/300) ===")
    for c in nulls:
        v = safety[c]; pref = 3 if (c in {"random_label_control", "NULL_label", "NULL_cov_strong_auc0.81", "NULL_cov_strong_auc0.94"}) else 7
        print(f"    {c:26s} B7={v['b7']}/300 CP95u={v['cp95u']:.4f} <=7:{v['le7']} (pref<={pref})")
    print(f"  === UTILITY: POS_concept={posc}/300 (>=20 strong,>0 weak) POS+cov={poscc}/300 (>=15) -> {util} ===")
    print(f"  >>> B7.1 VERDICT: {verdict}")

    with open(f"{RD}/b7_stage1_rows.jsonl", "w") as f:
        for r in rows_out: f.write(json.dumps(r, default=str) + "\n")
    out = dict(scope="B7.1 exposed full dev replay; dual-witness old_B3 AND B6_plain; development-only; NOT confirmatory; NOT deployable; NOT a universal type-I guarantee; NO tag; B6-FM NOT used",
               primary_rule="B7_primary_confirm = (old_b3_state==CONCEPT_CONFIRMED) AND (b6_state==CONCEPT_CONFIRMED_B6)",
               base_seed=300_000_000, n_per_condition=N, per_condition=tables,
               primary_safety=safety, primary_safety_pass=bool(safety_pass), utility_tier=util, verdict=verdict,
               set_conservative="allow set subset of old_B3 confirms AND B6_plain confirms; validity depends on each witness validity for its component null; NOT a universal type-I guarantee")
    json.dump(out, open(f"{RD}/b7_stage1_tables.json", "w"), indent=1, default=str)
    print(f"\n  saved {RD}/ (b7_stage1_rows.jsonl, b7_stage1_tables.json)")


if __name__ == "__main__":
    main()
