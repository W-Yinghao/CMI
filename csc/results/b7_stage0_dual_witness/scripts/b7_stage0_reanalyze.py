"""CSC B7-primary Stage-0 dual-witness canary (development-only; exposed row-level re-analysis of the COMMITTED B6.0
cohorts; NO new compute; NO tag; NOT confirmatory; NOT deployable; NOT a universal type-I guarantee).

B7-primary (PINNED) = old_B3_confirm AND B6_plain_confirm  -- a nuisance-DISJUNCTION rule: the OLD B3 fixed-margin
Y-null is the PRIOR-shift witness, B6.0 plain C-null is the COVARIATE-process witness; a concept alert is allowed only
when BOTH nuisance explanations are rejected. Decision is STATE-based (each witness's own gated confirm state, with its
own gates); p_dual = max(p_old, p_C) is a continuous DIAGNOSTIC only. Set-conservative: the allow set is a subset of
old_B3 confirms AND of B6_plain confirms; formal validity depends on each witness being valid for its component null.
B6-FM / old&FM / plain&FM / triple are SECONDARY diagnostics only -- they never change the primary.
"""
import os, sys, json, hashlib
import numpy as np

B60DIR = "/home/infres/yinwang/realeeg_feas/b6_canary"
OUTDIR = "/home/infres/yinwang/realeeg_feas/b7_stage0"
FMDIR = "/home/infres/yinwang/realeeg_feas/b6fm_canary"     # secondary only
CONDS = ["NULL_cov_soft", "NULL_cov_plus_label_soft", "NULL_cov_strong_auc0.81", "NULL_cov_strong_auc0.94",
         "NULL_label", "random_label_control", "POS_concept", "POS_concept_plus_cov"]
NOCONCEPT = {"NULL_cov_soft", "NULL_cov_plus_label_soft", "NULL_cov_strong_auc0.81", "NULL_cov_strong_auc0.94",
             "NULL_label", "random_label_control"}
N = 50
# OLD B3 states where the test ran and DECIDED (valid); others are abstain/invalid.
OLD_DECIDED = {"CONCEPT_CONFIRMED", "NO_CONCEPT_EVIDENCE_AFTER_PAIR_AUDIT"}


def _read(p): return [json.loads(l) for l in open(p) if l.strip()]


def b7_state(old_confirm, old_valid, b6_confirm, b6_valid):
    if (not old_valid) or (not b6_valid):
        return "UNIDENTIFIABLE_OR_INVALID"
    if old_confirm and b6_confirm:
        return "DUAL_CONCEPT_ALERT"
    return "NO_ACTIONABLE_CONCEPT_EVIDENCE"


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rows_out = []
    tables = {}
    fm_map = {}
    for c in CONDS:
        for r in _read(f"{FMDIR}/b6fm_canary_{c}_0.jsonl"):
            fm_map[r["task_id"]] = r
    for c in CONDS:
        recs = {r["task_id"]: r for r in _read(f"{B60DIR}/b6_canary_{c}_0.jsonl")}
        if len(recs) != N:
            print(f"FAIL-CLOSED: {c} has {len(recs)} != {N}"); sys.exit(2)
        gt_noconcept = c in NOCONCEPT
        # DISJOINT partition (Option A): each cohort in EXACTLY one bucket -> both+old_only+b6_plain_only+neither_valid+invalid == 50
        cnt = dict(old_confirm=0, b6_confirm=0, both=0, old_only=0, b6_plain_only=0, neither_valid=0, invalid=0,
                   DUAL_CONCEPT_ALERT=0, NO_ACTIONABLE_CONCEPT_EVIDENCE=0, UNIDENTIFIABLE_OR_INVALID=0,
                   fm_confirm=0, old_and_fm=0, plain_and_fm=0, triple=0)
        p_olds, p_cs, p_duals = [], [], []
        for tid, r in sorted(recs.items()):
            old_state = str(r.get("old_b3_state"))
            old_valid = old_state in OLD_DECIDED
            old_confirm = (old_state == "CONCEPT_CONFIRMED")
            b6_state = str(r.get("b6_state"))
            b6_valid = bool(r.get("b6_valid")) and not bool(r.get("b6_abstain_lock"))
            b6_confirm = (b6_state == "CONCEPT_CONFIRMED_B6")            # GATED state, NOT the p-only flag
            st = b7_state(old_confirm, old_valid, b6_confirm, b6_valid)
            b7_confirm = (st == "DUAL_CONCEPT_ALERT")
            p_old = r.get("old_fixed_margin_p")          # DIAGNOSTIC proxy (OLD decision also uses studentized+LCB gates)
            p_c = r.get("p_C_meanT")
            p_dual = max(p_old, p_c) if isinstance(p_old, (int, float)) and isinstance(p_c, (int, float)) else None
            if isinstance(p_old, (int, float)) and p_old == p_old: p_olds.append(p_old)
            if isinstance(p_c, (int, float)) and p_c == p_c: p_cs.append(p_c)
            if p_dual is not None and p_dual == p_dual: p_duals.append(p_dual)
            # secondary: B6-FM (never primary)
            fm = fm_map.get(tid, {})
            fm_confirm = (str(fm.get("fm_state")) == "CONCEPT_CONFIRMED")
            cnt[st] += 1
            cnt["old_confirm"] += old_confirm; cnt["b6_confirm"] += b6_confirm; cnt["fm_confirm"] += fm_confirm
            valid = old_valid and b6_valid          # confirm implies valid, so 'both' cohorts are always valid
            if not valid:
                cnt["invalid"] += 1
            elif old_confirm and b6_confirm:
                cnt["both"] += 1
            elif old_confirm:
                cnt["old_only"] += 1
            elif b6_confirm:
                cnt["b6_plain_only"] += 1
            else:
                cnt["neither_valid"] += 1
            cnt["old_and_fm"] += (old_confirm and fm_confirm)
            cnt["plain_and_fm"] += (b6_confirm and fm_confirm)
            cnt["triple"] += (old_confirm and b6_confirm and fm_confirm)
            rows_out.append(dict(task_id=tid, condition=c, ground_truth_noconcept=gt_noconcept,
                                 old_b3_state=old_state, old_confirm=bool(old_confirm), old_valid=bool(old_valid),
                                 b6_plain_state=b6_state, b6_plain_confirm=bool(b6_confirm), b6_valid=bool(b6_valid),
                                 b7_state=st, b7_primary_confirm=bool(b7_confirm),
                                 b7_false_confirm=bool(b7_confirm and gt_noconcept),
                                 b7_true_confirm=bool(b7_confirm and not gt_noconcept),
                                 p_old_fixed_margin=p_old, p_C_plain=p_c, p_dual=p_dual,
                                 fm_confirm_secondary=bool(fm_confirm), diagnostic_only=True))
        # DISJOINT partition must sum to N, and invalid-bucket must equal the UNIDENTIFIABLE_OR_INVALID state (fail-closed)
        part_sum = cnt["both"] + cnt["old_only"] + cnt["b6_plain_only"] + cnt["neither_valid"] + cnt["invalid"]
        if part_sum != N:
            print(f"FAIL-CLOSED: {c} disjoint partition sums to {part_sum} != {N}"); sys.exit(2)
        if cnt["invalid"] != cnt["UNIDENTIFIABLE_OR_INVALID"] or cnt["both"] != cnt["DUAL_CONCEPT_ALERT"]:
            print(f"FAIL-CLOSED: {c} partition/state mismatch invalid={cnt['invalid']}/{cnt['UNIDENTIFIABLE_OR_INVALID']} both={cnt['both']}/{cnt['DUAL_CONCEPT_ALERT']}"); sys.exit(2)
        tables[c] = dict(n=N, ground_truth_noconcept=gt_noconcept,
                         old_confirm=cnt["old_confirm"], b6_plain_confirm=cnt["b6_confirm"],
                         b7_primary_confirm=cnt["DUAL_CONCEPT_ALERT"],
                         b7_false_confirm=(cnt["DUAL_CONCEPT_ALERT"] if gt_noconcept else None),
                         b7_true_confirm=(cnt["DUAL_CONCEPT_ALERT"] if not gt_noconcept else None),
                         partition_both=cnt["both"], partition_old_only=cnt["old_only"],
                         partition_b6_plain_only=cnt["b6_plain_only"], partition_neither_valid=cnt["neither_valid"],
                         partition_invalid=cnt["invalid"], partition_sum=part_sum,
                         state_DUAL_ALERT=cnt["DUAL_CONCEPT_ALERT"], state_NO_ACTIONABLE=cnt["NO_ACTIONABLE_CONCEPT_EVIDENCE"],
                         state_UNIDENTIFIABLE_OR_INVALID=cnt["UNIDENTIFIABLE_OR_INVALID"],
                         median_p_old=float(np.median(p_olds)) if p_olds else None,
                         median_p_C_plain=float(np.median(p_cs)) if p_cs else None,
                         median_p_dual=float(np.median(p_duals)) if p_duals else None,
                         secondary_fm_confirm=cnt["fm_confirm"], secondary_old_and_fm=cnt["old_and_fm"],
                         secondary_plain_and_fm=cnt["plain_and_fm"], secondary_triple=cnt["triple"])

    # print main table
    print(f"{'condition':26s} {'GT':>6} {'old':>4} {'B6pl':>5} {'B7prim':>7} | DISJOINT: {'both':>4} {'oOnly':>5} {'B6Only':>6} {'neithV':>6} {'inval':>5} {'sum':>4} | {'medp_dual':>9}")
    print("-" * 128)
    for c in CONDS:
        t = tables[c]
        print(f"{c:26s} {'NOCON' if t['ground_truth_noconcept'] else 'CONC':>6} {t['old_confirm']:>4} {t['b6_plain_confirm']:>5} "
              f"{t['b7_primary_confirm']:>7} | {t['partition_both']:>13} {t['partition_old_only']:>5} {t['partition_b6_plain_only']:>6} "
              f"{t['partition_neither_valid']:>6} {t['partition_invalid']:>5} {t['partition_sum']:>4} | "
              f"{(t['median_p_dual'] if t['median_p_dual'] is not None else float('nan')):>9.3f}")

    # hard screen (development screen, NOT a CP safety claim)
    screen = dict(NULL_cov_soft=tables["NULL_cov_soft"]["b7_primary_confirm"] <= 5,
                  NULL_cov_plus_label=tables["NULL_cov_plus_label_soft"]["b7_primary_confirm"] <= 5,
                  strong81=tables["NULL_cov_strong_auc0.81"]["b7_primary_confirm"] <= 3,
                  strong94=tables["NULL_cov_strong_auc0.94"]["b7_primary_confirm"] <= 3,
                  NULL_label=tables["NULL_label"]["b7_primary_confirm"] <= 3,
                  random=tables["random_label_control"]["b7_primary_confirm"] <= 3,
                  POS_concept_pos=tables["POS_concept"]["b7_primary_confirm"] > 0)
    passed = all(screen.values())
    print(f"\n  >>> B7.0 hard screen (development screen, NOT CP safety): {screen}")
    print(f"  >>> B7.0 canary screen PASSED: {passed}  (POS_concept={tables['POS_concept']['b7_primary_confirm']}/50)")

    # persist
    with open(f"{OUTDIR}/b7_stage0_rows.jsonl", "w") as f:
        for r in rows_out:
            f.write(json.dumps(r, default=str) + "\n")
    inter = {c: dict(both=tables[c]["partition_both"], old_only=tables[c]["partition_old_only"],
                     b6_plain_only=tables[c]["partition_b6_plain_only"], neither_valid=tables[c]["partition_neither_valid"],
                     invalid=tables[c]["partition_invalid"], secondary_old_and_fm=tables[c]["secondary_old_and_fm"],
                     secondary_plain_and_fm=tables[c]["secondary_plain_and_fm"], secondary_triple=tables[c]["secondary_triple"])
             for c in CONDS}
    json.dump(inter, open(f"{OUTDIR}/b7_stage0_intersections.json", "w"), indent=1)
    out_tables = dict(
        scope="B7-primary Stage-0 dual-witness canary; exposed row-level re-analysis of committed B6.0 cohorts; development-only; NOT confirmatory; NOT deployable; NOT a universal type-I guarantee",
        primary_rule="B7_primary_confirm = (old_b3_state==CONCEPT_CONFIRMED) AND (b6_state==CONCEPT_CONFIRMED_B6)",
        diagnostic="p_dual = max(p_old_fixed_margin, p_C_plain) -- DIAGNOSTIC ONLY; primary decision is STATE-based",
        set_conservative="allow set is a SUBSET of old_B3 confirms AND of B6_plain confirms; formal validity depends on each witness's validity for its component null; NOT a universal type-I guarantee",
        witnesses="OLD B3 fixed-margin Y-null = PRIOR-shift witness; B6.0 plain C-null = COVARIATE-process witness",
        base_seed=200_000_000, n_per_condition=N, per_condition=tables, hard_screen=screen, hard_screen_passed=bool(passed),
        secondary_note="B6-FM (old_and_fm / plain_and_fm / triple) reported as SECONDARY diagnostics only; never primary")
    json.dump(out_tables, open(f"{OUTDIR}/b7_stage0_tables.json", "w"), indent=1, default=str)
    manifest = dict(scope=out_tables["scope"], primary_rule=out_tables["primary_rule"], base_seed=200_000_000,
                    conditions=CONDS, n_per_condition=N, source_b6_0_dir=B60DIR, source_b6_fm_dir_secondary=FMDIR,
                    b6_0_row_sha256={c: hashlib.sha256(open(f"{B60DIR}/b6_canary_{c}_0.jsonl", "rb").read()).hexdigest() for c in CONDS},
                    forbidden=["old_and_FM as primary", "best-of-B6", "router/observed_T threshold", "p-recalibration",
                               "P(C|Z,S,Y)", "within-class Z balancing", "oracle field", "condition-label rule", "confirmatory tag"],
                    diagnostic_only=True, not_confirmatory=True)
    json.dump(manifest, open(f"{OUTDIR}/b7_stage0_manifest.json", "w"), indent=1, default=str)
    print(f"\n  saved {OUTDIR}/ (rows.jsonl, tables.json, intersections.json, manifest.json)")


if __name__ == "__main__":
    main()
