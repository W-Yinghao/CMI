"""P3.0b/c four judgment tables from the merged internal forensic (1800 cohorts, 3 seed blocks)."""
import json, statistics as st
M = json.load(open("/home/infres/yinwang/realeeg_feas/p3_forensic/p3_internal_forensic_merged.json"))
R = M["per_cohort"]
def sub(cond=None, fc=None, tc=None):
    out=[r for r in R if (cond is None or r["condition"]==cond)]
    if fc is not None: out=[r for r in out if r.get("false_confirm")==fc]
    if tc is not None: out=[r for r in out if r.get("true_confirm")==tc]
    return out
def med(rows,k):
    v=[r[k] for r in rows if isinstance(r.get(k),(int,float)) and r[k]==r[k]]
    return st.median(v) if v else float("nan")
def frac(rows,pred):
    return sum(1 for r in rows if pred(r))/len(rows) if rows else float("nan")

print("############ TABLE 1: reproducibility of false-confirm across 3 seed blocks ############")
for cond in ("NULL_cov","NULL_cov_plus_label","NULL_label","random_label_control","POS_concept","POS_concept_plus_cov"):
    per=[sum(1 for r in R if r["seed_block"]==b and r["condition"]==cond and (r.get("false_confirm") or r.get("true_confirm"))) for b in M["seed_blocks"]]
    kind="FC" if cond in ("NULL_cov","NULL_cov_plus_label","NULL_label","random_label_control") else "TC"
    print(f"  {cond:22s} {kind} per-block = {per}  (pooled {sum(per)}/300)")

print("\n############ TABLE 2: null-calibration signature (NULL_cov: false-confirm vs clean) ############")
fc=sub("NULL_cov",fc=True); cl=sub("NULL_cov",fc=False)
print(f"  NULL_cov: {len(fc)} false-confirm vs {len(cl)} clean")
for k in ("observed_T","null_mean_T","null_sd_T","T_z","fixed_margin_p","standard_null_p",
          "studentized_p","studentized_stat","studentized_null_mean","studentized_null_sd","subject_consistency_lcb"):
    print(f"    {k:22s} FC_med={med(fc,k):10.5f}   clean_med={med(cl,k):10.5f}")
print(f"    fixed_margin_p AT FLOOR (<=0.0051): FC={frac(fc,lambda r:r['fixed_margin_p']<=0.0051):.2f}  clean={frac(cl,lambda r:r['fixed_margin_p']<=0.0051):.2f}")
print(f"    studentized_p <= 0.025 (budget):    FC={frac(fc,lambda r:r['studentized_p']<=0.025):.2f}  clean={frac(cl,lambda r:r['studentized_p']<=0.025):.2f}")
print(f"    LCB_budget>0 not needed here; state gate is studentized_p. mean-T (fixed_margin) is SATURATED for both.")

print("\n############ TABLE 3: subject dominance (few strong vs many weak) ############")
for label,rows in [("NULL_cov FALSE-confirm",sub("NULL_cov",fc=True)),
                   ("NULL_cov clean",sub("NULL_cov",fc=False)),
                   ("POS_concept TRUE-confirm",sub("POS_concept",tc=True))]:
    print(f"  {label:26s} frac_delta_pos={med(rows,'frac_delta_pos'):.3f}  top1_share={med(rows,'top1_abs_share'):.3f}  "
          f"top3_share={med(rows,'top3_abs_share'):.3f}  gini={med(rows,'gini_abs'):.3f}  n={len(rows)}")

print("\n############ TABLE 4: generator mismatch (method fixed-margin vs standard-null diagnostic) ############")
fc=sub("NULL_cov",fc=True)
print(f"  NULL_cov false-confirm: fixed_margin_p_med={med(fc,'fixed_margin_p'):.4f}  standard_null_p_med={med(fc,'standard_null_p'):.4f}")
print(f"    would_confirm_under_standard_null (nonmargin): {frac(fc,lambda r:r['would_confirm_under_standard_null']):.2f} of FC")
print(f"    -> BOTH nulls saturated => under-dispersion is NOT specific to fixed-margin conditioning.")
print(f"    (oracle-generator null = pass 2, to distinguish 'null estimation' vs 'statistic invalid'.)")

print("\n############ DECISIVE: NULL_cov false-confirm vs POS_concept true-confirm (same signature?) ############")
fc=sub("NULL_cov",fc=True); tc=sub("POS_concept",tc=True)
for k in ("observed_T","null_sd_T","T_z","studentized_stat","studentized_p","subject_consistency_lcb","mean_delta","frac_delta_pos","top1_abs_share"):
    print(f"    {k:22s} NULLcov_FC_med={med(fc,k):10.5f}   POS_TC_med={med(tc,k):10.5f}")
