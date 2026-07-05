import json, statistics as st
M=json.load(open("/home/infres/yinwang/realeeg_feas/b4_stage1b/b4_stage1b_merged.json")); R=M["per_cohort"]
def sub(cond,strat): return [r for r in R if r.get("condition")==cond and r.get("stratum")==strat and r.get("b4c_applicable")]
def nconf(rows): return sum(1 for r in rows if r.get("b4c_confirm"))
def nmconf(rows): return sum(1 for r in rows if r.get("method_confirm"))
def med(rows,k):
    v=[r[k] for r in rows if isinstance(r.get(k),(int,float)) and r[k]==r[k]]; return st.median(v) if v else float("nan")
print("############ B4c HARD SAFETY SCREEN (NULL m+/o- <=5/20 pass; >=10/20 HARD FAIL ; clean m-/o- <=1/20) ############")
print(f"  {'stratum':32s} {'n':>3s} | {'method':>6s} {'B4c':>4s} {'oracle':>6s}  cap")
b4c_pass=True; hard_fail=False
for cond in ("NULL_cov","NULL_cov_plus_label"):
    for strat,cap in (("m+/o-",5),("m-/o-",1)):
        rows=sub(cond,strat); a=nconf(rows); m=nmconf(rows); o=sum(1 for r in rows if r.get("archived_oracle_confirm"))
        if strat=="m+/o-" and a>=10: hard_fail=True
        if a>cap: b4c_pass=False
        print(f"  {cond+':'+strat:32s} {len(rows):>3d} | {m:>6d} {a:>4d} {o:>6d}  <= {cap}")
print("\n############ POS USEFULNESS (m+/o+ : >=6/20 pref, >0 min) ############")
for cond in ("POS_concept","POS_concept_plus_cov"):
    rows=sub(cond,"m+/o+"); print(f"  {cond:22s} m+/o+ n={len(rows)} | method={nmconf(rows)} B4c={nconf(rows)} oracle={sum(1 for r in rows if r.get('archived_oracle_confirm'))}")
print("\n############ OBSERVED-T SHRINKAGE (NULL_cov m+/o-) + NULL-DISPERSION ############")
nc=sub("NULL_cov","m+/o-")
print(f"  NULL_cov m+/o-: B3 observed_T median={med(nc,'archived_B3_observed_T'):.5f}  B4c observed_T median={med(nc,'b4c_observed_T'):.5f}  (shrinkage {(1-med(nc,'b4c_observed_T')/med(nc,'archived_B3_observed_T'))*100:.0f}%)")
allr=[r for r in R if r.get('b4c_applicable')]
print(f"  null_sd median: method {med(allr,'method_null_sd'):.6f} | B4a {med(allr,'b4a_null_sd'):.6f} | B4b {med(allr,'b4b_null_sd'):.6f} | B4c {med(allr,'b4c_null_sd'):.6f} | oracle {med(allr,'archived_oracle_null_sd_T'):.6f}")
print(f"  NULL_cov m+/o- p: method_fmP@floor(<=0.0051) {sum(1 for r in nc if (r.get('method_fixed_margin_p') or 1)<=0.0051)}/{len(nc)}  | B4c_fmP@floor {sum(1 for r in nc if r['b4c_fixed_margin_p']<=0.0051)}/{len(nc)}")
print(f"  NULL_cov m+/o- studentized_p median: B4c={med(nc,'b4c_studentized_p'):.3f}  (method~0.012, oracle~0.264)")
print(f"\n  ELIGIBILITY: B4c hard-safety {'PASS' if b4c_pass else 'FAIL'}  | hard-fail-trigger(>=10/20)={hard_fail}")
