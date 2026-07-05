"""B4 Stage 1 canary pass-criteria tables from the merged canary."""
import json, statistics as st
M=json.load(open("/home/infres/yinwang/realeeg_feas/b4_stage1/b4_stage1_merged.json")); R=M["per_cohort"]
def sub(cond,strat): return [r for r in R if r.get("condition")==cond and r.get("stratum")==strat]
def n_conf(rows,arm): return sum(1 for r in rows if r.get(f"{arm}_confirm"))
def med(rows,k):
    v=[r[k] for r in rows if isinstance(r.get(k),(int,float)) and r[k]==r[k]]; return st.median(v) if v else float("nan")
print("############ HARD SAFETY SCREEN (per arm; NULL m+/o- <=5/20 ; NULL clean m-/o- <=1/20) ############")
print(f"  {'stratum':34s} {'n':>3s} | {'method':>7s} {'B4a':>5s} {'B4b':>5s} {'oracle(arch)':>12s}")
screen={"b4a":[],"b4b":[]}
for cond in ("NULL_cov","NULL_cov_plus_label"):
    for strat,cap in (("m+/o-",5),("m-/o-",1)):
        rows=sub(cond,strat)
        ma,a,b=n_conf(rows,"method"),n_conf(rows,"b4a"),n_conf(rows,"b4b")
        oarch=sum(1 for r in rows if r.get("archived_oracle_confirm"))
        for arm,c in (("b4a",a),("b4b",b)):
            if c>cap: screen[arm].append(f"{cond}:{strat} {c}>{cap}")
        print(f"  {cond+':'+strat:34s} {len(rows):>3d} | {ma:>7d} {a:>5d} {b:>5d} {oarch:>12d}   (cap {cap})")
print("\n############ POS USEFULNESS SCREEN (POS m+/o+ : >=6/20 preferred, >0 minimum) ############")
for cond in ("POS_concept","POS_concept_plus_cov"):
    rows=sub(cond,"m+/o+")
    ma,a,b=n_conf(rows,"method"),n_conf(rows,"b4a"),n_conf(rows,"b4b")
    oarch=sum(1 for r in rows if r.get("archived_oracle_confirm"))
    print(f"  {cond:22s} m+/o+ n={len(rows)} | method={ma} B4a={a} B4b={b} oracle(arch)={oarch}")
print("\n############ NULL-DISPERSION SCREEN (median null_sd; must move method->oracle) ############")
allr=[r for r in R if r.get("fidelity_ok")]
print(f"  method null_sd median = {med(allr,'method_null_sd'):.6f}")
print(f"  B4a    null_sd median = {med(allr,'b4a_null_sd'):.6f}   (ratio B4a/method = {med(allr,'b4a_null_sd')/med(allr,'method_null_sd'):.2f})")
print(f"  B4b    null_sd median = {med(allr,'b4b_null_sd'):.6f}   (ratio B4b/method = {med(allr,'b4b_null_sd')/med(allr,'method_null_sd'):.2f})")
print(f"  oracle null_sd median (archived) = {med(allr,'archived_oracle_null_sd_T'):.6f}   (ratio oracle/method = {med(allr,'baseline_inflation_ratio_oracle_over_method'):.2f})")
nc=[r for r in allr if r.get('condition')=='NULL_cov' and r.get('stratum')=='m+/o-']
print(f"\n  NULL_cov m+/o- studentized_p medians: method={med(nc,'method_studentized_p'):.3f} B4a={med(nc,'b4a_studentized_p'):.3f} B4b={med(nc,'b4b_studentized_p'):.3f} oracle={med(nc,'archived_oracle_studentized_p'):.3f}")
print(f"  method fixed_margin_p at floor(<=0.0051) on NULL_cov m+/o-: {sum(1 for r in nc if r['method_fixed_margin_p']<=0.0051)}/{len(nc)}  |  B4a: {sum(1 for r in nc if r['b4a_fixed_margin_p']<=0.0051)}/{len(nc)}")
print("\n############ ELIGIBILITY SUMMARY ############")
for arm in ("b4a","b4b"):
    print(f"  {arm}: hard-safety {'PASS' if not screen[arm] else 'FAIL '+str(screen[arm])}")
