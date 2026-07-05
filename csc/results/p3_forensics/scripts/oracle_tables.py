"""P3.0d Tables A-D from the merged oracle diagnostic."""
import json, statistics as st
M=json.load(open("/home/infres/yinwang/realeeg_feas/p3_forensic/oracle/p3_oracle_merged.json"))
R=M["per_cohort"]; FLOOR=M["oracle_p_floor"]; A=0.025
def sel(cond=None,**kw):
    o=[r for r in R if (cond is None or r["condition"]==cond)]
    for k,v in kw.items(): o=[r for r in o if r.get(k)==v]
    return o
def rate(rows,key): return sum(1 for r in rows if r.get(key))/len(rows) if rows else float("nan")
def med(rows,k):
    v=[r[k] for r in rows if isinstance(r.get(k),(int,float)) and r[k]==r[k]]; return st.median(v) if v else float("nan")
def fl(rows,k,thr): return sum(1 for r in rows if isinstance(r.get(k),(int,float)) and r[k]<=thr)/len(rows) if rows else float("nan")

def n_invalid(rows): return sum(1 for r in rows if r.get("oracle_applicable") is False or r.get("method_invalid"))
print("############ TABLE A: all-cohort calibration (method vs oracle null) ############")
print("(rates over ALL n cohorts; n_inval = method-invalid cohorts [nan T], counted as non-confirms)")
print(f"  {'condition':22s} {'n':>4s} {'n_inval':>7s} {'m_conf':>7s} {'o_conf':>7s} | {'m_stP<=.025':>11s} {'o_stP<=.025':>11s} | {'o_stP_med':>9s}")
for c in ("NULL_cov","NULL_cov_plus_label","POS_concept","POS_concept_plus_cov"):
    rows=sel(c); gt_nc = c in ("NULL_cov","NULL_cov_plus_label")
    mconf = rate(rows,"method_false_confirm") if gt_nc else rate(rows,"method_true_confirm")
    oconf = rate(rows,"oracle_false_confirm") if gt_nc else rate(rows,"oracle_true_confirm")
    print(f"  {c:22s} {len(rows):>4d} {n_invalid(rows):>7d} {mconf:>7.3f} {oconf:>7.3f} | "
          f"{fl(rows,'method_studentized_p',A):>11.2f} {fl(rows,'oracle_studentized_p',A):>11.2f} | {med(rows,'oracle_studentized_p'):>9.4f}")

print("\n############ TABLE B: selected failure mechanism (NULL_cov method-FC vs matched clean) ############")
fc=sel("NULL_cov",method_false_confirm=True); cl=sel("NULL_cov",method_false_confirm=False)
print(f"  NULL_cov: {len(fc)} method false-confirm, {len(cl)} clean")
print(f"    oracle_confirm_rate:  FC={rate(fc,'oracle_false_confirm'):.3f}   clean={rate(cl,'oracle_false_confirm'):.3f}")
for k in ("oracle_fixed_margin_p","oracle_studentized_p","observed_T","oracle_null_sd_T","studentized_stat","lcb_budget"):
    print(f"    {k:22s} FC_med={med(fc,k):10.5f}   clean_med={med(cl,k):10.5f}")
print(f"    -> of the {len(fc)} method false-confirms, oracle still confirms {sum(1 for r in fc if r['oracle_false_confirm'])} "
      f"({rate(fc,'oracle_false_confirm')*100:.0f}%).")

print("\n############ TABLE C: POS separation (does oracle preserve true concept?) ############")
for c in ("POS_concept","POS_concept_plus_cov"):
    rows=sel(c); tc=sel(c,method_true_confirm=True)
    print(f"  {c}: method_TC={rate(rows,'method_true_confirm'):.3f} oracle_TC={rate(rows,'oracle_true_confirm'):.3f} "
          f"| of method-TC, oracle keeps {rate(tc,'oracle_true_confirm')*100:.0f}%  | T_med(TC)={med(tc,'observed_T'):.5f} o_stP_med(TC)={med(tc,'oracle_studentized_p'):.4f}")

print("\n############ TABLE D: decision matrix (method x oracle) ############")
print(f"  {'condition':22s} {'m+/o+':>6s} {'m+/o-':>6s} {'m-/o+':>6s} {'m-/o-':>6s}")
for c in ("NULL_cov","NULL_cov_plus_label","POS_concept","POS_concept_plus_cov"):
    rows=sel(c); gt_nc=c in ("NULL_cov","NULL_cov_plus_label")
    mk = "method_false_confirm" if gt_nc else "method_true_confirm"
    ok_ = "oracle_false_confirm" if gt_nc else "oracle_true_confirm"
    pp=sum(1 for r in rows if r[mk] and r[ok_]); pm=sum(1 for r in rows if r[mk] and not r[ok_])
    mp=sum(1 for r in rows if not r[mk] and r[ok_]); mm=sum(1 for r in rows if not r[mk] and not r[ok_])
    print(f"  {c:22s} {pp:>6d} {pm:>6d} {mp:>6d} {mm:>6d}")
print("  legend: m+/o- = method confirms, oracle does NOT -> fitted-null artifact (Case 1).")
print("          m+/o+ = confirms under BOTH -> statistic extreme under true generator (Case 2).")
