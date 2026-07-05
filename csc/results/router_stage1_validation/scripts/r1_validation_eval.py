"""Evaluate the LOCKED tau_R1 on the fresh held-out validation block (base 90e6, all 6). tau is NOT re-derived."""
import json, math
try:
    from scipy.stats import beta
    def cpu(k,n,a=0.05): return 1.0 if k==n else float(beta.ppf(1-a,k+1,n-k))
except Exception:
    def cpu(k,n,a=0.05): return min(1.0,(k+1.645*math.sqrt(k+1))/n)
import statistics as st
LOCK=json.load(open("/home/infres/yinwang/realeeg_feas/router_stage1/router_stage1_tau_lock.json"))
TAU=LOCK["tau_R1"]
V=json.load(open("/home/infres/yinwang/realeeg_feas/router_stage1/validation/r1_validation_merged.json"))["per_cohort"]
CONDS=["NULL_cov","NULL_cov_plus_label","NULL_label","random_label_control","POS_concept","POS_concept_plus_cov"]
NULLS={"NULL_cov","NULL_cov_plus_label","NULL_label","random_label_control"}
def mconf(r): return r.get("b3_state")=="CONCEPT_CONFIRMED"
def T(r):
    t=r.get("observed_T"); return t if isinstance(t,(int,float)) and t==t else None
def invalid(r): return T(r) is None or r.get("b3_state") in (None,"") or not isinstance(r.get("b3_state"),str)
rows={c:[r for r in V if r["condition"]==c] for c in CONDS}
print(f"=== HELD-OUT VALIDATION @ LOCKED tau_R1={TAU} (comparison >=) ===")
tab={}
print(f"  {'condition':22s} {'n':>3s} {'ninv':>4s} {'mconf':>5s} {'allow':>5s} {'rate':>6s} {'CP95u':>7s} {'medT_allow':>10s} {'medT_abst':>9s}")
for c in CONDS:
    rs=rows[c]; ninv=sum(1 for r in rs if invalid(r))
    mconf_n=sum(1 for r in rs if mconf(r) and T(r) is not None)
    allowed=[r for r in rs if mconf(r) and T(r) is not None and T(r)>=TAU]
    abst=[r for r in rs if not (mconf(r) and T(r) is not None and T(r)>=TAU)]
    a=len(allowed); cp=cpu(a,300) if c in NULLS else None
    mta=st.median([T(r) for r in allowed]) if allowed else float("nan")
    mtb=st.median([T(r) for r in abst if T(r) is not None]) if abst else float("nan")
    tab[c]=dict(n=300,n_invalid=ninv,method_confirm=mconf_n,router_allow=a,allow_rate=round(a/300,4),cp95u=(round(cp,4) if cp else None))
    print(f"  {c:22s} {300:>3d} {ninv:>4d} {mconf_n:>5d} {a:>5d} {a/300:>6.3f} {(cp if cp else 0):>7.4f} {mta:>10.5f} {mtb:>9.5f}")
nc,ncl,nl,rnd,pos,posc=[tab[c]["router_allow"] for c in CONDS]
print(f"\n  MAIN SUMMARY (allow counts /300): tau_R1 | NULL_cov={nc} | NULL_cov+label={ncl} | NULL_label={nl} | rand={rnd} | POS={pos} | POS+cov={posc}")
# endpoints
prim = (nc<=7 and ncl<=7)
sec_fail = (nl>7 or rnd>7)
util = ("STRONG" if (pos>=20 and posc>=15) else "WEAK" if (prim and pos>0) else "FAIL")
verdict = ("SAFETY-FAIL" if (not prim or sec_fail) else
           "SAFETY-PASS + STRONG utility -> eligible for frozen-protocol discussion" if util=="STRONG" else
           "SAFETY-PASS + WEAK utility (safe-but-weak)" if util=="WEAK" else
           "SAFETY-PASS + ZERO utility (abstention-only boundary)")
print(f"\n  PRIMARY safety (NULL_cov & NULL_cov+label <=7): {'PASS' if prim else 'FAIL'} ({nc},{ncl})")
print(f"  SECONDARY (NULL_label & rand; fail if >7): NULL_label={nl} rand={rnd} -> {'FAIL' if sec_fail else 'ok'}")
print(f"  UTILITY: POS_concept={pos} (>=20 strong,>0 weak) POS_cov={posc} (>=15) -> {util}")
print(f"\n  >>> R1 VERDICT: {verdict}")
json.dump(dict(diagnostic_only=True,tau_R1=TAU,tau_R1_source="locked from calibration; NOT re-derived",
  per_condition=tab,main_summary=dict(NULL_cov=nc,NULL_cov_plus_label=ncl,NULL_label=nl,random_label_control=rnd,POS_concept=pos,POS_concept_plus_cov=posc),
  primary_safety_pass=prim,secondary_fail=sec_fail,utility_tier=util,verdict=verdict),
  open("router_stage1/router_stage1_validation_tables.json","w"),indent=1,default=str)
print("\nsaved router_stage1_validation_tables.json")
