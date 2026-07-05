"""Router Stage R0 exposed threshold frontier (development-only; not confirmatory; not deployable; not a
concept certificate). MONOTONE abstention rule on the existing exposed P3.0b/c forensic:
    allow  if  B3 method_confirm == True  AND  observed_T >= tau
    else   abstain / no actionable evidence
-> can ONLY remove B3 confirmations, never add. Primary deployable score = observed_T. No oracle field, no
condition label in the DECISION (condition used only to EVALUATE per kind, never pooled). Deterministic grid."""
import json, sys, math
try:
    from scipy.stats import beta
    def cp_upper(k,n,a=0.05): return 1.0 if k==n else float(beta.ppf(1-a,k+1,n-k))
except Exception:
    def cp_upper(k,n,a=0.05): return min(1.0,(k+ 1.645*math.sqrt(k+1))/n)  # crude fallback
SRC="/home/infres/yinwang/realeeg_feas/p3_forensic/p3_internal_forensic_merged.json"
R=json.load(open(SRC))["per_cohort"]
CONDS=["NULL_cov","NULL_cov_plus_label","NULL_label","random_label_control","POS_concept","POS_concept_plus_cov"]
NULLS={"NULL_cov","NULL_cov_plus_label","NULL_label","random_label_control"}
def mconf(r): return r.get("b3_state")=="CONCEPT_CONFIRMED"
def T(r):
    t=r.get("observed_T"); return t if isinstance(t,(int,float)) and t==t else None
rows={c:[r for r in R if r["condition"]==c] for c in CONDS}
mc={c:sum(1 for r in rows[c] if mconf(r)) for c in CONDS}
# GUARDRAIL: allow-set is a SUBSET of method-confirms by construction
confT=sorted(set(round(T(r),7) for r in R if mconf(r) and T(r) is not None))
grid=[0.0]+confT           # tau=0 = pure B3 (allow all confirms); each confT = a real cutpoint
def allow(c,tau): return sum(1 for r in rows[c] if mconf(r) and T(r) is not None and T(r)>=tau)
frontier=[]
for tau in grid:
    row=dict(tau=round(tau,7))
    for c in CONDS:
        a=allow(c,tau); row[c]=a
        if c in NULLS: row[c+"_cp95u"]=round(cp_upper(a,300),4)
    frontier.append(row)
# R1-eligibility screen per tau
def eligible(row):
    return (row["NULL_cov"]<=7 and row["NULL_cov_plus_label"]<=7 and row["NULL_label"]<=1
            and row["random_label_control"]<=1 and row["POS_concept"]>0)
elig=[r for r in frontier if eligible(r)]
print("############ R0 THRESHOLD FRONTIER (allow counts /300; CP95u for nulls) ############")
print(f"  {'tau':>9s} | {'NULLcov':>7s} {'NULLcl':>6s} {'NULLlab':>7s} {'rand':>4s} | {'POS':>4s} {'POScov':>6s} | {'NULLcov_cp95u':>13s}")
# print a readable subset: tau=0, and around the first-eligible + a few
show=[frontier[0]]
if elig: show.append(elig[0])
# sample every ~8th cutpoint
show += frontier[1::max(1,len(frontier)//12)]
seen=set()
for row in sorted({r["tau"]:r for r in show}.values(), key=lambda x:x["tau"]):
    print(f"  {row['tau']:>9.5f} | {row['NULL_cov']:>7d} {row['NULL_cov_plus_label']:>6d} {row['NULL_label']:>7d} {row['random_label_control']:>4d} | {row['POS_concept']:>4d} {row['POS_concept_plus_cov']:>6d} | {row['NULL_cov_cp95u']:>13.4f}")
print(f"\n  method_confirm baseline (/300): " + " ".join(f"{c}={mc[c]}" for c in CONDS))
print(f"\n############ R1-ELIGIBLE thresholds (NULLcov&NULLcl allow<=7, NULLlab&rand<=1, POSconcept>0) ############")
if elig:
    e=elig[0]  # smallest tau meeting safety (most POS retained)
    print(f"  smallest eligible tau = {e['tau']:.5f}")
    print(f"    NULL_cov allow={e['NULL_cov']}/300 (cp95u {e['NULL_cov_cp95u']}) | NULL_cov_plus_label allow={e['NULL_cov_plus_label']}/300 (cp95u {e['NULL_cov_plus_label_cp95u']})")
    print(f"    NULL_label={e['NULL_label']} random={e['random_label_control']}")
    print(f"    POS_concept allow={e['POS_concept']}/300 (retention {e['POS_concept']/mc['POS_concept']*100:.0f}% of {mc['POS_concept']} method-confirms)")
    print(f"    POS_concept_plus_cov allow={e['POS_concept_plus_cov']}/300 (retention {e['POS_concept_plus_cov']/mc['POS_concept_plus_cov']*100:.0f}%)")
    print(f"    PREFERRED utility: POS_concept>=20 -> {'YES' if e['POS_concept']>=20 else 'no'} ; POS_cov>=15 -> {'YES' if e['POS_concept_plus_cov']>=15 else 'no'}")
    pref=[r for r in elig if r["POS_concept"]>=20 and r["POS_concept_plus_cov"]>=15]
    print(f"    thresholds meeting BOTH safety AND preferred utility: {len(pref)}")
    print(f"\n  >>> R0 VIABILITY: {'VIABLE frontier exists (safety + some POS)' if e['POS_concept']>0 else 'NOT viable'}")
else:
    print("  NO threshold satisfies the safety screen with any POS retained -> router line NOT viable -> write boundary paper")
# save frontier
json.dump(dict(diagnostic_only=True,not_confirmatory=True,not_deployable=True,not_a_certificate=True,
               rule="allow if method_confirm AND observed_T>=tau (monotone; subset of B3 confirms)",
               primary_score="observed_T", method_confirm_baseline=mc, n_grid=len(grid),
               frontier=frontier, r1_eligible_smallest_tau=(elig[0]["tau"] if elig else None)),
          open("router_stage0/router_stage0_frontier.json","w"), indent=1, default=str)
print("\nsaved router_stage0_frontier.json")
