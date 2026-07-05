"""P3.0 four-table report from forensics_diagnostics.json (no new compute).
Quantifies whether NULL_cov false-confirms CONCENTRATE in high-session-AUC / low-overlap cohorts, and the
threshold-sensitivity cost of an overlap/ESS gate on true POS_concept power."""
import json, statistics as st
D = json.load(open("/home/infres/yinwang/realeeg_feas/forensics_diagnostics.json"))

def mwu_auc(pos, neg):
    """Mann-Whitney rank AUC = P(score_pos > score_neg); 0.5 = no separation. Robust to n."""
    pos=[x for x in pos if x==x]; neg=[x for x in neg if x==x]
    if not pos or not neg: return float("nan")
    wins=ties=0
    for a in pos:
        for b in neg:
            if a>b: wins+=1
            elif a==b: ties+=1
    return (wins+0.5*ties)/(len(pos)*len(neg))

def tbl(cond, direction):
    rows=D[cond]; conf=[r for r in rows if r["confirmed"]]; noc=[r for r in rows if not r["confirmed"]]
    print(f"\n===== {cond}  (confirmed={len(conf)}, not={len(noc)}) =====")
    print(f"  {'metric':16s} {'conf_median':>11s} {'not_median':>11s} {'conf_mean':>10s} {'not_mean':>10s} {'sep_AUC':>8s}  interp")
    for key,hi_bad in [("session_auc",True),("overlap_frac",False),("ess_frac",False),("max_w_ratio",True)]:
        c=[r[key] for r in conf if r[key]==r[key]]; n=[r[key] for r in noc if r[key]==r[key]]
        if not c or not n:
            print(f"  {key:16s}  (no confirmed)"); continue
        # separation AUC: does the metric rank-predict false-confirm? high_bad -> use metric; else use -metric
        sep = mwu_auc(c,n) if hi_bad else mwu_auc([-x for x in c],[-x for x in n])
        flag = "CONCENTRATED" if sep>=0.65 else ("weak" if sep>=0.58 else "NO separation")
        print(f"  {key:16s} {st.median(c):11.4f} {st.median(n):11.4f} {st.mean(c):10.4f} {st.mean(n):10.4f} {sep:8.3f}  {flag}")

print("############ TABLE 1-3: does false-confirm concentrate in high-AUC / low-overlap? ############")
print("(sep_AUC = rank-prob that a false-confirm cohort is MORE covariate-problematic than a clean one;")
print(" 0.50 = no concentration, >=0.65 = concentrated. For overlap/ess a LOWER value is 'more problematic'.)")
for c in ("NULL_cov","NULL_cov_plus_label","POS_concept"):
    tbl(c, None)

print("\n\n############ TABLE 4: overlap/ESS gate threshold sensitivity ############")
print("A gate emits UNIDENTIFIABLE_COVARIATE_SUPPORT when the metric is below the threshold, overriding B3.")
print("Good gate = converts NULL_cov false-confirms to abstain WITHOUT killing POS_concept true-confirms.\n")
nullcov=D["NULL_cov"]; pospos=D["POS_concept"]
nc_fc=[r for r in nullcov if r["confirmed"]]                       # 15 false-confirms to catch
pos_tc=[r for r in pospos if r["confirmed"]]                       # 32 true-confirms to protect
import numpy as np
for metric in ("overlap_frac","ess_frac"):
    print(f"  -- gate on {metric} (abstain if < thr) --")
    vals=sorted(set(round(r[metric],4) for r in nullcov+pospos))
    print(f"     {'thr':>8s} {'NULLcov_FC_gated':>16s} {'POS_TC_gated(lost)':>19s} {'specificity':>12s}")
    for thr in np.quantile([r[metric] for r in nullcov+pospos],[0.05,0.10,0.25,0.50,0.75]):
        fc_g=sum(1 for r in nc_fc if r[metric]<thr)
        tc_g=sum(1 for r in pos_tc if r[metric]<thr)
        # specificity = fraction of gated-decisions that are false-confirms (want high); guard /0
        spec = fc_g/(fc_g+tc_g) if (fc_g+tc_g)>0 else float("nan")
        print(f"     {thr:8.4f} {fc_g:>7d}/{len(nc_fc):<7d} {tc_g:>7d}/{len(pos_tc):<7d}   {spec:12.3f}")
print("\n(If FC_gated and TC_gated fall at similar rates -> the gate is NON-SPECIFIC: it cannot remove false")
print(" confirmations without proportionally destroying true concept power.)")
