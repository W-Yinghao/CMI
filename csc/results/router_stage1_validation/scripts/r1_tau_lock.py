"""Lock tau_R1 from the CALIBRATION merged (null-only). Deterministic grid = sorted unique observed_T among
calibration B3-method-confirms + sentinels {0, +inf}. tau_R1 = smallest tau s.t. NULL_cov allow<=3/300 AND
NULL_cov_plus_label allow<=3/300 (conservative). Also report tau_boundary(<=7) and tau_zero(0). NULL-ONLY."""
import json, sys, math, struct
try:
    from scipy.stats import beta
    def cpu(k,n,a=0.05): return 1.0 if k==n else float(beta.ppf(1-a,k+1,n-k))
except Exception:
    def cpu(k,n,a=0.05): return min(1.0,(k+1.645*math.sqrt(k+1))/n)
CAL="/home/infres/yinwang/realeeg_feas/router_stage1/calibration/r1_calibration_merged.json"
R=json.load(open(CAL))["per_cohort"]
NULLS=["NULL_cov","NULL_cov_plus_label"]
def mconf(r): return r.get("b3_state")=="CONCEPT_CONFIRMED"
def T(r):
    t=r.get("observed_T"); return t if isinstance(t,(int,float)) and t==t else None
rows={c:[r for r in R if r["condition"]==c] for c in NULLS}
mc={c:sum(1 for r in rows[c] if mconf(r)) for c in NULLS}
grid=[0.0]+sorted(set(round(T(r),9) for r in R if mconf(r) and T(r) is not None))+[float("inf")]
def allow(c,tau): return sum(1 for r in rows[c] if mconf(r) and T(r) is not None and T(r)>=tau)
def first(pred):
    for tau in grid:
        if all(pred(allow(c,tau)) for c in NULLS): return tau
    return float("inf")
tau_R1=first(lambda a: a<=3)
tau_boundary=first(lambda a: a<=7)
tau_zero=first(lambda a: a==0)
rank=grid.index(tau_R1)
hexf=(struct.pack('>d',tau_R1).hex() if tau_R1!=float("inf") else "inf")
lock=dict(diagnostic_only=True,null_only_calibration=True,not_deployable=True,
  tau_R1=tau_R1, tau_R1_hexfloat=hexf, grid_rank=rank, grid_size=len(grid), comparison=">=",
  tau_boundary_le7=tau_boundary, tau_zero=tau_zero,
  calibration_counts_at_tau_R1={c:allow(c,tau_R1) for c in NULLS},
  calibration_counts_at_boundary={c:allow(c,tau_boundary) for c in NULLS},
  cp95u_at_tau_R1={c:round(cpu(allow(c,tau_R1),300),4) for c in NULLS},
  method_confirm_baseline=mc,
  abstention_only=(tau_R1==float("inf")),
  note="locked from CALIBRATION only; evaluated ONCE on held-out; NOT the exposed R0 tau")
json.dump(lock, open("router_stage1/router_stage1_tau_lock.json","w"), indent=1, default=str)
print("=== TAU LOCK (calibration, null-only) ===")
print(f"  method_confirm baseline: {mc}")
print(f"  tau_R1 = {tau_R1} (hexfloat {hexf}, grid rank {rank}/{len(grid)})")
print(f"    calibration allow at tau_R1: {lock['calibration_counts_at_tau_R1']}  cp95u {lock['cp95u_at_tau_R1']}")
print(f"  tau_boundary(<=7) = {tau_boundary}  -> {lock['calibration_counts_at_boundary']}")
print(f"  tau_zero(=0) = {tau_zero}")
print(f"  abstention_only = {lock['abstention_only']}")
print("\nsaved router_stage1_tau_lock.json")
