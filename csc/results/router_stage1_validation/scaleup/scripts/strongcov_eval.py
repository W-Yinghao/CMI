"""Evaluate the strong-covariate NO-CONCEPT null at the LOCKED tau_R1 (NOT re-derived). Tests the router's predicted
failure mode: does type-I (allow @ tau) climb as the session covariate strengthens (delta 1.5->auc~0.80, 2.5->~0.92),
vs the soft-covariate R1 result NULL_cov 5/300 (auc~0.52)? Fail-closed. DEVELOPMENT-diagnostic, NO retune."""
import os, sys, json, math
import numpy as np
try:
    from scipy.stats import beta
    def cp_upper(k, n, a=0.05): return 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
except Exception:
    def cp_upper(k, n, a=0.05): return min(1.0, (k + 1.645 * math.sqrt(k + 1)) / n)

FEAS = "/home/infres/yinwang/realeeg_feas"
TAU = json.load(open(f"{FEAS}/router_stage1/router_stage1_tau_lock.json"))["tau_R1"]
SCDIR = f"{FEAS}/router_stage1/strongcov"
BASE = 100_000_000
DELTAS = [1.5, 2.5]
STARTS = (0,)   # consolidated: one 300-cohort file per delta


def _read(p): return [json.loads(l) for l in open(p) if l.strip()]
def T(r):
    t = r.get("observed_T"); return t if isinstance(t, (int, float)) and t == t else None
def med(xs):
    xs = [x for x in xs if x is not None and x == x]
    return float(np.median(xs)) if xs else float("nan")


def main():
    print(f"=== STRONG-COVARIATE null @ LOCKED tau_R1={TAU} (GT NO_CONCEPT; any allow = type-I) ===")
    print(f"  {'delta':>6} {'n':>4} {'auc_med':>8} {'mconf':>6} {'allow':>6} {'rate':>7} {'CP95u':>7} {'Tz_med':>7} {'ffp_floor':>9} {'invalid':>7}")
    rows = {}
    for d in DELTAS:
        recs = []
        for s in STARTS:
            p = f"{SCDIR}/sc_{BASE}_d{d}_{s}.jsonl"
            if not os.path.exists(p):
                print(f"FAIL-CLOSED: missing {p}"); sys.exit(2)
            pr = json.load(open(p + ".prov.json")) if os.path.exists(p + ".prov.json") else {}
            if pr.get("n_worker_errors", 1) != 0:
                print(f"FAIL-CLOSED: worker errors in {p}"); sys.exit(2)
            rs = _read(p)
            if any("__worker_error__" in r for r in rs):
                print(f"FAIL-CLOSED: worker-error rows in {p}"); sys.exit(2)
            recs.extend(rs)
        if len(recs) != 300:
            print(f"FAIL-CLOSED: delta {d} has {len(recs)} != 300"); sys.exit(2)
        mconf = sum(1 for r in recs if r.get("b3_state") == "CONCEPT_CONFIRMED")
        allow = sum(1 for r in recs if r.get("b3_state") == "CONCEPT_CONFIRMED" and T(r) is not None and T(r) >= TAU)
        invalid = sum(1 for r in recs if not r.get("valid", False))
        auc = med([r.get("session_auc") for r in recs])
        tz = med([r.get("T_z") for r in recs])
        ffp = [r.get("fixed_margin_p") for r in recs if r.get("fixed_margin_p") is not None]
        ffp_floor = float(np.mean([1.0 if (p is not None and p <= 0.005 + 1e-9) else 0.0 for p in ffp])) if ffp else float("nan")
        cp = cp_upper(allow, 300)
        rows[d] = dict(n=300, session_auc_med=auc, method_confirm=mconf, router_allow=allow,
                       allow_rate=round(allow / 300, 4), cp95u=round(cp, 4), T_z_med=tz,
                       ffp_floor=ffp_floor, n_invalid=invalid)
        print(f"  {d:>6} {300:>4} {auc:>8.3f} {mconf:>6} {allow:>6} {allow/300:>7.3f} {cp:>7.4f} {tz:>7.2f} {ffp_floor:>9.2f} {invalid:>7}")
    print(f"\n  reference: SOFT covariate (R1 held-out NULL_cov) session_auc~0.52 -> allow 5/300 (CP95u 0.0347)")
    breach = {d: rows[d]["router_allow"] for d in DELTAS if rows[d]["router_allow"] > 7}
    print(f"  >>> router type-I @ strong covariate: " + ", ".join(f"d{d}(auc{rows[d]['session_auc_med']:.2f})={rows[d]['router_allow']}/300" for d in DELTAS))
    print(f"  >>> CAP-BREACH (>7/300) at: {breach if breach else 'NONE -- router holds under strong covariate'}")
    out = dict(diagnostic_only=True, tau_R1=TAU, tau_source="locked; not re-derived", base=BASE,
               soft_covariate_reference=dict(condition="NULL_cov", session_auc_approx=0.52, allow=5, cp95u=0.0347),
               per_delta=rows, cap=7, breach=breach,
               router_holds_under_strong_covariate=bool(not breach))
    json.dump(out, open(f"{SCDIR}/strongcov_tables.json", "w"), indent=1, default=str)
    print(f"\n  saved {SCDIR}/strongcov_tables.json")


if __name__ == "__main__":
    main()
