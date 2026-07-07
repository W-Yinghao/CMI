"""B8.3 Phase-A FEASIBILITY smoke (development-only; NO science verdict). Checks: (1) the label-balanced case-control
audit selector is FEASIBLE on SM16 (audit_selected_n large enough, enough eligible subjects post-audit); (2) observed
C x Y imbalance == 0 EXACTLY (balanced by construction); (3) null draws re-apply the selector and stay balanced (low
infeasible-draw rate); (4) CONTRACT worlds reach a computed-T state, VIOLATION worlds refuse; (5) no Z leakage into the
selector (audit_select signature takes no Z). Fresh smoke base (never a canary base). Fail-loud on infeasibility."""
import os, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, "/home/infres/yinwang/realeeg_feas")
import inspect
import numpy as np
import realeeg_internal_forensic as IF
import realeeg_b8_3 as B83
import realeeg_b8_1 as B81

SMOKE_BASE = 774_000_000
M = IF.M
NC = 8          # Phase-A: 8 cohorts/condition, feasibility only
NB = 120        # closer to Phase-B NB=200 so estimability (ninv<=0.20*NB) is meaningfully exercised


def build(world, seed):
    rng = np.random.default_rng(seed)
    subj = rng.choice(IF.SUBJECTS, size=min(M, len(IF.SUBJECTS)), replace=False)
    sel = np.isin(IF.CACHE["subject"], subj)
    coh = {k: IF.CACHE[k][sel] for k in ("Z", "y", "subject", "session")}
    return B81.build_b8_1_cohort(world, coh, rng)


def main():
    # (5) no-Z-leakage structural check
    sig = list(inspect.signature(B83.audit_select).parameters)
    assert sig == ["C", "Y", "G", "Block", "seed"], f"audit_select signature leaks? {sig}"
    print(f"[b8.3 smoke] audit_select params = {sig} (NO Z/T) OK")

    fails = []
    print(f"\n{'world':34s} {'valid%':>6} {'selN_med':>8} {'selN_min':>8} {'nelig_med':>9} {'imb_max':>7} {'ninfeas_med':>11} {'states'}")
    for w in B83.WORLDS:
        kind, hc = B83.WORLDS[w]
        selN, nelig, imbs, ninf, states, valids = [], [], [], [], [], 0
        for r in range(NC):
            seed = SMOKE_BASE + list(B83.WORLDS).index(w) * 1000 + r
            Z, Y, C, G, Block, Dc, Ct, th, intended = build(w, seed)
            rr = B83.b8_3_certify(Z, Y, C, G, Block, Dc, Ct, th, m=M, seed=seed, n_boot=NB)
            selN.append(rr["audit_selected_n"]); nelig.append(rr["n_eligible"]); imbs.append(rr["audit_cxy_imbalance"])
            ninf.append(rr["null_infeasible_draws"]); states.append(str(rr["b8_state"]))
            valids += int(rr["contract_valid"])
        from collections import Counter
        sc = dict(Counter(states))
        print(f"{w:34s} {100*valids/NC:>5.0f}% {np.median(selN):>8.0f} {min(selN):>8.0f} {np.median(nelig):>9.0f} "
              f"{max(imbs):>7} {np.median(ninf):>11.0f} {sc}")
        # feasibility invariants
        if kind == "contract":
            if valids != NC:
                fails.append(f"{w}: CONTRACT valid only {valids}/{NC}")
            # observed balance must be EXACT (0) whenever a subset was selected
            bad_imb = [i for i in imbs if i > 0]
            if bad_imb:
                fails.append(f"{w}: observed C x Y imbalance != 0 ({bad_imb}) -- selector not balanced")
            # feasibility: audit sample must be non-degenerate + enough eligible subjects to compute T
            if min(selN) <= 0:
                fails.append(f"{w}: audit selection EMPTY in some cohort (infeasible)")
            if np.median(nelig) < 20:   # == min_confirm_pairs: feasibility must guarantee ALERT-capability (red-team wrznv3lin)
                fails.append(f"{w}: median post-audit eligible subjects {np.median(nelig):.0f} < 20 (min_confirm_pairs -> cannot ALERT)")
            # CONTRACT-FEASIBILITY (hard): the AUDIT CONTRACT must be feasible -- the selector must reach the STATISTIC
        # (ALERT/NO_ACTIONABLE/SAMPLER_INVALID), NOT be blocked by INSUFFICIENT_LABELS (which would mean the balanced
        # audit sample is too small). SAMPLER_INVALID = estimability-at-NB (a statistic property, reported separately),
        # NOT a contract-feasibility failure.
            insuf = sc.get("INSUFFICIENT_LABELS", 0)
            if insuf > 0:
                fails.append(f"{w}: {insuf}/{NC} INSUFFICIENT_LABELS -- balanced audit sample too small (CONTRACT infeasible)")
            reached = sc.get("B8_CONCEPT_ALERT", 0) + sc.get("NO_ACTIONABLE_CONCEPT_EVIDENCE", 0) + sc.get("SAMPLER_INVALID", 0)
            if reached < NC:
                fails.append(f"{w}: only {reached}/{NC} reached the statistic (contract-feasibility gap)")
            # ESTIMABILITY (reported, soft): SAMPLER_INVALID rate at this NB -- flag only if it DOMINATES (>25%)
            samp = sc.get("SAMPLER_INVALID", 0)
            if samp > 0.25 * NC:
                fails.append(f"{w}: SAMPLER_INVALID {samp}/{NC} > 25% at NB={NB} -- estimability concern (check whether it mitigates at Phase-B NB=200)")
            if np.median(ninf) > 0.5 * NB:
                fails.append(f"{w}: median null-infeasible draws {np.median(ninf):.0f}/{NB} > 50% -- null selection unstable")
        else:
            if sc.get("CONTRACT_INVALID_OR_UNIDENTIFIABLE", 0) != NC:
                fails.append(f"{w}: VIOLATION not fully refused ({sc})")

    print("\n=== Phase-A FEASIBILITY verdict ===")
    if fails:
        print("SMOKE_FEASIBILITY_FAILURES:")
        for f in fails: print("  " + f)
        print("B8_3_SMOKE_INFEASIBLE")
        sys.exit(1)
    print("B8_3_SMOKE_FEASIBLE (selector balanced+feasible; CONTRACT reach computed-T; VIOLATION refuse). NO science verdict.")


if __name__ == "__main__":
    main()
