"""B8.1 smoke (development-only): plumbing check on real SM16 geometry. Verifies (1) every CONTRACT world validates;
(2) every VIOLATION world REFUSES with a sensible reason; (3) the QUIET-confound worlds have AUC<=tau (an AUC-only gate
would PASS) yet provenance REFUSES; (4) b8_1_certify runs end-to-end on representatives (POS low p, prior_only runs).
NOT the canary. Fresh smoke seed (never the canary base). Fail-loud."""
import os, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, "/home/infres/yinwang/realeeg_feas")
import numpy as np
import realeeg_internal_forensic as IF
import realeeg_b8_1 as B8

SMOKE_BASE = 771_000_000    # smoke only; canary uses a different fresh base
M = IF.M


def build(world, seed):
    rng = np.random.default_rng(seed)
    subj = rng.choice(IF.SUBJECTS, size=min(M, len(IF.SUBJECTS)), replace=False)
    sel = np.isin(IF.CACHE["subject"], subj)
    coh = {k: IF.CACHE[k][sel] for k in ("Z", "y", "subject", "session")}
    return B8.build_b8_1_cohort(world, coh, rng), coh


def main():
    print("=== check_contract_b8_1 over ALL worlds (provenance gate + AUC diagnostic) ===")
    print(f"{'world':34s} {'intend':>6} {'valid':>5} {'match':>6} {'AUC':>6} {'supp':>5} {'reasons'}")
    fails = []
    for w in B8.WORLDS:
        seed = SMOKE_BASE + list(B8.WORLDS).index(w)
        (Z, Y, C, G, Block, Dc, Ct, th, intended), _ = build(w, seed)
        valid, d = B8.check_contract_b8_1(Z, C, G, Block, Dc, Ct, th, seed + 13)
        kind = B8.WORLDS[w][0]
        auc = d["within_block_C_Z_auc"]
        print(f"{w:34s} {str(intended)[:5]:>6} {str(valid)[:5]:>5} {d['provenance_match']:>6.3f} {auc:>6.3f} "
              f"{d['n_support_blocks']:>5} {','.join(d['invalid_reasons']) if d['invalid_reasons'] else '-'}")
        # invariants
        if kind == "contract" and not valid:
            fails.append(f"{w}: CONTRACT world REFUSED (reasons {d['invalid_reasons']})")
        if kind == "violation" and valid:
            fails.append(f"{w}: VIOLATION world VALIDATED (should refuse)")
        # quiet worlds MUST be refused by provenance (schedule-adherence). AUC<=tau is the DESIRED regime (provenance
        # catches what AUC misses) but is seed-variable; the canary quantifies the AUC<=tau fraction, so only WARN here.
        if w.startswith("VIOLATION_quiet"):
            if "assignment_not_following_schedule" not in d["invalid_reasons"]:
                fails.append(f"{w}: QUIET world not refused by schedule-adherence (reasons {d['invalid_reasons']})")
            if (auc == auc) and auc > B8.TAU_CONTRACT_AUC:
                print(f"    [warn] {w}: AUC {auc:.3f} > tau (this cohort caught by AUC too; want a majority <=tau -- canary checks)")

    print("\n=== b8_1_certify on representatives (NB=80) ===")
    reps = ["CONTRACT_NULL_balanced", "CONTRACT_NULL_prior_only", "CONTRACT_NULL_cov_plus_prior",
            "CONTRACT_POS_boundary", "CONTRACT_POS_boundary_plus_prior",
            "VIOLATION_cov_plus_prior", "VIOLATION_quiet_cov_no_concept", "VIOLATION_quiet_cov_plus_concept"]
    for w in reps:
        seed = SMOKE_BASE + 5000 + list(B8.WORLDS).index(w)
        (Z, Y, C, G, Block, Dc, Ct, th, intended), _ = build(w, seed)
        r = B8.b8_1_certify(Z, Y, C, G, Block, Dc, Ct, th, m=M, seed=seed, n_boot=80)
        print(f"{w:34s} -> {r['b8_state']:35s} valid={str(r['contract_valid'])[:5]:>5} "
              f"match={r['provenance_match']:.3f} p_meanT={r['p_exact_meanT']:.3f} p_stud={r['p_exact_stud']:.3f} "
              f"reasons={r['contract_invalid_reasons'] if r['contract_invalid_reasons'] else '-'}")
        if w.startswith("VIOLATION") and r["b8_state"] != "CONTRACT_INVALID_OR_UNIDENTIFIABLE":
            fails.append(f"{w}: certify did NOT refuse (state {r['b8_state']})")

    print("\n=== QUIET-world AUC distribution over 10 seeds (want majority <= tau=0.60 so provenance is the gate) ===")
    for w in ("VIOLATION_quiet_cov_no_concept", "VIOLATION_quiet_cov_plus_concept"):
        aucs, matches, prov = [], [], 0
        for s in range(10):
            seed = SMOKE_BASE + 9000 + s * 101 + list(B8.WORLDS).index(w)
            (Z, Y, C, G, Block, Dc, Ct, th, intended), _ = build(w, seed)
            valid, d = B8.check_contract_b8_1(Z, C, G, Block, Dc, Ct, th, seed + 13)
            aucs.append(d["within_block_C_Z_auc"]); matches.append(d["provenance_match"])
            prov += int("assignment_not_following_schedule" in d["invalid_reasons"])
        aucs = np.array(aucs)
        print(f"    {w:34s} AUC med={np.median(aucs):.3f} <=tau {(aucs <= B8.TAU_CONTRACT_AUC).sum()}/10 | "
              f"match med={np.median(matches):.3f} | provenance-refused {prov}/10")
        if (aucs <= B8.TAU_CONTRACT_AUC).sum() < 6:
            fails.append(f"{w}: only {(aucs <= B8.TAU_CONTRACT_AUC).sum()}/10 cohorts AUC<=tau (want majority; retune QUIET_FLIP_FRAC/axis)")
        if prov < 10:
            fails.append(f"{w}: only {prov}/10 provenance-refused (schedule-adherence must always fire)")

    if fails:
        print("\nSMOKE_FAILURES:")
        for f in fails: print("  " + f)
        sys.exit(1)
    print("\nB8_1_SMOKE_OK")


if __name__ == "__main__":
    main()
