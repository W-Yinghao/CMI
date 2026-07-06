"""B5.1 SSL-encoder canary merge + POSITIVE-CONTROL-FIRST readout (development-only). The B5.0 red-team set the bar:
before ANY safety reading, POS true-confirm must SEPARATE from NULL_cov false-confirm -- otherwise the certifier is
firing on covariate structure, not concept, and low NULL rates are vacuous. This script LEADS with that test.
Question: does a LEARNED (SSL) representation give the concept-vs-covariate separation the random B5.0 no-op lacked?
Fail-closed. NO tag, NOT confirmatory."""
import os, sys, json, math
import numpy as np
try:
    from scipy.stats import beta, fisher_exact
    def cp_upper(k, n, a=0.05): return 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    HAVE_FISHER = True
except Exception:
    def cp_upper(k, n, a=0.05): return min(1.0, (k + 1.645 * math.sqrt(k + 1)) / n)
    HAVE_FISHER = False

CDIR = "/home/infres/yinwang/realeeg_feas/b5_features/b5_1_ssl_encoder/canary"
CONDS = ["NULL_cov", "NULL_cov_plus_label", "NULL_label", "random_label_control", "POS_concept", "POS_concept_plus_cov"]
NOCONCEPT = {"NULL_cov", "NULL_cov_plus_label", "NULL_label", "random_label_control"}
N = 80


def _read(p): return [json.loads(l) for l in open(p) if l.strip()]
def med(xs):
    xs = [x for x in xs if x is not None and x == x]
    return float(np.median(xs)) if xs else float("nan")


def cohort_boot_ci(flags, B=2000, seed=40_000_000):
    x = np.asarray(flags, float)
    if len(x) == 0: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return (float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)))


def main():
    per = {}
    for c in CONDS:
        p = f"{CDIR}/b5_1_canary_{c}_0.jsonl"
        if not os.path.exists(p):
            print(f"FAIL-CLOSED: missing {p}"); sys.exit(2)
        recs = _read(p)
        if any("__worker_error__" in r for r in recs):
            print(f"FAIL-CLOSED: worker-error rows in {c}"); sys.exit(2)
        if len(recs) != N:
            print(f"FAIL-CLOSED: {c} has {len(recs)} != {N}"); sys.exit(2)
        per[c] = recs

    rows = {}
    print(f"\n{'condition':24s} {'GT':>9} {'confirm':>8} {'rate':>7} {'T_z_med':>8} {'ffp_floor':>9} {'auc':>6}")
    print("-" * 80)
    for c in CONDS:
        recs = per[c]; gt = "NO_CONC" if c in NOCONCEPT else "CONCEPT"
        conf = [1 if (r.get("false_confirm") or r.get("true_confirm")) else 0 for r in recs]
        k = sum(conf)
        tz = med([r.get("T_z") for r in recs])
        ffp = [r.get("fixed_margin_p") for r in recs if r.get("fixed_margin_p") is not None]
        ffp_floor = float(np.mean([1.0 if (x is not None and x <= 0.005 + 1e-9) else 0.0 for x in ffp])) if ffp else float("nan")
        auc = med([r.get("session_auc") for r in recs])
        rows[c] = dict(n=N, confirm=k, rate=round(k / N, 4), T_z_med=tz, ffp_floor=ffp_floor,
                       session_auc_med=auc, confirm_flags=conf)
        print(f"{c:24s} {gt:>9} {k:>8} {k/N:>7.3f} {tz:>8.2f} {ffp_floor:>9.2f} {auc:>6.3f}")

    # ---- POSITIVE-CONTROL-FIRST: does POS separate from NULL_cov? ----
    kp, kn = rows["POS_concept"]["confirm"], rows["NULL_cov"]["confirm"]
    sep_p = float("nan"); sep = None
    if HAVE_FISHER:
        # one-sided: POS confirm rate > NULL_cov confirm rate
        odds, sep_p = fisher_exact([[kp, N - kp], [kn, N - kn]], alternative="greater")
        sep = bool(sep_p < 0.05)
    print("\n=== POSITIVE-CONTROL-FIRST (the gate the B5.0 red-team required) ===")
    print(f"  POS_concept confirm {kp}/{N} vs NULL_cov false-confirm {kn}/{N}")
    print(f"  Fisher one-sided (POS>NULL) p={sep_p:.4g} -> POS SEPARATES from NULL: {sep}")
    if sep:
        print("  -> LEARNED features give concept-vs-covariate separation the random B5.0 no-op LACKED (interesting).")
    else:
        print("  -> NO separation: like B5.0, the SSL encoder does not yield decision-level concept power over the")
        print("     covariate-null baseline -> the canary is INCONCLUSIVE for safety (a LEARNED-encoder negative).")

    # ---- type-I bound on NULL_cov (cohort bootstrap; subject-overlap caveat) ----
    lo, hi = cohort_boot_ci(rows["NULL_cov"]["confirm_flags"])
    cp = cp_upper(kn, N)
    print(f"\n  NULL_cov false-confirm {kn}/{N}: cohort-boot 95% CI [{lo:.3f},{hi:.3f}], CP95u {cp:.4f}")
    print("  CAVEAT: cohorts share subjects (30 of 54) -> naive/cohort CIs anti-conservative; a subject-cluster")
    print("  bound is required before ANY 'safe' claim. Only meaningful IF POS separates (else vacuous).")

    tables = dict(scope="B5.1 SSL-encoder canary; development-only; NOT confirmatory; NO tag", base_seed=40_000_000,
                  n_per_condition=N, per_condition={c: {k: v for k, v in rows[c].items() if k != "confirm_flags"} for c in CONDS},
                  positive_control_first=dict(pos_concept_confirm=kp, null_cov_false_confirm=kn,
                      fisher_one_sided_p=sep_p, pos_separates_from_null=sep,
                      interpretation=("learned features give concept-vs-covariate separation (interesting)" if sep
                                      else "no decision-level separation -> inconclusive for safety (learned-encoder negative)")),
                  null_cov_type_I=dict(confirm=kn, cohort_boot_ci=[lo, hi], cp95u=cp,
                      caveat="cohorts share subjects; needs subject-cluster bound for a safe claim; only meaningful if POS separates"),
                  b5_1_cache_sha256=(per["NULL_cov"][0].get("b5_cache_sha256")))
    json.dump(tables, open(f"{CDIR}/b5_1_canary_tables.json", "w"), indent=1, default=str)
    print(f"\n  saved {CDIR}/b5_1_canary_tables.json")


if __name__ == "__main__":
    main()
