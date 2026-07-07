"""B9 exact randomization null (development-only, NO scientific claim). Resamples C* ONLY within the predeclared
randomization set -- a uniform permutation of C within each (subject, microblock, Y_design) stratum -- and recomputes the
SAME B3 contrast T(Y_design, Z, C*). Because Y_design is a PRE-ASSIGNMENT cue (a pre-treatment CAUSE of Z, not B8.3's
observed common-effect label), this permutation is collider-free CONDITIONAL on genuine within-stratum conditional
randomization of the executed trials -- the state_machine's Z-blind count-balance/prior/adherence checks are
necessary-but-not-sufficient; a Z-dependent, balance-and-prior-preserving differential attrition would re-open a C-Z path
that a Z-blind validator cannot see and must be excluded by the B9.1 acquisition protocol. NO fitted Y|Z, NO fitted C|Z
propensity, NO selection on Z. Fail-closed. The contrast T is byte-reused from csc.mininfo.paired_calibrated."""
import numpy as np
from csc.mininfo import paired_calibrated as PC

ALPHA = 0.025


def resample_C_within_contract(C, Y_design, subject, microblock, rng):
    """Uniform permutation of C within each (subject, microblock, Y_design) stratum = the predeclared randomization set.
    Preserves the per-stratum C x Y_design counts EXACTLY. Y_design pre-assignment -> no collider."""
    C = np.asarray(C); Cstar = C.copy()
    key = {}
    for i in range(len(C)):
        key.setdefault((int(subject[i]), int(microblock[i]), int(Y_design[i])), []).append(i)
    for idx in key.values():
        idx = np.asarray(idx)
        Cstar[idx] = C[idx][rng.permutation(len(idx))]
    return Cstar


def exact_null_test(Z, Y_design, C, subject, microblock, seed=0, rank=3, Creg=0.5, n_folds=PC.N_FOLDS,
                    min_epochs=PC.MIN_EPOCHS_PER_CONDITION, min_confirm_pairs=20, n_boot=200, alpha=ALPHA):
    """Observed T(Y_design, Z, C) + exact within-(subject,microblock,Y_design) randomization null. Returns a dict with
    p_meanT, p_stud, observed_T, estimable, and enough gate fields for the state machine. Fail-closed (degenerate folds /
    infeasible draws are counted, not silently dropped). NO fitted null. Returns state hints only; the state machine decides."""
    Z, Y_design, C, subject, microblock = (np.asarray(Z, float), np.asarray(Y_design), np.asarray(C),
                                           np.asarray(subject), np.asarray(microblock))
    out = dict(observed_T=float("nan"), p_meanT=1.0, p_stud=1.0, exact_null_mean_T=float("nan"),
               exact_null_sd_T=float("nan"), n_exact_invalid=0, n_eligible=0, estimable=False, ran_test=False)
    elig = PC.eligible_complete_pairs(C, subject, min_epochs)
    out["n_eligible"] = int(len(elig))
    if len(elig) < n_folds * 2 or len(np.unique(Y_design)) < 2 or len(np.unique(C)) != 2:
        out["reason"] = "insufficient_for_crossfit"; return out
    m = np.isin(subject, elig)
    Zq, Yq, Cq, Gq, MBq = Z[m], Y_design[m], C[m], subject[m], microblock[m]
    cl = np.array(sorted(np.unique(Yq)))
    folds, _ = PC._make_folds(elig, n_folds, seed)
    prep0 = PC._prep_folds(Zq, Cq, Gq, folds, "centered", rank, Creg)
    if prep0 is None:
        out["reason"] = "fold_prep_degenerate"; return out
    T_obs, ok, deltas = PC._T_cv(prep0, Yq, Cq, Gq, cl, Creg)
    if not ok:
        out["reason"] = "observed_crossfit_degenerate"; return out
    Z_obs = PC._studentize(deltas)["Z"]
    rng = np.random.default_rng(seed + 777)
    ge_t = ge_z = 1; ninv = 0; tstars = []
    for _ in range(n_boot):
        Cstar = resample_C_within_contract(Cq, Yq, Gq, MBq, rng)
        prep = PC._prep_folds(Zq, Cstar, Gq, folds, "centered", rank, Creg)
        if prep is None:
            ninv += 1; ge_t += 1; ge_z += 1; continue
        Ts, oks, ds = PC._T_cv(prep, Yq, Cstar, Gq, cl, Creg)
        if not oks:
            ninv += 1; ge_t += 1; ge_z += 1; continue
        Zs = PC._studentize(ds)["Z"]
        tstars.append(Ts); ge_t += int(Ts >= T_obs); ge_z += int(Zs >= Z_obs)
    estimable = ninv <= 0.20 * n_boot
    out.update(observed_T=float(T_obs), observed_Tz=float(Z_obs), p_meanT=ge_t / (n_boot + 1), p_stud=ge_z / (n_boot + 1),
               exact_null_mean_T=float(np.mean(tstars)) if tstars else float("nan"),
               exact_null_sd_T=float(np.std(tstars)) if tstars else float("nan"), n_exact_invalid=int(ninv),
               estimable=bool(estimable), ran_test=True, n_eligible=int(len(elig)),
               size_ok=bool(len(elig) >= min_confirm_pairs))
    return out
