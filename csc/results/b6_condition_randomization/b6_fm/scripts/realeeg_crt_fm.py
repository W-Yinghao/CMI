"""CSC B6-FM — FIXED-MARGIN / class-preserving condition-randomization null (reviewer-authorized 2026-07-06 after
B6.0 plain-C-null showed the covariate-root fix but a PRIOR-SHIFT estimand gap). Same idea as B6.0 (randomize the
CONDITION C ~ P(C|Z,S), fix Z,Y, recompute the byte-reused B3 T) BUT the resampler is restricted to SAME-CLASS swaps
within subject: swap C between i=(C=hi,Y=y) and j=(C=lo,Y=y) with the SAME class y. This preserves per-subject
condition counts AND per-(subject,condition,class) counts AND the global condition x class margins EXACTLY -> a pure
PRIOR/label shift (a P(Y|C) difference) becomes a HELD-FIXED nuisance and can no longer fire, while a rotated boundary
still changes P(Z|Y,C) so genuine concept still fires. Y enters ONLY as an exact-margin swap constraint -- NEVER in the
propensity P(C|Z,S) and NEVER in the decision score. Propensity + exact-CB acceptance reused from B6.0 realeeg_crt.
NOT a fitted-h0 repair. NO tag, NO validity claim.
"""
import numpy as np
from csc.mininfo import paired_calibrated as PC
import realeeg_crt as CRT                # reuse fit_propensity (P(C|Z,S), no Y) + _bin_entropy

ALPHA_BUDGET = 0.025
N_PCS = 8
MIN_MARGIN_SWAPS = 5.0                   # structural identifiability floor on feasible same-class swaps (disclosed)


def resample_C_fm(D, Y, g, e, lo, hi, rng, n_swaps=None):
    """Count-AND-class-margin-preserving C* via a Metropolis odds-swap chain restricted to SAME-CLASS pairs within
    subject. Swapping i=(hi,y) <-> j=(lo,y) preserves N(subject,condition,class) exactly -> global condition x class
    margins exact. Acceptance min(1, odds_j/odds_i). Y is used ONLY to constrain swaps to the same class."""
    D = np.asarray(D); Y = np.asarray(Y); Dstar = D.copy()
    logodds = np.log(e) - np.log(1 - e)
    subs = np.unique(g)
    sub_idx = {int(s): np.where(g == s)[0] for s in subs}
    nsw = int(n_swaps) if n_swaps else max(20 * len(D), 300)
    for _ in range(nsw):
        s = int(subs[rng.integers(len(subs))]); idx = sub_idx[s]
        cur = Dstar[idx]
        hip = idx[cur == hi]
        if len(hip) == 0:
            continue
        i = hip[rng.integers(len(hip))]; yi = Y[i]
        loj = idx[(cur == lo) & (Y[idx] == yi)]      # same-class lo trials of this subject
        if len(loj) == 0:
            continue
        j = loj[rng.integers(len(loj))]
        if np.log(rng.random() + 1e-300) < (logodds[j] - logodds[i]):
            Dstar[i] = lo; Dstar[j] = hi
    return Dstar


def _margins(D, Y):
    """condition x class count table as a dict {(c,y): n}."""
    D = np.asarray(D); Y = np.asarray(Y)
    return {(int(c), int(y)): int(((D == c) & (Y == y)).sum()) for c in np.unique(D) for y in np.unique(Y)}


def _feasible_swaps(D, Y, g, lo, hi):
    """sum over (subject,class) strata of min(#hi, #lo) = number of possible same-class hi<->lo swaps; plus the
    fraction of strata locked to a single condition (forced trials)."""
    D = np.asarray(D); Y = np.asarray(Y); g = np.asarray(g)
    tot = 0; nstrata = 0; single = 0
    for s in np.unique(g):
        for y in np.unique(Y[g == s]):
            m = (g == s) & (Y == y)
            if m.sum() == 0:
                continue
            nstrata += 1
            nhi = int(((D == hi) & m).sum()); nlo = int(((D == lo) & m).sum())
            tot += min(nhi, nlo)
            if nhi == 0 or nlo == 0:
                single += 1
    return tot, (single / nstrata if nstrata else float("nan")), nstrata


def crt_test_fm(Z, Y, D, groups, m, seed=0, rank=3, C=0.5, n_folds=PC.N_FOLDS,
                min_epochs=PC.MIN_EPOCHS_PER_CONDITION, min_confirm_pairs=20, n_boot=200,
                alpha_budget=ALPHA_BUDGET, n_pcs=N_PCS):
    """Fixed-margin C-randomization test. Replicates the certifier's eligible-pick + folds (same seed) so observed_T
    matches byte-for-byte, then builds the class-preserving C-null. Returns states {CONCEPT_CONFIRMED /
    NO_ACTIONABLE_CONCEPT_EVIDENCE / UNIDENTIFIABLE_MARGIN_LOCK / SAMPLER_INVALID / NEED_MORE_LABELS} + margin
    feasibility / fidelity / covariate-preservation / null-shape diagnostics."""
    Z = np.asarray(Z, float); Y = np.asarray(Y); D = np.asarray(D); G = np.asarray(groups)
    out = dict(fm_valid=False, fm_state="NEED_MORE_LABELS", observed_T_crt=float("nan"),
               p_C_FM_meanT=1.0, p_C_FM_stud=1.0, c_null_mean_T=float("nan"), c_null_sd_T=float("nan"),
               n_crt_invalid=0, propensity_auc=float("nan"), margin_feasible_swaps=float("nan"),
               frac_strata_single_condition=float("nan"), n_strata=0, unique_Cstar=0,
               margin_fidelity_max_err=float("nan"), max_subject_count_err=float("nan"),
               resampled_Cstar_auc=float("nan"), covariate_auc_gap=float("nan"), fm_confirm=False, n_eligible_crt=0)
    elig = PC.eligible_complete_pairs(D, G, min_epochs)
    if len(elig) < n_folds * 2:
        out["reason"] = f"only {len(elig)} eligible pairs"; return out
    rng0 = np.random.default_rng(seed)
    pick = rng0.choice(np.array(sorted(elig)), size=min(int(m), len(elig)), replace=False)
    mask = np.isin(G, pick); Zq, Yq, Dq, Gq = Z[mask], Y[mask], D[mask], G[mask]
    elig_q = PC.eligible_complete_pairs(Dq, Gq, min_epochs)
    if len(elig_q) < n_folds * 2:
        out["reason"] = "queried subset degenerate"; return out
    mq = np.isin(Gq, elig_q); Zq, Yq, Dq, Gq = Zq[mq], Yq[mq], Dq[mq], Gq[mq]
    cl = np.array(sorted(np.unique(Yq)))
    vals = np.array(sorted(np.unique(Dq)))
    if len(cl) < 2 or len(vals) != 2:
        out["reason"] = "need 2 classes and 2 conditions"; return out
    lo, hi = int(vals[0]), int(vals[1])
    folds, fhash = PC._make_folds(elig_q, n_folds, seed)
    prep0 = PC._prep_folds(Zq, Dq, Gq, folds, "centered", rank, C)
    if prep0 is None:
        out["reason"] = "fold prep degenerate"; return out
    T_obs, ok, deltas = PC._T_cv(prep0, Yq, Dq, Gq, cl, C)
    if not ok:
        out["reason"] = "observed cross-fit degenerate"; return out
    Z_obs = PC._studentize(deltas)["Z"]

    e, auc = CRT.fit_propensity(Zq, Dq, Gq, hi, n_pcs=n_pcs, C=CRT.PROP_C, seed=seed + 13)
    feas, frac_single, nstrata = _feasible_swaps(Dq, Yq, Gq, lo, hi)
    obs_margins = _margins(Dq, Yq)

    rng = np.random.default_rng(seed + 777)
    ge_t, ge_z, ninv, tstars, seen = 1, 1, 0, [], set()
    margin_err = 0; count_err = 0; resamp_aucs = []
    for _ in range(n_boot):
        Dstar = resample_C_fm(Dq, Yq, Gq, e, lo, hi, rng)
        # margin fidelity (must be exact by construction)
        st_margins = _margins(Dstar, Yq)
        margin_err = max(margin_err, max(abs(st_margins.get(k, 0) - v) for k, v in obs_margins.items()))
        for s in np.unique(Gq):
            count_err = max(count_err, abs(int((Dstar[Gq == s] == hi).sum()) - int((Dq[Gq == s] == hi).sum())))
        seen.add(Dstar.tobytes())
        # covariate_auc_gap is a MARGINAL-only preservation check (red-team: blind to the within-class covariate
        # that actually drives the NULL_label failure) -> keep only a cheap 3-sample estimate for the record.
        if len(resamp_aucs) < 3:
            try:
                resamp_aucs.append(CRT.fit_propensity(Zq, Dstar, Gq, hi, n_pcs=n_pcs, C=CRT.PROP_C, seed=seed + 5)[1])
            except Exception:
                pass
        prep = PC._prep_folds(Zq, Dstar, Gq, folds, "centered", rank, C)
        if prep is None:
            ninv += 1; ge_t += 1; ge_z += 1; continue
        Ts, oks, ds = PC._T_cv(prep, Yq, Dstar, Gq, cl, C)
        if not oks:
            ninv += 1; ge_t += 1; ge_z += 1; continue
        Zs = PC._studentize(ds)["Z"]
        tstars.append(Ts); ge_t += int(Ts >= T_obs); ge_z += int(Zs >= Z_obs)
    p_t = ge_t / (n_boot + 1); p_z = ge_z / (n_boot + 1)
    estimable = ninv <= 0.20 * n_boot
    cnull_sd = float(np.std(tstars)) if tstars else float("nan")
    resamp_auc = float(np.median(resamp_aucs)) if resamp_aucs else float("nan")
    fm_confirm = bool(estimable and p_t <= alpha_budget and p_z <= alpha_budget)
    size_ok = len(elig_q) >= min_confirm_pairs
    if not estimable or margin_err > 0:
        state = "SAMPLER_INVALID"
    elif feas < MIN_MARGIN_SWAPS or len(seen) < 3:   # margins leave essentially no randomization
        state = "UNIDENTIFIABLE_MARGIN_LOCK"
    elif fm_confirm and size_ok:
        state = "CONCEPT_CONFIRMED"
    elif fm_confirm:
        state = "NEED_MORE_LABELS"
    else:
        state = "NO_ACTIONABLE_CONCEPT_EVIDENCE"
    out.update(fm_valid=bool(estimable), fm_state=state, observed_T_crt=float(T_obs),
               p_C_FM_meanT=float(p_t), p_C_FM_stud=float(p_z),
               c_null_mean_T=float(np.mean(tstars)) if tstars else float("nan"), c_null_sd_T=cnull_sd,
               n_crt_invalid=int(ninv), propensity_auc=auc, margin_feasible_swaps=float(feas),
               frac_strata_single_condition=float(frac_single), n_strata=int(nstrata), unique_Cstar=int(len(seen)),
               margin_fidelity_max_err=float(margin_err), max_subject_count_err=float(count_err),
               resampled_Cstar_auc=resamp_auc, covariate_auc_gap=float(auc - resamp_auc) if resamp_auc == resamp_auc else float("nan"),
               fm_confirm=fm_confirm, n_eligible_crt=int(len(elig_q)), fold_hash=fhash)
    return out
