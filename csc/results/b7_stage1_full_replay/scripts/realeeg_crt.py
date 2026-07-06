"""CSC B6.0 — condition-randomization / covariate-process null (development-only; reviewer-authorized 2026-07-06).
Tests H0: Y ⊥ C | Z,S by resampling the CONDITION C from an estimated propensity P(C|Z,S) (count-preserving within
subject), NOT by resampling the label Y. The B3 cross-fit contrast T is BYTE-REUSED from the certifier
(paired_calibrated._prep_folds + _T_cv + _make_folds + eligible_complete_pairs + _studentize); ONLY the null generator
changes (C-resampling instead of the fixed-margin Y-resampling). observed_T is verified identical to the certifier's.

Propensity (v1, fixed + auditable): P(C = high | subject-centered Z PCs) via cross-fit OOF L2 logistic. It NEVER sees
Y, synthetic labels, or oracle fields. Resampler: within each subject, keep the observed condition count and draw C*
from the EXACT conditional-Bernoulli law implied by the propensity odds, via a Metropolis odds-swap chain that mirrors
the certifier's own fixed-margin sampler -> a conditional randomization that (i) reduces to a within-subject
permutation null when the covariate is uninformative (propensity flat), and (ii) freezes toward the observed C when
the covariate LOCKS the condition (propensity ~deterministic) -> the C-null then puts observed_T in its own bulk and
the test abstains. NOT a fitted-h0 label-null (that line = B4, CLOSED). NO tag.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from csc.mininfo import paired_calibrated as PC

ALPHA_BUDGET = 0.025           # matches the certifier's cross-budget alpha (comparable confirm rule)
N_PCS = 8
PROP_C = 1.0
MIN_EFF_RANDOMIZATION = 5.0    # structural identifiability floor (effective free within-subject swaps); disclosed


def _subject_center(Z, g):
    Zc = Z.copy().astype(float)
    for s in np.unique(g):
        m = g == s
        Zc[m] -= Zc[m].mean(0)
    return Zc


def fit_propensity(Z, D, g, hi, n_pcs=N_PCS, C=PROP_C, seed=0, n_folds=5):
    """OOF propensity e_i = P(D_i == hi | subject-centered Z PCs, S). NO Y. Returns (e, auc)."""
    b = (np.asarray(D) == hi).astype(int)
    Zc = _subject_center(Z, g)
    # top-n_pcs directions of the subject-centered features (label-free)
    U, S, Vt = np.linalg.svd(Zc - Zc.mean(0), full_matrices=False)
    F = Zc @ Vt[:min(n_pcs, Vt.shape[0])].T
    e = np.full(len(b), b.mean(), float)
    if len(np.unique(b)) == 2:
        kf = KFold(n_splits=min(n_folds, len(b)), shuffle=True, random_state=seed)
        for tr, te in kf.split(F):
            if len(np.unique(b[tr])) < 2:
                e[te] = b[tr].mean(); continue
            clf = LogisticRegression(C=C, max_iter=1000).fit(F[tr], b[tr])
            e[te] = clf.predict_proba(F[te])[:, 1]
    e = np.clip(e, 1e-6, 1 - 1e-6)
    auc = float(roc_auc_score(b, e)) if len(np.unique(b)) == 2 else float("nan")
    return e, auc


def _bin_entropy(e):
    e = np.clip(e, 1e-9, 1 - 1e-9)
    return -(e * np.log(e) + (1 - e) * np.log(1 - e)) / np.log(2.0)   # in [0,1] bits


def resample_C(D, g, e, lo, hi, rng, n_swaps=None):
    """Count-preserving within-subject C* drawn from the EXACT conditional-Bernoulli law implied by the propensity
    (P(config) ∝ prod_{i: C_i=hi} odds_i, at fixed per-subject count) via a Metropolis odds-swap chain that MIRRORS
    the certifier's fixed-margin sampler (sample_h0_fixed_condition_margins): propose swapping a hi-trial i and a
    lo-trial j WITHIN a subject, accept with prob min(1, odds_j/odds_i) = exp(logodds_j - logodds_i). Stationary law
    is exactly CB with fixed count. Flat propensity -> uniform count-preserving permutation; near-deterministic
    propensity -> acceptances vanish -> C* freezes toward observed C (correct lock behaviour). (Fixes the earlier
    Gumbel-top-k, which sampled Plackett-Luce, not CB, and diverged for informative propensity.)"""
    D = np.asarray(D); Dstar = D.copy()          # start from observed C (mirrors the certifier's null start)
    logodds = np.log(e) - np.log(1 - e)
    subs = np.unique(g)
    sub_idx = {int(s): np.where(g == s)[0] for s in subs}
    nsw = int(n_swaps) if n_swaps else max(20 * len(D), 300)   # same convention as the fixed-margin sampler
    for _ in range(nsw):
        s = int(subs[rng.integers(len(subs))]); idx = sub_idx[s]
        cur = Dstar[idx]
        hip = idx[cur == hi]; lop = idx[cur == lo]
        if len(hip) == 0 or len(lop) == 0:
            continue
        i = hip[rng.integers(len(hip))]; j = lop[rng.integers(len(lop))]
        if np.log(rng.random() + 1e-300) < (logodds[j] - logodds[i]):   # CB acceptance: swap hi<->lo
            Dstar[i] = lo; Dstar[j] = hi
    return Dstar


def crt_test(Z, Y, D, groups, m, seed=0, rank=3, C=0.5, n_folds=PC.N_FOLDS,
             min_epochs=PC.MIN_EPOCHS_PER_CONDITION, min_confirm_pairs=20, n_boot=200,
             alpha_budget=ALPHA_BUDGET, n_pcs=N_PCS):
    """C-randomization test. Replicates the certifier's eligible-pick + folds EXACTLY (same seed) so observed_T
    matches certify_paired_calibrated byte-for-byte, then builds the C-null instead of the Y-null."""
    Z = np.asarray(Z, float); Y = np.asarray(Y); D = np.asarray(D); G = np.asarray(groups)
    out = dict(b6_valid=False, b6_state="NEED_MORE_LABELS", observed_T_crt=float("nan"),
               p_C_meanT=1.0, p_C_stud=1.0, c_null_mean_T=float("nan"), c_null_sd_T=float("nan"),
               n_crt_invalid=0, propensity_auc=float("nan"), propensity_mean_entropy=float("nan"),
               eff_randomization=float("nan"), frac_condition_locked=float("nan"), b6_confirm=False,
               n_eligible_crt=0)
    elig = PC.eligible_complete_pairs(D, G, min_epochs)
    if len(elig) < n_folds * 2:
        out["reason"] = f"only {len(elig)} eligible pairs"; return out
    # SAME eligible-pick as certify_paired_calibrated (same seed) -> same queried subset
    rng0 = np.random.default_rng(seed)
    pick = rng0.choice(np.array(sorted(elig)), size=min(int(m), len(elig)), replace=False)
    mask = np.isin(G, pick); Zq, Yq, Dq, Gq = Z[mask], Y[mask], D[mask], G[mask]
    elig_q = PC.eligible_complete_pairs(Dq, Gq, min_epochs)
    if len(elig_q) < n_folds * 2:
        out["reason"] = "queried subset degenerate"; return out
    mq = np.isin(Gq, elig_q); Zq, Yq, Dq, Gq = Zq[mq], Yq[mq], Dq[mq], Gq[mq]
    cl = np.array(sorted(np.unique(Yq)))
    if len(cl) < 2:
        out["reason"] = "single class"; return out
    vals = np.array(sorted(np.unique(Dq)))
    if len(vals) != 2:
        out["reason"] = f"need 2 conditions, got {len(vals)}"; return out
    lo, hi = int(vals[0]), int(vals[1])
    folds, fhash = PC._make_folds(elig_q, n_folds, seed)
    prep0 = PC._prep_folds(Zq, Dq, Gq, folds, "centered", rank, C)
    if prep0 is None:
        out["reason"] = "fold prep degenerate"; return out
    T_obs, ok, deltas = PC._T_cv(prep0, Yq, Dq, Gq, cl, C)
    if not ok:
        out["reason"] = "observed cross-fit degenerate"; return out
    Z_obs = PC._studentize(deltas)["Z"]

    # propensity P(C|Z,S) on the queried subset -- NO Y
    e, auc = fit_propensity(Zq, Dq, Gq, hi, n_pcs=n_pcs, C=PROP_C, seed=seed + 13)
    ent = _bin_entropy(e)
    # per-subject effective randomization = sum of within-subject entropies (bits ~ effective free swaps)
    eff = 0.0; locked = 0; nsub = len(np.unique(Gq))
    for s in np.unique(Gq):
        sm = Gq == s
        eff += float(ent[sm].sum())
        if ent[sm].mean() < 0.1:   # near-deterministic within subject
            locked += 1
    frac_locked = locked / nsub if nsub else float("nan")

    # C-null: resample D* (count-preserving, propensity-weighted), rebuild prep, recompute T*
    rng = np.random.default_rng(seed + 777)
    ge_t, ge_z, ninv, tstars = 1, 1, 0, []
    for _ in range(n_boot):
        Dstar = resample_C(Dq, Gq, e, lo, hi, rng)
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
    b6_confirm = bool(estimable and p_t <= alpha_budget and p_z <= alpha_budget)
    size_ok = len(elig_q) >= min_confirm_pairs
    if not estimable:
        state = "NEED_MORE_LABELS"
    elif eff < MIN_EFF_RANDOMIZATION:
        state = "UNIDENTIFIABLE_DUE_TO_COVARIATE_LOCK"          # structural identifiability (disclosed floor)
    elif b6_confirm and size_ok:
        state = "CONCEPT_CONFIRMED_B6"
    elif b6_confirm:
        state = "NEED_MORE_LABELS"
    else:
        state = "NO_CONCEPT_EVIDENCE_B6"
    out.update(b6_valid=bool(estimable), b6_state=state, observed_T_crt=float(T_obs),
               p_C_meanT=float(p_t), p_C_stud=float(p_z),
               c_null_mean_T=float(np.mean(tstars)) if tstars else float("nan"),
               c_null_sd_T=float(np.std(tstars)) if tstars else float("nan"), n_crt_invalid=int(ninv),
               propensity_auc=auc, propensity_mean_entropy=float(ent.mean()),
               eff_randomization=float(eff), frac_condition_locked=float(frac_locked),
               b6_confirm=b6_confirm,
               b6_confirm_ignoring_lock=bool(estimable and b6_confirm and size_ok),  # what confirm would be w/o the lock gate
               n_eligible_crt=int(len(elig_q)), fold_hash=fhash)
    return out
