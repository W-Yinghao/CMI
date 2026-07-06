"""CSC B8.0 information-contract / randomized paired audit (development-only; reviewer-authorized 2026-07-06 after B7.1
closed the observational null-repair line). CORE INVERSION: do NOT estimate the null on observational data; REQUIRE a
KNOWN randomization contract (condition C assigned by balanced randomization WITHIN subject x block, so C _|_ Z within
block by design) -> use an EXACT block-stratified randomization null (resample C* from the DECLARED within-block
randomization, NOT a fitted propensity); REFUSE (CONTRACT_INVALID_OR_UNIDENTIFIABLE) when the contract is not met. The
B3 contrast T is byte-reused from the certifier; only the null source changes (known randomization vs fitted). NO tag.

EMULATOR uses Lee2019 SM16 geometry as a realistic Z/subject substrate -- this is NOT validation (Lee2019 sessions are
not a real randomized audit). Two world classes: CONTRACT-SATISFYING (C balanced-randomized within pseudo-block) and
CONTRACT-VIOLATING (C confounded with Z, the B7.1 failure structures). Goal in violation worlds = correctly REFUSE.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from csc.mininfo import paired_calibrated as PC
from csc.mininfo import realeeg_engine as EG

ALPHA_BUDGET = 0.025
TAU_CONTRACT_AUC = 0.60      # DECLARED: C must be ~unpredictable from Z within block (block-adjusted AUC<=0.60) for a valid contract
MIN_SUPPORT_BLOCKS = 8       # DECLARED: >= this many blocks with BOTH conditions for adequate randomization support
BLOCK_SIZE = 8
LO, HI = 1, 2
PRIOR_SHIFT = 0.62           # DECLARED per-condition logit shift (~ the 0.35/0.65 prior used throughout B3-B7; consistency, not tuning)

# world -> (class, has_concept). CONTRACT_* satisfy the randomization contract (C balanced-randomized within block);
# VIOLATION_* confound C with Z (validator should REFUSE). has_concept marks worlds with a genuine C x Z boundary.
WORLDS = {
    "CONTRACT_NULL_balanced":           ("contract", False),
    "CONTRACT_NULL_cov_balanced":       ("contract", False),
    "CONTRACT_NULL_prior_only":         ("contract", False),
    "CONTRACT_NULL_cov_plus_prior":     ("contract", False),   # NEW: the B7.1-killer analogue UNDER a valid contract
    "CONTRACT_random_label":            ("contract", False),
    "CONTRACT_POS_boundary":            ("contract", True),
    "CONTRACT_POS_boundary_plus_prior": ("contract", True),    # FIXED: prior now logit-shift (concept preserved)
    "VIOLATION_cov_confound":           ("violation", False),  # within-block C = median split on leading PC off boundary
    "VIOLATION_cov_plus_prior":         ("violation", False),  # confound + prior (the B7.1 killer, observational)
    "VIOLATION_no_within_block_rand":   ("violation", False),  # C = whole-block (session-like) -> support fail
    "VIOLATION_borderline_confound":    ("violation", False),  # mild confound near tau=0.60 -> stress the threshold
    "VIOLATION_lowvar_confound":        ("violation", False),  # confound on a LOW-variance axis -> validator blind spot (documented)
}
NOCONCEPT = {w for w, (_, hc) in WORLDS.items() if not hc}


def _apply_prior(p, C, shift=PRIOR_SHIFT):
    """Prior/label nuisance as a per-condition logit MAIN EFFECT (P(Y|C) shifts) that PRESERVES the Z-dependence of p
    (so a concept C x Z interaction survives). Replaces the old label-reassignment which severed Y from Z."""
    lg = np.log(np.clip(p, 1e-6, 1 - 1e-6)) - np.log(np.clip(1 - p, 1e-6, 1 - 1e-6))
    lg = lg + np.where(np.asarray(C) == HI, shift, -shift)
    return 1.0 / (1.0 + np.exp(-lg))


def _blocks(subj, rng, block_size=BLOCK_SIZE):
    """Per-subject pseudo-blocks (shuffle within subject, chunk). Blocks are the randomization/matching unit."""
    Block = np.full(len(subj), -1, int); bid = 0
    for s in np.unique(subj):
        idx = np.where(subj == s)[0]; idx = idx[rng.permutation(len(idx))]
        for i in range(0, len(idx), block_size):
            Block[idx[i:i + block_size]] = bid; bid += 1
    return Block


def _balanced_within_block(Block, rng):
    """Assign C in {LO,HI} balanced within each block (C _|_ Z within block by design)."""
    C = np.full(len(Block), LO, int)
    for b in np.unique(Block):
        bi = np.where(Block == b)[0]; k = len(bi) // 2
        if k > 0:
            C[bi[rng.permutation(len(bi))[:k]]] = HI
    return C


def _label_free_axis(Z, w, k=0):
    """k-th singular direction of centered Z, orthogonalized against the boundary normal w (label-free)."""
    w_hat = w / (np.linalg.norm(w) + 1e-12)
    Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)[2]
    v = Vt[min(k, Vt.shape[0] - 1)].astype(float)
    v = v - (v @ w_hat) * w_hat
    return v / (np.linalg.norm(v) + 1e-12)


def build_b8_cohort(world, coh, rng, block_size=BLOCK_SIZE):
    """Return (Z, Y, C, G, Block, contract_intended). C is the CONDITION; G subject; Block the randomization unit."""
    Z, y0, subj = np.asarray(coh["Z"], float), np.asarray(coh["y"]), np.asarray(coh["subject"])
    G = subj.copy(); Block = _blocks(subj, rng, block_size)
    clf = EG._pooled_clf(Z, y0); w = clf.coef_.ravel().astype(float)
    p0 = clf.predict_proba(Z)[:, 1]
    kind, _ = WORLDS[world]

    if kind == "contract":                                   # C balanced-randomized WITHIN block -> C _|_ Z within block
        C = _balanced_within_block(Block, rng)
        if world == "CONTRACT_random_label":
            Y = rng.permutation(y0)
        elif world in ("CONTRACT_NULL_cov_balanced", "CONTRACT_NULL_cov_plus_prior"):
            v = _label_free_axis(Z, w)                         # covariate present but drawn INDEPENDENTLY of C -> balanced
            half = rng.permutation(len(Z))[:len(Z) // 2]
            Zc = Z.copy(); Zc[half] += 2.0 * float(np.std(Z @ v)) * v
            p = EG._pooled_clf(Zc, y0).predict_proba(Zc)[:, 1]
            Y = EG._draw(_apply_prior(p, C) if world.endswith("plus_prior") else p, rng)
        elif world in ("CONTRACT_POS_boundary", "CONTRACT_POS_boundary_plus_prior"):
            p2 = EG._rotate_proba(clf, Z, rng, deg=25.0)       # concept: rotate boundary in the C=HI trials (C x Z interaction)
            p = np.where(C == HI, p2, p0)
            Y = EG._draw(_apply_prior(p, C) if world.endswith("plus_prior") else p, rng)
        else:                                                # CONTRACT_NULL_balanced / CONTRACT_NULL_prior_only
            Y = EG._draw(_apply_prior(p0, C) if world == "CONTRACT_NULL_prior_only" else p0, rng)
        return Z, Y, C, G, Block, True

    # VIOLATION: C confounded with Z (NOT a valid randomization) -> the contract MUST be REFUSED
    if world == "VIOLATION_no_within_block_rand":            # C assigned per WHOLE BLOCK (session-like) -> 0 within-block support
        C = np.full(len(Z), LO, int); blks = np.unique(Block)
        C[np.isin(Block, blks[rng.permutation(len(blks))[:len(blks) // 2]])] = HI
        return Z, EG._draw(p0, rng), C, G, Block, False
    v = _label_free_axis(Z, w, k=(Z.shape[1] - 1) if world == "VIOLATION_lowvar_confound" else 0)  # last PC = low-variance blind spot
    proj = Z @ v
    if world == "VIOLATION_borderline_confound":            # mild within-block confound near tau (heavy noise)
        score = proj + 2.2 * float(np.std(proj)) * rng.standard_normal(len(proj))
        C = np.where(score > np.median(score), HI, LO); Y = EG._draw(p0, rng)
    elif world == "VIOLATION_cov_plus_prior":              # confound + prior (the B7.1 killer, observational)
        C = np.where(proj > np.median(proj), HI, LO); Y = EG._draw(_apply_prior(p0, C), rng)
    else:                                                   # VIOLATION_cov_confound / VIOLATION_lowvar_confound
        C = np.where(proj > np.median(proj), HI, LO); Y = EG._draw(p0, rng)
    return Z, Y, C, G, Block, False


def check_contract(Z, C, G, Block, seed, tau_auc=TAU_CONTRACT_AUC, min_support=MIN_SUPPORT_BLOCKS, n_pcs=8):
    """DATA-DRIVEN contract check (predeclared, NOT from the test result): (1) block-adjusted C~Z cross-fit AUC must be
    <= tau_auc (C unpredictable from Z given block -> randomized/balanced); (2) >= min_support blocks with BOTH
    conditions (randomization support). Returns (valid, diagnostics)."""
    C = np.asarray(C); b = (C == HI).astype(int)
    Zc = Z - Z.mean(0)
    Vt = np.linalg.svd(Zc, full_matrices=False)[2]
    F = Zc @ Vt[:min(n_pcs, Vt.shape[0])].T
    # block-adjusted features: append per-block mean-centering (removes block main effect so we test WITHIN-block C~Z)
    Fa = F.copy()
    for bl in np.unique(Block):
        m = Block == bl
        if m.sum() > 1: Fa[m] -= Fa[m].mean(0)
    e = np.full(len(b), b.mean(), float)
    if len(np.unique(b)) == 2:
        kf = KFold(n_splits=min(5, len(b)), shuffle=True, random_state=seed)
        for tr, te in kf.split(Fa):
            if len(np.unique(b[tr])) < 2: e[te] = b[tr].mean(); continue
            e[te] = LogisticRegression(C=1.0, max_iter=1000).fit(Fa[tr], b[tr]).predict_proba(Fa[te])[:, 1]
    auc = float(roc_auc_score(b, e)) if len(np.unique(b)) == 2 else float("nan")
    support = sum(1 for bl in np.unique(Block)
                  if len(np.unique(C[Block == bl])) >= 2 and (C[Block == bl] == HI).sum() >= 1 and (C[Block == bl] == LO).sum() >= 1)
    valid = (auc == auc) and (auc <= tau_auc) and (support >= min_support)
    return bool(valid), dict(within_block_C_Z_auc=auc, n_support_blocks=int(support),
                             n_blocks=int(len(np.unique(Block))), tau_auc=tau_auc, min_support=int(min_support))


def resample_C_exact(C, Y, G, Block, rng, fix_class_margins=True):
    """EXACT within-(subject,block[,class]) balanced permutation of C from the KNOWN randomization (NOT a fitted
    propensity). Preserves per-block condition counts (and per-(block,class) counts if fix_class_margins) exactly."""
    C = np.asarray(C); Y = np.asarray(Y); Cstar = C.copy()
    strata = {}
    for i in range(len(C)):
        key = (int(Block[i]), int(Y[i])) if fix_class_margins else (int(Block[i]),)
        strata.setdefault(key, []).append(i)
    for idx in strata.values():
        idx = np.asarray(idx)
        vals = C[idx]                                        # preserve the multiset of conditions in this stratum
        Cstar[idx] = vals[rng.permutation(len(idx))]         # uniform permutation = exact randomization within stratum
    return Cstar


def b8_certify(Z, Y, C, G, Block, m, seed=0, rank=3, Creg=0.5, n_folds=PC.N_FOLDS,
               min_epochs=PC.MIN_EPOCHS_PER_CONDITION, min_confirm_pairs=20, n_boot=200, alpha_budget=ALPHA_BUDGET):
    """Contract-first: if the randomization contract is not satisfied -> CONTRACT_INVALID_OR_UNIDENTIFIABLE (refuse).
    Else compute observed_T and the EXACT block-stratified randomization null; ALERT if both p-gates pass + size."""
    Z, Y, C, G, Block = np.asarray(Z, float), np.asarray(Y), np.asarray(C), np.asarray(G), np.asarray(Block)
    out = dict(b8_state="NEED_MORE_LABELS", contract_valid=False, observed_T=float("nan"),
               p_exact_meanT=1.0, p_exact_stud=1.0, exact_null_mean_T=float("nan"), exact_null_sd_T=float("nan"),
               n_exact_invalid=0, n_eligible=0)
    elig = PC.eligible_complete_pairs(C, G, min_epochs)
    if len(elig) < n_folds * 2:
        out.update(b8_state="INSUFFICIENT_LABELS", reason=f"{len(elig)} eligible"); return out
    rng0 = np.random.default_rng(seed)
    pick = rng0.choice(np.array(sorted(elig)), size=min(int(m), len(elig)), replace=False)
    mask = np.isin(G, pick); Zq, Yq, Cq, Gq, Bq = Z[mask], Y[mask], C[mask], G[mask], Block[mask]
    elig_q = PC.eligible_complete_pairs(Cq, Gq, min_epochs)
    if len(elig_q) < n_folds * 2:
        out.update(b8_state="INSUFFICIENT_LABELS", reason="queried degenerate"); return out
    mq = np.isin(Gq, elig_q); Zq, Yq, Cq, Gq, Bq = Zq[mq], Yq[mq], Cq[mq], Gq[mq], Bq[mq]
    # CONTRACT CHECK (predeclared, before the test)
    valid, cdiag = check_contract(Zq, Cq, Gq, Bq, seed + 13)
    out.update(contract_valid=bool(valid), n_eligible=int(len(elig_q)), **{f"contract_{k}": v for k, v in cdiag.items()})
    if not valid:
        out["b8_state"] = "CONTRACT_INVALID_OR_UNIDENTIFIABLE"; return out
    cl = np.array(sorted(np.unique(Yq)))
    if len(cl) < 2 or len(np.unique(Cq)) != 2:
        out.update(b8_state="INSUFFICIENT_LABELS", reason="need 2 classes+conditions"); return out
    folds, _ = PC._make_folds(elig_q, n_folds, seed)
    prep0 = PC._prep_folds(Zq, Cq, Gq, folds, "centered", rank, Creg)
    if prep0 is None:
        out.update(b8_state="SAMPLER_INVALID", reason="fold prep degenerate"); return out
    T_obs, ok, deltas = PC._T_cv(prep0, Yq, Cq, Gq, cl, Creg)
    if not ok:
        out.update(b8_state="SAMPLER_INVALID", reason="observed cross-fit degenerate"); return out
    Z_obs = PC._studentize(deltas)["Z"]
    rng = np.random.default_rng(seed + 777)
    ge_t, ge_z, ninv, tstars = 1, 1, 0, []
    for _ in range(n_boot):
        Cstar = resample_C_exact(Cq, Yq, Gq, Bq, rng, fix_class_margins=True)   # EXACT block-stratified randomization
        prep = PC._prep_folds(Zq, Cstar, Gq, folds, "centered", rank, Creg)
        if prep is None: ninv += 1; ge_t += 1; ge_z += 1; continue
        Ts, oks, ds = PC._T_cv(prep, Yq, Cstar, Gq, cl, Creg)
        if not oks: ninv += 1; ge_t += 1; ge_z += 1; continue
        Zs = PC._studentize(ds)["Z"]
        tstars.append(Ts); ge_t += int(Ts >= T_obs); ge_z += int(Zs >= Z_obs)
    p_t, p_z = ge_t / (n_boot + 1), ge_z / (n_boot + 1)
    estimable = ninv <= 0.20 * n_boot
    alert = bool(estimable and p_t <= alpha_budget and p_z <= alpha_budget and len(elig_q) >= min_confirm_pairs)
    if not estimable:
        state = "SAMPLER_INVALID"
    elif alert:
        state = "B8_CONCEPT_ALERT"
    else:
        state = "NO_ACTIONABLE_CONCEPT_EVIDENCE"
    out.update(b8_state=state, observed_T=float(T_obs), p_exact_meanT=float(p_t), p_exact_stud=float(p_z),
               exact_null_mean_T=float(np.mean(tstars)) if tstars else float("nan"),
               exact_null_sd_T=float(np.std(tstars)) if tstars else float("nan"), n_exact_invalid=int(ninv))
    return out
