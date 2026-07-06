"""CSC B8.1 class-balanced randomized-audit contract (development-only; reviewer-authorized 2026-07-06 after B8.0
MIXED). SINGLE change vs B8.0 that fixes the two disclosed B8.0 limitations:

  B8.0 stratified the EXACT null by the OBSERVED audit label Y (realeeg_b8.resample_C_exact key=(Block,Y)). Y depends
  on C (prior shift) AND on Z, so Y is a POST-TREATMENT COLLIDER -> conditioning on it left residual within-(block,Y)
  C-Z dependence -> prior_only over-alert 5/50.

  B8.1 makes class/prior balance PART OF THE RANDOMIZATION CONTRACT, not post-hoc analysis conditioning:
  (1) design_class Dc = the REAL cued class y0 -- a PRE-ASSIGNMENT / pre-outcome design variable (the MI cue precedes
      any synthetic condition assignment; Dc _|_ C by the randomization). NOT the observed/generated post-hoc Y.
  (2) condition C is balanced-randomized WITHIN (subject x block x Dc) strata -- class-balanced randomization.
  (3) the EXACT null resamples C* WITHIN (block, Dc) strata (a PRE-ASSIGNMENT stratifier -> NO collider).
  (4) a HARD PROVENANCE GATE: a randomization TABLE is generated BEFORE any label/concept injection + hash-pinned; the
      certifier verifies (a) table integrity (hash), (b) the executed C FOLLOWS the registered schedule (adherence),
      (c) within-(block,Dc) support, (d) balance. The block-adjusted C~Z AUC is a SECONDARY fail-closed DIAGNOSTIC,
      NOT the primary gate -- so a QUIET confound that slips tau=0.60 is still REFUSED by provenance (closes the B8.0
      stress-design gap). Contract-first: an invalid contract REFUSES before T is ever computed -- so even a genuine
      concept under an invalid contract (quiet_cov_plus_concept) is REFUSED, never alerted.

The B3 contrast T and every certifier gate (studentized, estimability, min_confirm_pairs) are BYTE-UNCHANGED from B8.0
-- NO statistic change, NO feature/montage change, NO p-value retuning, NO B7/router/oracle field. EMULATOR uses
Lee2019 SM16 geometry -- NOT validation. NO tag.
"""
import hashlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from csc.mininfo import paired_calibrated as PC
from csc.mininfo import realeeg_engine as EG

ALPHA_BUDGET = 0.025
TAU_CONTRACT_AUC = 0.60      # SECONDARY diagnostic (block-adjusted C~Z); NOT the primary gate (provenance is)
MIN_SUPPORT_BLOCKS = 8       # >= this many (block,Dc) strata with BOTH conditions (randomization support)
BLOCK_SIZE = 8
LO, HI = 1, 2
PRIOR_SHIFT = 0.62           # per-condition logit shift (~0.35/0.65 prior used throughout B3-B8.0; consistency)
BALANCE_TOL = 1              # per-(block,Dc) |n_HI - n_LO| <= this for a valid balanced schedule (odd strata -> 1)
QUIET_FLIP_FRAC = 0.12       # fraction of the registered schedule flipped toward Z in the QUIET-confound violations

# world -> (class, has_concept). CONTRACT_* satisfy the class-balanced randomization contract (C balanced-randomized
# within (block,Dc); executed C == registered schedule). VIOLATION_* deviate from the registered schedule (provenance
# must REFUSE). has_concept marks a genuine C x Z boundary -- note quiet_cov_plus_concept has a REAL concept yet is a
# VIOLATION: the correct behaviour is REFUSE (contract-first), NOT alert.
WORLDS = {
    "CONTRACT_NULL_balanced":           ("contract", False),
    "CONTRACT_NULL_prior_only":         ("contract", False),   # PRIMARY B8.1 target: was 5/50 in B8.0 (collider)
    "CONTRACT_NULL_cov_plus_prior":     ("contract", False),   # must STAY controlled (B8.0 was 1/50)
    "CONTRACT_random_label":            ("contract", False),
    "CONTRACT_POS_boundary":            ("contract", True),    # power (must survive the design_class-stratified null)
    "CONTRACT_POS_boundary_plus_prior": ("contract", True),
    "VIOLATION_cov_session":            ("violation", False),  # strong C=f(Z) confound (whole cohort) -> REFUSE
    "VIOLATION_prior_shift":            ("violation", False),  # C tracks the label axis p0(Z) + Y prior -> REFUSE
    "VIOLATION_cov_plus_prior":         ("violation", False),  # the B7.1 killer (confound + prior) -> REFUSE
    "VIOLATION_condition_lock":         ("violation", False),  # C locked within (block,Dc) -> no support -> REFUSE
    "VIOLATION_quiet_cov_no_concept":   ("violation", False),  # QUIET deviation (low AUC), no concept -> REFUSE via provenance
    "VIOLATION_quiet_cov_plus_concept": ("violation", True),   # QUIET deviation + REAL concept -> REFUSE via provenance (stress cell)
}
NOCONCEPT = {w for w, (_, hc) in WORLDS.items() if not hc}
INVALID_REASONS = ("missing_randomization_table", "assignment_hash_mismatch", "assignment_not_following_schedule",
                   "insufficient_support", "condition_lock_no_within_stratum", "balance_diagnostic_fail",
                   "block_confounding_auc")


# ---------------------------------------------------------------- generative helpers (pre-assignment first)
def _apply_prior(p, C, shift=PRIOR_SHIFT):
    """Per-condition logit MAIN EFFECT on P(Y|C) that PRESERVES the Z-dependence of p (a concept C x Z survives)."""
    lg = np.log(np.clip(p, 1e-6, 1 - 1e-6)) - np.log(np.clip(1 - p, 1e-6, 1 - 1e-6))
    lg = lg + np.where(np.asarray(C) == HI, shift, -shift)
    return 1.0 / (1.0 + np.exp(-lg))


def _blocks(subj, rng, block_size=BLOCK_SIZE):
    """Per-subject pseudo-blocks (shuffle within subject, chunk). The randomization/matching unit."""
    Block = np.full(len(subj), -1, int); bid = 0
    for s in np.unique(subj):
        idx = np.where(subj == s)[0]; idx = idx[rng.permutation(len(idx))]
        for i in range(0, len(idx), block_size):
            Block[idx[i:i + block_size]] = bid; bid += 1
    return Block


def _registered_schedule(Block, Dc, rng):
    """PRE-ASSIGNMENT randomization TABLE: C balanced within each (block, design_class) stratum (class-balanced
    randomization -> C _|_ (Z, Dc) within block by design). Generated BEFORE any label/concept injection."""
    Block = np.asarray(Block); Dc = np.asarray(Dc)
    C = np.full(len(Block), LO, int)
    for b in np.unique(Block):
        for d in np.unique(Dc[Block == b]):
            si = np.where((Block == b) & (Dc == d))[0]
            k = len(si) // 2
            if k > 0:
                C[si[rng.permutation(len(si))[:k]]] = HI
    return C


def _schedule_hash(C_table, Block, Dc):
    """Hash-pin the registered schedule (order-invariant per (block,Dc) stratum via sorted composition)."""
    Block = np.asarray(Block); Dc = np.asarray(Dc); C_table = np.asarray(C_table)
    rows = []
    for b in sorted(np.unique(Block)):
        for d in sorted(np.unique(Dc[Block == b])):
            m = (Block == b) & (Dc == d)
            rows.append((int(b), int(d), int((C_table[m] == HI).sum()), int((C_table[m] == LO).sum())))
    return hashlib.sha1(str(sorted(rows)).encode()).hexdigest()[:16]


def _label_free_axis(Z, w, k=0):
    """k-th singular direction of centered Z, orthogonalized against the boundary normal w (label-free)."""
    w_hat = w / (np.linalg.norm(w) + 1e-12)
    Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)[2]
    v = Vt[min(k, Vt.shape[0] - 1)].astype(float)
    v = v - (v @ w_hat) * w_hat
    return v / (np.linalg.norm(v) + 1e-12)


def _flip_toward(C_table, proj, frac, rng):
    """Deviate from the registered schedule: flip a RANDOM `frac` subset of trials so C tracks sign(proj) on a
    low-variance axis. Random (not most-extreme) selection + low-variance axis = a QUIET confound whose C~Z signal is
    weak on the top-8 PCs (an AUC gate misses it, reliably <= tau) while schedule-adherence (match < 1) REFUSES it."""
    C = np.asarray(C_table).copy(); n = len(C)
    k = int(round(frac * n))
    if k <= 0:
        return C
    tgt = np.where(proj > np.median(proj), HI, LO)
    cand = np.where(C != tgt)[0]                          # trials not already at the confound target
    if len(cand) == 0:
        return C
    sel = rng.choice(cand, size=min(k, len(cand)), replace=False)
    C[sel] = tgt[sel]
    return C


def build_b8_1_cohort(world, coh, rng, block_size=BLOCK_SIZE):
    """Return (Z, Y, C, G, Block, Dc, C_table, table_hash, contract_intended). ORDER (enforced): blocks + design_class
    Dc(=real cued class y0) + registered schedule C_table + hash are generated BEFORE C is executed and BEFORE any
    label/concept injection. CONTRACT worlds execute C == C_table; VIOLATION worlds deviate from it."""
    Z, y0, subj = np.asarray(coh["Z"], float), np.asarray(coh["y"]), np.asarray(coh["subject"])
    G = subj.copy(); Block = _blocks(subj, rng, block_size)
    Dc = y0.copy()                                            # design_class = PRE-ASSIGNMENT cued class
    C_table = _registered_schedule(Block, Dc, rng)           # registered schedule (pre-injection)
    table_hash = _schedule_hash(C_table, Block, Dc)
    clf = EG._pooled_clf(Z, y0); w = clf.coef_.ravel().astype(float)
    p0 = clf.predict_proba(Z)[:, 1]
    kind, _ = WORLDS[world]

    if kind == "contract":                                   # execute the registered schedule EXACTLY
        C = C_table.copy()
        if world == "CONTRACT_random_label":
            Y = rng.permutation(y0)
        elif world == "CONTRACT_NULL_cov_plus_prior":        # balanced covariate (drawn _|_ C) + prior, no interaction
            v = _label_free_axis(Z, w)
            half = rng.permutation(len(Z))[:len(Z) // 2]
            Zc = Z.copy(); Zc[half] += 2.0 * float(np.std(Z @ v)) * v
            p = EG._pooled_clf(Zc, y0).predict_proba(Zc)[:, 1]
            Y = EG._draw(_apply_prior(p, C), rng)
        elif world in ("CONTRACT_POS_boundary", "CONTRACT_POS_boundary_plus_prior"):
            p2 = EG._rotate_proba(clf, Z, rng, deg=25.0)       # concept: rotate boundary in C=HI trials (C x Z)
            p = np.where(C == HI, p2, p0)
            Y = EG._draw(_apply_prior(p, C) if world.endswith("plus_prior") else p, rng)
        else:                                                # CONTRACT_NULL_balanced / CONTRACT_NULL_prior_only
            Y = EG._draw(_apply_prior(p0, C) if world == "CONTRACT_NULL_prior_only" else p0, rng)
        return Z, Y, C, G, Block, Dc, C_table, table_hash, True

    # VIOLATION: the executed assignment DEVIATES from the registered schedule -> provenance must REFUSE
    if world == "VIOLATION_condition_lock":                  # C constant within (block,Dc) -> no within-stratum support
        C = np.full(len(Z), LO, int)
        for b in np.unique(Block):
            if rng.random() < 0.5:
                C[Block == b] = HI
        return Z, EG._draw(p0, rng), C, G, Block, Dc, C_table, table_hash, False
    if world == "VIOLATION_prior_shift":                     # C tracks the LABEL axis p0(Z) (deviates from schedule) + Y prior
        C = np.where(p0 > np.median(p0), HI, LO)
        return Z, EG._draw(_apply_prior(p0, C), rng), C, G, Block, Dc, C_table, table_hash, False
    if world in ("VIOLATION_quiet_cov_no_concept", "VIOLATION_quiet_cov_plus_concept"):
        # QUIET: deviate from the schedule along a LOW-VARIANCE axis (last PC) -> block-adjusted top-8-PC AUC stays
        # <= tau (an AUC-only gate would PASS) but schedule-adherence (H3) REFUSES. The concept (if any) is on the
        # high-variance boundary plane, so it is a genuine C x Z that would alert IF the contract passed.
        vq = _label_free_axis(Z, w, k=Z.shape[1] - 1)
        C = _flip_toward(C_table, Z @ vq, QUIET_FLIP_FRAC, rng)
        if world == "VIOLATION_quiet_cov_plus_concept":
            p2 = EG._rotate_proba(clf, Z, rng, deg=25.0)
            Y = EG._draw(np.where(C == HI, p2, p0), rng)
        else:
            Y = EG._draw(p0, rng)
        return Z, Y, C, G, Block, Dc, C_table, table_hash, False
    v = _label_free_axis(Z, w)
    proj = Z @ v
    if world == "VIOLATION_cov_plus_prior":                  # the B7.1 killer: strong confound + prior
        C = np.where(proj > np.median(proj), HI, LO)
        return Z, EG._draw(_apply_prior(p0, C), rng), C, G, Block, Dc, C_table, table_hash, False
    # VIOLATION_cov_session: strong C=f(Z) confound
    C = np.where(proj > np.median(proj), HI, LO)
    return Z, EG._draw(p0, rng), C, G, Block, Dc, C_table, table_hash, False


# ---------------------------------------------------------------- contract validator (HARD provenance + diagnostics)
def _block_adjusted_auc(Z, C, Block, seed, n_pcs=8):
    """SECONDARY diagnostic: block-adjusted cross-fit AUC(C~Z) (WITHIN-block C predictability from Z). NOT the gate."""
    C = np.asarray(C); b = (C == HI).astype(int)
    if len(np.unique(b)) < 2:
        return float("nan")
    Zc = Z - Z.mean(0)
    Vt = np.linalg.svd(Zc, full_matrices=False)[2]
    F = Zc @ Vt[:min(n_pcs, Vt.shape[0])].T
    Fa = F.copy()
    for bl in np.unique(Block):
        m = Block == bl
        if m.sum() > 1: Fa[m] -= Fa[m].mean(0)
    e = np.full(len(b), b.mean(), float)
    kf = KFold(n_splits=min(5, len(b)), shuffle=True, random_state=seed)
    for tr, te in kf.split(Fa):
        if len(np.unique(b[tr])) < 2: e[te] = b[tr].mean(); continue
        e[te] = LogisticRegression(C=1.0, max_iter=1000).fit(Fa[tr], b[tr]).predict_proba(Fa[te])[:, 1]
    return float(roc_auc_score(b, e))


def check_contract_b8_1(Z, C, G, Block, Dc, C_table, table_hash, seed,
                        tau_auc=TAU_CONTRACT_AUC, min_support=MIN_SUPPORT_BLOCKS, bal_tol=BALANCE_TOL,
                        integrity_ok=None):
    """DATA-DRIVEN, PREDECLARED (before T). Returns (valid, diag). HARD provenance gate (primary):
    (H1) a registered schedule exists; (H2) its hash matches the pin (integrity, checked on the FULL registered
    schedule -- pass integrity_ok from the caller when auditing a subsample, else it is recomputed here on the arrays
    given); (H3) executed C FOLLOWS the schedule (adherence: C == C_table); (H4) within-(block,Dc) support >=
    min_support; (H5) the schedule is balanced within (block,Dc). SECONDARY fail-closed diagnostic: (D1) block-adjusted
    C~Z AUC <= tau_auc. Reason set fail-closed; AUC reported for EVERY cohort (so quiet confounds refused by provenance
    while AUC<=tau are visible)."""
    C = np.asarray(C); Dc = np.asarray(Dc); Block = np.asarray(Block)
    reasons = []
    # H1 registered table exists
    if C_table is None or len(C_table) != len(C):
        reasons.append("missing_randomization_table")
        return False, dict(invalid_reasons=reasons, provenance_match=float("nan"), within_block_C_Z_auc=float("nan"),
                           n_support_blocks=0, n_strata=0, max_margin=float("nan"), tau_auc=tau_auc)
    C_table = np.asarray(C_table)
    # H2 hash integrity of the REGISTERED schedule (subsample-invariant: caller passes integrity_ok for a subsample)
    hash_ok = bool(integrity_ok) if integrity_ok is not None else (_schedule_hash(C_table, Block, Dc) == table_hash)
    if not hash_ok:
        reasons.append("assignment_hash_mismatch")
    # H3 adherence: executed C follows the registered schedule
    match = float(np.mean(C == C_table))
    if match < 1.0:
        reasons.append("assignment_not_following_schedule")
    # H4/H5 support + balance within (block, Dc) strata
    n_support = 0; margins = []; locked = True
    for b in np.unique(Block):
        for d in np.unique(Dc[Block == b]):
            m = (Block == b) & (Dc == d)
            nhi = int((C[m] == HI).sum()); nlo = int((C[m] == LO).sum())
            if nhi >= 1 and nlo >= 1:
                n_support += 1; locked = False
            margins.append(abs(nhi - nlo))
    max_margin = int(max(margins)) if margins else 0
    if n_support < min_support:
        reasons.append("insufficient_support")
    if locked:
        reasons.append("condition_lock_no_within_stratum")
    # balance evaluated on the REGISTERED schedule (the contract's design), independent of the executed deviation
    tab_margins = []
    for b in np.unique(Block):
        for d in np.unique(Dc[Block == b]):
            m = (Block == b) & (Dc == d)
            tab_margins.append(abs(int((C_table[m] == HI).sum()) - int((C_table[m] == LO).sum())))
    if tab_margins and max(tab_margins) > bal_tol:
        reasons.append("balance_diagnostic_fail")
    # D1 secondary AUC diagnostic (reported always; fail-closed)
    auc = _block_adjusted_auc(Z, C, Block, seed)
    if not (auc == auc) or auc > tau_auc:
        reasons.append("block_confounding_auc")
    valid = (len(reasons) == 0)
    return bool(valid), dict(invalid_reasons=sorted(set(reasons)), provenance_match=match,
                             hash_ok=bool(hash_ok), within_block_C_Z_auc=auc, n_support_blocks=int(n_support),
                             n_strata=int(len(tab_margins)), max_margin=max_margin, tau_auc=tau_auc,
                             min_support=int(min_support))


def resample_C_exact_b8_1(C, Dc, Block, rng):
    """EXACT within-(block, DESIGN_CLASS) balanced permutation of C -- the KNOWN class-balanced randomization. Dc is a
    PRE-ASSIGNMENT stratifier (NOT the observed post-treatment Y), so there is NO collider. Preserves per-(block,Dc)
    condition counts exactly."""
    C = np.asarray(C); Dc = np.asarray(Dc); Block = np.asarray(Block); Cstar = C.copy()
    strata = {}
    for i in range(len(C)):
        strata.setdefault((int(Block[i]), int(Dc[i])), []).append(i)
    for idx in strata.values():
        idx = np.asarray(idx)
        Cstar[idx] = C[idx][rng.permutation(len(idx))]       # uniform permutation = exact class-balanced randomization
    return Cstar


def b8_1_certify(Z, Y, C, G, Block, Dc, C_table, table_hash, m, seed=0, rank=3, Creg=0.5, n_folds=PC.N_FOLDS,
                 min_epochs=PC.MIN_EPOCHS_PER_CONDITION, min_confirm_pairs=20, n_boot=200, alpha_budget=ALPHA_BUDGET):
    """Contract-FIRST: refuse (CONTRACT_INVALID_OR_UNIDENTIFIABLE) if the class-balanced randomization contract is not
    satisfied (hard provenance gate), BEFORE T is ever computed. Else compute observed_T + the EXACT (block,Dc)
    randomization null; ALERT iff both p-gates pass + estimable + size. NO world/oracle/B7 field enters."""
    Z, Y, C, G, Block, Dc = map(np.asarray, (Z, Y, C, G, Block, Dc))
    Z = Z.astype(float)
    out = dict(b8_state="NEED_MORE_LABELS", contract_valid=False, observed_T=float("nan"),
               p_exact_meanT=1.0, p_exact_stud=1.0, exact_null_mean_T=float("nan"), exact_null_sd_T=float("nan"),
               n_exact_invalid=0, n_eligible=0, provenance_match=float("nan"), contract_invalid_reasons=[])
    # H2 integrity on the FULL registered schedule (subsample-invariant), computed once before any subsampling
    integ = None if C_table is None else (_schedule_hash(np.asarray(C_table), Block, Dc) == table_hash)
    elig = PC.eligible_complete_pairs(C, G, min_epochs)
    if len(elig) < n_folds * 2:
        out.update(b8_state="INSUFFICIENT_LABELS", reason=f"{len(elig)} eligible"); return out
    rng0 = np.random.default_rng(seed)
    pick = rng0.choice(np.array(sorted(elig)), size=min(int(m), len(elig)), replace=False)
    mask = np.isin(G, pick)
    Zq, Yq, Cq, Gq, Bq, Dq = Z[mask], Y[mask], C[mask], G[mask], Block[mask], Dc[mask]
    Tq = None if C_table is None else np.asarray(C_table)[mask]
    elig_q = PC.eligible_complete_pairs(Cq, Gq, min_epochs)
    if len(elig_q) < n_folds * 2:
        out.update(b8_state="INSUFFICIENT_LABELS", reason="queried degenerate"); return out
    mq = np.isin(Gq, elig_q)
    Zq, Yq, Cq, Gq, Bq, Dq = Zq[mq], Yq[mq], Cq[mq], Gq[mq], Bq[mq], Dq[mq]
    Tq = None if Tq is None else Tq[mq]
    # CONTRACT CHECK (hard provenance gate; predeclared; before the test). integrity from the FULL schedule.
    valid, cdiag = check_contract_b8_1(Zq, Cq, Gq, Bq, Dq, Tq, table_hash, seed + 13, integrity_ok=integ)
    out.update(contract_valid=bool(valid), n_eligible=int(len(elig_q)),
               provenance_match=float(cdiag.get("provenance_match", float("nan"))),
               contract_invalid_reasons=list(cdiag.get("invalid_reasons", [])),
               **{f"contract_{k}": v for k, v in cdiag.items() if k not in ("invalid_reasons", "provenance_match")})
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
        Cstar = resample_C_exact_b8_1(Cq, Dq, Bq, rng)         # EXACT (block, design_class) randomization null
        prep = PC._prep_folds(Zq, Cstar, Gq, folds, "centered", rank, Creg)
        if prep is None: ninv += 1; ge_t += 1; ge_z += 1; continue
        Ts, oks, ds = PC._T_cv(prep, Yq, Cstar, Gq, cl, Creg)
        if not oks: ninv += 1; ge_t += 1; ge_z += 1; continue
        Zs = PC._studentize(ds)["Z"]
        tstars.append(Ts); ge_t += int(Ts >= T_obs); ge_z += int(Zs >= Z_obs)
    p_t, p_z = ge_t / (n_boot + 1), ge_z / (n_boot + 1)
    estimable = ninv <= 0.20 * n_boot
    alert = bool(estimable and p_t <= alpha_budget and p_z <= alpha_budget and len(elig_q) >= min_confirm_pairs)
    state = "SAMPLER_INVALID" if not estimable else ("B8_CONCEPT_ALERT" if alert else "NO_ACTIONABLE_CONCEPT_EVIDENCE")
    out.update(b8_state=state, observed_T=float(T_obs), p_exact_meanT=float(p_t), p_exact_stud=float(p_z),
               exact_null_mean_T=float(np.mean(tstars)) if tstars else float("nan"),
               exact_null_sd_T=float(np.std(tstars)) if tstars else float("nan"), n_exact_invalid=int(ninv))
    return out
