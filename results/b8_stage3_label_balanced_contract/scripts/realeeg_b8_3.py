"""CSC B8.3 label-balanced / case-control randomized-audit contract (development-only; reviewer-authorized 2026-07-07
after B8.2 FALSIFIED B8.1's decision-level stability -- prior_only 18/300 & cov_plus_prior 13/300 both fail <=7/300; the
prior-bearing nulls retain a collider/mean-T residual that B8.1's studentized both-gate only masked, unstably).

CORE REDESIGN (contract, NOT statistic/gate/p-value). B8.1 tried to remove label-prior via design_class + the both-gate;
the prior residual survived (B8.2 falsified stability). B8.3 addresses the prior with the AUDIT SAMPLING CONTRACT: a
predeclared, deterministic, Z-blind CASE-CONTROL selector balances the audited sample so P(Y|C) is equal by construction;
the EXACT null RE-APPLIES the same selector under every randomized assignment C*.

HONEST SCOPE (design red-team wrznv3lin, folded in BEFORE Phase B): count-balancing removes the FIRST-ORDER P(Y|C) main
effect EXACTLY (equal per-condition label composition; C x Y imbalance == 0). It does NOT by construction remove two
residual prior channels -- (a) SELECTION-INTENSITY asymmetry: observed-C carries the prior, so the selector discards more
trials than under randomized C* (observed selected_n < null mean) -> an obs-vs-null asymmetry; (b) within-Y C-Z dependence:
stratifying on the COLLIDER Y (Y<-C prior, Y<-Z) opens a C-Z path count-balancing does not close. Both are the SAME family
that failed B8.2. Whether they re-inflate prior_only/cov_plus_prior is EMPIRICAL -- exactly what Phase B's mean-T-alone
<=7/300 PRIMARY screen measures. This is a genuine test, NOT a guaranteed fix; a breach -> B9 / estimand-narrowing (NOT a
gate/mean-T retune).

NARROWED ESTIMAND: label-balanced AUDIT-POPULATION boundary evidence -- condition-specific boundary/interaction evidence
in a predeclared label-balanced audit sample -- NOT a natural-prevalence deployment concept certificate. Prior MAIN effect
is out of the estimand (removed by the sampling contract); prior-carrying residuals (above) are tested, not assumed gone.

REUSE (byte-frozen from committed B8.1): build_b8_1_cohort (12 worlds), check_contract_b8_1 (hard provenance gate H3), and
resample_C_exact_b8_1 (within-(block,Dc) class-balanced randomization null). The B3 contrast T (PC._T_cv/_prep_folds/
_studentize) is byte-reused. The ONLY new machinery is the audit selector + its re-application inside the null. NO fitted
Y|Z or C|Z; NO oracle/B7/observed_T/router; NO threshold/p-value tuning. Emulator (Lee2019 SM16) -- NOT validation. NO tag.
"""
import hashlib
import numpy as np
import realeeg_b8_1 as B81
from csc.mininfo import paired_calibrated as PC

ALPHA_BUDGET = B81.ALPHA_BUDGET          # 0.025, unchanged
LO, HI = B81.LO, B81.HI
WORLDS = B81.WORLDS                      # same 12 conditions as B8.1/B8.2
MIN_EPOCHS_AUDIT = 4                     # min selected trials per condition per subject on the BALANCED audit subset
                                         # (case-control balancing halves per-stratum counts; feasibility tested in Phase-A smoke)


def _stratum_seed(seed, g, b, y):
    """Deterministic per-(subject,block,label) seed so the selector's tie-breaking is INDEPENDENT of C and of other
    strata (=> A(.,Y) is a pure deterministic function of (C,Y,G,Block,seed), re-applicable identically under any C*)."""
    return int(hashlib.sha1(f"{int(seed)}:{int(g)}:{int(b)}:{int(y)}".encode()).hexdigest()[:12], 16)


def audit_select(C, Y, G, Block, seed):
    """PREDECLARED label-balanced CASE-CONTROL audit selector. Within each (subject, block, audited-label Y) stratum,
    select k = min(#C==HI, #C==LO) trials from EACH condition (seeded choice) -> the selected audit sample has EQUAL
    per-condition label composition by construction (P(Y|C) balanced). Uses ONLY C,Y,G,Block,seed -- NEVER Z, T, or any
    p-value/result. Deterministic given seed (per-stratum seeded).

    INVARIANT (design red-team wrznv3lin, low): the mask is a pure fn of (C,Y,G,Block,seed) but is NOT invariant to a
    global ROW REORDERING of the inputs (rng.choice picks in array order). b8_3_certify MUST call this with Yq,Gq,Bq in
    the SAME row order for the observed and every null draw (only C changes) -- do NOT sort/reshuffle rows between calls.
    Returns (mask, diag)."""
    C = np.asarray(C); Y = np.asarray(Y); G = np.asarray(G); Block = np.asarray(Block)
    mask = np.zeros(len(C), bool)
    n_strata = 0; n_infeasible_strata = 0
    for g in np.unique(G):
        for b in np.unique(Block[G == g]):
            gb = (G == g) & (Block == b)
            for y in np.unique(Y[gb]):
                idx = np.where(gb & (Y == y))[0]
                hi = idx[C[idx] == HI]; lo = idx[C[idx] == LO]
                n_strata += 1
                k = min(len(hi), len(lo))
                if k <= 0:
                    n_infeasible_strata += 1
                    continue
                rng = np.random.default_rng(_stratum_seed(seed, g, b, y))
                mask[rng.choice(hi, k, replace=False)] = True
                mask[rng.choice(lo, k, replace=False)] = True
    return mask, dict(n_strata=int(n_strata), n_infeasible_strata=int(n_infeasible_strata))


def _cxy_imbalance(C, Y, mask):
    """Max |#(C=HI,Y=y) - #(C=LO,Y=y)| over labels y on the SELECTED subset -- must be 0 for an exactly balanced audit."""
    C = np.asarray(C)[mask]; Y = np.asarray(Y)[mask]
    if len(C) == 0:
        return -1
    return int(max(abs((C[Y == y] == HI).sum() - (C[Y == y] == LO).sum()) for y in np.unique(Y)))


def _T_on_subset(Zs, Ys, Cs, Gs, folds, cl, rank, Creg):
    """Byte-reused B3 contrast on a selected audit subset with fixed subject-folds. Returns (T, ok, Zstud)."""
    prep = PC._prep_folds(Zs, Cs, Gs, folds, "centered", rank, Creg)
    if prep is None:
        return float("nan"), False, float("nan")
    T, ok, deltas = PC._T_cv(prep, Ys, Cs, Gs, cl, Creg)
    if not ok:
        return float("nan"), False, float("nan")
    return float(T), True, float(PC._studentize(deltas)["Z"])


def b8_3_certify(Z, Y, C, G, Block, Dc, C_table, table_hash, m, seed=0, rank=3, Creg=0.5, n_folds=PC.N_FOLDS,
                 min_epochs_audit=MIN_EPOCHS_AUDIT, min_confirm_pairs=20, n_boot=200, alpha_budget=ALPHA_BUDGET):
    """Contract-FIRST (reuse B8.1 hard provenance gate). Then: OBSERVED -> apply the label-balanced audit selector,
    compute T on the balanced audit subset. NULL -> for each within-(block,Dc) randomized C*, RE-APPLY the selector
    A(C*,Y) and recompute T on the newly-selected balanced subset (the selection process is PART of the null). ALERT =
    B8.1 both-gate on the audit subset. mean-T-alone is reported (now a PRIMARY screen, not just diagnostic). Fail-closed
    on infeasible selection under any draw (no silent drop). NO Z/T in the selector; NO fitted null."""
    Z, Y, C, G, Block, Dc = (np.asarray(Z, float), np.asarray(Y), np.asarray(C), np.asarray(G),
                             np.asarray(Block), np.asarray(Dc))
    out = dict(b8_state="NEED_MORE_LABELS", contract_valid=False, observed_T=float("nan"),
               p_exact_meanT=1.0, p_exact_stud=1.0, exact_null_mean_T=float("nan"), exact_null_sd_T=float("nan"),
               n_exact_invalid=0, n_eligible=0, provenance_match=float("nan"), contract_invalid_reasons=[],
               audit_selected_n=0, audit_cxy_imbalance=-1, audit_n_infeasible_strata=0, null_infeasible_draws=0,
               meanT_alone=False)
    integ = None if C_table is None else (B81._schedule_hash(np.asarray(C_table), Block, Dc) == table_hash)
    # query subsample (same as B8.1: whole subjects)
    elig0 = PC.eligible_complete_pairs(C, G, PC.MIN_EPOCHS_PER_CONDITION)
    if len(elig0) < n_folds * 2:
        out.update(b8_state="INSUFFICIENT_LABELS", reason=f"{len(elig0)} eligible pre-audit"); return out
    rng0 = np.random.default_rng(seed)
    pick = rng0.choice(np.array(sorted(elig0)), size=min(int(m), len(elig0)), replace=False)
    mask = np.isin(G, pick)
    Zq, Yq, Cq, Gq, Bq, Dq = Z[mask], Y[mask], C[mask], G[mask], Block[mask], Dc[mask]
    Tq = None if C_table is None else np.asarray(C_table)[mask]
    # CONTRACT CHECK (hard provenance gate; predeclared; BEFORE any selection/T) -- reuse B8.1 verbatim
    valid, cdiag = B81.check_contract_b8_1(Zq, Cq, Gq, Bq, Dq, Tq, table_hash, seed + 13, integrity_ok=integ)
    out.update(contract_valid=bool(valid), provenance_match=float(cdiag.get("provenance_match", float("nan"))),
               contract_invalid_reasons=list(cdiag.get("invalid_reasons", [])),
               **{f"contract_{k}": v for k, v in cdiag.items() if k not in ("invalid_reasons", "provenance_match")})
    if not valid:
        out["b8_state"] = "CONTRACT_INVALID_OR_UNIDENTIFIABLE"; return out
    # OBSERVED audit selection (predeclared, Z-blind)
    sel, sdiag = audit_select(Cq, Yq, Gq, Bq, seed + 31)
    out.update(audit_n_infeasible_strata=int(sdiag["n_infeasible_strata"]))
    if sel.sum() == 0:
        out.update(b8_state="INSUFFICIENT_LABELS", reason="audit selection empty"); return out
    Za, Ya, Ca, Ga = Zq[sel], Yq[sel], Cq[sel], Gq[sel]
    imb = _cxy_imbalance(Cq, Yq, sel)
    out.update(audit_selected_n=int(sel.sum()), audit_cxy_imbalance=int(imb))
    if imb != 0:
        out.update(b8_state="SAMPLER_INVALID", reason=f"observed audit C x Y imbalance {imb}"); return out
    elig = PC.eligible_complete_pairs(Ca, Ga, min_epochs_audit)
    out.update(n_eligible=int(len(elig)))
    if len(elig) < n_folds * 2 or len(np.unique(Ya)) < 2 or len(np.unique(Ca)) != 2:
        out.update(b8_state="INSUFFICIENT_LABELS", reason=f"{len(elig)} eligible post-audit"); return out
    ma = np.isin(Ga, elig); Za, Ya, Ca, Ga = Za[ma], Ya[ma], Ca[ma], Ga[ma]
    cl = np.array(sorted(np.unique(Ya)))
    folds, _ = PC._make_folds(elig, n_folds, seed)
    T_obs, ok, Z_obs = _T_on_subset(Za, Ya, Ca, Ga, folds, cl, rank, Creg)
    if not ok:
        out.update(b8_state="SAMPLER_INVALID", reason="observed audit cross-fit degenerate"); return out
    # NULL: within-(block,Dc) randomization RE-APPLYING the selector each draw (selection is PART of the null)
    rng = np.random.default_rng(seed + 777)
    ge_t = ge_z = 1; ninv = 0; ninfeas = 0; tstars = []; null_seln = []
    for _ in range(n_boot):
        Cstar = B81.resample_C_exact_b8_1(Cq, Dq, Bq, rng)
        sstar, _sd = audit_select(Cstar, Yq, Gq, Bq, seed + 31)          # RE-APPLY selector under C* (balances Cstar x Y)
        if sstar.sum() == 0 or _cxy_imbalance(Cstar, Yq, sstar) != 0:    # balance check on Cstar (the assignment selected on)
            ninfeas += 1; ninv += 1; ge_t += 1; ge_z += 1; continue      # fail-closed, NO silent drop
        null_seln.append(int(sstar.sum()))   # DIAGNOSTIC (red-team wrznv3lin): selection-intensity asymmetry obs vs null
        Zs, Ys, Cs, Gs = Zq[sstar], Yq[sstar], Cstar[sstar], Gq[sstar]
        eg = PC.eligible_complete_pairs(Cs, Gs, min_epochs_audit)
        if len(eg) < n_folds * 2 or len(np.unique(Ys)) < 2 or len(np.unique(Cs)) != 2:
            ninfeas += 1; ninv += 1; ge_t += 1; ge_z += 1; continue
        mg = np.isin(Gs, eg); Zs, Ys, Cs, Gs = Zs[mg], Ys[mg], Cs[mg], Gs[mg]
        Ts, oks, Zst = _T_on_subset(Zs, Ys, Cs, Gs, folds, cl, Creg=Creg, rank=rank)
        if not oks:
            ninv += 1; ge_t += 1; ge_z += 1; continue
        tstars.append(Ts); ge_t += int(Ts >= T_obs); ge_z += int(Zst >= Z_obs)
    p_t = ge_t / (n_boot + 1); p_z = ge_z / (n_boot + 1)
    estimable = ninv <= 0.20 * n_boot
    both_alert = bool(estimable and p_t <= alpha_budget and p_z <= alpha_budget and len(elig) >= min_confirm_pairs)
    meanT_alone = bool(estimable and p_t <= alpha_budget and len(elig) >= min_confirm_pairs)   # PRIMARY screen too
    state = "SAMPLER_INVALID" if not estimable else ("B8_CONCEPT_ALERT" if both_alert else "NO_ACTIONABLE_CONCEPT_EVIDENCE")
    out.update(b8_state=state, observed_T=float(T_obs), p_exact_meanT=float(p_t), p_exact_stud=float(p_z),
               exact_null_mean_T=float(np.mean(tstars)) if tstars else float("nan"),
               exact_null_sd_T=float(np.std(tstars)) if tstars else float("nan"),
               n_exact_invalid=int(ninv), null_infeasible_draws=int(ninfeas), meanT_alone=bool(meanT_alone),
               observed_Tz=float(Z_obs),
               null_selected_n_mean=float(np.mean(null_seln)) if null_seln else float("nan"),
               selection_intensity_asymmetry=float(int(sel.sum()) - np.mean(null_seln)) if null_seln else float("nan"))
    return out
