"""ACAR v3 DEV bake-off + S2/S4 gates (S5 split → train-once → calibrate → OOF records → S2 admissibility → S4
selection → refit). DESIGN/DEV stage — SYNTHETIC FIXTURES ONLY until DEV_DESIGN_LOCK; reads NO real DEV cohort values
and emits NO binding go/no-go. The FIRST real DEV run (after the tag) computes ONLY the S2 admissibility + S4 selection
gate; binding G2 / coverage / harmful-rate / two-site are LATER external Arm B (S6), not DEV.

Wiring (per disease, on that disease's pooled cohorts resolved through a SourceStateRegistry):
  outer folds over ALL subjects (each EVAL once, incl. fallback-only) → non-EVAL ELIGIBLE hash-split FIT/CAL → FIT
  hash-split TRAIN/VAL. predictor sees FIT only; conformal q sees CAL only (ONE joint score per eligible CAL subject);
  S2/S4 diagnostics aggregate on OOF EVAL. Each batch is executed ONCE; features + ΔR share that execution. Fallback
  batches/subjects are retained in EVAL accounting (identity, ΔR 0) but never fitted/CAL'd. The first OOF pass emits
  immutable OOF records; the C2 final floor is Q05 of their `scale_raw` (no second training pass).
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
import os
import pickle

import numpy as np

from .set_features import NON_IDENTITY
from .data import deployment_batch_digest, canon_subject
from .conformal import subject_joint_score, conformal_q, route
from .predictors import score as cand_score, upper_bound, HP
from .training import fit_candidate_earlystop, refit_candidate_fixed_epochs, final_epochs
from .splits import cv_assignment
from .loader import hash_subject_list, _digest, SourceStateRegistry, SourceStateArtifact

Z90 = 1.2815515594463558                                   # standard-normal 0.90 quantile
AUROC_GATE = 0.60                                          # G1 (per disease, ≥1 action)
WIDTH_REDUCTION = 0.30                                     # disease-macro width ≥30% below C0
COVERAGE_MIN = 0.15                                        # OOF adaptation coverage
C0_SEED = 0                                                # v2 binding-runner model seed (NOT seed_es)


def _subject_weighted(values_by_subject):
    """Each subject contributes total weight 1 (split across its records). Returns (flat values, flat weights)."""
    vals, wts = [], []
    for v in values_by_subject.values():
        v = np.asarray(v, float)
        if not len(v):
            continue
        vals.append(v); wts.append(np.full(len(v), 1.0 / len(v)))
    if not vals:
        raise ValueError("no subject values")
    return np.concatenate(vals), np.concatenate(wts)


def _subject_macro_mean(by_subject):
    """Mean over subjects of each subject's mean — each subject totals weight 1 (a 100-batch subject does not dominate
    a 1-batch subject). Used for width AND MAE in BOTH the candidate and the C0 replay (apples-to-apples)."""
    ms = [float(np.mean(v)) for v in by_subject.values() if len(v)]
    return float(np.mean(ms)) if ms else float("inf")


def _weighted_quantile(values, weights, q):
    """Deterministic weighted empirical quantile (lower-tail, normalized weights, sorted)."""
    v = np.asarray(values, float); w = np.asarray(weights, float)
    order = np.argsort(v, kind="stable"); v = v[order]; w = w[order]
    cw = np.cumsum(w); cw /= cw[-1]
    return float(v[int(np.searchsorted(cw, q, side="left"))]) if cw[-1] > 0 else float("nan")


# ============================================================================================== pool / subject index
def _subject_batches(batches):
    idx = {}
    for b in batches:
        c = canon_subject(b.subject)
        slot = idx.setdefault(c, {"key": b.subject, "eligible": [], "fallback": []})
        (slot["fallback"] if b.fallback else slot["eligible"]).append(b)
    return idx


def _eligible_subjects(idx):
    return [v["key"] for v in idx.values() if v["eligible"]]


def pool_digest(batches) -> str:
    return _digest(b"POOL/1", sorted(deployment_batch_digest(b).encode() for b in batches))


def _as_registry(reg_or_art, disease):
    if isinstance(reg_or_art, SourceStateRegistry):
        return reg_or_art
    if isinstance(reg_or_art, SourceStateArtifact):
        return SourceStateRegistry(disease).add(reg_or_art)
    raise TypeError("expected SourceStateRegistry or SourceStateArtifact")


# ------------------------------------------------------------------------------------------------ execution cache
def _c0_vector(state, p0, z0, za, pa):
    """BIT-FOR-BIT v2 feature vector: reuses acar.features.paired_features/context_features/feature_vector on the SAME
    captured execution (z_pre=identity embedding, z_post=action embedding-or-None). 11-dim, NaN→0, v2 ordering."""
    from acar.features import paired_features, context_features, feature_vector
    return feature_vector(paired_features(p0, pa, z0, za), context_features(state, za, pa))


def disease_exec_cache(registry, batches, labels):
    """Execute every ELIGIBLE batch EXACTLY ONCE (the adapters are the expensive, candidate-independent step) and cache
    the captured WindowActionSets + ΔR + execution hashes + the v2 C0 vectors. Keyed by deployment_batch_digest. Reused
    across all folds, candidates, and the C0 replay — so the whole DEV bake-off triggers each batch's execution once."""
    cache = {}; states = {}
    for b in batches:
        if b.fallback:
            continue
        sa = registry.resolve(b)
        if sa.source_state_ref not in states:
            states[sa.source_state_ref] = sa._ephemeral_state()
        st = states[sa.source_state_ref]
        exe = sa.execute(b)
        p0 = np.asarray(exe.p0, float); z0 = np.asarray(exe.z0, float)
        c0feat = {a: _c0_vector(st, p0, z0, None if za is None else np.asarray(za, float), np.asarray(pa, float))
                  for a, za, pa in exe.per_action}
        cache[deployment_batch_digest(b)] = {
            "exe": exe, "was": exe.window_action_sets(sa), "c0feat": c0feat,
            "dr": dict(exe.labeled_risk_record(labels).delta_r_by_action)}
    return cache


def _preds_and_dr(artifact, batch, cache):
    c = cache[deployment_batch_digest(batch)]
    preds = {a: artifact.predict(c["was"][a]) for a in NON_IDENTITY}
    return preds, c["dr"]


# ----------------------------------------------------------------------------------------------------- OOF records
@dataclass(frozen=True)
class OOFRecord:
    candidate: str
    disease: str
    subject: str          # canon
    batch_digest: str
    fold: int
    action: str
    delta_r: float
    point: float
    upper_center: float
    scale_raw: float      # NaN for C1/C3
    scale_used: float     # NaN for C1/C3
    score: float
    q: float
    upper: float          # U_a = upper_bound(pred, q)
    chosen: str           # router choice for this batch (same across its actions)


def _train_examples(subjects, idx, cache):
    """CANONICAL-order TrainExamples (subjects by canon, batches by digest) from ELIGIBLE batches, built from the cached
    single execution (features + ΔR + execution hashes all from that one pass)."""
    from .training import DeploymentFeatureRecord
    out = []
    want = {canon_subject(s) for s in subjects}
    for cc in sorted(want):
        slot = idx.get(cc)
        if slot is None:
            continue
        for b in sorted(slot["eligible"], key=deployment_batch_digest):
            c = cache[deployment_batch_digest(b)]; exe = c["exe"]
            dfr = DeploymentFeatureRecord(b.disease, b.subject, exe.deployment_batch_digest,
                                          tuple((a, c["was"][a], c["dr"][a]) for a in NON_IDENTITY),
                                          execution_sha256=exe.execution_sha256,
                                          action_outputs_sha256=exe.action_outputs_sha256)
            out += dfr.to_train_examples()
    return out


@dataclass(frozen=True)
class CandidateOOF:
    disease: str
    candidate: str
    records: tuple
    best_epochs: tuple
    fold_qs: tuple
    n_eval_eligible_batches: int
    n_eval_fallback_batches: int
    n_eval_subjects: int
    n_cal_scores_per_fold: tuple
    fold_provenance: tuple        # per-fold FIT/CAL/EVAL subject-list hashes + counts + m/k/q_raw/q_used


def _elig_batches(subjects, idx):
    return [b for s in subjects for b in idx[canon_subject(s)]["eligible"]]


def run_oof(disease, reg_or_art, batches, labels, candidate, alpha=0.10, delta=0.0, cache=None) -> CandidateOOF:
    registry = _as_registry(reg_or_art, disease)
    idx = _subject_batches(batches)
    eligible = _eligible_subjects(idx)
    if len(eligible) < HP["k_folds"]:
        raise ValueError(f"need >= k_folds={HP['k_folds']} eligible subjects; got {len(eligible)}")
    if cache is None:
        cache = disease_exec_cache(registry, batches, labels)
    all_subjects = [v["key"] for v in idx.values()]
    elig_canon = {canon_subject(s) for s in eligible}
    assignment, _all = cv_assignment(all_subjects, eligible=elig_canon)     # outer over ALL; FIT/CAL from eligible only
    records = []; best_epochs = []; fold_qs = []; n_elig = n_fb = 0; eval_subj = set(); n_cal = []; fold_prov = []
    for fa in assignment:
        tr = _train_examples(fa["train"], idx, cache)
        va = _train_examples(fa["val"], idx, cache)
        artifact, best = fit_candidate_earlystop(candidate, disease, tr, va, HP["seed_es"])
        best_epochs.append(best)
        cal_scores = []
        for s in fa["cal"]:
            slot = idx[canon_subject(s)]
            if not slot["eligible"]:
                continue
            sb = []
            for b in sorted(slot["eligible"], key=deployment_batch_digest):
                preds, dr = _preds_and_dr(artifact, b, cache)
                sb.append({a: (preds[a], dr[a]) for a in NON_IDENTITY})
            cal_scores.append(subject_joint_score(sb))
        q, k = conformal_q(cal_scores, alpha); fold_qs.append(float(q)); n_cal.append(len(cal_scores))
        q_used = max(q, 0.0) if candidate == "C2" else q       # C2 deploy clamps q⁺; C1/C3 use q
        fit_subs = [canon_subject(s) for s in fa["fit"]]; cal_subs = [canon_subject(s) for s in fa["cal"]]
        eval_subs = [canon_subject(s) for s in fa["eval"]]
        fold_prov.append({
            "fold": fa["fold"],
            "fit_subject_list_sha256": hash_subject_list(fa["fit"]) if fa["fit"] else "",
            "cal_subject_list_sha256": hash_subject_list(fa["cal"]) if fa["cal"] else "",
            "eval_subject_list_sha256": hash_subject_list(fa["eval"]) if fa["eval"] else "",
            "fit_subjects": fit_subs, "cal_subjects": cal_subs, "eval_subjects": eval_subs,
            "n_fit_subjects": len(fa["fit"]), "n_fit_batches": len(_elig_batches(fa["fit"], idx)),
            "n_cal_subjects": len(fa["cal"]), "n_cal_batches": len(_elig_batches(fa["cal"], idx)),
            "n_eval_subjects": len(fa["eval"]), "n_eval_batches": len(_elig_batches(fa["eval"], idx)),
            "m": len(cal_scores), "k": int(k), "q_raw": float(q),
            "q_used": (float(q_used) if math.isfinite(q_used) else None),
        })
        for s in fa["eval"]:
            slot = idx[canon_subject(s)]; eval_subj.add(canon_subject(s))
            for b in sorted(slot["eligible"], key=deployment_batch_digest):
                preds, dr = _preds_and_dr(artifact, b, cache)
                U = {a: upper_bound(preds[a], q) for a in NON_IDENTITY}
                chosen, _U = route(preds, q, delta)
                for a in NON_IDENTITY:
                    p = preds[a]
                    records.append(OOFRecord(candidate, disease, canon_subject(s), deployment_batch_digest(b), fa["fold"],
                                             a, float(dr[a]), float(p.point), float(p.upper_center),
                                             float(p.scale_raw) if p.scale_raw is not None else float("nan"),
                                             float(p.scale_used) if p.scale_used is not None else float("nan"),
                                             float(cand_score(p, dr[a])), float(q), float(U[a]), chosen))
                n_elig += 1
            n_fb += len(slot["fallback"])
    return CandidateOOF(disease, candidate, tuple(records), tuple(best_epochs), tuple(fold_qs),
                        n_elig, n_fb, len(eval_subj), tuple(n_cal), tuple(fold_prov))


# ===================================================================================== S2 gates (pure, fail-closed)
def _by_action(records):
    out = {a: [r for r in records if r.action == a] for a in NON_IDENTITY}
    return out


def _subject_balanced(values_by_subject):
    """(mean, var) with each subject weighted equally (mean of per-subject means; var from subject-equal moments)."""
    if not values_by_subject:
        raise ValueError("no subjects")
    mus = [float(np.mean(v)) for v in values_by_subject.values() if len(v)]
    m = float(np.mean(mus))
    vs = [float(np.mean((np.asarray(v) - m) ** 2)) for v in values_by_subject.values() if len(v)]
    return m, float(np.mean(vs))


def s2_c2_gate(records):
    """C2: per-action SUBJECT-EQUAL-WEIGHTED standardized-residual mean∈[-0.25,0.25], variance∈[0.5,2.0], positive-tail
    90th pct ∈[0.8,2.0]·z₀.₉₀. Each subject totals weight 1 (so a 100-batch subject does not dominate). FAIL-CLOSED."""
    res = {}; ok = True
    for a, rs in _by_action(records).items():
        if not rs:
            raise ValueError(f"C2 S2: no OOF records for action {a}")
        by_s = {}
        for r in rs:
            if not (math.isfinite(r.scale_used) and r.scale_used > 0):
                raise ValueError("C2 S2: non-positive scale_used")
            by_s.setdefault(r.subject, []).append((r.delta_r - r.upper_center) / r.scale_used)
        m, v = _subject_balanced(by_s)
        vals, wts = _subject_weighted(by_s)
        tail = _weighted_quantile(vals, wts, 0.90)              # subject-equal-weighted tail
        a_ok = (abs(m) <= 0.25) and (0.5 <= v <= 2.0) and (0.8 * Z90 <= tail <= 2.0 * Z90)
        res[a] = dict(mean=m, var=v, tail90=tail, ok=bool(a_ok)); ok = ok and a_ok
    res["pass"] = bool(ok)
    return res


def s2_c3_gate(records):
    """C3: per-action SUBJECT-EQUAL-WEIGHTED exceedance P(ΔR>q̂₉₀)∈[0.05,0.20]; positive-excess 95th pct ≤ 2·(OOF ΔR SD);
    q̂₉₀>q̂₅₀ everywhere. exceedance/excess-95/SD are subject-weighted (each subject totals weight 1)."""
    res = {}; ok = True
    for a, rs in _by_action(records).items():
        if not rs:
            raise ValueError(f"C3 S2: no OOF records for action {a}")
        by_exc, by_excess, by_dr = {}, {}, {}
        for r in rs:
            by_exc.setdefault(r.subject, []).append(1.0 if r.delta_r > r.upper_center else 0.0)
            by_excess.setdefault(r.subject, []).append(max(r.delta_r - r.upper_center, 0.0))
            by_dr.setdefault(r.subject, []).append(r.delta_r)
        ev, ew = _subject_weighted(by_exc); exc = float(np.sum(ev * ew) / np.sum(ew))   # weighted mean
        xv, xw = _subject_weighted(by_excess); excess95 = _weighted_quantile(xv, xw, 0.95)
        dv, dw = _subject_weighted(by_dr)
        dmean = float(np.sum(dv * dw) / np.sum(dw)); sd = float(np.sqrt(np.sum(dw * (dv - dmean) ** 2) / np.sum(dw)))
        crossing_ok = bool(all(r.upper_center > r.point for r in rs))
        a_ok = (0.05 <= exc <= 0.20) and (excess95 <= 2.0 * sd) and crossing_ok
        res[a] = dict(exceedance=exc, excess95=excess95, dr_sd=sd, crossing_ok=crossing_ok, ok=bool(a_ok))
        ok = ok and a_ok
    res["pass"] = bool(ok)
    return res


def maxa_dominance(records, max_share=0.60):
    """`max_a share_a ≤ max_share` with FRACTIONAL tie credit (action-order invariant). M_{s,a}=max_B score_{sBa};
    T_s=argmax_a; share_a=(1/N)Σ_s 1[a∈T_s]/|T_s|. Applies to C1/C2/C3 (the candidate's own nonconformity `score`)."""
    by_sa = {}
    for r in records:
        by_sa.setdefault(r.subject, {}).setdefault(r.action, []).append(r.score)
    subjects = sorted(by_sa)
    if not subjects:
        raise ValueError("dominance: no subjects")
    share = {a: 0.0 for a in NON_IDENTITY}
    for s in subjects:
        M = {a: max(by_sa[s].get(a, [-math.inf])) for a in NON_IDENTITY}
        mx = max(M.values())
        T = [a for a in NON_IDENTITY if M[a] == mx]
        for a in T:
            share[a] += 1.0 / len(T)
    share = {a: share[a] / len(subjects) for a in NON_IDENTITY}
    top = max(share.values())
    return dict(shares=share, max_share=float(top), ok=bool(top <= max_share))


def c2_floor_from_oof(records, quantile=None):
    """C2 final σ_min,a = Q05 of the OOF `scale_raw` (NOT scale_used) over the SAME records. Pinned numpy quantile."""
    quantile = HP["sigma_min_quantile"] if quantile is None else quantile
    sm = {}
    for a, rs in _by_action(records).items():
        raw = [r.scale_raw for r in rs if math.isfinite(r.scale_raw)]
        if not raw:
            raise ValueError(f"C2 floor: no scale_raw for action {a}")
        v = float(np.quantile(np.asarray(raw, float), quantile, method="linear"))
        if not (math.isfinite(v) and v > 0):
            raise ValueError(f"C2 floor[{a}] non-positive")
        sm[a] = v
    return sm


# ============================================================================================= C0 / v2 full-recipe replay
@dataclass(frozen=True)
class C0Report:
    disease: str
    red_router: float
    adaptation_coverage: float
    mae: float
    max_action_auroc: float
    width: float
    n_eval_eligible_batches: int
    n_eval_fallback_batches: int
    fold_provenance: tuple = ()    # per-fold m/k/q
    oof_digest: str = ""           # C0 OOF routing record digest


def run_c0(disease, reg_or_art, batches, labels, alpha=0.10, delta=0.0, cache=None) -> C0Report:
    """C0 = the v2 recipe (per-action `acar.regressor.ActionRegressor`: HGB≥40 / Ridge≥8 / constant, **seed 0**), the
    bit-for-bit v2 11-D feature vector, actually TRAINED on FIT, one-sided conformal q on CAL subject scores, ROUTED on
    EVAL — over the identical splits/pool. FALLBACK batches are RETAINED (identity, ΔR 0, not adapted) in the red /
    coverage denominators; MAE/width/AUROC use eligible batches (predictor output required)."""
    registry = _as_registry(reg_or_art, disease)
    idx = _subject_batches(batches); eligible = _eligible_subjects(idx)
    if cache is None:
        cache = disease_exec_cache(registry, batches, labels)
    elig_canon = {canon_subject(s) for s in eligible}
    assignment, _ = cv_assignment([v["key"] for v in idx.values()], eligible=elig_canon)
    chosen_dr = []; n_adapt = 0; n_total = 0; n_fb = 0
    by_subj_ae = {}; centers = []; harm = []                     # MAE / AUROC accumulators (eligible only)
    by_subj_w = {}                                               # width: subject-macro (matches the candidate)
    fold_prov = []; oof_items = []
    for fa in assignment:
        Xy = {a: ([], []) for a in NON_IDENTITY}
        for s in fa["fit"]:
            for b in sorted(idx[canon_subject(s)]["eligible"], key=deployment_batch_digest):
                c = cache[deployment_batch_digest(b)]
                for a in NON_IDENTITY:
                    Xy[a][0].append(c["c0feat"][a]); Xy[a][1].append(c["dr"][a])
        regs = {a: _fit_action_regressor(np.array(Xy[a][0]), np.array(Xy[a][1])) for a in NON_IDENTITY}
        cal = []
        for s in fa["cal"]:
            slot = idx[canon_subject(s)]
            if not slot["eligible"]:
                continue
            smax = -math.inf
            for b in sorted(slot["eligible"], key=deployment_batch_digest):
                c = cache[deployment_batch_digest(b)]
                for a in NON_IDENTITY:
                    smax = max(smax, c["dr"][a] - float(regs[a].predict([c["c0feat"][a]])[0]))
            cal.append(smax)
        q, k = conformal_q(cal, alpha)
        for s in fa["eval"]:
            slot = idx[canon_subject(s)]
            for b in sorted(slot["eligible"], key=deployment_batch_digest):
                c = cache[deployment_batch_digest(b)]
                ghat = {a: float(regs[a].predict([c["c0feat"][a]])[0]) for a in NON_IDENTITY}
                U = {a: ghat[a] + q for a in NON_IDENTITY}
                for a in NON_IDENTITY:
                    by_subj_ae.setdefault(s, []).append(abs(ghat[a] - c["dr"][a]))
                    centers.append(ghat[a]); harm.append(1 if c["dr"][a] > 0 else 0)
                    by_subj_w.setdefault(s, []).append(U[a] - ghat[a])
                elig_acts = [a for a in NON_IDENTITY if U[a] < -float(delta)]
                ch = min(elig_acts, key=lambda a: U[a]) if elig_acts else "identity"
                chosen_dr.append(c["dr"][ch] if ch != "identity" else 0.0); n_adapt += int(ch != "identity")
                n_total += 1
                oof_items.append(json.dumps([fa["fold"], canon_subject(s), deployment_batch_digest(b),
                                             {a: repr(ghat[a]) for a in NON_IDENTITY}, repr(q), ch],
                                            separators=(",", ":")).encode())
            for _b in slot["fallback"]:                          # fallback retained: identity, ΔR 0, not adapted
                chosen_dr.append(0.0); n_total += 1; n_fb += 1
        fold_prov.append({"fold": fa["fold"], "m": len(cal), "k": int(k), "q_raw": float(q),
                          "q_used": (float(q) if math.isfinite(q) else None)})
    red = -float(np.mean(chosen_dr)) if chosen_dr else 0.0
    cov = (n_adapt / n_total) if n_total else 0.0
    mae = _subject_macro_mean(by_subj_ae)
    au = _auroc(centers, harm) if centers else float("nan")
    width = _subject_macro_mean(by_subj_w)                        # subject-macro (matches the candidate)
    return C0Report(disease, red, float(cov), mae, float(au), width, n_total - n_fb, n_fb,
                    tuple(fold_prov), _digest(b"C0OOF/1", oof_items))


def _fit_action_regressor(X, y):
    from acar.regressor import ActionRegressor
    return ActionRegressor(seed=C0_SEED).fit(X, y)              # v2 model seed 0 (NOT seed_es)


# ===================================================================================== per-candidate disease metrics
def _auroc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="stable"); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts); starts = csum - counts
    avg = (starts + csum + 1) / 2.0
    ranks = avg[inv]
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


@dataclass(frozen=True)
class CandidateReport:
    disease: str
    candidate: str
    s2_pass: bool
    dominance_pass: bool
    max_action_auroc: float       # max over actions of center-AUROC vs 1[ΔR>0]
    mae: float                    # subject-clustered MAE of center m_c vs ΔR (eligible)
    width: float                  # subject-macro mean of (U_a - center)
    adaptation_coverage: float    # n_adapt / (eligible + fallback) batches
    red_router: float             # -mean chosen ΔR over ALL EVAL batches (fallback contributes 0)
    any_q_inf: bool               # ANY fold's q was +inf (q_finite criterion fails closed if so)
    s2_detail: dict               # per-action S2 raw diagnostics (C2/C3) or {} (C1)
    dominance_shares: dict        # per-action fractional max_a shares
    per_action_auroc: dict        # per-action center-AUROC vs 1[ΔR>0]
    c2_floor: dict


def develop_candidate(disease, reg_or_art, batches, labels, candidate, alpha=0.10, delta=0.0, cache=None) -> tuple:
    """OOF pass for one candidate + its S2 admissibility & S4 input statistics. FALLBACK batches are RETAINED (identity,
    ΔR 0, not adapted) in the red/coverage denominators; MAE/width/AUROC use eligible batches. Returns (report, oof)."""
    oof = run_oof(disease, reg_or_art, batches, labels, candidate, alpha, delta, cache=cache)
    recs = oof.records
    dom = maxa_dominance(recs)
    if candidate == "C2":
        s2 = s2_c2_gate(recs); floor = c2_floor_from_oof(recs)
    elif candidate == "C3":
        s2 = s2_c3_gate(recs); floor = {}
    else:
        s2 = {"pass": True}; floor = {}                          # C1: only dominance applies
    per_action_auroc = {}
    for a, rs in _by_action(recs).items():
        per_action_auroc[a] = _auroc([r.point for r in rs], [1 if r.delta_r > 0 else 0 for r in rs])   # center-AUROC
    aurocs = [v for v in per_action_auroc.values() if not math.isnan(v)]
    max_au = max(aurocs) if aurocs else float("nan")
    by_subj_w = {}; by_subj_ae = {}
    for r in recs:
        by_subj_w.setdefault(r.subject, []).append(r.upper - r.point)
        by_subj_ae.setdefault(r.subject, []).append(abs(r.point - r.delta_r))         # |center - ΔR|
    width = _subject_macro_mean(by_subj_w)
    mae = _subject_macro_mean(by_subj_ae)
    # router red + coverage: per-batch chosen action; denominator INCLUDES fallback batches (identity, ΔR 0)
    by_batch = {}
    drmap = {(r.subject, r.batch_digest, r.action): r.delta_r for r in recs}
    for r in recs:
        by_batch.setdefault((r.subject, r.batch_digest), r.chosen)
    chosen_dr = []; n_adapt = 0
    for (subj, bd), ch in by_batch.items():
        if ch == "identity":
            chosen_dr.append(0.0)
        else:
            chosen_dr.append(drmap[(subj, bd, ch)]); n_adapt += 1
    chosen_dr += [0.0] * oof.n_eval_fallback_batches            # fallback retained -> identity, ΔR 0
    n_total = len(by_batch) + oof.n_eval_fallback_batches
    red = -float(np.mean(chosen_dr)) if chosen_dr else 0.0
    cov = (n_adapt / n_total) if n_total else 0.0
    any_q_inf = any(not math.isfinite(q) for q in oof.fold_qs) if oof.fold_qs else True   # ANY +inf fold fails q_finite
    s2_detail = {a: d for a, d in s2.items() if a != "pass"}
    rep = CandidateReport(disease, candidate, bool(s2["pass"]), bool(dom["ok"]), float(max_au), mae, width, float(cov),
                          float(red), bool(any_q_inf), s2_detail, dict(dom["shares"]), per_action_auroc, floor)
    return rep, oof


# ================================================================================================ S4 gate + select (pure)
def s4_eligible(m, *, auroc_gate=AUROC_GATE, width_reduction=WIDTH_REDUCTION, coverage_min=COVERAGE_MIN):
    """The FULL S4 pre-lock admissibility (every criterion gates SELECT). `m` is a candidate's cross-disease dict:
    s2_pass, dominance_pass, pd_auroc, scz_mae, c0_scz_mae, width_macro, c0_width_macro, coverage_macro, red_macro,
    c0_red_macro, any_q_inf. Returns dict(criteria, eligible)."""
    crit = {
        "s2": bool(m["s2_pass"]),
        "dominance": bool(m["dominance_pass"]),
        "pd_auroc": (not math.isnan(m["pd_auroc"])) and m["pd_auroc"] >= auroc_gate,
        "scz_mae_not_worse": m["scz_mae"] <= m["c0_scz_mae"] + 1e-12,
        "width_30pct_below_c0": math.isfinite(m["width_macro"]) and
        m["width_macro"] <= (1.0 - width_reduction) * m["c0_width_macro"],
        "coverage": m["coverage_macro"] >= coverage_min,
        "red_positive": m["red_macro"] > 0.0,
        "red_not_below_c0": m["red_macro"] >= m["c0_red_macro"] - 1e-12,
        "q_finite": not bool(m["any_q_inf"]),
    }
    return {"criteria": crit, "eligible": all(crit.values())}


def s4_select(per_candidate, *, tie_tol=1e-4, order=("C2", "C3", "C1")):
    """Among candidates with `eligible=True`: take the MAX disease-macro `red_macro`, form the tie set
    {c : max_red − red_macro[c] ≤ tie_tol} (transitive — relative to the true max), pick smallest `width_macro`, then
    fixed order C2 ≺ C3 ≺ C1. No eligible candidate → `DEV_STOP / NO_LOCKBOX_CONSUMED`."""
    passers = {c: m for c, m in per_candidate.items() if m.get("eligible")}
    if not passers:
        return {"verdict": "DEV_STOP", "reason": "NO_LOCKBOX_CONSUMED", "selected": None}
    max_red = max(m["red_macro"] for m in passers.values())
    tie = {c: m for c, m in passers.items() if max_red - m["red_macro"] <= tie_tol}     # relative to the TRUE max
    min_w = min(m["width_macro"] for m in tie.values())
    finalists = [c for c, m in tie.items() if m["width_macro"] <= min_w + 1e-12]
    best = min(finalists, key=lambda c: order.index(c) if c in order else len(order))   # C2 ≺ C3 ≺ C1
    return {"verdict": "SELECT", "selected": best, "reason": "passed S2+S4", "max_red": max_red, "tie": sorted(tie)}


# ===================================================================================================== full DEV run
@dataclass(frozen=True)
class DevelopResult:
    candidate_selected: str
    verdict: str
    alpha: float
    delta: float
    per_disease: dict           # disease -> {candidate -> CandidateReport, "C0": C0Report}
    final_epochs: dict          # disease -> int (for the selected candidate)
    refit_sha256: dict          # disease -> artifact sha (selected candidate) or None on DEV_STOP
    eligible_subject_list_sha256: dict
    pool_digest: dict
    n_fallback_only_subjects: dict
    s4_inputs: dict             # candidate -> cross-disease S4 metric dict (incl per-criterion eligibility)
    final_artifacts: dict       # disease -> FittedCandidateArtifact (selected) or None  (refit ONCE here)
    final_c0: dict              # disease -> {action: ActionRegressor} (refit ONCE) or None
    final_c0_probe: dict        # disease -> {action: feature} for reload checks, or None
    best_fixed: dict            # disease -> action maximizing DEV OOF red (frozen)
    provenance: dict            # disease -> field hashes + per-candidate/C0 per-fold provenance + source_state_sha256
    env_lock_sha256: str = ""   # set by run_binding_dev (verified runtime lock); "" for non-binding run_dev


def best_fixed_action(cache, idx, eligible):
    """best-fixed = the action maximizing DEV OOF red (= −mean ΔR_a over eligible batches), per disease. Frozen."""
    sums = {a: [] for a in NON_IDENTITY}
    for s in eligible:
        for b in idx[canon_subject(s)]["eligible"]:
            dr = cache[deployment_batch_digest(b)]["dr"]
            for a in NON_IDENTITY:
                sums[a].append(dr[a])
    red = {a: -float(np.mean(sums[a])) if sums[a] else float("-inf") for a in NON_IDENTITY}
    best = max(NON_IDENTITY, key=lambda a: (red[a], -NON_IDENTITY.index(a)))
    return best, red


def oof_record_digest(records) -> str:
    items = []
    for r in sorted(records, key=lambda r: (r.subject, r.batch_digest, r.action)):
        items.append(json.dumps([r.candidate, r.disease, r.subject, r.batch_digest, r.fold, r.action, repr(r.delta_r),
                                 repr(r.point), repr(r.upper_center), repr(r.scale_raw), repr(r.scale_used),
                                 repr(r.score), repr(r.q), repr(r.upper), r.chosen], separators=(",", ":")).encode())
    return _digest(b"OOF/1", items)


def _final_c0(cache, idx, eligible):
    """Refit the v2 C0 regressors ONCE on the full eligible pool (deterministic; seed 0). Returns (regs, probe)."""
    Xy = {a: ([], []) for a in NON_IDENTITY}; probe = None
    for s in eligible:
        for b in sorted(idx[canon_subject(s)]["eligible"], key=deployment_batch_digest):
            c = cache[deployment_batch_digest(b)]
            for a in NON_IDENTITY:
                Xy[a][0].append(c["c0feat"][a]); Xy[a][1].append(c["dr"][a])
            if probe is None:
                probe = {a: np.asarray(c["c0feat"][a], float) for a in NON_IDENTITY}
    regs = {a: _fit_action_regressor(np.array(Xy[a][0]), np.array(Xy[a][1])) for a in NON_IDENTITY}
    return regs, probe


def _field_hashes(registry, batches, labels, eligible):
    from .loader import hash_deployment_input, hash_labels, hash_subject_list as _hsl
    return {"deployment_input_sha256": hash_deployment_input(batches), "label_sha256": hash_labels(labels),
            "subject_list_sha256": _hsl(eligible), "source_state_refs": list(registry.refs),
            "pool_digest": pool_digest(batches)}


def s4_metrics(reports, candidate, diseases):
    """Assemble a candidate's cross-disease S4 metric dict from per-disease CandidateReports + C0. width/red are
    disease-MACRO; PD AUROC is PD-specific; SCZ MAE is SCZ-specific (vs C0). C0 macro red/width too."""
    rc = {d: reports[d][candidate] for d in diseases}; c0 = {d: reports[d]["C0"] for d in diseases}
    macro = lambda f: float(np.mean([f(d) for d in diseases]))
    pd_au = rc["PD"].max_action_auroc if "PD" in rc else float("nan")
    scz_mae = rc["SCZ"].mae if "SCZ" in rc else float("inf")
    c0_scz_mae = c0["SCZ"].mae if "SCZ" in c0 else float("inf")
    return {
        "s2_pass": all(rc[d].s2_pass for d in diseases),
        "dominance_pass": all(rc[d].dominance_pass for d in diseases),
        "pd_auroc": pd_au, "scz_mae": scz_mae, "c0_scz_mae": c0_scz_mae,
        "width_macro": macro(lambda d: rc[d].width), "c0_width_macro": macro(lambda d: c0[d].width),
        "coverage_macro": macro(lambda d: rc[d].adaptation_coverage),
        "red_macro": macro(lambda d: rc[d].red_router), "c0_red_macro": macro(lambda d: c0[d].red_router),
        "any_q_inf": any(rc[d].any_q_inf for d in diseases),
    }


def run_dev(diseases_data, candidates=("C1", "C2", "C3"), alpha=0.10, delta=0.0, env_lock_sha256="") -> DevelopResult:
    """diseases_data: {disease: (registry_or_artifact, batches, labels)}. Bake-off on EACH disease, real C0/v2 replay
    over the identical pool, the FULL S4 admissibility (`s4_eligible`) + disease-macro SELECT, and (if SELECT) the final
    per-disease refit on the frozen eligible pool. SYNTHETIC orchestration — no real DEV value, no binding G2 verdict."""
    diseases = list(diseases_data)
    reports = {d: {} for d in diseases}; oofs = {d: {} for d in diseases}
    elig_sha = {}; pools = {}; nfb = {}; caches = {}; prov = {}; bestfix = {}
    for d, (reg, batches, labels) in diseases_data.items():
        registry = _as_registry(reg, d)
        idx = _subject_batches(batches); eligible = _eligible_subjects(idx)
        elig_sha[d] = hash_subject_list(eligible); pools[d] = pool_digest(batches)
        nfb[d] = sum(1 for v in idx.values() if not v["eligible"])
        caches[d] = disease_exec_cache(registry, batches, labels)
        for c in candidates:
            reports[d][c], oofs[d][c] = develop_candidate(d, registry, batches, labels, c, alpha, delta, cache=caches[d])
        reports[d]["C0"] = run_c0(d, registry, batches, labels, alpha, delta, cache=caches[d])
        assert pools[d] == replay_pool_digest(batches)
        bestfix[d], red_by_a = best_fixed_action(caches[d], idx, eligible)
        ss_sha = {art.source_state_ref: art.source_state_sha256 for art in registry._by_ref.values()}
        prov[d] = {"field_hashes": _field_hashes(registry, batches, labels, eligible), "best_fixed_red": red_by_a,
                   "source_state_sha256": ss_sha,
                   "c0": {"fold_provenance": list(reports[d]["C0"].fold_provenance),
                          "oof_digest": reports[d]["C0"].oof_digest},
                   "per_candidate": {c: {"fold_qs": list(oofs[d][c].fold_qs),
                                         "n_cal_scores_per_fold": list(oofs[d][c].n_cal_scores_per_fold),
                                         "fold_provenance": list(oofs[d][c].fold_provenance),
                                         "oof_digest": oof_record_digest(oofs[d][c].records)} for c in candidates}}
    per_cand = {}
    for c in candidates:
        m = s4_metrics(reports, c, diseases); m["eligibility"] = s4_eligible(m); m["eligible"] = m["eligibility"]["eligible"]
        per_cand[c] = m
    sel = s4_select(per_cand)
    final_ep = {}; refit_sha = {}; final_art = {}; final_c0 = {}; final_probe = {}
    if sel["verdict"] == "SELECT":
        c = sel["selected"]
        for d in diseases:
            registry = _as_registry(diseases_data[d][0], d)
            idx = _subject_batches(diseases_data[d][1]); eligible = _eligible_subjects(idx)
            fe = final_epochs(list(oofs[d][c].best_epochs)); final_ep[d] = fe
            all_ex = _train_examples(eligible, idx, caches[d])
            floor = reports[d][c].c2_floor if c == "C2" else {}
            art = refit_candidate_fixed_epochs(c, d, all_ex, fe, floor, HP["seed_es"])    # final refit — EXACTLY ONCE
            final_art[d] = art; refit_sha[d] = art.artifact_sha256
            final_c0[d], final_probe[d] = _final_c0(caches[d], idx, eligible)             # final C0 — EXACTLY ONCE
    else:
        for d in diseases:
            final_ep[d] = 0; refit_sha[d] = None; final_art[d] = None; final_c0[d] = None; final_probe[d] = None
    return DevelopResult(sel.get("selected"), sel["verdict"], float(alpha), float(delta), reports, final_ep, refit_sha,
                         elig_sha, pools, nfb, per_cand, final_art, final_c0, final_probe, bestfix, prov,
                         str(env_lock_sha256))


def replay_pool_digest(batches) -> str:
    """The C0/v2 replay MUST consume the identical pool (asserted in run_dev)."""
    return pool_digest(batches)


# ===================================================================================== binding entrypoint + frozen runner
def _group_cohorts(cohort_inputs):
    """Validate a list of CohortInput against the frozen seven cohorts and return {disease: (registry, batches, labels,
    [CohortInput])}. FAIL-CLOSED: exactly the config.DISEASE dataset IDs, one CohortInput (and one source-state ref) per
    dataset, no extras/missing — so two cohorts' source states cannot be swapped undetected."""
    from acar.config import DISEASE
    from .loader import CohortInput, SourceStateRegistry
    by_d = {}
    for ci in cohort_inputs:
        if not isinstance(ci, CohortInput):
            raise TypeError("binding DEV requires CohortInput objects")
        by_d.setdefault(ci.disease, []).append(ci)
    if set(by_d) != {"PD", "SCZ"}:
        raise ValueError("binding DEV requires both diseases {PD, SCZ}")
    out = {}
    for d, cis in by_d.items():
        ds_ids = [ci.dataset_id for ci in cis]
        if sorted(ds_ids) != sorted(DISEASE[d]):
            raise ValueError(f"{d} cohorts {sorted(ds_ids)} != frozen {sorted(DISEASE[d])}")
        if len(set(ds_ids)) != len(ds_ids):
            raise ValueError(f"{d}: duplicate cohort dataset_id")
        reg = SourceStateRegistry(d); batches = []; labels = {}
        for ci in cis:
            reg.add(ci.source_artifact)                        # rejects duplicate refs (one ref per cohort)
            batches += list(ci.batches); labels.update(ci.labels)
        if len(reg.refs) != len(DISEASE[d]):
            raise ValueError(f"{d}: registry has {len(reg.refs)} refs, expected {len(DISEASE[d])}")
        out[d] = (reg, batches, labels, cis)
    return out


def run_binding_dev(cohort_inputs, candidates=("C1", "C2", "C3"), alpha=0.10, delta=0.0) -> DevelopResult:
    """The binding DEV entrypoint. Input is the list of immutable CohortInput objects (NOT loose registries). FAIL-CLOSED
    on any deviation: candidates (C1,C2,C3), α=0.10, δ=0.0, the EXACT seven cohorts (one source-state ref each, dataset↔
    source bound inside CohortInput), the applied+verified runtime lock. There is NO `verify_env` bypass — synthetic
    orchestration tests must call the non-binding `run_dev` instead."""
    from .envlock import apply_runtime, verify_env_lock
    if tuple(candidates) != ("C1", "C2", "C3"):
        raise ValueError("binding DEV requires candidates (C1, C2, C3)")
    if float(alpha) != 0.10 or float(delta) != 0.0:
        raise ValueError("binding DEV requires alpha=0.10, delta=0.0")
    grouped = _group_cohorts(cohort_inputs)
    apply_runtime()                                             # set the locked deterministic runtime, THEN verify it
    env_sha = verify_env_lock()
    diseases_data = {d: (reg, batches, labels) for d, (reg, batches, labels, _cis) in grouped.items()}
    return run_dev(diseases_data, candidates, alpha, delta, env_lock_sha256=env_sha)


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _report_json(rep):
    return {"s2_pass": rep.s2_pass, "dominance_pass": rep.dominance_pass, "max_action_auroc": rep.max_action_auroc,
            "mae": rep.mae, "width": rep.width, "adaptation_coverage": rep.adaptation_coverage,
            "red_router": rep.red_router, "any_q_inf": rep.any_q_inf, "s2_detail": rep.s2_detail,
            "dominance_shares": rep.dominance_shares, "per_action_auroc": rep.per_action_auroc, "c2_floor": rep.c2_floor}


def _cohort_json(ci):
    m = ci.manifest
    return {"dataset_id": ci.dataset_id, "disease": ci.disease, "full_dump_path": ci.full_dump_path,
            "full_dump_sha256": m.full_dump_sha256, "source_fit_sha256": m.source_fit_sha256,
            "deployment_input_sha256": m.deployment_input_sha256, "label_sha256": m.label_sha256,
            "subject_list_sha256": m.subject_list_sha256, "n_subjects": m.n_subjects, "n_recordings": m.n_recordings,
            "n_windows": m.n_windows, "embedding_dim": m.embedding_dim, "schema_version": m.schema_version,
            "source_state_ref": ci.source_artifact.source_state_ref,
            "source_state_sha256": ci.source_artifact.source_state_sha256}


def _json_safe(o):
    """Recursively coerce to JSON-standard types. NO silent default=str; non-finite floats -> NOT_EVALUABLE sentinel."""
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, bool) or isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else {"value": None, "status": "NOT_EVALUABLE"}
    if o is None or isinstance(o, str):
        return o
    raise TypeError(f"non-JSON-safe object of type {type(o).__name__}")


def _write_json(path, obj):
    core = _json_safe(obj)
    blob = json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    core["manifest_sha256"] = hashlib.sha256(blob).hexdigest()
    with open(path, "w") as f:
        json.dump(core, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    return core["manifest_sha256"]


def freeze_dev_run(cohort_inputs, outdir, *, candidates=("C1", "C2", "C3"), alpha=0.10, delta=0.0,
                   protocol_commit=None, binding_command=None):
    """Run the binding DEV gate on the seven CohortInputs and FREEZE the outcome to a NON-OVERWRITABLE directory via an
    ATOMIC write (build in `<outdir>.tmp`, `os.rename` only on full success; temp removed on failure). This layer ONLY
    serializes the predictor + C0 the run produced (refit EXACTLY ONCE inside the run) — it never re-executes adapters
    or retrains. The S5/S6/S8/S9 manifest carries the env-lock hash, per-cohort LoadedDumpManifests, field-separated
    hashes, per-fold FIT/CAL/EVAL hashes+counts+m/k/q, OOF digests, C2 σ_min, best-fixed per disease, per-candidate
    diagnostics + S4 eligibility, source_state_sha256, predictor+C0 file SHA-256, protocol commit/tag/command, and its
    OWN manifest_sha256 (JSON is allow_nan=False; non-finite -> NOT_EVALUABLE; no silent str-coercion). On DEV_STOP a
    `DEV_STOP / NO_LOCKBOX_CONSUMED` marker is written and no artifacts."""
    if os.path.exists(outdir):
        raise FileExistsError(f"refusing to overwrite existing DEV output dir {outdir}")
    grouped = _group_cohorts(cohort_inputs)                     # validate cohorts before any run
    res = run_binding_dev(cohort_inputs, candidates, alpha, delta)
    diseases = list(grouped)
    tmpdir = outdir + ".tmp"
    if os.path.exists(tmpdir):
        raise FileExistsError(f"stale temp dir {tmpdir}")
    os.makedirs(tmpdir)
    manifest = {"verdict": res.verdict, "selected": res.candidate_selected, "alpha": res.alpha, "delta": res.delta,
                "env_lock_sha256": res.env_lock_sha256, "pool_digest": res.pool_digest,
                "eligible_subject_list_sha256": res.eligible_subject_list_sha256,
                "n_fallback_only_subjects": res.n_fallback_only_subjects, "final_epochs": res.final_epochs,
                "best_fixed": res.best_fixed, "provenance": res.provenance, "s4_inputs": res.s4_inputs,
                "candidate_reports": {d: {c: _report_json(res.per_disease[d][c]) for c in candidates} for d in diseases},
                "cohorts": {d: [_cohort_json(ci) for ci in grouped[d][3]] for d in diseases},
                "protocol": {"commit": protocol_commit, "tag": "acar-v3-dev-design-v1",
                             "binding_command": binding_command, "output_path": outdir}}
    try:
        if res.verdict != "SELECT":
            manifest["reason"] = "NO_LOCKBOX_CONSUMED"
            _write_json(os.path.join(tmpdir, "DEV_STOP.json"), manifest)
        else:
            saved = {}
            for d in diseases:
                art = res.final_artifacts[d]                       # the SAME artifact the run produced (no re-refit)
                ppath = os.path.join(tmpdir, f"predictor_{d}.pkl")
                with open(ppath, "wb") as f:
                    pickle.dump(art, f)
                with open(ppath, "rb") as f:
                    reloaded = pickle.load(f)
                reloaded.verify_integrity()                        # cryptographic reload check
                if reloaded.artifact_sha256 != art.artifact_sha256:
                    raise ValueError("saved predictor does not reload with an identical hash")
                if art.artifact_sha256 != res.refit_sha256[d]:
                    raise ValueError("frozen predictor hash != the run's recorded refit hash")
                c0regs = res.final_c0[d]; probe = res.final_c0_probe[d]
                c0path = os.path.join(tmpdir, f"c0_{d}.pkl")
                with open(c0path, "wb") as f:
                    pickle.dump(c0regs, f)
                with open(c0path, "rb") as f:
                    c0re = pickle.load(f)
                for a in NON_IDENTITY:
                    if float(c0re[a].predict([probe[a]])[0]) != float(c0regs[a].predict([probe[a]])[0]):
                        raise ValueError("saved C0 regressor does not reload identically")
                saved[d] = {"predictor_sha256": art.artifact_sha256, "predictor_file_sha256": _file_sha256(ppath),
                            "predictor_path": os.path.join(outdir, f"predictor_{d}.pkl"),
                            "c0_file_sha256": _file_sha256(c0path), "c0_path": os.path.join(outdir, f"c0_{d}.pkl"),
                            "eligible_subject_list_sha256": res.eligible_subject_list_sha256[d]}
            manifest["saved"] = saved
            _write_json(os.path.join(tmpdir, "manifest.json"), manifest)
        os.rename(tmpdir, outdir)                                  # ATOMIC: appear only on full success
    except BaseException:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)                  # no half-written non-overwritable dir on failure
        raise
    return res, manifest
