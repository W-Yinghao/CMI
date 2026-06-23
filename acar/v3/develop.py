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
import math

import numpy as np

from .set_features import NON_IDENTITY
from .data import deployment_batch_digest, canon_subject
from .conformal import subject_joint_score, conformal_q, route
from .predictors import score as cand_score, upper_bound, HP
from .training import fit_candidate_earlystop, refit_candidate_fixed_epochs, final_epochs
from .splits import cv_assignment
from .loader import hash_subject_list, _digest, SourceStateRegistry, SourceStateArtifact

Z90 = 1.2815515594463558                                   # standard-normal 0.90 quantile


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
def disease_exec_cache(registry, batches, labels):
    """Execute every ELIGIBLE batch EXACTLY ONCE (the adapters are the expensive, candidate-independent step) and cache
    the captured WindowActionSets + ΔR + execution hashes. Keyed by deployment_batch_digest. Reused across all folds,
    candidates, and the C0 replay — so the whole DEV bake-off triggers each batch's single execution only once."""
    cache = {}
    for b in batches:
        if b.fallback:
            continue
        sa = registry.resolve(b)
        exe = sa.execute(b)
        cache[deployment_batch_digest(b)] = {
            "exe": exe, "was": exe.window_action_sets(sa),
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
    records = []; best_epochs = []; fold_qs = []; n_elig = n_fb = 0; eval_subj = set(); n_cal = []
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
        q, _k = conformal_q(cal_scores, alpha); fold_qs.append(float(q)); n_cal.append(len(cal_scores))
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
                        n_elig, n_fb, len(eval_subj), tuple(n_cal))


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
    """C2: per-action subject-balanced standardized-residual mean∈[-0.25,0.25], variance∈[0.5,2.0], positive-tail 90th
    pct ∈[0.8,2.0]·z₀.₉₀. FAIL-CLOSED. Returns dict(action -> stats) + overall pass."""
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
        allr = np.concatenate([np.asarray(x) for x in by_s.values()])
        tail = float(np.quantile(allr, 0.90))
        a_ok = (abs(m) <= 0.25) and (0.5 <= v <= 2.0) and (0.8 * Z90 <= tail <= 2.0 * Z90)
        res[a] = dict(mean=m, var=v, tail90=tail, ok=bool(a_ok)); ok = ok and a_ok
    res["pass"] = bool(ok)
    return res


def s2_c3_gate(records):
    """C3: per-action exceedance P(ΔR>q̂₉₀)∈[0.05,0.20]; positive-excess 95th pct ≤ 2·(OOF ΔR SD); q̂₉₀>q̂₅₀ everywhere."""
    res = {}; ok = True
    for a, rs in _by_action(records).items():
        if not rs:
            raise ValueError(f"C3 S2: no OOF records for action {a}")
        dr = np.array([r.delta_r for r in rs]); q90 = np.array([r.upper_center for r in rs])
        q50 = np.array([r.point for r in rs])
        exc = float(np.mean(dr > q90))
        excess95 = float(np.quantile(np.maximum(dr - q90, 0.0), 0.95))
        sd = float(np.std(dr))
        crossing_ok = bool(np.all(q90 > q50))
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
def _c0_feature(was):
    """Batch-summary φ_a(B) for the v2 ActionRegressor: masked column-means of the per-window features ⊕ context."""
    vals = np.asarray(was.values, float); mask = np.asarray(was.availability_mask, float)
    denom = np.clip(mask.sum(0), 1.0, None)
    col_mean = (vals * mask).sum(0) / denom
    return np.concatenate([col_mean, np.asarray(was.context_values, float)])


@dataclass(frozen=True)
class C0Report:
    disease: str
    red_router: float
    adaptation_coverage: float
    n_eval_eligible_batches: int


def run_c0(disease, reg_or_art, batches, labels, alpha=0.10, delta=0.0, cache=None) -> C0Report:
    """C0 = the v2 recipe (per-action ActionRegressor: HGB≥40 / Ridge≥8 / constant), actually TRAINED on FIT,
    one-sided conformal q on CAL subject scores, ROUTED on EVAL — over the identical splits/pool (cached executions)."""
    from acar.regressor import ActionRegressor
    registry = _as_registry(reg_or_art, disease)
    idx = _subject_batches(batches); eligible = _eligible_subjects(idx)
    if cache is None:
        cache = disease_exec_cache(registry, batches, labels)
    elig_canon = {canon_subject(s) for s in eligible}
    assignment, _ = cv_assignment([v["key"] for v in idx.values()], eligible=elig_canon)
    chosen_dr = []; n_adapt = 0; n_elig = 0

    def feat_dr(b):
        c = cache[deployment_batch_digest(b)]
        return {a: _c0_feature(c["was"][a]) for a in NON_IDENTITY}, c["dr"]
    for fa in assignment:
        Xy = {a: ([], []) for a in NON_IDENTITY}
        for s in fa["fit"]:
            for b in sorted(idx[canon_subject(s)]["eligible"], key=deployment_batch_digest):
                feat, dr = feat_dr(b)
                for a in NON_IDENTITY:
                    Xy[a][0].append(feat[a]); Xy[a][1].append(dr[a])
        regs = {a: ActionRegressor(seed=HP["seed_es"]).fit(np.array(Xy[a][0]), np.array(Xy[a][1])) for a in NON_IDENTITY}
        cal = []
        for s in fa["cal"]:
            slot = idx[canon_subject(s)]
            if not slot["eligible"]:
                continue
            smax = -math.inf
            for b in sorted(slot["eligible"], key=deployment_batch_digest):
                feat, dr = feat_dr(b)
                for a in NON_IDENTITY:
                    smax = max(smax, dr[a] - float(regs[a].predict([feat[a]])[0]))
            cal.append(smax)
        q, _k = conformal_q(cal, alpha)
        for s in fa["eval"]:
            for b in sorted(idx[canon_subject(s)]["eligible"], key=deployment_batch_digest):
                feat, dr = feat_dr(b)
                U = {a: float(regs[a].predict([feat[a]])[0]) + q for a in NON_IDENTITY}
                elig_acts = [a for a in NON_IDENTITY if U[a] < -float(delta)]
                if elig_acts:
                    ch = min(elig_acts, key=lambda a: U[a]); chosen_dr.append(dr[ch]); n_adapt += 1
                else:
                    chosen_dr.append(0.0)
                n_elig += 1
    red = -float(np.mean(chosen_dr)) if chosen_dr else 0.0       # reduction vs identity (0); >0 is good
    cov = (n_adapt / n_elig) if n_elig else 0.0
    return C0Report(disease, red, float(cov), n_elig)


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
    max_action_auroc: float
    width: float
    adaptation_coverage: float
    red_router: float
    selectable: bool
    c2_floor: dict


def develop_candidate(disease, reg_or_art, batches, labels, candidate, alpha=0.10, delta=0.0, cache=None) -> tuple:
    """Run the OOF pass for one candidate and compute its S2 admissibility + S4 input metrics. Returns (report, oof)."""
    oof = run_oof(disease, reg_or_art, batches, labels, candidate, alpha, delta, cache=cache)
    recs = oof.records
    dom = maxa_dominance(recs)
    if candidate == "C2":
        s2 = s2_c2_gate(recs); floor = c2_floor_from_oof(recs)
    elif candidate == "C3":
        s2 = s2_c3_gate(recs); floor = {}
    else:
        s2 = {"pass": True}; floor = {}                          # C1 has no scale/quantile gate, only dominance
    # per-action OOF harm AUROC (score vs 1[ΔR>0]); width; coverage; red_router
    aurocs = []
    for a, rs in _by_action(recs).items():
        au = _auroc([r.upper for r in rs], [1 if r.delta_r > 0 else 0 for r in rs])
        if not math.isnan(au):
            aurocs.append(au)
    max_au = max(aurocs) if aurocs else float("nan")
    # width W_c (subject-macro mean of U_a - center); center = point (C1/C2) or q50=point (C3 point is q50)
    by_subj_w = {}
    for r in recs:
        by_subj_w.setdefault(r.subject, []).append(r.upper - r.point)
    width = float(np.mean([np.mean(v) for v in by_subj_w.values()]))
    # router red + coverage from the per-batch chosen action
    by_batch = {}
    for r in recs:
        by_batch.setdefault((r.subject, r.batch_digest), r.chosen)  # chosen identical across actions of a batch
    chosen_dr = []
    drmap = {(r.subject, r.batch_digest, r.action): r.delta_r for r in recs}
    n_adapt = 0
    for (subj, bd), ch in by_batch.items():
        if ch == "identity":
            chosen_dr.append(0.0)
        else:
            chosen_dr.append(drmap[(subj, bd, ch)]); n_adapt += 1
    red = -float(np.mean(chosen_dr)) if chosen_dr else 0.0
    cov = n_adapt / len(by_batch) if by_batch else 0.0
    selectable = bool(s2["pass"] and dom["ok"])                  # C1 selectable ONLY if dominance passes
    rep = CandidateReport(disease, candidate, bool(s2["pass"]), bool(dom["ok"]), float(max_au), width, float(cov),
                          float(red), selectable, floor)
    return rep, oof


# ================================================================================================ S4 select (pure)
def s4_select(per_candidate, *, tie_tol=1e-4, order=("C2", "C3", "C1")):
    """SELECT = max disease-macro OOF router NLL reduction among candidates passing all DEV pre-lock criteria; tie
    (|Δred|≤tie_tol) → smaller disease-macro width; residual tie → fixed order C2 ≺ C3 ≺ C1. `per_candidate` maps
    candidate -> dict(passes:bool, red_macro:float, width_macro:float). Returns dict(selected | DEV_STOP)."""
    passers = {c: m for c, m in per_candidate.items() if m["passes"]}
    if not passers:
        return {"verdict": "DEV_STOP", "reason": "NO_LOCKBOX_CONSUMED", "selected": None}
    best = None
    for c in sorted(passers, key=lambda x: order.index(x) if x in order else len(order)):
        m = passers[c]
        if best is None:
            best = c; continue
        bm = passers[best]
        if m["red_macro"] > bm["red_macro"] + tie_tol:
            best = c
        elif abs(m["red_macro"] - bm["red_macro"]) <= tie_tol:
            if m["width_macro"] < bm["width_macro"] - 1e-12:
                best = c
            # residual tie: keep earlier in fixed order (already iterating in order) -> no change
    return {"verdict": "SELECT", "selected": best, "reason": "passed S2+S4"}


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


def run_dev(diseases_data, candidates=("C1", "C2", "C3"), alpha=0.10, delta=0.0) -> DevelopResult:
    """diseases_data: {disease: (registry_or_artifact, batches, labels)}. Runs the bake-off on EACH disease, the C0
    replay over the identical pool, the disease-MACRO S4 selection, and (if a candidate is selected) the final per-
    disease refit on the frozen eligible pool. SYNTHETIC orchestration — no real DEV value, no binding verdict."""
    reports = {d: {} for d in diseases_data}; oofs = {d: {} for d in diseases_data}
    elig_sha = {}; pools = {}; nfb = {}
    caches = {}
    for d, (reg, batches, labels) in diseases_data.items():
        registry = _as_registry(reg, d)
        idx = _subject_batches(batches); eligible = _eligible_subjects(idx)
        elig_sha[d] = hash_subject_list(eligible); pools[d] = pool_digest(batches)
        nfb[d] = sum(1 for v in idx.values() if not v["eligible"])
        caches[d] = disease_exec_cache(registry, batches, labels)   # execute each batch ONCE for the whole disease
        for c in candidates:
            reports[d][c], oofs[d][c] = develop_candidate(d, registry, batches, labels, c, alpha, delta, cache=caches[d])
        reports[d]["C0"] = run_c0(d, registry, batches, labels, alpha, delta, cache=caches[d])
        assert pools[d] == replay_pool_digest(batches)         # C0 consumes the identical pool
    # disease-macro S4 inputs (a candidate must be selectable in EVERY disease)
    per_cand = {}
    for c in candidates:
        passes = all(reports[d][c].selectable for d in diseases_data)
        red_macro = float(np.mean([reports[d][c].red_router for d in diseases_data]))
        width_macro = float(np.mean([reports[d][c].width for d in diseases_data]))
        per_cand[c] = {"passes": passes, "red_macro": red_macro, "width_macro": width_macro}
    sel = s4_select(per_cand)
    final_ep = {}; refit_sha = {}
    if sel["verdict"] == "SELECT":
        c = sel["selected"]
        for d, (reg, batches, labels) in diseases_data.items():
            idx = _subject_batches(batches); eligible = _eligible_subjects(idx)
            fe = final_epochs(list(oofs[d][c].best_epochs)); final_ep[d] = fe
            all_ex = _train_examples(eligible, idx, caches[d])
            floor = reports[d][c].c2_floor if c == "C2" else {}
            art = refit_candidate_fixed_epochs(c, d, all_ex, fe, floor, HP["seed_es"])
            refit_sha[d] = art.artifact_sha256
    else:
        for d in diseases_data:
            final_ep[d] = 0; refit_sha[d] = None
    return DevelopResult(sel.get("selected"), sel["verdict"], float(alpha), float(delta), reports, final_ep, refit_sha,
                         elig_sha, pools, nfb)


def replay_pool_digest(batches) -> str:
    """The C0/v2 replay MUST consume the identical pool (asserted in run_dev)."""
    return pool_digest(batches)
