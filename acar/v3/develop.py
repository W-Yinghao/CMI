"""ACAR v3 DEV orchestration (S5 split-as-one-algorithm → train-once / calibrate / select / refit). DESIGN/DEV stage —
SYNTHETIC FIXTURES ONLY until DEV_DESIGN_LOCK; this module reads NO real DEV cohort values and produces NO binding
go/no-go. It wires the leak-respecting pipeline so it can be exercised deterministically end to end:

    per outer fold:  FIT subjects → predictor (early-stop on FIT's TRAIN/VAL)   [predictor sees FIT only]
                     CAL subjects → exactly ONE joint nonconformity score each → conformal q   [q sees CAL only]
                     EVAL subjects → route(preds, q, δ); fallback batches forced to identity    [diagnostics OOF only]
    then:            final_epochs = round_half_up(median_k(best_epoch_k+1));  OOF σ_min (C2);
                     refit ONCE on all ELIGIBLE subjects (eligibility = frozen split/inclusion ONLY, never residuals).

Each batch is executed EXACTLY once (SourceStateArtifact.execute) and both its predictor features and its ΔR are
derived from that one execution (loader binding). Fallback (<MIN_BATCH) subjects/batches are RETAINED for EVAL
accounting but never enter FIT/CAL fitting or a CAL score.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib

import numpy as np

from .set_features import NON_IDENTITY
from .data import deployment_batch_digest, canon_subject
from .conformal import subject_joint_score, conformal_q, route
from .training import fit_candidate_earlystop, refit_candidate_fixed_epochs, final_epochs
from .predictors import HP
from .splits import cv_assignment
from .loader import hash_subject_list, _digest


# --------------------------------------------------------------------------------------------- pool / subject indexing
def _subject_batches(batches):
    """{SubjectKey: {'eligible': [...], 'fallback': [...]}} keyed by canon for dedup; preserves SubjectKey objects."""
    idx = {}
    for b in batches:
        c = canon_subject(b.subject)
        slot = idx.setdefault(c, {"key": b.subject, "eligible": [], "fallback": []})
        (slot["fallback"] if b.fallback else slot["eligible"]).append(b)
    return idx


def _eligible_subjects(idx):
    """Subjects with >=1 non-fallback batch — the ONLY ones a predictor / CAL score / refit may use. Frozen inclusion
    rule; never residual-based."""
    return [v["key"] for v in idx.values() if v["eligible"]]


def pool_digest(batches) -> str:
    """Order-insensitive digest of the batch pool (the SAME pool the neural path and the v2/C0 replay must consume)."""
    return _digest(b"POOL/1", sorted(deployment_batch_digest(b).encode() for b in batches))


# ------------------------------------------------------------------------------------------------- one-execution preds
def _batch_preds_and_dr(artifact, source_artifact, batch, labels):
    """ONE execution -> (predictions per action, ΔR per action). Both derive from the same captured outputs."""
    exe = source_artifact.execute(batch)
    sets = exe.window_action_sets(source_artifact)
    preds = {a: artifact.predict(sets[a]) for a in NON_IDENTITY}
    dr = dict(exe.labeled_risk_record(labels).delta_r_by_action)
    return preds, dr


def _cal_score(artifact, source_artifact, subj_eligible, labels):
    """EXACTLY ONE joint nonconformity score for a CAL subject (max over its eligible batches × actions)."""
    subject_batches = []
    for b in subj_eligible:
        preds, dr = _batch_preds_and_dr(artifact, source_artifact, b, labels)
        subject_batches.append({a: (preds[a], dr[a]) for a in NON_IDENTITY})
    return subject_joint_score(subject_batches)


# ------------------------------------------------------------------------------------------------------- result types
@dataclass(frozen=True)
class FoldRun:
    fold: int
    candidate: str
    fit_subjects: tuple
    cal_subjects: tuple
    eval_subjects: tuple
    train_subjects: tuple
    val_subjects: tuple
    best_epoch: int
    q: float
    n_cal_scores: int
    n_eval_eligible_batches: int
    n_eval_fallback_batches: int
    oof_router_delta: float       # Σ chosen-action ΔR over EVAL eligible batches (identity contributes 0) — diagnostic


@dataclass(frozen=True)
class DevelopResult:
    disease: str
    candidate: str
    alpha: float
    delta: float
    folds: tuple
    final_epochs: int
    refit_artifact_sha256: str
    eligible_subject_list_sha256: str
    pool_digest: str
    n_eligible_subjects: int
    n_fallback_only_subjects: int     # retained for deployment/EVAL accounting; NEVER in FIT/CAL/predictor/refit


# ------------------------------------------------------------------------------------------------------ OOF + refit
def _train_examples(source_artifact, subjects, idx, labels):
    """TrainExamples from the given subjects' ELIGIBLE batches (single-execution DeploymentFeatureRecords). CANONICAL
    order — subjects by canon, batches by deployment_batch_digest — so the example sequence (and thus every downstream
    reduction, incl. floating-point sums) is independent of input batch order."""
    out = []
    want = {canon_subject(s) for s in subjects}
    for c in sorted(want):
        slot = idx.get(c)
        if slot is None:
            continue
        for b in sorted(slot["eligible"], key=deployment_batch_digest):
            exe = source_artifact.execute(b)
            out += exe.deployment_feature_record(source_artifact, b, labels).to_train_examples()
    return out


def run_develop(disease, source_artifact, batches, labels, candidate="C1", alpha=0.10, delta=0.0):
    """Full synthetic DEV orchestration for ONE disease + ONE candidate. Returns a DevelopResult; raises fail-closed on
    a degenerate fold (e.g. a candidate that never improves)."""
    if candidate not in ("C1", "C2", "C3"):
        raise ValueError("bad candidate")
    idx = _subject_batches(batches)
    eligible = _eligible_subjects(idx)
    n_fb_only = sum(1 for v in idx.values() if not v["eligible"])     # fallback-only subjects (deployment/EVAL only)
    if len(eligible) < HP["k_folds"]:
        raise ValueError(f"need >= k_folds={HP['k_folds']} eligible subjects; got {len(eligible)}")
    assignment, _all = cv_assignment(eligible)        # split over ELIGIBLE subjects (fallback-only never fitted/CAL'd)
    fold_runs = []
    for fa in assignment:
        tr_ex = _train_examples(source_artifact, fa["train"], idx, labels)
        va_ex = _train_examples(source_artifact, fa["val"], idx, labels)
        artifact, best_epoch = fit_candidate_earlystop(candidate, disease, tr_ex, va_ex, HP["seed_es"])
        # CAL: exactly one score per CAL subject that has an eligible batch
        cal_scores = []
        for s in fa["cal"]:
            slot = idx[canon_subject(s)]
            if slot["eligible"]:
                cal_scores.append(_cal_score(artifact, source_artifact, slot["eligible"], labels))
        q, _k = conformal_q(cal_scores, alpha)
        # EVAL: route eligible batches; fallback batches retained as identity (ΔR contribution 0)
        n_elig = n_fb = 0; router_delta = 0.0
        for s in fa["eval"]:
            slot = idx[canon_subject(s)]
            for b in slot["eligible"]:
                preds, dr = _batch_preds_and_dr(artifact, source_artifact, b, labels)
                chosen, _U = route(preds, q, delta)
                router_delta += 0.0 if chosen == "identity" else float(dr[chosen])
                n_elig += 1
            n_fb += len(slot["fallback"])
        fold_runs.append(FoldRun(
            fa["fold"], candidate,
            tuple(canon_subject(s) for s in fa["fit"]), tuple(canon_subject(s) for s in fa["cal"]),
            tuple(canon_subject(s) for s in fa["eval"]), tuple(canon_subject(s) for s in fa["train"]),
            tuple(canon_subject(s) for s in fa["val"]), best_epoch, float(q), len(cal_scores),
            n_elig, n_fb, float(router_delta)))
    fe = final_epochs([f.best_epoch for f in fold_runs])
    # final refit on ALL eligible subjects (frozen inclusion only) — C1: empty σ_min
    all_ex = _train_examples(source_artifact, eligible, idx, labels)
    sigma_min_oof = {} if candidate != "C2" else _oof_sigma_min(disease, source_artifact, batches, labels, candidate)
    refit = refit_candidate_fixed_epochs(candidate, disease, all_ex, fe, sigma_min_oof, HP["seed_es"])
    return DevelopResult(disease, candidate, float(alpha), float(delta), tuple(fold_runs), fe,
                         refit.artifact_sha256, hash_subject_list(eligible), pool_digest(batches), len(eligible),
                         n_fb_only)


def _oof_sigma_min(disease, source_artifact, batches, labels, candidate):
    """OOF C2 σ_min per action = Q05 of σ̂ over out-of-fold EVAL eligible batches. (Synthetic orchestration only.)"""
    idx = _subject_batches(batches); eligible = _eligible_subjects(idx)
    assignment, _ = cv_assignment(eligible)
    per = {a: [] for a in NON_IDENTITY}
    for fa in assignment:
        tr = _train_examples(source_artifact, fa["train"], idx, labels)
        va = _train_examples(source_artifact, fa["val"], idx, labels)
        artifact, _ = fit_candidate_earlystop(candidate, disease, tr, va, HP["seed_es"])
        for s in fa["eval"]:
            for b in idx[canon_subject(s)]["eligible"]:
                preds, _dr = _batch_preds_and_dr(artifact, source_artifact, b, labels)
                for a in NON_IDENTITY:
                    per[a].append(float(preds[a].scale_used))
    sm = {}
    for a in NON_IDENTITY:
        if not per[a]:
            raise ValueError(f"OOF sigma_min: no σ̂ for action {a}")
        sm[a] = float(np.quantile(per[a], HP["sigma_min_quantile"]))
    return sm


# --------------------------------------------------------------------------------------------------- C0 / v2 replay
def replay_pool_digest(batches) -> str:
    """The C0/v2 replay MUST consume the identical pool. This returns the same pool digest the neural path used, so the
    orchestration can assert set-identity of (subjects, batches) across both paths."""
    return pool_digest(batches)
