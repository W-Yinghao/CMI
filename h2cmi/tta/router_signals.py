"""Stage-B1b-2 router signals (review): three label-free certificates per candidate action,
combined later by an auditable conjunction rule (NOT a learned gate).

  A  empirical-null ELIGIBILITY -- is the action's held-out change-of-variable evidence gain
     beyond what fitting noise on a NO-shift (source) split produces? A one-sided conformal
     p-value against an action-specific source-null distribution. (Calibration only; it does NOT
     rank actions -- mean-subtraction was already shown insufficient.)
  B  cross-subject REPRODUCIBILITY -- does the transform replicate across DISJOINT target
     subjects (a shared domain effect), or is it per-subject noise? dispersion / direction
     cosine / effect-to-noise ratio of the per-subject parameters + cross-fit prediction JS.
  C  class STRUCTURE (cross-view, not self-pseudo-label) -- disc<->gen agreement change, high-
     confidence source-anchor preservation, and Soft Neighborhood Density change; + occupancy /
     entropy guards.

Every function is pure and operates on a frozen model + a ClassConditionalTTA + target embeddings,
reusing the existing source bundles (no retraining).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from h2cmi.tta.class_conditional import Transform
from h2cmi.grid_io import stable_hash_int

EPS = 1e-8


# ----------------------------------------------------------------- scores
def evidence_gain(tta, U: torch.Tensor, transform, pi_T) -> float:
    """Change-of-variable held-/in-set evidence gain of (transform, pi_T) over identity on U."""
    Tid = Transform(U.shape[1], "diag_affine", device=tta.device)
    pi_S = torch.tensor(tta.pi_S, dtype=torch.float32, device=tta.device)
    return tta._change_of_var_nll(U, Tid, pi_S) - tta._change_of_var_nll(U, transform, pi_T)


def loso_evidence_gain(tta, U, groups, spec, *, oracle_labels=None, pooled_ref=None, seed=0) -> float:
    """Mean over groups of: fit the action on the OTHER groups, score the held-out group's
    evidence gain. groups is an int array aligning to U rows. NaN if <2 groups or any fit set
    is below min_target."""
    groups = np.asarray(groups)
    uq = np.unique(groups)
    if len(uq) < 2:
        return float("nan")
    gains = []
    for g in uq:
        ev = groups == g
        fit = ~ev
        if fit.sum() < tta.cfg.min_target:
            continue
        ol = (np.asarray(oracle_labels)[fit] if oracle_labels is not None else None)
        f = tta.fit_variant(U[fit], spec, oracle_labels=ol, pooled_ref=pooled_ref, tta_seed=seed)
        gains.append(evidence_gain(tta, U[ev], f.transform, f.pi_T))
    return float(np.mean(gains)) if gains else float("nan")


def source_null_scores(tta, Us, src_subjects, spec, *, pooled_ref=None, n_draws=80,
                       fit_subjects=2, base_seed=0) -> np.ndarray:
    """Action-specific NO-shift null: repeatedly pick a held-out source subject + a random
    `fit_subjects`-subset of the OTHER source subjects, fit the action, score the held-out
    subject's evidence gain. Source needs no adaptation, so these are pure noise-fit gains.
    Pools across source sites (canonical/same-distribution). TRAINING-INCLUDED (source model saw
    these subjects) -- a prototype null; the final version must exclude them via nested retrain."""
    src_subjects = np.asarray(src_subjects)
    uq = np.unique(src_subjects)
    if len(uq) < fit_subjects + 1:
        return np.array([], dtype=float)
    out = []
    for d in range(n_draws):
        rng = np.random.default_rng((int(base_seed), d, stable_hash_int(spec.name) % 65536))
        held = uq[rng.integers(0, len(uq))]
        others = uq[uq != held]
        fit_subj = rng.choice(others, size=fit_subjects, replace=False)
        fit_mask = np.isin(src_subjects, fit_subj)
        ev_mask = src_subjects == held
        if fit_mask.sum() < tta.cfg.min_target or ev_mask.sum() == 0:
            continue
        f = tta.fit_variant(Us[fit_mask], spec, pooled_ref=pooled_ref, tta_seed=int(d))
        out.append(evidence_gain(tta, Us[ev_mask], f.transform, f.pi_T))
    return np.asarray(out, dtype=float)


def conformal_pvalue(target_score: float, null_scores: np.ndarray) -> float:
    """One-sided conformal p: P(null >= target). p=nan if target nan or no nulls. With m nulls
    the smallest achievable p is 1/(m+1) -- so m must be large for a strict Bonferroni level."""
    null = np.asarray(null_scores, float)
    null = null[np.isfinite(null)]
    if not np.isfinite(target_score) or len(null) == 0:
        return float("nan")
    return float((1 + int((null >= target_score).sum())) / (len(null) + 1))


# ----------------------------------------------------------------- B: reproducibility
def _theta(transform, sd_S) -> np.ndarray:
    """Scale-normalised parameter vector of a diag-affine transform: [a, b/sd_S]."""
    a = transform.a.detach().cpu().numpy() if hasattr(transform, "a") else \
        np.log(np.clip(np.linalg.svd(transform.matrix().detach().cpu().numpy(), compute_uv=False), 1e-6, None))
    b = transform.b.detach().cpu().numpy()
    sd = np.asarray(sd_S, float) + EPS
    return np.concatenate([a, b / sd])


def replicate_stability(tta, U, subject_ids, spec, *, sd_S, oracle_labels=None, pooled_ref=None,
                        decision_prior=None, base_seed=0) -> dict:
    """Disjoint cross-subject reproducibility: fit the action on each SINGLE subject; compare the
    per-subject parameters and the per-subject predictions on the held-out third subject."""
    subj = np.asarray(subject_ids)
    uq = np.unique(subj)
    K = tta.n_classes
    nan = dict(transform_relative_dispersion=float("nan"), transform_direction_cosine=float("nan"),
               transform_effect_to_noise_ratio=float("nan"), crossfit_prediction_js=float("nan"),
               crossfit_prediction_disagreement=float("nan"))
    if len(uq) < 3:
        return nan
    dp = np.full(K, 1.0 / K) if decision_prior is None else np.asarray(decision_prior, float)
    log_dp = torch.log(torch.tensor(dp, dtype=torch.float32, device=tta.device).clamp_min(EPS))
    fits = {}
    for g in uq:
        m = subj == g
        if m.sum() < tta.cfg.min_target:
            return nan
        ol = (np.asarray(oracle_labels)[m] if oracle_labels is not None else None)
        fits[g] = tta.fit_variant(U[m], spec, oracle_labels=ol, pooled_ref=pooled_ref, tta_seed=int(g))
    thetas = np.stack([_theta(fits[g].transform, sd_S) for g in uq])      # [G, 2d]
    mean = thetas.mean(0)
    # relative pairwise dispersion + direction cosine
    disp, cos = [], []
    for i in range(len(uq)):
        for j in range(i + 1, len(uq)):
            ti, tj = thetas[i], thetas[j]
            denom = EPS + 0.5 * (np.linalg.norm(ti) + np.linalg.norm(tj))
            disp.append(np.linalg.norm(ti - tj) / denom)
            c = float(ti @ tj / (np.linalg.norm(ti) * np.linalg.norm(tj) + EPS))
            cos.append(c)
    rms = np.sqrt(((thetas - mean) ** 2).sum(1).mean()) + EPS
    etn = float(np.linalg.norm(mean) / rms)                              # effect-to-noise ratio
    # cross-fit prediction agreement on the held-out third subject
    js_all, dis_all = [], []
    for held in uq:
        fitg = [g for g in uq if g != held]
        m = subj == held
        with torch.no_grad():
            preds = [tta.density.class_posterior(fits[g].transform.apply(U[m]), log_dp) for g in fitg]
        p, q = preds[0].clamp_min(EPS), preds[1].clamp_min(EPS)
        mid = (0.5 * (p + q)).clamp_min(EPS)
        js = 0.5 * (p * (p.log() - mid.log())).sum(1) + 0.5 * (q * (q.log() - mid.log())).sum(1)
        js_all.append(float(js.mean().cpu()))
        dis_all.append(float((p.argmax(1) != q.argmax(1)).float().mean().cpu()))
    return dict(transform_relative_dispersion=float(np.median(disp)),
                transform_direction_cosine=float(np.mean(cos)),
                transform_effect_to_noise_ratio=etn,
                crossfit_prediction_js=float(np.mean(js_all)),
                crossfit_prediction_disagreement=float(np.mean(dis_all)))


# ----------------------------------------------------------------- C: class structure
@torch.no_grad()
def _disc_gen(model, U, transform, log_prior):
    z = transform.apply(U)
    p_disc = F.softmax(model.head.disc_logits(z), dim=1)
    p_gen = model.head.density.class_posterior(z, log_prior)
    return p_disc, p_gen


@torch.no_grad()
def disc_gen_agreement(model, U, transform, log_prior) -> float:
    """Mean -JS(disc || gen) on U under the transform (higher = the two views agree more)."""
    p, q = _disc_gen(model, U, transform, log_prior)
    p, q = p.clamp_min(EPS), q.clamp_min(EPS)
    m = (0.5 * (p + q)).clamp_min(EPS)
    js = 0.5 * (p * (p.log() - m.log())).sum(1) + 0.5 * (q * (q.log() - m.log())).sum(1)
    return float(-js.mean().cpu())


@torch.no_grad()
def source_confidence_threshold(model, Us, log_prior, q=0.5) -> float:
    """Source-calibrated confidence floor: the q-quantile of max(disc,gen) confidence where disc
    and gen agree, on SOURCE embeddings."""
    Tid = Transform(Us.shape[1], "diag_affine", device=Us.device)
    p, qd = _disc_gen(model, Us, Tid, log_prior)
    agree = p.argmax(1) == qd.argmax(1)
    conf = torch.minimum(p.max(1).values, qd.max(1).values)
    a = conf[agree]
    return float(np.quantile(a.cpu().numpy(), q)) if a.numel() else 0.5


@torch.no_grad()
def anchor_flip_rate(model, U, transform, log_prior, conf_thresh) -> tuple[float, int]:
    """Among IDENTITY high-confidence disc/gen-agreement anchors, the fraction whose argmax FLIPS
    under the action. Returns (flip_rate, n_anchors)."""
    Tid = Transform(U.shape[1], "diag_affine", device=U.device)
    p0, q0 = _disc_gen(model, U, Tid, log_prior)
    anchor = (p0.argmax(1) == q0.argmax(1)) & (torch.minimum(p0.max(1).values, q0.max(1).values) >= conf_thresh)
    if anchor.sum() == 0:
        return float("nan"), 0
    pa, qa = _disc_gen(model, U, transform, log_prior)
    y0 = p0.argmax(1)[anchor]
    ya = (0.5 * (pa + qa)).argmax(1)[anchor]
    return float((ya != y0).float().mean().cpu()), int(anchor.sum())


@torch.no_grad()
def soft_neighborhood_density(features: torch.Tensor, temp: float = 0.05) -> float:
    """SND (Saito et al.): mean entropy of the row-softmax of the cosine-similarity matrix (self
    excluded). Higher = denser, more consistent neighborhoods (taken as 'better')."""
    f = F.normalize(features, dim=1)
    s = f @ f.t()
    s.fill_diagonal_(-1e4)
    p = F.softmax(s / temp, dim=1).clamp_min(EPS)
    return float((-(p * p.log()).sum(1)).mean().cpu())


@torch.no_grad()
def class_structure(model, U, transform, log_prior, *, snd_temp=0.05) -> dict:
    """delta_snd vs identity + occupancy/entropy guards on the adapted posterior."""
    Tid = Transform(U.shape[1], "diag_affine", device=U.device)
    snd_id = soft_neighborhood_density(Tid.apply(U), snd_temp)
    z = transform.apply(U)
    snd_ad = soft_neighborhood_density(z, snd_temp)
    r = model.head.density.class_posterior(z, log_prior)
    occ = r.sum(0) / max(1, U.shape[0])
    ent = float((-(r.clamp_min(EPS) * r.clamp_min(EPS).log()).sum(1)).mean().cpu())
    pred = r.argmax(1)
    eff = int(torch.unique(pred).numel())
    return dict(delta_snd=float(snd_ad - snd_id), min_class_occupancy=float(occ.min().cpu()),
                effective_class_count=eff, posterior_entropy=ent)
