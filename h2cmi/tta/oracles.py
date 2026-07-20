"""Oracle TTA diagnostics (review §"Oracle diagnostics").

Three named oracles isolate WHERE unsupervised class-conditional TTA fails:

  oracle_prior               true target class frequency as a FIXED pi_T; the transform is
                             still estimated unsupervised. If this fixes a failure, the
                             problem is PRIOR ESTIMATION.
  oracle_labels              true labels as one-hot responsibilities (skip the E-step),
                             fit the transform under the density. If this fixes it, the
                             problem is the EM RESPONSIBILITIES (soft assignment).
  oracle_supervised_transform cross-fitted supervised transform fit (labels on one half,
                             score the other), reporting the held-out evidence ceiling. If
                             even this does not help, the diagonal TRANSFORM FAMILY (or the
                             density geometry) is insufficient.

We deliberately do NOT provide an ``oracle_simulator_transform``: the simulator's known
mixing lives in sensor/source space and is followed by per-channel z-scoring and a
non-linear encoder, so there is generally no exact latent diagonal that inverts it. A
mechanism oracle would require an explicit pre-normalisation unmix-remix, not implemented.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch
import torch.nn.functional as F

from h2cmi.tta.class_conditional import ClassConditionalTTA, TTAResult, Transform


@contextlib.contextmanager
def _frozen_density(tta: ClassConditionalTTA):
    saved = [(p, p.requires_grad) for p in tta.density.parameters()]
    for p, _ in saved:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, req in saved:
            p.requires_grad_(req)


def _empirical_prior(labels: np.ndarray, K: int) -> np.ndarray:
    c = np.bincount(np.asarray(labels), minlength=K).astype(np.float64)
    return c / max(c.sum(), 1.0)


def _result(tta, U, T, pi_T, kind, extra=None) -> TTAResult:
    log_piS = torch.log(torch.tensor(tta.pi_S, dtype=torch.float32, device=tta.device).clamp_min(1e-8))
    nll_before = tta._identity_diagnostics(U, log_piS)
    diag = tta._diagnostics(U, T, pi_T, log_piS, nll_before)
    diag["oracle"] = kind
    if extra:
        diag.update(extra)
    return TTAResult(T, pi_T.detach().cpu().numpy(), adapted=True, diagnostics=diag)


def oracle_prior(tta: ClassConditionalTTA, U: torch.Tensor, labels: np.ndarray) -> TTAResult:
    """True prior fixed; transform estimated unsupervised."""
    U = U.detach().to(tta.device)
    pi_true = torch.tensor(_empirical_prior(labels, tta.n_classes), dtype=torch.float32, device=tta.device)
    with _frozen_density(tta):
        T, pi_T = tta._fit_transform(U, fixed_prior=pi_true)
    return _result(tta, U, T, pi_T, "prior")


def oracle_labels(tta: ClassConditionalTTA, U: torch.Tensor, labels: np.ndarray) -> TTAResult:
    """True labels as one-hot responsibilities; transform fit under the density."""
    U = U.detach().to(tta.device)
    y = torch.as_tensor(labels, dtype=torch.long, device=tta.device)
    resp = F.one_hot(y, tta.n_classes).float()
    pi_true = torch.tensor(_empirical_prior(labels, tta.n_classes), dtype=torch.float32, device=tta.device)
    with _frozen_density(tta):
        T, pi_T = tta._fit_transform(U, fixed_prior=pi_true, fixed_resp=resp)
    return _result(tta, U, T, pi_T, "labels")


def crossfit_supervised_gain(tta: ClassConditionalTTA, U: torch.Tensor, labels: np.ndarray) -> float:
    """2-fold cross-fitted held-out change-of-variable NLL gain using TRUE labels.

    The evidence ceiling: best the diagonal family + density can do with perfect labels,
    scored honestly out-of-fold."""
    U = U.detach().to(tta.device)
    y = np.asarray(labels)
    n = U.shape[0]
    perm = torch.randperm(n, device=U.device)
    half = n // 2
    Tid = Transform(U.shape[1], "diag_affine", device=tta.device)
    pi_S = torch.tensor(tta.pi_S, dtype=torch.float32, device=tta.device)
    gains = []
    with _frozen_density(tta):
        for fit_idx, ev_idx in ((perm[:half], perm[half:]), (perm[half:], perm[:half])):
            fi = fit_idx.cpu().numpy()
            resp = F.one_hot(torch.as_tensor(y[fi], dtype=torch.long, device=tta.device),
                             tta.n_classes).float()
            pi_fit = torch.tensor(_empirical_prior(y[fi], tta.n_classes), dtype=torch.float32, device=tta.device)
            T, pi = tta._fit_transform(U[fit_idx], fixed_prior=pi_fit, fixed_resp=resp)
            nll_id = tta._change_of_var_nll(U[ev_idx], Tid, pi_S)
            nll_ad = tta._change_of_var_nll(U[ev_idx], T, pi)
            gains.append(nll_id - nll_ad)
    return float(np.mean(gains))


def oracle_supervised_oof(tta: ClassConditionalTTA, U: torch.Tensor, labels: np.ndarray,
                          groups: np.ndarray | None = None):
    """TRUE out-of-fold supervised predictions (review blocker #2).

    Split into two folds by ``groups`` (target SUBJECT ids -> held-out-subject folds, not a
    random trial split); fit the supervised transform with labels on one fold, PREDICT the
    other; concatenate. The returned probabilities are genuinely held-out, so their bAcc/NLL
    is an honest supervised ceiling -- unlike the transductive ``oracle_labels`` whose
    full-data refit shares responsibilities with its own predictions.

    Returns (proba [N,K], info).
    """
    U = U.detach().to(tta.device)
    y = np.asarray(labels)
    n = U.shape[0]
    if groups is None or len(np.unique(groups)) < 2:
        groups = np.arange(n) % 2                      # fallback: random-ish trial split
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    foldA_g = set(uniq[::2].tolist())
    inA = np.isin(groups, list(foldA_g))
    proba = np.full((n, tta.n_classes), 1.0 / tta.n_classes, dtype=np.float64)
    with _frozen_density(tta):
        for fit_mask, pred_mask in ((~inA, inA), (inA, ~inA)):
            if fit_mask.sum() < 2 or pred_mask.sum() < 1:
                continue
            fm = torch.as_tensor(fit_mask, device=tta.device)
            pm = torch.as_tensor(pred_mask, device=tta.device)
            resp = F.one_hot(torch.as_tensor(y[fit_mask], dtype=torch.long, device=tta.device),
                             tta.n_classes).float()
            pi_fit = torch.tensor(_empirical_prior(y[fit_mask], tta.n_classes),
                                  dtype=torch.float32, device=tta.device)
            T, pi = tta._fit_transform(U[fm], fixed_prior=pi_fit, fixed_resp=resp)
            with torch.no_grad():
                z = T.apply(U[pm])
                proba[pred_mask] = tta.density.class_posterior(
                    z, torch.log(pi.clamp_min(1e-8))).cpu().numpy()
    return proba, dict(oof_groups=int(len(uniq)))


def oracle_supervised_transform(tta: ClassConditionalTTA, U: torch.Tensor, labels: np.ndarray) -> TTAResult:
    """Cross-fitted supervised transform: reports the held-out evidence ceiling and a
    full-data refit (with labels) to apply. (Transductive proxy; prefer oracle_supervised_oof
    for the accuracy ceiling.)"""
    gain = crossfit_supervised_gain(tta, U, labels)
    U = U.detach().to(tta.device)
    y = torch.as_tensor(labels, dtype=torch.long, device=tta.device)
    resp = F.one_hot(y, tta.n_classes).float()
    pi_true = torch.tensor(_empirical_prior(labels, tta.n_classes), dtype=torch.float32, device=tta.device)
    with _frozen_density(tta):
        T, pi_T = tta._fit_transform(U, fixed_prior=pi_true, fixed_resp=resp)
    return _result(tta, U, T, pi_T, "supervised_transform",
                   extra={"crossfit_supervised_gain": gain})


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    from h2cmi.config import DensityConfig, TTAConfig
    from h2cmi.density.student_t_mixture import ClassConditionalDensity
    torch.manual_seed(0)
    d, K = 8, 3
    dens = ClassConditionalDensity(d, K, DensityConfig(n_components=1, cov_rank=2, df=8.0))
    with torch.no_grad():
        dens.mu.zero_()
        for c in range(K):
            dens.mu[c, 0, c % d] = 4.0
        dens.log_s.fill_(-1.0)
    pi_S = np.full(K, 1.0 / K)
    rng = np.random.default_rng(0)
    yt = rng.integers(0, K, 400)
    zs = dens.mu[yt, 0] + 0.3 * torch.randn(400, d)
    a_true = torch.linspace(1.5, 0.6, d); b_true = 0.7 * torch.ones(d)
    U = ((zs - b_true) / a_true).detach()
    tta = ClassConditionalTTA(dens, pi_S, TTAConfig(em_iters=25), K)
    g_unsup = tta._crossfit_evidence_gain(U)
    g_sup = crossfit_supervised_gain(tta, U, yt)
    print("unsupervised held-out gain:", round(g_unsup, 4))
    print("supervised  held-out gain:", round(g_sup, 4))
    print("oracle_prior ok:", oracle_prior(tta, U, yt).adapted)
    print("oracle_labels ok:", oracle_labels(tta, U, yt).adapted)
    r = oracle_supervised_transform(tta, U, yt)
    print("oracle_supervised gain field:", round(r.diagnostics["crossfit_supervised_gain"], 4))
    assert g_sup >= g_unsup - 0.5, "supervised should not be worse than unsupervised"
    print("oracles self-test PASSED")
