"""Synthetic covariate-shift task for the risk-feasible trainer + acceptance report.

``X = [y-features | d-features] + noise``: the task ``Y`` is decodable from the causal
y-features alone, while the spurious d-features leak the domain into an ERM ``Z``. The trainer
can therefore drop the d-features (leakage down) WITHOUT raising the source risk (the task uses
the y-features) — a feasible leakage reduction. Recordings are grouped (multi-class) so the
outer leakage module's grouped cross-fit applies directly on the frozen ``Z``.
"""
from __future__ import annotations

import numpy as np
import torch

from ..leakage.critic import CriticConfig
from ..leakage.crossfit import FrozenFeatures, make_fold_plan
from ..leakage.ucb import bootstrap_ucb
from ..models import build_model
from ..support_graph import build_support_graph, counts_from_labels, empirical_class_prior
from .adversary import ConditionalDomainAdversary, reference_entropy_bar
from .bn import all_eval
from .primal_dual import TrainConfig, train_risk_feasible
from .selector import select_checkpoint


def make_covariate_shift(seed=0, n_domains=2, n_classes=2, recs_per_domain=4, per_cell=30,
                         ky=3, kd=3, m=20, y_scale=2.5, d_scale=1.2):
    rng = np.random.default_rng(seed)
    y, d, g = [], [], []
    gid = 0
    for dom in range(n_domains):
        for _ in range(recs_per_domain):
            for c in range(n_classes):
                y += [c] * per_cell; d += [dom] * per_cell; g += [gid] * per_cell
            gid += 1
    y, d, g = np.array(y), np.array(d), np.array(g)
    y_means = rng.standard_normal((n_classes, ky)) * y_scale   # causal (task) features
    # spurious (domain) features: decodable but NOT saturating (so the adversarial gradient
    # to the encoder does not vanish), and irrelevant to Y (so dropping them is feasible).
    d_means = rng.standard_normal((n_domains, kd)) * d_scale
    X = np.concatenate(
        [y_means[y] + 0.4 * rng.standard_normal((y.size, ky)),
         d_means[d] + 0.6 * rng.standard_normal((y.size, kd))],
        axis=1,
    )
    counts = counts_from_labels(d, y, n_domains=n_domains, n_classes=n_classes)
    sg = build_support_graph(counts, m=m, reference_prior=empirical_class_prior(counts))
    return X.astype(np.float32), y, d, g, sg


def _factory(in_dim, cfg):
    return lambda: build_model("mlp", in_dim=in_dim, n_classes=cfg.n_classes, z_dim=cfg.z_dim, hidden=cfg.enc_hidden)


def _frozen_Z(model_state, factory, X) -> np.ndarray:
    m = factory()
    m.load_state_dict(model_state)
    with all_eval(m), torch.no_grad():
        return m(torch.as_tensor(np.asarray(X), dtype=torch.float32)).z.numpy()


def evaluate_surrogate(Z, y, d, sg, cfg, steps=300) -> float:
    """Best-critic train leakage surrogate ``H_ref_bar − C_D`` on a FROZEN Z (fair before/after)."""
    torch.manual_seed(0)
    Zt = torch.as_tensor(np.asarray(Z), dtype=torch.float32)
    adv = ConditionalDomainAdversary(Zt.shape[1], sg, hidden=cfg.adv_hidden)
    opt = torch.optim.Adam(adv.parameters(), lr=1e-2)
    for _ in range(steps):
        opt.zero_grad()
        adv.domain_ce(Zt, y, d).backward()
        opt.step()
    with torch.no_grad():
        return reference_entropy_bar(sg) - float(adv.domain_ce(Zt, y, d).item())


_OUTER_CAPS = (0, 32)   # probe family for the OUTER score in this (linearly-shifted) demo


def _outer_leakage(Z, y, d, g, sg, B):
    feat = FrozenFeatures(Z, y, d, g)
    plan = make_fold_plan(feat, sg, n_folds=4, seed=0)
    res = bootstrap_ucb(feat, sg, plan, CriticConfig(capacities=_OUTER_CAPS), alpha=0.1, n_bootstrap=B, seed=0)
    return res["extractable_LQ_ov"], res["bootstrap_ucl"]


def _outer_point(model_state, factory, X, y, d, g, sg) -> float:
    """Best-critic outer leakage POINT estimate on a candidate's frozen Z (no bootstrap)."""
    from ..leakage.estimate import estimate_extractable_leakage
    Z = _frozen_Z(model_state, factory, X)
    feat = FrozenFeatures(Z, y, d, g)
    plan = make_fold_plan(feat, sg, n_folds=4, seed=0)
    return estimate_extractable_leakage(feat, sg, plan, CriticConfig(capacities=_OUTER_CAPS))["extractable_LQ_ov"]


def acceptance_report(seed=0, B=100, candidate_stride=15) -> dict:
    import dataclasses
    X, y, d, g, sg = make_covariate_shift(seed=seed)
    cfg = TrainConfig(seed=seed)
    res = train_risk_feasible(X, y, d, g, sg, cfg)
    factory = _factory(X.shape[1], cfg)
    # honest selection: rank FEASIBLE checkpoints by the best-critic OUTER leakage (injected
    # score_fn on the frozen representation), not the optimistic co-trained surrogate. Scoring
    # every epoch is costly, so subsample the trajectory (stride) + always keep the last epoch.
    cands = res.trajectory[::candidate_stride] + [res.trajectory[-1]]
    res_sub = dataclasses.replace(res, trajectory=cands)
    score_fn = lambda c: _outer_point(c.model_state, factory, X, y, d, g, sg)
    sel = select_checkpoint(res_sub, score_fn=score_fn, score_name="extractable_LQ_ov_point")

    Z_erm = _frozen_Z(res.erm_record.model_state, factory, X)
    Z_sel = _frozen_Z(sel.model_state, factory, X)
    sur_before = evaluate_surrogate(Z_erm, y, d, sg, cfg)
    sur_after = evaluate_surrogate(Z_sel, y, d, sg, cfg)
    lq_before, ucl_before = _outer_leakage(Z_erm, y, d, g, sg, B)
    lq_after, ucl_after = _outer_leakage(Z_sel, y, d, g, sg, B)

    rep = {
        "R_ERM_hat": res.erm_stage.R_ERM_hat,
        "tau": res.erm_stage.tau,
        "selected_R_src": sel.R_src,
        "realized_risk_gap": sel.R_src - res.erm_stage.R_ERM_hat,
        "train_leakage_surrogate_before": sur_before,
        "train_leakage_surrogate_after": sur_after,
        "extractable_LQ_ov_before": lq_before,
        "extractable_LQ_ov_after": lq_after,
        "bootstrap_ucl_before": ucl_before,             # ERM is pre-specified -> clean UCL
        "selection_bootstrap_ucl_after": ucl_after,     # selected adaptively -> optimistic
        "final_lambda": res.trajectory[-1].lam if res.trajectory else cfg.lambda_init,
        "selected_epoch": sel.selected_epoch,
        "used_erm_fallback": sel.used_erm_fallback,
    }
    return rep


def _demo() -> None:
    rep = acceptance_report()
    print("Risk-feasible trainer — acceptance report (covariate-shift synthetic)")
    print(f"  R_ERM_hat                  = {rep['R_ERM_hat']:.4f}")
    print(f"  tau (= R_ERM_hat + eps)    = {rep['tau']:.4f}")
    print(f"  selected_R_src             = {rep['selected_R_src']:.4f}")
    print(f"  realized_risk_gap          = {rep['realized_risk_gap']:+.4f}  (<= eps)")
    print(f"  train_leakage_surrogate    = {rep['train_leakage_surrogate_before']:.4f}"
          f"  ->  {rep['train_leakage_surrogate_after']:.4f}")
    print(f"  extractable_LQ_ov          = {rep['extractable_LQ_ov_before']:.4f}"
          f"  ->  {rep['extractable_LQ_ov_after']:.4f}")
    print(f"  bootstrap_ucl (ERM, clean) = {rep['bootstrap_ucl_before']:.4f}")
    print(f"  selection_bootstrap_ucl    = {rep['selection_bootstrap_ucl_after']:.4f}  (post-selection; recompute on audit split)")
    print(f"  final_lambda               = {rep['final_lambda']:.4f}")
    print(f"  selected_epoch             = {rep['selected_epoch']}")
    print(f"  used_erm_fallback          = {rep['used_erm_fallback']}")


if __name__ == "__main__":
    _demo()
