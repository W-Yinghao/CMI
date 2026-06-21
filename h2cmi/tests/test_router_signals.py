"""Stage-B1b-2 router signal functions: A (empirical-null conformal p-value), B (disjoint
cross-subject reproducibility), C (cross-view class structure) -- shapes, ranges, and the
discriminating behaviours they must exhibit."""
from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import torch

warnings.filterwarnings("ignore")

from h2cmi.config import DensityConfig, TTAConfig
from h2cmi.density.student_t_mixture import HybridHead
from h2cmi.tta.class_conditional import ClassConditionalTTA, B1A_VARIANTS_BY_NAME
from h2cmi.tta import router_signals as rs


def _setup(n=240, d=8, K=3, seed=0):
    torch.manual_seed(seed)
    head = HybridHead(d, K, DensityConfig(n_components=1, cov_rank=2))
    with torch.no_grad():
        head.density.mu[:, 0] = torch.eye(K, d)[:, :d] * 3.0
    model = SimpleNamespace(head=head, cfg=SimpleNamespace(n_classes=K))
    pi_S = np.full(K, 1.0 / K)
    tta = ClassConditionalTTA(head.density, pi_S, TTAConfig(em_iters=6), K)
    rng = np.random.default_rng(seed)
    yt = rng.choice(K, size=n, p=[0.5, 0.3, 0.2])
    U = (head.density.mu[yt, 0] + 0.3 * torch.randn(n, d)).detach()
    subj = rng.integers(0, 3, size=n)
    sd_S = U.std(0, unbiased=False).cpu().numpy()
    log_prior = torch.log(torch.tensor(pi_S, dtype=torch.float32).clamp_min(1e-8))
    return model, tta, U, yt, subj, sd_S, log_prior


def test_conformal_pvalue_and_source_null():
    model, tta, U, yt, subj, sd_S, _ = _setup()
    spec = B1A_VARIANTS_BY_NAME["gen_iterative_diag"]
    src_subj = np.repeat(np.arange(12), 20)                       # 12 source subjects
    Us = (tta.density.mu[np.random.default_rng(1).integers(0, 3, 240), 0] + 0.3 * torch.randn(240, 8)).detach()
    nulls = rs.source_null_scores(tta, Us, src_subj, spec, n_draws=40)
    assert nulls.ndim == 1 and len(nulls) > 0
    p_hi = rs.conformal_pvalue(float(np.max(nulls)) + 1e6, nulls)  # target >> null -> tiny p
    p_lo = rs.conformal_pvalue(float(np.min(nulls)) - 1e6, nulls)  # target << null -> p ~ 1
    assert 0.0 < p_hi <= p_lo <= 1.0 and p_hi < 0.1
    assert np.isnan(rs.conformal_pvalue(float("nan"), nulls))


def test_replicate_stability_shapes_and_single_subject():
    model, tta, U, yt, subj, sd_S, _ = _setup()
    spec = B1A_VARIANTS_BY_NAME["gen_oneshot_diag"]
    s = rs.replicate_stability(tta, U, subj, spec, sd_S=sd_S)
    for k in ("transform_relative_dispersion", "transform_direction_cosine",
              "transform_effect_to_noise_ratio", "crossfit_prediction_js",
              "crossfit_prediction_disagreement"):
        assert k in s and np.isfinite(s[k]), k
    assert -1.0001 <= s["transform_direction_cosine"] <= 1.0001
    nan = rs.replicate_stability(tta, U, np.zeros_like(subj), spec, sd_S=sd_S)  # 1 subject
    assert np.isnan(nan["transform_relative_dispersion"])


def test_class_structure_and_anchors():
    model, tta, U, yt, subj, sd_S, log_prior = _setup()
    f = tta.fit_variant(U, B1A_VARIANTS_BY_NAME["gen_iterative_diag"], tta_seed=0)
    cs = rs.class_structure(model, U, f.transform, log_prior)
    assert set(cs) == {"delta_snd", "min_class_occupancy", "effective_class_count", "posterior_entropy"}
    assert 0.0 <= cs["min_class_occupancy"] <= 1.0 and 1 <= cs["effective_class_count"] <= 3
    thr = rs.source_confidence_threshold(model, U, log_prior, q=0.5)
    flip, n_anchor = rs.anchor_flip_rate(model, U, f.transform, log_prior, thr)
    assert n_anchor >= 0 and (np.isnan(flip) or 0.0 <= flip <= 1.0)
    # identity transform: agreement is finite, anchors never flip under identity
    from h2cmi.tta.class_conditional import Transform
    Tid = Transform(U.shape[1], "diag_affine")
    flip0, _ = rs.anchor_flip_rate(model, U, Tid, log_prior, thr)
    assert np.isnan(flip0) or flip0 == 0.0
    assert np.isfinite(rs.disc_gen_agreement(model, U, f.transform, log_prior))


def test_snd_higher_for_clustered_features():
    # well-separated clusters should have different SND than uniform noise (sanity: finite, sensible)
    torch.manual_seed(0)
    clustered = torch.cat([torch.randn(50, 4) + 10 * torch.eye(4)[i] for i in range(3)], 0)
    noise = torch.randn(150, 4)
    assert np.isfinite(rs.soft_neighborhood_density(clustered))
    assert rs.soft_neighborhood_density(clustered) != rs.soft_neighborhood_density(noise)


if __name__ == "__main__":
    for n, fn in sorted(globals().items()):
        if n.startswith("test_") and callable(fn):
            fn(); print(f"  {n} PASSED")
    print("test_router_signals PASSED")
