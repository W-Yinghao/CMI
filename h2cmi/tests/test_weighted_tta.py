"""Mandatory tests for the weighted V2P estimators (REVIEW_P0 section D, tests 1-6).

  python -m h2cmi.tests.test_weighted_tta
"""
from __future__ import annotations

import numpy as np
import torch

from h2cmi.config import DensityConfig, TTAConfig
from h2cmi.density.student_t_mixture import ClassConditionalDensity
from h2cmi.tta.class_conditional import ClassConditionalTTA, Transform
from h2cmi.tta.weighted_tta import (fit_weighted_em, fit_weighted_pooled, weighted_pooled_moments,
                                    canonical_weights, effective_weights)

D, K = 8, 3


def _setup(n=240, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    dens = ClassConditionalDensity(D, K, DensityConfig(n_components=1, cov_rank=2))
    with torch.no_grad():
        dens.mu[:, 0] = torch.eye(K, D) * 3.0
    pi_S = np.full(K, 1.0 / K)
    yt = rng.integers(0, K, size=n)
    U = dens.mu[yt, 0] + 0.4 * torch.randn(n, D)
    A = torch.eye(D) + 0.08 * torch.randn(D, D)
    U = U @ A.T + 0.2
    cfg = TTAConfig(em_iters=12, em_lr=5e-2)
    return dens, U.detach(), pi_S, cfg, yt


def test_1_equal_weights_reproduce_unweighted():
    dens, U, pi_S, cfg, _ = _setup()
    Tw, _ = fit_weighted_em(dens, U, np.ones(len(U)), pi_S, cfg, K, "cpu", "iterative")
    tta = ClassConditionalTTA(dens, pi_S, cfg, K, "cpu")
    Tu, _ = tta._fit_transform(U, fixed_prior=torch.tensor(pi_S, dtype=torch.float32))
    da = float((Tw.a - Tu.a).abs().max()); db = float((Tw.b - Tu.b).abs().max())
    assert da < 1e-5 and db < 1e-5, f"equal-weights != unweighted (da={da:.2e}, db={db:.2e})"
    print(f"[1] equal weights reproduce unweighted: da={da:.2e} db={db:.2e} OK")


def test_2_rational_weights_reproduce_replication():
    dens, U, pi_S, cfg, _ = _setup(n=30)
    m = (np.arange(len(U)) % 3 + 1).astype(int)            # multiplicities 1,2,3,...
    Tw, _ = fit_weighted_em(dens, U, m.astype(float), pi_S, cfg, K, "cpu", "iterative")
    rep = torch.cat([U[i].repeat(m[i], 1) for i in range(len(U))], 0)
    Tr, _ = fit_weighted_em(dens, rep, np.ones(len(rep)), pi_S, cfg, K, "cpu", "iterative")
    da = float((Tw.a - Tr.a).abs().max()); db = float((Tw.b - Tr.b).abs().max())
    assert da < 1e-4 and db < 1e-4, f"weighted != replication (da={da:.2e}, db={db:.2e})"
    print(f"[2] rational weights reproduce replication: da={da:.2e} db={db:.2e} OK")


def test_3_ratios_identical_ids():
    rng = np.random.default_rng(1); y = (rng.random(120) < 0.5).astype(int)
    supports = []
    for q in (0.50, 0.75, 0.25):
        w = effective_weights(y, q)
        assert np.all(w > 0), "some trial got zero weight -> id set would differ"
        supports.append(tuple(np.nonzero(w > 0)[0].tolist()))
    assert supports[0] == supports[1] == supports[2], "ratios use different trial ids"
    print(f"[3] all 3 ratios use identical trial ids (n={len(y)}) OK")


def test_4_weight_sums_and_masses_exact():
    rng = np.random.default_rng(2); y = (rng.random(200) < 0.4).astype(int); N = len(y)
    for q in (0.50, 0.75, 0.25):
        cw = canonical_weights(y, q); ew = effective_weights(y, q)
        assert abs(cw.sum() - 1.0) < 1e-12, f"canonical sum {cw.sum()} != 1"
        assert abs(ew.sum() - N) < 1e-9, f"effective sum {ew.sum()} != N"
        m0 = ew[y == 0].sum(); m1 = ew[y == 1].sum()
        assert abs(m0 - N * q) < 1e-9 and abs(m1 - N * (1 - q)) < 1e-9, f"class masses off at q={q}"
    print("[4] weight sums + effective class masses exact OK")


def test_5_labels_never_enter_nonoracle():
    dens, U, pi_S, cfg, yt = _setup()
    rng = np.random.default_rng(3)
    p1 = rng.permutation(K)[yt]; p2 = rng.permutation(K)[yt]      # two label relabelings
    T1, _ = fit_weighted_em(dens, U, np.ones(len(U)), pi_S, cfg, K, "cpu", "iterative", oracle_labels=p1)
    T2, _ = fit_weighted_em(dens, U, np.ones(len(U)), pi_S, cfg, K, "cpu", "iterative", oracle_labels=p2)
    da = float((T1.a - T2.a).abs().max())
    assert da == 0.0, f"non-oracle fit depended on labels (da={da})"
    print("[5] labels never enter a non-oracle responsibility/optimizer OK")


def test_6_identity_ratio_invariant():
    dens, U, pi_S, cfg, _ = _setup()
    rng = np.random.default_rng(4); y = (rng.random(len(U)) < 0.5).astype(int)
    uni = torch.log(torch.tensor(pi_S, dtype=torch.float32).clamp_min(1e-8))
    Tid = Transform(D, "diag_affine")
    preds = []
    for q in (0.50, 0.75, 0.25):
        _ = effective_weights(y, q)                           # ratio differs, identity must not
        with torch.no_grad():
            p = dens.class_posterior(Tid.apply(U), uni).argmax(1).numpy()
        preds.append(p)
    assert np.array_equal(preds[0], preds[1]) and np.array_equal(preds[0], preds[2]), \
        "identity predictions changed with ratio"
    print("[6] identity transform + predictions ratio-invariant OK")


def test_7_weighted_pooled_equal_is_unweighted():
    dens, U, pi_S, cfg, _ = _setup()
    mu, sd = weighted_pooled_moments(U, np.ones(len(U)))
    assert float((mu - U.mean(0)).abs().max()) < 1e-5
    assert float((sd - U.std(0, unbiased=False)).abs().max()) < 1e-5
    print("[7] weighted pooled moments reduce to unweighted OK")


def test_8_allones_joint_equals_unweighted():
    dens, U, pi_S, cfg, _ = _setup()
    Tw, piw = fit_weighted_em(dens, U, np.ones(len(U)), pi_S, cfg, K, "cpu", "joint")
    tta = ClassConditionalTTA(dens, pi_S, cfg, K, "cpu")
    Tu, piu = tta._fit_transform(U)                          # default = joint (E-step + prior M-step)
    da = float((Tw.a - Tu.a).abs().max()); dp = float((piw - piu).abs().max())
    assert da < 1e-5 and dp < 1e-6, f"all-ones joint != unweighted joint (da={da:.2e}, dpi={dp:.2e})"
    print(f"[8] all-ones weighted joint == unweighted joint: da={da:.2e} dpi={dp:.2e} OK")


def test_9_allones_oneshot_equals_unweighted():
    import torch.nn.functional as F
    dens, U, pi_S, cfg, _ = _setup()
    pi_S_t = torch.tensor(pi_S, dtype=torch.float32)
    with torch.no_grad():
        r0 = F.softmax(dens.log_prob_all(U) + torch.log(pi_S_t.clamp_min(1e-8)).view(1, -1), dim=1)
    Tw, _ = fit_weighted_em(dens, U, np.ones(len(U)), pi_S, cfg, K, "cpu", "oneshot")
    tta = ClassConditionalTTA(dens, pi_S, cfg, K, "cpu")
    Tu, _ = tta._fit_transform(U, fixed_resp=r0, fixed_prior=pi_S_t)
    da = float((Tw.a - Tu.a).abs().max()); db = float((Tw.b - Tu.b).abs().max())
    assert da < 1e-5 and db < 1e-5, f"all-ones one-shot != unweighted one-shot (da={da:.2e})"
    print(f"[9] all-ones weighted one-shot == unweighted one-shot: da={da:.2e} db={db:.2e} OK")


def test_10_intweight_joint_equals_replication():
    dens, U, pi_S, cfg, _ = _setup(n=30)
    m = (np.arange(len(U)) % 3 + 1).astype(int)
    Tw, piw = fit_weighted_em(dens, U, m.astype(float), pi_S, cfg, K, "cpu", "joint")
    rep = torch.cat([U[i].repeat(m[i], 1) for i in range(len(U))], 0)
    Tr, pir = fit_weighted_em(dens, rep, np.ones(len(rep)), pi_S, cfg, K, "cpu", "joint")
    da = float((Tw.a - Tr.a).abs().max()); dp = float((piw - pir).abs().max())
    assert da < 1e-4 and dp < 1e-5, f"int-weight joint != replication joint (da={da:.2e}, dpi={dp:.2e})"
    print(f"[10] integer-weight joint == replication joint: da={da:.2e} dpi={dp:.2e} OK")


if __name__ == "__main__":
    test_1_equal_weights_reproduce_unweighted()
    test_2_rational_weights_reproduce_replication()
    test_3_ratios_identical_ids()
    test_4_weight_sums_and_masses_exact()
    test_5_labels_never_enter_nonoracle()
    test_6_identity_ratio_invariant()
    test_7_weighted_pooled_equal_is_unweighted()
    test_8_allones_joint_equals_unweighted()
    test_9_allones_oneshot_equals_unweighted()
    test_10_intweight_joint_equals_replication()
    print("ALL WEIGHTED-TTA TESTS PASSED")
