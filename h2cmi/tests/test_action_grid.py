"""Stage-B0: the fit/decision-prior separation invariant, and the action-attribution logic."""
from __future__ import annotations

import json
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

from h2cmi.config import DensityConfig, TTAConfig
from h2cmi.density.student_t_mixture import ClassConditionalDensity
from h2cmi.tta.class_conditional import ClassConditionalTTA
from h2cmi.analyze_action_grid import analyze
from sklearn.metrics import balanced_accuracy_score


# ----- TTA invariant: prior-only cannot change uniform-decision balanced accuracy --------
def test_prior_only_invariant_under_uniform_decision():
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
    yt = rng.choice(K, 400, p=[0.6, 0.3, 0.1])
    U = (dens.mu[yt, 0] * torch.linspace(1.5, 0.6, d) + 0.4).detach()
    tta = ClassConditionalTTA(dens, pi_S, TTAConfig(em_iters=20), K)
    uni = torch.log(torch.full((K,), 1.0 / K))

    def bacc_uni(res):
        z = res.transform.apply(U)
        return balanced_accuracy_score(yt, dens.class_posterior(z, uni).argmax(1).numpy())

    b_id = bacc_uni(tta.fit_action(U, "identity"))
    b_pr = bacc_uni(tta.fit_action(U, "prior_only"))
    assert abs(b_id - b_pr) < 1e-9, "prior-only changed uniform-decision bAcc"
    # prior-only must still recover a non-uniform fit prior (toward the true [0.6,0.3,0.1])
    pi_fit = tta.fit_action(U, "prior_only").pi_T
    assert pi_fit[0] > pi_fit[2] + 0.1, pi_fit


# ----- attribution logic on synthetic action-grid rows ------------------------------------
def _row(seed, site, scen, action, cmi, bacc, acc, nll):
    strict = dict(strict_disc_bacc=0.4, strict_gen_bacc=0.4, strict_blend_bacc=0.4,
                  disc_ece=0.1, gen_ece=0.1, blend_ece=0.1, disc_gen_disagreement=0.2)
    return dict(data_seed=seed, target_site=site, scenario=scen, action=action, cmi=cmi,
                bacc_uniform_decision=bacc, accuracy_target_prior=acc, nll_target_prior=nll, **strict)


# planted: identity baseline bAcc=0.70, acc=0.60, nll=1.0
PLAN = {  # (geom_bacc_gain, prior_acc_gain, prior_nll_gain, expect)
    "cov":                  (0.10, 0.00, 0.00, "geometry_driven"),
    "prior":                (0.00, 0.10, -0.30, "prior_driven"),
    "conditional_rotation": (-0.05, 0.00, 0.00, "neither"),
}


def _build(path):
    rows = []
    for scen, (gg, pa, pn, _) in PLAN.items():
        for seed in (0, 1, 2):
            for site in (0, 1):
                for cmi in ("off", "on"):
                    rows.append(_row(seed, site, scen, "identity", cmi, 0.70, 0.60, 1.0))
                    rows.append(_row(seed, site, scen, "prior_only", cmi, 0.70, 0.60 + pa, 1.0 + pn))
                    rows.append(_row(seed, site, scen, "geometry_only", cmi, 0.70 + gg, 0.60, 1.0))
                    rows.append(_row(seed, site, scen, "joint", cmi, 0.70 + gg, 0.60 + pa, 1.0 + pn))
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_action_attribution():
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "a.jsonl")
        _build(path)
        rep = analyze(path, n_boot=300, seed=0)
        for scen, (_, _, _, expect) in PLAN.items():
            assert rep["decisions"][scen]["code"] == expect, \
                f"{scen}: got {rep['decisions'][scen]['code']}, expected {expect}"
            # prior-only never moves balanced accuracy (the invariant), every scenario
            assert abs(rep["actions"][scen]["prior_effect_bacc"]["seed_mean"]) < 1e-6
        assert abs(rep["actions"]["cov"]["transform_effect"]["seed_mean"] - 0.10) < 0.01
        assert abs(rep["actions"]["prior"]["prior_effect_acc"]["seed_mean"] - 0.10) < 0.01


if __name__ == "__main__":
    test_prior_only_invariant_under_uniform_decision()
    test_action_attribution()
    print("test_action_grid PASSED")
