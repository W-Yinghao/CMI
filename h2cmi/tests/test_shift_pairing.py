"""Paired-simulator invariants: source identical across scenarios; mechanisms orthogonal;
order-independent."""
from __future__ import annotations

import numpy as np

from h2cmi.data.paired_simulator import PairedEEGSimulator


def _sample(scen, target=1, seed=0):
    sim = PairedEEGSimulator(3, 12, 64, data_seed=seed)
    return sim.sample(3, 2, 2, 12, target_site=target, scenario=scen)


def test_source_identical_across_scenarios():
    target = 1
    ref = _sample("no_shift", target)
    src = ref.site != target
    for scen in ("cov", "prior", "concept", "cov_prior", "cov_concept"):
        s = _sample(scen, target)
        ssrc = s.site != target
        assert np.array_equal(ref.X[src], s.X[ssrc]), f"{scen} perturbed SOURCE X"
        assert np.array_equal(ref.y[src], s.y[ssrc]), f"{scen} perturbed SOURCE y"
        assert np.array_equal(np.where(src)[0], np.where(ssrc)[0]), f"{scen} changed source indices"


def test_mechanisms_are_orthogonal():
    target = 1
    base = _sample("no_shift", target)
    cov = _sample("cov", target)
    concept = _sample("concept", target)
    prior = _sample("prior", target)
    bt, ct, pt, kt = (base.site == target, cov.site == target,
                      prior.site == target, concept.site == target)
    # cov / concept change the TARGET signal but NOT its labels (only prior changes labels)
    assert not np.array_equal(base.X[bt], cov.X[ct]), "cov did not change target X"
    assert np.array_equal(base.y[bt], cov.y[ct]), "cov changed target labels (should not)"
    assert not np.array_equal(base.X[bt], concept.X[kt]), "concept did not change target X"
    assert np.array_equal(base.y[bt], concept.y[kt]), "concept changed target labels (should not)"
    # prior DOES change target labels
    assert not np.array_equal(base.y[bt], prior.y[pt]), "prior did not change target labels"


def test_order_independent_and_deterministic():
    a = _sample("cov", 1)
    b = _sample("cov", 1)
    assert np.array_equal(a.X, b.X) and np.array_equal(a.y, b.y), "sampling not deterministic"


if __name__ == "__main__":
    test_source_identical_across_scenarios()
    test_mechanisms_are_orthogonal()
    test_order_independent_and_deterministic()
    print("test_shift_pairing PASSED")
