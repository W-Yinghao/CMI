"""Bayes-risk preservation (Prop. 1): at overlap=0 removing the nuisance subspace keeps
label accuracy while removing the leakage; under full overlap the selector refuses, so
accuracy is preserved trivially (identity)."""
import torch

from tos_cmi.data.synthetic import SynthSpec, make, make_collinear
from tos_cmi.subspace import SubspaceSelector
from tos_cmi.eval.proposition import bayes_risk_check


def _check_data(data, seed=0):
    s = data["spec"]
    sel = SubspaceSelector(s.d, s.n_cls, s.n_dom)
    sel.refresh(torch.tensor(data["Z"]), torch.tensor(data["y"]), torch.tensor(data["d"]))
    return bayes_risk_check(data, sel, seed=seed), sel


def _check(overlap, seed=0):
    return _check_data(make(SynthSpec(n=4000, overlap=overlap), seed=seed), seed=seed)


def test_risk_preserved_and_leakage_removed_orthogonal():
    p, sel = _check(0.0)
    assert not sel.is_identity
    # removing the nuisance subspace barely moves accuracy ...
    assert p["acc_drop"] < 0.05, p
    # ... but removes the conditional-domain leakage that was concentrated in it
    assert p["leak_nuis"] > 0.15, p
    assert p["leak_task"] < 0.5 * p["leak_full"] + 1e-6, p
    print("test_risk_preserved_and_leakage_removed_orthogonal: OK", p)


def test_refuses_under_overlap():
    p, sel = _check_data(make_collinear(seed=0))
    assert sel.is_identity
    assert abs(p["acc_drop"]) < 1e-6      # identity => task == full, risk untouched
    # there genuinely is leakage in Z that a naive global method would chase
    assert p["leak_full"] > 0.1, p
    print("test_refuses_under_overlap: OK", p)


if __name__ == "__main__":
    test_risk_preserved_and_leakage_removed_orthogonal()
    test_refuses_under_overlap()
    print("ALL PROPOSITION TESTS PASSED")
