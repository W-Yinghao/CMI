"""The selector's selected directions lie inside the planted nuisance span (precision~1)
when one is risk-feasible, and it degrades to identity (refuses to delete) when the domain
shift is collinear with the class discriminant.

Honest framing (reviewer's point): the default planted nuisance rank is 4 but the selector
keeps only the highest-ratio few, so this is a PRECISION claim (selection is real), NOT full
recovery (recall can be < 1). Both are reported."""
import torch

from tos_cmi.data.synthetic import SynthSpec, make, make_collinear
from tos_cmi.subspace import SubspaceSelector
from tos_cmi.eval.stability import precision_recall


def _fit(data):
    s = data["spec"]
    sel = SubspaceSelector(s.d, s.n_cls, s.n_dom)
    sel.refresh(torch.tensor(data["Z"]), torch.tensor(data["y"]), torch.tensor(data["d"]))
    return sel, data


def _select(overlap, seed=0):
    return _fit(make(SynthSpec(n=4000, overlap=overlap), seed=seed))


def test_selection_precise_within_planted_span():
    sel, data = _select(0.0)
    assert not sel.is_identity
    assert sel.report.k > 0
    pr = precision_recall(sel.report.basis, data["nuisance_basis"])
    assert pr["precision"] > 0.85, pr      # what we DELETE is genuinely nuisance
    print("test_selection_precise_within_planted_span: OK  k=%d  precision=%.3f recall=%.3f"
          % (sel.report.k, pr["precision"], pr["recall"]))


def test_identity_when_collinear():
    # domain shift collinear with the class discriminant -> no safe nuisance subspace
    sel, _ = _fit(make_collinear(seed=0))
    assert sel.is_identity, sel.summary()
    assert sel.report.k == 0
    # projector is the zero map -> projecting any Z gives zeros
    z = torch.randn(8, sel.z_dim)
    assert torch.allclose(sel.project(z), torch.zeros_like(z))
    print("test_identity_when_collinear: OK (refused to delete)")


def test_deletable_subspace_does_not_grow_with_overlap():
    ks = [_select(ov)[0].report.k for ov in (0.0, 0.4, 0.8)]
    # entanglement never *enlarges* the risk-feasible nuisance subspace
    assert ks[0] >= ks[-1], ks
    print("test_deletable_subspace_does_not_grow_with_overlap: OK  k(overlap)=", ks)


if __name__ == "__main__":
    test_selection_precise_within_planted_span()
    test_identity_when_collinear()
    test_deletable_subspace_does_not_grow_with_overlap()
    print("ALL SUBSPACE/IDENTITY TESTS PASSED")
