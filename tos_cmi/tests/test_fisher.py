"""F_Y concentrates on the label subspace; F_{D|Y} on the (conditional) domain subspace;
at overlap=0 the two are mutually orthogonal."""
import numpy as np
import torch

from tos_cmi.data.synthetic import SynthSpec, make
from tos_cmi.fisher import fisher_pair
from tos_cmi.eval.stability import subspace_overlap


def _top_eigvecs(S, k):
    w, V = torch.linalg.eigh(S.double())          # ascending
    return V[:, -k:].cpu().numpy()


def test_fisher_alignment_orthogonal_case():
    data = make(SynthSpec(n=4000, overlap=0.0), seed=0)
    Z = torch.tensor(data["Z"]); y = torch.tensor(data["y"]); d = torch.tensor(data["d"])
    s = data["spec"]
    F_DgY, F_Y = fisher_pair(Z, y, d, s.n_cls, s.n_dom)

    assert F_DgY.shape == (s.d, s.d) and F_Y.shape == (s.d, s.d)
    # PSD up to numerical tolerance
    assert torch.linalg.eigvalsh(F_DgY).min() > -1e-6
    assert torch.linalg.eigvalsh(F_Y).min() > -1e-6

    # F_Y's top directions span the label subspace; F_{D|Y}'s the nuisance subspace
    lab_vecs = _top_eigvecs(F_Y, data["label_basis"].shape[1])
    nuis_vecs = _top_eigvecs(F_DgY, data["nuisance_basis"].shape[1])
    assert subspace_overlap(lab_vecs, data["label_basis"]) > 0.85
    assert subspace_overlap(nuis_vecs, data["nuisance_basis"]) > 0.85

    # cross-alignment is small at overlap=0 (task-orthogonal nuisance)
    assert subspace_overlap(nuis_vecs, data["label_basis"]) < 0.30
    print("test_fisher_alignment_orthogonal_case: OK")


if __name__ == "__main__":
    test_fisher_alignment_orthogonal_case()
    print("ALL FISHER TESTS PASSED")
