"""F_Y concentrates on the label MEAN subspace; F_{D|Y} on the (conditional) domain MEAN
subspace; at overlap=0 the two are mutually (near-)orthogonal.

NOTE these are first-moment (mean-scatter) statistics; this test only certifies recovery of
PLANTED MEAN structure. It says nothing about covariance/nonlinear task or domain
information (see test_limits.py for the honest counterexample). Thresholds are kept margin-
safe and n is large so the assertion reproduces across environments (CPU, py3.9-3.13)."""
import torch

from tos_cmi.data.synthetic import SynthSpec, make
from tos_cmi.fisher import fisher_pair
from tos_cmi.eval.stability import subspace_cos2_similarity


def _top_eigvecs(S, k):
    w, V = torch.linalg.eigh(S.double())          # ascending
    return V[:, -k:].cpu().numpy()


def test_fisher_alignment_orthogonal_case():
    data = make(SynthSpec(n=6000, overlap=0.0), seed=0)
    Z = torch.tensor(data["Z"]); y = torch.tensor(data["y"]); d = torch.tensor(data["d"])
    s = data["spec"]
    F_DgY, F_Y = fisher_pair(Z, y, d, s.n_cls, s.n_dom)

    assert F_DgY.shape == (s.d, s.d) and F_Y.shape == (s.d, s.d)
    assert torch.linalg.eigvalsh(F_DgY).min() > -1e-6     # PSD up to tolerance
    assert torch.linalg.eigvalsh(F_Y).min() > -1e-6

    lab_vecs = _top_eigvecs(F_Y, data["label_basis"].shape[1])
    nuis_vecs = _top_eigvecs(F_DgY, data["nuisance_basis"].shape[1])
    sim_lab = subspace_cos2_similarity(lab_vecs, data["label_basis"])
    sim_nuis = subspace_cos2_similarity(nuis_vecs, data["nuisance_basis"])
    sim_cross = subspace_cos2_similarity(nuis_vecs, data["label_basis"])
    assert sim_lab > 0.80, sim_lab
    assert sim_nuis > 0.80, sim_nuis
    assert sim_cross < 0.35, sim_cross      # task-orthogonal (in mean) at overlap=0
    print(f"test_fisher_alignment_orthogonal_case: OK  lab={sim_lab:.3f} "
          f"nuis={sim_nuis:.3f} cross={sim_cross:.3f}")


if __name__ == "__main__":
    test_fisher_alignment_orthogonal_case()
    print("ALL FISHER TESTS PASSED")
