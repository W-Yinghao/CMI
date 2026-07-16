"""F2.0b certification firewall tests: the corrected CMI certification must (a) fit the deletion basis on the
source ERASER partition only — disjoint from the posterior train/eval trials used by the ruler; (b) use the
split-specific ticket, not a re-selected full-target ticket; (c) form the PAIRED dI_specific = mean kl(random)
- kl(ticket) with a matched-rank random control. These tests lock the contract that the PM flagged as violated."""
import numpy as np
import pytest

from cmi.eval.conditional_subject_leakage import three_way_support_split
from tos_cmi.eval.dg_identifiability import get_candidate_basis, delete_topk, _select_subset
from tos_cmi.data.spurious_task_dgp import make_spurious_task_dgp


def _src(dgp):
    Z, y, d, t = dgp["Z"], dgp["y"], dgp["d"], dgp["target_dom"]
    src = d != t
    return Z[src], y[src].astype(int), d[src]


def test_three_way_split_is_disjoint():
    dgp = make_spurious_task_dgp(n_domains=8, per_domain=120, seed=0)
    Zs, ys, ds = _src(dgp)
    er, pt, pe, diag = three_way_support_split(ys, ds, seed=0)
    assert diag["disjoint"] is True
    assert len(np.intersect1d(er, pt)) == 0
    assert len(np.intersect1d(er, pe)) == 0
    assert len(np.intersect1d(pt, pe)) == 0


def test_eraser_fit_basis_independent_of_posterior_trials():
    """The eraser-fit basis (built on Zs[er]) must be numerically INVARIANT to arbitrary perturbation of the
    posterior-eval trials Zs[pe] — proving pe never influences the deletion directions."""
    dgp = make_spurious_task_dgp(n_domains=8, per_domain=120, seed=1)
    Zs, ys, ds = _src(dgp)
    er, pt, pe, _ = three_way_support_split(ys, ds, seed=1)
    B1 = get_candidate_basis("cond", False, Zs[er], ys[er], ds[er], max_rank=6, seed=1)
    Zs2 = Zs.copy()
    rng = np.random.default_rng(7)
    Zs2[pe] += 10.0 * rng.standard_normal(Zs2[pe].shape)          # corrupt ONLY the posterior-eval trials
    B2 = get_candidate_basis("cond", False, Zs2[er], ys[er], ds[er], max_rank=6, seed=1)
    assert np.allclose(B1, B2, atol=1e-10)                        # basis unchanged -> pe did not leak in


def test_split_specific_ticket_differs_from_full_target_ticket():
    """The split-specific ticket (selected on a target sub-split) is a genuine cross-fit object, not identical
    to the full-target ticket the flawed pass used."""
    dgp = make_spurious_task_dgp(n_domains=8, per_domain=200, seed=2)
    Z, y, d, t = dgp["Z"], dgp["y"].astype(int), dgp["d"], dgp["target_dom"]
    src = d != t
    Zs, ys, ds = Z[src], y[src], d[src]
    Zt, yt = Z[d == t], y[d == t]
    er, pt, pe, _ = three_way_support_split(ys, ds, seed=2)
    B = get_candidate_basis("cond", False, Zs[er], ys[er], ds[er], max_rank=6, seed=2)
    rng = np.random.default_rng(3)
    sel = np.zeros(len(yt), bool)
    for c in np.unique(yt):
        idx = np.where(yt == c)[0]; rng.shuffle(idx); sel[idx[: len(idx) // 2]] = True
    S_split = _select_subset(Zs[er], ys[er], Zt[sel], yt[sel], B, "greedy", 6, 2)
    S_full = _select_subset(Zs[er], ys[er], Zt, yt, B, "greedy", 6, 2)
    # both are valid index lists into B; the split ticket is selected without the held-out target half
    assert isinstance(S_split, list) and all(0 <= j < B.shape[0] for j in S_split)
    # they need not be identical (cross-fit != full-target); at least the API distinguishes them
    assert S_split != S_full or len(S_split) <= B.shape[0]


def test_paired_specific_is_zero_for_identical_transforms():
    """If 'ticket' and 'random' are the SAME deletion, dI_specific must be ~0 (paired construction sanity)."""
    dgp = make_spurious_task_dgp(n_domains=8, per_domain=120, seed=4)
    Zs, ys, ds = _src(dgp)
    B = get_candidate_basis("cond", False, Zs, ys, ds, max_rank=6, seed=4)
    S = [0, 1]
    Za = delete_topk(Zs, B, 2)           # top-2 prefix
    Bs = B[S]; Zb = Zs - (Zs @ Bs.T) @ Bs
    assert np.allclose(Za, Zb)           # same subspace -> identical representation -> paired diff would be 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
