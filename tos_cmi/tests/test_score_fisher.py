"""Score-Fisher tests after the Phase-1.1 correctness patch.

Fast algebra tests (no probe training):
  * metric-aware projector is similarity-covariant;
  * select_from_fishers (eig + selection + projector) is EXACTLY covariant under z->Az
    (the transported-probe claim of fix 4);
  * checkpoint reload preserves is_identity via the active_k buffer.

Probe-training tests (run on SLURM):
  * RECOVERS covariance-only leakage where mean-scatter is blind (headline);
  * identity under collinear domain shift; leakage gate shut on pure noise;
  * refitted-probe selection is EMPIRICALLY stable under a coordinate rescaling (loose --
    Adam/weight-decay are not equivariant, so this is NOT an exact-covariance claim).

Targeted stressors (all-safe / all-dangerous+saturation / partial) and the source-risk UCB
rank gate are the next sub-steps, not here.
"""
import numpy as np
import torch

from tos_cmi.data.synthetic import (SynthSpec, make, make_collinear, make_covariance_only,
                                     apply_linear_transform)
from tos_cmi.score_fisher import (ScoreFisherConfig, ScoreFisherSelector, metric_projector,
                                   select_from_fishers)
from tos_cmi.subspace import SubspaceSelector
from tos_cmi.eval.stability import precision_recall, projection_distance


def _fast_cfg():
    return ScoreFisherConfig(epochs=200, hidden=64, gate_boot=200, n_perm_null=2)


# ----------------------------------------------------------------- fast algebra tests
def test_metric_projector_similarity_covariant():
    rng = np.random.default_rng(0); d, k = 8, 3
    Mh = rng.standard_normal((d, d)); M = Mh @ Mh.T + np.eye(d)
    V = rng.standard_normal((d, k))
    A = rng.standard_normal((d, d)) + 2 * np.eye(d)
    P = metric_projector(V, M)
    P_t = metric_projector(A @ V, np.linalg.inv(A).T @ M @ np.linalg.inv(A))
    assert np.linalg.norm(np.linalg.inv(A) @ P_t @ A - P) < 1e-8
    assert np.linalg.norm(P @ P - P) < 1e-8
    print("test_metric_projector_similarity_covariant: OK")


def test_select_from_fishers_exactly_covariant():
    """Transported-probe (exact): selection + projector covariant under z->Az."""
    d = 6
    G_DgY = np.diag([10.0, 1.0, 0.05, 0.05, 0.05, 0.05])     # e0 domain-rich
    G_Y = np.diag([0.01, 8.0, 0.01, 0.01, 0.01, 0.01])       # e0 label-light, e1 label-rich
    rng = np.random.default_rng(1)
    Mh = rng.standard_normal((d, d)); M = Mh @ Mh.T + np.eye(d)
    A = rng.standard_normal((d, d)) + 2.5 * np.eye(d)
    Ai = np.linalg.inv(A)
    cfg = ScoreFisherConfig()
    c1, _, _, _, _, _, P1 = select_from_fishers(G_DgY, G_Y, M, cfg)
    Gd2, Gy2, M2 = Ai.T @ G_DgY @ Ai, Ai.T @ G_Y @ Ai, Ai.T @ M @ Ai
    c2, _, _, _, _, _, P2 = select_from_fishers(Gd2, Gy2, M2, cfg)
    assert c1.size > 0 and np.array_equal(c1, c2), (c1, c2)
    assert np.linalg.norm(Ai @ P2 @ A - P1) < 1e-6
    print("test_select_from_fishers_exactly_covariant: OK  k=%d" % c1.size)


def test_checkpoint_preserves_identity():
    sf = ScoreFisherSelector(6, 2, 3, _fast_cfg())
    sf.active_k = torch.tensor(2, dtype=torch.long)          # pretend a non-identity selection
    sf.P = torch.eye(6)
    sf2 = ScoreFisherSelector(6, 2, 3, _fast_cfg())
    sf2.load_state_dict(sf.state_dict())
    assert not sf2.is_identity and int(sf2.active_k) == 2     # survives reload (report is None)
    print("test_checkpoint_preserves_identity: OK")


# ----------------------------------------------------------------- probe-training tests
def test_recovers_covariance_leakage_where_meanscatter_is_blind():
    data = make_covariance_only(n=4000, seed=0)
    s = data["spec"]
    ms = SubspaceSelector(s.d, s.n_cls, s.n_dom)
    ms.refresh(torch.tensor(data["Z"]), torch.tensor(data["y"]), torch.tensor(data["d"]))
    assert ms.is_identity, "mean-scatter should be blind to covariance leakage"
    sf = ScoreFisherSelector(s.d, s.n_cls, s.n_dom, _fast_cfg())
    rep = sf.refresh(torch.tensor(data["Z"]), torch.tensor(data["y"]), torch.tensor(data["d"]))
    assert rep.gate_open, sf.summary()
    assert not sf.is_identity, sf.summary()
    pr = precision_recall(rep.basis, data["nuisance_basis"])
    assert pr["precision"] > 0.6, (pr, sf.summary())
    print("test_recovers_covariance_leakage: OK", {"k": rep.k, "precision": round(pr["precision"], 3),
          "leak_gain": round(rep.leak_gain, 4)})


def test_identity_when_collinear():
    data = make_collinear(n=4000, seed=0)
    s = data["spec"]
    sf = ScoreFisherSelector(s.d, s.n_cls, s.n_dom, _fast_cfg())
    sf.refresh(torch.tensor(data["Z"]), torch.tensor(data["y"]), torch.tensor(data["d"]))
    assert sf.is_identity, sf.summary()
    print("test_identity_when_collinear: OK", sf.summary())


def test_gate_shut_on_pure_noise():
    data = make(SynthSpec(n=4000, sep_label=0.0, sep_dom=0.0), seed=0)
    s = data["spec"]
    sf = ScoreFisherSelector(s.d, s.n_cls, s.n_dom, _fast_cfg())
    rep = sf.refresh(torch.tensor(data["Z"]), torch.tensor(data["y"]), torch.tensor(data["d"]))
    assert not rep.gate_open, sf.summary()
    assert sf.is_identity
    print("test_gate_shut_on_pure_noise: OK", {"leak_lcb": round(rep.leak_lcb, 4)})


def test_refitted_probe_selection_empirically_stable_under_rescaling():
    """Refitted-probe (empirical, loose): re-train the whole selector on Z and on AZ; the
    selected subspace, mapped back, should be CLOSE -- not exact (non-convex training)."""
    data = make_covariance_only(n=4000, seed=0)
    s = data["spec"]
    rng = np.random.default_rng(3)
    A = np.diag(np.exp(rng.uniform(-0.5, 0.5, s.d))).astype(np.float32)   # diagonal rescaling
    cfg = _fast_cfg()
    sf1 = ScoreFisherSelector(s.d, s.n_cls, s.n_dom, cfg)
    r1 = sf1.refresh(torch.tensor(data["Z"]), torch.tensor(data["y"]), torch.tensor(data["d"]))
    data2 = apply_linear_transform(data, A)
    sf2 = ScoreFisherSelector(s.d, s.n_cls, s.n_dom, cfg)
    r2 = sf2.refresh(torch.tensor(data2["Z"]), torch.tensor(data2["y"]), torch.tensor(data2["d"]))
    assert r1.k > 0 and r2.k > 0, (sf1.summary(), sf2.summary())
    # map r2's basis back to original coords (basis transforms as A v): A^{-1} V2
    back = np.linalg.inv(A) @ r2.basis
    pd = projection_distance(r1.basis, back)
    assert pd < 0.8, pd        # close subspaces (loose), not identical
    print("test_refitted_probe_selection_empirically_stable: OK  proj_dist=%.3f" % pd)


if __name__ == "__main__":
    test_metric_projector_similarity_covariant()
    test_select_from_fishers_exactly_covariant()
    test_checkpoint_preserves_identity()
    test_recovers_covariance_leakage_where_meanscatter_is_blind()
    test_identity_when_collinear()
    test_gate_shut_on_pure_noise()
    test_refitted_probe_selection_empirically_stable_under_rescaling()
    print("ALL SCORE-FISHER TESTS PASSED")
