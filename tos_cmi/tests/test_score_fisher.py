"""Phase-1 score-Fisher acceptance tests (the novelty over the frozen mean-scatter baseline).

1. metric-aware projector is similarity-covariant under coordinate change (pure linear
   algebra -- isolates the projector formula from probe-training variance);
2. it RECOVERS covariance-only domain leakage that mean-scatter is blind to (headline);
3. it degrades to identity when domain is collinear with the discriminant;
4. its leakage gate stays shut on pure noise.

Stressors still to add as failing-first tests (next increment): XOR end-to-end recovery,
class-specific domain shift, imbalanced/missing (d,y) cells, full rescaling end-to-end.
"""
import numpy as np
import torch

from tos_cmi.data.synthetic import (SynthSpec, make, make_collinear, make_covariance_only,
                                     apply_linear_transform)
from tos_cmi.score_fisher import (ScoreFisherConfig, ScoreFisherSelector, metric_projector,
                                   select_score_fisher)
from tos_cmi.subspace import SubspaceSelector
from tos_cmi.eval.stability import precision_recall, projection_distance


def _fast_cfg():
    return ScoreFisherConfig(epochs=200, hidden=64, gate_boot=200)


def test_metric_projector_similarity_covariant():
    """P(A^{-T} M A^{-1}; A V) == A P(M; V) A^{-1}  ->  selection invariant to z->Az."""
    rng = np.random.default_rng(0)
    d, k = 8, 3
    Mh = rng.standard_normal((d, d)); M = Mh @ Mh.T + np.eye(d)        # SPD metric
    V = rng.standard_normal((d, k))
    Ah = rng.standard_normal((d, d)); A = Ah + 2 * np.eye(d)           # invertible
    P = metric_projector(V, M)
    P_t = metric_projector(A @ V, np.linalg.inv(A).T @ M @ np.linalg.inv(A))
    # map transformed projector back to original coords; should match P
    back = np.linalg.inv(A) @ P_t @ A
    assert np.linalg.norm(back - P) < 1e-8, np.linalg.norm(back - P)
    assert np.linalg.norm(P @ P - P) < 1e-8                            # idempotent
    print("test_metric_projector_similarity_covariant: OK")


def test_recovers_covariance_leakage_where_meanscatter_is_blind():
    data = make_covariance_only(n=4000, seed=0)
    s = data["spec"]
    # mean-scatter baseline: blind -> identity
    ms = SubspaceSelector(s.d, s.n_cls, s.n_dom)
    ms.refresh(torch.tensor(data["Z"]), torch.tensor(data["y"]), torch.tensor(data["d"]))
    assert ms.is_identity, "mean-scatter should be blind to covariance leakage"
    # score-Fisher: gate opens and it selects a subspace aligned with the carrier
    sf = ScoreFisherSelector(s.d, s.n_cls, s.n_dom, _fast_cfg())
    rep = sf.refresh(torch.tensor(data["Z"]), torch.tensor(data["y"]), torch.tensor(data["d"]))
    assert rep.gate_open, sf.summary()
    assert not sf.is_identity, sf.summary()
    pr = precision_recall(rep.basis, data["nuisance_basis"])
    assert pr["precision"] > 0.6, (pr, sf.summary())     # selection sits in the true carrier plane
    print("test_recovers_covariance_leakage: OK", {"k": rep.k, "precision": round(pr["precision"], 3),
          "critic_adv": round(rep.critic_adv, 3)})


def test_identity_when_collinear():
    data = make_collinear(n=4000, seed=0)
    s = data["spec"]
    sf = ScoreFisherSelector(s.d, s.n_cls, s.n_dom, _fast_cfg())
    sf.refresh(torch.tensor(data["Z"]), torch.tensor(data["y"]), torch.tensor(data["d"]))
    assert sf.is_identity, sf.summary()        # domain collinear with discriminant -> no safe subspace
    print("test_identity_when_collinear: OK", sf.summary())


def test_gate_shut_on_pure_noise():
    data = make(SynthSpec(n=4000, sep_label=0.0, sep_dom=0.0), seed=0)
    s = data["spec"]
    sf = ScoreFisherSelector(s.d, s.n_cls, s.n_dom, _fast_cfg())
    rep = sf.refresh(torch.tensor(data["Z"]), torch.tensor(data["y"]), torch.tensor(data["d"]))
    assert not rep.gate_open, sf.summary()     # no detectable leakage
    assert sf.is_identity
    print("test_gate_shut_on_pure_noise: OK", {"critic_adv_lb": round(rep.critic_adv_lb, 4)})


if __name__ == "__main__":
    test_metric_projector_similarity_covariant()
    test_recovers_covariance_leakage_where_meanscatter_is_blind()
    test_identity_when_collinear()
    test_gate_shut_on_pure_noise()
    print("ALL SCORE-FISHER TESTS PASSED")
