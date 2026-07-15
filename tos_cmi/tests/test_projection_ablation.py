"""Leakage-free projection ablation: P_N is estimated on selector-train only; metrics are
read on a disjoint probe-test. At overlap=0 removing the nuisance subspace keeps label
accuracy while removing the (linear) conditional-domain advantage; under collinear domain
shift the selector refuses, so accuracy is preserved trivially (identity)."""
import torch

from tos_cmi.data.synthetic import SynthSpec, make, make_collinear
from tos_cmi.eval.projection_ablation import linear_probe_projection_ablation


def test_risk_preserved_and_advantage_removed_orthogonal():
    data = make(SynthSpec(n=6000, overlap=0.0), seed=0)
    p, sel = linear_probe_projection_ablation(data, seed=0)
    assert not p["is_identity"]
    assert p["acc_drop"] < 0.05, p                 # removing P_N barely moves label accuracy
    assert p["domadv_nuis"] > 0.15, p             # the linear domain advantage sat in P_N
    assert p["domadv_task"] < 0.5 * p["domadv_full"] + 1e-6, p
    print("test_risk_preserved_and_advantage_removed_orthogonal: OK", p)


def test_refuses_under_collinear():
    data = make_collinear(n=6000, seed=0)
    p, sel = linear_probe_projection_ablation(data, seed=0)
    assert p["is_identity"]
    assert abs(p["acc_drop"]) < 1e-6              # identity => task == full
    assert p["domadv_full"] > 0.1, p             # there genuinely is leakage a naive method would chase
    print("test_refuses_under_collinear: OK", p)


if __name__ == "__main__":
    test_risk_preserved_and_advantage_removed_orthogonal()
    test_refuses_under_collinear()
    print("ALL PROJECTION-ABLATION TESTS PASSED")
