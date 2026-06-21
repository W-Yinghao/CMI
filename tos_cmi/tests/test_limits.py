"""Honest limitation: the current Fishers are FIRST-MOMENT (mean-scatter) statistics. On a
domain leakage that lives in the *covariance* (zero domain-mean), the selector is blind and
returns identity -- even though the domain is decodable from Z|Y by a quadratic probe. This
is the concrete evidence that the method is `label-mean-scatter-light`, NOT
`task/domain-orthogonal`, and motivates the score-Fisher / SPD extension (next phase)."""
import numpy as np
import torch

from tos_cmi.data.synthetic import make_covariance_only
from tos_cmi.subspace import SubspaceSelector
from tos_cmi.eval.projection_ablation import _cond_domain_adv


def test_mean_scatter_selector_blind_to_covariance_leakage():
    data = make_covariance_only(n=6000, seed=0)
    s = data["spec"]
    sel = SubspaceSelector(s.d, s.n_cls, s.n_dom)
    sel.refresh(torch.tensor(data["Z"]), torch.tensor(data["y"]), torch.tensor(data["d"]))

    # the first-moment selector sees ~no domain-MEAN shift -> identity (honest no-op)
    assert sel.is_identity, sel.summary()

    # but the leakage is REAL: a probe that looks at the VARIANCE along the carrier recovers
    # domain, while a purely linear probe cannot. We use the known carrier w only to PROVE the
    # leakage exists (not as part of the method) -- the squared projection (Z @ w)^2 is the
    # second-order feature the first-moment selector structurally cannot use.
    Z, y, d = data["Z"], data["y"], data["d"]
    w = data["nuisance_basis"][:, 0]
    Zw2 = ((Z @ w) ** 2)[:, None]
    n = len(Z); cut = n // 2
    lin = _cond_domain_adv(Z[:cut], y[:cut], d[:cut], Z[cut:], y[cut:], d[cut:], s.n_cls, s.n_dom)
    Zq = np.concatenate([Z, Zw2], 1)
    quad = _cond_domain_adv(Zq[:cut], y[:cut], d[:cut], Zq[cut:], y[cut:], d[cut:], s.n_cls, s.n_dom)
    assert quad > lin + 0.10, (lin, quad)
    print(f"test_mean_scatter_selector_blind_to_covariance_leakage: OK "
          f"(identity; linear domadv={lin:.3f} << variance-feature domadv={quad:.3f})")


if __name__ == "__main__":
    test_mean_scatter_selector_blind_to_covariance_leakage()
    print("ALL LIMITATION TESTS PASSED")
