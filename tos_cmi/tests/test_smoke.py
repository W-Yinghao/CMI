"""End-to-end: refresh -> Step-A critic fit -> Step-B penalty -> grad to encoder, with a
finite non-negative penalty; and the identity case is an exact no-op (penalty == 0)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from tos_cmi.data.synthetic import SynthSpec, make, make_collinear
from tos_cmi.config import PenaltyConfig
from tos_cmi.selective_cmi import SelectivePenalty


def _setup(data):
    s = data["spec"]
    Z = torch.tensor(data["Z"]); y = torch.tensor(data["y"]); d = torch.tensor(data["d"])
    enc = nn.Linear(s.d, s.d)
    sp = SelectivePenalty(s.d, s.n_cls, s.n_dom, priors=(data["y"], data["d"]),
                          pcfg=PenaltyConfig(lam=1.0))
    return data, enc, sp, Z, y, d


def test_end_to_end_step():
    data, enc, sp, Z, y, d = _setup(make(SynthSpec(n=2000, overlap=0.0), seed=0))
    zt = enc(Z)
    sp.refresh(zt.detach(), y, d)
    assert not sp.is_identity

    # Step A: critic fits on detached projected features
    la = sp.posterior_loss(zt.detach(), y, d)
    assert torch.isfinite(la) and la.item() >= 0

    # Step B: penalty is finite, non-negative (KL>=0), and back-props to the encoder
    pen = sp.penalty(zt, y, d)
    assert torch.isfinite(pen) and pen.item() >= -1e-6
    (F.cross_entropy(enc(Z), y) + pen).backward()
    assert enc.weight.grad is not None and torch.isfinite(enc.weight.grad).all()
    print("test_end_to_end_step: OK  penalty=%.4f" % pen.item())


def test_identity_is_noop():
    data, enc, sp, Z, y, d = _setup(make_collinear(n=2000, seed=0))
    zt = enc(Z)
    sp.refresh(zt.detach(), y, d)
    assert sp.is_identity
    assert sp.posterior_loss(zt.detach(), y, d).item() == 0.0
    assert sp.penalty(zt, y, d).item() == 0.0
    print("test_identity_is_noop: OK")


if __name__ == "__main__":
    test_end_to_end_step()
    test_identity_is_noop()
    print("ALL SMOKE TESTS PASSED")
