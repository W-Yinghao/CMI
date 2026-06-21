"""Conditional-domain adversary: critic/encoder gradient signs, no gradient on ineligible
cells, fixed p_ref class weighting under batch reweighting, and a safe no-op when there is no
comparable class.

Standalone (``python -m oaci.tests.test_train_adversary``) and pytest-compatible.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from oaci.train.adversary import ConditionalDomainAdversary
from oaci.support_graph import build_support_graph, counts_from_labels, empirical_class_prior


def _sg(d, y, m, nd, ncl, pref=None):
    counts = counts_from_labels(d, y, n_domains=nd, n_classes=ncl)
    return build_support_graph(counts, m=m, reference_prior=pref if pref is not None else empirical_class_prior(counts))


def _balanced(nd=2, ncl=2, per=40, seed=0):
    rng = np.random.default_rng(seed)
    y, d = [], []
    for dom in range(nd):
        for c in range(ncl):
            y += [c] * per
            d += [dom] * per
    y, d = np.array(y), np.array(d)
    Z = np.concatenate([np.eye(ncl)[y] * 0.5, rng.standard_normal((y.size, 3))], axis=1)
    Z[:, -1] += 3.0 * d                                  # a domain-separating direction
    return torch.tensor(Z, dtype=torch.float32), y, d


def test_critic_and_encoder_gradient_signs():
    Z, y, d = _balanced()
    sg = _sg(d, y, m=20, nd=2, ncl=2)
    adv = ConditionalDomainAdversary(Z.shape[1], sg)
    opt = torch.optim.Adam(adv.parameters(), lr=1e-2)
    cd0 = adv.domain_ce(Z, y, d).item()
    for _ in range(60):                                  # critic minimises C_D (encoder frozen)
        opt.zero_grad(); adv.domain_ce(Z, y, d).backward(); opt.step()
    cd_after_critic = adv.domain_ce(Z, y, d).item()
    assert cd_after_critic < cd0 - 1e-3, (cd0, cd_after_critic)

    # encoder ascends C_D: minimise -C_D over Z (critic frozen) -> C_D must rise
    for p in adv.parameters():
        p.requires_grad_(False)
    Zp = Z.clone().requires_grad_(True)
    optz = torch.optim.Adam([Zp], lr=5e-2)
    cd_start = adv.domain_ce(Zp, y, d).item()
    for _ in range(60):
        optz.zero_grad(); (-adv.domain_ce(Zp, y, d)).backward(); optz.step()
    assert adv.domain_ce(Zp, y, d).item() > cd_start + 1e-3


def test_ineligible_cells_have_no_explicit_adversary_gradient():
    # domain 2 holds only 5 class-0 samples (< m) -> (2, class0) ineligible -> no adversary term
    y = np.array([0] * 40 + [1] * 40 + [0] * 40 + [1] * 40 + [0] * 5)
    d = np.array([0] * 80 + [1] * 80 + [2] * 5)
    sg = _sg(d, y, m=20, nd=3, ncl=2)
    assert 2 not in sg.support_of_class[0]
    Z = torch.randn(y.size, 4, requires_grad=True)
    adv = ConditionalDomainAdversary(4, sg)
    adv.domain_ce(Z, y, d).backward()
    inelig = d == 2
    assert Z.grad[inelig].abs().sum().item() == 0.0      # ineligible rows: exactly zero gradient
    assert Z.grad[~inelig].abs().sum().item() > 0.0      # eligible rows do get gradient


def test_fixed_p_ref_under_batch_reweighting():
    Z, y, d = _balanced()
    pref = np.array([0.7, 0.3])                           # non-uniform so it differs from batch freq
    sg = _sg(d, y, m=20, nd=2, ncl=2, pref=pref)
    adv = ConditionalDomainAdversary(Z.shape[1], sg)
    assert adv.class_weights() == {0: 0.7, 1: 0.3}

    def manual(sample_weight=None):
        total = 0.0
        for yy in (0, 1):
            mask = (y == yy)
            labels = torch.tensor([adv.dmap[yy][int(dd)] for dd in d[mask]])
            ce = F.cross_entropy(adv.heads[str(yy)](Z[mask]), labels, reduction="none")
            if sample_weight is not None:
                w = torch.as_tensor(sample_weight)[mask].float()
                cem = (ce * w).sum() / w.sum()
            else:
                cem = ce.mean()
            total += pref[yy] * float(cem.item())          # p_ref weighting, NOT batch frequency
        return total

    assert abs(adv.domain_ce(Z, y, d).item() - manual()) < 1e-5
    w = np.ones(y.size); w[y == 0] *= 5.0                  # reweight within class 0
    assert abs(adv.domain_ce(Z, y, d, sample_weight=w).item() - manual(w)) < 1e-5
    assert adv.class_weights() == {0: 0.7, 1: 0.3}         # class weights still fixed p_ref


def test_no_comparable_classes_is_safe_noop():
    y = np.array([0] * 40 + [1] * 40)
    d = np.array([0] * 80)                                 # single domain -> no comparable class
    sg = _sg(d, y, m=20, nd=1, ncl=2)
    assert sg.comparable_classes == []
    Z = torch.randn(80, 4, requires_grad=True)
    cd = ConditionalDomainAdversary(4, sg).domain_ce(Z, y, d)
    assert float(cd.item()) == 0.0
    assert not cd.requires_grad                            # no graph -> safe no-op (no backprop)


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} train-adversary tests")


if __name__ == "__main__":
    _run_all()
