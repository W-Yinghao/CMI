"""Rare-cell paired-stream sampler + the four residual trainer fixes.

Coverage, exact prior restoration (mass-based, classifier-free), ineligible-cell handling in
each stream, microbatch-accumulation partition invariance, loud capacity failure, byte-exact
no-comparable no-op, exact-Lagrangian λ_floor=0, missing-cell mask respect, reproducibility.

Standalone (``python -m oaci.tests.test_rare_cell_sampler``) and pytest-compatible.
"""
from __future__ import annotations

import numpy as np
import torch

from oaci.config import SamplerConfig
from oaci.data.batch import (
    effective_prior_domain_given_y,
    effective_prior_y,
    fixed_prior_domain_given_y,
)
from oaci.data.sampler import RareCellSampler
from oaci.support_graph import build_support_graph, counts_from_labels, empirical_class_prior
from oaci.train.adversary import ConditionalDomainAdversary
from oaci.train.primal_dual import TrainConfig, effective_risk_weight, train_risk_feasible
from oaci.train.risk import source_risk
from oaci.train.selector import select_checkpoint, state_hash
from oaci.train.synthetic import make_covariate_shift


def _imbalanced(seed=0, m=10, ineligible=True):
    # 50:1 within-class domain imbalance; both domains eligible (>= m) in both classes.
    spec = {(0, 0): 500, (1, 0): 10, (0, 1): 10, (1, 1): 500}
    if ineligible:
        spec[(2, 0)] = 4                                   # (2,class0): below m -> ineligible
    y, d, g, gid = [], [], [], 0
    for (dom, cls), n in spec.items():
        for s in range(0, n, 30):                         # ~30 rows per recording
            cnt = min(30, n - s)
            y += [cls] * cnt; d += [dom] * cnt; g += [gid] * cnt; gid += 1
    y, d, g = np.array(y), np.array(d), np.array(g)
    n_dom = 3 if ineligible else 2
    counts = counts_from_labels(d, y, n_domains=n_dom, n_classes=2)
    sg = build_support_graph(counts, m=m, reference_prior=empirical_class_prior(counts))
    return y, d, g, sg


def _cfg(**kw):
    base = dict(min_per_eligible_cell=8, adv_microbatch_size=64, adv_accumulation_steps=1,
                task_batch_size=200, steps_per_epoch=3, seed=0)
    base.update(kw)
    return SamplerConfig(**base)


class _ConstLogits(torch.nn.Module):
    """Returns fixed logits regardless of input -> CE constant within each class (exact tests)."""
    def __init__(self, b):
        super().__init__(); self.b = torch.tensor(b)
    def forward(self, x):
        return self.b.expand(x.shape[0], self.b.shape[0])


def test_every_eligible_cell_is_covered_per_logical_step():
    y, d, g, sg = _imbalanced()
    s = RareCellSampler(y, d, g, sg, _cfg())
    lb = s.adv_logical_batch()
    covered = {(int(d[i]), int(y[i])) for i in lb.idx}
    assert covered == set(s.cells)
    assert s.eligible_cell_coverage(lb) == 1.0


def test_batch_never_redefines_support_or_p_ref():
    y, d, g, sg = _imbalanced()
    S_before = {k: list(v) for k, v in sg.support_of_class.items()}
    pref = sg.reference_prior.copy()
    s = RareCellSampler(y, d, g, sg, _cfg())
    for _ in range(5):
        s.adv_logical_batch(); s.task_batch()
    assert {k: list(v) for k, v in sg.support_of_class.items()} == S_before
    assert np.array_equal(sg.reference_prior, pref)
    assert s.U_cell[(0, 0)] == int(sg.cell_mass[0, 0])    # fixed unit count from the graph, not the batch


def test_ineligible_cells_never_enter_adversary_stream():
    y, d, g, sg = _imbalanced(ineligible=True)
    s = RareCellSampler(y, d, g, sg, _cfg())
    for _ in range(12):
        lb = s.adv_logical_batch()
        assert np.sum(d[lb.idx] == 2) == 0                # the ineligible (2,class0) cell


def test_ineligible_cells_still_enter_task_stream():
    y, d, g, sg = _imbalanced(ineligible=True)
    s = RareCellSampler(y, d, g, sg, _cfg(task_batch_size=400))
    seen = set()
    for _ in range(40):
        seen |= set(d[s.task_batch().idx].tolist())
    assert 2 in seen                                       # ineligible-domain rows DO appear in task


def test_unweighted_sampler_changes_effective_pD_givenY():
    y, d, g, sg = _imbalanced()
    s = RareCellSampler(y, d, g, sg, _cfg())
    lb = s.adv_logical_batch()
    raw = effective_prior_domain_given_y(lb.idx, np.ones(len(lb.idx)), y, d, sg)
    fixed = fixed_prior_domain_given_y(sg)
    for yy in sg.comparable_classes:
        assert np.allclose(raw[yy][1], [0.5, 0.5])         # equal k per cell -> uniform
        assert not np.allclose(raw[yy][1], fixed[yy][1], atol=0.05)


def test_adv_weights_restore_fixed_empirical_pD_givenY():
    y, d, g, sg = _imbalanced()
    s = RareCellSampler(y, d, g, sg, _cfg())
    lb = s.adv_logical_batch()
    wtd = effective_prior_domain_given_y(lb.idx, lb.weight, y, d, sg)
    fixed = fixed_prior_domain_given_y(sg)
    for yy in sg.comparable_classes:
        assert np.allclose(wtd[yy][1], fixed[yy][1], atol=1e-12)   # EXACT


def test_task_weights_restore_ce_target():
    y, d, g, sg = _imbalanced()
    s = RareCellSampler(y, d, g, sg, _cfg(task_batch_size=200))
    model = _ConstLogits([0.3, -0.7])
    X = torch.zeros(len(y), 1)
    yt = torch.tensor(y, dtype=torch.long)
    full = source_risk(model(X), yt, "ce", 2).item()
    tb = s.task_batch()
    mb = source_risk(model(X[tb.idx]), yt[tb.idx], "ce", 2, weight=torch.tensor(tb.weight)).item()
    assert abs(full - mb) < 1e-5


def test_task_weights_restore_balanced_ce_target():
    y, d, g, sg = _imbalanced()
    s = RareCellSampler(y, d, g, sg, _cfg(task_batch_size=200))
    model = _ConstLogits([0.1, 0.6])
    X = torch.zeros(len(y), 1)
    yt = torch.tensor(y, dtype=torch.long)
    full = source_risk(model(X), yt, "balanced_ce", 2).item()
    tb = s.task_batch()
    mb = source_risk(model(X[tb.idx]), yt[tb.idx], "balanced_ce", 2, weight=torch.tensor(tb.weight)).item()
    assert abs(full - mb) < 1e-5


def test_microbatch_partition_invariance():
    y, d, g, sg = _imbalanced()
    torch.manual_seed(0)
    adv = ConditionalDomainAdversary(4, sg, hidden=0)
    Z = torch.randn(len(y), 4)
    s = RareCellSampler(y, d, g, sg, _cfg(min_per_eligible_cell=8, adv_microbatch_size=8, adv_accumulation_steps=4))
    lb = s.adv_logical_batch()
    idx, w = lb.idx, lb.weight
    one = adv.domain_ce_contribution(Z[idx], y[idx], d[idx], w).item()                 # single chunk
    split = sum(adv.domain_ce_contribution(Z[mb.idx], y[mb.idx], d[mb.idx], mb.weight)
                for mb in lb.microbatches).item()                                       # sampler's split
    chunks = np.array_split(np.arange(len(idx)), 7)
    split2 = sum(adv.domain_ce_contribution(Z[idx[c]], y[idx[c]], d[idx[c]], w[c]) for c in chunks).item()
    assert abs(one - split) < 1e-6 and abs(one - split2) < 1e-6


def test_too_small_logical_batch_fails_loudly():
    y, d, g, sg = _imbalanced()                            # K_ov = 4 -> B_min = 4*16 = 64
    try:
        RareCellSampler(y, d, g, sg, _cfg(min_per_eligible_cell=16, adv_microbatch_size=8, adv_accumulation_steps=4))
    except ValueError as e:
        assert "B_min" in str(e)
    else:
        raise AssertionError("expected a loud failure when capacity < B_min")


def test_no_comparable_classes_is_byte_exact_erm_noop():
    X, y, d, g, sg = make_covariate_shift(seed=0, n_domains=1)
    assert sg.comparable_classes == []
    res = train_risk_feasible(X, y, d, g, sg, TrainConfig(seed=0, stage1_epochs=20, stage2_epochs=10))
    assert res.active is False and res.trajectory == []   # NO Stage-2 updates ran
    sel = select_checkpoint(res)
    assert sel.used_erm_fallback and sel.selected_epoch == -1
    assert sel.model_hash == res.erm_record.model_hash
    assert state_hash(sel.model_state) == state_hash(res.erm_record.model_state)


def test_lambda_floor_zero_matches_exact_lagrangian():
    for lam in (0.0, 0.05, 1.5, 7.0):
        assert effective_risk_weight(lam, 0.0) == lam      # floor 0 -> coefficient == λ exactly
    assert effective_risk_weight(0.05, 0.1) == 0.1         # nonzero floor: a relaxation, not λ


def test_missing_cell_mask_is_respected():
    y, d, g, sg = _imbalanced(ineligible=False)            # cells (0,0),(1,0),(0,1),(1,1)
    counts = counts_from_labels(d, y, n_domains=2, n_classes=2)
    masked = counts.copy(); masked[1, 1] = 0              # DELETE cell (1,1)
    sg_m = build_support_graph(masked, m=10, reference_prior=empirical_class_prior(counts))
    assert 1 not in sg_m.support_of_class[1]               # class1 now lacks domain1
    s = RareCellSampler(y, d, g, sg_m, _cfg())
    for _ in range(10):
        lb = s.adv_logical_batch()
        assert np.sum((d[lb.idx] == 1) & (y[lb.idx] == 1)) == 0   # masked cell never sampled


def test_seed_reproducibility():
    y, d, g, sg = _imbalanced()
    a = RareCellSampler(y, d, g, sg, _cfg(seed=5))
    b = RareCellSampler(y, d, g, sg, _cfg(seed=5))
    c = RareCellSampler(y, d, g, sg, _cfg(seed=6))
    la, lb_, lc = a.adv_logical_batch(), b.adv_logical_batch(), c.adv_logical_batch()
    assert np.array_equal(la.idx, lb_.idx) and np.allclose(la.weight, lb_.weight)
    assert not np.array_equal(la.idx, lc.idx)
    assert np.array_equal(a.task_batch().idx, b.task_batch().idx)


def test_minibatch_trainer_integration_runs():
    # exercise the sampler-driven Stage-2 path end-to-end (paired streams + microbatch accumulation)
    from oaci.data.sampler_demo import make_imbalanced
    X, y, d, g, sg = make_imbalanced(seed=0)
    sampler = RareCellSampler(y, d, g, sg, _cfg(min_per_eligible_cell=8, adv_microbatch_size=32,
                                                adv_accumulation_steps=1, task_batch_size=64, steps_per_epoch=2))
    cfg = TrainConfig(seed=0, stage1_epochs=20, stage2_epochs=3, warmup_steps=4, critic_steps=2,
                      adv_hidden=8, z_dim=8, enc_hidden=16)
    res = train_risk_feasible(X, y, d, g, sg, cfg, sampler=sampler)
    assert len(res.trajectory) == 3
    assert select_checkpoint(res).model_state is not None


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} rare-cell-sampler tests")


if __name__ == "__main__":
    _run_all()
