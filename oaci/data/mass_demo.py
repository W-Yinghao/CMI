"""Sample-mass invariance demo: units with 1 / 2 / 10 / 100 windows; duplicate the 100-window
unit wholesale and re-split its mass, then show every estimand is unchanged. NO training.
Run: ``python -m oaci.data.mass_demo``.
"""
from __future__ import annotations

import numpy as np
import torch

from ..config import SamplerConfig
from ..data.batch import effective_prior_domain_given_y, effective_prior_y
from ..data.eeg.units import cell_mass as _cell_mass
from ..data.eeg.units import eligibility_counts
from ..data.sampler import RareCellSampler
from ..leakage.critic import CriticConfig
from ..leakage.crossfit import FrozenFeatures, make_fold_plan
from ..leakage.estimate import estimate_extractable_leakage
from ..leakage.ucb import bootstrap_ucb
from ..support_graph import build_support_graph
from ..train.adversary import ConditionalDomainAdversary
from ..train.risk import balanced_ce, source_risk

SIZES = [1, 2, 10, 100] + [1] * 8       # showcase 1/2/10/100 + small units (bootstrap robustness)


def _build(seed=0, dim=4, dup100=False):
    rng = np.random.default_rng(seed)
    Z, y, d, g, b, unit = [], [], [], [], [], []
    uid = 0
    for dom in (0, 1):
        for cls in (0, 1):
            for sz in SIZES:
                n = sz * 2 if (dup100 and sz == 100) else sz   # duplicate the 100-window unit wholesale
                base = rng.standard_normal((sz, dim))          # the unit's content (fixed)
                rows = np.vstack([base, base]) if (dup100 and sz == 100) else base
                for j in range(n):
                    Z.append(rows[j % sz]); y.append(cls); d.append(dom); g.append(uid)
                    b.append(1.0 / n); unit.append(f"u{uid}")
                uid += 1
    return (np.array(Z), np.array(y), np.array(d), np.array(g), np.array(b, float),
            np.array(unit, dtype=object))


def _measures(Z, y, d, g, b, unit, sg):
    feat = FrozenFeatures(Z, y, d, g, b)
    plan = make_fold_plan(feat, sg, n_folds=2, seed=0)
    cfg = CriticConfig(capacities=(0, 8))
    torch.manual_seed(0)                                          # SAME adversary weights before/after
    adv = ConditionalDomainAdversary(Z.shape[1], sg, hidden=0).double()
    Zt, yt, bt = torch.tensor(Z), torch.tensor(y), torch.tensor(b)
    # logits are a DETERMINISTIC function of the row content, so duplicated rows get equal logits
    W = np.random.default_rng(42).standard_normal((Z.shape[1], 2))
    L = torch.tensor(Z @ W)
    elig = np.where(np.isin(d, [0, 1]))[0]
    streamed = sum(adv.domain_ce_contribution(torch.tensor(Z[c]), y[c], d[c], torch.tensor(b[c]))
                   for c in np.array_split(elig, 5)).item()
    est = estimate_extractable_leakage(feat, sg, plan, cfg)
    boot = bootstrap_ucb(feat, sg, plan, cfg, alpha=0.1, n_bootstrap=100, seed=0)
    s = RareCellSampler(y, d, g, sg, SamplerConfig(min_per_eligible_cell=4, task_batch_size=8),
                        sample_mass=b, mass_unit_id=unit)
    tb = s.task_batch(); lb = s.adv_logical_batch()
    return {
        "task_ce": source_risk(L, yt, "ce", 2, weight=bt).item(),
        "balanced_ce": balanced_ce(L, yt, 2, weight=bt).item(),
        "domain_ce_full": adv.domain_ce(Zt, y, d, importance_weight=bt).item(),
        "domain_ce_streamed": streamed,
        "fold_hash": hash(tuple(sorted(plan.fold_of_group.items()))),
        "extractable_LQ_ov": est["extractable_LQ_ov"],
        "bootstrap_ucl": boot["bootstrap_ucl"],
        "eff_py": effective_prior_y(tb.idx, tb.weight, y, 2).round(4).tolist(),
        "eff_pd_y0": effective_prior_domain_given_y(lb.idx, lb.weight, y, d, sg)[0][1].round(4).tolist(),
    }


def _demo() -> None:
    Z, y, d, g, b, unit = _build(dup100=False)
    nelig = eligibility_counts(d, y, unit, 2, 2); M = _cell_mass(d, y, b, 2, 2)
    sg = build_support_graph(nelig, m=2, cell_mass=M)
    cell_err = float(np.max(np.abs(M - np.array([[_cell_mass(d, y, b, 2, 2)[i, j] for j in range(2)] for i in range(2)]))))
    m_before = _measures(Z, y, d, g, b, unit, sg)
    Z2, y2, d2, g2, b2, u2 = _build(dup100=True)
    # support graph is identical (same units & mass) -> reuse sg
    m_after = _measures(Z2, y2, d2, g2, b2, u2, sg)

    print("Sample-mass invariance demo (units: 1/2/10/100 windows; 100-window unit duplicated)")
    um = np.array([b[unit == u].sum() for u in np.unique(unit)])
    print(f"  unit mass min/max           = {um.min():.6f} / {um.max():.6f}")
    print(f"  cell-mass consistency error = {cell_err:.2e}")
    print(f"  n_rows before/after         = {len(y)} / {len(y2)}  (eligibility units unchanged)")
    keys = ["task_ce", "balanced_ce", "domain_ce_full", "domain_ce_streamed",
            "extractable_LQ_ov", "bootstrap_ucl"]
    maxd = 0.0
    for k in keys:
        diff = abs(m_before[k] - m_after[k]); maxd = max(maxd, diff)
        print(f"  {k:22s} {m_before[k]:+.6f} -> {m_after[k]:+.6f}   |Δ|={diff:.2e}")
    print(f"  fold_plan hash before/after = {m_before['fold_hash']} / {m_after['fold_hash']}"
          f"  ({'SAME' if m_before['fold_hash']==m_after['fold_hash'] else 'DIFFERENT'})")
    print(f"  effective p(y)              = {m_before['eff_py']} -> {m_after['eff_py']}")
    print(f"  effective p(d|y=0)          = {m_before['eff_pd_y0']} -> {m_after['eff_pd_y0']}")
    print(f"  MAX absolute discrepancy    = {maxd:.2e}")


if __name__ == "__main__":
    _demo()
