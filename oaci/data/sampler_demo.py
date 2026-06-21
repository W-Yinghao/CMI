"""Acceptance demo for the paired-stream rare-cell sampler under heavy cell imbalance (50:1).

Reports coverage, raw vs importance-weighted ``p(d|y)``, weighted-mass error, ESS, replacement
rate, unique-recording fraction, and full-vs-minibatch agreement of task risk and ``C_D``.
Run: ``python -m oaci.data.sampler_demo``.
"""
from __future__ import annotations

import numpy as np
import torch

from ..config import SamplerConfig
from ..support_graph import build_support_graph, counts_from_labels, empirical_class_prior
from ..train.adversary import ConditionalDomainAdversary
from ..train.primal_dual import Encoder, TaskHead
from ..train.risk import source_risk
from .batch import (
    effective_prior_domain_given_y,
    fixed_prior_domain_given_y,
    weighted_ess,
)
from .sampler import RareCellSampler


def make_imbalanced(seed=0, m=10):
    """2 domains x 2 classes with 50:1 within-class domain imbalance, plus one INELIGIBLE
    (d=2, class 0) cell of 5 rows (< m). Recordings (groups) span ~50 rows each."""
    rng = np.random.default_rng(seed)
    spec = {(0, 0): 500, (0, 1): 10, (1, 0): 10, (1, 1): 500, (2, 0): 5}  # last is ineligible
    y, d, g = [], [], []
    gid = 0
    for (dom, cls), n in spec.items():
        for s in range(0, n, 50):                       # ~50 rows per recording
            cnt = min(50, n - s)
            y += [cls] * cnt; d += [dom] * cnt; g += [gid] * cnt; gid += 1
    y, d, g = np.array(y), np.array(d), np.array(g)
    ym = rng.standard_normal((2, 3)) * 2.5
    dm = rng.standard_normal((3, 3)) * 1.2
    X = np.concatenate([ym[y] + 0.4 * rng.standard_normal((y.size, 3)),
                        dm[d] + 0.6 * rng.standard_normal((y.size, 3))], axis=1).astype(np.float32)
    counts = counts_from_labels(d, y, n_domains=3, n_classes=2)
    sg = build_support_graph(counts, m=m, reference_prior=empirical_class_prior(counts))
    return X, y, d, g, sg


def _demo() -> None:
    X, y, d, g, sg = make_imbalanced()
    cfg = SamplerConfig(min_per_eligible_cell=16, adv_microbatch_size=32, adv_accumulation_steps=4,
                        task_batch_size=128, steps_per_epoch=5, seed=0)
    sampler = RareCellSampler(y, d, g, sg, cfg)

    lb = sampler.adv_logical_batch()
    raw = effective_prior_domain_given_y(lb.idx, np.ones(len(lb.idx)), y, d, sg)
    wtd = effective_prior_domain_given_y(lb.idx, lb.weight, y, d, sg)
    fixed = fixed_prior_domain_given_y(sg)
    mass_err = max(float(np.max(np.abs(wtd[yy][1] - fixed[yy][1]))) for yy in sg.comparable_classes)
    ess = [weighted_ess(lb.weight[(y[lb.idx] == yy) & (d[lb.idx] == dd)])
           for (yy) in sg.comparable_classes for dd in sg.support_of_class[yy]]
    # ineligible (d=2,class0) must NOT appear in the adversary stream
    inelig_adv = int(np.sum(d[lb.idx] == 2))

    # fixed model: full vs minibatch task risk and C_D
    Xt, yt = torch.tensor(X), torch.tensor(y, dtype=torch.long)
    torch.manual_seed(0)
    enc, head = Encoder(X.shape[1], 8, 16), TaskHead(8, 2)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=5e-3)
    for _ in range(80):
        opt.zero_grad(); source_risk(head(enc(Xt)), yt, "balanced_ce").backward(); opt.step()
    adv = ConditionalDomainAdversary(8, sg, hidden=16)
    optd = torch.optim.Adam(adv.parameters(), lr=1e-2)
    with torch.no_grad():
        Zf = enc(Xt)
    for _ in range(150):
        optd.zero_grad(); adv.domain_ce(Zf, y, d).backward(); optd.step()

    with torch.no_grad():
        full_task = float(source_risk(head(enc(Xt)), yt, "balanced_ce").item())
        tb = sampler.task_batch()
        mb_task = float(source_risk(head(enc(Xt[tb.idx])), yt[tb.idx], "balanced_ce",
                                    weight=torch.tensor(tb.weight)).item())
        full_cd = float(adv.domain_ce(enc(Xt), y, d).item())
        lb2 = sampler.adv_logical_batch()
        mb_cd = float(sum(adv.domain_ce_contribution(enc(Xt[mb.idx]), y[mb.idx], d[mb.idx], mb.weight)
                          for mb in lb2.microbatches).item())

    print("Rare-cell sampler — acceptance report (50:1 cell imbalance)")
    print(f"  n_eligible_cells            = {sampler.K_ov}")
    print(f"  logical_adv_batch_size      = {sampler.logical_adv_batch_size}")
    print(f"  eligible_cell_coverage      = {sampler.eligible_cell_coverage(lb):.3f}")
    for yy in sg.comparable_classes:
        print(f"  class {yy}: raw  p(d|y) = {np.round(raw[yy][1],3)}  -> weighted = {np.round(wtd[yy][1],3)}"
              f"   (fixed = {np.round(fixed[yy][1],3)})")
    print(f"  max_weighted_mass_error     = {mass_err:.2e}")
    print(f"  importance-weight ESS       = min {min(ess):.2f}  median {float(np.median(ess)):.2f}")
    print(f"  ineligible rows in adv stream = {inelig_adv}  (must be 0)")
    print(f"  replacement_rate            = {sampler.replacement_rate:.3f}")
    print(f"  unique_recording_fraction   = {sampler.unique_recording_fraction(lb):.3f}")
    print(f"  full_vs_minibatch_task_risk = {full_task:.4f} vs {mb_task:.4f}  (|Δ|={abs(full_task-mb_task):.3f})")
    print(f"  full_vs_minibatch_domain_ce = {full_cd:.4f} vs {mb_cd:.4f}  (|Δ|={abs(full_cd-mb_cd):.3f})")


if __name__ == "__main__":
    _demo()
