"""Eval acceptance demo: 3 unequal-size domains where POOLED bAcc looks fine but a small domain
is clearly harmed — so pooled vs mean/worst-domain vs worst-paired-delta diverge. Reports the
full panel. Run: ``python -m oaci.eval.synthetic``.
"""
from __future__ import annotations

import numpy as np

from .artifacts import PredictionBundle, align_pair
from .bootstrap import make_bootstrap_plan, paired_ci
from .calibration import (
    mean_domain_nll,
    nll_per_sample,
    pooled_nll,
    top_label_ece,
    worst_domain_nll,
)
from .metrics import (
    domain_baccs,
    mean_domain_bacc,
    pooled_bacc,
    worst_domain_bacc,
    worst_paired_delta_bacc,
)
from .noninferiority import noninferiority, source_risk_noninferiority, superiority
from .sweep import post_fragmentation_curve_average

CLASSES = ["A", "B"]
MARGIN = 2.0


def _population(seed=0, sizes=(6, 3, 2), per_group=(50, 40, 20)):
    """3 domains of unequal size; each group carries both classes (balanced)."""
    rng = np.random.default_rng(seed)
    sid, y, dom, grp, gid = [], [], [], [], 0
    for d, (ng, pg) in enumerate(zip(sizes, per_group)):
        for _ in range(ng):
            half = pg // 2
            labels = [0] * half + [1] * (pg - half)
            for lab in labels:
                y.append(lab); dom.append(d); grp.append(gid)
            gid += 1
    y = np.array(y); dom = np.array(dom); grp = np.array(grp)
    sid = np.arange(len(y))
    return sid, y, dom, grp


_METHOD_ID = {"ERM": 0, "OACI": 1, "global-LPC": 2, "uniform": 3}


def _bundle(method, acc_by_domain, pop, level=0, role="target_audit", seed=0):
    sid, y, dom, grp = pop
    # deterministic seed (no hash(): PYTHONHASHSEED would make the demo non-reproducible)
    rng = np.random.default_rng([seed, _METHOD_ID.get(method, 9), level])
    pred = y.copy()
    for d, acc in acc_by_domain.items():
        m = dom == d
        flip = rng.random(m.sum()) > acc
        idx = np.where(m)[0]
        pred[idx[flip]] = 1 - y[idx[flip]]
    logits = np.zeros((len(y), 2))
    logits[np.arange(len(y)), pred] = MARGIN
    return PredictionBundle(sample_id=sid, logits=logits, y=y, domain=dom, group=grp,
                            method=method, seed=seed, split_id="demo", split_role=role,
                            deletion_level=level, class_names=CLASSES, risk_metric="balanced_ce")


def _balanced_ce(logits, y, classes=(0, 1)):
    per = nll_per_sample(logits, y)
    return float(np.mean([per[np.asarray(y) == c].mean() for c in classes]))


def _demo() -> None:
    pop = _population()
    # ERM solid everywhere; OACI slightly up on big domains but HARMS the small domain 2.
    erm = _bundle("ERM", {0: 0.85, 1: 0.84, 2: 0.86}, pop, role="target_audit")
    oaci = _bundle("OACI", {0: 0.88, 1: 0.86, 2: 0.70}, pop, role="target_audit")
    a, b = align_pair(oaci, erm)
    classes = [0, 1]
    n_groups = len(np.unique(a.group))

    print("Eval acceptance report — 3 unequal domains, hidden small-domain harm")
    print(f"  audit_population_hash      = {a.eval_population_hash}")
    print(f"  n_domains/n_groups/n_eval  = 3 / {n_groups} / 3")
    for name, bnd in [("ERM", b), ("OACI", a)]:
        print(f"  [{name}] pooled={pooled_bacc(bnd.y, bnd.pred, classes):.3f}"
              f"  mean-domain={mean_domain_bacc(bnd.y, bnd.pred, bnd.domain, classes):.3f}"
              f"  worst-domain={worst_domain_bacc(bnd.y, bnd.pred, bnd.domain, classes):.3f}")
    wpd = worst_paired_delta_bacc(a.y, a.pred, b.pred, a.domain, classes)
    print(f"  worst_paired_delta_bAcc    = {wpd:+.3f}  (<0 -> a domain harmed despite pooled gain)")
    for name, bnd in [("ERM", b), ("OACI", a)]:
        print(f"  [{name}] NLL pooled={pooled_nll(bnd.logits, bnd.y):.3f}"
              f"  mean-domain={mean_domain_nll(bnd.logits, bnd.y, bnd.domain):.3f}"
              f"  worst-domain={worst_domain_nll(bnd.logits, bnd.y, bnd.domain):.3f}"
              f"  ECE={top_label_ece(bnd.logits, bnd.y):.3f}")

    # paired clustered bootstrap on worst_paired_delta_bAcc (worst domain recomputed per replicate)
    plan = make_bootstrap_plan(a.domain, a.group, a.y, reference_classes=classes, n_boot=400, seed=0)
    delta_fn = lambda idx: worst_paired_delta_bacc(a.y[idx], a.pred[idx], b.pred[idx], a.domain[idx], classes)
    ci = paired_ci(plan, wpd, delta_fn, alpha=0.05)
    print(f"  worst_paired_delta CI      basic[{ci['basic_lcl']:+.3f},{ci['basic_ucl']:+.3f}]"
          f"  pct[{ci['percentile_lcl']:+.3f},{ci['percentile_ucl']:+.3f}]  invalid_rate={ci['invalid_draw_rate']:.3f}")
    print(f"  target-bAcc NI (δ=0.02)    = {noninferiority(ci, 0.02, higher_is_better=True)}"
          f"   superiority = {superiority(ci, higher_is_better=True)}")
    # worst-domain selection frequencies across replicates
    worst_dom = []
    for idx in plan.replicates:
        bvals = domain_baccs(a.y[idx], a.pred[idx], a.domain[idx], classes)
        worst_dom.append(min(bvals, key=bvals.get))
    vals, cnts = np.unique(worst_dom, return_counts=True)
    freq = {int(v): round(c / len(worst_dom), 3) for v, c in zip(vals, cnts)}
    print(f"  worst-domain selection freq= {freq}")

    # source-risk noninferiority (audit metric == training metric == balanced_ce)
    src_erm = _bundle("ERM", {0: 0.90, 1: 0.90, 2: 0.90}, pop, role="source_audit", seed=1)
    src_oaci = _bundle("OACI", {0: 0.89, 1: 0.89, 2: 0.89}, pop, role="source_audit", seed=1)
    sa, sb = align_pair(src_oaci, src_erm)
    src_plan = make_bootstrap_plan(sa.domain, sa.group, sa.y, reference_classes=classes, n_boot=400, seed=0)
    src_dhat = _balanced_ce(sa.logits, sa.y) - _balanced_ce(sb.logits, sb.y)
    src_dfn = lambda idx: _balanced_ce(sa.logits[idx], sa.y[idx]) - _balanced_ce(sb.logits[idx], sb.y[idx])
    src_ci = paired_ci(src_plan, src_dhat, src_dfn, alpha=0.05)
    epsilon = 0.05
    print(f"  realized_constraint_gap    = {src_dhat:+.4f}  (trainer-guard-style point gap)")
    print(f"  source-risk audit ΔbalCE   = {src_dhat:+.4f}  UCL={src_ci['basic_ucl']:+.4f}")
    print(f"  source-risk NI (ε={epsilon})    = "
          f"{source_risk_noninferiority(src_ci, epsilon, 'balanced_ce', 'balanced_ce')}")

    # post-fragmentation curve average ΔA_post over a tiny sweep (fixed audit population)
    levels = [0, 1, 2]; first_frag = 1
    oaci_worst = [worst_domain_bacc(*_wd(_bundle("OACI", {0: 0.88, 1: 0.86, 2: dd}, pop, level=l), classes))
                  for l, dd in zip(levels, [0.84, 0.75, 0.70])]
    erm_worst = [worst_domain_bacc(*_wd(b, classes)) for _ in levels]
    dA = post_fragmentation_curve_average(levels, [o - e for o, e in zip(oaci_worst, erm_worst)], first_frag)
    print(f"  post-fragmentation ΔA_post = {dA:+.3f}  (ℓ_f={first_frag}; OACI worst {np.round(oaci_worst,3)})")


def _wd(bundle, classes):
    return (bundle.y, bundle.pred, bundle.domain, classes)


if __name__ == "__main__":
    _demo()
