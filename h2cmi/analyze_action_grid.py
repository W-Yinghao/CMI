"""Stage-B0 action-decomposition analysis (review §3/§7-B0).

Attributes each scenario's adaptation effect to the TRANSFORM vs the PRIOR, using the
fit/decision-prior separation, and reports the multi-head (disc/gen/blend) strict picture.

Per scenario (seed-clustered; sites and the CMI arm averaged within a seed):
  transform_effect  = geometry_only − identity   on balanced acc (UNIFORM decision prior)
  prior_effect_bacc = prior_only    − identity   on bAcc (≈ 0 by construction -- a check)
  prior_effect_acc  = prior_only    − identity   on ordinary accuracy (TARGET prior)
  prior_effect_nll  = prior_only    − identity   on NLL (TARGET prior; negative = better)
  joint_effect_bacc = joint         − identity   on bAcc
  joint_vs_best     = joint − max(geometry_only, prior_only)   on bAcc

A scenario is labelled geometry-driven / prior-driven (calibration, not bAcc) / both /
neither, which is the mechanism→action map the gate should later route on.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from h2cmi.analyze_shift_grid import canon, _seed_cluster

EPS = 0.02


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line); r["scenario"] = canon(r["scenario"]); rows.append(r)
    return rows


def _by(rows, **eq):
    return [r for r in rows if all(r.get(k) == v for k, v in eq.items())]


def _unit_vals(rows, scenario, action, metric):
    """(seed,site) -> mean over the CMI arm of `metric` for a given action."""
    by = defaultdict(list)
    for r in _by(rows, scenario=scenario, action=action):
        by[(r["data_seed"], r["target_site"])].append(r[metric])
    return {u: float(np.mean(v)) for u, v in by.items()}


def _contrast(rows, scenario, a, b, metric):
    va, vb = _unit_vals(rows, scenario, a, metric), _unit_vals(rows, scenario, b, metric)
    return {u: va[u] - vb[u] for u in set(va) & set(vb)}


def _strict(rows, scenario, cmi=None):
    sel = _by(rows, scenario=scenario, action="identity") if cmi is None \
        else _by(rows, scenario=scenario, action="identity", cmi=cmi)
    if not sel:
        return {}
    keys = ("strict_disc_bacc", "strict_gen_bacc", "strict_blend_bacc",
            "disc_ece", "gen_ece", "blend_ece", "disc_gen_disagreement")
    return {k: float(np.mean([r[k] for r in sel])) for k in keys}


def decide_action(c) -> tuple[str, str]:
    tg = c["transform_effect"]["seed_mean"]
    pa = c["prior_effect_acc"]["seed_mean"]
    pn = c["prior_effect_nll"]["seed_mean"]
    if tg > EPS and pa > EPS:
        return ("both", f"geometry (+{tg:.3f} bAcc) AND prior (+{pa:.3f} acc) both help")
    if tg > EPS:
        return ("geometry_driven", f"transform recovers bAcc (+{tg:.3f}); prior secondary")
    if pa > EPS or pn < -0.05:
        return ("prior_driven", f"prior helps accuracy/NLL ({pa:+.3f} acc, {pn:+.3f} NLL) "
                                "but NOT balanced acc -> prior-only, identity geometry")
    return ("neither", f"no action recovers bAcc (geom {tg:+.3f}, prior-acc {pa:+.3f})")


def analyze(path, n_boot=2000, seed=0) -> dict:
    rows = load(path)
    scenarios = sorted({r["scenario"] for r in rows})
    rep = dict(source=path, n_rows=len(rows), cluster="seed", scenarios=scenarios,
               actions={}, strict={}, decisions={})
    for s in scenarios:
        c = dict(
            transform_effect=_seed_cluster(_contrast(rows, s, "geometry_only", "identity",
                                                     "bacc_uniform_decision"), n_boot, seed),
            prior_effect_bacc=_seed_cluster(_contrast(rows, s, "prior_only", "identity",
                                                      "bacc_uniform_decision"), n_boot, seed),
            prior_effect_acc=_seed_cluster(_contrast(rows, s, "prior_only", "identity",
                                                     "accuracy_target_prior"), n_boot, seed),
            prior_effect_nll=_seed_cluster(_contrast(rows, s, "prior_only", "identity",
                                                     "nll_target_prior"), n_boot, seed),
            joint_effect_bacc=_seed_cluster(_contrast(rows, s, "joint", "identity",
                                                      "bacc_uniform_decision"), n_boot, seed),
        )
        # joint vs the best single action (per unit)
        g = _unit_vals(rows, s, "geometry_only", "bacc_uniform_decision")
        p = _unit_vals(rows, s, "prior_only", "bacc_uniform_decision")
        j = _unit_vals(rows, s, "joint", "bacc_uniform_decision")
        units = set(g) & set(p) & set(j)
        c["joint_vs_best"] = _seed_cluster({u: j[u] - max(g[u], p[u]) for u in units}, n_boot, seed)
        rep["actions"][s] = c
        rep["strict"][s] = dict(pooled=_strict(rows, s),
                                off=_strict(rows, s, "off"), on=_strict(rows, s, "on"))
        rep["decisions"][s] = dict(zip(("code", "message"), decide_action(c)))
    return rep


def _print(rep):
    print(f"\n=== action-decomposition ({rep['n_rows']} rows, cluster=seed) {rep['source']} ===")
    print(f"\n{'scenario':<24} {'geom→bAcc':>10} {'prior→bAcc':>11} {'prior→acc':>10} "
          f"{'prior→NLL':>10} {'joint→bAcc':>11} {'joint−best':>11}  decision")
    for s, c in rep["actions"].items():
        d = rep["decisions"][s]
        print(f"{s:<24} {c['transform_effect']['seed_mean']:>+10.3f} "
              f"{c['prior_effect_bacc']['seed_mean']:>+11.3f} {c['prior_effect_acc']['seed_mean']:>+10.3f} "
              f"{c['prior_effect_nll']['seed_mean']:>+10.3f} {c['joint_effect_bacc']['seed_mean']:>+11.3f} "
              f"{c['joint_vs_best']['seed_mean']:>+11.3f}  [{d['code']}]")
    print("\n-- multi-head strict bAcc (disc / gen / blend) + disc-gen disagreement, by CMI --")
    print(f"{'scenario':<24} {'off disc/gen/blend':>22} {'on disc/gen/blend':>22} {'disagree off/on':>16}")
    for s, st in rep["strict"].items():
        o, n = st["off"], st["on"]
        if o and n:
            print(f"{s:<24} {o['strict_disc_bacc']:.2f}/{o['strict_gen_bacc']:.2f}/{o['strict_blend_bacc']:.2f}".ljust(47)
                  + f"{n['strict_disc_bacc']:.2f}/{n['strict_gen_bacc']:.2f}/{n['strict_blend_bacc']:.2f}".ljust(22)
                  + f"{o['disc_gen_disagreement']:.2f}/{n['disc_gen_disagreement']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rep = analyze(args.inp, args.n_boot, args.seed)
    _print(rep)
    if args.out:
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2, default=float)
        print("\nsaved ->", args.out)


if __name__ == "__main__":
    main()
