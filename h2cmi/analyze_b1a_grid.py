"""Frozen Stage-B1a analysis (pre-registered BEFORE the GPU grid runs).

Five contrasts localise WHY the joint EM underperforms geometry-only. Each contrast is a
difference of two variants' per-unit metric; the identity baseline cancels, so:

  full-target flavour : metric = bacc_uniform        (in-sample transductive bAcc)
  grouped-OOF flavour : metric = grouped_oof_bacc     (per-target-subject LOSO held-out bAcc)

  C_feedback        = bacc[gen_oneshot]  - bacc[gen_iterative]        (full)  >=0.02 -> iterative
                      responsibility feedback is HARMFUL
  C_responsibility  = oof [oracle_diag]  - oof [gen_oneshot]          (OOF)   >=0.02 -> responsibility
                      ESTIMATION is the bottleneck (oracle labels would fix it)
  C_class_cond      = (bacc|oof)[gen_oneshot] - (bacc|oof)[pooled_empirical]  >=0.02 -> p(z|y) is
                      load-bearing vs a class-free pooled moment match (reported BOTH flavours)
  C_family          = oof [oracle_lowrank] - oof [oracle_diag]        (OOF)   >=0.02 -> the diagonal
                      family is the bottleneck for rotation (low-rank helps under the resp ceiling)
  C_prior_coupling  = bacc[gen_iterative] - bacc[joint]              (full)  >=0.02 -> the prior
                      M-step coupling is harmful (geometry-only beats joint)

Statistical unit is the SEED: average the 5 target sites within a seed, then cluster-bootstrap
over seeds. Development threshold |mean| >= 0.02. Hard-null (difficulty=hard, matched_domain_null)
is judged by a separate SAFETY panel, never by the contrasts.

  python -m h2cmi.analyze_b1a_grid --in results/h2cmi/b1a_standard.jsonl \
      --out results/h2cmi/b1a_standard.report.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

THRESH = 0.02
IDENTITY = "identity"

# (name, variant_a, variant_b, flavour) -- contrast = metric[a] - metric[b]
CONTRASTS = (
    ("C_feedback",        "gen_oneshot_diag",       "gen_iterative_diag",    "full"),
    ("C_responsibility",  "oracle_oneshot_diag",    "gen_oneshot_diag",      "oof"),
    ("C_class_cond_full", "gen_oneshot_diag",       "pooled_empirical_diag", "full"),
    ("C_class_cond_oof",  "gen_oneshot_diag",       "pooled_empirical_diag", "oof"),
    ("C_family",          "oracle_oneshot_lowrank", "oracle_oneshot_diag",   "oof"),
    ("C_prior_coupling",  "gen_iterative_diag",     "joint_iterative_diag",  "full"),
)
_METRIC = {"full": "bacc_uniform", "oof": "grouped_oof_bacc"}


def _load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def _cluster_bootstrap(per_seed_site, *, n_boot=10000, seed=0):
    """per_seed_site: dict seed -> list of site-level values. Average sites within a seed, then
    resample seeds with replacement. Returns (mean, ci_lo, ci_hi, n_seeds)."""
    seeds = sorted(per_seed_site)
    per_seed = np.array([np.mean(per_seed_site[s]) for s in seeds], dtype=float)
    per_seed = per_seed[np.isfinite(per_seed)]
    n = len(per_seed)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(per_seed.mean())
    if n == 1:
        return mean, float("nan"), float("nan"), 1
    rng = np.random.default_rng(seed)
    boot = per_seed[rng.integers(0, n, size=(n_boot, n))].mean(1)
    return mean, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), n


def _per_seed_site(rows, metric):
    """metric value per (seed, site) for one variant within one scenario."""
    out = defaultdict(dict)                                    # seed -> {site: value}
    for r in rows:
        v = r.get(metric)
        if v is not None and v == v:                          # finite
            out[r["data_seed"]][r["target_site"]] = float(v)
    return {s: list(d.values()) for s, d in out.items()}


def _contrast(rows_by_variant, a, b, metric, *, n_boot, seed):
    """Per (seed,site): metric[a]-metric[b]; cluster-bootstrap over seeds."""
    ra, rb = rows_by_variant.get(a, []), rows_by_variant.get(b, [])
    ma = {(r["data_seed"], r["target_site"]): r.get(metric) for r in ra}
    mb = {(r["data_seed"], r["target_site"]): r.get(metric) for r in rb}
    per = defaultdict(list)
    for k in set(ma) & set(mb):
        va, vb = ma[k], mb[k]
        if va is not None and vb is not None and va == va and vb == vb:
            per[k[0]].append(float(va) - float(vb))
    return _cluster_bootstrap(per, n_boot=n_boot, seed=seed)


def hard_null_safety(rows_by_variant, *, occ_floor=0.05) -> dict:
    """Safety panel for difficulty=hard matched_domain_null (NOT the contrasts). All adaptation
    must be near-inert: |delta bAcc|<=0.01, prediction_disagreement<=0.02, no OOF NLL worsening,
    no responsibility/occupancy collapse, and identity must be in the OOF oracle-best set."""
    out = {}
    id_rows = rows_by_variant.get(IDENTITY, [])
    id_oof_nll = np.nanmean([r.get("grouped_oof_nll", np.nan) for r in id_rows]) if id_rows else np.nan
    identity_best = any(r.get("oracle_best_variant") == IDENTITY for r in id_rows)
    for v, rows in rows_by_variant.items():
        if v == IDENTITY:
            continue
        dbacc = np.nanmean([abs(r.get("delta_bacc_uniform", np.nan)) for r in rows])
        disagree = np.nanmean([r.get("prediction_disagreement", np.nan) for r in rows])
        oof_nll = np.nanmean([r.get("grouped_oof_nll", np.nan) for r in rows])
        occ = np.nanmin([r.get("final_class_occupancy", np.nan) for r in rows])
        out[v] = dict(
            abs_delta_bacc=float(dbacc), pred_disagreement=float(disagree),
            oof_nll=float(oof_nll), min_class_occupancy=float(occ),
            ok_delta_bacc=bool(dbacc <= 0.01), ok_disagreement=bool(disagree <= 0.02),
            ok_oof_nll=bool(not (oof_nll > id_oof_nll + 1e-9)),
            ok_no_collapse=bool(occ >= occ_floor))
    return dict(identity_in_oracle_best=bool(identity_best),
                identity_oof_nll=float(id_oof_nll), per_variant=out,
                all_safe=bool(identity_best and all(
                    all(x.get(k) for k in ("ok_delta_bacc", "ok_disagreement", "ok_oof_nll", "ok_no_collapse"))
                    for x in out.values())))


def analyze(rows, *, n_boot=10000, boot_seed=0) -> dict:
    by_diff_scen = defaultdict(lambda: defaultdict(list))     # (diff,scen) -> variant -> rows
    for r in rows:
        by_diff_scen[(r.get("difficulty", "standard"), r["scenario"])][r["variant"]].append(r)
    report = {"threshold": THRESH, "n_rows": len(rows), "contrasts": {}, "hard_null_safety": {}}
    for (diff, scen), rbv in sorted(by_diff_scen.items()):
        key = f"{diff}/{scen}"
        if diff == "hard" and scen == "matched_domain_null":
            report["hard_null_safety"][key] = hard_null_safety(rbv)
            continue
        cs = {}
        for name, a, b, flav in CONTRASTS:
            mean, lo, hi, ns = _contrast(rbv, a, b, _METRIC[flav], n_boot=n_boot, seed=boot_seed)
            cs[name] = dict(flavour=flav, a=a, b=b, mean=mean, ci_lo=lo, ci_hi=hi, n_seeds=ns,
                            meets_threshold=bool(abs(mean) >= THRESH) if mean == mean else False,
                            ci_excludes_zero=bool(lo == lo and (lo > 0 or hi < 0)))
        report["contrasts"][key] = cs
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    rep = analyze(_load(args.inp), n_boot=args.n_boot)
    txt = json.dumps(rep, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(txt)
    for scen, cs in rep["contrasts"].items():
        print(f"[{scen}]")
        for name, c in cs.items():
            flag = "*" if c["meets_threshold"] else " "
            print(f"  {flag} {name:18s} {c['mean']:+.3f} [{c['ci_lo']:+.3f},{c['ci_hi']:+.3f}] "
                  f"n_seeds={c['n_seeds']} ({c['flavour']})")
    for scen, s in rep["hard_null_safety"].items():
        print(f"[{scen}] HARD-NULL SAFETY: all_safe={s['all_safe']} identity_best={s['identity_in_oracle_best']}")
    if args.out:
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
