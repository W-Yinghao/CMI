#!/usr/bin/env python
"""E1 aggregate — subject-cluster bootstrap over the per-fold spectrum JSONs (freeze-before-aggregate).

REFUSES to aggregate a partial matrix: requires every (dataset, seed, fold) cell present (9+12 folds x 3
seeds). Primary endpoints, each with a subject-cluster (fold) bootstrap 95% CI, seeds grouped within a fold:
  * corr(tau_erm, delta_lambda) over matched direction-pairs pooled within resampled folds  (predict > 0)
  * top-subject-direction exact-head reliance, ERM vs CIGL                                    (predict rises)
  * subject effective rank, ERM vs CIGL                                                       (predict falls)
  * top-2 energy concentration + top-direction head alignment, ERM vs CIGL                    (predict rise)
Writes results/spectrum/summary.json. Run only after BOTH datasets' fleets complete.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
FOLDS = {"BNCI2014_001": 9, "BNCI2015_001": 12}
SEEDS = (0, 1, 2)


def _load_cells(spec_dir, dataset):
    cells = {}
    for seed in SEEDS:
        for fold in range(FOLDS[dataset]):
            fp = spec_dir / f"{dataset}_seed{seed}_fold{fold}.json"
            if fp.exists():
                cells[(seed, fold)] = json.loads(fp.read_text())
    return cells


def _fold_bootstrap(fold_values, stat, n_boot=10000, seed=0):
    """Cluster bootstrap over folds (the outer subject unit). fold_values: list keyed by fold; `stat` maps a
    resampled list of fold-entries -> scalar. Returns (point, lo, hi)."""
    folds = list(fold_values)
    if not folds:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    point = stat(folds)
    boots = []
    for _ in range(n_boot):
        samp = [folds[i] for i in rng.integers(0, len(folds), len(folds))]
        v = stat(samp)
        if np.isfinite(v):
            boots.append(v)
    return float(point), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def _corr_stat(fold_entries):
    tau = np.concatenate([np.asarray(e["tau"]) for e in fold_entries])
    dlam = np.concatenate([np.asarray(e["dlam"]) for e in fold_entries])
    if len(tau) < 3 or np.std(tau) < 1e-12 or np.std(dlam) < 1e-12:
        return np.nan
    return float(np.corrcoef(tau, dlam)[0, 1])


def _paired_mean_stat(key):
    return lambda fold_entries: float(np.mean([e[key] for e in fold_entries]))


def aggregate_dataset(spec_dir, dataset, n_boot):
    cells = _load_cells(spec_dir, dataset)
    expected = len(SEEDS) * FOLDS[dataset]
    if len(cells) != expected:
        return {"dataset": dataset, "status": "INCOMPLETE", "have": len(cells), "expected": expected,
                "missing": [f"seed{s}_fold{f}" for s in SEEDS for f in range(FOLDS[dataset])
                            if (s, f) not in cells]}
    # firewall gate: refuse any cell that did not pass source-only firewall (fail-loud, reason-coded)
    bad = [k for k, c in cells.items() if not c.get("firewall_ok", False)]
    if bad:
        return {"dataset": dataset, "status": "FIREWALL_FAIL", "failed_cells": [f"seed{s}_fold{f}" for s, f in bad]}
    # group by fold (seeds travel together): average seed-level scalars within a fold
    by_fold = defaultdict(list)
    for (seed, fold), c in cells.items():
        by_fold[fold].append(c)
    fold_corr, fold_corr_rank, fold_scalar = [], [], []
    for fold, cs in by_fold.items():
        tau = np.concatenate([[p["tau_erm"] for p in c["pairs"]] for c in cs])
        dlam = np.concatenate([[p["delta_lambda"] for p in c["pairs"]] for c in cs])
        fold_corr.append({"tau": tau, "dlam": dlam})
        taur = np.concatenate([[p["tau_erm"] for p in c["pairs_rank"]] for c in cs])
        dlamr = np.concatenate([[p["delta_lambda"] for p in c["pairs_rank"]] for c in cs])
        fold_corr_rank.append({"tau": taur, "dlam": dlamr})
        fold_scalar.append({
            "top_rel_erm": np.mean([c["top_dir_reliance_erm"] for c in cs]),
            "top_rel_cigl": np.mean([c["top_dir_reliance_cigl"] for c in cs]),
            "eff_erm": np.mean([c["erm"]["effective_rank"] for c in cs]),
            "eff_cigl": np.mean([c["cigl"]["effective_rank"] for c in cs]),
            "top2_erm": np.mean([c["erm"]["top2_energy_concentration"] for c in cs]),
            "top2_cigl": np.mean([c["cigl"]["top2_energy_concentration"] for c in cs]),
            "align_erm": np.mean([c["top_dir_alignment_erm"] for c in cs]),
            "align_cigl": np.mean([c["top_dir_alignment_cigl"] for c in cs]),
            "d_top_rel": np.mean([c["top_dir_reliance_cigl"] - c["top_dir_reliance_erm"] for c in cs]),
            "d_eff": np.mean([c["cigl"]["effective_rank"] - c["erm"]["effective_rank"] for c in cs]),
        })

    def ci(key):
        return _fold_bootstrap(fold_scalar, _paired_mean_stat(key), n_boot=n_boot)

    corr = _fold_bootstrap(fold_corr, _corr_stat, n_boot=n_boot)
    corr_rank = _fold_bootstrap(fold_corr_rank, _corr_stat, n_boot=n_boot)
    return {
        "dataset": dataset, "status": "COMPLETE", "n_folds": len(by_fold),
        "corr_tau_dlambda": {"point": corr[0], "ci95": [corr[1], corr[2]], "predict": ">0",
                             "pairing": "cosine (common raw ambient, optimal assignment)"},
        "corr_tau_dlambda_rank_pairing": {"point": corr_rank[0], "ci95": [corr_rank[1], corr_rank[2]],
                                          "predict": ">0", "pairing": "energy-rank (robustness)"},
        "delta_top_dir_reliance_cigl_minus_erm": {"point": ci("d_top_rel")[0], "ci95": list(ci("d_top_rel")[1:]),
                                                  "predict": ">0 (rises)"},
        "delta_effective_rank_cigl_minus_erm": {"point": ci("d_eff")[0], "ci95": list(ci("d_eff")[1:]),
                                                "predict": "<0 (falls)"},
        "top_dir_reliance": {"erm": ci("top_rel_erm")[0], "cigl": ci("top_rel_cigl")[0]},
        "effective_rank": {"erm": ci("eff_erm")[0], "cigl": ci("eff_cigl")[0]},
        "top2_energy_concentration": {"erm": ci("top2_erm")[0], "cigl": ci("top2_cigl")[0]},
        "top_dir_head_alignment": {"erm": ci("align_erm")[0], "cigl": ci("align_cigl")[0]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec_dir", default=str(REPO / "results" / "spectrum"))
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()
    spec_dir = Path(args.spec_dir)
    out = {"experiment": "E1_subject_spectrum", "inference_unit": "held_out_subject_fold; seeds grouped",
           "n_boot": args.n_boot, "datasets": {}}
    for ds in FOLDS:
        out["datasets"][ds] = aggregate_dataset(spec_dir, ds, args.n_boot)
    incomplete = [ds for ds, r in out["datasets"].items() if r.get("status") != "COMPLETE"]
    out["aggregate_valid"] = (len(incomplete) == 0)
    if incomplete:
        print(f"REFUSING full aggregate — incomplete datasets: {incomplete}")
    (spec_dir / "summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({ds: out["datasets"][ds].get("status") for ds in FOLDS}, indent=2))
    print(f"-> {spec_dir/'summary.json'}")


if __name__ == "__main__":
    main()
