"""
CSC Route B3-P2.3 — EXPANDED development map. METHOD LOCKED to `pc_centered` (h1_basis="pc",
condition_coding="centered", rank=3, C=0.5, min_confirm_pairs=20). NO full_z / RFF / score-space; NO
sweeps. Simulator-only, dev-only; NOT a freeze/confirmatory; NO real EEG. Goal: show `pc_centered` is not
a P2.2 fluke — stress controls (incl. covariate+label, missing-pair, unequal-epochs) under a star-grid of
difficulty axes, with exact CP bounds, before any freeze candidate.

  python -m csc.mininfo.run_b3_p23 --clusters 24 --jobs 24 --out csc/results/b3_p23_dev_map.json
  python -m csc.mininfo.run_b3_p23 --canary                       # 4 clusters, baseline+1 stress
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np

from csc.protocol import _cp_bound
from csc.sim.shift_simulator import SimConfig, make_geom
from .paired_sim import make_paired_target, PAIRED_TRUTH
from .paired_certifier import (certify_paired, CONCEPT_CONFIRMED, NO_CONCEPT_EVIDENCE,
                               NEED_MORE_LABELS, INVALID_PAIR, UNIDENTIFIABLE)

# method lock (machine-readable; written into the artifact)
METHOD_LOCK = dict(method="pc_centered", h1_basis="pc", condition_coding="centered", rank=3, C=0.5,
                   min_confirm_pairs=20, alpha=0.05, decide_n=20, development_only=True)

CONTROLS = ["clean", "paired_covariate", "paired_label", "random_label",
            "paired_covariate_plus_label", "missing_pair", "unequal_epochs_extreme"]
POSITIVES = ["paired_concept", "paired_concept_plus_cov", "paired_pure_conditional"]
KINDS = CONTROLS + POSITIVES

# star-grid: baseline + one-axis-off stress presets (cfg_* -> SimConfig; rest -> make_paired_target)
SCENARIOS = {
    "baseline":         {},
    "high_subject_tau": {"cfg_subject_tau": 1.0},
    "high_nuisance":    {"cov_scale": 18.0},
    "imbalanced":       {"base_prior": [0.60, 0.25, 0.15]},
    "label_noise":      {"label_noise": 0.10},
    "few_epochs":       {"cfg_epochs_min": 6, "cfg_epochs_max": 12},
}


def _one_cell(kind, scen_name, scen, m, cluster_seed, n_subjects, n_boot):
    cfg = SimConfig(seed=cluster_seed,
                    subject_tau=scen.get("cfg_subject_tau", SimConfig.subject_tau),
                    epochs_min=scen.get("cfg_epochs_min", SimConfig.epochs_min),
                    epochs_max=scen.get("cfg_epochs_max", SimConfig.epochs_max))
    geom = make_geom(cfg, np.random.default_rng(cluster_seed))
    Z, Y, D, G, truth = make_paired_target(
        kind, geom, cfg, n_subjects=n_subjects, seed=10_000 + cluster_seed,
        cov_scale=scen.get("cov_scale", 10.0), base_prior=scen.get("base_prior"),
        label_noise=scen.get("label_noise", 0.0), pure_cond_frac=scen.get("pure_cond_frac", 0.35))
    log = certify_paired(Z, Y, D, G, m=m, alpha=METHOD_LOCK["alpha"], decide_n=METHOD_LOCK["decide_n"],
                         min_confirm_pairs=METHOD_LOCK["min_confirm_pairs"], h1_basis="pc",
                         condition_coding="centered", rank=3, C=METHOD_LOCK["C"], n_boot=n_boot,
                         seed=cluster_seed)
    confirmed = (log["state"] == CONCEPT_CONFIRMED)
    cbc = log.get("classes_by_condition", {}) or {}
    coverage_fail = any(v < 2 for v in cbc.values()) if cbc else False
    return dict(scenario=scen_name, cluster_seed=int(cluster_seed), kind=kind, truth=truth,
                label_budget_m=int(m), n_subjects_total=int(n_subjects),
                n_queried_subjects=log.get("n_queried_subjects", 0),
                n_labeled_subject_conditions=log.get("n_labeled_subject_conditions", 0),
                n_labeled_epochs=log.get("n_labeled_epochs", 0),
                condition_coding=log.get("condition_coding"), h1_basis=log.get("h1_basis"),
                rank=3, C=log.get("C_used"), min_confirm_pairs=log.get("min_confirm_pairs"),
                T=log.get("T"), p_value=log.get("p_value"), null_mean=log.get("null_mean"),
                null_sd=log.get("null_sd"), n_boot_invalid=log.get("n_boot_invalid", 0),
                classes_by_condition=cbc, n_pairs=log.get("n_pairs", 0), state=log["state"],
                reason=log.get("reason", ""), false_confirm_flag=bool(confirmed and truth == "NO_CONCEPT"),
                power_flag=bool(confirmed and truth == "CONCEPT"),
                would_confirm_without_guard=log.get("would_confirm_without_guard", False),
                class_coverage_fail=bool(coverage_fail))


def _agg_cell(rs, truth, n_boot):
    n = len(rs)
    confirmed = sum(r["state"] == CONCEPT_CONFIRMED for r in rs)
    states = {}
    for r in rs:
        states[r["state"]] = states.get(r["state"], 0) + 1
    blk = dict(n=n, confirmed=confirmed, confirm_rate=(confirmed / n if n else None),
               would_confirm=sum(bool(r["would_confirm_without_guard"]) for r in rs),
               state_counts=states,
               mean_invalid_frac=float(np.mean([r["n_boot_invalid"] / n_boot for r in rs])) if n else None,
               class_coverage_fails=sum(bool(r["class_coverage_fail"]) for r in rs),
               need_more_rate=states.get(NEED_MORE_LABELS, 0) / n if n else None,
               no_concept_rate=states.get(NO_CONCEPT_EVIDENCE, 0) / n if n else None,
               invalid_pair_rate=states.get(INVALID_PAIR, 0) / n if n else None,
               unidentifiable_rate=states.get(UNIDENTIFIABLE, 0) / n if n else None)
    if truth == "NO_CONCEPT":
        blk["false_confirm_count"] = confirmed
        blk["false_confirm_cp_upper"] = _cp_bound(confirmed, n, side="upper") if n else 1.0
    else:
        blk["power"] = confirmed / n if n else None
        blk["power_cp_lower"] = _cp_bound(confirmed, n, side="lower") if n else 0.0
    return blk


def run(clusters=24, ms=(0, 10, 20, 30), n_subjects=36, n_boot=200, base_seed=0,
        scenarios=None, n_jobs=1, out=None, quiet=True):
    if quiet:
        warnings.filterwarnings("ignore")
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(v, "1")
    scn = scenarios or SCENARIOS
    tasks = [(k, sn, scn[sn], m, base_seed + c)
             for sn in scn for k in KINDS for m in ms for c in range(clusters)]
    if n_jobs and n_jobs != 1:
        from joblib import Parallel, delayed
        recs = Parallel(n_jobs=n_jobs)(
            delayed(_one_cell)(k, sn, sp, m, s, n_subjects, n_boot) for k, sn, sp, m, s in tasks)
    else:
        recs = [_one_cell(k, sn, sp, m, s, n_subjects, n_boot) for k, sn, sp, m, s in tasks]

    agg = {}
    for sn in scn:
        for k in KINDS:
            for m in ms:
                rs = [r for r in recs if r["scenario"] == sn and r["kind"] == k
                      and r["label_budget_m"] == m]
                agg[f"{sn}|{k}|m{m}"] = {**dict(scenario=sn, kind=k, truth=PAIRED_TRUTH[k], m=m),
                                         **_agg_cell(rs, PAIRED_TRUTH[k], n_boot)}

    payload = dict(kind="CSC Route B3-P2.3 EXPANDED development map (pc_centered LOCKED)",
                   status="DEVELOPMENT only -- simulator/dev seeds; NOT a freeze/confirmatory; NO real EEG. "
                          "No finite-sample type-I control is claimed.",
                   method_lock=METHOD_LOCK, scenarios=list(scn), controls=CONTROLS, positives=POSITIVES,
                   clusters_per_cell=clusters, label_budgets=list(ms), n_subjects=n_subjects,
                   n_boot=n_boot, base_seed=base_seed, n_jobs=n_jobs, endpoints=agg, per_cluster=recs)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[b3p23] wrote {out}")
    # console: per-scenario confirm-rate table (controls should stay low, positives high)
    for sn in scn:
        print(f"=== B3-P2.3 [{sn}] confirm rate (pc_centered; controls low / positives high) ===")
        print("kind                       truth        " + "  ".join(f"m{m:>2}" for m in ms))
        for k in KINDS:
            row = "  %-23s %-10s " % (k, PAIRED_TRUTH[k])
            row += "  ".join("%.2f" % (agg[f'{sn}|{k}|m{m}']['confirm_rate'] or 0.0) for m in ms)
            print(row)
    return payload


def main():
    ap = argparse.ArgumentParser(description="CSC Route B3-P2.3 expanded dev runner (pc_centered locked).")
    ap.add_argument("--clusters", type=int, default=24)
    ap.add_argument("--ms", type=int, nargs="+", default=[0, 10, 20, 30])
    ap.add_argument("--n_subjects", type=int, default=36)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--base_seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--canary", action="store_true", help="4 clusters, baseline+high_nuisance only")
    ap.add_argument("--out", type=str, default="csc/results/b3_p23_dev_map.json")
    a = ap.parse_args()
    scen = {k: SCENARIOS[k] for k in ("baseline", "high_nuisance")} if a.canary else None
    run(clusters=(4 if a.canary else a.clusters), ms=tuple(a.ms), n_subjects=a.n_subjects,
        n_boot=a.n_boot, base_seed=a.base_seed, scenarios=scen, n_jobs=a.jobs,
        out=(a.out.replace(".json", "_canary.json") if a.canary else a.out))


if __name__ == "__main__":
    main()
