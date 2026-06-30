"""
CSC Route B3-P2.4 — calibrated EXPANDED development map. Method LOCKED `pc_centered_calibrated`
(pair-integrity guard + eligible-complete-pair guard + class-balanced cross-fitted test). Same grid as
P2.3 (10 kinds x 6 scenarios x m{0,10,20,30} x 24 clusters) for direct comparison. Simulator-only,
dev-only; NOT a freeze/confirmatory; NO real EEG.

  python -m csc.mininfo.run_b3_p24 --clusters 24 --jobs 24 --out csc/results/b3_p24_dev_map.json
  python -m csc.mininfo.run_b3_p24 --canary
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
from .paired_calibrated import (certify_paired_calibrated, CALIBRATION_VERSION, PAIR_INTEGRITY_MIN,
                                MIN_EPOCHS_PER_CONDITION, N_FOLDS)
from .paired_certifier import (CONCEPT_CONFIRMED, NO_CONCEPT_EVIDENCE, NEED_MORE_LABELS, INVALID_PAIR,
                               UNIDENTIFIABLE)
from .run_b3_p23 import KINDS, CONTROLS, POSITIVES, SCENARIOS

METHOD_LOCK = dict(method="pc_centered_calibrated", h1_basis="pc", condition_coding="centered", rank=3,
                   C=0.5, min_confirm_pairs=20, alpha=0.05, decide_n=20, n_folds=N_FOLDS,
                   pair_integrity_min=PAIR_INTEGRITY_MIN, min_epochs_per_condition=MIN_EPOCHS_PER_CONDITION,
                   calibration_version=CALIBRATION_VERSION, development_only=True)


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
    log = certify_paired_calibrated(Z, Y, D, G, m=m, alpha=METHOD_LOCK["alpha"],
                                    decide_n=METHOD_LOCK["decide_n"],
                                    min_confirm_pairs=METHOD_LOCK["min_confirm_pairs"], rank=3,
                                    C=METHOD_LOCK["C"], n_folds=N_FOLDS, n_boot=n_boot, seed=cluster_seed)
    confirmed = (log["state"] == CONCEPT_CONFIRMED)
    return dict(scenario=scen_name, cluster_seed=int(cluster_seed), kind=kind, truth=truth,
                label_budget_m=int(m), n_subjects_total=log.get("n_subjects_total"),
                n_complete_pairs=log.get("n_complete_pairs"), pair_integrity=log.get("pair_integrity"),
                missing_condition_fraction=log.get("missing_condition_fraction"),
                n_eligible_complete_pairs=log.get("n_eligible_complete_pairs"),
                n_queried_subjects=log.get("n_queried_subjects", 0),
                n_eligible_queried=log.get("n_eligible_queried", 0),
                min_epochs_per_condition=log.get("min_epochs_per_condition"),
                class_cell_counts_by_condition=log.get("class_cell_counts_by_condition", {}),
                class_balance_gate_status=log.get("class_balance_gate_status"),
                n_folds=log.get("n_folds"), fold_hash=log.get("fold_hash"),
                T_cv=log.get("T_cv"), p_value_cv=log.get("p_value_cv"),
                null_mean_cv=log.get("null_mean_cv"), null_sd_cv=log.get("null_sd_cv"),
                n_boot_invalid=log.get("n_boot_invalid", 0), calibration_version=CALIBRATION_VERSION,
                state=log["state"], reason=log.get("reason", ""),
                would_confirm_without_guard=log.get("would_confirm_without_guard", False),
                false_confirm_flag=bool(confirmed and truth == "NO_CONCEPT"),
                power_flag=bool(confirmed and truth == "CONCEPT"))


def _agg(rs, truth, n_boot):
    n = len(rs); confirmed = sum(r["state"] == CONCEPT_CONFIRMED for r in rs)
    states = {}
    for r in rs:
        states[r["state"]] = states.get(r["state"], 0) + 1
    pis = [r["pair_integrity"] for r in rs if r["pair_integrity"] is not None]
    blk = dict(n=n, confirmed=confirmed, confirm_rate=(confirmed / n if n else None),
               would_confirm=sum(bool(r["would_confirm_without_guard"]) for r in rs), state_counts=states,
               need_more_rate=states.get(NEED_MORE_LABELS, 0) / n if n else None,
               invalid_pair_rate=states.get(INVALID_PAIR, 0) / n if n else None,
               no_concept_rate=states.get(NO_CONCEPT_EVIDENCE, 0) / n if n else None,
               mean_eligible_pairs=float(np.mean([r["n_eligible_complete_pairs"] or 0 for r in rs])) if n else None,
               pair_integrity_min=float(np.min(pis)) if pis else None,
               pair_integrity_mean=float(np.mean(pis)) if pis else None,
               crossfit_invalid_cells=sum((r["n_boot_invalid"] or 0) > 0 for r in rs),
               mean_invalid_frac=float(np.mean([(r["n_boot_invalid"] or 0) / n_boot for r in rs])) if n else None)
    if truth == "NO_CONCEPT":
        blk["false_confirm_count"] = confirmed
        blk["false_confirm_cp_upper"] = _cp_bound(confirmed, n, side="upper") if n else 1.0
    else:
        blk["power"] = confirmed / n if n else None
        blk["power_cp_lower"] = _cp_bound(confirmed, n, side="lower") if n else 0.0
    return blk


def run(clusters=24, ms=(0, 10, 20, 30), n_subjects=36, n_boot=200, base_seed=0, scenarios=None,
        n_jobs=1, out=None, quiet=True):
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
                                         **_agg(rs, PAIRED_TRUTH[k], n_boot)}

    payload = dict(kind="CSC Route B3-P2.4 calibrated EXPANDED development map (pc_centered_calibrated)",
                   status="DEVELOPMENT only -- simulator/dev seeds; NOT a freeze/confirmatory; NO real EEG. "
                          "No finite-sample type-I control is claimed.",
                   method_lock=METHOD_LOCK, scenarios=list(scn), controls=CONTROLS, positives=POSITIVES,
                   clusters_per_cell=clusters, label_budgets=list(ms), n_subjects=n_subjects,
                   n_boot=n_boot, base_seed=base_seed, n_jobs=n_jobs, endpoints=agg, per_cluster=recs)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[b3p24] wrote {out}")
    for sn in scn:
        print(f"=== B3-P2.4 [{sn}] confirm rate (pc_centered_calibrated; controls low / positives high) ===")
        print("kind                       truth        " + "  ".join(f"m{m:>2}" for m in ms))
        for k in KINDS:
            row = "  %-23s %-10s " % (k, PAIRED_TRUTH[k])
            row += "  ".join("%.2f" % (agg[f'{sn}|{k}|m{m}']['confirm_rate'] or 0.0) for m in ms)
            print(row)
    return payload


def main():
    ap = argparse.ArgumentParser(description="CSC Route B3-P2.4 calibrated dev runner (pc_centered_calibrated).")
    ap.add_argument("--clusters", type=int, default=24)
    ap.add_argument("--ms", type=int, nargs="+", default=[0, 10, 20, 30])
    ap.add_argument("--n_subjects", type=int, default=36)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--base_seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--canary", action="store_true")
    ap.add_argument("--out", type=str, default="csc/results/b3_p24_dev_map.json")
    a = ap.parse_args()
    scen = {k: SCENARIOS[k] for k in ("baseline", "high_subject_tau")} if a.canary else None
    run(clusters=(4 if a.canary else a.clusters), ms=tuple(a.ms), n_subjects=a.n_subjects,
        n_boot=a.n_boot, base_seed=a.base_seed, scenarios=scen, n_jobs=a.jobs,
        out=(a.out.replace(".json", "_canary.json") if a.canary else a.out))


if __name__ == "__main__":
    main()
