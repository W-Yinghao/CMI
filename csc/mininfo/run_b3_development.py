"""
CSC Route B3-P2.2 — DEVELOPMENT runner (paired minimal-information certificate). Simulator-only,
dev-only; per-cluster logging. NOT a freeze, NOT a confirmatory run, NO real EEG. Independent target
clusters (one fresh geom seed per cluster); endpoints certificate-level with exact CP bounds.

PRIMARY h1 = full_z (R1: condition x FULL Z_std, fixed L2 C=0.5*3/d); pc (P2.1 low-rank) is run on the
SAME data as the "relative-to-P2.1" baseline. min_confirm_pairs=20 guard (positive claim only at m>=20);
RAW would-confirm reported separately as a label-efficiency diagnostic.

  python -m csc.mininfo.run_b3_development --clusters 24 --jobs 32 --out csc/results/b3_p22_dev_map.json
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

KINDS = list(PAIRED_TRUTH)
NO_CONCEPT_KINDS = [k for k, v in PAIRED_TRUTH.items() if v == "NO_CONCEPT"]
CONCEPT_KINDS = [k for k, v in PAIRED_TRUTH.items() if v == "CONCEPT"]


def _one_cluster(kind, m, cluster_seed, n_subjects, decide_n, alpha, n_boot, min_confirm_pairs):
    cfg = SimConfig(seed=cluster_seed)
    geom = make_geom(cfg, np.random.default_rng(cluster_seed))
    Z, Y, D, G, truth = make_paired_target(kind, geom, cfg, n_subjects=n_subjects,
                                           seed=10_000 + cluster_seed)
    # PRIMARY = full_z (B3-P2.2 R1); pc = P2.1 baseline on the SAME data (exact "relative-to-P2.1" compare)
    log = certify_paired(Z, Y, D, G, m=m, alpha=alpha, decide_n=decide_n, n_boot=n_boot,
                         min_confirm_pairs=min_confirm_pairs, h1_basis="full_z", seed=cluster_seed)
    pc = certify_paired(Z, Y, D, G, m=m, alpha=alpha, decide_n=decide_n, n_boot=n_boot,
                        min_confirm_pairs=min_confirm_pairs, h1_basis="pc", seed=cluster_seed)
    confirmed = (log["state"] == CONCEPT_CONFIRMED)
    return dict(cluster_seed=int(cluster_seed), kind=kind, truth=truth, label_budget_m=int(m),
                label_unit="subject_condition", state=log["state"], p_value=log["p_value"],
                T=log["T"], n_paired_available=log["n_paired_available"],
                n_queried_subjects=log.get("n_queried_subjects", log.get("n_queried", 0)),
                n_labeled_subject_conditions=log.get("n_labeled_subject_conditions", 0),
                n_labeled_epochs=log.get("n_labeled_epochs", 0),
                n_pairs=log.get("n_pairs", 0), classes_by_condition=log.get("classes_by_condition", {}),
                n_boot_invalid=log.get("n_boot_invalid", 0), valid=log["valid"], reason=log["reason"],
                h1_basis=log.get("h1_basis"), C_used=log.get("C_used"),
                n_features_interaction=log.get("n_features_interaction"),
                min_confirm_pairs=log.get("min_confirm_pairs"),
                would_confirm_without_min_pairs=log.get("would_confirm_without_min_pairs", False),
                false_confirm_flag=bool(confirmed and truth == "NO_CONCEPT"),
                power_flag=bool(confirmed and truth == "CONCEPT"),
                pc_state=pc["state"], pc_p_value=pc["p_value"],
                pc_would_confirm=pc.get("would_confirm_without_min_pairs", False))


def run(clusters=24, ms=(0, 5, 10, 20, 30), n_subjects=36, decide_n=20, alpha=0.05, n_boot=200,
        base_seed=0, min_confirm_pairs=20, n_jobs=1, out=None, quiet=True):
    if quiet:
        warnings.filterwarnings("ignore")
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(v, "1")
    tasks = [(k, m, base_seed + c) for k in KINDS for m in ms for c in range(clusters)]
    if n_jobs and n_jobs != 1:
        from joblib import Parallel, delayed
        recs = Parallel(n_jobs=n_jobs)(
            delayed(_one_cluster)(k, m, s, n_subjects, decide_n, alpha, n_boot, min_confirm_pairs)
            for k, m, s in tasks)
    else:
        recs = [_one_cluster(k, m, s, n_subjects, decide_n, alpha, n_boot, min_confirm_pairs)
                for k, m, s in tasks]

    # aggregate endpoints per (kind, m): full_z guarded confirm + RAW would-confirm (label-efficiency) +
    # pc P2.1 baseline (same data) for the "relative-to-P2.1" criterion.
    agg = {}
    for k in KINDS:
        for m in ms:
            rs = [r for r in recs if r["kind"] == k and r["label_budget_m"] == m]
            n = len(rs)
            confirmed = sum(r["state"] == CONCEPT_CONFIRMED for r in rs)
            would = sum(bool(r["would_confirm_without_min_pairs"]) for r in rs)
            pc_conf = sum(r["pc_state"] == CONCEPT_CONFIRMED for r in rs)
            states = {}
            for r in rs:
                states[r["state"]] = states.get(r["state"], 0) + 1
            blk = dict(kind=k, truth=PAIRED_TRUTH[k], m=m, n=n,
                       confirmed=confirmed, confirm_rate=(confirmed / n if n else None),
                       would_confirm=would, would_confirm_rate=(would / n if n else None),
                       pc_confirmed=pc_conf, pc_confirm_rate=(pc_conf / n if n else None),
                       state_counts=states)
            if PAIRED_TRUTH[k] == "NO_CONCEPT":
                blk["false_confirm_cp_upper"] = _cp_bound(confirmed, n, side="upper") if n else 1.0
            else:
                blk["power_cp_lower"] = _cp_bound(confirmed, n, side="lower") if n else 0.0
            agg[f"{k}@m{m}"] = blk

    payload = dict(kind="CSC Route B3-P2.2 DEVELOPMENT map (full_z R1 + min_confirm_pairs; pc baseline)",
                   status="DEVELOPMENT only -- simulator, dev seeds; NOT a freeze/confirmatory; NO real EEG.",
                   h1_basis="full_z", pc_baseline=True, min_confirm_pairs=min_confirm_pairs,
                   positive_claim_only_at_m_ge=min_confirm_pairs,
                   clusters_per_cell=clusters, label_budgets=list(ms), decide_n=decide_n, alpha=alpha,
                   n_boot=n_boot, base_seed=base_seed, n_subjects=n_subjects, n_jobs=n_jobs,
                   kinds=KINDS, no_concept_kinds=NO_CONCEPT_KINDS, concept_kinds=CONCEPT_KINDS,
                   endpoints=agg, per_cluster=recs)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[b3] wrote {out}")
    # console: full_z guarded CONFIRM, then full_z RAW would-confirm, then pc baseline guarded confirm
    def _table(title, key):
        print(f"=== B3-P2.2 {title} ===")
        print("kind                      truth        " + "  ".join(f"m{m:>2}" for m in ms))
        for k in KINDS:
            row = "  %-22s %-10s " % (k, PAIRED_TRUTH[k])
            row += "  ".join("%.2f" % (agg[f'{k}@m{m}'][key] or 0.0) for m in ms)
            print(row)
    _table("full_z CONFIRM rate (guarded, m>=%d only)" % min_confirm_pairs, "confirm_rate")
    _table("full_z would-confirm rate (RAW, label-efficiency)", "would_confirm_rate")
    _table("pc baseline CONFIRM rate (P2.1, same data)", "pc_confirm_rate")
    return payload


def main():
    ap = argparse.ArgumentParser(description="CSC Route B3-P2.2 development runner (paired; dev-only).")
    ap.add_argument("--clusters", type=int, default=24)
    ap.add_argument("--ms", type=int, nargs="+", default=[0, 5, 10, 20, 30])
    ap.add_argument("--n_subjects", type=int, default=36)
    ap.add_argument("--decide_n", type=int, default=20)
    ap.add_argument("--min_confirm_pairs", type=int, default=20)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--base_seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", type=str, default="csc/results/b3_p22_dev_map.json")
    a = ap.parse_args()
    run(clusters=a.clusters, ms=tuple(a.ms), n_subjects=a.n_subjects, decide_n=a.decide_n,
        min_confirm_pairs=a.min_confirm_pairs, n_boot=a.n_boot, base_seed=a.base_seed, n_jobs=a.jobs,
        out=a.out)


if __name__ == "__main__":
    main()
