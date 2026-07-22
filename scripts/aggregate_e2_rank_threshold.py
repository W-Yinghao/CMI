#!/usr/bin/env python
"""E2 aggregate — subject-cluster bootstrap over the per-dump rank-threshold JSONs (freeze-before-aggregate).

REFUSES a partial matrix (9 subjects x 3 seeds = 27 dumps per backbone). Inference unit = held-out subject
(LOSO dump); seeds grouped within subject; subject-cluster bootstrap 95% CI over the 9 subjects.

Reports, per backbone:
  * r_D (= k_mean_complete): the analytic conditional-independence rank (Thm 1 primary quantity)
  * k_probe_chance + redundancy_rank (r_D - k_probe_chance): the weaker linear-probe rank + redundancy story
  * exact-head safety (ONLY for head_exact backbones): dim(S_D in ker/row), min principal angle,
    logit_change_remove_SD_relative
Never conflates r_D with k_probe_chance. Writes results/rank_threshold/summary.json.
"""
from __future__ import annotations
import argparse, glob, json
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
N_SUBJ, N_SEED = 9, 3


def _cluster_boot(by_subject, key, n_boot=10000, seed=0):
    """Subject-cluster bootstrap of the mean of `key` (seeds averaged within subject first)."""
    subj_means = []
    for s, rows in by_subject.items():
        vals = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
        if vals:
            subj_means.append(float(np.mean(vals)))
    if not subj_means:
        return None
    v = np.asarray(subj_means); rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(1)
    return {"point": float(v.mean()), "ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
            "n_subjects": len(v)}


def aggregate_backbone(recs, n_boot):
    by_subject = defaultdict(list)
    for r in recs:
        by_subject[int(r["target_subject"])].append(r)
    head_exact = bool(recs[0]["head_exact"])
    out = {"backbone": recs[0]["backbone"], "n_dumps": len(recs), "n_subjects": len(by_subject),
           "head_exact": head_exact, "exact_head_clause_reported": head_exact,
           "d_z": int(recs[0]["d_z"]),
           "r_D": _cluster_boot(by_subject, "r_D", n_boot),
           "k_mean_complete": _cluster_boot(by_subject, "k_mean_complete", n_boot),
           "k_probe_chance": _cluster_boot(by_subject, "k_probe_chance", n_boot),
           "redundancy_rank": _cluster_boot(by_subject, "redundancy_rank", n_boot),
           "subject_effective_rank": _cluster_boot(by_subject, "subject_effective_rank", n_boot)}
    if head_exact:
        out["exact_head_safety"] = {
            "dim_SD_in_ker": _cluster_boot(by_subject, "dim_SD_in_ker", n_boot),
            "dim_SD_in_row": _cluster_boot(by_subject, "dim_SD_in_row", n_boot),
            "min_angle_SD_row_deg": _cluster_boot(by_subject, "min_angle_SD_row_deg", n_boot),
            "logit_change_remove_SD_relative": _cluster_boot(by_subject, "logit_change_remove_SD_relative", n_boot),
            "predict": "S_D subset ker(W tilde): dim_SD_in_row small, min_angle_SD_row large, logit_change small",
        }
    else:
        out["note"] = "no verified exact linear head; exact-head clause NOT reported (head-free endpoints only)"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO / "results" / "rank_threshold"))
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()
    d = Path(args.dir)
    by_bb = defaultdict(list)
    for fp in glob.glob(str(d / "*.json")):
        if Path(fp).name == "summary.json":
            continue
        r = json.loads(Path(fp).read_text())
        by_bb[r["backbone"]].append(r)
    out = {"experiment": "E2_rank_threshold", "inference_unit": "held_out_subject (dump); seeds grouped",
           "n_boot": args.n_boot, "backbones": {}}
    for bb in ("EEGNet", "TSMNet"):
        recs = by_bb.get(bb, [])
        expected = N_SUBJ * N_SEED
        if len(recs) != expected:
            out["backbones"][bb] = {"status": "INCOMPLETE", "have": len(recs), "expected": expected}
        else:
            out["backbones"][bb] = {"status": "COMPLETE", **aggregate_backbone(recs, args.n_boot)}
    out["aggregate_valid"] = all(v.get("status") == "COMPLETE" for v in out["backbones"].values())
    (d / "summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({bb: out["backbones"][bb].get("status") for bb in out["backbones"]}, indent=2))
    if not out["aggregate_valid"]:
        print("REFUSING full aggregate — some backbone incomplete.")
    print(f"-> {d/'summary.json'}")


if __name__ == "__main__":
    main()
