#!/usr/bin/env python
"""E2 runner — linear removability threshold r_D + head-geometry overlaps (Theorem 1).

Reuse (no retrain): banked frozen EEGNet(16) + TSMNet(210) LOSO dumps on BCI-IV-2a (BNCI2014_001),
`tos_cmi/results/tos_cmi_eeg_frozen/BNCI2014_001_{backbone}_LOSO/subX_erm_lam0_seedY.npz`. Each dump stores
Z_source [N,dz] + logits_source [N,K] (source-only already) + subject_source. The linear head is RECOVERED
from (Z,logits) and FAIL-CLOSED verified (TSMNet exact; EEGNet not -> head_exact=False, exact-head clause
reported as probe/indicative).

  probe:  python scripts/run_e2_rank_threshold.py --backbone TSMNet --probe
  fleet:  python scripts/run_e2_rank_threshold.py --backbone TSMNet --seeds 0 1 2
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cmi.eval.rank_threshold import rank_threshold_fold           # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TOS_ROOT = Path("/home/infres/yinwang/CMI_AAAI_tos/tos_cmi/results/tos_cmi_eeg_frozen")
DATASET = "BNCI2014_001"


def dumps_for(tos_root: Path, backbone, seed):
    d = tos_root / f"{DATASET}_{backbone}_LOSO"
    return sorted(d.glob(f"sub*_erm_lam0_seed{seed}.npz"))


def run_dump(path, n_perm, seed):
    z = np.load(path, allow_pickle=True)
    rec = rank_threshold_fold(z["Z_source"], z["y_source"], z["subject_source"], z["logits_source"],
                              seed=seed, n_perm=n_perm)
    rec.update(backbone=str(z["backbone"]), dataset=str(z["dataset"]),
               target_subject=int(z["target_subject"]), seed=int(z["seed"]),
               z_dim_declared=int(z["z_dim"]))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["EEGNet", "TSMNet"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tos_root", default=str(DEFAULT_TOS_ROOT))
    ap.add_argument("--out_dir", default=str(REPO / "results" / "rank_threshold"))
    ap.add_argument("--n_perm", type=int, default=50)
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    tos_root = Path(args.tos_root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if args.probe:
        paths = dumps_for(tos_root, args.backbone, args.seeds[0])
        if not paths:
            raise SystemExit(f"no dumps for {args.backbone} seed{args.seeds[0]} under {tos_root}")
        rec = run_dump(paths[0], n_perm=max(8, args.n_perm // 6), seed=args.seeds[0])
        qc = {
            "PROBE": f"{args.backbone} {Path(paths[0]).name}",
            "d_z": rec["d_z"], "head_exact": rec["head_exact"],
            "head_replay_max_abs_diff": round(rec["head_replay_max_abs_diff"], 3),
            "r_D": rec["r_D"], "energy_rank_99": rec["energy_rank_99"],
            "dim_SD_in_ker": rec["dim_SD_in_ker"], "dim_SD_in_row": rec["dim_SD_in_row"],
            "min_angle_SD_row_deg": round(rec["min_angle_SD_row_deg"], 1),
            "min_angle_SD_ker_deg": round(rec["min_angle_SD_ker_deg"], 1),
            "exact_head_clause_reported": rec["exact_head_clause_reported"],
            "logit_change_remove_SD_relative": (None if rec["logit_change_remove_SD_relative"] is None
                                                else round(rec["logit_change_remove_SD_relative"], 4)),
            "k_mean_complete_(=r_D analytic)": rec["k_mean_complete"],
            "k_probe_chance_(weaker)": rec["k_probe_chance"],
            "redundancy_rank_(r_D - probe)": rec["redundancy_rank"],
            "sweep_len": len(rec["sweep"]),
            "resid_subj_bacc_k1": round(rec["sweep"][0]["resid_subject_bacc_linear"], 3),
            "resid_subj_bacc_last": round(rec["sweep"][-1]["resid_subject_bacc_linear"], 3),
        }
        print(json.dumps(qc, indent=2))
        print("\nQC NOTE: single-dump numbers are for gating only; NOT an aggregate.")
        return

    n = 0
    for seed in args.seeds:
        for p in dumps_for(tos_root, args.backbone, seed):
            fp = out_dir / f"{args.backbone}_{Path(p).stem}.json"
            if fp.exists():
                n += 1; continue
            fp.write_text(json.dumps(run_dump(p, n_perm=args.n_perm, seed=seed), indent=2))
            n += 1
    print(f"[e2] {args.backbone}: {n} dump-cells present under {out_dir}")


if __name__ == "__main__":
    main()
