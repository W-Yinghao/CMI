#!/usr/bin/env python
"""E1 runner — subject-information x exact-head-use spectrum (Theorem 2) over banked DGCNN audit sidecars.

Reuse (no retrain): the confirmed cmi_trace_p0p1 objective-comparison .audit.npz sidecars (ERM + CIGL,
seeds 0/1/2, all LOSO folds), which carry graph_z + a VERIFIED linear head. Provenance is validated
(config_sha256 in the run manifest must match the local configs/cmi_trace_p0p1.yaml) before any reuse.

Per (dataset, seed, fold): compute the whitened subject spectrum for ERM and CIGL, pair directions by
principal angle, emit per-fold JSON. Aggregation (corr(tau,Δλ), eff-rank, top-dir reliance, energy, alignment
with subject-cluster bootstrap CI) runs ONLY when the full matrix is present (freeze-before-aggregate).

  probe:  python scripts/run_e1_spectrum.py --dataset BNCI2014_001 --probe --fold 0 --seed 0
  fleet:  python scripts/run_e1_spectrum.py --dataset BNCI2014_001 --seeds 0 1 2 --k_spec 16 --n_perm 50
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cmi.eval.audit_npz import load_audit_npz          # noqa: E402
from cmi.eval.subject_spectrum import subject_spectrum, paired_delta_lambda  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_ROOT = Path("/home/infres/yinwang/CMI_AAAI_cmitrace/results/cmi_trace_p0p1")
METHODS = ("erm", "cigl_graph_node")
FOLDS = {"BNCI2014_001": 9, "BNCI2015_001": 12}


def validate_provenance(audit_root: Path):
    """Fail-loud provenance gate: the banked run's config_sha256 must match the local frozen config."""
    man = audit_root / "manifest.json"
    if not man.exists():
        raise SystemExit(f"ProvenanceError: no manifest at {man}")
    m = json.loads(man.read_text())
    local = hashlib.sha256((REPO / "configs" / "cmi_trace_p0p1.yaml").read_bytes()).hexdigest()
    if m.get("config_sha256") != local:
        raise SystemExit(f"ProvenanceError: config_sha256 mismatch\n manifest={m.get('config_sha256')}\n local={local}")
    return m


def sidecar_path(audit_root: Path, dataset, fold, method, seed):
    d = audit_root / "objective_comparison" / dataset / "audit"
    hits = sorted(d.glob(f"{dataset}_fold{fold}_sub*_{method}_seed{seed}.audit.npz"))
    return hits[0] if hits else None


def run_fold(audit_root, dataset, seed, fold, k_spec, n_perm, n_random, out_dir):
    specs = {}
    for method in METHODS:
        p = sidecar_path(audit_root, dataset, fold, method, seed)
        if p is None:
            raise FileNotFoundError(f"missing sidecar: {dataset} fold{fold} {method} seed{seed}")
        data = load_audit_npz(str(p))
        specs[method] = subject_spectrum(data, k_spec=k_spec, n_perm=n_perm, n_random=n_random, seed=seed)
    pairs_cos = paired_delta_lambda(specs["erm"], specs["cigl_graph_node"], mode="cosine")
    pairs_rank = paired_delta_lambda(specs["erm"], specs["cigl_graph_node"], mode="rank")
    row = {
        "dataset": dataset, "seed": int(seed), "fold": int(fold),
        "firewall_ok": bool(specs["erm"]["firewall_passed"] and specs["cigl_graph_node"]["firewall_passed"]),
        "erm": {k: specs["erm"][k] for k in ("effective_rank", "top2_energy_concentration", "firewall_passed",
                                             "head_replay_verified", "target_subject")},
        "cigl": {k: specs["cigl_graph_node"][k] for k in ("effective_rank", "top2_energy_concentration",
                                                          "firewall_passed", "head_replay_verified")},
        "top_dir_reliance_erm": specs["erm"]["directions"][0]["tau_ce_reliance"],
        "top_dir_reliance_cigl": specs["cigl_graph_node"]["directions"][0]["tau_ce_reliance"],
        "top_dir_alignment_erm": specs["erm"]["directions"][0]["head_alignment"],
        "top_dir_alignment_cigl": specs["cigl_graph_node"]["directions"][0]["head_alignment"],
        "pairs": pairs_cos, "pairs_rank": pairs_rank,       # primary=cosine (common-ambient); rank=robustness
        "spectrum_erm": specs["erm"]["directions"], "spectrum_cigl": specs["cigl_graph_node"]["directions"],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{dataset}_seed{seed}_fold{fold}.json").write_text(json.dumps(row, indent=2))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(FOLDS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--audit_root", default=str(DEFAULT_AUDIT_ROOT))
    ap.add_argument("--out_dir", default=str(REPO / "results" / "spectrum"))
    ap.add_argument("--k_spec", type=int, default=16)
    ap.add_argument("--n_perm", type=int, default=50)
    ap.add_argument("--n_random", type=int, default=50)
    ap.add_argument("--probe", action="store_true", help="run a single (fold,seed) and print QC, no aggregate")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    audit_root = Path(args.audit_root)
    man = validate_provenance(audit_root)
    out_dir = Path(args.out_dir)

    if args.probe:
        row = run_fold(audit_root, args.dataset, args.seed, args.fold,
                       args.k_spec, args.n_perm, args.n_random, out_dir)
        qc = {
            "PROBE": f"{args.dataset} seed{args.seed} fold{args.fold}",
            "provenance_config_sha_ok": True,
            "erm_firewall": row["erm"]["firewall_passed"], "erm_head_verified": row["erm"]["head_replay_verified"],
            "cigl_firewall": row["cigl"]["firewall_passed"], "cigl_head_verified": row["cigl"]["head_replay_verified"],
            "erm_eff_rank": round(row["erm"]["effective_rank"], 3),
            "cigl_eff_rank": round(row["cigl"]["effective_rank"], 3),
            "top_dir_reliance_erm": round(row["top_dir_reliance_erm"], 4),
            "top_dir_reliance_cigl": round(row["top_dir_reliance_cigl"], 4),
            "n_pairs": len(row["pairs"]),
            "lambda_finite": all(np.isfinite(p["delta_lambda"]) for p in row["pairs"]),
            "tau_finite": all(np.isfinite(d["tau_ce_reliance"]) for d in row["spectrum_erm"]),
            "single_fold_corr_tau_dlambda": round(float(np.corrcoef(
                [p["tau_erm"] for p in row["pairs"]], [p["delta_lambda"] for p in row["pairs"]])[0, 1]), 3)
            if len(row["pairs"]) > 2 else None,
        }
        print(json.dumps(qc, indent=2))
        print("\nQC NOTE: single-fold numbers are for gating only; NOT an aggregate. Do not interpret.")
        return

    # fleet mode: run all folds x seeds, then guarded aggregate
    n_folds = FOLDS[args.dataset]
    done = []
    for seed in args.seeds:
        for fold in range(n_folds):
            fp = out_dir / f"{args.dataset}_seed{seed}_fold{fold}.json"
            if fp.exists():
                done.append(fp); continue
            run_fold(audit_root, args.dataset, seed, fold, args.k_spec, args.n_perm, args.n_random, out_dir)
            done.append(fp)
    expected = len(args.seeds) * n_folds
    print(f"[e1] {args.dataset}: {len(done)}/{expected} fold-cells present "
          f"(aggregate handled by scripts/aggregate_e1_spectrum.py once BOTH datasets complete).")


if __name__ == "__main__":
    main()
