"""MCC estimator audit runner (GPU, env icml). Per (dataset, target_subject, seed) bundle: load the SAME (hash-
verified) ERM warm-up as the MCC rounds, take its continuation-train source split, and compute the EXACT
full-source MCC gradient (two-pass, BN frozen) vs the K=4 / K=16 episodic estimators + shuffled controls, plus the
per-cell diagnostics (A_K / B_K / SNR_K, one-step prototype-WSCI, true-vs-shuffle). NO training. Writes one cell
json + .done. Only the project owner may stop a scientific line; manuscript FROZEN.

  python -m scripts.run_mcc_estimator_audit --bundle-index 0 --device cuda --out-dir results/cmi_trace_mcc_estimator_audit
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from tos_cmi.train.run_mcc_arms import enumerate_bundles, _warmup, _source_val_split
from tos_cmi.train.mechanism_consistency import class_pairs
from tos_cmi.eval import mcc_estimator_audit as MEA


def _audit_cell(bb, Xa, ya, da, device, R, K_large, seed):
    subs = sorted(np.unique(da).tolist()); classes = sorted(np.unique(ya).tolist()); pairs = class_pairs(classes)
    # feasible large-K: min cell count (so every subject-class cell can supply K without replacement blowing up)
    min_cell = min(int(((da == s) & (ya == c)).sum()) for s in subs for c in classes)
    K16 = min(K_large, max(4, min_cell)); k16_reason = "ok" if K16 == K_large else f"capped_min_cell={min_cell}"
    rng = np.random.default_rng(seed)
    g_full, means, gmu, L_full = MEA.exact_population_gradient(bb, Xa, ya, da, device, shuffle=False, rng=rng)
    g_full_sh, _, _, _ = MEA.exact_population_gradient(bb, Xa, ya, da, device, shuffle=True, rng=np.random.default_rng(seed + 1))
    g4 = MEA.episodic_theta_gradients(bb, Xa, ya, da, device, K=4, R=R, seed=seed, shuffle=False)
    g16 = MEA.episodic_theta_gradients(bb, Xa, ya, da, device, K=K16, R=R, seed=seed, shuffle=False)
    d4 = MEA.gradient_diagnostics(g_full, g4); d16 = MEA.gradient_diagnostics(g_full, g16)
    dw_full = MEA.one_step_prototype_wsci(means, gmu, subs, classes, pairs)
    dw_k4 = MEA.episodic_prototype_one_step_wsci(bb, Xa, ya, da, device, 4, R, seed, means, subs, classes, pairs)
    dw_k16 = MEA.episodic_prototype_one_step_wsci(bb, Xa, ya, da, device, K16, R, seed, means, subs, classes, pairs)
    return dict(A_4=d4["A_K"], A_16=d16["A_K"], B_4=d4["B_K"], B_16=d16["B_K"], SNR_4=d4["SNR_K"], SNR_16=d16["SNR_K"],
                full_grad_norm=d4["full_grad_norm"], mean_grad_norm_4=d4["mean_grad_norm"], mean_grad_norm_16=d16["mean_grad_norm"],
                dw_full=dw_full, dw_k4=dw_k4, dw_k16=dw_k16, K16=K16, k16_reason=k16_reason, L_full=L_full,
                true_vs_shuffle_reldist=float(np.linalg.norm(g_full - g_full_sh) / (np.linalg.norm(g_full) + 1e-12)),
                true_vs_shuffle_cos=MEA._cos(g_full, g_full_sh), n_subjects=len(subs), n_source=int(len(ya)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-index", type=int); ap.add_argument("--list-bundles", action="store_true")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out-dir", default="results/cmi_trace_mcc_estimator_audit")
    ap.add_argument("--cache-dir", default="results/cmi_trace_mcc/warmup_cache")
    ap.add_argument("--verify-warmup-from", default="results/cmi_trace_mcc"); ap.add_argument("--warmup-epochs", type=int, default=300)
    ap.add_argument("--R", type=int, default=64); ap.add_argument("--K-large", type=int, default=16)
    a = ap.parse_args()
    bundles = enumerate_bundles()
    if a.list_bundles:
        for i, (ds, s, sd) in enumerate(bundles):
            print(f"{i}\t{ds}\tsub{s}\tseed{sd}")
        print(f"# {len(bundles)} cells"); return
    ds, subj, seed = bundles[a.bundle_index]
    print(f"[mcc-audit] bundle {a.bundle_index} {ds} sub{subj} seed{seed} device={a.device} R={a.R}", flush=True)
    bb, Xtr, ytr, dtr, n_cls, Xte, yte, dataset, classes, meta_arr, warm_hash, X_shape = _warmup(
        ds, subj, seed, a.device, a.warmup_epochs, 64, a.cache_dir)
    if a.verify_warmup_from:
        prev = Path(a.verify_warmup_from) / f"{ds}_sub{subj}_seed{seed}.manifest.json"
        if prev.exists():
            pw = json.loads(prev.read_text()).get("warmup_hash")
            assert pw == warm_hash, f"warm-up hash mismatch vs MCC round: {warm_hash} != {pw}"
            print(f"[mcc-audit] warm-up hash verified == MCC round ({pw})", flush=True)
    tr_idx, _ = _source_val_split(dtr, ytr, seed=seed)               # SAME continuation-train source as MCC training
    res = _audit_cell(bb, Xtr[tr_idx], ytr[tr_idx], dtr[tr_idx], a.device, a.R, a.K_large, seed)
    row = dict(dataset=ds, subject=str(subj), seed=int(seed), warmup_hash=warm_hash, **res)
    outd = Path(a.out_dir); outd.mkdir(parents=True, exist_ok=True)
    stem = f"cell_{a.bundle_index:03d}_{ds}_sub{subj}_seed{seed}"
    (outd / f"{stem}.json").write_text(json.dumps(row, indent=2, default=float))
    (outd / f"{stem}.done").write_text(f"{warm_hash}\n")
    print(f"  A_4={res['A_4']:+.3f} A_16={res['A_16']:+.3f} (A16-A4={res['A_16']-res['A_4']:+.3f}) SNR_4={res['SNR_4']:.3f} "
          f"SNR_16={res['SNR_16']:.3f} dw_full/k4={res['dw_full']:+.4f}/{res['dw_k4']:+.4f} "
          f"trueVSshuf cos={res['true_vs_shuffle_cos']:+.3f} K16={res['K16']}({res['k16_reason']})", flush=True)


if __name__ == "__main__":
    main()
