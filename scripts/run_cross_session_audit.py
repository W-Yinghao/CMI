"""Cross-Session Objective Audit runner (GPU, env icml; NO training). Per bundle at the hash-verified ERM warm-up:
build source-only cross-session weights, compute the encoder-param gradients of CS-RW-MCC / CS-Risk / LOSO-RW /
their permuted controls / the ordinary source task, and a NON-DEPLOYABLE target FUTURE-SESSION task gradient
(target labels used ONLY to score alignment). Primary endpoint = A_o = cos(g_o, g_target); secondary = normalized
one-step target loss. Writes one cell json + .done. Manuscript FROZEN; only the project owner stops a scientific
line.

  python -m scripts.run_cross_session_audit --bundle-index 0 --device cuda --out-dir results/cmi_trace_cross_session_audit
"""
from __future__ import annotations
import argparse, copy, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from tos_cmi.train.run_mcc_arms import enumerate_bundles, _warmup, _source_val_split
from tos_cmi.eeg.feature_dump import _forward_dump
from tos_cmi.train import cross_session_objective as CS
from tos_cmi.train.risk_weighted_mcc import source_loso_excess_risk_weights, permute_weights


def _one_step_target_loss(bb, base_sd, g, Xt, yt, mask, device, alpha=0.1, bs=512):
    """Loss on the target FUTURE session after theta' = theta - alpha*g/||g||, minus the base loss (negative = the
    source objective's direction reduces target loss). Restores the model afterward."""
    gn = g / (np.linalg.norm(g) + 1e-12); idx = np.where(mask)[0]

    def _ce():
        bb.eval()
        with torch.no_grad():
            tot = 0.0
            for i in range(0, len(idx), bs):
                j = idx[i:i + bs]; lg = bb(torch.tensor(Xt[j], dtype=torch.float32).to(device))[0]
                tot += float(F.cross_entropy(lg, torch.tensor(np.asarray(yt)[j], dtype=torch.long, device=device), reduction="sum"))
        return tot / len(idx)
    base = _ce()
    off = 0
    with torch.no_grad():
        for p in bb.parameters():
            n = p.numel(); p.add_(torch.tensor(-alpha * gn[off:off + n].reshape(p.shape), dtype=p.dtype, device=device)); off += n
    stepped = _ce(); bb.load_state_dict(base_sd)
    return stepped - base


def _audit(ds, subj, seed, device, cache_dir, verify_dir):
    bb, Xtr, ytr, dtr, n_cls, Xte, yte, dataset, classes, meta_arr, warm_hash, X_shape = _warmup(
        ds, subj, seed, device, 300, 64, cache_dir)
    if verify_dir:
        prev = Path(verify_dir) / f"{ds}_sub{subj}_seed{seed}.manifest.json"
        if prev.exists():
            assert json.loads(prev.read_text()).get("warmup_hash") == warm_hash, "warm-up hash mismatch"
    base_sd = copy.deepcopy(bb.state_dict())
    tr_idx, _ = _source_val_split(dtr, ytr, seed=seed)
    Xs, ys, dsub = Xtr[tr_idx], ytr[tr_idx], dtr[tr_idx]
    sess_s = np.asarray(meta_arr["session_source"])[tr_idx]
    Zs = _forward_dump(bb, Xs, device)[1]
    # weights (source-only)
    csw = CS.cross_session_risk_weights(Zs, ys, dsub, sess_s)
    W_cs, subs, pairs = csw["weights"], csw["subs"], csw["pairs"]
    W_cs_perm = permute_weights(W_cs, subs, pairs, seed=seed)
    W_loso = source_loso_excess_risk_weights(Zs, ys, dsub)["weights"]
    # target FUTURE session mask (target labels used ONLY below for alignment; never in any weight/objective)
    sess_t = np.asarray(meta_arr["session_target"]); _, later_t = CS._early_late(sess_t); tgt_future = np.isin(sess_t, list(later_t))
    if tgt_future.sum() == 0:
        tgt_future = np.ones(len(yte), bool)
    # gradients (w.r.t. encoder)
    g_cs_rw = CS.exact_weighted_mcc_gradient(bb, Xs, ys, dsub, W_cs, device)
    g_cs_rw_perm = CS.exact_weighted_mcc_gradient(bb, Xs, ys, dsub, W_cs_perm, device)
    g_cs_risk = CS.weighted_late_task_gradient(bb, Xs, ys, dsub, csw["is_late"], W_cs, device)
    g_cs_risk_perm = CS.weighted_late_task_gradient(bb, Xs, ys, dsub, csw["is_late"], W_cs_perm, device)
    g_loso = CS.exact_weighted_mcc_gradient(bb, Xs, ys, dsub, W_loso, device)
    g_task = CS.task_gradient(bb, Xs, ys, device)
    g_target = CS.task_gradient(bb, Xte, yte, device, mask=tgt_future)     # AUDIT-ONLY
    A = {k: CS.cos(g, g_target) for k, g in dict(cs_rw=g_cs_rw, cs_rw_perm=g_cs_rw_perm, cs_risk=g_cs_risk,
                                                 cs_risk_perm=g_cs_risk_perm, loso=g_loso, task=g_task).items()}
    onestep = dict(cs_rw=_one_step_target_loss(bb, base_sd, g_cs_rw, Xte, yte, tgt_future, device),
                   cs_rw_perm=_one_step_target_loss(bb, base_sd, g_cs_rw_perm, Xte, yte, tgt_future, device),
                   cs_risk=_one_step_target_loss(bb, base_sd, g_cs_risk, Xte, yte, tgt_future, device),
                   cs_risk_perm=_one_step_target_loss(bb, base_sd, g_cs_risk_perm, Xte, yte, tgt_future, device))
    return dict(dataset=ds, subject=str(subj), seed=int(seed), warmup_hash=warm_hash, weight_status=csw["status"],
                effective_weight_support=csw["effective_weight_support"], max_weight=csw["max_weight"],
                A=A, dA_cs_rw=A["cs_rw"] - A["cs_rw_perm"], dA_cs_risk=A["cs_risk"] - A["cs_risk_perm"],
                dA_cs_rw_vs_loso=A["cs_rw"] - A["loso"], onestep_target_loss=onestep,
                dOne_cs_rw=onestep["cs_rw"] - onestep["cs_rw_perm"], dOne_cs_risk=onestep["cs_risk"] - onestep["cs_risk_perm"],
                mean_source_late_risk=float(np.mean(list(csw["r"].values()))), n_subjects=len(subs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-index", type=int); ap.add_argument("--list-bundles", action="store_true")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out-dir", default="results/cmi_trace_cross_session_audit")
    ap.add_argument("--cache-dir", default="results/cmi_trace_mcc/warmup_cache"); ap.add_argument("--verify-from", default="results/cmi_trace_mcc")
    a = ap.parse_args()
    bundles = enumerate_bundles()
    if a.list_bundles:
        [print(f"{i}\t{ds}\tsub{s}\tseed{sd}") for i, (ds, s, sd) in enumerate(bundles)]; print(f"# {len(bundles)}"); return
    ds, subj, seed = bundles[a.bundle_index]
    print(f"[cs-audit] bundle {a.bundle_index} {ds} sub{subj} seed{seed}", flush=True)
    row = _audit(ds, subj, seed, a.device, a.cache_dir, a.verify_from)
    outd = Path(a.out_dir); (outd / "cells").mkdir(parents=True, exist_ok=True)
    stem = f"cell_{a.bundle_index:03d}_{ds}_sub{subj}_seed{seed}"
    (outd / "cells" / f"{stem}.json").write_text(json.dumps(row, indent=2, default=float))
    (outd / "cells" / f"{stem}.done").write_text(row["weight_status"] + "\n")
    print(f"  A: cs_rw={row['A']['cs_rw']:+.3f} cs_risk={row['A']['cs_risk']:+.3f} loso={row['A']['loso']:+.3f} task={row['A']['task']:+.3f} | "
          f"dA_cs_rw={row['dA_cs_rw']:+.3f} dA_cs_risk={row['dA_cs_risk']:+.3f} dA_vs_loso={row['dA_cs_rw_vs_loso']:+.3f} | "
          f"1step cs_rw={row['dOne_cs_rw']:+.4f} cs_risk={row['dOne_cs_risk']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
