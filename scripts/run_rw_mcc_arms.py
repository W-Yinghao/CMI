"""Risk-Weighted MCC 3-arm continuation trainer (GPU, env icml). Per bundle: load the hash-verified ERM warm-up,
compute the source-LOSO excess-risk RW weights ONCE on the warm-up source (frozen), then fork 3 arms continuing 20
epochs from that SAME warm-up: A=ERM-continue / B=true RW-MCC / C=pairwise weight-permuted RW-MCC. lambda_RW=1.0.
Same dump blob as the MCC rounds (so the oracle / aggregators re-audit). Target X/Y evaluation only. Manuscript
FROZEN; only the project owner stops a scientific line.

  python -m scripts.run_rw_mcc_arms --bundle-index 0 --device cuda --out-dir results/cmi_trace_rw_mcc
"""
from __future__ import annotations
import argparse, copy, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from cmi.models.backbones import build_backbone
from tos_cmi.train.run_mcc_arms import enumerate_bundles, _warmup, _source_val_split, _bacc, _dump_arm
from tos_cmi.eeg.feature_dump import _forward_dump
from tos_cmi.train.mechanism_consistency import BalancedSubjectClassSampler, contrast_norm, effective_rank
from tos_cmi.train.risk_weighted_mcc import source_loso_excess_risk_weights, permute_weights, weight_hash, rw_mcc_loss

ARMS = ["A_erm_continue", "B_rw_true", "C_rw_wperm"]


def _continue_rw(warm_sd, Xtr, ytr, dtr, n_cls, X_shape, arm, weights, device, cont_epochs, lr, K, lam_rw, ramp, seed):
    bb = build_backbone("EEGNet", X_shape[1], X_shape[2], n_cls, device=device); bb.load_state_dict(copy.deepcopy(warm_sd))
    params = list(bb.parameters()); assert sum(p.numel() for p in params if p.requires_grad) == sum(p.numel() for p in params)
    tr_idx, va_idx = _source_val_split(dtr, ytr, seed=seed)
    samp = BalancedSubjectClassSampler(dtr[tr_idx], ytr[tr_idx], K=K,
                                       n_batches=max(1, len(tr_idx) // (len(np.unique(dtr)) * n_cls * K)), seed=seed)
    opt = torch.optim.AdamW(params, lr=lr)
    best = dict(val=-1.0, sd=None, ep=-1); diag = dict(rw_loss=[], ce_loss=[])
    for ep in range(cont_epochs):
        lam_t = 0.0 if arm == "A_erm_continue" else lam_rw * min(1.0, (ep + 1) / max(1, ramp))
        bb.train()
        for local in samp:
            gi = tr_idx[local]
            xb = torch.tensor(Xtr[gi], dtype=torch.float32).to(device)
            yb = torch.tensor(ytr[gi], dtype=torch.long).to(device); db = torch.tensor(dtr[gi], dtype=torch.long).to(device)
            logits, z = bb(xb); ce = F.cross_entropy(logits, yb)
            if arm == "A_erm_continue" or weights is None:
                loss = ce; rw_v = 0.0
            else:
                rw, _ = rw_mcc_loss(z, yb, db, weights); loss = ce + lam_t * rw; rw_v = float(rw.detach())
            opt.zero_grad(); loss.backward(); opt.step()
            diag["rw_loss"].append(rw_v); diag["ce_loss"].append(float(ce.detach()))
        val = _bacc(bb, Xtr[va_idx], ytr[va_idx], device)
        if val > best["val"]:
            best = dict(val=val, sd=copy.deepcopy(bb.state_dict()), ep=ep)
    bb.load_state_dict(best["sd"])
    with torch.no_grad():
        Zc = torch.tensor(_forward_dump(bb, Xtr[tr_idx][:2048], device)[1], dtype=torch.float32)
    diag.update(selected_epoch=best["ep"], source_val_bacc=best["val"], effective_rank=effective_rank(Zc),
                contrast_norm=contrast_norm(Zc, ytr[tr_idx][:2048], dtr[tr_idx][:2048]),
                mean_rw_loss=float(np.mean(diag["rw_loss"])) if diag["rw_loss"] else 0.0,
                mean_ce_loss=float(np.mean(diag["ce_loss"])))
    return bb, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-index", type=int); ap.add_argument("--list-bundles", action="store_true")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out-dir", default="results/cmi_trace_rw_mcc")
    ap.add_argument("--cache-dir", default="results/cmi_trace_mcc/warmup_cache"); ap.add_argument("--verify-warmup-from", default="results/cmi_trace_mcc")
    ap.add_argument("--warmup-epochs", type=int, default=300); ap.add_argument("--cont-epochs", type=int, default=20)
    ap.add_argument("--K", type=int, default=4); ap.add_argument("--lam-rw", type=float, default=1.0); ap.add_argument("--ramp", type=int, default=5)
    ap.add_argument("--cont-lr", type=float, default=1e-4)
    a = ap.parse_args()
    bundles = enumerate_bundles()
    if a.list_bundles:
        [print(f"{i}\t{ds}\tsub{s}\tseed{sd}") for i, (ds, s, sd) in enumerate(bundles)]; print(f"# {len(bundles)} bundles"); return
    ds, subj, seed = bundles[a.bundle_index]
    print(f"[rw-mcc] bundle {a.bundle_index} {ds} sub{subj} seed{seed}", flush=True)
    bb, Xtr, ytr, dtr, n_cls, Xte, yte, dataset, classes, meta_arr, warm_hash, X_shape = _warmup(
        ds, subj, seed, a.device, a.warmup_epochs, 64, a.cache_dir)
    if a.verify_warmup_from:
        prev = Path(a.verify_warmup_from) / f"{ds}_sub{subj}_seed{seed}.manifest.json"
        if prev.exists():
            assert json.loads(prev.read_text()).get("warmup_hash") == warm_hash, "warm-up hash mismatch"
    warm_sd = copy.deepcopy(bb.state_dict())
    # RW weights computed ONCE at the warm-up source (frozen); permuted control
    tr_idx, _ = _source_val_split(dtr, ytr, seed=seed)
    Za = _forward_dump(bb, Xtr[tr_idx], a.device)[1]
    wout = source_loso_excess_risk_weights(Za, ytr[tr_idx], dtr[tr_idx])
    subs, pairs, W = wout["subs"], wout["pairs"], wout["weights"]; Wp = permute_weights(W, subs, pairs, seed=seed)
    arm_weights = {"A_erm_continue": None, "B_rw_true": W, "C_rw_wperm": Wp}
    manifest = dict(dataset=ds, subject=str(subj), seed=seed, warmup_hash=warm_hash, lam_rw=a.lam_rw,
                    weight_status=wout["status"], weight_hash=weight_hash(W, subs, pairs), perm_weight_hash=weight_hash(Wp, subs, pairs),
                    positive_weight_fraction=wout["positive_weight_fraction"], effective_weight_support=wout["effective_weight_support"],
                    max_weight=wout["max_weight"], arms={})
    for arm in ARMS:
        b2, diag = _continue_rw(warm_sd, Xtr, ytr, dtr, n_cls, X_shape, arm, arm_weights[arm], a.device,
                                a.cont_epochs, a.cont_lr, a.K, a.lam_rw, a.ramp, seed)
        p = _dump_arm(b2, arm, ds, subj, seed, meta_arr, Xtr, ytr, Xte, yte, classes, a.device, warm_hash, a.lam_rw, diag, a.out_dir)
        tgt = _bacc(b2, Xte, yte, a.device)
        manifest["arms"][arm] = dict(dump=Path(p).name, target_bacc=tgt, selected_epoch=diag["selected_epoch"],
                                     source_val_bacc=diag["source_val_bacc"], eff_rank=diag["effective_rank"],
                                     contrast_norm=diag["contrast_norm"], mean_rw_loss=diag["mean_rw_loss"])
        print(f"  {arm}: target_bAcc={tgt:.4f} sel_ep={diag['selected_epoch']} effrank={diag['effective_rank']:.2f} "
              f"cnorm={diag['contrast_norm']:.3f} rw={diag['mean_rw_loss']:.4f}", flush=True)
    outd = Path(a.out_dir); (outd / f"{ds}_sub{subj}_seed{seed}.manifest.json").write_text(json.dumps(manifest, indent=2, default=float))
    (outd / f"{ds}_sub{subj}_seed{seed}.done").write_text(f"{warm_hash}\t{wout['status']}\n")
    print(f"[rw-mcc] cell done: {ds} sub{subj} seed{seed} weight_status={wout['status']}", flush=True)


if __name__ == "__main__":
    main()
