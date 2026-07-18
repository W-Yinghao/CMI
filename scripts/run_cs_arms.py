"""Cross-Session 5-arm continuation trainer (GPU, env icml). Per bundle from the SAME ERM warm-up: A=ERM-continue /
B=CS-RW-MCC / C=weight-permuted CS-RW-MCC / D=direct cross-session risk / E=permuted direct-risk. Weights = source
early->late instability, frozen at warm-up. lambda=1.0. Records the exact-gradient target alignment as a CO-DIAGNOSTIC
(target labels audit-only for that). Target X/Y otherwise evaluation only. Manuscript FROZEN.

  python -m scripts.run_cs_arms --bundle-index 0 --device cuda --out-dir results/cmi_trace_cs_arms
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
from tos_cmi.train.risk_weighted_mcc import rw_mcc_loss, permute_weights, source_loso_excess_risk_weights
from tos_cmi.train import cross_session_objective as CS

ARMS = ["A_erm_continue", "B_cs_rw_mcc", "C_cs_rw_perm", "D_cs_risk", "E_cs_risk_perm"]


def _continue(warm_sd, Xtr, ytr, dtr, sess_tr, n_cls, X_shape, arm, W, Wp, v, vp, device, cont_epochs, lr, K, lam, ramp, seed):
    bb = build_backbone("EEGNet", X_shape[1], X_shape[2], n_cls, device=device); bb.load_state_dict(copy.deepcopy(warm_sd))
    params = list(bb.parameters()); assert sum(p.numel() for p in params if p.requires_grad) == sum(p.numel() for p in params)
    tr_idx, va_idx = _source_val_split(dtr, ytr, seed=seed)
    samp = BalancedSubjectClassSampler(dtr[tr_idx], ytr[tr_idx], K=K, n_batches=max(1, len(tr_idx) // (len(np.unique(dtr)) * n_cls * K)), seed=seed)
    opt = torch.optim.AdamW(params, lr=lr)
    best = dict(val=-1.0, sd=None, ep=-1); diag = dict(aux=[], ce=[])
    for ep in range(cont_epochs):
        lam_t = 0.0 if arm == "A_erm_continue" else lam * min(1.0, (ep + 1) / max(1, ramp))
        bb.train()
        for local in samp:
            gi = tr_idx[local]
            xb = torch.tensor(Xtr[gi], dtype=torch.float32).to(device)
            yb = torch.tensor(ytr[gi], dtype=torch.long).to(device); db = torch.tensor(dtr[gi], dtype=torch.long).to(device)
            logits, z = bb(xb); ce = F.cross_entropy(logits, yb); aux_v = 0.0
            if arm == "A_erm_continue":
                loss = ce
            elif arm == "B_cs_rw_mcc":
                aux, _ = rw_mcc_loss(z, yb, db, W); loss = ce + lam_t * aux; aux_v = float(aux.detach())
            elif arm == "C_cs_rw_perm":
                aux, _ = rw_mcc_loss(z, yb, db, Wp); loss = ce + lam_t * aux; aux_v = float(aux.detach())
            elif arm == "D_cs_risk":
                aux = CS.direct_risk_loss(logits, ytr[gi], dtr[gi], sess_tr[gi], v, device); loss = ce + lam_t * aux; aux_v = float(aux.detach())
            else:  # E_cs_risk_perm
                aux = CS.direct_risk_loss(logits, ytr[gi], dtr[gi], sess_tr[gi], vp, device); loss = ce + lam_t * aux; aux_v = float(aux.detach())
            opt.zero_grad(); loss.backward(); opt.step()
            diag["aux"].append(aux_v); diag["ce"].append(float(ce.detach()))
        val = _bacc(bb, Xtr[va_idx], ytr[va_idx], device)
        if val > best["val"]:
            best = dict(val=val, sd=copy.deepcopy(bb.state_dict()), ep=ep)
    bb.load_state_dict(best["sd"])
    with torch.no_grad():
        Zc = torch.tensor(_forward_dump(bb, Xtr[tr_idx][:2048], device)[1], dtype=torch.float32)
    diag.update(selected_epoch=best["ep"], source_val_bacc=best["val"], effective_rank=effective_rank(Zc),
                contrast_norm=contrast_norm(Zc, ytr[tr_idx][:2048], dtr[tr_idx][:2048]),
                mean_aux_loss=float(np.mean(diag["aux"])) if diag["aux"] else 0.0, mean_ce_loss=float(np.mean(diag["ce"])))
    return bb, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-index", type=int); ap.add_argument("--list-bundles", action="store_true")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out-dir", default="results/cmi_trace_cs_arms")
    ap.add_argument("--cache-dir", default="results/cmi_trace_mcc/warmup_cache"); ap.add_argument("--verify-warmup-from", default="results/cmi_trace_mcc")
    ap.add_argument("--warmup-epochs", type=int, default=300); ap.add_argument("--cont-epochs", type=int, default=20)
    ap.add_argument("--K", type=int, default=4); ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--ramp", type=int, default=5)
    ap.add_argument("--cont-lr", type=float, default=1e-4)
    a = ap.parse_args()
    bundles = enumerate_bundles()
    if a.list_bundles:
        [print(f"{i}\t{ds}\tsub{s}\tseed{sd}") for i, (ds, s, sd) in enumerate(bundles)]; print(f"# {len(bundles)}"); return
    ds, subj, seed = bundles[a.bundle_index]
    print(f"[cs-arms] bundle {a.bundle_index} {ds} sub{subj} seed{seed}", flush=True)
    bb, Xtr, ytr, dtr, n_cls, Xte, yte, dataset, classes, meta_arr, warm_hash, X_shape = _warmup(
        ds, subj, seed, a.device, a.warmup_epochs, 64, a.cache_dir)
    if a.verify_warmup_from:
        prev = Path(a.verify_warmup_from) / f"{ds}_sub{subj}_seed{seed}.manifest.json"
        if prev.exists():
            assert json.loads(prev.read_text()).get("warmup_hash") == warm_hash, "warm-up hash mismatch"
    warm_sd = copy.deepcopy(bb.state_dict())
    tr_idx, _ = _source_val_split(dtr, ytr, seed=seed)
    sess_full = np.asarray(meta_arr["session_source"])
    Zs = _forward_dump(bb, Xtr[tr_idx], a.device)[1]
    csw = CS.cross_session_risk_weights(Zs, ytr[tr_idx], dtr[tr_idx], sess_full[tr_idx])
    subs, pairs, W = csw["subs"], csw["pairs"], csw["weights"]; Wp = permute_weights(W, subs, pairs, seed=seed)
    v = CS.per_trial_cs_weights(W, subs, pairs, csw["classes"]); vp = CS.per_trial_cs_weights(Wp, subs, pairs, csw["classes"])
    # CO-DIAGNOSTIC: exact-gradient target alignment (target labels audit-only, not used in any arm)
    try:
        sess_t = np.asarray(meta_arr["session_target"]); _, later_t = CS._early_late(sess_t); tf = np.isin(sess_t, list(later_t))
        if tf.sum() == 0:
            tf = np.ones(len(yte), bool)
        Xs, ys, dsub = Xtr[tr_idx], ytr[tr_idx], dtr[tr_idx]
        gt = CS.task_gradient(bb, Xte, yte, a.device, mask=tf)
        align = dict(cs_rw=CS.cos(CS.exact_weighted_mcc_gradient(bb, Xs, ys, dsub, W, a.device), gt),
                     cs_risk=CS.cos(CS.weighted_late_task_gradient(bb, Xs, ys, dsub, csw["is_late"], W, a.device), gt),
                     loso=CS.cos(CS.exact_weighted_mcc_gradient(bb, Xs, ys, dsub, source_loso_excess_risk_weights(Zs, ys, dsub)["weights"], a.device), gt),
                     task=CS.cos(CS.task_gradient(bb, Xs, ys, a.device), gt))
    except Exception as e:
        align = dict(error=str(e)[:80])
    manifest = dict(dataset=ds, subject=str(subj), seed=seed, warmup_hash=warm_hash, lam=a.lam, weight_status=csw["status"],
                    effective_weight_support=csw["effective_weight_support"], mean_source_late_risk=float(np.mean(list(csw["r"].values()))),
                    gradient_alignment_diagnostic=align, arms={})
    for arm in ARMS:
        b2, diag = _continue(warm_sd, Xtr, ytr, dtr, sess_full, n_cls, X_shape, arm, W, Wp, v, vp, a.device,
                             a.cont_epochs, a.cont_lr, a.K, a.lam, a.ramp, seed)
        p = _dump_arm(b2, arm, ds, subj, seed, meta_arr, Xtr, ytr, Xte, yte, classes, a.device, warm_hash, a.lam, diag, a.out_dir)
        tgt = _bacc(b2, Xte, yte, a.device)
        manifest["arms"][arm] = dict(dump=Path(p).name, target_bacc=tgt, selected_epoch=diag["selected_epoch"],
                                     source_val_bacc=diag["source_val_bacc"], eff_rank=diag["effective_rank"],
                                     contrast_norm=diag["contrast_norm"], mean_aux_loss=diag["mean_aux_loss"])
        print(f"  {arm}: tgt_bAcc={tgt:.4f} sel_ep={diag['selected_epoch']} effrank={diag['effective_rank']:.2f} aux={diag['mean_aux_loss']:.4f}", flush=True)
    outd = Path(a.out_dir); (outd / f"{ds}_sub{subj}_seed{seed}.manifest.json").write_text(json.dumps(manifest, indent=2, default=float))
    (outd / f"{ds}_sub{subj}_seed{seed}.done").write_text(f"{warm_hash}\t{csw['status']}\n")
    print(f"[cs-arms] cell done: {ds} sub{subj} seed{seed} align={align}", flush=True)


if __name__ == "__main__":
    main()
