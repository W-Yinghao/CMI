"""Cross-Session FULL-STRENGTH oracle-ceiling trainer (Track A; GPU, env icml). Same 5 arms and the SAME hash-verified
ERM warm-up per bundle as run_cs_arms, but three execution changes ONLY, per the PM's fixed protocol:
  lambda = 1.0 from epoch 0   (NO ramp)
  train exactly 20 epochs
  NO source-val checkpoint rollback  (the DUMP + primary state is the FINAL, epoch-20 model)
Everything else identical: same warm-up hash, batch schedule, matched dropout, optimizer/LR, source-only weights,
EEGNet, both datasets, all LOSO subjects, seeds 0/1/2. Per-epoch metrics are recorded at {0,1,2,5,10,15,20}
(target bAcc included -> this is an ORACLE-CEILING diagnostic; target labels enter the trajectory ONLY, never the
training objective or any weight). Primary endpoint = epoch 20; trajectory = mechanism diagnostic; any max-over-e>=5
gain is a NON-DEPLOYABLE target-epoch oracle upper bound. Manuscript FROZEN.

  python -m scripts.run_cs_fullstrength --bundle-index 0 --device cuda --out-dir results/cmi_trace_cs_fullstrength
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
from tos_cmi.train.risk_weighted_mcc import rw_mcc_loss, permute_weights
from tos_cmi.train import cross_session_objective as CS

ARMS = ["A_erm_continue", "B_cs_rw_mcc", "C_cs_rw_perm", "D_cs_risk", "E_cs_risk_perm"]
SAVE_EPOCHS = [0, 1, 2, 5, 10, 15, 20]


def _task_margin(bb, X, y, device):
    """Mean (correct-class logit - max other-class logit) over X."""
    logits = _forward_dump(bb, X, device)[0]; y = np.asarray(y).astype(int)
    corr = logits[np.arange(len(y)), y]; tmp = logits.copy(); tmp[np.arange(len(y)), y] = -1e9
    return float((corr - tmp.max(1)).mean())


def _source_late_risk(bb, X, y, is_late, device):
    """Mean CE on source LATE-session trials (unweighted diagnostic of how late-session task loss evolves)."""
    idx = np.where(np.asarray(is_late))[0]
    if idx.size == 0:
        return float("nan")
    logits = _forward_dump(bb, X[idx], device)[0]
    return float(F.cross_entropy(torch.tensor(logits), torch.tensor(np.asarray(y)[idx].astype(int)), reduction="mean"))


def _epoch_metrics(bb, Xtr, ytr, dtr, tr_idx, va_idx, is_late_full, Xte, yte, device, aux_val):
    zc = torch.tensor(_forward_dump(bb, Xtr[tr_idx][:2048], device)[1], dtype=torch.float32)
    return dict(source_val_bacc=_bacc(bb, Xtr[va_idx], ytr[va_idx], device),
                aux_loss=float(aux_val),
                source_late_risk=_source_late_risk(bb, Xtr[tr_idx], ytr[tr_idx], is_late_full[tr_idx], device),
                target_bacc=_bacc(bb, Xte, yte, device),
                eff_rank=float(effective_rank(zc)),
                contrast_norm=float(contrast_norm(zc, ytr[tr_idx][:2048], dtr[tr_idx][:2048])),
                task_margin=_task_margin(bb, Xtr[va_idx], ytr[va_idx], device))


def _continue_fs(warm_sd, Xtr, ytr, dtr, is_late_full, n_cls, X_shape, arm, W, Wp, v, vp, device, cont_epochs, lr, K, lam, seed, Xte, yte):
    bb = build_backbone("EEGNet", X_shape[1], X_shape[2], n_cls, device=device); bb.load_state_dict(copy.deepcopy(warm_sd))
    params = list(bb.parameters()); assert sum(p.numel() for p in params if p.requires_grad) == sum(p.numel() for p in params)
    tr_idx, va_idx = _source_val_split(dtr, ytr, seed=seed)
    samp = BalancedSubjectClassSampler(dtr[tr_idx], ytr[tr_idx], K=K, n_batches=max(1, len(tr_idx) // (len(np.unique(dtr)) * n_cls * K)), seed=seed)
    opt = torch.optim.AdamW(params, lr=lr)
    torch.manual_seed(seed)                       # MATCHED DROPOUT across arms (see run_cs_arms)
    if str(device) != "cpu":
        torch.cuda.manual_seed_all(seed)
    per_epoch = {}
    lam_t = 0.0 if arm == "A_erm_continue" else lam    # FULL STRENGTH from epoch 0, NO ramp
    bb.eval(); per_epoch[0] = _epoch_metrics(bb, Xtr, ytr, dtr, tr_idx, va_idx, is_late_full, Xte, yte, device, 0.0)
    for ep in range(cont_epochs):
        bb.train(); aux_run = []
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
                aux = CS.direct_risk_loss(logits, ytr[gi], dtr[gi], is_late_full[gi], v, device); loss = ce + lam_t * aux; aux_v = float(aux.detach())
            else:  # E_cs_risk_perm
                aux = CS.direct_risk_loss(logits, ytr[gi], dtr[gi], is_late_full[gi], vp, device); loss = ce + lam_t * aux; aux_v = float(aux.detach())
            opt.zero_grad(); loss.backward(); opt.step(); aux_run.append(aux_v)
        e1 = ep + 1
        if e1 in SAVE_EPOCHS:
            bb.eval(); per_epoch[e1] = _epoch_metrics(bb, Xtr, ytr, dtr, tr_idx, va_idx, is_late_full, Xte, yte, device, float(np.mean(aux_run)) if aux_run else 0.0)
    bb.eval()                                     # NO rollback: final (epoch-20) model is the primary/dump state
    diag = dict(per_epoch=per_epoch, selected_epoch=cont_epochs, source_val_bacc=per_epoch[cont_epochs]["source_val_bacc"],
                effective_rank=per_epoch[cont_epochs]["eff_rank"], contrast_norm=per_epoch[cont_epochs]["contrast_norm"],
                mean_aux_loss=per_epoch[cont_epochs]["aux_loss"])
    return bb, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-index", type=int); ap.add_argument("--list-bundles", action="store_true")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out-dir", default="results/cmi_trace_cs_fullstrength")
    ap.add_argument("--cache-dir", default="results/cmi_trace_mcc/warmup_cache"); ap.add_argument("--verify-warmup-from", default="results/cmi_trace_mcc")
    ap.add_argument("--warmup-epochs", type=int, default=300); ap.add_argument("--cont-epochs", type=int, default=20)
    ap.add_argument("--K", type=int, default=4); ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--cont-lr", type=float, default=1e-4)
    a = ap.parse_args()
    bundles = enumerate_bundles()
    if a.list_bundles:
        [print(f"{i}\t{ds}\tsub{s}\tseed{sd}") for i, (ds, s, sd) in enumerate(bundles)]; print(f"# {len(bundles)}"); return
    ds, subj, seed = bundles[a.bundle_index]
    print(f"[cs-fs] bundle {a.bundle_index} {ds} sub{subj} seed{seed}", flush=True)
    bb, Xtr, ytr, dtr, n_cls, Xte, yte, dataset, classes, meta_arr, warm_hash, X_shape = _warmup(
        ds, subj, seed, a.device, a.warmup_epochs, 64, a.cache_dir)
    if a.verify_warmup_from:
        prev = Path(a.verify_warmup_from) / f"{ds}_sub{subj}_seed{seed}.manifest.json"
        if prev.exists():
            assert json.loads(prev.read_text()).get("warmup_hash") == warm_hash, "warm-up hash mismatch"
    warm_sd = copy.deepcopy(bb.state_dict())
    tr_idx, _ = _source_val_split(dtr, ytr, seed=seed)
    sess_full = np.asarray(meta_arr["session_source"])
    is_late_full = np.isin(sess_full, list(CS._early_late(sess_full)[1]))
    Zs = _forward_dump(bb, Xtr[tr_idx], a.device)[1]
    csw = CS.cross_session_risk_weights(Zs, ytr[tr_idx], dtr[tr_idx], sess_full[tr_idx])
    subs, pairs, W = csw["subs"], csw["pairs"], csw["weights"]; Wp = permute_weights(W, subs, pairs, seed=seed)
    v = CS.per_trial_cs_weights(W, subs, pairs, csw["classes"]); vp = CS.per_trial_cs_weights(Wp, subs, pairs, csw["classes"])
    manifest = dict(dataset=ds, subject=str(subj), seed=seed, warmup_hash=warm_hash, lam=a.lam, protocol="full_strength_no_ramp_no_rollback",
                    weight_status=csw["status"], effective_weight_support=csw["effective_weight_support"],
                    mean_source_late_risk_weight=float(np.mean(list(csw["r"].values()))), save_epochs=SAVE_EPOCHS, arms={})
    for arm in ARMS:
        b2, diag = _continue_fs(warm_sd, Xtr, ytr, dtr, is_late_full, n_cls, X_shape, arm, W, Wp, v, vp, a.device,
                                a.cont_epochs, a.cont_lr, a.K, a.lam, seed, Xte, yte)
        p = _dump_arm(b2, arm, ds, subj, seed, meta_arr, Xtr, ytr, Xte, yte, classes, a.device, warm_hash, a.lam, diag, a.out_dir)
        manifest["arms"][arm] = dict(dump=Path(p).name, per_epoch=diag["per_epoch"], target_bacc_final=diag["per_epoch"][a.cont_epochs]["target_bacc"],
                                     source_val_bacc_final=diag["per_epoch"][a.cont_epochs]["source_val_bacc"])
        tj = " ".join(f"e{e}:{diag['per_epoch'][e]['target_bacc']:.3f}" for e in SAVE_EPOCHS if e in diag['per_epoch'])
        print(f"  {arm}: tgt_traj[{tj}] aux20={diag['per_epoch'][a.cont_epochs]['aux_loss']:.4f}", flush=True)
    outd = Path(a.out_dir); (outd / f"{ds}_sub{subj}_seed{seed}.manifest.json").write_text(json.dumps(manifest, indent=2, default=float))
    (outd / f"{ds}_sub{subj}_seed{seed}.done").write_text(f"{warm_hash}\t{csw['status']}\n")
    print(f"[cs-fs] cell done: {ds} sub{subj} seed{seed}", flush=True)


if __name__ == "__main__":
    main()
