"""Phase 2.0 -- frozen-feature dump. Train a backbone on one LOSO fold (source = all non-target
subjects) for a given method (erm | lpc_prior, lam), then dump the FROZEN latent Z + logits +
labels + domain/subject/session for the source (whole; the score-Fisher diagnostic does its own
group-aware S_sel/S_gate split via cluster_id=subject) and the held-out target (report only).

Reuses the validated cmi/ training stack (NO trainer modification): cmi.train.trainer.train_model
+ cmi.models.backbones.build_backbone + cmi.data.moabb_data. The target subject never enters the
selector / gate / lambda / calibration -- it is dumped only for the final report."""
from __future__ import annotations
import json
import subprocess
from pathlib import Path
import numpy as np
import torch

from cmi.paths import configure_offline_moabb
from cmi.data.moabb_data import load, domain_labels
from cmi.models.backbones import build_backbone
from cmi.train.trainer import train_model


def _git_sha(path):
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _remap(d):
    """Contiguous domain ids 0..n-1 (train_model expects dense domain labels)."""
    u = np.unique(d); m = {v: i for i, v in enumerate(u)}
    return np.array([m[v] for v in d], dtype=np.int64), len(u)


@torch.no_grad()
def _forward_dump(bb, X, device, bs=256):
    """One eval pass -> (logits [N,C] float32, Z [N,z_dim] float32)."""
    bb.eval(); lg, zs = [], []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i + bs], dtype=torch.float32).to(device)
        logits, z = bb(xb)
        lg.append(logits.detach().cpu().numpy()); zs.append(z.detach().cpu().numpy())
    return np.concatenate(lg).astype("float32"), np.concatenate(zs).astype("float32")


def dump_fold(dataset, target_subject, method, lam, seed, out_path, *, backbone="TSMNet",
              epochs=300, bs=64, warmup=40, device="cuda", tmin=0.5, tmax=3.5, resample=128,
              domain_mode="subject"):
    """Train on source (LOSO leaving target_subject out), dump frozen Z/logits/meta to out_path.npz."""
    configure_offline_moabb()
    X, y, meta, classes = load(dataset, tmin=tmin, tmax=tmax, resample=resample)
    n_cls = len(classes)
    dom_all, _ = domain_labels(meta, domain_mode)
    subj = meta["subject"].to_numpy(); sess = meta["session"].astype(str).to_numpy()

    te = (subj == target_subject); tr = ~te
    if te.sum() == 0:
        raise ValueError("target_subject %r not in %s" % (target_subject, dataset))
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    dtr, n_dom = _remap(dom_all[tr])

    torch.manual_seed(seed); np.random.seed(seed)
    bb = build_backbone(backbone, X.shape[1], X.shape[2], n_cls, device=device)
    bb, _post, diag = train_model(bb, Xtr, ytr, dtr, n_cls, method=method, lam=lam,
                                  epochs=epochs, bs=bs, warmup=warmup, device=device, seed=seed)

    lg_src, Z_src = _forward_dump(bb, Xtr, device)
    lg_tgt, Z_tgt = _forward_dump(bb, Xte, device)
    root = Path(__file__).resolve().parents[2]
    blob = dict(
        Z_source=Z_src, Z_target=Z_tgt, logits_source=lg_src, logits_target=lg_tgt,
        y_source=ytr.astype("int64"), y_target=yte.astype("int64"),
        domain_source=dtr, subject_source=subj[tr].astype("int64"),
        subject_target=subj[te].astype("int64"), session_source=sess[tr], session_target=sess[te],
        dataset=dataset, backbone=backbone, method=method, lam=np.float64(lam),
        seed=np.int64(seed), target_subject=np.int64(target_subject), n_cls=np.int64(n_cls),
        n_dom_source=np.int64(n_dom), z_dim=np.int64(Z_src.shape[1]), classes=np.array(classes),
        tmin=np.float64(tmin), tmax=np.float64(tmax), resample=np.int64(resample),
        domain_mode=domain_mode, git_sha=_git_sha(root), tsmnet_sha=_git_sha(root / "repos/TSMNet"),
        train_diag=json.dumps({k: float(v) for k, v in (diag or {}).items()
                               if isinstance(v, (int, float))}))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **blob)
    return out_path
