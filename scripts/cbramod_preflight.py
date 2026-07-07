#!/usr/bin/env python
"""FSR Phase 8A — CBraMod preflight (backup / control substrate; PM). Single-stage EEG foundation encoder
(criss-cross channel-time transformer). Confirms the LOCAL pretrained CBraMod backbone loads and produces a
frozen pooled window embedding on dummy + REAL SHU-MI motor-imagery EEG. No tokenizer (so no VQ-collapse risk),
no target labels, no training. Tries CPU (CBraMod forward has no hardcoded .cuda()); --device cuda to force GPU.
Writes results/fsr_cbramod_preflight/cbramod_preflight.json."""
import argparse, glob, json, os, sys, time
from pathlib import Path
import numpy as np
import scipy.io as sio
from scipy.signal import resample

CBRAMOD = "/home/infres/yinwang/eeg2025/CBraMod"
CKPT = "/home/infres/yinwang/eeg2025/NIPS/Cbramod_pretrained_weights.pth"
SHU = "/projects/EEG-foundation-model/SHU-MI-cbramod/mat"
OUT = Path("results/fsr_cbramod_preflight")


def zscore(x):
    return (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, CBRAMOD)
    import torch
    torch.set_num_threads(4)
    from models.cbramod import CBraMod
    dev = torch.device(args.device)

    rep = dict(device=args.device, torch=torch.__version__,
               weights=dict(path=CKPT, mb=round(os.path.getsize(CKPT) / 1e6, 1)))
    bb = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
    miss, unexp = bb.load_state_dict(sd, strict=False)
    bb.proj_out = torch.nn.Identity()
    bb.eval().to(dev)
    rep["load"] = dict(n_ckpt_tensors=len(sd), missing_keys=len(miss), unexpected_keys=len(unexp),
                       missing_sample=list(miss)[:6], unexpected_sample=list(unexp)[:6],
                       n_params=sum(p.numel() for p in bb.parameters()))

    # ---- dummy forward ----
    xd = torch.randn(2, 19, 4, 200, device=dev)
    with torch.no_grad():
        t0 = time.time(); fd = bb(xd); dtd = time.time() - t0
    rep["dummy_forward"] = dict(input_shape=list(xd.shape), feats_shape=list(fd.shape),
                                finite=bool(np.isfinite(fd.detach().cpu().numpy()).all()), sec=round(dtd, 3))

    # ---- REAL SHU-MI forward ----
    f = sorted(glob.glob(f"{SHU}/*.mat"))[0]
    m = sio.loadmat(f); data = m["data"].astype(np.float32)      # (trials,32,1000) @250Hz 4s
    B = min(8, data.shape[0])
    x = resample(data[:B], 800, axis=-1)[:, :19, :]              # 250->200Hz, 19 ch
    x = zscore(x).reshape(B, 19, 4, 200)
    xt = torch.tensor(x, dtype=torch.float32, device=dev)
    with torch.no_grad():
        t0 = time.time(); fr = bb(xt); dtr = time.time() - t0
        pooled = fr.mean(dim=(1, 2))                             # (B,200) pooled window embedding
    fr_np = fr.detach().cpu().numpy(); pl = pooled.detach().cpu().numpy()
    pdist = float(np.mean([np.linalg.norm(pl[i] - pl[j]) for i in range(B) for j in range(i + 1, B)])) if B > 1 else None
    rep["real_forward_shu"] = dict(source=f.split("/")[-1], n_trials=B, input_shape=list(xt.shape),
                                   feats_shape=list(fr.shape), pooled_shape=list(pooled.shape),
                                   finite=bool(np.isfinite(fr_np).all()),
                                   pooled_mean_pairwise_L2=round(pdist, 4) if pdist is not None else None,
                                   pooled_nondegenerate=bool(pdist is None or pdist > 1e-6), sec=round(dtr, 3))

    load_ok = rep["load"]["missing_keys"] == 0 and rep["load"]["unexpected_keys"] == 0
    real_ok = rep["real_forward_shu"]["finite"] and rep["real_forward_shu"]["pooled_nondegenerate"]
    rep["preflight_gate"] = dict(
        backbone_loads_clean=bool(load_ok), real_eeg_forward_ok=bool(real_ok), single_stage_no_tokenizer=True,
        runs_on_cpu=bool(args.device == "cpu"),
        frozen_feature_dump_available=bool(load_ok and real_ok),
        verdict=("GO (control/backup): frozen CBraMod backbone loads clean and yields a non-degenerate pooled "
                 "window embedding on real SHU-MI EEG; single-stage (no VQ tokenizer), so no code-collapse risk."
                 if (load_ok and real_ok) else "PARTIAL/NO-GO: see per-stage detail."))
    (OUT / "cbramod_preflight.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep["preflight_gate"], indent=2))
    print("load: ckpt=%d missing=%d unexpected=%d params=%d" % (rep["load"]["n_ckpt_tensors"],
          rep["load"]["missing_keys"], rep["load"]["unexpected_keys"], rep["load"]["n_params"]))
    print("real SHU: feats", rep["real_forward_shu"]["feats_shape"], "pooled", rep["real_forward_shu"]["pooled_shape"],
          "finite", rep["real_forward_shu"]["finite"], "L2", rep["real_forward_shu"]["pooled_mean_pairwise_L2"])


if __name__ == "__main__":
    main()
