#!/usr/bin/env python
"""FSR Phase 8B closure — CodeBrain temporal-token side-check (ONE-TIME, per PM/FSR_46 STOP-6: NO retrain/tuning).
Preflight saw n_unique_temporal=1 on 8 real SHU trials under /100 AND z-score. This confirms whether the pretrained
temporal codebook is GLOBALLY collapsed across MANY diverse real trials + a canonical 30-patch window (the only
remaining cheap checks). Frequency codes reported for contrast. If still ~1 unique -> temporal_token_status=collapsed,
excluded from main analysis. GPU. Writes temporal_token_sidecheck.csv + status."""
import csv, glob, json, os, sys
from pathlib import Path
import numpy as np
import scipy.io as sio
from scipy.signal import resample

CODEBRAIN = "/home/infres/yinwang/CodeBrain"
CKPT_TOK = "/home/infres/yinwang/eeg2025/NIPS/CodeBrain/Checkpoints/CodeBrain_Tokenizer.pth"
SHU = "/projects/EEG-foundation-model/SHU-MI-cbramod/mat"
BNCI = "/projects/EEG-foundation-model/datalake/raw/MNE-bnci-data/database/data-sets/001-2014"
OUT = Path("results/fsr_codebrain_cbramod_8b")


def unwrap(sd):
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def zscore(x):
    return (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6)


def main():
    sys.path.insert(0, CODEBRAIN)
    import torch
    import Models.modeling_tokenizer as MT  # noqa
    from timm.models import create_model
    torch.set_num_threads(4)
    dev = torch.device("cuda")
    tok = create_model("tfdual_vq", pretrained=False, EEG_size=6000, n_code_t=4096, n_code_f=4096, code_dim=32)
    tok.load_state_dict(unwrap(torch.load(CKPT_TOK, map_location="cpu", weights_only=False)), strict=False)
    tok.eval().to(dev)
    rows = []

    def probe(name, X):   # X (n, C, npatch, 200) float
        C = X.shape[1]; ic = list(range(C + 1))
        Tt, Tf = [], []
        with torch.no_grad():
            for i in range(0, len(X), 64):
                xt = torch.tensor(X[i:i + 64], dtype=torch.float32, device=dev)
                tt, tf = tok.get_codebook_indices(xt, input_chans=ic)
                Tt.append(tt.cpu().numpy().ravel()); Tf.append(tf.cpu().numpy().ravel())
        Tt = np.concatenate(Tt); Tf = np.concatenate(Tf)
        def ent(a):
            p = np.bincount(a, minlength=4096).astype(float); p = p[p > 0] / p.sum(); return float(-(p * np.log(p)).sum())
        r = dict(probe=name, n_trials=int(len(X)), n_tokens=int(Tt.size),
                 temporal_n_unique=int(np.unique(Tt).size), temporal_entropy=round(ent(Tt), 4),
                 freq_n_unique=int(np.unique(Tf).size), freq_entropy=round(ent(Tf), 4))
        rows.append(r); print(r)
        return r

    # (a) MANY real SHU trials (~1000), 32 ch, 4 patches, z-score
    Xs = []
    for f in sorted(glob.glob(f"{SHU}/*.mat"))[:12]:
        d = sio.loadmat(f)["data"].astype(np.float32)
        x = resample(d, 800, axis=-1); x = zscore(x).reshape(len(x), 32, 4, 200)
        Xs.append(x)
    Xs = np.concatenate(Xs, 0)[:1000]
    a = probe("shu_1000trials_32ch_4patch_zscore", Xs)

    # (b) canonical 30-patch REAL window from BNCI continuous (30 s @200Hz), 22 ch
    m = sio.loadmat(f"{BNCI}/A01T.mat", struct_as_record=False, squeeze_me=True)
    run = [r for r in m["data"] if np.atleast_1d(getattr(r, "y", [])).size > 0][0]
    Xc = getattr(run, "X")[:, :22].T            # (22, T) @250
    Xc = resample(Xc, int(Xc.shape[1] * 200 / 250), axis=-1)
    win = Xc[:, :6000]                          # 30 s @200Hz
    win = zscore(win).reshape(1, 22, 30, 200)
    b = probe("bnci_1window_22ch_30patch_zscore", win)

    collapsed = bool(a["temporal_n_unique"] <= 4 and b["temporal_n_unique"] <= 4)
    status = "collapsed" if collapsed else "diverse"
    (OUT / "temporal_token_sidecheck.csv").write_text(
        "probe,n_trials,n_tokens,temporal_n_unique,temporal_entropy,freq_n_unique,freq_entropy\n" +
        "".join(f"{r['probe']},{r['n_trials']},{r['n_tokens']},{r['temporal_n_unique']},{r['temporal_entropy']},{r['freq_n_unique']},{r['freq_entropy']}\n" for r in rows))
    (OUT / "temporal_token_status.json").write_text(json.dumps(dict(
        temporal_token_status=status, frequency_token_status="usable" if a["freq_n_unique"] > 50 else "check",
        note=("One-time side-check (no retrain/tuning). Temporal codebook globally collapsed on real EEG (SHU 1000 "
              "trials + BNCI canonical 30-patch window) => excluded from main analysis; frequency codes diverse. "
              "CodeBrain's temporal-token interpretability claims are NOT transported to our audit (C23)." if collapsed
              else "Temporal codes show diversity on some probe; re-examine before excluding."),
        probes=rows), indent=2) + "\n")
    print("TEMPORAL_TOKEN_STATUS:", status)


if __name__ == "__main__":
    main()
