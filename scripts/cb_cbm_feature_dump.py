#!/usr/bin/env python
"""FSR Phase 8B — frozen feature dump for CodeBrain + CBraMod (see docs/FSR_46). Loads a dataset's raw EEG,
resamples to 200 Hz, per-trial/per-channel z-scores, patches to (C,n_patch,200), and dumps the frozen encoder
POOLED window embedding F0 in R^200 (+ CodeBrain frequency-token ids), with subject id d, task label y, session.
Deterministic (eval + no_grad + masking off); QC verifies F0 stability across repeats and batch groupings.
No target labels touched (pure feature extraction). GPU for CodeBrain, CPU ok for CBraMod.

    <eeg2025 python> scripts/cb_cbm_feature_dump.py --model codebrain --dataset shu --device cuda
"""
import argparse, glob, json, os, sys, time
from pathlib import Path
import numpy as np
import scipy.io as sio
from scipy.signal import resample

CODEBRAIN = "/home/infres/yinwang/CodeBrain"
CBRAMOD = "/home/infres/yinwang/eeg2025/CBraMod"
CKPT_CB_ENC = "/home/infres/yinwang/eeg2025/NIPS/CodeBrain/Checkpoints/CodeBrain.pth"
CKPT_CB_TOK = "/home/infres/yinwang/eeg2025/NIPS/CodeBrain/Checkpoints/CodeBrain_Tokenizer.pth"
CKPT_CBM = "/home/infres/yinwang/eeg2025/NIPS/Cbramod_pretrained_weights.pth"
SHU = "/projects/EEG-foundation-model/SHU-MI-cbramod/mat"
SHU_CH = ["FP1","FP2","FZ","F3","F4","F7","F8","FC1","FC2","FC5","FC6","CZ","C3","C4","T3","T4","A1","A2",
          "CP1","CP2","CP5","CP6","PZ","P3","P4","T5","T6","PO3","PO4","OZ","O1","O2"]
OUT = Path("results/fsr_codebrain_cbramod_8b")


def unwrap(sd):
    if isinstance(sd, dict):
        for k in ("model", "state_dict", "model_state_dict"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]; break
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def set_determinism(torch):
    import random
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # required for deterministic cuBLAS
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:                                                          # force batch-size-invariant conv/matmul algos
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def load_shu():
    """yield (X (n,32,1000)@250Hz, y (n,), subject_id, session_id) per session .mat."""
    for f in sorted(glob.glob(f"{SHU}/*.mat")):
        base = os.path.basename(f)
        sub = int(base.split("sub-")[1][:3]); ses = int(base.split("ses-")[1][:2])
        m = sio.loadmat(f)
        yield m["data"].astype(np.float32), m["labels"].astype(int).ravel(), sub, ses, base


def preprocess(x_raw, native_hz=250):
    """(n,C,T)@native -> (n,C,n_patch,200) @200Hz, per-trial/per-channel z-score, patch=1s."""
    n, C, T = x_raw.shape
    T200 = int(round(T * 200 / native_hz))
    x = resample(x_raw, T200, axis=-1)
    x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6)
    npatch = T200 // 200
    x = x[:, :, :npatch * 200].reshape(n, C, npatch, 200)
    return x.astype(np.float32), npatch


def build_codebrain(torch, device):
    sys.path.insert(0, CODEBRAIN)
    from Models.SSSM import SSSM
    import Models.modeling_tokenizer as MT  # noqa
    from timm.models import create_model
    enc = SSSM(in_channels=200, res_channels=200, skip_channels=200, out_channels=200, num_res_layers=8,
               diffusion_step_embed_dim_in=200, diffusion_step_embed_dim_mid=200, diffusion_step_embed_dim_out=200,
               s4_lmax=570, s4_d_state=64, s4_dropout=0.0, s4_bidirectional=True, s4_layernorm=True,
               codebook_size_t=4096, codebook_size_f=4096, if_codebook=False)
    em, eu = enc.load_state_dict(unwrap(torch.load(CKPT_CB_ENC, map_location="cpu", weights_only=False)), strict=False)
    enc.proj_out = torch.nn.Sequential(); enc.eval().to(device)
    tok = create_model("tfdual_vq", pretrained=False, EEG_size=6000, n_code_t=4096, n_code_f=4096, code_dim=32)
    tok.load_state_dict(unwrap(torch.load(CKPT_CB_TOK, map_location="cpu", weights_only=False)), strict=False)
    tok.eval().to(device)

    def fwd(xt):                     # (B,C,n_patch,200) -> pooled (B,200)
        feats = enc(xt); return feats.mean(dim=(1, 2))

    def freq_tokens(xt):
        C = xt.shape[1]
        _, tok_f = tok.get_codebook_indices(xt, input_chans=list(range(C + 1)))
        return tok_f

    return fwd, freq_tokens, dict(missing=len(em), unexpected=len(eu), params=sum(p.numel() for p in enc.parameters()))


def build_cbramod(torch, device):
    sys.path.insert(0, CBRAMOD)
    from models.cbramod import CBraMod
    bb = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
    sd = torch.load(CKPT_CBM, map_location="cpu", weights_only=False)
    m, u = bb.load_state_dict(unwrap(sd), strict=False)
    bb.proj_out = torch.nn.Identity(); bb.eval().to(device)

    def fwd(xt):
        feats = bb(xt); return feats.mean(dim=(1, 2))
    return fwd, None, dict(missing=len(m), unexpected=len(u), params=sum(p.numel() for p in bb.parameters()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["codebrain", "cbramod"], required=True)
    ap.add_argument("--dataset", choices=["shu"], default="shu")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=64)
    args = ap.parse_args()
    (OUT / "embeddings").mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(4); set_determinism(torch)
    dev = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    fwd, freqfn, loadinfo = (build_codebrain if args.model == "codebrain" else build_cbramod)(torch, dev)

    F0, Y, D, SES, FREQH = [], [], [], [], []
    qc_rows = []
    t_start = time.time()
    for x_raw, y, sub, ses, base in load_shu():
        x, npatch = preprocess(x_raw, 250)
        xt = torch.tensor(x, dtype=torch.float32, device=dev)
        pooled = []
        with torch.no_grad():
            for i in range(0, len(xt), args.bs):
                pooled.append(fwd(xt[i:i + args.bs]).cpu().numpy())
        pooled = np.concatenate(pooled, 0)
        F0.append(pooled); Y.append(y); D.append(np.full(len(y), sub)); SES.append(np.full(len(y), ses))
        # frequency-token histogram per trial (CodeBrain only) — coarse 4096->64 bins for compactness
        if freqfn is not None:
            with torch.no_grad():
                tf = freqfn(xt[:min(len(xt), 64)]).cpu().numpy()
            FREQH.append(dict(subject=sub, session=ses, n=int(min(len(xt), 64)),
                              n_unique=int(np.unique(tf).size), entropy=float(_entropy(tf, 4096))))
        # determinism QC on the first session only: repeat + batch-grouping
        if base == os.path.basename(sorted(glob.glob(f"{SHU}/*.mat"))[0]):
            with torch.no_grad():
                a = fwd(xt[:32]).cpu().numpy()
                b = fwd(xt[:32]).cpu().numpy()                          # repeat identical
                c1 = fwd(xt[:16]).cpu().numpy(); c2 = fwd(xt[16:32]).cpu().numpy()
                c = np.concatenate([c1, c2], 0)                         # different batch grouping
            qc_rows.append(dict(model=args.model, dataset=args.dataset,
                                repeat_max_abs_diff=float(np.max(np.abs(a - b))),
                                batchgroup_max_abs_diff=float(np.max(np.abs(a - c))),
                                deterministic=bool(np.max(np.abs(a - b)) < 1e-5 and np.max(np.abs(a - c)) < 1e-4),
                                finite=bool(np.isfinite(pooled).all())))
        print(f"[{args.model}:{args.dataset}] {base} n={len(y)} done", flush=True)

    F0 = np.concatenate(F0, 0); Y = np.concatenate(Y); D = np.concatenate(D); SES = np.concatenate(SES)
    np.savez(OUT / "embeddings" / f"{args.model}_{args.dataset}_F0.npz", X=F0, y=Y, d=D, ses=SES)
    man = dict(model=args.model, dataset=args.dataset, n_trials=int(len(Y)),
               n_subjects=int(np.unique(D).size), n_sessions_per_subj=int(np.unique(SES).size),
               channels=SHU_CH, n_channels=len(SHU_CH), native_hz=250, target_hz=200, patch=200, n_patch=int(npatch),
               classes=sorted(int(c) for c in np.unique(Y)), embed_dim=int(F0.shape[1]),
               load=loadinfo, freq_token_hist=FREQH, qc=qc_rows, sec=round(time.time() - t_start, 1))
    (OUT / f"feature_dump_manifest_{args.model}_{args.dataset}.json").write_text(json.dumps(man, indent=2) + "\n")
    print(f"WROTE {args.model}_{args.dataset}_F0.npz  X={F0.shape} subjects={np.unique(D).size} "
          f"classes={sorted(int(c) for c in np.unique(Y))}  QC={qc_rows}")


def _entropy(ids, K):
    p = np.bincount(ids.ravel(), minlength=K).astype(float); p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum())


if __name__ == "__main__":
    main()
