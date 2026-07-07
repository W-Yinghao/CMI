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

    def fwd(xt):                     # (B,C,n_patch,200) -> full feats (B,C,n_patch,200)
        f = enc(xt)                  # SSSM does x.squeeze() -> drops batch dim when B==1; restore shape
        return f if f.dim() == 4 else f.reshape(xt.shape[0], xt.shape[1], xt.shape[2], -1)

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
        return bb(xt)                # full feats (B,C,n_patch,200)
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

    def pool(feats):                 # feats (B,C,np,200) -> F0 pooled (B,200), F1 per-channel (B,C*200)
        B, C, npp, D = feats.shape
        f0 = feats.mean(axis=(1, 2))
        f1 = feats.mean(axis=2).reshape(B, C * D)
        return f0, f1

    F0, F1, Y, D, SES, FREQH = [], [], [], [], [], []
    qc_rows = []
    t_start = time.time()
    files = sorted(glob.glob(f"{SHU}/*.mat"))
    for x_raw, y, sub, ses, base in load_shu():
        x, npatch = preprocess(x_raw, 250)
        xt = torch.tensor(x, dtype=torch.float32, device=dev)
        f0l, f1l = [], []
        with torch.no_grad():
            for i in range(0, len(xt), args.bs):
                f0b, f1b = pool(fwd(xt[i:i + args.bs]).cpu().numpy())
                f0l.append(f0b); f1l.append(f1b)
        f0 = np.concatenate(f0l, 0); f1 = np.concatenate(f1l, 0)
        F0.append(f0); F1.append(f1); Y.append(y); D.append(np.full(len(y), sub)); SES.append(np.full(len(y), ses))
        if freqfn is not None:
            with torch.no_grad():
                tf = freqfn(xt[:min(len(xt), 64)]).cpu().numpy()
            FREQH.append(dict(subject=sub, session=ses, n=int(min(len(xt), 64)),
                              n_unique=int(np.unique(tf).size), entropy=float(_entropy(tf, 4096))))
        if base == os.path.basename(files[0]):     # determinism QC: repeat + batch-grouping, on F0 and F1
            with torch.no_grad():
                a0, a1 = pool(fwd(xt[:32]).cpu().numpy())
                b0, b1 = pool(fwd(xt[:32]).cpu().numpy())
                c0a, c1a = pool(fwd(xt[:16]).cpu().numpy()); c0b, c1b = pool(fwd(xt[16:32]).cpu().numpy())
                c0 = np.concatenate([c0a, c0b], 0); c1 = np.concatenate([c1a, c1b], 0)
            qc_rows.append(dict(model=args.model, dataset=args.dataset, bs=args.bs,
                                F0_repeat_max=float(np.max(np.abs(a0 - b0))), F0_batchgroup_max=float(np.max(np.abs(a0 - c0))),
                                F1_repeat_max=float(np.max(np.abs(a1 - b1))), F1_batchgroup_max=float(np.max(np.abs(a1 - c1))),
                                deterministic_repeat=bool(np.max(np.abs(a0 - b0)) < 1e-5 and np.max(np.abs(a1 - b1)) < 1e-5),
                                finite=bool(np.isfinite(f0).all() and np.isfinite(f1).all())))
        print(f"[{args.model}:{args.dataset}] {base} n={len(y)} done", flush=True)

    F0 = np.concatenate(F0, 0); F1 = np.concatenate(F1, 0)
    Y = np.concatenate(Y); D = np.concatenate(D); SES = np.concatenate(SES)
    suffix = f"_bs{args.bs}" if args.bs != 64 else ""
    np.savez(OUT / "embeddings" / f"{args.model}_{args.dataset}_F0{suffix}.npz", X=F0, y=Y, d=D, ses=SES)
    np.savez(OUT / "embeddings" / f"{args.model}_{args.dataset}_F1{suffix}.npz", X=F1, y=Y, d=D, ses=SES,
             n_channels=len(SHU_CH), chan_dim=200)
    man = dict(model=args.model, dataset=args.dataset, bs=args.bs, n_trials=int(len(Y)),
               n_subjects=int(np.unique(D).size), n_sessions_per_subj=int(np.unique(SES).size),
               channels=SHU_CH, n_channels=len(SHU_CH), native_hz=250, target_hz=200, patch=200, n_patch=int(npatch),
               classes=sorted(int(c) for c in np.unique(Y)), F0_dim=int(F0.shape[1]), F1_dim=int(F1.shape[1]),
               F1_desc="per-channel: mean over patches, keep channels, flatten C*200 (preserves MI spatial lateralization)",
               load=loadinfo, freq_token_hist=FREQH, qc=qc_rows, sec=round(time.time() - t_start, 1))
    (OUT / f"feature_dump_manifest_{args.model}_{args.dataset}{suffix}.json").write_text(json.dumps(man, indent=2) + "\n")
    print(f"WROTE {args.model}_{args.dataset} F0={F0.shape} F1={F1.shape} subjects={np.unique(D).size} QC={qc_rows}")


def _entropy(ids, K):
    p = np.bincount(ids.ravel(), minlength=K).astype(float); p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum())


if __name__ == "__main__":
    main()
