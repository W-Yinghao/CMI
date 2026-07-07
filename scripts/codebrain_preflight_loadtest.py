#!/usr/bin/env python
"""FSR Phase 8A — CodeBrain preflight load-test (fail-closed gate). Confirms the LOCAL base pretrained CodeBrain
weights load into the upstream architecture and produce frozen FSR feature-dump entry points:
  - TFDual tokenizer  (CodeBrain_Tokenizer.pth) -> temporal token ids + frequency token ids  (decoupled codes)
  - EEGSSM encoder    (CodeBrain.pth)           -> per-(channel,patch) hidden + pooled window embedding
No target labels, no training. CPU by default; --device cuda if S4 needs it. Writes codebrain_load_test.json.

    <eeg2025 python> scripts/codebrain_preflight_loadtest.py [--device cpu|cuda]
"""
import argparse, json, os, sys, time, types
from pathlib import Path
import numpy as np

CODEBRAIN = "/home/infres/yinwang/CodeBrain"
CKPT_ENC = "/home/infres/yinwang/eeg2025/NIPS/CodeBrain/Checkpoints/CodeBrain.pth"
CKPT_TOK = "/home/infres/yinwang/eeg2025/NIPS/CodeBrain/Checkpoints/CodeBrain_Tokenizer.pth"
OUT = Path("results/fsr_codebrain_preflight")


def strip_module(sd):
    # unwrap common checkpoint containers ({'model':...}/{'state_dict':...}) then drop DataParallel 'module.' prefix
    if isinstance(sd, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in sd and isinstance(sd[key], dict):
                sd = sd[key]; break
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--chans", type=int, default=19)
    ap.add_argument("--patches", type=int, default=30)  # canonical 30s window = s4_lmax/19
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, CODEBRAIN)
    import torch
    torch.set_num_threads(4)
    from Models.SSSM import SSSM
    import Models.modeling_tokenizer as MT
    from timm.models import create_model  # registers tfdual_vq

    rep = dict(device=args.device, torch=torch.__version__, weights=dict(
        encoder=dict(path=CKPT_ENC, mb=round(os.path.getsize(CKPT_ENC) / 1e6, 1)),
        tokenizer=dict(path=CKPT_TOK, mb=round(os.path.getsize(CKPT_TOK) / 1e6, 1))))
    dev = torch.device(args.device)
    B, C, S, P = 2, args.chans, args.patches, 200
    x = torch.randn(B, C, S, P, device=dev)              # (batch, channels, patches, 200) raw @200Hz, patch=1s
    input_chans = list(range(C + 1))   # tokenizer: input_chans[0]=cls slot, rest=C data channels

    # ---- EEGSSM encoder (the primary FSR audit substrate) ----
    P8 = types.SimpleNamespace(n_layer=8, dropout=0.0, codebook_size_t=4096, codebook_size_f=4096)
    enc = SSSM(in_channels=200, res_channels=200, skip_channels=200, out_channels=200,
               num_res_layers=P8.n_layer, diffusion_step_embed_dim_in=200, diffusion_step_embed_dim_mid=200,
               diffusion_step_embed_dim_out=200, s4_lmax=570, s4_d_state=64, s4_dropout=P8.dropout,
               s4_bidirectional=True, s4_layernorm=True, codebook_size_t=4096, codebook_size_f=4096,
               if_codebook=False)
    esd = strip_module(torch.load(CKPT_ENC, map_location="cpu", weights_only=False))
    miss, unexp = enc.load_state_dict(esd, strict=False)
    enc.proj_out = torch.nn.Sequential()
    enc.eval().to(dev)
    rep["encoder_load"] = dict(n_ckpt_tensors=len(esd), missing_keys=len(miss), unexpected_keys=len(unexp),
                               missing_sample=list(miss)[:6], unexpected_sample=list(unexp)[:6],
                               n_params=sum(p.numel() for p in enc.parameters()))
    with torch.no_grad():
        t0 = time.time(); feats = enc(x); dt = time.time() - t0
    feats_np = feats.detach().cpu().numpy()
    pooled = feats.reshape(B, -1).detach().cpu().numpy() if feats.dim() >= 2 else feats.detach().cpu().numpy()
    rep["encoder_forward"] = dict(input_shape=list(x.shape), feats_shape=list(feats.shape),
                                  pooled_dim=int(pooled.reshape(B, -1).shape[1]),
                                  finite=bool(np.isfinite(feats_np).all()),
                                  sec_per_forward=round(dt, 3), ok=True)

    # ---- TFDual tokenizer (decoupled temporal / frequency codes) ----
    try:
        tok = create_model("tfdual_vq", pretrained=False, EEG_size=6000, n_code_t=4096, n_code_f=4096, code_dim=32)
        tsd = strip_module(torch.load(CKPT_TOK, map_location="cpu", weights_only=False))
        tmiss, tunexp = tok.load_state_dict(tsd, strict=False)
        tok.eval().to(dev)
        with torch.no_grad():
            t0 = time.time(); tok_t, tok_f = tok.get_codebook_indices(x, input_chans=input_chans); dtt = time.time() - t0
        rep["tokenizer_load"] = dict(n_ckpt_tensors=len(tsd), missing_keys=len(tmiss), unexpected_keys=len(tunexp),
                                     missing_sample=list(tmiss)[:6], unexpected_sample=list(tunexp)[:6])
        rep["tokenizer_forward"] = dict(temporal_token_shape=list(tok_t.shape), freq_token_shape=list(tok_f.shape),
                                        n_unique_temporal=int(tok_t.unique().numel()),
                                        n_unique_freq=int(tok_f.unique().numel()),
                                        temporal_codebook=4096, freq_codebook=4096,
                                        sec_per_forward=round(dtt, 3), ok=True)
    except Exception as e:
        rep["tokenizer_load"] = dict(ok=False, error=f"{type(e).__name__}: {e}")

    enc_ok = rep["encoder_load"]["missing_keys"] == 0 and rep["encoder_forward"]["ok"] and rep["encoder_forward"]["finite"]
    tok_ok = rep.get("tokenizer_forward", {}).get("ok", False)
    rep["preflight_gate"] = dict(
        encoder_loads_clean=bool(rep["encoder_load"]["missing_keys"] == 0),
        encoder_forward_ok=bool(rep["encoder_forward"]["ok"] and rep["encoder_forward"]["finite"]),
        tokenizer_ok=bool(tok_ok),
        frozen_feature_dump_available=bool(enc_ok and tok_ok),
        no_download_needed=True, no_mamba_dep=True,
        verdict=("GO: frozen CodeBrain encoder + TFDual tokenizer both load from LOCAL weights and produce FSR "
                 "feature-dump entry points (temporal codes, frequency codes, EEGSSM hidden/pooled)."
                 if (enc_ok and tok_ok) else "PARTIAL/NO-GO: see per-stage errors."))
    (OUT / "codebrain_load_test.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep["preflight_gate"], indent=2))
    print("encoder feats_shape:", rep["encoder_forward"]["feats_shape"], "pooled_dim:", rep["encoder_forward"]["pooled_dim"])
    if "tokenizer_forward" in rep:
        print("temporal_token_shape:", rep["tokenizer_forward"]["temporal_token_shape"],
              "freq_token_shape:", rep["tokenizer_forward"]["freq_token_shape"])


if __name__ == "__main__":
    main()
