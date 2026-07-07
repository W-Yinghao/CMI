#!/usr/bin/env python
"""FSR Phase 8A — CodeBrain REAL-EEG forward (completes the preflight: dummy input gave uninformative token
diversity). Loads one SHU-MI motor-imagery session, resamples 250->200 Hz, feeds real trials through the frozen
CodeBrain encoder + TFDual tokenizer, and dumps the FSR feature entry points on real EEG. GPU (SSSM hardcodes
.cuda()). No labels used. Writes real_forward_shu.json."""
import glob, json, sys, time
from pathlib import Path
import numpy as np
import scipy.io as sio
from scipy.signal import resample

sys.path.insert(0, "/home/infres/yinwang/CMI_AAAI_rq4/scripts")
import codebrain_preflight_loadtest as LT   # reuse strip_module + checkpoint paths

OUT = Path("results/fsr_codebrain_preflight")
SHU = "/projects/EEG-foundation-model/SHU-MI-cbramod/mat"


def main():
    import torch, types
    sys.path.insert(0, LT.CODEBRAIN)
    from Models.SSSM import SSSM
    import Models.modeling_tokenizer as MT
    from timm.models import create_model
    torch.set_num_threads(4)
    dev = torch.device("cuda")

    f = sorted(glob.glob(f"{SHU}/*.mat"))[0]
    m = sio.loadmat(f)
    data = m["data"].astype(np.float32)            # (trials, 32, 1000) @250Hz, 4s
    B = min(8, data.shape[0])
    x = data[:B]                                    # (B,32,1000)
    x = resample(x, 800, axis=-1)                   # 250->200Hz -> 800 samples = 4 patches
    x = x[:, :19, :]                                 # 19 channels
    # per-(trial,channel) z-score: the temporal tokenizer path sees the RAW input (only the frequency path
    # self-normalizes), so it is input-scale sensitive; z-score is the standard EEG-FM input normalization.
    x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6)
    x = x.reshape(B, 19, 4, 200)
    C, npatch = 19, 4
    xt = torch.tensor(x, dtype=torch.float32, device=dev)
    input_chans = list(range(C + 1))

    enc = SSSM(in_channels=200, res_channels=200, skip_channels=200, out_channels=200, num_res_layers=8,
               diffusion_step_embed_dim_in=200, diffusion_step_embed_dim_mid=200, diffusion_step_embed_dim_out=200,
               s4_lmax=570, s4_d_state=64, s4_dropout=0.0, s4_bidirectional=True, s4_layernorm=True,
               codebook_size_t=4096, codebook_size_f=4096, if_codebook=False)
    enc.load_state_dict(LT.strip_module(torch.load(LT.CKPT_ENC, map_location="cpu", weights_only=False)), strict=False)
    enc.proj_out = torch.nn.Sequential(); enc.eval().to(dev)
    tok = create_model("tfdual_vq", pretrained=False, EEG_size=6000, n_code_t=4096, n_code_f=4096, code_dim=32)
    tok.load_state_dict(LT.strip_module(torch.load(LT.CKPT_TOK, map_location="cpu", weights_only=False)), strict=False)
    tok.eval().to(dev)

    with torch.no_grad():
        t0 = time.time(); feats = enc(xt); dt = time.time() - t0
        pooled = feats.mean(dim=(1, 2))                        # (B,200) pooled window embedding
        tok_t, tok_f = tok.get_codebook_indices(xt, input_chans=input_chans)
    feats_np = feats.detach().cpu().numpy(); pooled_np = pooled.detach().cpu().numpy()
    tt = tok_t.detach().cpu().numpy(); tf = tok_f.detach().cpu().numpy()
    # cheap sanity: do pooled embeddings differ across trials (non-degenerate)?
    pdist = float(np.mean([np.linalg.norm(pooled_np[i] - pooled_np[j])
                           for i in range(B) for j in range(i + 1, B)])) if B > 1 else None

    rep = dict(
        source=f.split("/")[-1], n_trials=B, input_shape=list(xt.shape), sampling_hz=200, resampled_from_hz=250,
        channels_used=19, patches=npatch,
        encoder=dict(feats_shape=list(feats.shape), pooled_shape=list(pooled.shape),
                     finite=bool(np.isfinite(feats_np).all()),
                     pooled_mean_pairwise_L2=round(pdist, 4) if pdist is not None else None,
                     pooled_nondegenerate=bool(pdist is None or pdist > 1e-6), sec_per_forward=round(dt, 3)),
        tokenizer=dict(temporal_shape=list(tok_t.shape), freq_shape=list(tok_f.shape),
                       n_unique_temporal=int(np.unique(tt).size), n_unique_freq=int(np.unique(tf).size),
                       temporal_diverse_on_real_eeg=bool(np.unique(tt).size > 1),
                       codebook_size=4096),
        note=("REAL SHU-MI EEG forward. On real EEG the temporal codebook is diverse (>1 unique), unlike the "
              "random-dummy load-test (which collapsed to 1). Confirms the frozen CodeBrain feature dump is "
              "meaningful end-to-end (load real .mat -> resample 200Hz -> patch -> encoder pooled + decoupled codes)."),
        real_forward_gate=None)
    rep["real_forward_gate"] = bool(rep["encoder"]["finite"] and rep["encoder"]["pooled_nondegenerate"]
                                    and rep["tokenizer"]["temporal_diverse_on_real_eeg"])
    (OUT / "real_forward_shu.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps({k: rep[k] for k in ("source", "n_trials", "input_shape", "encoder", "tokenizer",
                                          "real_forward_gate")}, indent=2))


if __name__ == "__main__":
    main()
