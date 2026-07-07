# CodeBrain frozen feature-dump schema (FSR Phase 8 audit entry points)

Confirmed against the local base weights (`CodeBrain.pth` 60.9 MB EEGSSM, `CodeBrain_Tokenizer.pth` 112.3 MB
TFDual) on GPU. **All frozen (`no_grad`), no target labels, no training.** CodeBrain inference requires GPU (the
SSSM sliding-window mask hardcodes `.cuda()` at `Models/SSSM.py:129`).

## Canonical input
```
x : (B, C, n_patch, 200)   float, raw EEG resampled to 200 Hz, 1 s patches (200 samples/patch)
C        = channels mapped to CodeBrain montage (pretraining used channel_list = range(20); tensor 19 ch)
n_patch  = seconds per window; pretraining window = 30 s => C*n_patch <= 570 (= s4_lmax)
input_chans = list(range(C))   # channel-embedding index into the tokenizer's montage vocabulary
```
For MI (≈4 s trials): tile/pad to a fixed `n_patch`; keep `C*n_patch ≤ 570`. Encoder (SSSM, conv/S4) is
channel/length-flexible; the **tokenizer's NeuralTransformer pos/channel embeddings are montage-tied** → feed the
canonical layout (19 ch × 30 patch worked; 20 ch × 5 patch mismatched the pos-embed).

## Feature entry points (per window)
| id | tensor | shape | what it is | FSR use |
|----|--------|-------|------------|---------|
| F0_tok_temporal | `token_t` | (B, C·n_patch) int in [0,4096) | TFDual **temporal** codebook ids (time-domain decoupled code) | L1 subject-decodability of temporal codes; token subject- vs class-specificity |
| F1_tok_frequency | `token_f` | (B, C·n_patch) int in [0,4096) | TFDual **frequency** codebook ids (spectral decoupled code) | L1 subject-decodability of frequency codes — *is subject info spectral?* |
| F2_tok_quant_t/f | `quantize_t`,`quantize_f` | (B, C·n_patch, 32) float | continuous quantized code vectors | linear-probe leakage/reliance on the code space |
| F3_enc_hidden | `feats` (EEGSSM, `proj_out`=Id) | (B, C, n_patch, 200) float | per-(channel,patch) encoder hidden | L1/L4/L5 per-channel + per-time reliance; spatial-subject audit |
| F4_enc_pooled | mean over (C,n_patch) of F3 | (B, 200) float | pooled window embedding (the primary substrate) | L1 subject probe, L4 task-head alignment, L5 subspace intervention, L6 target consequence |
| F4b_enc_flat | flatten F3 | (B, C·n_patch·200) | full encoder features (as CodeBrain downstream head input) | matches `model_for_shu` head input; heavy — pool for probes |

## FSR ladder mapping on CodeBrain (Phase 8B/8C)
- **L1 leakage** — linear/subject-probe accuracy on F0/F1/F3/F4 separately → *where* subject identity lives
  (temporal code vs frequency code vs encoder hidden vs pooled). Key CodeBrain-specific question.
- **L4 task-head alignment** — cosine of a trained linear task head with the subject subspace of F4.
- **L5 subject reliance** — accuracy drop of the task head on F4 after erasing its subject subspace (as in 4B/PC1).
- **L6 target consequence** — held-out-target true-label bAcc after the F4 subspace intervention (firewall: target
  labels final scoring only).
- **Scaling axis (8C)** — recompute L1/L4/L5/L6 as a function of `N_source` subjects, at fixed vs growing total
  trials (subject-diversity vs sample-size decomposition).

## Load recipe (from `model_for_shu.py`, verified)
```python
enc = SSSM(in_channels=200,res_channels=200,skip_channels=200,out_channels=200, num_res_layers=8,
           diffusion_step_embed_dim_in/mid/out=200, s4_lmax=570, s4_d_state=64, s4_dropout=0.,
           s4_bidirectional=True, s4_layernorm=True, codebook_size_t=4096, codebook_size_f=4096, if_codebook=False)
sd = torch.load("CodeBrain.pth", map_location="cpu", weights_only=False)     # strip "module." prefix (k[7:])
enc.load_state_dict({k[7:]:v for k,v in sd.items()}, strict=False)            # 269 tensors, 0 missing/unexpected
enc.proj_out = nn.Sequential(); enc.eval().cuda()                            # frozen; feats = enc(x)
tok = create_model("tfdual_vq", EEG_size=6000, n_code_t=4096, n_code_f=4096, code_dim=32)   # weights_only=False
tok.load_state_dict({k[7:] if k.startswith('module.') else k: v for k,v in torch.load('CodeBrain_Tokenizer.pth',
    weights_only=False).items()}, strict=False); tok.eval().cuda()
tok_t, tok_f = tok.get_codebook_indices(x, input_chans=list(range(C)))       # decoupled temporal/frequency codes
```
