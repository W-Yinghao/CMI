# CIGL_69A2 — Internal full-capacity Conformer preflight (CPU engineering only)

```
Branch project/metacmi-eegnet-conformer. Answers the PM directive: ConformerMini parity/leakage results are
NOT a Conformer-family verdict; validate on the ORIGINAL Conformer or an equal-param model. Because the OFFICIAL
braindecode EEGConformer is UNAVAILABLE in eeg2025, we therefore use an INTERNAL full-capacity `conformer_full`
arm (official-geometry-INSPIRED, NOT the official model). This preflight builds that high-capacity arm and
proves it is audit-compatible (feature_z + probe R3), source-only-LOSO ready, and leak-proof. NO GPU launched.
```

## Naming guardrail (PM-required)
`EEGConformerFull` / `conformer_full` is an **internal full-capacity / official-geometry-inspired Conformer
arm**. It is **NOT** to be called the "official EEG Conformer". The honest framing everywhere is:
> official braindecode implementation unavailable in eeg2025; we therefore use an internal full-capacity
> `conformer_full` arm.

## Official-import status (recorded, env-dependent)
- **Official braindecode `EEGConformer` is NOT importable in `eeg2025`**: `ImportError: cannot import name
  'BNCI2014001' from 'moabb.datasets'` (the known eager-import break; `moabb` in this env lacks that symbol).
- ⇒ the official class cannot be run here without env surgery. The preflight therefore provides the **internal
  full-capacity `conformer_full` arm** (equal-param, official-geometry-inspired), and records the import status
  so we can also run the literal official model later if/when a compatible env is available.

## `EEGConformerFull` (`cmi/models/sanity_backbones.py`) — the high-capacity arm
Faithful to Song et al. 2022 (arXiv:2106.11170):
- **Tokenizer (official kernels):** Conv2d(1,40,(1,25)) temporal (valid) → Conv2d(40,40,(chans,1)) spatial →
  BN → ELU → AvgPool2d((1,75),(1,15)) → Dropout(0.5) → Conv2d(40,40,(1,1)) projection → tokens [B,T',40].
- **Encoder:** depth **6**, emb **40**, **10** heads, FF expansion 4, GELU, dropout 0.5.
- **Head:** the official **3-layer MLP** (Linear→ELU→Drop→Linear→ELU→Drop→Linear), **NOT** a single linear.
- **Capacity:** z_dim 800; **369,732 params (2a) / 355,266 (2015) ≈ 7.8–10.2× EEGConformerMini** (47k / 35k).
- Contract `forward(x)->(logits, feature_z)`, feature_z = flattened transformer output = MLP-head input.
- Official pool geometry needs n_times ≳ 100 (temporal conv is valid); the pool is clamped only for tiny
  synthetic smokes, and equals the official (75,15) on real EEG.

## Audit compatibility — probe fallback (the PM-allowed "probe-compatible head")
Because the head is an MLP, **head-replay is not exact** → `extract_task_head` returns `unsupported` →
`save_fold_audit` writes `replay_ok=False` → **R3 falls back to the source-fit probe**
(`removal_mode='probe_replay'`, `firewall_passed=True`). So the Full Conformer yields a *representation*-reliance
claim (source-fit probe), not the *classifier*-reliance (head-replay) claim ConformerMini's single-linear head
supports. This is the honest, expected trade for the fuller architecture.

## Preflight checks (`tests/test_conformer_full_preflight.py`, 7 CPU pass)
build+forward finite (2a/2015); **≥5× Mini params** (genuinely high-capacity); MLP head ⇒ `extract_task_head`
unsupported; `forward_feature_capture` shapes; **R3 artifact probe-compatible** (`validate==[]`, not
head_replay, `removal_mode='probe_replay'`, `firewall_passed`); **no target-label leakage** (flipping target y
leaves the source-fit probe reliance unchanged); official-import status recorded (never fails the suite).

## End-to-end gate path (already wired; no gate code change)
`scripts/run_metacmi_gate.py --backbones conformer_full` works unchanged: `build_sanity_backbone` knows
`conformer_full`; `save_fold_audit` auto-dispatches to `forward_feature_capture` (no `forward_graph`); the audit
uses the probe fallback. CPU dry-run runs ERM / MetaCE / MetaCMI-Direct β0.1 on `conformer_full`
(MetaCMI-Direct applies the removal through the MLP head — a valid non-linear reliance objective).

## Decision / recommended next (NO GPU until PM approval)
`EEGConformerFull` is a **valid, high-capacity, audit-compatible, leak-proof** validation arm and is ready for
full-Conformer LOSO seed0 whenever approved. Proposed sequence:
1. **Full-Conformer ERM seed0 LOSO** ({2a, 2015}, source-only) = 21 runs — is the *fuller* Conformer stronger
   than EEGNetMini (the question ConformerMini answered only at parity)?
2. **Any positive Phase-2 MetaCMI signal on ConformerMini must be re-validated here** (EEGConformerFull) before a
   "CMI produces a stronger model" claim.
Held: no GPU; official braindecode arm deferred to a compatible env; static-DGCNN route stays frozen.
```
```
