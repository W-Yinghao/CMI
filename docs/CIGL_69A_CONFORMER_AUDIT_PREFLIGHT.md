# CIGL_69A — MetaCMI backbones + Conformer audit preflight (ERM seed0)

```
Branch project/metacmi-eegnet-conformer (off e619713, after the static-DGCNN CIGL/FCIGL/dCIGL route was frozen as
method-level negative). NEW PARADIGM: source-episodic Meta-CMI on STRONG non-graph EEG decoders, CMI/audit on
feature_z (pre-classifier), not graph_z. Phase 1 (this doc) = backbone scaffold + ERM audit preflight.
Two backbones: EEGNetMini (controlled anchor) + EEGConformerMini (high-capacity arm). MetaCMI = Phase 2.
```

## Why two backbones
Per PM: the "can CMI produce a stronger model?" question must NOT be gated behind a low-capacity anchor.
- **EEGNetMini** — controlled anchor; validated, stable, low-risk → clean mechanism judgment.
- **EEGConformerMini** — high-capacity arm → tests whether CMI transfers to a stronger model.
If only Conformer: a MetaCMI win can't be separated from Conformer capacity; a loss can't be separated from
Conformer instability. The anchor disambiguates.

## Backbones (`cmi/models/sanity_backbones.py`)
- `EEGNetMini` (existing): `forward(x[B,C,T]) -> (logits, z)`, single `nn.Linear` head over flattened features.
- **`EEGConformerMini` (new, faithful-minimal, NOT official/braindecode):** shallow conv tokenizer (temporal
  conv → spatial conv collapsing channels → BN/ELU → windowed avgpool) → small Transformer encoder (depth 2,
  4 heads, emb 32) → flatten → single `nn.Linear` head. Same `(logits, feature_z)` contract; feature_z =
  flattened transformer output. **Head-replay exact** (0.0 diff): `logits = feature_z @ headᵀ + b` in eval.
  z_dim probed dynamically (2a 544, 2015 1696); ~35–43k params.

## Naming/claim guard (REQUIRED)
For the non-graph backbones (EEGNetMini / EEGConformerMini): **`graph_z` in the `.audit.npz` means the generic
pre-classifier `feature_z`** (stored there only to reuse the head-replay-verified R3 slot), and **`node_z` is a
`[N,1,1]` dummy schema filler**. **No graph/node leakage claim is made for these CNN/transformer backbones** — the
audit measures label-conditional *subject* leakage on `feature_z`, not any graph/node structure.
`EEGConformerMini` / `EEGNetMini` are project-internal minimal pure-torch reimplementations, **NOT** the official
EEGNet / braindecode EEG Conformer.

## feature_z audit (no graph objects)
`cmi/eval/head_export.py`: `forward_feature_capture` (forward→(logits, feature_z, dummy node_z)) +
`save_fold_audit` auto-dispatch (uses forward_graph for graph backbones, forward for feature backbones). feature_z
is stored in the **graph_z slot** of the `.audit.npz` (head-replay verification is hardwired there) with a
`[N,1,1]` dummy node_z. So the SAME R3 / head-replay / reliance machinery runs unchanged on feature_z — verified
`replay_ok`, `evaluate_reliance` in `head_replay` mode, firewall passed. node_z is never read by R3.

## Preflight tests (`tests/test_metacmi_preflight.py`, 10 CPU pass)
both backbones × 2a/2015 dims build + **exact feature_z head-replay**; Conformer is the internal Mini (has
`.transformer`, not braindecode); `forward_feature_capture` shapes + linear replay; **feature_z audit is
R3-consumable** (validate clean, replay_ok, head_replay mode, firewall); **no target-label leakage** (flipping
target y leaves the source-fit R3 subspace unchanged); no dependency breakage (all Mini decoders build in eeg2025,
no braindecode).

## ERM seed0 preflight gate (`scripts/run_metacmi_gate.py`)
2 backbones × ERM × BNCI2014_001 (9) + BNCI2015_001 (12) = **42 GPU runs**, full LOSO, seed 0, source-only,
source-val early-stop, target eval-only. Per (backbone, fold): metrics JSON + Pareto row + verified feature_z
`.audit.npz` + firewall metadata. Question: **is Conformer stronger than EEGNetMini under strict source-only
LOSO, and is its feature_z auditable (R3 + head-replay)?**

## Preflight pass criteria (PM)
- **Backbone pass:** `Conformer ERM target_bacc ≥ EEGNetMini ERM` (or comparable) with source-val/target not
  collapsing. If Conformer ERM is unstable, do NOT interpret CMI on it.
- feature_z auditable on both (replay_ok, R3 consumable) — already verified on synthetic; real-EEG confirmation
  is the gate.

## Phase 2 (only if preflight healthy) — see CIGL_69B
CPU scaffold built + tested in `docs/CIGL_69B_METACMI_SCAFFOLD.md`: `metace` (CE(meta_train)+ρ·CE(meta_heldout))
+ `metacmi_direct` β0.1 (SymKL removal on meta_heldout; projector fit on meta_train subjects only). First GPU
gate reuses the frozen CIGL_69A ERM comparators (no ERM rerun) → adds only the 2 new methods × 2 backbones ×
(9+12) folds. **NO β=0.5 in the first gate.** GPU HELD until the Phase-1 readout + PM approval.

## Interpretation guardrail (Mini ≠ official)
`EEGConformerMini` is internal faithful-minimal, NOT official/braindecode. A positive MetaCMI signal on it =
*"CMI helps an audit-compatible ConformerMini; validate on full/official (or equal-param) Conformer next"* —
never "CMI works on Conformer". A negative on Mini does not kill the Conformer family. Any family-level claim
needs the full/fuller Conformer arm (**CIGL_69A2 full-Conformer preflight**), not Mini alone.

## Frozen constraints
No old-CIGL/FCIGL/dCIGL sweep, no P10, no baseline zoo, no architecture search without CMI, no official-braindecode
backbone, no manuscript push. Static-DGCNN reliance-control route stays frozen (method-level negative).
```
```
