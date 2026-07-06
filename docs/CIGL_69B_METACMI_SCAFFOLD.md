# CIGL_69B — MetaCMI CPU-only scaffold (source-episodic Meta-CMI on feature_z)

```
Branch project/metacmi-eegnet-conformer. Phase 2 scaffold ONLY — CPU-built + CPU-tested, NO GPU launch.
Built in parallel with the queued Phase-1 ERM preflight (885120/885121). Phase-2 GPU is HELD until the
Phase-1 readout + PM approval. This doc adds the two Phase-2 trainer methods and their firewall tests.
```

## Methods (`cmi/train/trainer.py`, `META_METHODS = {"metace", "metacmi_direct"}`)
Within `source_train` only, the source SUBJECTS are episodically partitioned (deterministic per epoch,
`rng = default_rng(seed+ep)`) into `meta_train` vs `meta_heldout` (both always non-empty).
- **MetaCE** — `loss = CE(meta_train) + ρ·CE(meta_heldout)`. ρ = `meta_rho` (default 1.0). Trains a model that
  must generalize across a held-out *source* subject group — a source-episodic DG baseline, no CMI term.
- **MetaCMI-Direct** — MetaCE `+ warm·β·SymKL(h(z_mh), h((I−SᵀS)·z_mh))`. `S` = the label-conditional subject
  subspace (k=2) fit on `meta_train` feature_z ONLY (`_feature_subject_projector`), applied to the
  `meta_heldout` feature_z `z_mh`; `h` = the single linear head (exact replay). β = `fcigl_strength` (Phase-2
  first gate = 0.1). Penalizes how much the prediction on pseudo-target subjects *changes* when the
  meta_train subject direction is removed → a direct source-episodic reliance-control objective.

## Firewall (asserted in `tests/test_metacmi_scaffold.py`, 8 CPU tests)
- the episodic partition covers **exactly** the source-train subjects, disjoint, both sides ≥1;
- the `meta_train` projector is a **pure function of the meta_train rows** — corrupting meta_heldout/target
  rows leaves `S`,`P` unchanged;
- a monkeypatch spy proves `_feature_subject_projector` is only ever fit on a **strict subset of source
  subjects** (meta_train) during training — never all-source / meta_heldout / target;
- the SymKL penalty is **active** (β>0 changes the learned model; the term is logged, finite, ≥0);
- deterministic under a fixed seed; **fail-closed** (ValueError) on a backbone with no linear head.
`meta_rho`/`meta_train_frac` are only read inside the `uses_meta` branch → the CIGL_69A ERM path is byte-inert.

## Gate wiring (`scripts/run_metacmi_gate.py`)
`--methods metace metacmi_direct_beta0.1` now route through `train_model(..., meta_rho, meta_train_frac)`;
per-fold feature_z `.audit.npz` (head-replay verified) + firewall metadata unchanged. CPU dry-run (both
backbones × both methods) writes rows with `replay_ok=True`.

## Interpretation guardrail (REQUIRED — carries into Phase 2 and any readout)
`EEGConformerMini` is an **internal faithful-minimal** transformer (emb 32, depth 2, 4 heads), **NOT** the
official EEG Conformer / braindecode. Therefore:
- a **positive** MetaCMI signal on `EEGConformerMini` may be stated only as *"CMI helps an audit-compatible
  ConformerMini; validate on the full / official (or equal-param) Conformer next"* — **never** "CMI works on
  Conformer";
- a **negative** result on `EEGConformerMini` does **not** kill the Conformer family (could be Mini's small
  capacity / setup), and must be checked against the full/fuller Conformer preflight (CIGL_69A2) before any
  family-level claim.
Any Conformer-family conclusion requires the **full/official (or equal-param) Conformer** arm (CIGL_69A2), not
Mini alone.

## Held constraints
No GPU launch for Phase 2. No β=0.5 in the first gate. No change to the queued CIGL_69A ERM jobs. No revival
of static-DGCNN CIGL/FCIGL/dCIGL. No external data.
```
```
