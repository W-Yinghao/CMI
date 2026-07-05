# CIGL_68 — Direct-reliance CMI (the training-time R3)

```
Branch project/cigl-direct-reliance-cmi off frozen project/cigl-functional-cmi @ c6f4934 (keeps CIGL_65/67
artifact compatibility). LAST justified method-development probe on the CMI line. Directly optimizes the
counterfactual reliance that every proxy failed to control: prediction BEFORE the source-only subject-subspace
removal must match prediction AFTER.
```

## Why (the gap so far)
- CIGL_65: global measured `I(Z;D|Y)` ↓ but classifier reliance not ↓.
- CIGL_66: residual subject subspace is task-head-aligned (a diagnostic correlate).
- CIGL_67: `fcigl_align` reduces the alignment scalar (sig, both datasets) but **reliance still not ↓** (R3 delta
  ns across seeds 0/1/2). Alignment is diagnostic, not a sufficient causal target.
- CIGL_68: stop optimizing proxies — optimize the **counterfactual reliance itself** (the training-time R3).

## Objective (`cmi/train/trainer.py`, `DCIGL_METHODS = {dcigl_consistency}`)
Same DGCNN adapter + graphcmi graph/node CMI terms (λg=λn=0.010) + the same **source-train-only, detached,
periodically-refreshed** label-conditional subject-subspace projector `P = I − SᵀS` on `graph_z` (k=2; NO target,
NO source-val). Let `logits = head(graph_z)`, `logits_rem = head((I−SᵀS)graph_z)`:

```
L = CE(logits, y) + λg·CIGL_graph + λn·CIGL_node
    + β·SymKL(softmax(logits), softmax(logits_rem))   # prediction BEFORE ≈ AFTER removal (the training-time R3)
    + γ·CE(logits_rem, y)                              # removed rep must still classify
```

**Distinct from the killed `fcigl_removal_aug`:** removal_aug only required `z_removed` to classify; it did **not**
constrain the *original* prediction's dependence on the removed directions. `dcigl_consistency` penalizes the
before-vs-after prediction change directly — this is exactly what R3 measures.

## Pre-registered (NO sweep beyond this)
`β ∈ {0.1, 0.5}`; `γ = 0.5` (fixed); `k = 2`; `λg = λn = 0.010`. Two variants:
`dcigl_consistency_beta0.1`, `dcigl_consistency_beta0.5`. No new η/α, no k sweep, no λ sweep, no stacking, no new
baselines, no architecture change.

## Seed0 gate (this phase)
`scripts/run_cigl_direct_reliance_gate.py` — 2 variants × BNCI2014_001 (9) + BNCI2015_001 (12) = **42 GPU runs**,
full LOSO, seed 0, same adapter/firewall/audit+R3+head-replay export. Reference comparators (ERM, old CIGL, CDAN,
FCIGL-align) **not rerun** — compared against frozen CIGL_65/67.

## Seed0 success criteria (PM-pinned)
- **strong:** `target ≥ CIGL−0.005 AND R3_task_drop_k2 < CIGL−0.01 AND < FCIGL-align AND leakage below ERM AND
  random_ctrl ≈ 0` (and `target > CIGL` = CMI-driven stronger model).
- **functional:** target retained AND R3 task_drop decreases meaningfully AND leakage below ERM → expand best β.
- **fail:** R3 unchanged, or target drops materially, or leakage rebounds near ERM, or random control unstable →
  **stop CMI method search**, freeze the two-level→three-level measurement→reliance gap as the scientific story.

## Engineering tests (`tests/test_dcigl.py`, 6 CPU pass; full regression 227)
registered + trains finite + term logged; **SymKL math** (symmetric, 0 when identical, >0 when different);
**behavioral: on entangled data (subject spuriously carries class), dcigl β>0 makes predictions MORE invariant to
subject-subspace removal than β=0** (the objective works); **dcigl ≠ removal_aug** (lower SymKL under removal);
projector source-only + deterministic; fails closed on non-graph backbone. Synthetic = engineering only.

## Firewall / integrity (same as R2/functional gates)
source-train-only projector (excludes source-val + target); target eval-only; same adapter; head-replay export +
verify per fold; random_subspace R3 control. Stop-and-report on: NaN, replay failure, projector touching
source-val/target, backbone-path divergence, unconsumable artifacts, random control non-zero.
```
```
