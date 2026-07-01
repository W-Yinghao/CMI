# CIGL_47 — FB-LGG-DualCMI Roadmap

Main-line successor to the static `DGCNNGraph` SOTA track.

## 0. Decision context (PI gate, 2026-07-01)

Pilot 1 (`CIGL_46`) closed the static `DGCNNGraph` SOTA track. Across seeds 0/1/2 the dual-CMI
objective did not beat ERM on either dataset. Evidence:
`results/graphdualcmi_pilot/G2_SUMMARY.md` (+ `G2_AGGREGATE.csv`).

Per the PI-precommitted rules, Rule B + Rule D fired:

- Rule B — BNCI2015 seed-0 dual +2.9pp gain did not survive seeds 0/1/2 (dual −2.4pp mean).
- Rule D — BNCI2014 all near-chance with source bAcc = 1.000 (cross-subject transfer failure).

`DGCNNGraph` is retained as a diagnostic / audit backbone only. This is consistent with the CIGL
v0.6 manuscript, which scoped static DGCNN as a bounded graph/node leakage-audit backbone with no
SOTA / edge-CMI / cross-architecture / beyond-MI claim.

The new main line is `FB-LGG-DualCMI`. The goal is no longer "make DGCNNGraph work"; it is:

> A stronger EEG graph decoder whose ERM backbone already has credible cross-subject target bAcc,
> then add active dual-CMI regularization.

Static DGCNN showed that regularizing a weak/overfitting graph backbone just hurts or goes flat. So
the new track fixes three root causes first, before any lambda/gamma tuning:

1. a stronger backbone;
2. an actually-active decoder residual;
3. source-only early stopping (plus dropout) to stop catastrophic source memorization.

Protocol unchanged: non-GPU design + scaffolding + CPU tests first, pushed per-step for review. GPU
stays frozen until a fresh run-spec is approved. Target labels may be used only for after-the-fact
`target_eval` metrics. No changes to CITA/DualPC/Tri-CMI.

## 1. Why static DGCNNGraph is diagnostic-only (evidence)

| dataset            | ERM mean | dual delta | dec-only delta | graphcmi delta | source bAcc | zero_graph |
|--------------------|----------|------------|----------------|----------------|-------------|------------|
| BNCI2014_001 (.25) | 0.349    | -0.020     | -0.008         | -0.018         | 1.000       | ~chance    |
| BNCI2015_001 (.50) | 0.594    | -0.024     | -0.014         | +0.011         | 1.000       | ~chance    |

- The graph branch IS load-bearing (zero_graph -> chance), so the failure is not "graph ignored" —
  the learned graph features simply do not transfer across subjects.
- `source bAcc = 1.000` everywhere -> 300 fixed epochs with no source-domain validation over-memorize
  source.
- The decoder residual was dormant (`dec_js_res ~ 3e-4`; the gate never fired), so `I(Y;D|Z)` was
  never actually exercised — not falsified, untested.

## 2. FB-LGG-DualCMI architecture

New backbone (a new model, not a repair of `DGCNNGraph`).

- Class: `FBLGGDualCMIBackbone`   Registry name: `FBLGGGraph`
- File: `cmi/models/fb_lgg_dualcmi.py`
- Contract (5-tuple, with a DISTINCT `fused_z`):

```text
forward_graph(x) -> (logits, graph_z, node_z, edge_logits_or_none, fused_z)
```

`fused_z` may differ from `graph_z` (SOTA decoding needs a temporal/CNN branch alongside the graph
branch). The classifier reads `fused_z`; the decoder residual acts on `fused_z`.

### 2.1 Temporal stem (filterbank)

Multi-kernel-size temporal filterbank, channel-preserving:

```text
Input:  B x C x T
per band (several kernel sizes):
    Conv (learnable) -> BatchNorm -> square -> windowed avg-pool -> log
concat bands -> node features  (B x C x F_node)
```

Fixed bandpass / Sinc-style filters are a later refinement, not required for the first scaffold.

### 2.2 Local graph stage (channel groups)

Channel-name-aware electrode grouping with a crash-proof index-partition fallback:

```text
groups (min): frontal / central / parietal / occipital / temporal
per group: local node projection -> within-group GCN -> pool to a group token
```

Real 10-20 montages for the MI datasets are wired as presets in the runner (see section 8), so
grouping is name-aware on real electrodes, not a contiguous index partition. If the preset is absent
or the channel count disagrees, the runner warns loudly and falls back to the index partition.

### 2.3 Global graph stage

```text
group tokens -> global GCN with a SHARED learned A0 -> readout = graph_z
```

No free dynamic adjacency in v1. Free per-sample A(x) was the v0.6 failure mode (subject
fingerprint). If reintroduced later it must be a constrained residual A(x) = A0 + dA(x) with explicit
Frobenius / L1 / symmetry / anatomical-prior penalties. Out of scope for the first scaffold.

### 2.4 Fusion + ablations + dropout

```text
fused_z = gate([graph_z, temporal_z])
logits  = classifier(fused_z)
```

Dropout (default 0.25, configurable) is applied to node features, `graph_z`, `temporal_z`, and
`fused_z` — regularization against the source memorization seen in G2 (source bAcc = 1.000).

Ablations expose both branch contributions so the runner can verify neither branch is a hidden bypass:

```text
ablate(x, "zero_graph")      graph_z -> 0
ablate(x, "zero_temporal")   temporal_z -> 0
ablate(x, "permute_nodes")   shuffle node/channel identity
```

With fusion, `zero_graph` no longer collapses to chance — it should DROP materially; a hidden
temporal-only bypass would show a near-zero drop. The runner records
`ablate_zero_graph_target_bacc`, `ablate_zero_temporal_target_bacc`, and
`ablate_permute_nodes_target_bacc` (backbones declare their supported modes via `meta.ablation_modes`).

## 3. Objective: keep dual-CMI, make the decoder residual ACTIVE

Same objective family as CIGL_46, but the decoder term must be exercised and instrumented:

```text
L = CE(Y | fused_z)
  + lambda_g    * I~(graph_z ; D | Y)                          # encoder graph CMI (GLS, ref=marginal)
  + lambda_node * (1/C) sum_v I~(Z_v ; D | Y)                  # encoder node CMI  (GLS)
  + lambda_edge * I~(A ; D | Y)                                # encoder edge CMI  (only if per-sample edge)
  + gamma_dec   * [ dec_scale * JS(h, h0) - dec_margin ]_+     # ACTIVE decoder residual I(Y;D|Z)
  + Omega_graph
```

### 3.1 Config grammar (backward-compatible)

```text
graphdualpc:<lambda_g>:<lambda_node>:<lambda_edge>:<gamma_dec>
graphdualpc:<lambda_g>:<lambda_node>:<lambda_edge>:<gamma_dec>:<dec_scale>
```

- 4-field form: `dec_scale = 1.0` (byte-identical to CIGL_46).
- 5-field form: `dec_scale = <dec_scale>`.

Example: `graphdualpc:0.010:0.010:0.000:0.100:300` sets `dec_scale = 300`.

### 3.2 Activation diagnostics (emitted every graphdualpc run)

```text
dec_js_res_raw        raw JS residual
dec_js_res_scaled     dec_scale * raw
loss_dec              gamma_dec * warm * [dec_scale*JS - dec_margin]_+
loss_dec_over_ce      loss_dec / loss_ce
dec_gate_active_frac  fraction of batches where the gate fired
```

Activation target: `loss_dec_over_ce` in ~1%-10%.
`< 0.1%` => dormant (as in CIGL_46); `> 20%` => decoder term dominating. The concrete `dec_scale` is
chosen AFTER a real-EEG CPU tiny run reports `loss_dec_over_ce` (section 7), not guessed.

## 4. Split encoder/decoder posterior heads

Because `fused_z != graph_z`, the CIGL_46 single shared `post` object is not clean. Separate,
independently-parameterized heads:

```text
post_graph : q(D | graph_z, Y)                         # encoder graph CMI
post_node  : q(D | node_z, e_v, Y)                     # encoder node CMI (node-id embedding)
post_dec   : q(Y | fused_z), h0(Y|fused_z,D), h(Y|fused_z,D)   # decoder residual
```

For DGCNNGraph (z_dec == graph_z) `post_dec` aliases `post`, so that path stays byte-identical to
CIGL_46. An undeclared distinct `fused_z` still fails closed.

## 5. Source-only early stopping (fix source bAcc = 1.000)

`source bAcc = 1.000` at 300 fixed epochs is catastrophic source memorization. Opt-in early stopping,
never touching target labels:

```text
outer target subject : locked (LOSO)
source subjects      : split into source-train / source-val BY SUBJECT
early-stop metric    : source-val balanced accuracy (tie-break: lower source-val CE)
best-epoch restore   : reload best-source-val weights before target_eval
```

First scaffold uses a deterministic seeded held-out source subject (upgradeable to inner LOSO). Run
metadata recorded: `source_val_subjects`, `best_epoch`, `best_source_val_bacc`,
`final_train_source_bacc`, `final_val_source_bacc`.

## 6. Scaffolding status (non-GPU)

| step | deliverable | status |
|------|-------------|--------|
| P3-A | this roadmap | done |
| P3-B | `FBLGGGraph` backbone (5-tuple, ablations, group builder, dropout) | done |
| P3-C | graphdualpc encoder/decoder head split | done |
| P3-D | `dec_scale` grammar + activation diagnostics | done |
| P3-E | source-only early stopping + best-epoch restore | done |
| P3-F | hardening: graphcmi 5-tuple fix, zero_temporal metric, ch_names presets, dropout, docs | done |

All CPU-only, committed and pushed per-step. No GPU until a fresh run-spec is approved.

## 7. Real-EEG CPU calibration (P3-G; before any GPU run-spec)

A CPU tiny run on real EEG (few subjects, few epochs) to confirm the pipeline and pick an active
`dec_scale`. It is NOT a target-accuracy check — only source-side training diagnostics:

```text
FBLGGGraph + graphcmi runs (no 5-tuple crash)
FBLGGGraph + graphdualpc runs; source_val early-stop metadata present
dec_scale sweep -> loss_dec_over_ce enters ~1%-10%
zero_graph / zero_temporal / permute_nodes metrics all present
```

`dec_scale` selection rule (source-side only, no target accuracy):

```text
loss_dec_over_ce < 0.1%   -> dormant, reject
1% - 10%                  -> acceptable candidate
> 20%                     -> too strong (reject unless CE/source not degraded)
```

## 8. First FB-LGG GPU pilot — PLACEHOLDER (NOT approved)

Two staged sub-pilots; submit a run-spec only after P3-F + P3-G.

### Stage F0 — validate the ERM backbone first

```text
Datasets : BNCI2014_001 target_indices 0,1  |  BNCI2015_001 target_indices 0,9
Seed     : 0 only
Backbone : FBLGGGraph
Config   : erm:0
Flags    : --source_val_early_stop
```

Gate: FBLGGGraph ERM must beat DGCNNGraph ERM meaningfully, especially BNCI2014_001 (which was
near-chance). If ERM does not clear the cross-subject bar, fix the backbone/early-stopping BEFORE
adding any dual-CMI.

### Stage F1 — active decoder / dual-CMI (only if F0 passes)

```text
graphcmi <small-lambda>
graphdualpc <decoder-active>     # dec_scale from P3-G calibration
graphdualpc <dual-active>        # dec_scale from P3-G calibration
```

Do NOT rerun DGCNNGraph; CIGL_46 G2 is the reference. Detailed GPU spec (epochs/bs/lr/walltime/abort
criteria) comes later under the standard run-spec template. CIGL_45 audit-hardening remains a fallback.
