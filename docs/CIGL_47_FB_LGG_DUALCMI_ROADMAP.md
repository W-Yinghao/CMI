# CIGL_47 — FB-LGG-DualCMI Roadmap (main-line successor to static DGCNNGraph)

## 0. Decision context (PI gate, 2026-07-01)

Pilot 1 (`CIGL_46`) closed the **static `DGCNNGraph` SOTA track**: across seeds 0/1/2 the dual-CMI
objective did **not** beat ERM on either dataset (see `results/graphdualcmi_pilot/G2_SUMMARY.md`).
Per the PI-precommitted rules, **Rule B + Rule D** fired:

- **Rule B** — BNCI2015 seed-0 dual +2.9pp gain did not survive seeds 0/1/2 (dual −2.4pp mean).
- **Rule D** — BNCI2014 all near-chance with source bAcc = 1.000 (cross-subject transfer failure).

`DGCNNGraph` is retained as a **diagnostic / audit backbone only**. This is consistent with the CIGL
v0.6 manuscript, which scoped static DGCNN as a bounded graph/node leakage-audit backbone with **no**
SOTA / edge-CMI / cross-architecture / beyond-MI claim. The negative SOTA result contradicts nothing in
v0.6; CIGL_45 audit-hardening remains an independent fallback.

**The new main line is `FB-LGG-DualCMI`.** The goal is no longer "make DGCNNGraph work"; it is:

> A stronger EEG graph decoder whose **ERM backbone already has credible cross-subject target bAcc**,
> then add **active** dual-CMI regularization.

Static DGCNN showed that regularizing a weak/overfitting graph backbone just hurts or goes flat. So the
new track fixes three root causes first — **(i) a stronger backbone, (ii) an actually-active decoder
residual, (iii) source-only early stopping to stop catastrophic source memorization** — before any λ/γ
tuning.

**Protocol unchanged.** Non-GPU design + scaffolding + CPU tests first, pushed per-step for PI review.
GPU stays frozen until a fresh run-spec is approved. Target labels may be used only for after-the-fact
`target_eval` metrics. No changes to CITA/DualPC/Tri-CMI.

---

## 1. Why static DGCNNGraph is diagnostic-only (evidence)

| dataset | ERM mean | dual Δ | decoder-only Δ | graphcmi Δ | source bAcc | zero_graph |
|---|---|---|---|---|---|---|
| BNCI2014_001 (.25) | 0.349 | −0.020 | −0.008 | −0.018 | 1.000 | ≈chance |
| BNCI2015_001 (.50) | 0.594 | −0.024 | −0.014 | +0.011 | 1.000 | ≈chance |

- The graph branch **is** load-bearing (zero_graph → chance), so the failure is not "graph ignored" —
  the learned graph features simply do not transfer across subjects.
- `source bAcc = 1.000` everywhere → 300 fixed epochs with no source-domain validation over-memorize
  source.
- The decoder residual was **dormant** (`dec_js_res ≈ 3e-4`, `dec_ce_res ≈ 0`; `[JS−τ]_+` never fired),
  so `I(Y;D|Z)` was never actually exercised — **not falsified, untested**.

---

## 2. FB-LGG-DualCMI architecture spec

New backbone (a **new** model, not a repair of `DGCNNGraph`).

- **Class:** `FBLGGDualCMIBackbone`   **Registry name:** `FBLGGGraph`
- **File:** `cmi/models/fb_lgg_dualcmi.py`
- **Contract:**
  ```python
  forward_graph(x) -> (logits, graph_z, node_z, edge_logits_or_none, fused_z)
  ```
  Unlike `DGCNNGraph`, `fused_z` **may differ** from `graph_z` (SOTA decoding needs a temporal/CNN
  branch alongside the graph branch). The classifier reads `fused_z`; decoder-CMI acts on `fused_z`.

### 2.1 Temporal stem (filterbank)

Replace the weak single temporal summary with a filterbank-style front end:

```text
Input:  B × C × T
per temporal band / kernel group:
    Conv (Conv1d/Conv2d, learnable multi-kernel-size), Norm (BN/GroupNorm), ELU, dropout,
    windowed log-variance / temporal pooling
concat bands -> node features  (B × C × F_node)
```

First scaffold: learnable temporal kernels at several kernel sizes. Fixed bandpass / Sinc-style filters
are a later refinement, not required for the first scaffold.

### 2.2 Local graph stage (channel groups)

Channel-name-aware electrode grouping with an index-based fallback (must not crash if channel names
differ or are absent):

```text
groups (min): frontal / central / parietal / occipital / temporal / motor-central
per group: local node projection -> local graph conv/attention -> pool to a group token
```

For BNCI2014_001 (22 ch) and BNCI2015_001 (13 ch) the group builder reads montage/channel names when
available and falls back to contiguous index partitions otherwise.

### 2.3 Global graph stage

```text
tokens (group tokens or electrode tokens) -> global graph with a SHARED learned A0
                                          -> ChebConv / GCN / graph attention
                                          -> global readout = graph_z
```

**No free dynamic adjacency in v1.** Free per-sample `A(x)` was the v0.6 failure mode (subject
fingerprint). If reintroduced later it must be a *constrained residual* `A(x) = A0 + ΔA(x)` with
explicit Frobenius / L1 / symmetry / anatomical-prior penalties. **Out of scope for the first scaffold.**

### 2.4 Fusion + ablations

```text
fused_z = gate([graph_z, temporal_z])
logits  = classifier(fused_z)
```

Expose ablations so the first GPU gate can verify the graph branch contributes (no hidden CNN-only
bypass making "graph" look good):

```python
ablate(x, mode="zero_graph")      # graph_z -> 0
ablate(x, mode="permute_nodes")   # shuffle node/channel identity
ablate(x, mode="zero_temporal")   # temporal_z -> 0  (new mode)
```

---

## 3. Objective: keep dual-CMI, make the decoder residual ACTIVE

Same objective family as CIGL_46, but the decoder term must be exercised and instrumented:

```text
L = CE(Y|fused_z)
  + λ_g · Ĩ(graph_z ; D | Y)                       # encoder graph CMI (GLS, reference=marginal)
  + λ_n · (1/C) Σ_v Ĩ(Z_v ; D | Y)                 # encoder node CMI (GLS)
  + λ_e · Ĩ(A ; D | Y)                             # encoder edge CMI (only if per-sample edge exists)
  + γ_dec · [ dec_scale · JS(h(Y|fused_z,D), h0(Y|fused_z,D)) − τ ]_+   # ACTIVE decoder residual
  + Ω_graph
```

### 3.1 Config grammar (backward-compatible)

```text
graphdualpc:<λg>:<λnode>:<λedge>:<γdec>[:<dec_scale>]
```

- `graphdualpc:0.010:0.010:0.000:0.100`            → `dec_scale = 1.0` (unchanged, back-compat)
- `graphdualpc:0.010:0.010:0.000:0.100:<s>`        → `dec_scale = s`

Training term:
```python
r_dec_raw = post_dec.dec_js_residual(fused_z, db, weight=wb)
r_dec     = dec_scale * r_dec_raw
loss     += gamma_dec * warm * relu(r_dec - dec_margin)
```

### 3.2 Activation diagnostics (must be emitted every run)

```json
{ "dec_js_res_raw": ..., "dec_js_res_scaled": ..., "dec_gate_active_frac": ...,
  "loss_ce": ..., "loss_graph": ..., "loss_node": ..., "loss_dec": ...,
  "loss_dec_over_ce": ... }
```

**Activation target:** `loss_dec_over_ce ≈ 1%–10%`.
`<0.1%` ⇒ dormant (as in CIGL_46); `>20%` ⇒ decoder term dominating. Concrete `γ_dec`/`dec_scale`
values are chosen **after** a CPU tiny run reports `loss_dec_over_ce`, not guessed.

---

## 4. Split encoder/decoder posterior heads

Because `fused_z ≠ graph_z`, the CIGL_46 single shared `post` object is no longer clean. Use separate,
independently-parameterized heads:

```text
post_graph : q(D | graph_z, Y)            # encoder graph CMI
post_node  : q(D | node_z, e_v, Y)        # encoder node CMI (node-id embedding)
post_dec   : q(Y | fused_z), h0(Y | fused_z, D), h(Y | fused_z, D)   # decoder residual
```

```python
r_graph = post_graph.reg(graph_z, yb, weight=wb, reference="marginal")
r_node  = post_node.reg(node_z, yb, weight=wb)
r_dec   = post_dec.dec_js_residual(fused_z, db, weight=wb)
```

**Never** silently reuse one `post` object across `graph_z` and `fused_z`. The distinct-`fused_z`
NotImplementedError in the CIGL_46 trainer is *replaced* by this split (P3-C).

---

## 5. Source-only early stopping (fix source bAcc = 1.000)

`source bAcc = 1.000` at 300 fixed epochs is catastrophic source memorization. Add source-only
validation and best-epoch restore. **Target labels are never used for selection.**

```text
outer target subject: locked (LOSO)
source subjects: split into source-train / source-val BY SUBJECT
early-stop metric: source-val balanced accuracy   (tie-break: lower source-val CE)
best-epoch restore: reload best-source-val weights before target_eval
```

First scaffold uses a deterministic split (upgrade to inner LOSO later):
```text
source_val_subject = first source subject after the target in seeded subject order
```
The split must never leak the target subject and never touch target labels before final eval.

Run metadata:
```json
{ "source_val_subjects": [...], "best_epoch": ..., "best_source_val_bacc": ...,
  "final_train_source_bacc": ..., "final_val_source_bacc": ... }
```

---

## 6. Non-GPU task plan (P3-A … P3-E)

| step | deliverable | tests |
|---|---|---|
| **P3-A** | this roadmap `docs/CIGL_47_FB_LGG_DUALCMI_ROADMAP.md` | — |
| **P3-B** | `FBLGGGraph` backbone: `forward_graph` 5-tuple (distinct `fused_z`), filterbank stem, local/global graph, fusion, channel-group builder (name-aware + index fallback), `ablate(zero_graph/permute_nodes/zero_temporal)`; register in `build_backbone` | 2a=22ch & 2015=13ch shapes; 5-tuple; zero_graph/zero_temporal/permute_nodes each change logits |
| **P3-C** | graphdualpc head split: `post_graph`/`post_dec`(+`node_post`/`edge_post`); distinct `fused_z` no longer raises; separate params | distinct `fused_z` runs; `post_graph`≠`post_dec` params; decoder diagnostics finite |
| **P3-D** | `dec_scale` grammar (backward-compat) + activation diagnostics | grammar parses w/ and w/o `dec_scale`; `dec_js_res_raw/scaled`, `loss_dec`, `loss_dec_over_ce`, `dec_gate_active_frac` present & finite |
| **P3-E** | source-only early stopping + best-epoch restore + metadata | firewall still passes (no target labels pre-eval); `source_val_subjects`/`best_epoch` recorded |

All CPU-only. Each step is committed and pushed for PI review. **No GPU** until a fresh run-spec is
approved.

---

## 7. First FB-LGG GPU pilot — PLACEHOLDER (NOT approved; submit separately)

```text
Datasets:  BNCI2014_001 target_indices 0,1   |   BNCI2015_001 target_indices 0,9
Seeds:     0 only (first)
Configs:   FBLGGGraph erm:0
           FBLGGGraph graphdualpc <decoder-active>
           FBLGGGraph graphdualpc <dual-active>
           FBLGGGraph graphcmi <small-λ>
Reference: CIGL_46 G2 (static DGCNNGraph) — do NOT rerun DGCNNGraph.
```

**First gate (to be refined in the run-spec):**
```text
FBLGG ERM beats DGCNNGraph ERM meaningfully (especially BNCI2014_001, where DGCNN was near-chance),
AND graphdualpc does not harm source-val / target,
AND decoder residual is active (loss_dec_over_ce in ~1–10%),
AND the graph branch contributes (zero_graph ablation drops materially).
```

If FBLGG ERM does **not** clear the cross-subject-transfer bar, the objective work is premature — fix
the backbone/early-stopping first. Detailed GPU spec (epochs/bs/lr/walltime/abort criteria) comes later
under the standard run-spec template.
