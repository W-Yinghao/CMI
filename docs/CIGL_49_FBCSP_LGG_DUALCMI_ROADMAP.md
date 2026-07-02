# CIGL_49 — FBCSP-LGG-DualCMI Roadmap (P5 architecture patch)

## 0. Motivation (from P4 / CIGL_48)

F0 showed FBLGGGraph ERM **fails** BNCI2014 (4-class) cross-subject while **passing** BNCI2015 (binary).
P4's decisive test: on the decodable 2a fold, classical **CSP+LDA 0.483 > DGCNN 0.403 > FBLGG 0.306** —
FBLGG underperforms even a linear CSP because it **omits spatial collapse** (channels stay as graph nodes)
and delegates spatial integration to a graph that is too weak for 4-class discrimination.

**P5 fix: add the missing FBCSP-style spatial-spectral branch** in parallel with the graph branch, fused
by a bounded 3-way gate. Keep the graph branch (load-bearing on 2015). Do NOT replace the graph. Method:
`FBCSP-LGG-DualCMI`; backbone registry name `FBCSPLGGGraph`. Non-GPU scaffold first; **F1/CMI and GPU
remain frozen**.

## 1. Architecture — FBCSPLGGGraph (subclass of FBLGGDualCMIBackbone)

Three branches → one bounded gated fusion → one classifier:

```text
graph_z    : local-global electrode graph (central_strip_v1)        [B, z_dim]        (from FBLGG)
temporal_z : channel-mean temporal readout                          [B, temp_dim]     (from FBLGG)
spatial_z  : FBCSP per-band spatial projection + log-variance        [B, spatial_z_dim] (NEW, P5)
fused_z    : 3-way softmax-gated combination of {graph, temporal, spatial}  [B, fused_z_dim] (distinct)
logits     : head3(fused_z)
```

### 1.1 FBCSP spatial branch (the CSP recipe, learned)

```text
for each temporal band k (learnable temporal conv):
    band signal      : Conv2d(1, n_filt, (1,kern)) -> BN               [B, n_filt, C, T']
    spatial project  : grouped Conv2d(n_filt, n_filt*K, (C,1))         [B, n_filt*K, 1, T']
    log-variance     : log(var over T')                               [B, n_filt*K]
concat bands -> Linear -> spatial_z                                    [B, spatial_z_dim]
```

This is exactly the CSP feature path — per-band **discriminative spatial filters** + **log-variance** —
that P4 showed FBLGG was missing. Regularized by dropout (+ weight decay via the trainer). `max_norm` on
the spatial filters is available but off in v1 (kept minimal; no Riemannian layers).

### 1.2 Bounded 3-way fusion (instrumented)

```text
gp, tp, sp = fuse_g(graph_z), fuse_t(temporal_z), fuse_s(spatial_z)   # each [B, fused_z_dim]
gate       = softmax(gate3([gp; tp; sp]))                             # [B, 3], sums to 1
fused_z    = gate_graph*gp + gate_temporal*tp + gate_spatial*sp
```

Softmax over branches is bounded and sums to 1, so **no branch can unconstrainedly dominate the scale**.
The per-batch gate mean/std for each branch are exposed via `gate_summary(x)` and recorded by the runner
(`gate_graph_mean/std`, `gate_temporal_mean/std`, `gate_spatial_mean/std`) — aggregate only, no per-trial
files.

## 2. Contract (P5-C)

```text
forward_graph(x) -> (logits, graph_z, node_z, edge_logits=None, fused_z)   # 5-tuple, distinct fused_z
forward(x)       -> (logits, fused_z)
backbone.last_aux = {graph_z, temporal_z, spatial_z, fused_z}  (detached, for diagnostics)
backbone.last_gate = [B,3] softmax weights
ablate(x, mode)  -> logits ; modes = zero_graph / zero_temporal / zero_spatial / permute_nodes
meta.ablation_modes = (zero_graph, zero_temporal, zero_spatial, permute_nodes)
```

The runner records `ablate_zero_graph/zero_temporal/zero_spatial/permute_nodes_target_bacc` (the
ablation loop is generic over `meta.ablation_modes`). graphdualpc head-split works (distinct fused_z →
separate `post_dec`). Source-only early stopping and the target-label firewall are unchanged.

## 3. Grouping (P5 requirement)

Keeps `central_strip_v1` (CIGL_48/P3-H). The **graph branch** does topology / local-global relations;
the **spatial branch** does CSP-style discriminative spatial filtering. Both MI datasets resolve the
preset (BNCI2014 → 9 named groups, BNCI2015 → 5) with no index fallback.

## 4. Fold-difficulty context (P5-A)

CSP fold-difficulty map (all 9 BNCI2014 folds): 4/9 subjects are CSP cross-subj decodable (>0.40): subj
1, 3, 8, 9. subj 2 (F0 fold 1) is hard (0.248). See `docs/CIGL_49_FOLD_DIFFICULTY_APPENDIX.md` +
`results/fblgg_f0/BNCI2014_CSP_ALL_FOLDS.csv`. This informs (does not decide) the F0 fold choice.

## 5. Status / what remains frozen

- **Done (this branch):** FBCSPLGGGraph backbone + registry + runner wiring (zero_spatial ablation, gate
  diagnostics, central_strip_v1) + 9 CPU tests + CPU smoke. FBLGGGraph / DGCNNGraph untouched.
- **ERM-only first.** The first FBCSP-LGG GPU gate is ERM (below), submitted for approval, not run.
- **Frozen:** F1 / graphcmi / graphdualpc / dec_scale=300 / λ,γ sweep / edge λ / dynamic adjacency /
  more FB-LGG seeds / GPU. `dec_scale=300` stays the F1 default candidate, unused until a backbone clears
  the 4-class gate.

## 6. Proposed FBCSP-LGG F0 GPU run-spec (NOT approved; submit-only)

```text
Branch / SHA : project/fbcsp-lgg-dualcmi-scaffold @ <commit>
Env          : eeg2025; MNE readable mirror; private HOME per job; CUDA fail-closed
Backbone     : FBCSPLGGGraph      Config: erm:0      Flags: --source_val_early_stop
Datasets     : BNCI2014_001 target_indices {0,1}(+ optionally 7 = decodable subj8)  |  BNCI2015_001 {0,9}
Seeds        : 0 (then 1,2 as the ERM gate seed-stability extension, same as F0)
Train        : epochs 300, bs 64, lr 1e-3, warmup 40, n_inner 2
Gate         : FBCSPLGG ERM must beat FBLGG (2a 0.296) AND DGCNN (2a 0.342) on 2a — ideally reach the CSP
               cross-subj band (~0.40+) on decodable folds; must not regress BNCI2015 (0.627). Report the
               3-way gate means (is spatial_z used on 2a?) + all 4 branch ablations.
```
