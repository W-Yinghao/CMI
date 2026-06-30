# CIGL Phase 3A-G — Graph Backbone Redesign Results (BNCI2014_001, fold-0)

> **EXPLORATORY diagnostic — NOT a benchmark / SOTA result.** One dataset, one LOSO fold, 3 seeds,
> source-only, **ERM only (no CMI regularization)**. Tests whether a graph-compatible backbone can learn
> the fold *through the graph path*; it is not a method claim.

## 1–6. Run provenance

```bash
sbatch scripts/sbatch_cigl_phase3a_graph_backbone_redesign_bnci001.sh
# -> python scripts/run_cigl_phase3a_graph_backbone_redesign.py --dataset BNCI2014_001 --device cuda --fold 0 \
#      --seeds 0 1 2 --epochs 80 --probe_epochs 100 --leak_n_perm 10
```

| field | value |
|---|---|
| SLURM job id | **876507** |
| partition / node | multi-partition `A100,V100,V100-32GB,A40` (default QOS) → scheduled on **node09** (after a `QOSMaxGRESPerUser` queue wait) |
| runtime | ~10–15 min compute (sacct accounting unavailable; log mtime 10:35) |
| branch / commit_hash | `project/cigl-phase3a-graph-backbone-redesign` / `58a0a6511e97104ae0c1566dae7e3588523891ee` |
| config_hash | `0f92756c94d3` |
| environment | conda `eeg2025`, torch 2.6.0+cu124 (CUDA) |
| dataset / fold / held-out target | BNCI2014_001 / fold-0 / **subject 1** (never used in training/extraction/probe) |
| source subjects | 2–9; enc-train 3232, probe-pool 1376 |
| seeds / classes / chance / floor | 0,1,2 / 4 / **0.25** / success floor **0.45** |

## 7–8. Candidate table (ERM only; per-candidate means over 3 seeds)

`src` = `source_probe` (the **only** success metric). `tgt` = `target_eval`, **evaluation-only**.

| candidate | edge | train bAcc/F1 | **src bAcc/F1** | src per-seed | tgt bAcc/F1 | train−src gap |
|---|---|---|---|---|---|---|
| `shallow_graph_stem` | dynamic | 1.000 / 1.000 | **0.331 / 0.331** | 0.339, 0.318, 0.335 | 0.322 / 0.318 | **+0.669** |
| `eegnet_graph_stem` | dynamic | 0.961 / 0.961 | **0.335 / 0.335** | 0.342, 0.334, 0.328 | 0.307 / 0.305 | **+0.626** |
| `dgcnn_forward_graph_adapter` | static | 0.745 / 0.745 | **0.458 / 0.457** | 0.481, 0.451, 0.442 | 0.431 / 0.425 | +0.287 |

## 9. Per-candidate pass / fail (source-only)

| candidate | src≥0.45 mean | ≥2/3 seeds | train≫chance | forward_graph valid | graph_z/node_z finite+nondegen | graph path used | **PASS** |
|---|---|---|---|---|---|---|---|
| `shallow_graph_stem` | ✗ (0.331) | ✗ (0/3) | ✓ (1.000) | ✓ | ✓ | ✗ | **✗** |
| `eegnet_graph_stem` | ✗ (0.335) | ✗ (0/3) | ✓ (0.961) | ✓ | ✓ | ✗ | **✗** |
| `dgcnn_forward_graph_adapter` | ✓ (0.458) | ✓ (2/3) | ✓ (0.745) | ✓ | ✓ | ✓ | **✓** |

## 10. Graph-usage checks (both deltas, on the source probe-pool)

| candidate | source_probe | zero_graph Δ | permute_nodes Δ | graph_path_used |
|---|---|---|---|---|
| `shallow_graph_stem` | 0.331 | +0.081 | +0.086 | False |
| `eegnet_graph_stem` | 0.335 | +0.085 | +0.083 | False |
| `dgcnn_forward_graph_adapter` | 0.458 | **+0.208** | **+0.211** | **True** |

For DGCNN, zeroing the graph readout (and permuting node content) collapses source toward chance
(0.458 → ~0.25) → logits route through `graph_z`, **no CNN bypass**. The two dynamic backbones show
*small* deltas **only because their source is already at chance** (there is no generalizable source
signal to ablate) — this is **not** evidence of a bypass; it follows from their overfitting.

## 11–12. Edge objects / leakage

- **Dynamic candidates** (`edge_logits_dynamic=true`, per-sample `A(x)` [B,C,C]) — light audit (n_perm=10,
  magnitude sanity, **p not required <0.05**): `shallow_graph_stem` graph/node/edge KL **1.02 / 0.62 /
  0.77**; `eegnet_graph_stem` **0.57 / 0.25 / 0.66**. **Caveat:** these are **overfit** models (train
  ≈ 1.0, source ≈ chance), so — exactly like GraphCMINet — their leakage sits on a non-task-capable
  encoder and is **not** a usable basis for a CIGL method.
- **DGCNN adapter**: `edge_logits_dynamic=false`, `edge_logits=None`. Its adjacency is shared/static, so
  there is **no per-sample edge object and no dynamic-edge leakage claim**; its edge audit is skipped by
  design (graph/node leakage would be the reviewer's n_perm=50 re-audit if this path is chosen).

## 13–14. Selection + firewall flags

- **`selected_successful_graph_backbones` = `['dgcnn_forward_graph_adapter']`** (source-only).
  `any_graph_backbone_succeeds=true`, `dynamic_edge_backbone_succeeds=false`, `only_static_adapter_succeeds=true`.
- `used_target_labels_for_training=false`, `used_target_labels_for_selection=false`,
  `used_target_covariates=false`, `target_eval_is_evaluation_only=true`,
  `graph_backbone_selection_uses_target_eval=false`, `cmi_regularization_used=false`.

## 15. Recommended decision — **Decision B** *(pending reviewer)*

**Decision B — only the static `dgcnn_forward_graph_adapter` succeeds; dynamic-edge CIGL is not
supported.** The DGCNN adapter is a valid task-capable graph backbone (source 0.458, ≥2/3 seeds, graph
path used — zero_graph/permute_nodes deltas +0.208/+0.211, no bypass) — but its adjacency is **static**,
so it exposes **no per-sample edge object**. Both **dynamic**-adjacency backbones **fail by
overfitting**: train 0.96–1.0 with source at chance and a +0.63–0.67 gap, *independent of the temporal
stem* (ShallowConvNet **and** EEGNet behave the same).

The current dynamic-adjacency designs overfit and carry strong leakage; this is **consistent with**
per-sample adjacency acting as a subject-fingerprint channel, but **this run does not causally isolate
`A(x)` as the only source of memorization** — it could equally be the *combination* of dynamic adjacency
+ flatten readout + data size + regularization scale. So we do **not** claim dynamic edge is
task-critical (and certainly not the opposite-of-helpful in a causal sense); we only observe that the
current dynamic designs do not generalize on this fold.

**Next authorized step would be (reviewer-gated): a graph/node CIGL path on a task-capable graph
backbone — not edge-CMI.** Two concrete options for the reviewer to choose between: (a) pursue
graph/node leakage on the **DGCNN adapter** (static adjacency; start with its graph/node Gate-2-style
audit at n_perm=50); or (b) design a **constrained dynamic adjacency** (e.g. low-rank / shared-base +
small regularized per-sample perturbation) so a dynamic-edge backbone can *generalize* — a new design
question, **not** authorized here. **The CIGL regularizer remains NOT authorized**, and no full LOSO /
SEED / λ-grid / SOTA is warranted until a task-capable graph backbone with verified leakage is in hand.
Generated per-candidate JSON are gitignored; this doc is the tracked record.

---

*Note (lettering): the runner's internal verdict string labels this same outcome "C"
(only-static-adapter-passes); mapped to the reviewer's decision-rule lettering it is **Decision B**. The
reviewer's lettering is authoritative throughout this doc.*
