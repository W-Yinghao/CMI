# CIGL Phase 3A-G — Task-Capable Graph Backbone Redesign (BNCI2014_001, fold-0)

> **EXPLORATORY diagnostic — NOT a benchmark / SOTA result.** One dataset, one LOSO fold, source-only,
> **ERM only (no CMI regularization)**. The goal is a graph-compatible backbone that *learns the task
> through the graph path*; it is not yet a method claim.

## Motivation (from Phase 3A-S, Decision A)

Phase 3A-S showed the strict source-only fold-0 is **learnable**: EEGNet/ShallowConvNet/DeepConvNet reach
`source_probe` 0.52–0.56 and even DGCNN clears 0.46, while GraphCMINet stays at 0.334. So the bottleneck
is **GraphCMINet's specific design**, not the protocol and not graph learning in general. The most likely
culprit: GraphCMINet's `PowerLayer` collapses the **entire** time axis to one power value per
(channel, filter), discarding the temporal dynamics that ShallowConvNet keeps via windowed pooling.

Phase 3A-G redesigns the **graph-compatible task backbone** (ERM only) so CIGL's leakage objects can
later sit on a backbone that actually learns the task. **No CMI regularizer is run here.**

## Candidate designs (small named set — not a grid)

`known-good temporal stem → per-channel node features → graph propagation → graph readout → logits`

| candidate | temporal stem | adjacency / edge | node identity |
|---|---|---|---|
| `shallow_graph_stem` | ShallowConvNet windowed log-power per channel | **dynamic** per-sample `A(x)`, `edge_logits` exposed | preserved |
| `eegnet_graph_stem` | EEGNet temporal blocks per channel (**no** spatial collapse) | **dynamic** per-sample `A(x)` | preserved |
| `dgcnn_forward_graph_adapter` | existing DGCNN temporal encoder | **shared/static** adjacency → `edge_logits=None`, `edge_logits_dynamic=false` | preserved |

All in `cmi/models/graph_task_backbones.py` (pure torch, **no PyG**, no per-edge neural heads). Each
exposes `forward_graph(x) → (logits, graph_z, node_z, edge_logits_or_none)` and per-arch metadata
(`graph_compatible`, `edge_logits_dynamic`, `node_identity_preserved`). The static DGCNN adapter does
**not** fabricate a dynamic edge object. `dgcnn_forward_graph_adapter.forward` is identical to the
DGCNN that cleared 0.46 in Phase 3A-S, so it is a credible practical lower bound.

## Why ERM-only, and the strict source-only rule

There is no point regularizing leakage on a near-chance backbone (Phase 3A-R). Phase 3A-G only asks
*can a graph-compatible backbone learn the task at all*. Protocol (identical to Phase 3A-S, reused
verbatim): target subject excluded from training and `source_probe`; **target labels are
evaluation-only** (never training/early-stop/normalization/selection); success judged on `source_probe`
only. Summary records `graph_backbone_selection_uses_target_eval=false`, `cmi_regularization_used=false`.

## Graph-usage check (anti-bypass)

To rule out a "graph-compatible" model that secretly solves the task with a CNN bypass while the graph
objects are decorative, the runner ablates on the **source** probe-pool:

- **`zero_graph`** — replace the graph readout `graph_z` with zeros before the classifier; `source_probe`
  must collapse toward chance (logits depend only on the graph readout — there is no parallel path).
- **`permute_nodes`** — permute `node_z` across the batch before readout; `source_probe` must drop
  (the readout uses node content, not a constant).

A candidate counts the graph path as *used* iff `zero_graph` drops `source_probe` by ≥ 0.10. (Local
validation on a learnable synthetic: all three learn to 1.000 and `zero_graph` → exactly chance 0.250.)

## Pass / fail criteria (per candidate; source-only)

A candidate **PASSES** iff: (1) mean `source_probe` bAcc ≥ 0.45; (2) ≥ 2/3 seeds individually ≥ 0.45;
(3) `train` bAcc ≥ chance + 0.10; (4) `target_eval` is evaluation-only (never selection); (5)
`forward_graph` returns valid `graph_z`/`node_z` (finite, nodes = channels); (6) `graph_z`/`node_z`
non-degenerate (std > tol); (7) graph path used (zero_graph drop ≥ 0.10); (8) no CMI regularization.
**Optional**: light leakage audit (`n_perm=10`) on `graph_z`/`node_z`/`edge_logits` for dynamic-edge
candidates only — a magnitude sanity check, **p is not required < 0.05**; skipped for the static adapter
(its adjacency is not a per-sample object).

## Next decisions (reviewer-gated)

- **A** — a **dynamic-edge** graph backbone passes → next is the **repaired-backbone Gate-2 leakage
  audit (n_perm=50)**; still no regularizer until leakage is verified on the repaired backbone.
- **B** — no graph-compatible backbone passes → pause the CIGL method path; keep the diagnostic framework.
- **C** — only the **static DGCNN adapter** passes → pursue a **graph/node** CIGL path, **not** edge-CMI,
  unless a validated dynamic-edge object is introduced.
- **D** — a candidate learns the task but the **graph-usage check fails** → that architecture is invalid
  for CIGL (a bypass); do not proceed with it.

## Run

```bash
# CPU dry-run (pipeline + firewall + graph-usage; no GPU):
python scripts/run_cigl_phase3a_graph_backbone_redesign.py --dry_run_synthetic --device cpu --seeds 0 1 --epochs 3 --leak_n_perm 5

# Real run (after reviewer approval of the dry-run):
sbatch scripts/sbatch_cigl_phase3a_graph_backbone_redesign_bnci001.sh
#  -> --dataset BNCI2014_001 --device cuda --fold 0 --seeds 0 1 2 --epochs 80 --probe_epochs 100 --leak_n_perm 10
```

**Not authorized** (unchanged): no CIGL regularizer, no λ sweep, no full LOSO, no SEED/DEAP, no SOTA
table, no Edge-CMI method claim, no Phase 3B, no diagnostic-only final pivot. Outputs land in
`results/cigl/phase3a_graph_backbone_redesign/` (generated JSON gitignored; the tracked record will be
`docs/CIGL_23_...` after the real run).
