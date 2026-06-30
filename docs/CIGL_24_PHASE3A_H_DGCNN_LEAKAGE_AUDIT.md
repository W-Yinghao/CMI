# CIGL Phase 3A-H — DGCNN Adapter Graph/Node Leakage Audit (BNCI2014_001, fold-0)

> **EXPLORATORY diagnostic — NOT a benchmark / SOTA result, and NOT a regularizer.** One dataset, one
> LOSO fold, source-only, ERM only. Asks whether the **task-capable** DGCNN adapter's learned graph
> objects carry label-conditional source-domain leakage.

## Motivation (from Phase 3A-G, Decision B)

Phase 3A-G found that **only** `dgcnn_forward_graph_adapter` learns BNCI2014_001 fold-0 as a graph
backbone (source 0.458, ≥2/3 seeds, graph path used); the dynamic-edge stems overfit. So the only
task-capable graph backbone in hand is the **static-adjacency** DGCNN adapter. Before any regularizer we
must answer the prerequisite diagnostic:

> Do the DGCNN adapter's `graph_z` and `node_z` carry label-conditional domain leakage `I(Z;D|Y)` under
> the same strict source-only fold-0?

## Why graph/node **only** (no edge)

The DGCNN adapter's adjacency is **shared/static** (`edge_logits_dynamic=false`, `edge_logits=None`):
there is **no per-sample edge object**, so an edge-level `I(A;D|Y)` audit is **meaningless** here. The
**edge audit is skipped** with an explicit reason — it is **never faked** with a dummy/broadcast
adjacency. A dedicated `audit_graph_node_objects` wrapper (byte-for-byte the graph/node blocks of the
Phase-2 `audit_graph_objects`, minus edge) is used.

## Strict source-only rule + no regularization

Identical protocol to Phase 3A-S/3A-G (reused verbatim): target subject excluded from training and
`source_probe`; **target labels/covariates never** touch training, early stopping, normalization, model
selection, probe fitting, or the audit; `target_eval` is evaluation-only. **No CMI regularization** is
run (`cmi_regularization_used=false`). The audit's permutation null permutes D **within label on the
probe-training split only and retrains the probe** (the established Gate-2 correctness pattern).

## Audit definition

- **graph**: `I(Z_g;D|Y)` proxy — held-out conditional-domain probe `q(D | graph_z, Y)`, leakage =
  `E KL(q ‖ π_y(D))`.
- **node**: `(1/C) Σ_v I(Z_v;D|Y)` proxy — shared probe over flattened (trial, channel) rows, plus a
  per-channel `node_leakage_map`.
- **edge**: **skipped** (`edge_audit_skipped=true`, reason recorded).
- Significance per object per seed: `clears_null = kl_mean > permutation_mean AND permutation_p ≤
  gate_alpha (0.05)`, with `n_perm=50` on the real run.

## Pass / fail criteria

The audit **passes** iff (source-only): (1) DGCNN `source_probe` bAcc ≥ 0.45 mean; (2) ≥ 2/3 seeds
individually ≥ 0.45; (3) graph-usage check passes (zero_graph drop ≥ 0.10 — logits route through the
graph readout, both `zero_graph` and `permute_nodes` deltas reported); (4) `graph_z` **or** `node_z`
clears the null in ≥ 2/3 seeds; (5) if **node** leakage is claimed, the `node_leakage_map` is **stable**
across seeds above a channel-permutation null (`above_random`, non-degenerate); (6) target
labels/covariates were not used for training, selection, probe fitting, normalization, or the audit.

## Next decisions (reviewer-gated)

- **A** — graph/node leakage **exists** (clears null) on the task-capable DGCNN → a **graph/node
  regularizer pilot** on the DGCNN adapter **may be considered** (still reviewer-gated; not edge-CMI).
- **B** — **no** graph/node leakage clears the null → **pause** the CIGL method path; the defensible
  result becomes the diagnostic story (leakage + overfitting in dynamic graph objects, no method).
- **C** — DGCNN **task fails** on rerun (source < 0.45) → the Phase 3A-G DGCNN result is unstable →
  return to backbone diagnosis.

## Run

```bash
# CPU dry-run (pipeline + firewall + edge-skip; no GPU):
python scripts/run_cigl_phase3a_dgcnn_leakage_audit.py --dry_run_synthetic --device cpu --seeds 0 1 --epochs 3 --probe_epochs 5 --n_perm 5

# Real run (after reviewer approval of the dry-run):
sbatch scripts/sbatch_cigl_phase3a_dgcnn_leakage_audit_bnci001.sh
#  -> --dataset BNCI2014_001 --device cuda --fold 0 --seeds 0 1 2 --epochs 80 --probe_epochs 100 --n_perm 50 --gate_alpha 0.05
```

**Not authorized** (unchanged): no CIGL regularizer, no λ sweep, no full LOSO, no SEED/DEAP, no SOTA, no
edge-CMI claim, no faked dynamic edge, no Phase 3B, no CITA/DualPC/Tri-CMI changes, no PyG, no per-edge
neural heads. Outputs land in `results/cigl/phase3a_dgcnn_leakage_audit/` (generated JSON gitignored; the
tracked record will be `docs/CIGL_25_...` after the real run).
