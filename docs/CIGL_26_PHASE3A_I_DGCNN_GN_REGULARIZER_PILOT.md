# CIGL Phase 3A-I — DGCNN Graph/Node CMI Regularizer Pilot (BNCI2014_001, fold-0)

> **EXPLORATORY pilot — NOT a benchmark / SOTA result.** One dataset, one LOSO fold, source-only,
> **graph/node CMI only (no edge term)**. Small λ ladder on a **modest** baseline; this is a
> controllability probe, not a method claim.

## Motivation (from Phase 3A-H)

Phase 3A-H established that the **task-capable** static DGCNN adapter (source ~0.458, 2/3 seeds, graph
path used) carries **strong, significant, stable** graph/node leakage (graph KL ~8× / node KL ~15× the
permutation mean; p=0.020 in 3/3 seeds; node-map corr 0.945). That makes graph/node leakage a **valid
regularization target** — but controllability was **not** proven. Phase 3A-I asks the narrow question:

> Can graph/node CMI regularization **reduce** the verified graph/node leakage **without destroying** the
> (already modest) DGCNN task baseline?

## Graph/node only — no edge

The DGCNN adapter's adjacency is **shared/static** (`edge_logits_dynamic=false`, `edge_logits=None`):
**no edge term, no edge audit, no per-edge heads** — `edge_audit_skipped=true`, reason recorded; never
faked. The trainer's `graphcmi` branch is used with `lambda_edge=0` and a guard that skips the edge term
when `edge_logits is None` (GraphCMINet's edge path is unchanged); it fails closed if `lambda_edge≠0` is
requested on an edge-less backbone.

## Strict source-only firewall

Reused verbatim from Phase 3A-H: target subject excluded from training and `source_probe`; **target
labels/covariates never** touch training, early stopping, normalization, config selection,
confirmation-label choice, probe fitting, or the audit. **Selection and confirmation labels are chosen
source-only** (`selection_uses_target_eval=false`, `confirmation_label_selection_uses_target_eval=false`);
`target_eval` enters **only** a final reported retention verdict. Training-time posterior heads drive the
regularizer; **evidence uses fresh held-out audit probes** with retrained within-label permutation nulls
(Step-A heads are discarded for evidence).

## Configs (graph/node only — small λ ladder)

| config | lambda_g | lambda_node |
|---|---|---|
| `erm_fixed` | 0.000 | 0.000 |
| `graph_001` | 0.001 | 0.000 |
| `node_001` | 0.000 | 0.001 |
| `graph_node_001` | 0.001 | 0.001 |
| `graph_003` | 0.003 | 0.000 |
| `node_003` | 0.000 | 0.003 |
| `graph_node_003` | 0.003 | 0.003 |
| `graph_node_010` | 0.010 | 0.010 |

No larger λ and **no edge λ** in this pilot. The DGCNN baseline is modest, so the ladder starts small.

## Audit + confirmation protocol

Per config × seed: train DGCNN adapter (ERM for `erm_fixed`, else `graphcmi` graph/node), then audit
frozen `graph_z`/`node_z` with **fresh** probes (`n_perm=20`), `clears_null = kl_mean > permutation_mean
AND permutation_p ≤ 0.05`. **Source-only Pareto selection** picks `source_only_reducers` (≥30% graph or
node KL reduction vs `erm_fixed` in ≥2/3 seeds AND source drop ≤0.02 AND source ≥0.45), `best_pareto`,
and `best_graph_node`. **Confirmation labels** = `{erm_fixed} ∪ {best_pareto} ∪ {best_graph_node}`
(source-only) are re-audited at **`n_perm_confirm=50`**.

## Pass / fail criteria

A **strong pass** requires: (1) `erm_fixed` reproduces the DGCNN baseline (mean source ≥ 0.45, ≥2/3 seeds);
(2) ≥1 graph/node config reduces graph **or** node KL by **≥30%** vs `erm_fixed` in **≥2/3 seeds**;
(3) source retained — mean source ≥ 0.45 **and** source drop ≤ **0.02** vs `erm_fixed`; (4) target_eval
drop ≤ **0.05** vs `erm_fixed` (evaluation-only); (5) fresh held-out audit probes (Step-A heads not reused
as evidence); (6) `n_perm=50` confirmation for `erm_fixed`, best source-only Pareto, and best graph_node.

- **Borderline** — ≥30% reduction but source falls into **[0.43, 0.45)**: a **tradeoff signal only**, not
  a method pass.
- **Fail** — no config reduces graph/node leakage without source loss → pause / reframe as a
  diagnostic / regularization-target analysis.

## Decision rules (reviewer-gated)

- **A** — leakage reduced AND source+target retained → candidate for multi-fold confirmation (still no
  edge-CMI, no full LOSO/SEED/λ-grid/SOTA).
- **B (tradeoff)** — leakage reduced, source retained, but target/headroom thin → tradeoff signal, no
  method win.
- **C** — no meaningful leakage reduction without source loss → diagnostic framework / redesign.
- **D** — `erm_fixed` fails to reproduce the DGCNN baseline → return to DGCNN stability diagnosis.

## Run

```bash
# CPU dry-run (pipeline + firewall + edge-skip; no GPU):
python scripts/run_cigl_phase3a_dgcnn_gn_regularizer_pilot.py --dry_run_synthetic --device cpu --seeds 0 1 --epochs 3 --probe_epochs 5 --n_perm 5 --n_perm_confirm 5

# Real run (after reviewer approval of the dry-run):
sbatch scripts/sbatch_cigl_phase3a_dgcnn_gn_regularizer_pilot_bnci001.sh
#  -> --seeds 0 1 2 --epochs 80 --probe_epochs 100 --n_perm 20 --n_perm_confirm 50 --gate_alpha 0.05
```

**Not authorized** (unchanged): edge-CMI / edge regularization, per-edge heads, PyG, full LOSO, SEED/DEAP,
large λ-grid, SOTA, CITA/DualPC/Tri-CMI changes. **Warning:** the DGCNN baseline is modest (~0.458; one
seed dips below 0.45), so task headroom is thin — do not declare a method win on a borderline tradeoff.
Outputs land in `results/cigl/phase3a_dgcnn_gn_regularizer_pilot/` (generated JSON gitignored; the tracked
record will be `docs/CIGL_27_...` after the real run).
