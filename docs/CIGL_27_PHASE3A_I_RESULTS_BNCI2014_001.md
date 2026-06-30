# CIGL Phase 3A-I — DGCNN Graph/Node CMI Regularizer Pilot Results (BNCI2014_001, fold-0)

> **EXPLORATORY pilot — NOT a benchmark / SOTA result.** One dataset, one LOSO fold, 3 seeds,
> source-only, **graph/node CMI only (no edge term)**. A controllability probe on a **modest** baseline;
> a pilot pass here is a *candidate for multi-fold confirmation*, not a method win.

## 1–6. Run provenance

```bash
sbatch scripts/sbatch_cigl_phase3a_dgcnn_gn_regularizer_pilot_bnci001.sh
# -> python scripts/run_cigl_phase3a_dgcnn_gn_regularizer_pilot.py --dataset BNCI2014_001 --device cuda \
#      --fold 0 --seeds 0 1 2 --epochs 80 --probe_epochs 100 --n_perm 20 --n_perm_confirm 50 --gate_alpha 0.05
```

| field | value |
|---|---|
| SLURM job id | **876887** |
| partition / node | multi-partition `A100,V100,V100-32GB,A40` (default QOS) → **node09** |
| runtime | ~35–45 min (8 configs × 3 seeds + n_perm=20 audits + n_perm=50 confirmation; log mtime 13:48) |
| branch / commit_hash | `project/cigl-phase3a-dgcnn-gn-regularizer-pilot` / `bb2858ae33608051a7deb9d355e7f0ea1d6ee7a2` |
| config_hash | `107640bdc446` |
| environment | conda `eeg2025`, torch 2.6.0+cu124 (CUDA) |
| dataset / fold / held-out target | BNCI2014_001 / fold-0 / **subject 1** |
| candidate | `dgcnn_forward_graph_adapter` (static adjacency; **no edge term**) |
| seeds / classes / chance / floor | 0,1,2 / 4 / 0.25 / 0.45 · n_perm=20 (pilot), 50 (confirm) · gate_alpha=0.05 |

## 7. Config table (graph/node only — no edge λ)

| config | λ_g | λ_node | config | λ_g | λ_node |
|---|---|---|---|---|---|
| `erm_fixed` | 0.000 | 0.000 | `graph_003` | 0.003 | 0.000 |
| `graph_001` | 0.001 | 0.000 | `node_003` | 0.000 | 0.003 |
| `node_001` | 0.000 | 0.001 | `graph_node_003` | 0.003 | 0.003 |
| `graph_node_001` | 0.001 | 0.001 | `graph_node_010` | 0.010 | 0.010 |

## 8. ERM_fixed reproduction check

mean source 0.458; per-seed **0.481 / 0.451 / 0.442**; **2/3 seeds ≥ 0.45** → `erm_reproduces=true`
(reproduces the Phase 3A-H DGCNN baseline). Note seed2 (0.442) dips below the floor — the baseline is thin.

## 9–11. Per-config table (source / target / leakage; drops & reductions vs `erm_fixed`)

`tgt` = `target_eval` (evaluation-only). Reduction% = (erm KL − cfg KL)/erm KL; `Ns` = seeds with ≥30%
reduction. `gc`/`nc` = graph/node `clears_null` seed counts.

| config | source bAcc (drop) | tgt bAcc (drop) | graph KL (red%, Ns) | node KL (red%, Ns) | gc / nc |
|---|---|---|---|---|---|
| `erm_fixed` | 0.458 (—) | 0.431 (—) | 1.261 (—) | 0.521 (—) | 3 / 3 |
| `graph_001` | 0.478 (−0.021) | 0.446 (−0.014) | 1.324 (−5%, 0) | 0.538 (−3%, 0) | 3 / 3 |
| `node_001` | 0.478 (−0.020) | 0.447 (−0.016) | 1.333 (−6%, 0) | 0.528 (−1%, 0) | 3 / 3 |
| `graph_node_001` | 0.477 (−0.020) | 0.447 (−0.016) | 1.319 (−5%, 0) | 0.526 (−1%, 0) | 3 / 3 |
| `graph_003` | 0.472 (−0.014) | 0.448 (−0.017) | 1.275 (−1%, 0) | 0.528 (−1%, 0) | 3 / 3 |
| `node_003` | 0.477 (−0.019) | 0.444 (−0.013) | 1.318 (−5%, 0) | 0.510 (2%, 0) | 3 / 3 |
| `graph_node_003` | 0.471 (−0.013) | 0.446 (−0.014) | 1.251 (1%, 0) | 0.497 (5%, 0) | 3 / 3 |
| **`graph_node_010`** | **0.469 (−0.011)** | **0.447 (−0.016)** | **0.658 (48%, 3)** | **0.304 (42%, 3)** | **3 / 3** |

(Negative source/target "drop" = the config slightly *exceeds* `erm_fixed`; no task cost.)

## 12. Graph-usage deltas (all configs)

`graph_path_used = true` for every config (zero_graph and permute_nodes both collapse source toward
chance; no CNN bypass). For `graph_node_010`: zero_graph and permute_nodes drops remain ≈ +0.2.

## 13. Confirmation (n_perm=50) — `erm_fixed` and `graph_node_010` (source-only selected)

| config | graph clears | graph KL | node clears | node KL |
|---|---|---|---|---|
| `erm_fixed` | 3/3 | 1.261 | 3/3 | 0.521 |
| `graph_node_010` | 3/3 | **0.663** | 3/3 | **0.305** |

Per-seed (p=0.020 throughout — the n_perm=50 floor): `erm_fixed` graph kl 1.272/1.265/1.246 (perm
~0.16), node 0.518/0.502/0.544 (perm ~0.034); `graph_node_010` graph kl 0.734/0.706/0.549 (perm ~0.11),
node 0.308/0.321/0.284 (perm ~0.023); source 0.461/0.471/0.475. The **~47% graph / ~41% node reduction
is confirmed at n_perm=50**, and the reduced leakage **still clears the null** (reduced, not eliminated).

## 14. Source-only selection

`source_only_reducers = ['graph_node_010']`; `best_pareto = graph_node_010`; `best_graph_node =
graph_node_010`; `confirmation_labels = ['erm_fixed', 'graph_node_010']`;
`confirmation_label_selection_uses_target_eval = false`. (Selection used `source_probe` + leakage only.)

## 15. Final target-retention verdict (reported only)

`final_target_retaining_reducers = ['graph_node_010']` (target drop −0.016 ≤ 0.05). `target_eval` was used
**only** for this reported verdict, never for selection or confirmation-label choice.

## 16. Edge — explicitly skipped (not faked)

`edge_regularization_used=false`, `edge_logits_dynamic=false`, `edge_audit_skipped=true`,
`edge_skip_reason="static/shared adjacency: edge_logits=None; no per-sample edge object"`.

## 17. Firewall flags

`used_target_labels_for_training=false`, `used_target_labels_for_selection=false`,
`used_target_covariates=false`, `target_eval_is_evaluation_only=true`, `selection_uses_target_eval=false`,
`confirmation_label_selection_uses_target_eval=false`.

## 18. Recommended decision — **A (pilot pass → multi-fold confirmation)** *(pending reviewer)*

Against the strict method-pass gate, **`graph_node_010` passes all six criteria**: (1) `erm_fixed`
reproduces (0.458, 2/3 seeds); (2) ≥30% reduction — graph **48%** / node **42%** — in **3/3** seeds;
(3) source retained (0.469 ≥ 0.45, 3/3 seeds; drop −0.011 ≤ 0.02); (4) target retained (drop −0.016 ≤
0.05); (5) **confirmed at n_perm=50** (reduction holds; still clears null); (6) fresh held-out audit
probes. None of the disqualifiers apply (source ≥ 0.45, source drop ≤ 0.02, **both** source and target
improved — not target-only — and confirmation passed). So this is **Decision A: the graph/node regularizer
reduces verified leakage without damaging the modest task → a candidate for multi-fold confirmation.**
This is the **first** time in the project that label-conditional graph/node leakage is reduced **without**
task collapse (Phase 3A reduced GraphCMINet leakage ~99.9% but destroyed the task; here ~42–48% reduction
costs **no** task).

**Honest caveats — this is a pilot pass, NOT a method win:**
- **Only the top of the λ ladder works.** λ ≤ 0.003 produce 0–5% reduction (0 seeds ≥30%); the entire
  effect is `graph_node_010` (λ=0.010), at the **edge** of the authorized ladder. A larger λ is **not**
  authorized and might either reduce more or start costing task — unknown.
- **Partial controllability, not erasure.** Leakage is reduced ~42–48% but **still clears the null** 3/3
  at n_perm=50 (gKL 0.663, nKL 0.305 remain highly significant). The fingerprint is dented, not removed.
- **Thin baseline.** Absolute task is ~0.458–0.469, barely above the 0.45 floor (erm seed2 = 0.442 < floor).
  The "no task cost" is real but on a weak decoder; this is not a strong-accuracy regime.
- **One fold, one dataset, 3 seeds.** No multi-fold/LOSO/SEED here. The Decision-A meaning is exactly
  "candidate for **multi-fold confirmation**" — replication is required before any method claim, and
  **no SOTA/benchmark claim is made**.

**Next authorized step would be (reviewer-gated): multi-fold confirmation of `graph_node_010`** (the same
graph/node-only, source-only, edge-skipped protocol across folds) before any method framing. The CIGL
**edge**-CMI path stays out of scope; no full LOSO/SEED/λ-grid/SOTA without explicit authorization.
Generated per-config JSON are gitignored; this doc is the tracked record.
