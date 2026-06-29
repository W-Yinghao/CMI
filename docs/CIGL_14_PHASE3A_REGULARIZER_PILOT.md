# CIGL Phase 3A — Regularizer-Effect Pilot

Gate-2 established (exploratory, `docs/CIGL_13`) that a source-only GraphCMINet-ERM encoder's graph
objects carry significant label-conditional subject leakage. Phase 3A asks the next question —
**controllability**:

> Can graph / node / edge CMI regularization **reduce** that leakage **without destroying task
> performance**?

This is a **pilot**, not a benchmark: one dataset (BNCI2014_001), one LOSO fold (fold-0, same held-out
subject as Gate-2), a fixed small set of lambda configs, 3 seeds. No full LOSO, no SEED/DEAP, no SOTA,
no λ grid search. If Phase 3A fails, expanding to all folds / other datasets is premature.

Runner: `scripts/run_cigl_phase3a_regularizer_pilot.py`. Launcher: `scripts/sbatch_cigl_phase3a_bnci001.sh`.
Results doc (after the real run): `docs/CIGL_15_PHASE3A_RESULTS_BNCI2014_001.md`.

## Configs (exactly seven; ERM = graphcmi:0:0:0)

| label | graphcmi (λ_g:λ_node:λ_edge) |
|---|---|
| erm | `graphcmi:0.0:0.0:0.0` |
| graph_only | `graphcmi:0.3:0.0:0.0` |
| node_only | `graphcmi:0.0:0.3:0.0` |
| edge_only | `graphcmi:0.0:0.0:0.1` |
| graph_node | `graphcmi:0.3:0.3:0.0` |
| full_cigl | `graphcmi:0.3:0.3:0.1` |
| low_full_cigl | `graphcmi:0.1:0.1:0.03` |

## Strict source-only rule

- Held-out target subject (fold-0) is excluded from encoder training, feature extraction, and the probe.
- Three disjoint source subsets per (config, seed): encoder-train (GraphCMINet), probe-pool (frozen
  features for the audit + held-out source task metric), and within the probe-pool a support-aware
  (Y,D) split into probe-train / probe-val for the domain probe.

## Target-label rule (strict)

Target labels are used **only** for after-the-fact `target_eval` metrics, flagged `evaluation_only`.
They are **never** used for training, early stopping, **config selection**, normalization, probe
fitting, or the leakage audit. The summary records `used_target_labels_for_training=false`,
`used_target_labels_for_selection=false`, `target_labels_used_for="evaluation_only metrics"`.

## Audit rule (fresh probes, not Step-A heads)

Leakage evidence comes from `audit_graph_objects` — **freshly-trained held-out probes** on the frozen
`graph_z / node_z / edge_logits`, with the retrained within-train permutation null
(`clears_null = kl_mean > permutation_mean AND permutation_p <= gate_alpha`, α=0.05). The trainer's
Step-A domain heads are reported separately (`stepA_*` critic-quality diagnostics) and are **not** used
as leakage evidence.

**Two passes:**

- **Pass 1** — all **7** configs × seeds at **`n_perm=20`**. Per-seed JSON saved as
  `<dataset>_fold<fold>_<config>_seed<seed>_nperm20.json`. Reductions named
  `{obj}_pass1_leakage_reduction_vs_erm` (vs the **pass-1 ERM**).
- **Pass 2 (confirmation)** — **ERM, full_cigl, and the best-Pareto config** re-audited at
  **`n_perm=50`**. Because the encoder init is seeded **before** construction, the same `(config,seed)`
  is the **same frozen model** as pass-1, re-audited only at higher permutation power. **Per-seed
  confirmation records are retained** (`<dataset>_fold<fold>_confirm_<config>_seed<seed>_nperm50.json`
  and `summary.confirmation_per_seed[config][seed][obj] = {kl_mean, permutation_mean, permutation_p,
  positive_excess, clears_null, gate_alpha}`). Confirmation reductions are computed **against the
  confirmation ERM** (`{obj}_confirm_leakage_reduction_vs_confirm_erm`), never the pass-1 ERM.

**best-Pareto selection** is computed from **source-only** metrics (graph+node leakage reduction
penalized by source-task drop) — never from `target_eval`. The candidate set excludes **only** ERM, so
**`full_cigl` is eligible** to be `best_pareto_config`.

## Metrics (per config, averaged over seeds)

- `source_probe` balanced accuracy & macro-F1 (held-out source probe-pool).
- `target_eval` balanced accuracy & macro-F1 — **evaluation_only**.
- graph/node/edge `kl_mean`, `permutation_mean`, `permutation_p`, `positive_excess`, `clears_null`.
- graph/node/edge `{obj}_pass1_leakage_reduction_vs_erm` (pass-1) /
  `{obj}_confirm_leakage_reduction_vs_confirm_erm` (confirmation) and `clears_null` counts.
- Step-A graph/node/edge `dom_acc` and losses; `reg_graph/node/edge`.
- Every per-seed record (pass-1 and confirmation) stamps `used_target_labels_for_training=false`,
  `used_target_labels_for_selection=false`, `used_target_covariates=false`,
  `target_eval_is_evaluation_only=true`, plus `dataset`, `fold`, `commit_hash`, `config_hash`, `seed`.

## Pass / fail criteria (reviewer-defined)

PASS if **all** hold:

1. at least one config reduces **graph or node** leakage by **≥30%** vs ERM in **≥2/3 seeds**;
2. `source_probe` balanced-accuracy drop **≤3** points (vs ERM);
3. `target_eval` balanced-accuracy drop **≤5** points (evaluation_only);
4. audit probes are **freshly trained held-out** probes (not Step-A heads);
5. `graph_node` **or** `full_cigl` is a Pareto-improving candidate (task vs leakage).

These thresholds are applied **by the reviewer** from the artifacts; the runner reports evidence and a
proposed next path but makes **no** pass/fail or A/B/C/D decision. Use the **confirmation** (`n_perm=50`)
clears_null counts and `confirm_leakage_reduction_vs_confirm_erm` for the binding read; pass-1 (`n_perm=20`)
is the directional screen.

**Edge term:** stays in the main method only if it materially reduces edge KL without task collapse and
adds value beyond graph/node. ⚠️ Gate-2 found edge **leakage** strong but the edge **map** only weakly
seed-stable — so if edge helps only weakly or destabilizes task, the main method becomes **Graph+Node
CIGL** and edge-CMI moves to ablation/appendix.

### Next-path recommendation (Claude proposes, reviewer decides)

- **A — full CIGL** · **B — graph+node only** · **C — node-only** · **D — diagnostic-only** (if no
  config controls leakage without task collapse).

## Acceptance

```bash
pytest -q tests/test_graphcmi_backbone.py tests/test_graph_regularizers.py tests/test_graph_leakage.py \
         tests/test_probe_splits.py tests/test_graph_map_stability.py tests/test_phase2_real_runner.py \
         tests/test_phase3a_runner.py
python scripts/smoke_graphcmi.py --device cpu
python scripts/smoke_graph_leakage.py --device cpu
python scripts/run_cigl_phase3a_regularizer_pilot.py --dry_run_synthetic --device cpu \
    --seeds 0 1 --n_perm 5 --n_perm_confirm 5 --epochs 3 --probe_epochs 5
# real (GPU/sbatch): sbatch scripts/sbatch_cigl_phase3a_bnci001.sh
# real (GPU/sbatch): sbatch scripts/sbatch_cigl_phase3a_bnci001.sh
```
