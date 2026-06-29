# CIGL Phase 3A-R — Baseline Repair + Gentle-λ Re-pilot

Phase 3A (`docs/CIGL_15`, `docs/CIGL_16`) showed CMI regularization **controls** leakage but **failed**
the task-preserving gate, with a **near-chance** GraphCMINet-ERM baseline (source bAcc ≈ 0.33 on a
4-class task). The bottleneck is now **baseline adequacy**, not leakage control. Phase 3A-R answers, in
order, on BNCI2014_001 fold-0, strictly source-only:

> **A.** Can a small named set of GraphCMINet-ERM variants reach a **non-degenerate** source-only
> baseline? **B.** If so, does a **gentle** CMI micro-ladder (λ ≤ ~0.05) buy a **task-preserving**
> leakage reduction on that baseline?

Runner: `scripts/run_cigl_phase3a_baseline_repair.py`. Launcher:
`scripts/sbatch_cigl_phase3a_baseline_repair_bnci001.sh`.

## Strict rules (unchanged)

- **Source-only.** Held-out target subject excluded from training, feature extraction, probe, and the
  audit. Per-channel z-score (when used) is fitted on **source enc-train only**.
- **Target labels** are used **only** for after-the-fact `target_eval` metrics (`evaluation_only`),
  never for training, early stopping, normalization, **config/baseline selection**, probe fitting, or
  the leakage audit.
- **No λ grid** — Part B is a small named micro-ladder. No full LOSO, no SEED/DEAP, no SOTA.

## Part A — baseline candidates (small, named; one knob each vs `current_default`)

| candidate | change |
|---|---|
| current_default | GraphCMINet defaults (feat=16, hidden=32, hops=2), classbal sampler |
| source_channel_zscore | + per-channel z-score fitted on source enc-train |
| stronger_graphcmi_backbone | feat=32, hidden=64, hops=2 |
| lower_lr_longer | lr=3e-4, epochs=150 |
| no_classbal_sampler | sampler="raw" |
| ce_balance_check | class-balanced CE (`balance=True`) |

Per candidate × seed: train **GraphCMINet-ERM** (λ_g=λ_node=λ_edge=0); record train bAcc/F1,
`source_probe` bAcc/F1, `target_eval` bAcc/F1 (evaluation-only), train−source gap, and a **light** audit
(`n_perm=10`) graph/node/edge KL for characterization.

**Sanity controls:** `overfit_small_source` (tiny balanced source subset → train bAcc should be ≫
chance), `label_shuffle_control` (source labels shuffled → `source_probe` should be ≈ chance).

**Baseline adequacy gate (source-only):** at least one candidate has `source_probe` bAcc **≥ 0.45**
(or **≥ current_default + 0.10**), **and** the controls behave (overfit ≫ chance, shuffle ≈ chance). The
selected baseline is the passing candidate with the highest `source_probe` bAcc (never `target_eval`).

**If the gate FAILS:** STOP — recommend architecture/preprocessing diagnosis. Part B is skipped and CIGL
is **not** claimed as a method.

## Part B — gentle micro-ladder (only if Part A passes)

On the selected baseline config: `erm_fixed (0:0:0)`, `graph_node_003 (0.003:0.003:0)`,
`graph_node_01 (0.01:0.01:0)`, `graph_node_03 (0.03:0.03:0)`, `graph_only_01 (0.01:0:0)`,
`node_only_01 (0:0.01:0)`, `edge_only_03 (0:0:0.03)`, `edge_only_10 (0:0:0.10)`,
`full_01 (0.01:0.01:0.003)`, `full_03 (0.03:0.03:0.01)`. Audit all configs at `n_perm=20`, then a
**confirmation re-audit at `n_perm=50`** of `erm_fixed` + the task-preserving winners (or the best
graph/node reducer if none), with **per-seed records retained** (`confirmation`,
`confirmation_per_seed`, `*_confirm_*_seed*_nperm50.json`; same frozen model, higher permutation power).

**Gentle gate (PASS, a reviewer VERDICT — not a selection):** at least one graph/node-capable config
reduces **graph or node** KL by **≥30%** vs fixed ERM in **≥2/3 seeds**, with **source** bAcc drop
**≤3 pt** AND **target** bAcc drop **≤5 pt** (target_eval enters only this reported verdict, never
config selection for training). The best-baseline and any config *selection* use source-only metrics.

## Decision branches (reviewer decides)

- Baseline repaired **and** a gentle config is task-preserving → revive **graph+node / full CIGL**.
- Only `edge_only` stays task-preserving → narrow to **Edge-CMI**.
- No task-preserving tradeoff at any credible baseline → pivot to a **diagnostic framework**.
- Baseline **not** repairable → architecture/preprocessing diagnosis before any further method claim.

## Acceptance

```bash
pytest -q tests/test_phase3a_baseline_repair.py tests/test_phase3a_runner.py
python scripts/run_cigl_phase3a_baseline_repair.py --dry_run_synthetic --device cpu \
    --seeds 0 1 --epochs 3 --probe_epochs 5 --n_perm 5
# real (GPU/sbatch, after reviewer approval): sbatch scripts/sbatch_cigl_phase3a_baseline_repair_bnci001.sh
```

The runner reports evidence and a gate verdict but makes **no** A/B/C/D decision. No real GPU run until
the dry-run/tests are reported and the reviewer approves.
