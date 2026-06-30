# CIGL Phase 3A-J — Fixed-Config Multi-Fold Confirmation (BNCI2014_001)

> **EXPLORATORY confirmation — NOT a benchmark / SOTA result.** A replication test of ONE fixed
> source-only candidate (`graph_node_010`) across BNCI2014_001 LOSO folds. No λ grid, no new configs, no
> edge term, no second dataset, no SOTA framing.

## Motivation (from Phase 3A-I)

Phase 3A-I passed as a single-fold (fold-0) pilot: the fixed candidate `graph_node_010`
(λ_g=λ_node=0.010, no edge) reduced DGCNN graph/node leakage ~42–48% (confirmed at n_perm=50) **without**
source/target task collapse. That is a positive signal on **one** fold. Phase 3A-J asks the only
scientifically meaningful next question: **does that same fixed candidate replicate across folds, or was
it a fold-0 artifact?**

## Why the candidate is fixed, and why fold-0 is "dev"

`graph_node_010` was **selected** on fold-0 (Phase 3A-I), so fold-0 is a **development fold** and is **not
independent confirmation**. The candidate is now **frozen** — no λ search, no new configs:

| config | λ_g | λ_node | λ_edge |
|---|---|---|---|
| `erm_fixed` | 0.000 | 0.000 | 0.000 |
| `graph_node_010` | 0.010 | 0.010 | 0.000 |

The **primary confirmation set is folds 1–8**; the main decision is based on folds 1–8 (not all folds
pooled). Reporting groups: `fold0_dev` (separate), `folds1_8_confirmation` (primary), and
`all_folds_descriptive` (includes fold-0, descriptive only).

## Strict source-only firewall, no edge

Configs are **fixed** (no selection step at all). Target labels/covariates never touch training, early
stopping, normalization, probe fitting, or the audit; `target_eval` is an **evaluation-only guardrail**
(`selection_uses_target_eval=false`). DGCNN's adjacency is static → **no edge term, no edge audit**
(`edge_regularization_used=false`, `edge_audit_skipped=true`, reason recorded; never faked). Per
fold/config/seed: train DGCNN (ERM or graph/node CMI), then audit frozen `graph_z`/`node_z` with **fresh**
held-out probes at `n_perm=50`; reductions are paired by **fold and seed**:
`reduction = (ERM_KL − graph_node_010_KL) / ERM_KL`.

## Pass / fail criteria (primary = folds 1–8; 6/8 and 5/8 at n=8, generalized by fraction)

1. **ERM baseline adequacy** — `erm_fixed` source mean ≥ 0.45 and ≥2/3 seeds ≥ 0.45, in **≥6/8** folds.
2. **ERM leakage target exists** — `erm_fixed` graph **or** node leakage clears null in ≥2/3 seeds, in
   **≥6/8** folds.
3. **Regularizer effect** — `graph_node_010` reduces graph **or** node KL by **≥30%** vs fold/seed-matched
   `erm_fixed` in ≥2/3 seeds, in **≥5/8** folds.
4. **Source task retention** — `graph_node_010` source mean ≥ 0.45 and source drop ≤ **0.02** vs
   `erm_fixed`, in **≥5/8** folds.
5. **Target task guardrail** — `target_eval` drop ≤ **0.05** vs `erm_fixed`, evaluation-only (a guardrail,
   **not** a selection criterion).
6. **Source-only firewall** — target labels/covariates do not affect training/selection/normalization/
   probe-fit/audit.
7. **Edge absent** — `edge_regularization_used=false`, `edge_audit_skipped=true`, no edge-CMI claim.

`confirmed` = criteria 1–4 all hold on folds 1–8. **If fold-0 passes but folds 1–8 do not, the result is
NOT confirmed.**

## Decision rules (reviewer-gated)

- **A — confirmed** (criteria 1–4 on folds 1–8) → proceed to method framing or a second-dataset
  confirmation.
- **B — partial** (leakage target exists and some reduction/retention, but below the confirmation
  thresholds) → bounded finding / refine; not a full method claim.
- **C — not confirmed** (no meaningful replicated reduction) → pilot-only; no method claim.
- **D — ERM baseline unstable** (adequacy fails across folds) → return to DGCNN stability diagnosis.

## Run

```bash
# CPU dry-run (pipeline + firewall + dev-fold split; no GPU):
python scripts/run_cigl_phase3a_dgcnn_gn_multifold_confirmation.py --dry_run_synthetic --device cpu --folds 0 1 2 --seeds 0 1 --epochs 3 --probe_epochs 5 --n_perm 5

# Real run (after reviewer approval of the dry-run):
sbatch scripts/sbatch_cigl_phase3a_dgcnn_gn_multifold_confirmation_bnci001.sh
#  -> --folds 0 1 2 3 4 5 6 7 8 --seeds 0 1 2 --epochs 80 --probe_epochs 100 --n_perm 50 --gate_alpha 0.05
```

**Not authorized** (unchanged): λ grid / new configs, edge-CMI / edge regularization, per-edge heads, PyG,
SEED/DEAP, second dataset, SOTA, CITA/DualPC/Tri-CMI changes. Outputs land in
`results/cigl/phase3a_dgcnn_gn_multifold_confirmation/` (generated JSON gitignored; the tracked record
will be `docs/CIGL_29_...` after the real run).
