# CIGL Phase 3A-K — Fixed-Config Second-Dataset Confirmation (BNCI2015_001)

> **EXPLORATORY externality test — NOT a benchmark / SOTA result.** Replication of ONE fixed source-only
> candidate (`graph_node_010`) on a SECOND MI dataset. No λ grid, no new configs, no edge term, no SOTA.

## Motivation (from Phase 3A-J)

Phase 3A-J confirmed the fixed candidate `graph_node_010` (λ_g=λ_node=0.010, no edge) across **all
BNCI2014_001 LOSO folds** (primary folds 1–8 = 8/8 on every criterion). That is a credible
**single-dataset** method signal. Phase 3A-K asks the externality question:

> Does the SAME fixed `graph_node_010` replicate on a **second** MI dataset (BNCI2015_001) under the same
> source-only, graph/node-only, edge-skipped protocol?

## Why BNCI2015_001, and why the candidate stays fixed

BNCI2015_001 is a separate MI dataset (different subjects, channels, and task structure) that is close
enough to test replication — **binary** Left/Right MI. The candidate is **frozen** (no λ search, no new
configs):

| config | λ_g | λ_node | λ_edge |
|---|---|---|---|
| `erm_fixed` | 0.000 | 0.000 | 0.000 |
| `graph_node_010` | 0.010 | 0.010 | 0.000 |

`graph_node_010` was selected on **BNCI2014_001**, not here, so on BNCI2015_001 there is **no dev fold —
every LOSO fold is a confirmation fold.**

## Loader / preprocessing (finalized after diligence)

BNCI2015_001 is 2-class MI **right_hand vs feet** — not left/right hand — so MOABB's `LeftRightImagery`
rejects it. The loader is fixed (scoped to BNCI2015_001 only; all other datasets unchanged) to use
**`MotorImagery(n_classes=2, events=["right_hand","feet"])`**.

- **Paradigm:** `MotorImagery` (because it is right_hand vs feet, not left/right hand).
- **Primary resample: 128 Hz** — for protocol consistency with the BNCI2014_001 confirmation (every Phase
  3A run used 128 Hz). `notes/preprocessing_decision.md` mentions 250 Hz; that is a **known note, not
  changed here** to avoid introducing a new preprocessing variable into a fixed-protocol externality test.
- **Window: tmin=0.5, tmax=3.5 (kept).** MOABB places epochs at `dataset.interval[0]`; BNCI2015_001 has
  `interval=[0,5]`, so [0.5, 3.5] s lies inside the declared MI interval (same code path as 2a's [2,6]).
  **This phase does NOT run a window-sensitivity sweep** — only a preflight/metadata guard records the
  window vs the declared interval.
- **Class-count guard:** if the loaded labels are not exactly the two classes (right_hand/feet → 2-class),
  the runner stops before training and reports for reviewer re-authorization.
- **If ERM adequacy fails**, the correct next step is **preprocessing/window diagnosis (Decision D)** in a
  *separately authorized* window-sensitivity phase — **do not silently widen or move the window**.

A `--preflight_only` mode (no training, no probes, no silent downloads) verifies the dataset/paradigm/
classes/subject-count/channel-and-time-shape from the local cache and writes a preflight JSON recording
`moabb_paradigm`, `events`, `resample` (+rationale/note), `tmin/tmax`, `dataset_interval`, and
`window_inside_declared_interval`. The real run records the same under `meta.preprocessing`.

## Binary-dataset threshold logic (chance = 0.50)

BNCI2014_001 is 4-class (chance 0.25, floor 0.45); BNCI2015_001 is binary (chance 0.50), so the
adequacy/retention floors are **raised** and the per-seed floor is distinct from the mean floor:

- ERM adequacy: ERM mean source ≥ **0.60** and ≥2/3 seeds ≥ **0.55**, in **≥8/12** folds.
- ERM leakage target exists: ERM graph **or** node clears null in ≥2/3 seeds, in **≥8/12** folds.
- Regularizer effect: `graph_node_010` reduces graph **or** node KL by **≥30%** vs fold/seed-matched ERM
  in ≥2/3 seeds, in **≥7/12** folds.
- Source retention: `graph_node_010` mean source ≥ **0.60** and source drop ≤ **0.02**, in **≥7/12** folds.
- Target guardrail: `target_eval` drop ≤ **0.05** (evaluation-only).

**Class-count guard:** the runner determines `n_classes`/`chance_bacc` from the loaded labels. If the
loaded dataset is **not** binary (`n_classes != 2`), it **fails before training** and asks for reviewer
approval — the binary floors must not be applied to a non-binary set. (Thresholds generalize by fraction:
⌈8/12·n⌉ and ⌈7/12·n⌉, equal to 8 and 7 at n=12.)

## Strict source-only firewall, no edge, no λ grid

Configs are **fixed** (no selection). Target labels/covariates never touch training/normalization/
probe-fit/audit; `target_eval` is an evaluation-only guardrail (`selection_uses_target_eval=false`).
DGCNN's adjacency is static → **no edge term/audit** (`edge_regularization_used=false`,
`edge_audit_skipped=true`, not faked). Per fold/config/seed: train DGCNN (ERM or graph/node CMI), audit
frozen `graph_z`/`node_z` with **fresh** held-out probes at `n_perm=50`; reductions paired by fold+seed.
A non-default `--dataset` requires `--allow_non_default_dataset`.

## Three-layer verdict (target guardrail is evaluation-only but REQUIRED for Decision A)

Criteria 1–4 are source-side; criterion 5 (target guardrail) is **evaluation-only** — it never touches
training, normalization, probe fitting, config choice, or selection (configs are fixed), but it **is**
part of the final reported verdict, because it decides whether we can call the externality "confirmed
enough for method framing." The runner reports three explicit layers:

- **`source_only_confirmed`** = criteria 1–4 (the source-only replication result).
- **`target_guardrail_pass`** = target_eval drop ≤ 0.05 in **≥ ⌈7/12·n⌉** folds (≥7/12; evaluation-only).
- **`confirmed_with_target_guardrail`** = `source_only_confirmed AND target_guardrail_pass` (the final
  Decision-A condition).

## Decision rules (reviewer-gated)

- **A — `confirmed_with_target_guardrail = true`** (source-only confirmed **and** target guardrail held)
  → cross-dataset method framing may begin.
- **B — partial** — `source_only_confirmed = true` but `target_guardrail_pass = false`, **or** strong
  partial replication → bounded single-dataset method signal (NOT method framing).
- **C — not confirmed** — source-only confirmation fails for reduction/retention reasons → BNCI2014_001-
  only finding.
- **D — ERM baseline unstable** on the second dataset → dataset/backbone diagnosis.

A run whose source-side criteria pass but whose `target_eval` collapses is **B (partial)**, never A.

## Run

```bash
# CPU dry-run (binary synthetic; pipeline + firewall + threshold logic; no GPU):
python scripts/run_cigl_phase3a_dgcnn_gn_second_dataset_confirmation.py --dry_run_synthetic --device cpu --seeds 0 1 --epochs 3 --probe_epochs 5 --n_perm 5

# Real run (after reviewer approval of the dry-run):
sbatch scripts/sbatch_cigl_phase3a_dgcnn_gn_second_dataset_confirmation_bnci2015_001.sh
#  -> --dataset BNCI2015_001 --device cuda --seeds 0 1 2 --epochs 80 --probe_epochs 100 --n_perm 50 --gate_alpha 0.05
```

**Not authorized** (unchanged): λ grid / new configs, edge-CMI / edge regularization, per-edge heads, PyG,
SEED/DEAP, Lee2019_MI (yet), SOTA, CITA/DualPC/Tri-CMI changes. If `n_classes != 2`, **stop for
re-authorization**. Outputs land in `results/cigl/phase3a_dgcnn_gn_second_dataset_confirmation/`
(generated JSON gitignored; the tracked record will be `docs/CIGL_31_...` after the real run).
