# FSR_51 — Trained-Representation Subject-Scaling Pilot Protocol (Phase 8D; pre-registration)

**Project FSR — Phase 8D.** Pre-registration of a **staged trained-representation** subject-scaling pilot. 8C-1
showed the frozen encoder + full-pool PCA **cannot** test whether subject **diversity** changes the representation
(pairwise-L1 is training/N-independent by construction). 8D adds a **small trainable adapter** so the representation
**can** respond to source-subject count — the only clean way to ask the original question. **CBraMod primary**
(deterministic, stronger 64ch task); CodeBrain exploratory. **No full fine-tuning, no specialist baselines, no new
datasets/channels/target-panel** (STOP). Design-red-teamed before the run; adversarially verified after.

## Primary scientific question
> When a small representation adapter is **trained with more source subjects**, does it change **representation-level
> subject separability** and/or **subject-subspace reliance**, or only task performance (sample-size)?

This is what 8C-1 could not test (frozen). **Not** a leaderboard; **not** full fine-tuning.

## Models / input / arms
- **Input feature:** the frozen **F1 spatial** feature → the **fixed-a-priori PCA d=128** space `z` from 8C (same
  fixed space; PCA fit once on the 90-subject source pool, source-only). The adapter is trained **on top of `z`**.
- **CBraMod primary; CodeBrain exploratory** (run only if cheap / not weak-task).
- **Arms:**
  - **A0 Frozen** — `z` + linear head (= the 8C-1 baseline; no representation training).
  - **A1 Adapter** (PRIMARY trainable-representation arm) — `z' = z + W2·σ(W1·z)` (bottleneck), LayerNorm; adapter +
    linear head trained **jointly** on **source** task labels. Fixed architecture; **bottleneck_dim ∈ {16, 32}**
    (two pre-declared sizes; selected on **source-val**, never target). No other hyperparameter sweep.
  - **A2 Last-block/shallow rep fine-tune** — **optional**, pilot only **if A1 shows a signal** (8D-PASS).
- Audit metrics are recomputed on the **adapted** representation `z'` (A1) vs `z` (A0).

## Data / design (PILOT — not full grid)
PhysioNetMI, 105 analyzable subjects, runs 4/8/12 imagined L/R fist (8C-0 pins), **frozen 15-subject target panel**,
90-subject source pool. **N_source ∈ {2, 8, all}** (pilot). **growing** {2,8,all}; **fixed** {2,8}. **subset_seeds
= 3**, **train_seeds = 2** (adapter random inits). N=all single composition. Target panel + exclusions unchanged.

## Training / selection / firewall
Adapter + head trained by SGD/Adam on **source-train** task labels (fixed epochs/lr, pre-declared; no target-driven
early stopping). **bottleneck_dim selected on source-val bAcc only.** Per-cell **task gate = source-val bAcc ≥ 0.58**
→ L4/L5/L6 interpretable, else `WEAK_TASK_NOT_INTERPRETED`. **No target label** in adapter training, head, bottleneck
selection, PCA, subset selection, or rank. z-score per-trial within-window. `trained_rep_target_label_firewall.json`
logs reads. Target labels enter **only** `target_bacc` and L6.

## Metrics (recomputed on the trained representation)
- **Representation-level L1 (the key):** mean pairwise subject separability (2-way, run-held-out train 4/8 test 12)
  **on `z'`** (A1) vs `z` (A0). Because `z'` is trained with `N_source` subjects, its L1 **can** move with N — the
  test the frozen design lacked. Report A0-vs-A1 × N × condition.
- **Task:** target bAcc, macro-F1, source-val bAcc (gate).
- **L4/L5/L6** (gated): task-head↔subject-subspace alignment; L5 subject-subspace erase vs **variance-matched**
  null (rank `min(8, K(N−1))`, per removed-variance); L6 target consequence. Same rules as 8C-1.

## 8D gates (pre-registered)
- **8D-PASS (any → expand to full grid {2,4,8,16,all}, seeds 10, train_seeds 3):**
  1. A1 **fixed-trials** N=8 beats N=2 target by **≥ +0.02 bAcc** with **no increase** in L5 subject reliance; or
  2. A1 **pairwise-L1 decreases** with N_source while target stable/improves; or
  3. A1 target improves while **L1 stays high but L5 stays null** (subject retained, not a task lever).
- **8D-NULL (STOP Phase 8; no full FT, no baselines):** A1 ≈ A0 frozen — no target gain, no L1 change, no L5 change.
- **8D-RISK (do NOT interpret as scaling):** A1 improves source-val but **hurts target**, or L5 **increases**, or
  results are **seed-dependent** → write as overfitting / unstable adapter.

## Wording
- **Allowed:** "a trained adapter with more source subjects does / does not change representation-level subject
  separability / subject-subspace reliance / target performance."
- **Forbidden:** "foundation encoders become subject-invariant"; "subject diversity does not reduce subject
  information" (still design-limited unless A1 actually moves L1); full-FT / SOTA / leaderboard claims; reading
  8C-1's frozen L1 as a diversity result.

## Outputs (`results/fsr_trained_rep_scaling/`)
```
adapter_architecture_manifest.json   trained_rep_run_manifest.csv
trained_rep_task_performance.csv     trained_rep_pairwise_l1.csv (A0 vs A1 x N x condition)
trained_rep_l4_alignment.csv         trained_rep_l5_replay.csv
trained_rep_l6_consequence.csv       fixed_vs_growing_adapter_summary.csv
trained_rep_target_label_firewall.json  trained_rep_verdict.json
```
`trained_rep_verdict.json`: `{"primary_model":"CBraMod","arms":["A0_frozen","A1_adapter"],
"n_source_grid_pilot":[2,8,"all"],"fixed_trials_grid":[2,8],"subset_seeds":3,"train_seeds":2,
"adapter_signal":"none|target_gain|l1_reduction|reliance_change|mixed|unstable","expand_to_full_grid":null,
"target_labels_used_for_selection":false}`.

## STOP rules
```text
1  full foundation-model fine-tuning / >2 predeclared adapter sizes / post-hoc size sweep.
2  target labels in adapter train / head / bottleneck selection / subset / rank / early stop.
3  CodeBrain-only rescue; specialist baselines; new dataset/channel-map/target-panel.
4  A1 target gain that is seed-dependent (train_seeds disagree) -> report unstable, do not interpret.
5  source-val too small at N=2 to select bottleneck reliably -> report, do not force.
6  adapter training nondeterministic beyond a disclosed tolerance -> pin seeds / report.
```

## After 8D
8D-PASS → expand full grid (return for PM review first). 8D-NULL → **stop Phase 8**; write the frozen+lightly-trained
conclusion as an FSR extension/appendix. PC2 paused; Paper 1 unaffected; **Paper 2 frozen** (8B+8C+8D become a
follow-up axis only if 8D shows a real trained-representation effect).
