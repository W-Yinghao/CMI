# FSR_51 — Trained-Representation Subject-Scaling Pilot Protocol (Phase 8D; pre-registration, v2 red-teamed)

**Project FSR — Phase 8D.** Pre-registration of a staged trained-representation subject-scaling pilot. 8C-1 showed
the frozen encoder + full-pool PCA **cannot** test whether subject diversity changes the representation (pairwise-L1
is training/N-independent by construction). **v2 incorporates a design red-team (agent a606486f) that found 2
BLOCKERs + 4 MAJORs** — critically, that a task-only residual adapter likely reproduces the 8C-1 null one level up.
All fixed below. CBraMod primary; CodeBrain exploratory. **No full fine-tuning, no baselines, no new
datasets/channels/target-panel** (STOP). Adversarially verified after.

## Primary scientific question
> When a small representation adapter is **trained with more source subjects**, does it change **representation-level
> subject separability** (measured **where training has room to reorganize it**) and/or **subject-subspace
> reliance**, or only task performance (capacity/sample-size)?

## The BL-1 fix — measure L1 where training can move it, and prove the metric CAN move
The residual `z' = z + δ` on a **fixed** PCA `z`, trained on **task labels only**, has **no pressure** to change
subject structure; subject identity passes through and L1-on-`z'` would stay ~0.95 (ceiling) — the same structural
null. Therefore:
- **Primary L1 locus = the bottleneck code `h = σ(W1·z)`** (fully trained, no passthrough) and **δ alone**; L1-on-`z'`
  is reported but **secondary** (passthrough-dominated).
- **Positive control arm A1-SE (mandatory):** an adapter trained with an **auxiliary subject-confusion loss**
  (adversarial subject classifier / subject-invariance penalty) — proves the L1 metric *can* fall below ceiling on
  `h`. **If even A1-SE cannot push L1-on-`h` materially below ceiling, the metric is saturated → all 8D L1 nulls are
  `l1_untestable_metric_saturated`, declared up front, not read as "no diversity effect."**
- **Variance-share precondition:** if `‖δ‖/‖z‖` is below a pre-declared threshold (residual dominates), L1-on-`z'` is
  `design_limited`. Reported per cell.

## Models / input / arms
Input = frozen **F1 spatial** → fixed-a-priori **PCA d=128** `z` (8C, source-only). Adapter on top of `z`.
- **A0 Frozen** — `z` + linear head (8C-1 baseline).
- **A1 Adapter** — `z' = z + W2·σ(W1·z)` + LayerNorm; adapter+head trained on **source** task labels.
- **A1-SE** — A1 + subject-confusion aux loss (**positive control**, not a scaling arm).
- **A2** — last-block pilot, only if A1 shows a signal.
- **bottleneck_dim FIXED = 16 across all N** (MA-2/MN-1: removes capacity confound from the scaling contrast);
  {16,32} source-val selection is a **secondary** analysis only, never the scaling contrast.

## Data / design (PILOT) — capacity & total-trials isolated (MA-2)
PhysioNetMI 105 subj, runs 4/8/12, **frozen 15-target panel**, 90-source pool. **N_source ∈ {2,8,all}** growing;
**{2,8}** fixed. **The fixed arm EQUALIZES total training trials across N=2 and N=8** (equal total; per-subject
shrinks) — pinned in `trained_rep_run_manifest.csv`. **subset_seeds = 3; train_seeds = 5** (MA-3; ≥5 on the primary
fixed cells). N=all single composition. The **diversity contrast is within-A1, fixed-trials, matched-capacity
(dim=16)** — never A1-vs-A0 (which is capacity, not diversity).

## Held-out L1 subjects (MA-1) + firewall
**L1-on-`h`/δ is computed on source subjects HELD OUT from adapter training** (designate adapter-train vs
L1-eval **disjoint** source-subject subsets within each cell; L1 test run 12 also excluded from adapter train).
Adapter+head trained by Adam on source-train (fixed epochs/lr, pre-declared from **source-only** considerations,
not tuned on 8C/target). **bottleneck fixed (no target selection).** Per-cell **task gate = source-val bAcc ≥ 0.58**.
CUDA determinism pinned (`use_deterministic_algorithms`, cuDNN deterministic, seeds); tolerance disclosed (MN-4).
**Re-assert 0-overlap of the NEW 8D subsets with the target panel** (MN-4). Target labels enter **only**
`target_bacc` + L6.

## Metrics
- **Representation L1 (key):** mean pairwise subject separability (2-way, run-held-out train 4/8 test 12) on **`h`**
  (primary) + δ + `z'` (secondary), on **held-out** source subjects, per arm × N × condition. Report the A1-SE
  positive-control L1-on-`h` and the `‖δ‖/‖z‖` share.
- **Task:** target bAcc, macro-F1, source-val bAcc; **per-cell train-seed SD** (MA-3).
- **L4/L5/L6** (gated): as 8C-1; **erase/null rank FIXED across N** for the scaling contrast (MN-3); L5 null =
  variance-matched; if L5 is gate-failed at N=2 **or** N=8, gate 1 is **not evaluable** (MA-4).

## Staging (PM): 8D-0 metric-power gate FIRST, then 8D-1 scaling
**8D-0 (run first; CBraMod only, N_source=8 only, subset_seeds=3, train_seeds=5; arms A0 + A1-SE):** run **only** the
subject-erase positive control to prove the bottleneck-code L1 metric can move. **8D-0 PASS iff A1-SE:**
held-out-source pairwise separability on `h` **drops vs A0 by ≥0.10 absolute OR ≥20% relative**, **CI lower > 0**,
**source-val task does NOT collapse**, firewall clean. **8D-0 FAIL → STOP Phase 8** (do NOT try another adapter
architecture — that starts a method search): *"the light-adapter design cannot test representation-level
subject-scaling because even a subject-erasure positive control does not move bottleneck subject separability
(metric saturated)."* **8D-1 runs only if 8D-0 passes.** Records `trained_rep_metric_power_gate.csv`.

## 8D-1 gates (pre-registered; CI-based, MA/BL fixes)
- **8D-PASS (any → expand full grid {2,4,8,16,all}, seeds 10, train_seeds 5; PM review first):**
  1. **within-A1, fixed-trials, matched-capacity:** N=8 − N=2 target bAcc **clustered CI (by the 15-target panel)
     excludes 0 and ≥ +0.02**, **with L5 evaluable and not increasing** (min L5 movement pre-declared); or
  2. A1 **L1-on-`h` decreases** with N (clustered/held-out) while target stable/improves, **and A1-SE proved the
     metric can move**; or
  3. target improves while **L1-on-`h` stays high but L5 stays null** (subject retained, not a task lever).
  The primary gate requires a **clustered CI**, not a point difference; report the **minimum detectable effect** at
  (subset=3, train=5, clustered) — **if MDE > +0.02 the pilot cannot PASS as designed** → raise seeds or set the
  threshold to the MDE (BL-2).
- **8D-NULL (STOP Phase 8; no full FT/baselines):** A1 ≈ A0 (no target gain, no `h`-L1 change, no L5 change) **and
  A1-SE proved the metric can move** (else → `l1_untestable_metric_saturated`, not NULL).
- **8D-RISK (do NOT interpret):** A1 improves source-val but hurts target; or L5 increases; or **train-seed SD >
  gate margin** (seed-dependent) → overfitting/unstable, not scaling.

## Wording
- **Allowed:** "a trained adapter with more source subjects does / does not change subject separability **in the
  bottleneck code** / reliance / performance"; "the L1 metric is / is not saturated (per A1-SE)."
- **Forbidden:** "foundation encoders become subject-invariant"; reading a capacity gain (A1>A0) as diversity;
  reading a saturated-metric non-result as a diversity null; full-FT/SOTA.

## Outputs (`results/fsr_trained_rep_scaling/`)
```
adapter_architecture_manifest.json   trained_rep_run_manifest.csv (subsets, fixed-total budget, overlap re-assert)
trained_rep_task_performance.csv (+ train-seed SD)   trained_rep_pairwise_l1.csv (h / delta / z', A0/A1/A1-SE)
trained_rep_l4_alignment.csv   trained_rep_l5_replay.csv   trained_rep_l6_consequence.csv
fixed_vs_growing_adapter_summary.csv   trained_rep_target_label_firewall.json   trained_rep_verdict.json
```
`trained_rep_verdict.json`: as v1 + `{"l1_locus":"bottleneck_code_h","positive_control_metric_can_move":null,
"delta_variance_share":null,"diversity_contrast":"withinA1_fixedtrials_dim16_subjectsheldout","gate_mde_clustered":null,
"adapter_signal":"none|design_limited|l1_untestable_metric_saturated|target_gain|l1_reduction|reliance_change|mixed|unstable"}`.

## STOP rules
```text
1  full FT / >2 predeclared adapter sizes / post-hoc size sweep / bottleneck varied across N in the scaling contrast.
2  target labels in adapter train / head / bottleneck selection / subset / rank / early stop.
3  CodeBrain-only rescue; specialist baselines; new dataset/channel-map/target-panel.
4  A1 target gain seed-dependent (train-seed SD > gate margin) -> unstable, do not interpret.
5  source-val too small at N=2 to gate reliably -> report, do not force (bottleneck already fixed=16).
6  A1-SE cannot move L1-on-h below ceiling -> metric saturated -> L1 untestable (declare, not NULL).
```

## After 8D
8D-PASS → expand full grid (PM review first). 8D-NULL / metric-saturated → **stop Phase 8**; write the frozen +
lightly-trained conclusion as an FSR extension/appendix. PC2 paused; Paper 1 unaffected; **Paper 2 frozen**.
