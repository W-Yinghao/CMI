# FSR_49 — CodeBrain + CBraMod 8C Subject-Scaling Results (Phase 8C-1)

**Project FSR — Phase 8C-1.** Source-subject-count scaling audit of frozen EEG foundation encoders (protocol
FSR_48 v2; PM 8C-1 terms). **CBraMod primary, CodeBrain exploratory.** PhysioNetMI (105 subjects, imagined
left/right fist, runs 4/8/12), F1 spatial, PCA d=128 (fit **once** on the 90-subject source pool), per-cell task
gate (source-val ≥0.58), primary L1 = mean pairwise subject separability (2-way, run-held-out), L5 vs
variance-matched null, 15-subject frozen target panel, target labels final-scoring-only. `N_source` growing
{2,4,8,16,all} + fixed {2,4,8} × 10 seeds (N=all single composition). **Adversarially verified — the framing below
is the verification-mandated one (an under-claim relative to the raw slopes, with a scope limit made explicit).**

## Headline (verifier-corrected)
> On frozen PhysioNetMI — a **weak** MI task (target bAcc chance 0.5 → peak 0.64) — cross-subject transfer improves
> **modestly with more source TRIALS** (+0.036/log-N, robust to dropping N=all; concentrated at N≥16). At **fixed**
> trials, added subject **diversity** shows **no *detectable*** benefit over the narrow feasible range {2,4,8}
> (an **underpowered null**, not proof of absence). Subject identity stays ~**95%**-decodable and subject-subspace
> erasure **never helps the task at any N** — but that ~95% is a **fixed, training-independent property of the
> frozen encoder** (ceiling-saturated), so **this design confirms "subject encoded ≠ task lever" (extends 8B) yet
> cannot test whether diversity *erases* subjects.** The only real effect is **sample-size**; the FSR-relevant
> **diversity** and **reliance-scaling** claims are **NULL/untestable** → **no specialist baselines.**

## Numbers (all reproduce; CIs are un-clustered → treat as lower bounds on uncertainty)
| CBraMod (primary), slope vs log(N) | value [cell-level CI] | reading (downgraded per MAJOR-2) |
|---|---|---|
| target bAcc, **growing** {2,4,8,16,all} | +0.036 [0.028, 0.045] | increases; over the **shared** {2,4,8} only +0.021 |
| target bAcc, **fixed** {2,4,8} | +0.002 [−0.011, 0.015] | no detectable effect (CI admits ±0.02; N=2 fixed≡growing) |
| pairwise subject separability | −0.002; **level ~0.95** | **fixed property of the frozen encoder; N-independent by design** |
| L5 subject reliance (gated, n=22) | ~0 (abs) | erasure **never** materially helps; beats variance null only **4/22** |

Means (CBraMod growing target bAcc): N=2 0.524 → N=4 0.516 → N=8 0.553 → N=16 0.596 → all 0.644 — **the rise is at
N≥8/16**, where the **fixed** arm cannot follow (feasibility-truncated), so "growing rises, fixed flat" partly
reflects a **range** difference, not only a mechanism difference (MAJOR-4).

## What this does and does NOT show (the scope correction)
- **DOES (extends 8B):** in the frozen encoder + linear head, the **subject subspace is not a task lever** — erasing
  it never materially changes source-val or target accuracy, at **any** `N_source` (absolute L5 ≈ 0; beats a
  variance null in only 4/22 gated cells). "Subject encoded ≠ harmful reliance" holds in the scaling regime too.
- **DOES (mundane):** transfer improves with more source **trials** (sample-size), N≥16-driven.
- **DOES NOT (design-limited):** it **cannot** test "does subject **diversity** erase subject information," because
  pairwise-L1 is measured in a **fixed source-pool PCA space with a frozen encoder** → it is training/condition/N-
  **independent by construction** (~0.95 for any subject pair, ceiling-saturated). The flat L1-vs-N is **structural,
  not a finding.** A real diversity test needs a *trained* representation (fine-tuning / re-fit per N), which is out
  of scope (frozen-encoder audit; no fine-tuning per PM).
- **DOES NOT (underpowered):** the fixed-trials "no diversity effect" is over **3 levels {2,4,8}** (N=2 fixed≡growing),
  weak task, un-clustered CI admitting ±0.02 → **"no detectable effect,"** not "no effect."

## Mandatory caveats (disclose-all-cells)
1. **Weak-task regime** (like BNCI, unlike decodable SHU-MI): target bAcc 0.52–0.64, **gate pass CBraMod 0.52 /
   CodeBrain 0.39** — ~half the cells fail; L4/L5/L6 on the gated ~half; perf-slope/L1 computed ungated.
2. **CIs are cell-level, not clustered** by the shared 15-target panel (harness docstring over-claimed "clustered");
   they are **over-precise** — signs are robust, magnitudes/CIs are lower bounds on uncertainty.
3. **Pairwise-L1 ~0.95 ceiling-saturated AND N-independent by design** (MAJOR-1) — reported for completeness, **not**
   a diversity result. Raw N-way accuracy is descriptive-only.
4. **Fixed only {2,4,8}**; **CodeBrain exploratory** (gate 0.39, same direction, weaker).

## Firewall (verified CLEAN)
All 41 plan subsets contain **zero** target-panel subjects (90 distinct source subjects, none in the 15-panel). PCA
source-pool only; head/gate/standardization on source-train; L1 on source subjects; rank from N only. **Target
labels enter only `target_bacc` and L6** (final scoring). No leak (verifier-confirmed).

## Verdict / next
FSR-relevant claims (subject-**diversity** effect; **reliance scaling**) are **NULL or untestable by this design**;
only the **sample-size** effect survives, and the **"subject encoded ≠ task lever" result extends to scaling**. Per
PM (null for the interesting claims → no baselines), **`proceed_to_specialist_baselines = False`.** Phase-8
contribution = **encoder-level FSR audit** (8B decodable-task: subject encoded ≠ relied-upon, architecture-general)
**+ its scaling boundary** (8C: the subject subspace is never a task lever across `N_source`; transfer gains are
data-driven; the frozen design cannot address diversity-erasure). PC2 paused; Paper 1 unaffected; **Paper 2 frozen**
— whether 8B+8C become a follow-up is a later decision (and would need a *trained*-representation arm to make a real
diversity claim).

## Deliverables (`results/fsr_codebrain_cbramod_8c/`)
`subject_scaling_performance.csv`, `subject_scaling_pairwise_l1.csv`, `subject_scaling_nway_l1_descriptive.csv`,
`subject_scaling_l4_alignment.csv`, `subject_scaling_l5_replay.csv`, `subject_scaling_l6_consequence.csv`,
`fixed_vs_growing_trials.csv`, `task_gate_by_cell.csv`, `model_condition_slope_summary.csv`,
`subject_scaling_mixed_effects.json` (with the un-clustered-CI + L1-is-structural notes),
`codebrain_cbramod_8c_verdict.json`, `target_label_firewall.json`.
