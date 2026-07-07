# FSR_39 — Head-Only Learned-Reliance Results (Phase 7B)

**Project FSR — Phase 7B.** Results of the staged, fail-closed head-only learned-reliance experiment (FSR_38).
CPU-only; freeze the Phase-4B branch latents, retrain **only** a linear head on the concatenated branch latents
(96-dim) under a source subject↔class skew. 8 fresh confirm seeds × 21 folds × ρ∈{0,0.5,0.8}. Scripts + raw CSVs
on `project/fsr-rq4-refit`. Independently verified (SOUND; the framing below is the verification-mandated scoped
one — this is an *under-claim*, not a robustness finding).

## Headline — 7B-0 learnability gate FAILS (fail-closed); 7B-1 does NOT run
Under source subject-class **prevalence-reweighting on true labels** (which induces the subject↔class correlation
*exactly* — weighted `frac(c_d|subject)` = 0.25/0.50/0.80 with ρ, global `P(y)` held to max-dev 0.0), a linear
head shows **no learned subject→c_d reliance** and **preserves the task**: the mechanism is inert. Per the
fail-closed protocol, the target-harm/repair stage (7B-1) **does not run**.

## Verified numbers (8 confirm seeds × 21 folds, clustered bootstrap over folds)
| ρ | cd-pred-bias (H1−H0) | l5 subspace-reliance (H1−H0) | task drop (reweight) | task drop (POS-CTRL corrupt) |
|---|---|---|---|---|
| 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.5 | −0.008 [−0.016, −0.002] | −0.000 [−0.003, +0.002] | +0.007 | **+0.079** |
| 0.8 | −0.017 [−0.039, +0.005] | −0.004 [−0.012, +0.003] | +0.036 | **+0.179** |

- **Subject is strongly decodable** from the concat latents by a linear (head3-class) probe: 0.866 vs chance
  0.106. So the signal is *available* to the head.
- **The head does not learn the shortcut under reweighting:** cd-pred-bias is flat-to-negative (CI at ρ=0.8
  includes 0), subject-subspace reliance is flat, and held-in **task bAcc is preserved** (drop only +0.036 at
  ρ=0.8). The task signal is a *sufficient statistic* for the reweighted true labels, so there is no gradient
  pressure toward the non-generalizing subject signal.
- **The gate has power (positive control):** a task-**conflicting** label-corruption shortcut (relabel a fraction
  of each subject's non-`c_d` samples *as* `c_d`) **collapses** held-in task bAcc (drop +0.179 at ρ=0.8, ≈5× the
  reweighting drop). So the reweighting-null is **mechanism-specific**, not a dead/underpowered gate. Kish
  effective-n fraction at ρ=0.8 is 0.735 (a mild, disclosed power reduction; CIs remain meaningful).

## What this licenses — and the mandatory scope caveat
- **Allowed:** "Source subject-class **prevalence-reweighting** (on true labels) does **not** induce a learned
  head-level subject→c_d reliance on the frozen 4B representation; the head uses the generalizing task signal and
  preserves task accuracy. The learnability gate fails-closed, so the target-harm/repair stage does not run."
- **Mandatory caveat (this is NOT "the head resists weaponization"):**
  1. The Q5a positive-control *leg* — a head demonstrably **learning** the source shortcut and *then* declining to
     harm the target — was **never achieved**: only subject decodability (0.866) passed. Per STOP-rule 8, no
     physiology/robustness/"resists" claim is licensed. The verdict is **"no head-level weaponization
     *demonstrated* under prevalence-reweighting,"** not "cannot be weaponized."
  2. A task-**conflicting label-corruption** weaponization design is **untested** (the positive control shows it
     *would* be detectable via task-bAcc collapse, but measuring a *learned reliance* there needs a
     transfer-aware detector, since an arbitrary per-subject `c_d` does not make the held-in cd-bias metric
     *rise*). This is a **new pre-registration**, not part of 7B.
  3. `n_rows_kept_frac=1.0` (reweighting drops no rows) is **not** unchanged power: Kish eff-n falls to 0.735 at
     ρ=0.8.
- **Forbidden:** "the head resists weaponization"; "natural EEG subject signal is safe / can never be harmful";
  reporting the reweighting-null as a robustness finding.

## Why this deepens the picture (recoverability)
The result sharpens the recoverability theory (FSR_36/37): **prevalence skew on true labels induces a subject↔
class *prevalence correlation*, not a task-conflicting *shortcut*** — because the true task features remain a
sufficient statistic for the (reweighted) labels, the head has no incentive to key on subject. Head-level
weaponization of a naturally present, decodable subject signal therefore requires a **task-conflicting spurious
correlation** (label corruption), i.e. a genuine R2/R3-inducing mechanism, not mere prevalence stress. This
mirrors the Prior-Decoupled discipline: a marginal/prevalence manipulation is not, by itself, an action-relevant
shortcut.

## PC2 implication (strengthened, cheaply)
This says **do not spend GPU on PC2 as designed** (prevalence-stress refit): the same prevalence mechanism is
inert on the cheap head probe — where a shortcut is *easiest* to induce and detect — and a full backbone refit has
*more* capacity to fit the true task, so it is *even less* likely to weaponize. A genuine learned-shortcut test
would require a **label-corruption-style** mechanism (a new pre-registration with a transfer-aware detector), not
the current PC2. `pc2_gpu_gate` stays **paused**; head-only cannot authorize PC2 regardless (FSR_38 Q8).

## Deliverables
`results/fsr_head_only_learned_reliance/`: `head_learnability_gate.csv`, `head_skew_manifest.csv`,
`head_result_interpretation_table.csv`, `head_verdict.json` (gate_pass=false, proceed_to_7B1=false,
positive_control_gate_has_power=true). The full-stage CSVs (dose-response / target-harm / repair) are **absent by
design** — 7B-1 did not run.

## Manuscript impact
A short future-work / mechanism line (not a headline result): *"A cheap head-only probe shows that source
prevalence-reweighting does not weaponize the (decodable) natural subject signal into a learned head-level
reliance — the task signal remains a sufficient statistic — so a learned-shortcut demonstration would require a
task-conflicting (label-corruption) mechanism; this bounds where learned reliance is even inducible, and defers
the costly learned-reliance GPU study."*
