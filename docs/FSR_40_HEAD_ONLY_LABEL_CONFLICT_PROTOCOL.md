# FSR_40 — Head-Only Task-Conflict Weaponization Protocol (Phase 7C; pre-registration)

**Project FSR — Phase 7C.** Pre-registration of the **last cheap mechanism experiment** (PM). 7B-0 showed source
subject-class *prevalence-reweighting on true labels* does **not** weaponize the head (the task signal stays a
sufficient statistic). 7C asks the complementary question: does a **task-conflicting**, subject-correlated
training label (label corruption that fights the true task) weaponize the naturally present subject signal into a
**transferable, target-harmful** learned reliance? **Primary = weaponization test, not repair.** CPU-only,
head-only, frozen 4B latents; no GPU/backbone-retrain/CMI/fbdualpc/target-fit/Lee2019/new-primitive/post-hoc-γ.
Design-red-teamed before the run. After 7C, the manuscript freezes regardless of outcome.

## Scientific question
> When the training objective contains a subject-correlated, **task-conflicting** signal, can a head trained on
> frozen EEG representations weaponize the naturally present subject information into a **cross-subject
> transferable, target-harmful** learned reliance?

This is a **controlled task-conflict positive control**, not natural label noise, not a clinical claim, not a PC2
substitute. It completes the bridge: *prevalence skew → not weaponized (7B); task-conflict skew → ?*

## Data + heads
Frozen Phase-4B branch latents, concat `[graph_z|temporal_z|spatial_z]` (96-dim; spatial is the strongest
subject-leaky/load-bearing/coupled branch), 21 LOSO folds, source trains the head, target scores only. Heads
(all source-only, fixed linear architecture mirroring `head3`, no sweep; 8 fresh confirm seeds + dev 0):
- **H0** — clean balanced labels (baseline).
- **Hprev** — 7B prevalence-reweighting head (negative mechanism control; expected inert).
- **Hconflict** — subject-correlated **task-conflict** labels (PRIMARY weaponization arm).
- **Hrandom** — matched-rate *balanced* label corruption **without subject structure** (label-noise control).
- **Hshuffle** — same corruption rate / same P(y), but the subject→c_d mapping **shuffled** (subject-structure
  control; multiple shuffle seeds → null band).
- **Hreg** — source-only training-time subject-invariance-regularized head (optional mitigation; secondary).

## The task-conflict label construction (must be exactly P(y)-preserving)
Assign each source subject a deterministic, class-balanced spurious class `c_d` (cyclic by subject rank). Corrupt
labels toward `c_d` via **deterministic paired label swaps** that keep the **global label histogram exactly
unchanged**:
```
For each pair of source subjects (d, d') with c_d = a != b = c_{d'}:
  swap the labels of  k  samples of subject d that are TRUE-class b  with
                      k  samples of subject d' that are TRUE-class a.
  -> d gains a-labels (its own c_d), d' gains b-labels (its own c_{d'});
     the swap relabels k of (b->a) and k of (a->b) -> global histogram net zero.
Each sample is swapped at most once (tracked). k grows with gamma.
```
The relabeled samples are **task-conflicting**: their true features say class $b$ but the training label is $a$,
so the head cannot satisfy them with the task signal — it must key on subject to fit them. Verify in the output:
**`max_abs_delta_global_Py = 0`** (exact integer swap). `Hrandom` uses the same number of relabels but chosen
without subject structure (still histogram-exact); `Hshuffle` uses the same construction with `c_d` shuffled
across subjects.

## Stress levels
`γ ∈ {0, 0.2, 0.4}` (pre-registered; **+0.6 optional, pre-declared**). **Primary `γ = 0.4`.** `γ` is never chosen
after seeing target harm (STOP). The construction can **saturate below γ** (partner-class / per-subject-budget
caps), so the run records the **achieved** conflict fraction `2·n_swaps/N_train` and the gate **requires the
achieved rate to track γ monotonically** (γ=0 → 0 < γ=0.2 < γ=0.4); a saturated, non-tracking dose fails-closed.

## Two gates (both required before any weaponization claim) — memorize (a) vs transfer (b)
### Gate Q7C-a — HELD-IN learnability (the head CAN memorize the engineered shortcut)
Measured on the **TRAINING** subjects (the data the head saw): on the **relabeled (task-conflicting) subset**,
`Hconflict`'s fit (accuracy against the corrupted labels) **exceeds a task-only floor** (`H0`'s fit on the same
rows, ≈0 by construction) and **rises with γ**, under an achieved rate that tracks γ and an exactly-P(y)-preserving
histogram. This is a **linear-learnability capability check** — *can* a linear head on frozen 4B latents key on
subject to satisfy the conflict at all. **If Q7C-a fails → STOP:** "a linear head cannot memorize the subject
shortcut under this task-conflict; no weaponization inference." (Subject-reliance-on-train columns are reported as
**diagnostics only**, not gate conditions — the subspace-erasure L5 is a weak instrument once true-task is degraded.)

### Gate Q7C-b — TRANSFER to unseen subjects (the shortcut TRANSFERS, not just memorizes)
Measured on **held-OUT source subjects** (pseudo-targets the head never trained on): the **STRUCTURED** conflict
head's **held-out true-task harm** must beat **BOTH** controls — the **subject-shuffle band** (`Hshuffle`, `c_d`
scrambled across subjects, `N_SHUFFLE=5` seeds → null band) **and** the matched **random-noise** control
(`Hrandom`) — monotone in γ, i.e. the harm is due to the subject **structure**, not generic corruption. L4/L5
reliance deltas vs the shuffle band are reported as **mechanism diagnostics**, not hard gate conditions.
**If Q7C-a passes but Q7C-b fails →** "the head can **memorize** subject-correlated corruption, but the harm does
**not** require the subject structure (a scrambled control does equal/more damage) — no transferable cross-subject
weaponization under this protocol" (a real, reportable result, not a target-weaponization claim).

## Primary result (weaponization)
**Confirmed** iff **all**: Q7C-a passes (memorizable) AND Q7C-b passes (held-out true-task harm beats BOTH the
shuffle band and random-noise, monotone in γ) AND **target true-label bAcc decreases vs H0** (clustered CI < 0)
AND firewall clean. **Allowed:** "controlled task-conflicting source labels can weaponize naturally present subject
information into a transferable, target-harmful head-level reliance." **Forbidden:** "natural EEG training contains
such label corruption"; any natural-harm/clinical claim.
**Not confirmed** (Q7C-a passes, Q7C-b or target harm fails, or harm does not beat the shuffle band): "the head
**memorizes** subject-correlated corrupted labels but the harm is **not subject-structure-specific / does not
transfer** to unseen subjects under this protocol." **Underpowered** (Q7C-a fails): "a linear head cannot memorize
the shortcut; no weaponization inference."

## Repair (SECONDARY only)
Only the existing arms on `Hconflict` at `γ=0.4`: **E4, E4b, ERASE** (target-$X$; theory predicts R3 → fail) and
**Hreg** (source-only training-time). No new primitive, no repair competition. If Hconflict weaponizes but
E4/E4b/ERASE fail → "learned reliance is R3-like; target-$X$ repair unavailable." If Hreg helps → "training-time
source-visible mitigation may reduce induced reliance." Gate inherits 4F/4G corrections (clustered CI, structural
veto/task-safety, leave-one-dataset-out = 2-dataset consistency check).

## Firewall
Target labels only final scoring (`TargetScorer`). `c_d`, the swap construction, γ, all head training, and repair
fits are source-only. Recorded in `target_label_firewall.json`.

## Outputs (`results/fsr_head_only_label_conflict/`)
Two aggregators, staged: `aggregate_head_conflict_gate.py` (Q7C-a) then, only if it passes,
`aggregate_head_conflict_transfer.py` (Q7C-b + weaponization + repair).
```
label_conflict_manifest.csv            # per fold/seed/gamma: c_d, n_swaps, achieved_conflict_frac, Py-delta, shuffle_nsw
global_label_histogram_check.csv       # max_abs_delta_global_Py (must be 0) per fold/seed/gamma
heldin_learnability_gate.csv           # Q7C-a: conflict-subset fit vs task-only floor (TRAIN) + reliance diagnostics
pseudo_target_transferability_gate.csv # Q7C-b: held-out true-task drop vs H0 / shuffle band / random + L5/L4 diag
dose_response_reliance.csv             # L4(target,label-free) + L5(source held-out) vs gamma
target_harm.csv                        # H0 - Hconflict (and Hshuffle) target true-label bAcc vs gamma (via TargetScorer)
repair_secondary_results.csv           # E4/E4b/ERASE/Hreg on Hconflict (gamma=0.4)
target_label_firewall.json             # TargetScorer read count; l4 is label-free; reliance dose on SOURCE held-out
label_conflict_verdict.json            # Q7C-a verdict, then updated in place with Q7C-b + weaponization + repair
```
`label_conflict_verdict.json` (key fields):
```json
{"heldin_learnability_pass": null, "pseudo_target_transferability_pass": null,
 "weaponization_confirmed": null, "primary_gamma": 0.4, "global_Py_preserved": null,
 "target_harm": null, "target_harm_ci": [null,null], "repair_claim_level": null,
 "target_labels_used_for_fit": false, "target_labels_used_for_selection": false,
 "pc2_gpu_gate": "paused", "gate_verdict": {"...": "Q7C-a"}, "transfer_verdict": {"...": "Q7C-b + weaponization + repair"}}
```

## STOP rules
```text
1  target labels used for fit / c_d / swap construction / gamma or hyperparameter selection / repair fit.
2  GPU / backbone retrain / CMI / fbdualpc / new repair primitive / architecture-or-hyperparameter sweep.
3  global label histogram not exactly preserved (max_abs_delta_global_Py != 0) -> re-construct.
4  weaponization claimed without BOTH Q7C-a (learnability) AND Q7C-b (transferability) passing.
5  weaponization claimed without beating BOTH Hrandom (label-noise) AND Hshuffle (subject-structure) controls.
6  gamma changed after seeing target harm; or a corruption rate tuned on target bAcc.
7  repair promoted to primary; or a new repair primitive added.
8  CLAIM-LOCK: written as natural label noise / natural-harm / clinical / DG / SOTA; or "natural EEG training
   contains this corruption". 7C is a CONTROLLED task-conflict positive control only.
```

## Framing (fixed)
7C is the last cheap head-only bridge. With 7B it forms the mechanism statement: *prevalence correlation is not a
shortcut; a task-conflicting, subject-correlated, and transferable signal is.* Repair is secondary and interpreted
through the recoverability classes (FSR_36/37). **PC2 stays paused.** After 7C, Paper 2 freezes; Paper 1
(Prior-Decoupled TTA) is independent and not blocked.
