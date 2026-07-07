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
after seeing target harm (STOP).

## Two gates (both required before any weaponization claim)
### Gate Q7C-a — held-in learnability (the head DID learn the engineered shortcut)
On **source held-in** data: `Hconflict` fits the corrupted labels (corrupted-label accuracy rises with γ), and its
subject→c_d shortcut score exceeds the `Hrandom`/`Hshuffle` null band, monotone in γ. **If Q7C-a fails → STOP:**
"the controlled label-conflict did not create a learned shortcut in the head class; no weaponization inference."
(7B's positive control already proved the metrics have power.)

### Gate Q7C-b — source-pseudo-target transferability (NEW; the shortcut TRANSFERS, not memorizes)
A source-ID shortcut may merely memorize training subject identities and not transfer to an unseen subject. So:
```
Within source subjects: hold out one/more source subjects as PSEUDO-TARGETS;
train Hconflict on the remaining source with the task-conflict labels;
evaluate the pseudo-target on TRUE labels + reliance (L4/L5) + true-task bAcc.
```
Q7C-b passes iff the pseudo-target shows **increased subject reliance and reduced true-task bAcc** vs H0, beyond
the `Hshuffle` null. **If Q7C-a passes but Q7C-b fails →** "the head can **memorize** source subject-correlated
corruption, but it does **not** create transferable cross-subject harmful reliance under this protocol" (a real,
reportable result — not a target-weaponization claim).

## Primary result (weaponization)
**Confirmed** iff **all**: Q7C-a passes AND Q7C-b passes AND `Hconflict` L4/L5 subject-reliance rises over
`Hrandom`/`Hshuffle` AND **target true-label bAcc decreases vs H0** (clustered CI < 0) AND the effect grows with
γ AND firewall clean. **Allowed:** "controlled task-conflicting source labels can weaponize naturally present
subject information into learned head-level reliance." **Forbidden:** "natural EEG training contains such label
corruption"; any natural-harm/clinical claim.
**Not confirmed** (Q7C-a passes, Q7C-b or target harm fails): "the head fits subject-correlated corrupted labels
but the shortcut does not transfer to unseen subjects/target under this protocol." **Underpowered** (Q7C-a fails):
"the label-conflict stress did not create a learned shortcut; no weaponization inference."

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
```
label_conflict_manifest.csv            # per fold/seed/gamma: c_d, n_swaps, achieved global-Py delta
global_label_histogram_check.csv       # max_abs_delta_global_Py (must be 0) per fold/seed/gamma
heldin_learnability_gate.csv           # Q7C-a: corrupted-label acc + shortcut score vs Hrandom/Hshuffle
pseudo_target_transferability_gate.csv # Q7C-b: pseudo-target reliance + true-task bAcc drop
dose_response_reliance.csv             # L4/L5 vs gamma for the heads
target_harm.csv                        # H0 - Hconflict target true-label bAcc vs gamma
repair_secondary_results.csv           # E4/E4b/ERASE/Hreg on Hconflict (gamma=0.4)
random_label_noise_controls.csv        # Hrandom detail
subject_mapping_shuffle_controls.csv   # Hshuffle null band
target_label_firewall.json
label_conflict_verdict.json
```
`label_conflict_verdict.json` (key fields):
```json
{"heldin_learnability_pass": null, "pseudo_target_transferability_pass": null,
 "weaponization_confirmed": null, "primary_gamma": 0.4, "global_Py_preserved": null,
 "hconflict_vs_hshuffle_reliance": null, "target_harm": null, "target_harm_ci": [null,null],
 "target_labels_used_for_fit": false, "target_labels_used_for_selection": false,
 "repair_claim_level": "none|secondary|training_time_partial", "pc2_gpu_gate": "paused"}
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
