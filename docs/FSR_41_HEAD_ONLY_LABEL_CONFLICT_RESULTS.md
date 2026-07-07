# FSR_41 — Head-Only Task-Conflict Weaponization Results (Phase 7C)

**Project FSR — Phase 7C.** Results of the staged, fail-closed head-only **task-conflict** weaponization test
(pre-registration FSR_40). CPU-only; frozen Phase-4B branch latents; retrain **only** a linear head on the
concatenated branch latents (96-dim) under a subject-correlated, **task-conflicting** label corruption injected by
deterministic **paired label swaps** that hold the global label histogram **exactly** (`max_abs_delta_global_Py=0`
on all 504 rows). 8 fresh confirm seeds × 21 LOSO folds × 2 datasets (BNCI2014_001 4-class, BNCI2015_001 binary) ×
γ∈{0,0.2,0.4}. Scripts + raw CSVs on `project/fsr-rq4-refit`. **Design-red-teamed before the run** (2 BLOCKERs +
4 MAJORs fixed — the memorize-vs-transfer gate separation is the key fix) and **adversarially results-verified**
(all numbers reproduce; firewall clean; 3 over-claim corrections applied — the framing below is the verified one).

## Headline — the head can MEMORIZE the corruption but does NOT weaponize it (fail-closed at Q7C-b)
Subject-keyed, task-conflicting label corruption (P(y) held exact) is **learnable in-sample** by a linear head on
frozen EEG latents — conflict-subset fit **0.70** vs a **0.20** clean-head floor, dose-monotone — but it is **not
weaponizable**: on **held-out** subjects and on the **target**, its true-task harm **never exceeds a subject-
scrambled control** (`beats_shuffle=False` on **both** datasets), and the fit is **not localized to the subject
subspace** (erasure/ERASE diagnostics ≤ 0). It **beats matched random noise** (+0.078), so the damage is **generic
subject-blocked corruption harm** that transfers, **not** a subject-**structure-specific / natural-signal-aligned**
transferable reliance. **Weaponization NOT confirmed.** With 7B (prevalence skew not even *learned*), neither cheap
head-level manipulation turns naturally-present subject leakage into a transferable harmful reliance.

## Q7C-a — held-in learnability: PASS (in-sample fittability only)
On the **training** subjects, the conflict head satisfies the relabeled (task-conflicting) subset far beyond a
task-only floor, and the achieved corruption rate tracks γ (construction-capped, disclosed):

| γ | achieved conflict frac | conflict-subset fit | fit − task-only floor (CI_lo) |
|---|---|---|---|
| 0.0 | 0.000 | — | — |
| 0.2 | 0.175 | 0.491 | 0.288 (0.251) |
| 0.4 | 0.343 | 0.700 | 0.496 (0.463) |

- **What this licenses:** *the linear head can MEMORIZE the subject-conditional, task-conflicting relabeling
  in-sample (fit beyond the clean-head floor, dose-monotone).* This is a **capability check** — unlike 7B, where
  prevalence skew was **not even learned** (task signal a sufficient statistic).
- **What it does NOT license (verifier-mandated):** it is **not** "a subject *shortcut* was learned." The
  subject-subspace-erasure diagnostic is **negative** (`l5_conflict − l5_shuffle` = −0.032/−0.043; vs random
  −0.004/−0.011) and ERASE repair is null-to-negative — the fit is **not attributable to the estimated k=2 subject
  subspace**. Memorizing labels *keyed on* subject identity does not imply the head *uses the identifiable subject
  signal*; a linear head can satisfy the subject-conditional relabel via non-subspace/idiosyncratic directions.

## Q7C-b — transfer to unseen subjects: FAIL (not subject-structure-specific)
On **held-out** source subjects (pseudo-targets), clustered bootstrap over folds, γ=0.4:

| contrast (structured conflict head) | pooled | BNCI2014_001 (4-cls, 9 folds) | BNCI2015_001 (binary, 12 folds) |
|---|---|---|---|
| true-task drop **vs H0** | +0.112 [0.095, 0.129] | +0.129 [0.110, 0.147] | +0.100 [0.075, 0.125] |
| true-task drop **vs random-noise** | **+0.078 [0.063, 0.092]** | +0.100 [0.081, 0.118] | +0.061 [0.047, 0.076] |
| true-task drop **vs subject-shuffle** | −0.010 [−0.018, −0.001] | **−0.001 [−0.013, +0.010]** | −0.017 [−0.027, −0.005] |

- **Structured corruption transfers harm** (vs H0 > 0, both datasets) and **beats matched random label-noise**
  (vs random > 0, both datasets) — so *subject-blocked* corruption does transfer damage.
- **But it does NOT exceed the subject-shuffle control** (`beats_shuffle=False`) on **either** dataset — the harm
  is **not subject-structure-specific**. The subject→c_d **structure** is not the load-bearing part of the harm; a
  scrambled subject→class map does **equal-or-more** damage.
- **Fragility disclosed (verifier):** the mildly-negative pooled vs-shuffle is significant **only pooled** and is
  **driven by the binary dataset**; on BNCI2014_001 the CI **crosses zero** (a null). The robust, both-dataset-
  consistent statement is **`beats_shuffle=False`**, *not* "shuffle is more harmful." A negative structured-minus-
  shuffle is read as a **construction asymmetry**, not evidence that scrambling harms more (a target-side check
  confirms Hshuffle−Hconflict crosses 0). `monotone=False` on the vs-shuffle gap is **consistent** with no
  structured effect.
- Reliance diagnostics (weak instrument, disclosed): l5(c−shuf) +0.001 [−0.013,+0.014], l4(c−shuf) −0.004
  [−0.010,+0.003] — both include 0.
- **⇒ pseudo_target_transferability_pass = False.**

## Target harm — real but GENERIC (not weaponization)
Target true-label bAcc **drops** (H0 − Hconflict = +0.062 [0.033, 0.091]; both datasets +0.054 / +0.068). But per
the shuffle contrast this is **generic corruption harm** (subject-scrambled corruption harms the target
equal-or-more), **not** a *subject-structured target weaponization*. `weaponization_confirmed = False`.

## Repair (SECONDARY only; correctly gated to `repair_claim_level = None`)
On the primary-γ conflict head (target-X + one training-time arm), vs the Hconflict target bAcc:

| arm | gain over Hconflict | CI_lo | frac of harm gap |
|---|---|---|---|
| **E4** (first-moment mean-align, target-X) | **+0.028** | **0.008** | 0.46 |
| E4b (second-moment shrink, target-X) | −0.006 | −0.010 | — |
| ERASE (subject-subspace, target-X) | −0.008 | −0.033 | — |
| Hreg (training-time subspace-erased) | −0.004 | −0.028 | — |

- **E4 recovers ~46% of the harm gap** (CI excludes 0; both datasets CI_lo>0, 2015 barely at 0.002) — but this is
  recovery of **generic, first-moment** corruption-induced shift, **consistent with 4F's first-moment deterministic
  scope**, and E4 was run **only** on Hconflict (not shown specific vs shuffle/random). **It is NOT repair of a
  demonstrated weaponization — none was demonstrated.** `repair_claim_level` is therefore **`None`** (aggregator-
  enforced: repair claims are gated OFF unless weaponization is confirmed). E4b/ERASE/Hreg are null-to-negative,
  consistent with 4G (no second-moment / erasure repair).

## Firewall (verified clean)
`target_labels_used_for_fit = false`, `target_labels_used_for_selection = false`; all target reads go through
`p4e.TargetScorer` (score-only, 13 reads/fold accounted); `l4` is label-free (head-weight/subspace cosine); the
reliance dose is reported on **source** held-out (source labels). `c_d`, the swap construction, γ, head training,
and every repair operator are source-/target-X-only. (`target_label_firewall.json`, 168 fold-seed rows.)

## Why this deepens the picture (mechanism bridge with 7B)
- **7B (prevalence skew):** the subject↔class correlation is induced *exactly*, but the head does **not even learn**
  it — the task signal is a sufficient statistic; nothing to weaponize.
- **7C (task-conflict skew):** the head **does learn** (memorizes) the subject-conditional corruption in-sample,
  **but** the resulting transferable harm is **not subject-structure-specific** (ties the shuffle control; beats
  only random noise) and **not localized to the subject subspace**.
- **Together:** across the two cheapest head-level manipulations — where a shortcut is *easiest* to induce and
  detect — a naturally-present, decodable subject signal does **not** become a **transferable, structure-specific
  harmful reliance**. Subject-*blocked* corruption transfers generic harm; the *specific* natural-signal alignment
  does not. This is the FSR thesis at the mechanism level: a correlation (prevalence) or even a learnable
  subject-conditional corruption is not, by itself, a transferable harmful **shortcut**.

## PC2 implication (stays paused; strengthened again, cheaply)
The mechanism where weaponization is *easiest* (cheap linear head, deliberate task-conflict) **fails** to produce
structure-specific transferable harm; a full backbone refit has **more** capacity to fit the true task and is if
anything **less** likely to weaponize. `pc2_gpu_gate` stays **paused**; head-only cannot authorize PC2 regardless
(FSR_38 Q8).

## Mandatory caveats / forbidden claims
- **Allowed:** "a subject-keyed, task-conflicting label corruption is *learnable in-sample* by a linear head on
  frozen 4B latents but does **not** weaponize into a *transferable, subject-structure-specific* target harm — its
  transferable harm ties a subject-scrambled control and is not localized to the subject subspace."
- **Forbidden:** "a subject shortcut was learned" (Q7C-a is in-sample fittability, not subspace reliance);
  "scrambling is more harmful than the true structure" (construction asymmetry, not an effect); "E4 repairs the
  weaponization" (none demonstrated; E4 recovers generic first-moment harm only); any "natural EEG training
  contains this corruption" / natural-harm / clinical / DG / SOTA claim. **7C is a CONTROLLED task-conflict
  positive-control probe.**

## Deliverables (`results/fsr_head_only_label_conflict/`)
`heldin_learnability_gate.csv` (504), `pseudo_target_transferability_gate.csv` (504), `target_harm.csv` (504),
`repair_secondary_results.csv` (168), `dose_response_reliance.csv` (504), `label_conflict_manifest.csv`,
`global_label_histogram_check.csv`, `target_label_firewall.json`, `label_conflict_verdict.json`
(`heldin_learnability_pass=true`, `pseudo_target_transferability_pass=false`, `weaponization_confirmed=false`,
`repair_claim_level=null`, per-dataset breakdown, `beats_shuffle_in_any_single_dataset=false`, `pc2_gpu_gate=paused`).

## Manuscript impact
A short mechanism line (not a headline result): *"Two cheap head-only probes bound where learned subject reliance
is inducible: source prevalence-reweighting does not weaponize the (decodable) natural subject signal (the task
remains a sufficient statistic; 7B), and a subject-keyed, task-conflicting label corruption — though learnable
in-sample — does not weaponize either, because its transferable harm is generic (ties a subject-scrambled control,
beats only random noise) and is not localized to the subject subspace (7C). Neither turns naturally-present subject
leakage into a transferable, structure-specific harmful reliance on the frozen representation."* **Phase 7C closes
the cheap-mechanism line; the manuscript now freezes. PC2 stays paused.**
