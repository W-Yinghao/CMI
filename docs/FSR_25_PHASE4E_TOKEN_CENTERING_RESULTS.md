# FSR_25 — Phase 4E: Branch-local Token-Neutralization Repair (results)

**Project FSR — Phase 4E.** Results of the pre-registered token-neutralization repair (FSR_24). CPU-only on
frozen 4B dumps; deterministic sha256 tokens; **3 fresh confirm seeds** [20260707/8/9] × 21 folds = **63 confirm
seed-folds** (+ dev seed 0 for mechanism). No GPU/retrain/CMI/fbdualpc/target-fit. Scripts + raw CSVs on
`project/fsr-rq4-refit` @ `2617238`. Verdict independently recomputed + firewall-audited + robustness-checked
(verification `w7lf8vcwi`; GO-WITH-EDITS, all applied). **The three sections below are SEPARATE and must not be
fused.**

> **FROZEN CONCLUSION (PM, not to be changed) — E4E-1/2/3.**
> - **E4E-1 (binding):** `repair_claim_level = none`. Not partial. E4 is not a certified repair.
> - **E4E-2 (descriptive allowed):** "E4 produced the first within-scope deployable-repair **signal** for a
>   controlled first-moment shortcut, but did not clear the frozen confirmatory bar." **Forbidden:** "E4 repairs
>   the shortcut / Phase 4E shows deployable repair / E4 passed."
> - **E4E-3 (methodological, for the NEXT protocol only):** a task-destructive comparator must not veto a
>   task-safe repair via netted recovery; future protocols must gate comparators by raw recovery + clean-task
>   safety first. This does **not** modify the Phase 4E verdict.
> The corrected confirmatory test is **Phase 4F** (FSR_26/27) on **fresh** seeds — Phase 4E is not re-scored.

---

## 1. Confirmatory ledger entry (BINDING) — `repair_claim_level = none`
Firewall **PASS** (all target labels read only via `TargetScorer.score`; fit/selection/α source-only;
source-heldout selection reproduces at zero target reads). `none` is a **criterion outcome, not a firewall
violation**. It is driven **solely** by the frozen clause *"E4 must beat ERASE on netted recovery"*:
`E4 − ERASE = −0.0094` bAcc, CI [−0.0246, +0.0031] — a statistical **tie**, not a loss. Harm was established:
the injected shortcut costs pooled **+0.0426 bAcc [+0.026, +0.061]** (63 seed-folds; 19/63 anti-harm folds
handled by pooled ratio-of-means). E0 exact subtraction = 1.0 (sanity). **We do not overturn this verdict.**

## 2. Descriptive (does NOT meet the pre-registered bar) — E4 is a real within-scope positive
The **deployable primary E4** (full-space first-moment mean alignment `z − λ(mean(z_T) − μ_src)`) **descriptively
reverses** the injected **first-moment** offset token-specifically and task-safely (**a signal, not a certified
repair** — binding level `none`):

| arm | raw recovery (injected) | clean-target drop | netted recovery | verdict role |
|---|---|---|---|---|
| **E4** full mean-align | **+1.15** (restores 0.453→0.502 ≥ orig every seed) | **−0.021 (helps clean)** | **0.655 [0.246, 0.907]** | primary |
| E1 subspace-restricted | +0.93 | −0.018 | 0.516 | secondary (adds nothing: E1−E4 = −0.006) |
| E3 random-subspace | +0.13 | −0.001 | 0.102 | control |
| ERASE (=PC1 R2) | **−0.35 (worsens injected)** | **+0.052 (hurts clean)** | 0.875 | control (see §3) |

E4 point-beats random centering E3 by **+0.0235 bAcc [+0.005, +0.041]** (netted, bAcc units). E1's subspace
restriction adds nothing over full centering (E1−E4 = −0.006, CI < 0), exactly as the mechanism predicts
(`u_tsub` is out of `S`; captured_fraction median 0.68).

**Mandatory caveats (this is why it is descriptive, not a pass):**
- **Fails one frozen clause** (beat ERASE, §3) → binding level `none`. E4 would meet the **partial** bar but for
  that single clause.
- **"E4 > random E3" is pool-only, NOT leave-one-seed-out robust:** the +0.0235 margin (lower CI +0.005, and
  barely above the pre-registered DELTA=0.02) **crosses 0** when confirm seed 20260709 is dropped (+0.0205,
  CI [−0.0014, +0.0423]); every single-seed CI spans 0. Significance lives only in the full 3-seed × 63-fold
  pool. The E4 netted **point estimate** is robust (jackknife 0.646/0.673/0.647; per-dataset 2014=0.54 /
  2015=0.70; drop-anti-harm=0.68) but statistically nonzero only pooled.
- **Construction-favorable, low-SNR scope:** first-moment mean-alignment repairing a first-moment (constant-
  offset) injection is close to built-in (the deployable analogue of the E0 oracle); harm is only +0.043 bAcc
  on a sub-0.5 four-class task; E4 **overshoots** orig (0.502 > 0.496), so the repair is entangled with generic
  alignment gain. This does **not** demonstrate general shortcut repair.
- **Robust sub-claims:** E4 *reverses* the injected offset (raw recovery positive in every seed/dataset/
  anti-harm cut) and is *task-safe on clean* (−0.021, negative every seed). These two are not carried by one
  fold/seed/dataset.

## 3. Post-hoc methodological finding (for the NEXT pre-registration ONLY — not this run's verdict)
The frozen "beat ERASE on netted" gate is **mis-specified**, and this is the reason a real within-scope E4
positive lands as `none`:
- ERASE's netted 0.875 is a **regression-to-a-common-floor artifact**, not token removal. Per confirm seed,
  erasure collapses **both** injected and clean to the same ~0.44 attractor (seed7 inj0.438/cln0.441; seed8
  0.442/0.443; seed9 0.433/0.446), annihilating the +0.043 clean-injected gap. Injected does **not** restore
  toward orig 0.496 — both arms fall to a shared floor, and the higher-starting clean arm falls farther,
  manufacturing a positive netted "gap." ERASE's **raw** recovery is **negative** (−0.35: it worsens the very
  shortcut it should remove) and it fails task-safety by ~5× (clean drop +0.052).
- The pre-registered **clean-drop ≤ 0.01 task-safety gate** (FSR_24 selection) was applied to E4 but **never to
  the ERASE control** before letting ERASE's netted number veto E4. A corrected protocol must **disqualify any
  comparator arm failing task-safety** (`clean_drop ≤ SAFE_DROP` AND `raw_recovery > 0`) **before** the netted
  comparison.
- **This is a specification gap for the next freeze — NOT a license to re-score this run.** ERASE's task-
  destructiveness was known and pre-registered before freeze, so invoking it *after* it became the sole blocker
  would be goalpost-moving. To **claim** E4 we must amend the frozen protocol (task-safety gate on comparators)
  **and re-run on new fresh confirm seeds** (also powering the leave-one-seed-out E4>E3 test) — evaluated once.

## PC2 GPU gate
`pc2_gpu_gate = paused` (bound to `level ∈ {partial, strong}`; `none` → paused). **PC2 stays paused / no-go.**
Eligibility is earned only via a corrected FROZEN protocol re-run reaching ≥ partial, not by reinterpreting this
frozen run.

## Recommendation (PM decision)
Phase 4E produced the project's **first within-scope deployable-repair signal**: full first-moment target-batch
mean alignment (E4) reverses a known injected constant-offset shortcut, token-specifically and task-safely,
where erasure (task-destructive) and a counterfactual head (Phase 4D) failed — but it did **not** clear the
frozen bar, its discriminative significance is pool-only, and its scope is construction-favorable. Two honest
options:
1. **Amend + re-confirm (recommended if a repair claim is wanted):** add the pre-registered comparator
   task-safety gate; re-run on ≥3 **new** fresh confirm seeds (more seeds to power leave-one-seed-out); if E4
   clears ≥ partial there, **then** PC2 becomes eligible. This is the only firewall-clean path to a claim.
2. **Accept `none` and stop the repair line:** manuscript reads "verification + attribution succeed; a
   first-moment mean-alignment repair shows a within-scope, construction-favorable signal but does not clear a
   pre-registered bar; deployable repair of general shortcuts remains unresolved."

## Manuscript impact (Result 4, current)
*"Erasure — even task-orthogonalized — does not repair the injected shortcut (it regresses accuracy to a floor).
A counterfactual head (4D) ties a random control. A deployable first-moment mean-alignment (4E) reverses the
constant-offset injection token-specifically and task-safely (raw recovery 1.15, netted 0.65, beats random),
but this is a construction-favorable, pool-only-significant, within-scope demonstration that did not clear the
pre-registered comparator bar. Verification and attribution succeed; certified general repair remains open."*
