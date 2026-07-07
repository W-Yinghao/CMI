# FSR_30 — Phase 4G: Controlled Second-Moment Repair Positive Control (results)

**Project FSR — Phase 4G.** Results of the pre-registered controlled second-moment repair test (FSR_29). CPU-only;
8 fresh confirm seeds × 21 folds × 2 injection types (varmod primary, covtoken secondary) = **168 seed-folds**
(+ dev 0). Scripts + raw CSVs on `project/fsr-rq4-refit`. Verdict independently recomputed (byte-identical),
firewall-audited, and adversarially probed (SOUND).

## Headline — `repair_claim_level = none` (a clean, mechanistically-attributed negative)
On a strictly mean-null second-moment shortcut injected into `spatial_z` (harm **+0.031 bAcc [0.007, 0.058]**;
mean-displacement **exactly 0**, excess-variance along `v_c` in **100%** of folds; mechanical-identity 1.2% vs
4F's 73%), **covariance-shrinkage repair does not beat random-direction shrinkage at the source-selected
operating point** — `E4b − E3 = +0.005 bAcc` (clustered CI **[−0.005, 0.014]**, below the 0.02 bar) — and **even
oracle shrinkage along the *true* injected direction gains only +0.004** (CI [−0.005, 0.012]). The estimator
recovers `v_c` (overlap 0.71) and matches the oracle, so this is a **genuine weakness of covariance-shrinkage,
not mis-estimation** (`fail_attribution = genuinely_weak_second_moment_repair`).

**Scope boundary (the scientific point):** a **deterministic first-moment** offset is exactly invertible by
mean-alignment (Phase 4F, scoped-strong), but a **stochastic second-moment** per-sample perturbation — which has
already scattered samples across the decision boundary — is **not** recoverable from batch-level covariance
statistics, even with oracle knowledge of the direction. The deployable repair family is confined to
**first-moment / deterministic-offset** shortcuts.

## Verified numbers (varmod primary; clustered bootstrap over folds)
| quantity | value |
|---|---|
| pooled harm | +0.0308 [0.007, 0.058] → established |
| mean displacement / mech-identity | 0.0 (exact) / 1.2% |
| E4 first-moment netted | **−0.78** (insufficient — mean-alignment hurts a mean-null injection) → `mean_null_pass` |
| E4b netted recovery | 0.18 [−0.42, 0.45] (CI includes 0) |
| **E4b − E3 (specificity)** | **+0.005 [−0.005, 0.014]** → `beats_e3 = False` |
| **ORACLE-E4b − E3** | **+0.004 [−0.005, 0.012]** → `oracle_beats_e3 = False` |
| est-dir·v_c overlap / injection-dominance | 0.71 / 1.12 |
| ERASE valid_repair | False (raw −0.68, clean drop +0.029 → task-destructive negative control) |
| leave-one-dataset-out | fail (both datasets sub-DELTA); covtoken secondary also `none` |

## Mandatory honesty qualifier — α dose-response
There **is** a monotone injection-strength dose-response in the specificity gain: `E4b − E3` = −0.001 (α=1) →
+0.017 (α=2) → **+0.023 (α=3, > DELTA)**; the oracle tracks it (+0.021 at α=3). **But** the DELTA-crossing lives
**only at α=3**, which is the pre-registered **injection-dominant near-tautology** regime ("undo an injection we
can fully see in the covariance"), and the **source-only stress rule selects α ∈ {1,2}** (85/82/1 split, zero
fallbacks). So:
- **At the honest operating point (source-selected α), second-moment repair is `none`.**
- A direction-specific advantage clearing the bar emerges **only** in the injection-dominant regime, which
  certifies nothing general (the same tautology caveat as 4F, here correctly *excluded* by the α selection).
- **Do not state** "second-moment shortcuts are unconditionally unrepairable" — the accurate claim is **"not
  repairable at the source-selected operating point; a DELTA-clearing direction-specific advantage exists only in
  the injection-dominant α=3 regime."**

## Firewall
Clean: target labels flow only into `TargetScorer.score` (34 reads/fold-injtype = 1 orig + 3α×11 arms, no hidden
reads); `excess_dirs`/`shrink_along` use covariances of target-X + source only; `k,λ,α` source-only. E4b, oracle,
and the diagnostics are all target-X-only.

## What this licenses / does not
- **Licenses (negative, informative):** covariance-shrinkage (E4b) — even oracle-directed — **does not repair a
  controlled second-moment stochastic shortcut at the operating point**; the repair family established in 4F is
  **first-moment-specific**. C14 (E4b repairs controlled second-moment shortcuts) resolves to **not established**.
- **Does not license:** any learned/natural/general/DG/SOTA repair claim; any "second-moment repair works";
  re-scoring 4E/4F.
- **PC2 implication (sharpens FSR_31):** a **learned** reliance will carry higher-moment / stochastic structure,
  which 4G shows is **not** repairable by the current first/second-moment primitives — so spending PC2 GPU on
  E4/E4b against a learned reliance is expected to be an **expensive negative**. `pc2_gpu_gate = paused`,
  `pc2_gpu_run_authorized = false`.

## Manuscript impact (Result 4, current)
*"Repair is scope-bounded to first-moment deterministic offsets. A deployable first-moment mean-aligner inverts a
controlled constant-offset injection (4F, scoped-strong, largely by construction); but a controlled mean-null
second-moment (stochastic variance) injection is not repaired by covariance-shrinkage at the source-selected
operating point — even with oracle knowledge of the injected direction (4G, none; a DELTA-clearing advantage
appears only in the injection-dominant regime). Verification, localization, and attribution succeed; deployable
repair of anything beyond a deterministic first-moment offset — and of natural/learned shortcuts — remains open."*
