# FSR_29 — Phase 4G: Controlled Second-Moment Repair Positive Control (pre-registration)

**Project FSR — Phase 4G.** CPU-only controlled positive control testing whether a **second-moment / covariance**
repair primitive (**E4b**) repairs a **controlled second-moment branch-local shortcut** where the Phase-4F
first-moment primitive (**E4**) should be **insufficient**. **Frozen before the run + design-red-teamed.** No GPU/
retrain/CMI/fbdualpc/target-fit. Reuses the 4E/4F arm scaffolding + the Phase-4F-corrected gate (structural veto,
clustered bootstrap, **leave-one-DATASET-out binding**, netting, non-identity report).

## Core question
> Does **E4b second-moment (covariance) alignment** repair a controlled **zero-mean second-moment** injected
> shortcut on `spatial_z`, where **E4 first-moment mean alignment is insufficient** (by construction of a
> mean-null injection)?
A pass certifies only `controlled_second_moment_only` (and, per the diagnostics below, is **regime-conditional**
on injection-dominance); it does **not** certify learned/natural/general/DG/SOTA repair. The **likely honest
landing** (design-red-team + smoke): E4 is insufficient and E4b−E3 is **real but sub-DELTA** in the fair regime
(injection not dominating the covariance) — a `none`/`partial` that is *informative*: "second-moment repair shows
genuine but sub-threshold direction-specific signal; it only clears the bar when the injection dominates the
covariance (near-tautological)." The oracle-E4b + dominance diagnostics make the landing cleanly attributable.

## Primary branch
`spatial_z` (Phase 4B: max subject-leaky, load-bearing, functionally coupled, task-entangled; PC1/4F primary).

## Injection — ZERO-MEAN second-moment shortcut (strictly mean-nulled)
Feasibility-verified (harm +0.04–0.06 bAcc, E4-insufficient, E4b-recovers). Two injection types (both reported;
`varmod` primary). Deterministic sha256 `token_seed`; **8 fresh confirm seeds** `[20260721..20260728]` + dev 0.
Class-directed `v_c` and subject `u_d` are source-only (as PC1/4F). Per LOSO fold, per token_seed:

- **`varmod` (primary, 4G-B): class-directed variance modulation.** `P_i = α·scale·s_i·unit(v_{c})` with `s_i`
  a per-sample **Rademacher** (`±1`, mean 0); `c = c_d` (source subject majority) / `c_target = hash(tsub) mod
  n_cls` (target). Adds variance along the class-directed direction with **zero batch mean**.
- **`covtoken` (secondary, 4G-A): subject×class rank-2 covariance token.** `P_i = α·scale·(e^u_i·unit(u_d) +
  e^v_i·unit(v_{c}))`, `e^u_i, e^v_i` per-sample Rademacher — a subject–class-coupled covariance shift.
- **Strict mean-null (mandatory):** subtract the per-batch mean of the perturbation, `P ← P − mean_batch(P)`, so
  `mean(z_inj) − mean(z_clean) = 0` **exactly** → a first-moment aligner (E4) subtracts nothing new. Injected
  `z_inj = z + P`.
- **Scale/α (source-only stress rule):** `scale` calibrated so that at α=1 the source-heldout per-sample
  class-directed logit **standard deviation** ≈ median source correct-vs-runnerup margin; `alpha_star` = smallest
  `α ∈ {1,2,3}` with source-heldout induced harm-proxy (logit-std) ≥ `FRAC=1.0`×margin. α never from target bAcc.

## Repair arms (target-X-only + source; target labels score only; all NETTED vs clean-target)
- **E4 — first-moment mean alignment** `z − λ(mean(z_T) − μ_src)`. **Expected insufficient** on a mean-null
  injection (the mean-null sanity check).
- **E4b — second-moment excess-variance shrinkage (PRIMARY).** Estimate the **excess-variance directions**
  `top-k eigvecs of (Σ_T − Σ_S)` (`Σ_T` from **injected target-X**, `Σ_S` from source-X), then **shrink the
  deviation along each toward source variance** (`z ← z − λ(1−r)·(z−mean_T)·qqᵀ`, `r = min(√(qᵀΣ_S q / qᵀΣ_T q),
  1)` — shrink only, first moment untouched). `k, λ` selected source-heldout. Target-X-only, no target labels.
  *(Design note: a full-CORAL variant shares its whitening `Σ_T^{−1/2}` with any random-target recolor, so
  `CORAL-vs-random` cannot isolate specificity — the direction shrunk must be the ablated axis, hence
  excess-direction vs random-direction below.)*
- **E3 — random-direction shrinkage control.** The **identical** shrink operator applied to **k random
  directions** (matched) instead of the estimated excess directions. `E4b > E3` iff **identifying the injected
  excess direction** matters (not generic variance-shrinkage).
- **ORACLE-E4b — shrink along the TRUE injected direction** (`v_c`, and `u_d` for covtoken; known, so
  **non-deployable / headline-excluded**). Bounds "if the excess direction were known exactly, how much repair is
  possible" → makes a sub-threshold E4b **cleanly attributable** (oracle repairs but E4b doesn't ⇒ the estimator
  `dirs_exc` missed `v_c`; oracle also fails ⇒ genuinely-weak second-moment repair). Design-red-team fix.
- **ERASE — subspace erasure** (project out the excess-variance direction) — **negative control**, task-safety
  gated (structural, per 4F; cannot veto if task-destructive).

### Interpretability diagnostics (design-red-team wjdzttrhu — reported alongside every E4b−E3)
The estimator `dirs_exc = top-k eigvec(Σ_T − Σ_S)` mixes the injection with the natural target-vs-source domain
shift, so E4b−E3 only clears DELTA when the **injection dominates** the covariance gap (which re-enters the 4F
near-tautology). To keep pass/fail honest and interpretable we report, per fold/seed:
- **`injection_dominance_index`** = (injected variance along `v_c`) / (top-k excess eigenvalue mass). Near 1 ⇒
  the injection dominates (a pass there certifies little beyond "undo an injection we can fully see"); small ⇒
  the fair regime (E4b−E3 is expected small).
- **`est_dir_vc_overlap`** = `|dirs_exc · unit(v_c)|` (did the estimator recover the true direction?).
- **`arm_dir_overlap`** = `|dirs_exc(injected) · dirs_exc(clean)|` (is netting comparing matched vs mismatched
  axes?).
- **`oracle_e4b`** recovery + `oracle_beats_e3` → the `fail_attribution` field.
A pass is reported as **regime-conditional** with its dominance index; it is NOT read as general second-moment
repair.
- **Clean-target arms** for E4/E4b/E3 → token-specific NETTED = injected-effect − clean-effect (removes generic
  covariance-TTA). **Non-identity subset** reported (a second-moment injection should produce few/no mechanical-
  identity rows — itself a useful contrast to 4F's 73%).

## Pass / fail (inherits every Phase-4F correction)
Pooled over 8 confirm seeds × 21 folds × {injection type}, at `alpha_star`, NETTED; **all gate CIs CLUSTERED**
over `(dataset, target_subject)` folds. Constants `HARM_FLOOR=0.02`, `DELTA=0.02`, `SAFE_DROP=0.01`.

1. **Harm gate.** Pooled induced harm ≥ 0.02 AND clustered CI lower > 0. Else `none`.
2. **Mean-null sanity.** Batch mean displacement ≈ 0 AND **E4 first-moment does NOT explain most recovery**
   (E4 netted recovery small / not ≥ E4b). If E4 also repairs, the injection is not purely second-moment → report
   but do **not** license E4b scope expansion.
3. **E4b repair gate.** E4b task-safe (`clean_drop ≤ 0.01`, `raw_recovery > 0`, repaired > injected, not
   regression-to-floor) AND `E4b_netted_gain − E3_netted_gain ≥ DELTA` AND clustered CI lower > 0.
4. **E4b comparator gate.** E4b beats the structural veto set {E4, E3} on netted gain; ERASE negative-control
   only. **Leave-one-DATASET-out is BINDING** (the 4F lesson): a `strong` claim must survive dropping the
   carrying dataset; a `partial` reports the per-dataset split with the descriptive-N=2 caveat.

**Grades** (magnitude = point; nonzero-ness = clustered CI lower bound):
```
partial:  harm est AND mean-null AND E4b task-safe AND E4b beats E3 (>=DELTA, clustered CI_lo>0)
          AND E4b beats {E4,E3} AND E4b netted recovery > 0.30 (non-identity) AND per-dataset sign-consistent.
strong:   partial AND E4b netted recovery >= 0.50 AND leave-one-DATASET-out passes (both datasets est+specific).
none:     any gate fails; OR E4 explains most recovery (injection not second-moment); OR E4b <= E3.
```
`repair_claim_scope = controlled_second_moment_only`.

## Outputs (`results/fsr_phase4g_second_moment/`)
```
# harness:
phase4g_manifest.csv               # per fold/seed/injtype: alpha_star, lambda*, eps*, margin/scale
phase4g_injection_sanity.csv       # harm, E0-analog, injection-type
phase4g_mean_null_check.csv        # per fold/seed: batch mean displacement, E4 netted (should be ~0)
phase4g_covariance_shift_check.csv # target excess-variance vs source along injected direction
phase4g_repair_results.csv         # per fold/seed/injtype/alpha: E4/E4b/E3/ERASE injected+clean bAcc, raw+netted
phase4g_random_controls.csv        # E3/ERASE detail
phase4g_target_label_firewall.json
# aggregator:
phase4g_leave_one_dataset_out.csv
phase4g_verdict.json
```
`phase4g_verdict.json` (key fields):
```json
{"primary_branch": "spatial_z", "injection_type": "second_moment_controlled",
 "fresh_confirm_seeds": [20260721,"...",20260728], "harm_established": null,
 "mean_null_pass": null, "e4_first_moment_sufficient": null,
 "primary_repair": "E4b_second_moment_alignment",
 "e4b_task_safe": null, "e4b_netted_recovery": null, "e4b_netted_recovery_ci_clustered": [null,null],
 "e4b_minus_e3_netted_gain": null, "e4b_minus_e3_ci_clustered": [null,null],
 "e4_netted_recovery": null, "e4b_minus_e4_netted_gain": null,
 "leave_one_dataset_out_pass": null, "per_dataset_sign_consistent": null,
 "mechanical_identity_frac": null, "e4b_netted_recovery_nonidentity": null,
 "erase_valid_repair": false,
 "target_labels_used_for_fit": false, "target_labels_used_for_selection": false,
 "repair_claim_level": "none|partial|strong", "repair_claim_scope": "controlled_second_moment_only",
 "pc2_gpu_gate": "paused|eligible_for_review"}
```

## STOP rules
```text
1  target labels used for fit / injection / alpha / lambda-eps / comparator eligibility.
2  GPU / retrain / CMI / fbdualpc / hyperparameter sweep beyond frozen lambda,eps,k grid.
3  injection not strictly mean-nulled (batch mean displacement not ~0) -> not a second-moment control.
4  E4 first-moment explains most recovery but E4b scope-expansion still claimed.
5  leave-one-DATASET-out omitted, or a task-destructive arm allowed to veto E4b.
6  headline uses the ratio over the absolute netted gain, or omits the non-identity subset.
7  CLAIM-LOCK: written as learned / natural / general / DG / SOTA repair, or "E4b surgically removes";
   or presented as re-scoring Phase 4E/4F. Scope = controlled second-moment injected positive control ONLY.
```

## PC2 posture (unchanged)
`pc2_gpu_gate` starts `paused`. A 4G E4b ≥ partial makes it `eligible_for_review` (ledger/PC2-E4 readiness only,
FSR_31); **no GPU run** — PC2 GPU authorization needs PM review + ≥3 datasets + ρ dose-response + budget.
