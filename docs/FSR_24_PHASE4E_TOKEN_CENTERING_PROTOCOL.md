# FSR_24 — Phase 4E: Branch-local Token-Neutralization Repair (pre-registration)

**Project FSR — Phase 4E.** Pre-registration of a **new** repair primitive, replacing the failed Phase-4D
counterfactual head (FSR_22: `repair_claim_level = none`). **Frozen before the result run + design-red-teamed
(w4adgd9z3).** CPU-only on the frozen 4B dumps + checkpoints; **no GPU, no retrain, no CMI/fbdualpc, no
architecture search, no target-label fit.**

> **Design-red-team reframing (material — flagged to PM).** The PM's proposal made **E1 (subspace-restricted
> centering)** the primary and **E4 (full-space centering)** a control. The red-team proved this is backwards:
> the injected `tok_tgt = unit(unit(u_{tsub}) + unit(v_{c_target}))` is a **constant per-batch offset**, so the
> full mean difference `mean(z_T) − μ_src` contains **100%** of it, whereas E1's subspace `S`
> (between-subject scatter of injected source) **cannot span `u_{tsub}`** — a *fresh* per-subject direction not
> among the source `u_d` (in dim 32, `‖P_S u_{tsub}‖ ≈ √(k/32) ≈ 0.18–0.35`). E1 can only touch the
> class-directed `v_{c_target}` half — the exact task-coupled direction where erasure already gave *negative*
> recovery. So **E4 (full first-moment mean alignment) is the deployable, strictly-stronger token remover**, and
> E1 is geometrically incapable of removing the token's dominant half. Accordingly: **E4 is the primary
> deployable arm; E1 is secondary** ("does restricting to `S` add anything?"), and the whole thing is netted
> against a **clean-target arm** because mean-alignment is generic TTA. This *serves* the PM's stated goal
> (estimate + neutralize the additive token without breaking task signal) — it just uses the mechanistically
> correct operator.

## Mechanistic premise (verified against PC1 code)
`tok_tgt` rows are identical across the target batch (subject + class fixed), so the harm is a **constant
additive offset** `α·scale·tok_tgt` on every target `spatial_z`. Hence (FSR_20/22): E0 exact subtraction → 1.0;
subspace **erasure** → negative (kills task *variance*); a counterfactual head → ties random. The correct repair
subtracts the **batch-mean offset** (constant, preserving within-batch task variance) — the question is whether
a *deployable* (target-unlabeled) mean estimate does this, and whether it is genuine token removal vs generic
first-moment TTA.

## Reproducible, seedable, multi-seed injection
Token built from an explicit `token_seed` via a deterministic `sha256`-derived RNG (**not** Python `hash()`), so
it is reproducible across processes. `u_d, u_{tsub}` = unit vecs from `default_rng(seed_int(token_seed,"u",id))`;
`c_target = seed_int(token_seed,"ct",tsub) mod n_cls`; `c_d` = source majority (tie-break
`seed_int(token_seed,"c",d)`); `v_{b,c}` = mean source `∂logit_c/∂z_b` over **all** source samples (deterministic);
`scale` = PC1 rule (`max(median source margin, 0.2)` floor). **Seeds: `DEV_SEED=0`** (development/mechanism, may
be inspected freely) **and three fresh confirm seeds `CONFIRM_SEEDS = [20260707, 20260708, 20260709]`** (pinned
here **before** the run). **The verdict is driven by the multi-seed CONFIRM aggregate** (M6); DEV_SEED is
mechanism-only and may not inform any confirm-seed choice (STOP-10).

## α selection — source-only stress rule (the ONLY α mechanism; overrides any grid argmax)
Per fold/seed, over `α ∈ {0.5, 1.0, 1.5, 2.0}`:
```
alpha_star = smallest α such that the SOURCE-heldout injected class-directed logit shift
             >= FRAC * (median source correct-vs-runnerup margin),   FRAC = 1.0
```
`FRAC=1.0` (justified independently of FSR_22: require the injection to move a held-out source decision by **at
least one full source margin** — a source-defined "the shortcut is real" threshold, not a target-harm knob). α
is **never** chosen by target bAcc/harm, and α_star is **not** folded into the hyperparameter argmax. Per-fold
`stress_unmet` (no grid α meets threshold → α_star=2.0 fallback) is **disclosed** in the verdict so a degenerate
"always 2.0" collapse is visible.

## Repair arms (all use target X only + source; target labels score only)
`μ_src` = **per-class-averaged** mean of clean source `spatial_z` (source-only; per-class to avoid the
class-prior confound — P2); `z_T` = injected target batch; `S` = top-`k` between-subject mean-scatter of
injected source (source-only); `P_S` = projector onto `S`.

- **E0 — exact token subtraction** (attribution upper bound, **non-deployable**, excluded from headline):
  `z − α·scale·tok_tgt` → recovery ≈ 1.0 (internal sanity anchor).
- **E4 — full-space first-moment mean alignment** (**PRIMARY, deployable**): `z − λ·(mean(z_T) − μ_src)`.
  Removes the full constant offset (⊇ the whole token). Deployable (target-X + source mean).
- **E1 — subspace-restricted centering** (**SECONDARY**): `z − λ·P_S(mean(z_T) − μ_src)`. Tested only as "does
  restricting to the source token subspace add anything over E4?" Expected: no (u-part out of `S`).
- **E2 — counterfactual token marginalization** (**exploratory**): `z_clean = z − λ·P_S(mean(z_T)−μ_src)`; then
  `logits = mean_{c} head3(_fuse3(g, temporal, z_clean + α·scale·unit(V[c])))` over the balanced class set.
- **E3 — random-subspace centering** (**control**): E1 with `S` a random `k`-dim subspace (sanity check, not a
  sufficient gate — S is token-aligned by construction so E1>E3 is near-tautological, P1).
- **ERASE — subspace erasure** (**named control**, = PC1 R2): `z − P_S(z)` (projection, kills variance). E1/E4
  must **beat erasure**.
- **CLEAN-target arms** (**required, M2**): E4 and E1 applied to the **un-injected** target. Define
  **token-specific effect** `= [bAcc(E on injected) − bAcc_injected] − [bAcc(E on clean) − bAcc_orig]` — nets
  out generic mean-alignment TTA. **The verdict uses the netted (token-specific) recovery**, not raw recovery.
- **injected floor**: `bAcc_injected`.

`recovery_fraction = (bAcc_repaired − bAcc_injected)/(bAcc_orig − bAcc_injected)` (raw); **netted recovery**
subtracts the clean-target TTA gain in the numerator. Pooled ratio-of-means over folds×confirm-seeds; anti-harm
folds (denominator ≤ 0) handled by a pre-declared **difference-of-means** fallback and disclosed count (M6).

## Mechanism-capture instrumentation (M1 — computed + dumped BEFORE any target scoring)
Per fold/seed, into `phase4e_mechanism_capture.csv`: `cos(unit(u_{tsub}), S)`; `captured_fraction =
‖P_S(α·scale·tok_tgt)‖ / ‖α·scale·tok_tgt‖`, split into u-part vs v-part capture. **Pre-registered:** if median
`captured_fraction < 0.5`, E1 is declared **mechanistically out-of-scope for the u-part** and its ceiling is
geometric, not scientific — E1 may then only be reported as "centers the in-`S` (v/domain) component".

## Source-only selection (no target labels; no target-harm gating; re-derived per seed — M5)
`k ∈ {1,2,4}`, `λ ∈ {0.5,1.0}` selected on a **source-heldout pseudo-target** (hold out one source subject,
inject with the source-only token protocol, estimate `S/μ_src/δ` from the remaining source, score with **source**
labels; rotate + average). Select `(k,λ)` maximizing source-heldout **netted** recovery subject to
source-heldout task-safety (clean drop ≤ 0.01) and a positive gap over E3-random. **STOP-10:** the selection is a
pure deterministic source-only function **re-derived from scratch on each confirm seed**; DEV_SEED real-target
recovery may **not** inform `(k,λ)` reused on confirm seeds. `phase4e_manifest.csv` emits dev-vs-confirm
`(k*, λ*, α_star)` side-by-side; an external pass must reproduce the selection reading **zero** target labels (P4).

## Pass / fail (pre-registered; multi-seed CONFIRM aggregate; NETTED recovery)
Establish-harm gate at `α_star` (pooled netted denominator `bAcc_orig − bAcc_injected` ≥ 0.02 AND boot CI > 0,
else `none`). Constants `SAFE_DROP=0.01`, `DELTA=0.02`. **Primary = E4** (deployable); E1 is judged only as a
possible improvement over E4.

- **strong**: E4 **netted** recovery ≥ 0.50 AND E4 beats ERASE and E3 on pooled netted bAcc (margin ≥ DELTA,
  bootstrap CI > 0) AND source-heldout selection clean AND pooled netted bAcc_repaired > bAcc_injected AND
  firewall clean.
- **partial (moderate)**: E4 netted recovery > E3/ERASE (point) AND netted bAcc_repaired > bAcc_injected AND
  source-heldout hyperparams transfer, but recovery < 0.50.
- **none**: E4 netted ≤ E3/ERASE, OR netted bAcc_repaired ≤ bAcc_injected, OR harm not established, OR
  hyperparams do not transfer.
- **E1 sub-claim** ("token-subspace neutralization adds value"): licensed **only** if E1 netted > E4 netted with
  bootstrap CI > 0 AND attributable to task preservation. **If E4 ≥ E1 within CI → the claim is "first-moment
  target-batch mean alignment neutralizes the constant token; the S-subspace machinery adds nothing"** (M3).

`repair_claim_level ∈ {none, partial, strong}`. **E0 (oracle) and E3/ERASE (controls) excluded from headline.**

## Outputs (`results/fsr_phase4e_token_centering/`)
```
phase4e_manifest.csv                     # per fold/seed: token_seed, alpha grid, alpha_star, stress_unmet, k*, λ*, dev-vs-confirm
phase4e_mechanism_capture.csv            # cos(u_tsub,S), captured_fraction (u/v split) -- BEFORE scoring
phase4e_source_heldout_selection.csv     # source-heldout netted recovery / task-safety / random-gap per (k,λ)
phase4e_alpha_rule.csv                   # per fold/seed: source-heldout logit-shift vs FRAC*margin, alpha_star
phase4e_token_centering_results.csv      # per fold/seed/alpha: E0/E1/E2/E3/E4/ERASE injected+clean bAcc, raw+netted recovery
phase4e_clean_target_netting.csv         # per fold/seed: E on clean vs injected -> token-specific netted effect
phase4e_random_controls.csv              # E3/ERASE detail
phase4e_fresh_seed_confirmatory.csv      # 3 CONFIRM seeds aggregate (verdict driver)
phase4e_target_label_firewall.json       # TargetScorer read budget per fold
phase4e_verdict.json
```
`phase4e_verdict.json` (key fields):
```json
{"fresh_token_seeds": [20260707,20260708,20260709], "dev_seed": 0,
 "alpha_selected_by_source_only": true, "alpha_star_confirm_per_seed": null, "stress_unmet_frac": null,
 "harm_established_confirm": null, "mechanism_captured_fraction_median": null, "e1_mechanistically_in_scope": null,
 "primary_repair": "E4_full_mean_alignment",
 "e4_netted_recovery": null, "e4_netted_recovery_ci": [null,null], "e4_raw_recovery": null,
 "e1_netted_recovery": null, "e1_minus_e4_netted_bacc": null, "e1_minus_e4_netted_bacc_ci": [null,null],
 "e1_adds_value_over_e4": null,
 "e3_random_recovery": null, "erase_recovery": null, "exact_subtraction_recovery": null,
 "clean_target_tta_gain": null, "source_heldout_transfers": null, "source_val_task_safe": null,
 "target_labels_used_for_fit": false, "target_labels_used_for_selection": false,
 "repair_claim_level": "none|partial|strong", "pc2_gpu_gate": "paused|eligible"}
```

## STOP rules
```text
1  target labels used for E1/E2/E4 fit, S/μ estimation, k/λ/alpha selection, or dictionary construction.
2  Phase 4E requires GPU / retrain / CMI / fbdualpc / new model training.
3  hyperparameter SWEEP beyond the pre-declared source-heldout grid (k in {1,2,4}, λ in {0.5,1.0}).
4  recovery reported on selected positive folds only (pool ALL folds x confirm seeds; no fold dropped by harm).
5  primary (E4) not compared against E3 random AND ERASE (pooled netted bAcc + bootstrap CI).
6  α chosen by target bAcc/harm; headline recovery reported at any α other than the source-selected alpha_star.
7  DEV_SEED (freely inspected) used as the sole basis for the claim (verdict = multi-seed CONFIRM aggregate).
8  headline uses E0 (oracle) or E3/ERASE (controls); or E1 crowned while E4 >= E1 within CI.
9  recovery reported WITHOUT netting out the clean-target TTA effect (raw-only claim).
10 DEV_SEED real-target recovery (any grid cell) used to pick k/λ/alpha_star reused on a CONFIRM seed; selection
   must be re-derived source-only per seed and externally reproducible with zero target-label reads.
11 CLAIM-LOCK: results written as (i) a natural shortcut, (ii) a DG method / SOTA, or (iii) "spatial leakage is
   naturally harmful". This repairs an INJECTED positive-control shortcut only. E0/E3/E4 recovery magnitudes are
   NOT byte-comparable to PC1/FSR_20/22 (new sha256 token + all-sample V); compare within-phase only.
```

## Framing (fixed)
Phase 4E asks whether a **deployable, target-unlabeled first-moment mean-alignment** (E4), optionally restricted
to the source token subspace (E1), can **neutralize a known injected constant additive shortcut** while
preserving task-coupled spatial signal — netted against generic TTA. Most likely honest outcome (red-team
prediction): E4 ≈ E0 on the token but the gain is largely first-moment TTA (nets down against clean-target), and
E1's subspace restriction adds nothing. A pass licenses *"deployable first-moment mean alignment neutralizes the
constant token (net of TTA)"* — **not** a DG method, SOTA, natural-harm claim, or "token-subspace" claim unless
E1 > E4 with CI. **PC2 GPU gate: GO only if E4 (or E1) ≥ moderate (partial) pass, net of TTA.**
