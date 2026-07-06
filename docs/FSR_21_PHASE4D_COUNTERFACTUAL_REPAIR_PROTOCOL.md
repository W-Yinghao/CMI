# FSR_21 — Phase 4D: Counterfactual / Task-Protected Repair (pre-registration)

**Project FSR — Phase 4D.** Pre-registration of a *task-protected* repair for the injected harmful
branch-local shortcut of PC1 (FSR_19/FSR_20). PC1 proved the FSR protocol **detects, localizes, and
exactly attributes** a known harmful shortcut, but that **erasure is not a repair** — even *oracle*
subspace removal has negative recovery, because the harm rides a **task-coupled** (class-directed) logit
direction. Phase 4D asks whether a repair that **preserves the task direction instead of projecting it out**
can recover the harm where erasure cannot.

**Frozen before run** (this file committed to `project/functional-shortcut-reliance` and design-red-teamed
before any Phase-4D result exists). **CPU-only** on the frozen 4B dumps + checkpoints. **No GPU refit, no
CMI/fbdualpc, no architecture search, no hyperparameter sweep, no target-label fit, no backbone retrain.**
Only a small adapter on the frozen `spatial_z` branch is trained, **source-only**.

## Estimand (unchanged from PC1)
Per LOSO fold, at injection strength α, with the frozen backbone `head3(_fuse3(graph_z, temporal_z, ·))`:
```
recovery_fraction = (bAcc_repaired − bAcc_injected) / (bAcc_orig − bAcc_injected)
```
Pooled across folds (ratio of pooled means, not mean-of-per-fold-ratios — the PC1 small-denominator fix).
`bAcc_orig` = clean target bAcc; `bAcc_injected` = target bAcc after PC1 spatial-token injection; `bAcc_repaired`
= target bAcc after the repair arm. **Goal:** raise repaired above injected *while preserving task-useful
spatial information* (source-val clean-task safety gate), **without** blind subspace erasure.

## Injection (same construction as PC1; reproducible)
Primary branch `spatial_z`; α ∈ {1.0, 2.0} (PC1's repair grid; α never chosen by target bAcc). Token =
`unit(normalize(u_subj) + normalize(v_{class}))` via PC1's `token_matrix` **verbatim**; source-margin-normalized
`scale` with the PC1 `max(margin, 0.2)` floor and `s_shift`-at-`c_d`; source token uses source subject `u_d` +
source-assigned class `c_d` (majority source label, PC1 hash tie-break); target token uses `u_{tsub}` +
`c_target = hash(tsub) mod n_cls` (deterministic hash, **never** target labels). Class-directed
`v_{b,c}` = mean source `∂logit_c/∂z_b` through the frozen head, computed over **all** source samples
(`nsamp=len`, deterministic — no RNG-order dependence); the **same** `V` drives the injected `v_{c_target}` and
D2's task subspace `T`, so "protect `T` ⇒ protect the harm" holds by construction, and `k=2` is pinned for all
subspaces.

> **Reproducibility & comparability (design-red-team fix).** PC1's `u/c_target/cd` use Python `hash()`, which
> is `PYTHONHASHSEED`-salted per process; PC1 (FSR_20) was run **without** a pinned seed and dumped no tokens,
> so byte-identical token directions **cannot** be recovered retroactively. Phase 4D therefore pins
> `PYTHONHASHSEED=0` (recorded in `phase4d_repair_manifest.csv`) so its **own** injection is reproducible across
> processes, and establishes comparability with PC1 at the **induced-harm-magnitude** level: it reports pooled
> `induced_harm = bAcc_orig − bAcc_injected` and checks it lands in PC1's FSR_20 range (α=1 ≈ +0.04, α=2 ≈ +0.07).
> We do **not** claim byte-identity to the FSR_20 PC1 run.

## Repair arms (2 primary + controls, all source-fit, target labels score only)

### D0 — Exact token subtraction (attribution upper bound; **not** a repair claim)
Reuse PC1 R0: subtract the known per-sample token `−α·scale·tok_tgt`. Recovers ≈1.0 *because it knows the
exact per-sample nuisance vector*. Reported only to bound attributability; **not deployable**, **not** counted
as a repair pass.

### D1 — Counterfactual token-consistency adapter (**PRIMARY**)
A small residual MLP `A: spatial_z → spatial_z` (`A(z) = z + MLP(z)`, one hidden layer, fixed width),
trained **source-only** on the frozen branch latents; the backbone (`graph_z`, `temporal_z`, `_fuse3`,
`head3`) stays frozen. Repaired logits = `head3(_fuse3(graph_z, temporal_z, A(spatial_z)))`.

Counterfactual augmentation (source only): for each source **train** sample `(g_i, t_i, s_i, y_i)`, build the
clean view `s_i` and `K` token-injected views `s_i + α·scale·token_k`. Each `token_k = unit(unit(u_rand) +
unit(v_{c_k}))` where **`u_rand` is a fresh random unit direction** (the token family = {random subject-like
direction + a genuine class direction}) and — **critically (design-red-team BLOCKER fix)** — **`c_k` is drawn
uniform over classes, independent of `y_i`**. Drawing the token-class independent of the label forces the
CE+consistency objective to enforce token-**invariance** rather than token-**exploitation**: a source-correlated
class (`c_d` = majority label) would reward an adapter that *uses* the token, which then *amplifies* harm on the
target where `c_target` is arbitrary. The target token `u_{tsub}+v_{c_target}` lies in this family's
distribution (`u_{tsub}` a random-like direction, `v_{c_target}` a class direction seen in augmentation).
Loss (all fixed a priori, no sweep):
```
L = mean_k CE( head3(_fuse3(g_i, t_i, A(s_i + α·scale·token_k))), y_i )     # recover the task under the token
  + λ_cons · consistency( logits over the K views )                          # be invariant to which token
  + λ_l2   · MSE( head3(_fuse3(g_i,t_i,A(s_i))) , frozen_original_logits_i ) # stay close to clean original
```
`consistency` = variance of the K view-logits (push toward token-invariance). Fixed `K=4`, `λ_cons=1`,
`λ_l2=1`, hidden `64`, `epochs=150`, Adam `lr=1e-3` — declared in the harness header, unchanged after the run.

**Selection + u-generalization diagnostic (design-red-team fix).** Hold out a fraction of *source subjects*
(`val_subject_frac=0.3`). CE/consistency/l2 gradients use **train subjects only**. Pick the epoch with the best
**source-val injected-task bAcc**, where the val injection uses each **held-out subject's real `u_d`** + a
**y-decorrelated (uniform) class** — so a higher score genuinely means invariance (not exploitation) **and** the
metric doubles as the pre-registered **u-generalization diagnostic**: it measures invariance transfer to
subject directions never in the training gradient. No target labels, no target early-stopping. At eval, apply
the selected `A` to the **target injected** `spatial_z`.

**Fail-diagnosis (pre-registered).** `u_{tsub}` is a fresh direction; a source-only per-sample adapter might be
structurally unable to identify a token in an unseen direction. Therefore, if D1 fails on the target **and** the
source-val injected-task bAcc (u-generalization diagnostic) is also low, the fail is attributed to the
**token-shift / u-generalization gap**, *not* to the inadequacy of the task-protected repair idea. A clean
"repair inadequate" conclusion requires D1 to generalize invariance on held-out **source** subjects yet still
fail on target.

Rationale (why this is *task-protected*, not erasure): the adapter is trained to be **invariant to the token
family while preserving the source label**, so it *ignores the spurious token axis and keeps the genuine task
axis* — rather than projecting out the class-directed subspace (which PC1 showed also removes task signal). It
never sees target data or target labels; target labels are read only through a `TargetScorer.score()` guard
(final bAcc), whose access count is recorded in `phase4d_target_label_firewall.json`.

### D2 — Task-orthogonalized subject erasure (**SECONDARY**)
A principled erasure baseline. Estimate the subject subspace `S` from source **injected** `spatial_z`
(between-subject mean-scatter SVD, top-k). Estimate the task subspace `T` = span of the source class-directed
directions `{v_{b,c}}` (QR-orthonormalized). Remove **only the component of `S` orthogonal to `T`**
(`S_⊥T = S − S·Q_T·Q_Tᵀ`, re-orthonormalized), then erase `S_⊥T` from the **target injected** `spatial_z`.
*Predeclared expectation:* recovery is **limited**, because PC1's harm lives *in* `T` (class-directed) — so
protecting `T` protects the harm. D2 exists to (i) show a better-motivated erasure than blind LEACE/INLP still
under-repairs when harm is task-coupled, and (ii) contrast with D1.

### D3 — Random matched controls (**required**)
To show any D1 gain is not generic smoothing/regularization (design-red-team disambiguation — one control, no
ambiguity):
- **D3a random-perturbation adapter:** **identical** architecture / epochs / loss / selection rule to D1, with
  exactly one difference — its `K` per-view perturbations are **pure random unit directions of matched norm
  (`α·scale`) with NO class-`v` structure** (`token_k = unit(u_rand)`, no `+v_{c_k}`). So D1 and D3a differ
  *only* in whether the perturbation lives in the real token `(u+v_c)` subspace. If D1 does not beat D3a on
  pooled repaired bAcc, the token-subspace structure adds nothing.
- **D3b random-subspace erasure:** matched-dimension (`k=2`) random subspace erased from the target injected
  branch (control for D2). Reuses PC1's random-k.

### D4 — Injected, no repair (**baseline**)
PC1 R0 injected `bAcc_injected` — the floor every arm must beat.

## Sign / firewall
`task_drop = bAcc_orig − bAcc_erased` (as Phase 4B). Firewall: adapter fit, consistency targets, subspace
estimation, and **all** arm selection use source data + source labels only; α is fixed (not target-tuned);
target labels are used **only** to compute the final `bAcc_*` for scoring. `phase4d_target_label_firewall.json`
records this per fold.

## Pass / fail (pre-registered, PM Phase-4D criteria — design-red-team hardened)
Primary arm **D1**, at the **pre-registered constant α=1.0** (α=2.0 reported as a predeclared secondary; the
headline verdict is **α=1.0 only** — promoting α=2.0 or a max-over-α recovery to the headline is a STOP). All
quantities are **pooled over ALL folds** (ratio-of-pooled-means for recovery; no fold is dropped by any
target-harm threshold). Constants: `HARM_FLOOR=0.02`, `DELTA_BACC=0.02`, `SAFE_DROP=0.01`.

**Establish-harm gate (precondition).** If pooled `induced_harm = bAcc_orig − bAcc_injected` at α=1.0 is not
clearly positive (`≥ HARM_FLOOR` **and** bootstrap CI excludes 0), `recovery_fraction` is undefined →
`repair_claim_level = none` ("harm not established"). Denominator + CI recorded in the verdict.

**`strong`** (requires all):
```
D1 recovery_fraction ≥ 0.70
D1 beats D3a on POOLED REPAIRED bAcc by ≥ DELTA_BACC AND the bootstrap CI of (D1_repaired − D3a_repaired) > 0
source-val clean-task drop (mean) ≤ SAFE_DROP        (task-safety gate)
pooled bAcc_repaired > pooled bAcc_injected           (target improved)
```

**`partial`** (requires all):
```
0 < D1 recovery_fraction < 0.70
D1 beats D3a on pooled repaired bAcc (point margin > 0)
source-val clean-task drop (mean) ≤ SAFE_DROP
pooled bAcc_repaired > pooled bAcc_injected
```
→ "counterfactual repair **substantially (if 0.50–0.70) / partially (if <0.50)** recovers the injected shortcut
harm." The `0.50–0.70` band is **`partial`**, never written with the word "strong".

**`none`** (any of): harm not established, OR D1 ≤ D3a, OR clean-task safety fails, OR `bAcc_repaired ≤
bAcc_injected`. → "FSR verifies and localizes harmful shortcuts, but repair remains unresolved; erasure **and**
a simple counterfactual head are insufficient." (Not a project failure — it sharpens
verification-vs-intervention. If D1's u-generalization diagnostic is also low, the fail is a token-shift
artifact, not a repair-idea verdict.)

`repair_claim_level ∈ {none, partial, strong}`, set by the aggregator. **Erasure arms D0/D2/D3b are contrast
baselines only and are excluded from the headline (`erasure_arms_excluded_from_headline=true`).**

## Outputs (`results/fsr_phase4d_repair/`)
```
phase4d_repair_manifest.csv          # per fold: dataset, target_subject, alpha, scale, arch/loss knobs
phase4d_source_val_selection.csv     # per fold: selected epoch, source-val injected-task bAcc, clean-task drop
phase4d_counterfactual_consistency.csv  # per fold: pre/post consistency (view-logit variance), gain
phase4d_target_recovery.csv          # per fold/alpha: bAcc_orig, injected, D0/D1/D2/D3a/D3b repaired + recovery
phase4d_random_control.csv           # per fold/alpha: D1 − D3a recovery margin
phase4d_verdict.json                 # pooled recovery, pass flags, repair_claim_level, firewall flags
phase4d_target_label_firewall.json   # per-fold firewall attestation
```
`phase4d_verdict.json` schema (key fields):
```json
{"counterfactual_repair_pass": null, "repair_claim_level": "none|partial|strong",
 "primary_alpha": 1.0, "alpha_is_preregistered_constant": true, "alpha_selection_used_target": false,
 "harm_established_alpha1": null,
 "injection_harm_denominator_alpha1": null, "injection_harm_denominator_alpha1_ci": [null, null],
 "recovery_fraction_alpha1": null, "recovery_fraction_alpha1_ci": [null, null], "recovery_fraction_alpha2": null,
 "d1_repaired_bacc_alpha1": null, "d3a_repaired_bacc_alpha1": null, "injected_bacc_alpha1": null,
 "d1_minus_random_bacc_alpha1": null, "d1_minus_random_bacc_alpha1_ci": [null, null],
 "d1_minus_random_bacc_alpha2": null, "delta_bacc_margin": 0.02,
 "random_control_beaten_point": null, "random_control_beaten_strong": null,
 "target_improved_over_injected": null,
 "source_val_task_safe": null, "source_val_clean_task_drop_mean_alpha1": null,
 "source_val_clean_task_drop_ci_alpha1": [null, null],
 "u_generalization_val_inj_bacc_alpha1": null,
 "task_orth_erasure_recovery_alpha1": null, "randk_erasure_recovery_alpha1": null,
 "exact_subtraction_recovery_alpha1": null, "erasure_arms_excluded_from_headline": true,
 "target_labels_used_for_fit": false, "target_labels_used_for_selection": false,
 "target_labels_used_for_final_eval_only": true}
```

## STOP rules
```text
1  target labels used for adapter fit / consistency target / subspace estimation / arm or epoch/alpha selection.
2  Phase 4D requires GPU or a backbone retrain.
3  any architecture / hyperparameter *sweep* (only one fixed config is allowed).
4  recovery reported on selected positive folds only (must pool ALL folds; no fold dropped by target harm).
5  D1 not compared against the D3a random-structure control (pooled repaired-bAcc margin + bootstrap CI).
6  source-val clean-task safety gate skipped or computed with target labels.
7  D1 gain claimed while D1 ≈ D3a (generic-regularization confound not ruled out).
8  epoch/arm/ALPHA selection touches target bAcc; or the α=1.0-primary rule is derived from harm, not a constant.
9  the headline verdict promotes α=2.0 (or a max-over-α recovery) instead of the pre-registered α=1.0.
10 recovery_fraction reported when the establish-harm gate fails (denominator not clearly positive).
11 CLAIM-LOCK: any output frames ERASURE (D2 task-orth or D3b random-k) as improving target — regardless of
   their measured recovery, D2/D3b are contrast baselines only and cannot carry the repair headline.
12 CLAIM-LOCK: results written as (i) a natural shortcut (vs injected positive control), (ii) a new DG method
   or SOTA, or (iii) "spatial leakage is (naturally) harmful". All three framings are stop-worthy verbatim.
```

**Disclosed limitation (not a STOP).** With only 8–13 source subjects, the ~0.3 source-subject val split
(2–3 subjects) is used for *both* epoch selection and the clean-task safety gate; the gate is therefore mildly
optimistic and its 0.01 threshold is within subject-sampling noise. It is reported with a bootstrap CI over
source subjects and evaluated at the selected epoch on clean (no-token) inputs; treat a bare pass as suggestive,
not definitive.

## Framing (fixed)
Phase 4D repairs an **injected** positive-control shortcut, not a natural one (Phase 4B verdict remains
`NO_VERIFIED_HARMFUL_BRANCH_SHORTCUT`). A D1 pass licenses *"a counterfactual / task-protected repair recovers
a known harmful branch-local shortcut where erasure (even oracle) fails"* — it does **not** license a natural
DG method or a claim that source prevalence creates this shortcut. If D1 fails, the honest result is that
verification succeeds but repair is unresolved. **PC2 (learned-shortcut GPU refit) is GO only if D1 reaches at
least a moderate pass here** (per PM); otherwise PC2 stays a paper protocol.
