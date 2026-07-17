# Mechanism-Subspace Oracle — PRE-REG AMENDMENT 03 (SHARED-NULL CONDITIONAL ESTIMAND). SPEC ONLY — M1 HOLD.

Supersedes the P0.5 shared-overlap rejection-sampling control of amendment 02 and re-states the estimand. Approved
by PM 2026-07-17 after the Stage-C smoke + a 4-agent adversarial panel showed the amendment-02 primary control was
infeasible by construction and the total-energy alternative was the wrong invariant. Config: `cmi_trace_
mechanism_subspace_oracle_v4.yaml`. Manuscript FROZEN. Only the PROJECT OWNER may explicitly stop a scientific
line; every negative routes with a graded verdict + failure record, never a closeout word.

## The estimand (restated, sharper)
Let `C̄ = mean_d C_d` be the shared class-contrast mechanism (Helmert contrasts, source Ledoit-Wolf-whitened
metric). Its row space `S = row(C̄) = range(G_shared)` has `rank(G_shared) ≤ C−1` (≈3 for 4-class BNCI2014, ≈1 for
2-class BNCI2015 at p=16). Let `N ∈ R^{p×q}`, `span(N) = S⊥` be the NUMERICAL null space of the shared mechanism
(SVD threshold τ = 1e-7 on `C̄`). Inside that null space define the projected disagreement
`G_dis^N = Nᵀ G_dis N`. The mechanism dictionary is

    B_mech = N · TopEig(G_dis^N).

**Scientific question (confirmatory):** *within the feasible space that does NOT delete the shared class-contrast
mechanism, are the directions most aligned with cross-subject mechanism DISAGREEMENT more enriched for
future-session-harmful deletable mechanism than RANDOM directions in the SAME shared-null space?*

This is a CONDITIONAL RANDOMIZATION test: hold "shared-mechanism overlap = 0" fixed (matched), vary only
"alignment to G_dis" (the object under test). The generalized eigenproblem `G_dis v = ρ(G_shared+ηI)v` of
amendment 02 is retained ONLY as a sensitivity basis (it approximates the same null-space object because for
v∈ker(G_shared), ρ(v)=vᵀG_dis v/(η‖v‖²)); the explicit null-projected eigensystem above is now PRIMARY.

## A03.1 — Primary contrast basis (explicit null projection)
    shared_span   = row(C̄)                                  # (C-1)-dim
    N             = null(C̄) via SVD, singular-value threshold τ = 1e-7        # q ≈ p-(C-1)
    G_dis^N       = Nᵀ G_dis N                               # q x q
    r             = min(8, numrank_{1e-7}(G_dis^N))          # dictionary rank
    B_contrast    = N · TopEig_r(G_dis^N)                    # p x r, ambient-orthonormal
NON-DEGENERACY GATE (fail-closed, do NOT manufacture a no-variation control):
    require  q > r  (null_dim > dictionary_rank)  AND  numrank(G_dis^N) > 0 .
If `q ≤ r` → emit `SHARED_NULL_CONTROL_LOW_DOF` (the null space is too small to admit a distinct random control);
if `numrank(G_dis^N)=0` → `TASK_MECHANISM_BELOW_RESOLUTION`. Sensitivity: generalized-eig basis of amendment 02.

## A03.2 — Primary specificity control = SHARED_NULL_HAAR (conditional randomization)
Draw the random control as Haar-random rank-`r` subspaces WITHIN the shared-null space:
    B_rand ~ Haar( Gr(r, S⊥) )  ==  N · Haar(Gr(r, q))   (orthonormalized in ambient coords).
Per cell: **2 independent blocks × 100 dictionaries** each. Every random dictionary shares with the informed
dictionary: the SAME shared-null feasible space, the SAME rank r, the SAME exhaustive action family (subsets
rank≤3), the SAME `Y_cal` access, and the SAME source-safety filter. It differs ONLY in alignment to `G_dis`.
This is NOT confounding — it is conditional randomization on `shared_overlap=0`. Degeneracy is checked, not
assumed: report the informed-vs-random subspace-overlap DISTRIBUTION and `G_dis` capture fractions; if the random
draws have ~no geometric/utility variation (e.g. q≈r), the cell is flagged `SHARED_NULL_CONTROL_LOW_DOF`, not
silently accepted. AMBIENT Haar (whole `R^p`) is retained as a generic low-rank SECONDARY control. total-energy
`(G_shared+G_dis)` matching is REJECTED (it lets the control carry 27–48% deployable shared content, whose
deletion hurts +0.09–0.13 source bAcc, so the arms are not equally task-bearing; and it collapses to ambient at
p=16 while failing the powered gate at n_keep=100).
NOTE: shared-null guarantees only ZERO AVERAGE class-contrast mechanism, NOT full task-safety, so the source-LOSO
safety gate (P0.4) still runs INDEPENDENTLY on both informed and random.

## A03.3 — One fold-level G_shared shared by all families; null-projected secondaries
`G_shared` (and hence `N`) is computed ONCE per fold from the contrast construction and REUSED by every family, so
rule/grad/B_cond all get a matched shared-null control (amendment 02 gave only contrast a matched control). Primary
secondary bases are NULL-PROJECTED:
    B_rule^N = N · Eig_r(Nᵀ G_rule N),   B_grad^N = N · Eig_r(Nᵀ G_grad N).
Full/unprojected rule & grad bases are retained as SENSITIVITY only.

## A03.4 — Required diagnostics (per informed AND per random dictionary, all persisted)
    shared_null_dim, dictionary_rank, selected_indices, selected_rank, Gdis_capture_fraction,
    shared_overlap, source_safety_{mean,median,worst}, calibration_utility, query_utility,
    projector_hash, dictionary_hash, random_block, random_id, config_hash, git_sha, feature_hash
Also report per cell: `tr(P_B G_dis) / tr(P_N G_dis)` (fraction of null-space disagreement captured by the
selected projector) and the informed–random subspace-overlap distribution.

## P0.1 — SYMMETRIC safety gate (safe-vs-safe, unc-vs-unc)
Informed and random controls must use the SAME safety treatment on both arms. Report BOTH contrasts separately:
    ΔU_safe_specific = U_informed_safe − E[U_random_safe]        (both source-LOSO-safe filtered)
    ΔU_unc_specific  = U_informed_unc  − E[U_random_unc]         (both unconstrained)
FORBIDDEN: comparing safe-informed against unconstrained-random (the amendment-02 asymmetry). The confirmatory
gate uses ΔU_safe_specific; ΔU_unc_specific is reported alongside.

## P0.2 — B_rule = EXACT joint shared-residual ridge (not the pooled-then-residual approximation)
Solve the joint convex program exactly:
    min_{W0,{ΔW_d}}  Σ_d ‖Y_d − X_d(W0+ΔW_d)‖² + λ0‖W0‖² + λΔ Σ_d‖ΔW_d‖²,   λ0=1, λΔ=10.
Via block normal equations / sparse Cholesky (or block-coordinate descent to a KKT tolerance). Add a KKT-residual
test (stationarity ‖∇‖ ≤ tol at the returned solution). `G_rule = Σ_d ΔW_dᵀ ΔW_d`. λΔ∈{1,100} sensitivity.

## P0.3 — B_grad = EQUAL-SUBJECT class-conditional gradient
(1) per subject-class cell `g_{d,y} = E[∇_z ℓ | D=d, Y=y]`; (2) equal-weight over subjects
`ḡ_y = (1/m) Σ_d g_{d,y}` (NOT a trial-count-weighted pooled mean); (3) all `(d,y)` residuals `r_{d,y}=g_{d,y}−ḡ_y`
enter `G_grad = Σ r rᵀ` with equal weight. ONE fresh source-standardized logistic head. Trial-duplication
invariance: duplicating one subject's trials must not systematically rotate the basis (tested).

## P0.4 — Cell-specific random seeds + two blocks
Random dictionary seed = hash(dataset, backbone, heldout_subject, model_seed, basis_family, control_family,
block_id, random_id) — so Monte-Carlo error is INDEPENDENT across cells (amendment-02 reused block 0/1 across all
cells). BOTH the shared-null-Haar and ambient controls run 2 independent blocks (amendment 02 ran the matched
control as a single block).

## P0.5 — Full builder + artifact contract
Every builder returns `{raw_matrix, raw_singular_values, numerical_rank, orthonormal_basis,
generalized_eigenvalues (contrast/sensitivity), config_hash}` with the raw matrix persisted BEFORE
orthonormalization. Every informed AND random dictionary row is written individually (not just a random-utility
vector), with selected subsets and safety statistics, so each projector and control is reconstructable post-M1.

## Statistics — EXACT sign-flip confirmatory permutation p
Effect uncertainty: subject-cluster bootstrap CI (10k) as before. CONFIRMATORY p: aggregate `dU_safe_specific` to
one value per target subject, then an EXACT one-sided SIGN-FLIP permutation test over subjects (BNCI2014 ≤ 2⁹=512,
BNCI2015 ≤ 2¹²=4096 sign patterns — enumerate exactly). Holm across the two confirmatory dataset p-values. The
non-centered bootstrap p is retained as SENSITIVITY only, never the sole confirmatory p. Route A (per
route_stage_result) requires: contrast/EEGNet, LCB95(ΔU_safe_specific)>0, Holm-adjusted sign-flip p<0.05 on ≥1
dataset, other dataset UCB95>−0.01, AND `specificity_control == SHARED_NULL_HAAR` (matched control actually ran).

## Blocker 1 — DGCNN keeps the SAME future-session estimand (no session-free split)
DGCNN must be evaluated with the identical calibration-session→future-session macro split as EEGNet. Route: (1)
strict metadata BACKFILL of a `session_target` sidecar iff a trial-level key gives element-wise `y_target`
agreement + deterministic order (count-match alone is INSUFFICIENT); else (2) frozen DGCNN re-inference persisting
`trial_id, session_source, session_target, run_id, event_id, y_{source,target}, subj_source, feature_hash,
checkpoint_hash`, with a feature-PARITY audit (max|Z_new−Z_old|, label identity, logits identity, order identity).
A session-free random split for DGCNN is NOT approved.

## Sequence
D1 = this amendment + config v4 (first commit). D2 = implementation repair. D3 = DGCNN metadata backfill or
re-inference. D4 = the 13 pinned tests. D5 = 4th real-EEG engineering smoke (2ds × 2bb × 2subj × seed0 × 4
families; random 2×10, same shared-null support / actions / safety / firewall / artifact pipeline; engineering
only — NO scientific weight). Full M1 (126-cell fail-resumable array) HOLD until D1–D5 pass PM review.
