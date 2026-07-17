# Mechanism-Subspace Oracle — PRE-REG AMENDMENT 02 (M0.2 IMPLEMENTATION CONTRACT). SPEC ONLY — M1 HOLD.

Freezes the implementation details so no two "YAML-conforming" implementations can reach different M1 verdicts.
Does NOT change the science (M0.1 amendment 01 stands). Config: `cmi_trace_mechanism_subspace_oracle_v3.yaml`.
Manuscript FROZEN. Only the PROJECT OWNER may explicitly stop a scientific line.

## P0.1 — contrast coding (Helmert) + weighting + fail-closed
Construction uses an orthonormal Helmert contrast matrix `H ∈ R^{(C−1)×C}` with `H Hᵀ = I`, `H 1 = 0`. For
subject d with class-mean matrix `M_d ∈ R^{C×p}` (row c = whitened mean of class c for subject d): `C_d = H M_d`
(the (C−1) whitened contrast rows). `C̄ = (1/m) Σ_d C_d`. Then
`G_shared = C̄ᵀ C̄ / (C−1)` and `G_dis = Σ_d (C_d − C̄)ᵀ (C_d − C̄) / (m (C−1))`. ALL subjects and classes equal
weight; a source subject MISSING any class -> that cell `FAILS_CLOSED` (no silent re-weighting). The 6 pairwise
contrasts (4-class) are saved as an INTERPRETIVE artifact only; they do NOT enter the scatter (Helmert spans the
same core subspace without pairwise redundancy).

## P0.2 — shrinkage η is executable
`η = 0.05 · tr(G_shared) / p`. If `tr(G_shared) < 1e-8` -> emit `TASK_MECHANISM_BELOW_RESOLUTION` (do NOT let ηI
manufacture eigenvectors from noise). `eta_rel ∈ {0.01, 0.10}` is a sensitivity sweep only (not the primary gate).

## P0.3 — dictionary + search budget (exhaustive, not greedy)
`dictionary_max_rank = 8`; `selected_rank ∈ {0,1,2,3}`; search = EXHAUSTIVE over ALL subsets of rank ≤ 3 of the
rank-8 dictionary (identity included) = `C(8,1)+C(8,2)+C(8,3) = 92` non-empty actions. Every random dictionary
has the SAME rank 8, the SAME 92-action exhaustive family, and the SAME `Y_cal` access. (Exhaustive avoids the
prefix/greedy expressivity gap that bit the earlier line.)

## P0.4 — readout + source-safety
All utility uses ONE fresh readout: source-standardized L2 logistic (`C=1`, `solver=lbfgs`), balanced accuracy
(== the deployable `_bacc`). Source safety = `mean_d [ bAcc_full_sourceLOSO − bAcc_P_sourceLOSO ] ≤ 0.02`; save
mean / median / worst-subject drop / positive-negative sign count (primary gate uses the MEAN only). Each basis
family reports BOTH the UNCONSTRAINED and the SOURCE-SAFE hindsight oracle (so unconstrained-positive but
safe-empty routes to training-time reshaping, not identity-masked).

## P0.5 — shared-overlap-matched random (executable + fail-closed)
For an informed dictionary B, its sorted per-direction profile `q_j = (b_jᵀ G_shared b_j)/tr(G_shared)`,
`q_(1) ≥ … ≥ q_(r)`. Each block: generate 5000 Haar-random rank-8 dictionaries, sort each q-profile, keep the
100 with min profile-L2 distance to B's profile (NO target outcome used), report matching RMSE. Fail-closed:
require `profile_rmse ≤ 0.02` AND `absolute_total_overlap_gap ≤ 0.01`, else `SHARED_MATCH_CONTROL_FAILED` (NEVER
silently downgrade to ambient while calling it task-overlap-matched).

## P0.6 — primary gate + multiplicity
CONFIRMATORY primary: basis = `contrast_disagreement`, backbone = `EEGNet`, both datasets. M1 enrichment (route
A) requires ALL of: (1) ≥1 dataset one-sided Holm-adjusted p<0.05 (across the confirmatory family); (2) same
dataset cluster-CI `LCB95(dU) > 0`; (3) other dataset not clearly harmful `UCB95(dU) > −0.01`. SECONDARY: DGCNN
positive only -> `BACKBONE_SPECIFIC`; rule/grad positive only -> `READOUT_OR_ESTIMATOR_DEPENDENT`; `B_cond`
positive -> negative-reference anomaly (control review). A secondary hit does NOT unlock M2 without an
independent confirmatory rerun or a new seed block.

## P0.7 — secondary estimators (exact)
`B_rule` = shared-plus-residual MULTIVARIATE RIDGE on class-centered one-hot targets: `W_d = W_0 + ΔW_d`,
`min Σ_d ‖Y_d − X_d(W_0+ΔW_d)‖² + λ0‖W_0‖² + λΔ Σ‖ΔW_d‖²`, primary `λ0=1, λΔ=10` (source-whitened, no target
tuning); `G_rule = Σ_d ΔW_dᵀ ΔW_d`; `λΔ ∈ {1,100}` sensitivity. `B_grad` = ONE fresh standardized logistic head
fit on ALL source; `g_{d,y} = E[∇_z ℓ | D=d, Y=y]`, `r_{d,y} = g_{d,y} − ḡ_y`, `G_grad = Σ_{d,y} r r ᵀ`
(class+subject balanced, rank-thresholded). DGCNN stored head = SENSITIVITY only (avoid EEGNet-fresh vs
DGCNN-stored estimator asymmetry in the primary).

## Basis builder return contract
Every builder returns `{raw_matrix, raw_singular_values, numerical_rank, orthonormal_basis (dictionary),
generalized_eigenvalues (contrast only), config_hash}`. Raw spectrum saved BEFORE orthonormalization.

## Pipeline (unchanged, restated)
1) source-only construction; 2) non-deployable subset/rank selection with `Y_cal` (exhaustive, ≤3); 3) score on
`T_query` (session-macro). Result routing per M0.1 amendment 01; every non-ENRICHED verdict writes
`{failure_layer, evidence, learned_lesson, next_hypothesis, next_experiment}`.

## Sequence
Stage B = this contract (first commit) + `tos_cmi/eval/mechanism_subspace.py` + runner + aggregator + the 17
tests. Stage C = real-EEG engineering smoke (2 ds × 2 backbones × 2 subjects × seed 0 × 4 families; random
blocks 2×10 in smoke, same action budget). Full M1 (126 cells) HOLD until Stage B/C are reviewed.
