# REVIEW_P0_FROZEN — preregistration for the bounded P0 correction phase

Branch: exp/h2cmi-review-p0-corrections, forked from exp/h2cmi-responsibility-qxu @ 09e9249.
Env: conda 'icml' — torch 2.8.0+cu128, moabb 1.2.0, mne 1.8.0, scikit-learn 1.5.2, scipy 1.13.1,
numpy 1.26.4. NO push. Frozen historical tags/artifacts must NOT be altered, overwritten, or retagged.
This is a CORRECTION phase, NOT method search. Committed BEFORE any outcome-producing run.

## The two P0 mismatches (independently audited + VERIFIED in code)
P0-1 DECISION-PRIOR CONFOUND. `current_joint` evaluates with the joint's ESTIMATED prior fj.pi_T as the
  DECISION prior (run_v2.py:140-144; run_w1_mi.py:45-47; run_w2_sleep.py:158-160) while identity/pooled/
  CC use uniform. So the prior-M-step "harm/help" delta conflated GEOMETRY with the DECISION PRIOR.
P0-2 V2P POOL CONSTRUCTION. run_v2p._pool_indices takes `seed` but never uses it and slices i0[:n0]/
  i1[:n1] (contiguous-first), so different prevalence RATIOS used DIFFERENT trial subsets -> geometry
  movement was confounded with trial-set differences (not the same trials reweighted).
Also: W1/W2 trained ONE source seed (build_cfg/sleep_cfg seed=0); W2 used the FIRST 20 valid paired-
night subjects (benchmark_subjects -> two[:20]).
NOTE: old frozen results stand as-is under their tags; this phase produces NEW, separately-named
artifacts and does not reinterpret them.

## A. Separate FIT prior from DECISION prior (frozen)
Per target unit, fit joint EM EXACTLY ONCE -> (T_J geometry, pi_J estimated prior). Produce four
predictions from the SAME (T_J, pi_J):
    identity_uniform            = decode(I,   uniform)
    identity_joint_prior        = decode(I,   pi_J)
    joint_geometry_uniform      = decode(T_J, uniform)   # joint-FIT geometry, uniform decision
    joint_geometry_joint_prior  = decode(T_J, pi_J)      # == the old current_joint
"joint_geometry_uniform" = the joint-fit geometry evaluated with a UNIFORM decision; it is NOT a
geometry estimator that excluded the prior M-step. Balanced-accuracy PRIMARY results ALWAYS use the
uniform decision prior.
Also fit (all decoded with uniform unless noted):
    fixed_iterative_geometry_uniform   (gen_iterative, prior pinned at reference; uniform decision)
    fixed_reference_oneshot_uniform    (gen_oneshot_diag; uniform)
    pooled_uniform                     (pooled_empirical_diag; uniform)
    latent_im_diag_uniform             (renamed from SPDIM; latent IM-recentering diag; uniform)
    source_recolored_ea                (renamed from EA; raw-space recolor-to-source)
REAL-DATA PRIOR-M-STEP GEOMETRY CONTRAST = fixed_iterative_geometry_uniform - joint_geometry_uniform.
Save for every branch: pi_J, transform (log-scale a + translation b), transform norm, prediction hash,
probabilities, balanced acc, ordinary acc, macro-F1, NLL, Brier, ECE.
EXACT prediction decomposition (must verify numerically, |residual|<1e-9):
    G          = B(T_J, uniform) - B(I, uniform)
    P          = B(I, pi_J)      - B(I, uniform)
    Interaction= [B(T_J,pi_J)-B(T_J,uniform)] - [B(I,pi_J)-B(I,uniform)]
    full_joint_delta = B(T_J,pi_J)-B(I,uniform) == G + P + Interaction

## B. W1 correction (unseen-subject MI LOSO, same-backbone)
Datasets BNCI2014_001 L/R, Cho2017, Lee2019_MI. Protocol = EXISTING disjoint earlier-target-block
adaptation / later-target-block evaluation. Source seeds {0,1,2}: reuse seed-0 frozen bundles ONLY
after validating exact source weights + data hash + preprocessing signature + source-training config;
train seeds 1,2 with identical config except seed; NO target-label model selection. Unit = target
subject. Analysis: (i) average technical source seeds WITHIN target subject; (ii) per-dataset paired
estimates + percentile cluster bootstrap; (iii) subject-weighted aggregate; (iv) dataset-equal macro
average; (v) joint_geometry_uniform - pooled_uniform paired CI; (vi) leave-one-dataset-out descriptive;
(vii) negative-change rates at delta<0, <-0.01, <-0.02.
W1 PRIMARY (confirmatory): fixed_iterative_geometry_uniform - joint_geometry_uniform (prior-M-step
geometry effect, uniform decision). All else secondary/exploratory.

## C. W2 correction + expansion (Sleep-EDF Sleep-Cassette, frozen preprocessing)
Audit ALL subjects; freeze the COMPLETE list with two valid nights after existing QC (do NOT hard-code
78/20; save actual IDs + N). Primary protocol: night1 = unlabeled adapt, night2 = eval. Secondary
protocol: night2 first contiguous half = adapt, second half = eval. Source seeds {0,1,2}; reuse
matching seed-0 bundles where valid; no hyperparameter changes. Record per unit: rho_source,
rho_adaptation_night, rho_evaluation_night, pi_J, JS(source,adapt), JS(adapt,eval), JS(source,eval),
JS(pi_J,eval), per-stage recall W/N1/N2/N3/REM, per-subject + aggregate normalized confusion matrices,
all section-A prediction branches. W2 PRIMARY (confirmatory): joint_geometry_uniform - identity_uniform
in balanced accuracy. Decision-prior DIAGNOSTIC: joint_geometry_joint_prior - joint_geometry_uniform.
Regressions vs prevalence divergence = EXPLORATORY, not causal.

## D. V2P_WEIGHTED_PREVALENCE_INTERVENTION (new; old V2P artifacts untouched)
Same supported cross-session units as corrected V2P; source seeds {0,1,2}. Per target unit: ONE fixed
adaptation reservoir + ONE fixed evaluation set; ALL ratios use EXACTLY the same trial IDs in the same
temporal positions (reweight, do NOT resubset). For class-0 mass q in {0.50,0.75,0.25}:
    w_i = q/n0 if y_i==0 ; w_i = (1-q)/n1 if y_i==1  (normalized).
Labels used ONLY by the offline weight-builder + the oracle diagnostic; deployed estimators receive
embeddings + sample weights, never labels. Weighted versions of: pooled moments; fixed-reference
one-shot soft conditioning; fixed-prior iterative soft conditioning; joint EM. Weighted joint EM:
E-step unchanged given params; geometry Q uses weighted averages; prior M-step uses weighted
responsibility averages; regularization scale matches the original MEAN-objective convention. Add
oracle_label_conditional (true labels, diagnostic only).
MANDATORY TESTS: (1) equal weights reproduce the existing unweighted estimator; (2) rational weights
reproduce explicit sample replication; (3) all three ratios have identical unique trial IDs + eval IDs;
(4) weight sums + effective class masses exact; (5) target labels never enter any non-oracle
responsibility/optimizer; (6) identity transform + predictions ratio-invariant.
Primary geometry outputs: log-scale vector; translation vector; log-scale displacement from 1:1;
translation displacement from 1:1; mean eval-embedding displacement from 1:1; symmetric 3:1/1:3
displacement; vector slope between 3:1 and 1:3. Secondary: predicted occupancy (uniform decision);
balanced acc (uniform); ordinary acc/NLL under uniform AND estimated joint prior; pi_J.
V2P PRIMARY (confirmatory): fixed-reference displacement nonzero; fixed-reference vs pooled
displacement; oracle-label-conditional displacement = misspecification diagnostic. Unit = pair/session
target unit nested within subject; average source seeds within unit, then subject-cluster bootstrap
preserving all sessions/pairs of a sampled subject.

## E. Statistics
10,000 percentile bootstrap replicates, deterministic stable seeds. W1 = stratified resampling of
subjects within dataset. W2 = subject bootstrap. V2P = subject-cluster bootstrap preserving repeated
pair/session units. Source seeds are NOT independent observations (averaged within unit first). ONE
predeclared PRIMARY contrast per panel (above); all else secondary/exploratory; Holm correction only
within explicitly declared confirmatory families. RENAME "harm rate" -> "negative-change rate"
everywhere; thresholds 0, -0.01, -0.02.

## F. Naming
"SPDIM" same-backbone comparator -> "Latent-IM-Diag" (NOT an official SPD-manifold SPDIM reproduction).
"EA" in reports -> "source-recolored Euclidean alignment". Do NOT run official SPDIM; do NOT rerun
BTTA-DG.

## G. Provenance + outputs
NEW runners/analyzers (do not alter old frozen-artifact interpretation); shared implementation changes
stay backward compatible. Outputs: REVIEW_P0_FROZEN.md, REVIEW_P0_RESULTS.md, review_p0.report.json,
review_p0.sha256, exact expected-key manifests, source-bundle hashes, env lock, SLURM job IDs, exact
commands, row counts, excluded-unit list with preregistered reason, prediction hashes, bootstrap config.
Large raw JSONL stays outside git (filename + SHA-256 recorded). Run the FULL test suite before
submission and after merge. icml env; ALL GPU work via SLURM (no login-node GPU training); parallelize
via job arrays + exact-key merging. If a frozen bundle fails hash/config validation -> STOP that unit
and report; do NOT silently relax provenance. After jobs: merge only exact expected keys; run all
analyzers; generate the frozen report; commit code/prereg/results in SEPARATE local commits; do NOT
push; print commit SHAs, job IDs, artifact hashes, actual W2 subject count, final results table.

## PROHIBITED
target-only gates; new operator families; target-label tuning; dataset-specific hyperparameters; new
prevalence ratios; new datasets; official SPDIM work; BTTA-DG reruns; changes to old terminal tags;
outcome-driven protocol changes; pushing to remote.
