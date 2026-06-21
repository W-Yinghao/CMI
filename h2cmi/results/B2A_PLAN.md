# A* terminal decision + B2a scoping

## A* TERMINAL DECISION (immutable — do not re-tune)
- **A_STAR_FAIL** (preregistered): neither N1 nor N2 satisfies the complete acceptance set.
- **N1 frozen as the abstention COMPONENT** — nested source-null calibration gives a safe,
  nontrivial adapt-vs-abstain gate (dev seeds 0–2, diagonal actions): std false-adapt 0.03,
  coverage 0.36, non-null mean ΔbAcc +0.044, hard-null all safe. Not a universal "WHEN solved" —
  not yet faced fresh seeds / extended actions / real EEG.
- **N2 retired** — stability veto adds no conditional info worth its coverage cost (cov 0.05).
- **Target-only action ranking TERMINATED** — top-1 0.40 AND regret 0.038 (real cost, not ties).
  Scoped claim only: unlabeled target statistics fail to identify the best operator within the
  tested diagonal family. NOT a universal impossibility claim.
- **No further target-only heuristic permitted.**

### Descriptive decomposition (read-only, `b2a_decomposition.json`; verdict unchanged)
- adaptation can help on **81%** of shift units (some action > identity); on nulls the in-sample
  best beats identity on **37%** (pure overfit — exactly what the gate must, and does, suppress).
- oracle-best operator among adapt cases: **pooled 62%**, gen_iterative 23%, gen_oneshot 15%.
- always-one-operator mean ΔbAcc on shifts: **pooled +0.053** > gen_oneshot +0.045 ≈ gen_iter
  +0.044; oracle ceiling +0.089. KEY: always-pooled (+0.053) already BEATS N1's per-unit target
  selection (+0.044) — a good DEFAULT operator + the gate beats the failed target-only selector.

## B2a: metadata-conditioned diagonal operator selection (next phase)
Diagonal-only first; SPD/rotation only AFTER the selection architecture is proven, and EVERY
action-set expansion recomputes the empirical max-null calibration. Dev seeds 0–2 only.

### Deploy rule (strictly decoupled)
    a_m       = g(Δmetadata)                       # metadata picks WHICH operator
    a_deploy  = a_m   if  Z_{a_m} > τ  (frozen N1 gate)   else  identity
- N1 only decides IF the metadata-chosen operator has enough source-null-calibrated evidence;
  N1 score is NOT used for operator ranking.
- metadata missing / conflicting / unseen-mechanism → identity.

### Mechanism-level action set (freeze the estimator BEFORE looking at B target eval)
    identity
    pooled_empirical_diag
    canonical_fixed_prior_class_conditional_diag
- gen_oneshot vs gen_iterative are ESTIMATOR variants, not acquisition mechanisms — do NOT let
  metadata choose between them. Pick ONE canonical class-conditional implementation by SOURCE-ONLY
  nested pseudo-target episodes (source-only dominant; tie → parsimony), decided before B target eval.

### Metadata inputs = deployable mechanism descriptions of the source→target relation
device/acquisition family · reference scheme · montage · channel set & layout relation ·
sampling-rate/preprocessing differences · site/center/session hierarchy relation · cohort/
ascertainment (where genuinely available).
- FORBIDDEN: hidden synthetic scenario names (e.g. cov_prior); raw site-ID lookup tables (memorise
  5 sites). Site only as hierarchy/grouping relation or source-side random effect.
- Map = low-capacity auditable rule table first. If learned: only from source pseudo-target
  episodes with outer-target-site exclusion; NO target labels / target oracle action / A* outcome.

### Frozen evaluation (dev seeds 0–2; aggregate sites within seed, then cluster by seed)
Comparators: identity · always-pooled · always-canonical-class-conditional · failed N1 target-stat
ranking · metadata-map + frozen-N1-gate · metadata-ORACLE upper bound.
Acceptance (same as A*): null false-adapt ≤0.10 · hard-null harm ≤0.10 · coverage ≥0.25 ·
non-null mean ΔbAcc >0 · non-null harm ≤0.20 · top-1 oracle ≥0.50 · action regret ≤0.02.
Also report: operator regret on gate-positive cases · metadata-selected action failing the N1 veto
rate · abstention from missing/unknown metadata · pooled↔class-conditional confusion matrix.
