# PRE-REGISTRATION — Partial-Identification Concept-Shift Certificates (csc)

> **CONFIRMATORY OUTCOME (frozen `csc-confirmatory-v1` / `dee8958`, SLURM 876329): scientific FAIL.**
> The pre-registered, frozen confirmatory run on UNSEEN synthetic clusters (K=1, `P_baseline`,
> base_seed 900000) **failed both endpoints** — false-certification control (forbidden 1/65, CP-UB
> 0.0709 > α=0.05) AND power (28/65, CP-LB 0.326 < bar 0.50). The development-observed identifiable
> core (dev power 0.83 / 0 forbidden) **did not survive confirmatory validation** — a development→
> confirmatory generalization gap that this freeze→unseen protocol was built to expose. Per the
> non-selection rules the result was committed, not rerun; thresholds/seeds/manifest/tag unchanged;
> `P_strong` not run. Direction (reviewer): the paper owns the **identifiability/abstention boundary
> + this audited negative result** (NOT a positive detector). See `notes/CSC_CONFIRMATORY_RESULT.md`
> and `notes/CSC_MANUSCRIPT_RESULT_MEMO.md`. NO method-revision round opened; NO P2.

Status: **DRAFT v1 (CSC-P0 + CSC-P1 implemented on the simulator; real-data run NOT frozen).**
Follows the project freeze-before-run discipline (`notes/FREEZE_PROTOCOL.md`,
`notes/A0_FALSIFICATION_FROZEN.md`). §3 numbers are *synthetic* validation that the method
and its self-tests behave as specified; they are **not** the result. The result is the
real-data run (§2), to be frozen (thresholds, splits, seeds, equivalence bands) before
execution — and only AFTER the CSC-P1 calibration is locked.

---

## 1. Hypothesis and output

A three-state certificate over an unlabeled target batch:
`COVARIATE_COMPATIBLE` / `CONCEPT_SUSPECT` / `UNIDENTIFIABLE`. Claim:

> Reading only unlabeled target `Z` (no target labels, no source examples), the certificate
> (a) controls FALSE certification of an unidentifiable shift (pure conditional, pure label, or
> out-of-atlas) at level `alpha` — by abstaining (THEORY §1 proves a Z-only certifier MUST be
> able to abstain); and (b) has **stable, direction-linked power** to flag a real, marginally-
> visible concept change.

What is STRUCTURAL vs STATISTICAL (CSC-P1.4.2/P1.4.3 #7): byte-identical targets always get the
SAME output (exact-pair indistinguishability) is structural; controlling clean/pure/label false
certification at `alpha` is a FINITE-SAMPLE STATISTICAL claim (a chance clean marginal can exceed
`tau_detect` + align with the atlas), measured by the exact-CP endpoint over independent clusters
— NOT an absolute "never". This is a statement about the certificate's **error profile**, not accuracy.
`COVARIATE_COMPATIBLE` is a *compatibility* claim, NOT an adaptation guarantee (THEORY §5).

---

## 2. Data design (real-data validation — to be frozen)

| condition | source | target | expected certificate | role |
|---|---|---|---|---|
| **PD medication ON/OFF** | PD ON/OFF paired within subject | held-out subjects | `CONCEPT_SUSPECT` (if oracle confirms a boundary move) | **positive-control candidate** |
| **SCZ / PD cross-site, same disease** | site A | site B | mostly `COVARIATE_COMPATIBLE` / `UNIDENTIFIABLE` | real null / weak signal |
| **synthetic covariate** | sim | nuisance `P(Z)` shift | `COVARIATE_COMPATIBLE` | covariate power |
| **synthetic boundary-coupled** | sim | concept + visible signature | `CONCEPT_SUSPECT` | concept power |
| **synthetic pure conditional** | sim | relabel-only, `Z` identical | `UNIDENTIFIABLE` | invisible-shift guard |
| **synthetic label shift** | sim | `P(Y)` skewed, `P(Z\|Y)` fixed | `UNIDENTIFIABLE` | **label-confound guard** |
| **synthetic label×covariate** | sim | label shift + covariate | `UNIDENTIFIABLE` | confounded-attribution guard |
| **domain-`D` permutation** | sim/real, `D` permuted within `Y` | — | residual evidence NON-significant | null calibration of `T` |
| **random-label-noise** | sim/real, labels noised | — | no spurious concept evidence | confound control |
| **single-class / disconnected domain** | degenerate support graph | — | `INVALID` → `UNIDENTIFIABLE` | invalid cases that MUST be rejected |

**PD medication is a positive-control CANDIDATE, not unconditional ground truth.** Whether
ON/OFF actually moves the task boundary must be confirmed by the §6 oracle analysis on
held-out labels; if the oracle says `COVARIATE_STABLE`, PD ON/OFF is not a valid concept
positive and cannot anchor the power claim.

Substrate reuse: real `Z` from the AAAI loaders (`cmi/data/*.py`) / the audited deployment
encoder dumps (A0 `erm:0 = CITA-no-LPC`). Real run adds: subject/site-**clustered**
inference, a permuted-`D` real null, and real-embedding semi-synthetic visible/invisible
shifts. `certify`, `certify_robust`, `analyze_source`, `nested_lodo` take ordinary
`(Z[n,d], y, D)` arrays.

---

## 3. Synthetic validation (CSC-P1.3 — FROZEN PATH)

The authoritative, machine-readable result is `csc/results/audit.json` (produced on SLURM via
`csc/run_audit.sbatch`): it records the `ProtocolConfig` manifest hash, both validity banks,
the cluster-level exact-Clopper-Pearson endpoints, and full provenance. ALL numbers there come
through `run_frozen_protocol` (calibrated `certify_robust`), not `certify(...)`. The block
below is an earlier CSC-P0 POINT-certificate smoke retained only for historical contrast — it
is NOT the frozen-path result.

```
source concept evidence valid: 10/10
                                COVARI  CONCEP  UNIDEN  forbid
clean            (NONE)              0       0      10       0
covariate        (COVARIATE)         5       0       5       0
boundary_coupled (CONCEPT_VISIBLE)   0      10       0       0
pure_conditional (CONCEPT_INVISIBLE) 0       0      10       0
label_shift      (LABEL_SHIFT)       0       0      10       0
label_covariate_mixed (LABEL_COV)    0       0      10       0

false certifications, total                 : 0  (across 60 certificates)
per-SEED clusters with any must-abstain miss : 0/10  -> 95% upper bound on rate = 0.259
power on VISIBLE concept (boundary_coupled) : 1.000
covariate -> COVARIATE_COMPATIBLE           : 0.500
```

**Honesty on the headline (corrected, CSC-P1.1).** These are **DEVELOPMENT** seeds (they
informed code/threshold choices) — *not* a confirmatory run. The inference unit is an
INDEPENDENT source seed, not a per-target certificate (the 4 must-abstain targets share a
source). With 0 cluster failures in `N=10` seeds the **exact one-sided 95% Clopper–Pearson**
upper bound is `1−0.05^(1/N) = 0.259` (the Rule-of-Three `3/N = 0.30` is only its
approximation) — NOT the 0.072 a wrong 40-independent-trials count would give. Controlling at
0.05 needs ≥ **59 independent source–target clusters**. So: **"simulator smoke passed;
false-certification control and stable power are NOT yet statistically established."**

The v0 label-shift counterexample (31–53% false `CONCEPT_SUSPECT` at skewed target priors)
is now **0/12** and a regression test.

Covariate→`COVARIATE_COMPATIBLE` coverage is **0.50** (down from 0.70): CSC-P1.1 now requires
the positive `cov_stable` equivalence evidence, so borderline-stable covariate shifts abstain
rather than certify. This is a *safe* (non-forbidden) miss; coverage is a freeze-sweep tuning
target (α / eps / bootstrap budget), not a gate to weaken.

---

## 4. Self-tests (all passing)

`tests/test_validity_gate` (single-class + disconnected + ill-posed → INVALID → abstain),
`tests/test_design_and_pairs` (reference coding full rank; complement(A,A)→0; dup-feature →
INVALID; identical-Z → identical certificate), `tests/test_null_calibration` (clean / invisible
/ label-shift abstain; covariate-only source shows no concept evidence), `tests/test_power`
(covariate → COMPATIBLE; visible concept → SUSPECT; concept evidence detected), `tests/
test_protocol` (frozen-path taxonomy with 0 forbidden; `calibrate_tau_detect` keyword
regression; cluster-aware path; deterministic manifest hash).

---

## 5. Open before real-data freeze

The CSC-P1.1 calibration machinery is implemented and self-consistent, but **not yet frozen**:

1. Run the frozen grid sweep over the §6 freeze list and pick by the lexicographic rule. The
   bands `eps_concept, eps_stable` must be tight enough (enough data / oracle bootstrap) that
   genuinely-stable folds land `COVARIATE_STABLE` rather than `AMBIGUOUS` (the current
   synthetic budget often leaves them ambiguous — the safe outcome, but uninformative).
2. **Atlas beyond the mean** — add covariance / higher-moment signatures, or keep returning
   `UNIDENTIFIABLE` for shape-only concept shifts (safe but low power).
3. Domain/subject-**clustered** inference and a permuted-`D` real null.
4. Scale `N` (independent source–target clusters) until the cluster-level Rule-of-Three bound
   reaches α for the must-abstain families.
5. Power must be validated on out-of-distribution targets / an oracle-confirmed real positive
   control — **not** on in-distribution LODO folds (§6).

---

## 6. Frozen path, two banks, manifest & selection (CSC-P1.3)

### 6.0 ONE certification path
Every certificate — dev sweep, unit tests, audit, confirmatory — goes through
`csc.protocol.run_frozen_protocol(Z_src,Y_src,D_src, Z_tgt, cfg, src_group_ids, tgt_group_ids)`:
```
analyze_source (cluster-aware)  ->  calibrate_thresholds (SOURCE ONLY, target-size + group matched)
                                ->  certify_robust (consensus confidence region, cluster-aware)
```
parameterised by ONE serializable `csc.protocol.ProtocolConfig` whose canonical-JSON **hash**
identifies the method. There is no other way to certify (the v0 measured `certify(...)`
directly, which is NOT the deployable path).

### 6.1 TWO separate validity banks (never one `valid_bank` for both)
* **CALIBRATION_NULL_BANK** (`csc.calibration.nested_lodo`, leave-one-domain-out): validates
  **false-concept control** on oracle-`COVARIATE_STABLE` folds and supplies threshold
  calibration. `calibration_null_bank_valid` ⇔ enough oracle-stable folds. It does **not**
  measure power (in-distribution folds have within-spread shifts; leave-all-concept-out leaves
  no training atlas).
* **OOD_POWER_BANK** (`csc.protocol.ood_power_bank`): validates **deployment power** on
  GENERATOR-TRUTH out-of-distribution targets through the *same frozen path*, with the FULL
  source (training retains its concept atlas). `ood_power_bank_valid` ⇔ enough fair visible
  targets. The real positive control is oracle-confirmed PD ON/OFF (§2).

### 6.2 Oracle bands are frozen INDEPENDENTLY of certificate performance
`oracle_eps_concept_ce`, `oracle_eps_stable_ce` (`0 < eps_stable < eps_concept`, CE/nats)
define the oracle TRUTH (two-sided equivalence with a refit-OOB bootstrap CI). They are fixed
BEFORE any certificate sweep. **Synthetic parameter selection uses GENERATOR TRUTH** (we built
the targets); the oracle is only an estimator sanity-check on the null bank. Widening
`eps_stable` to manufacture `COVARIATE_STABLE` folds is forbidden — if the bank has no stable
samples the result is `BANK_INVALID`, not a re-defined label. (Raising `oracle_boot` only
reduces Monte-Carlo error of the quantile, never the CI width from sample size / estimator
variance.)

### 6.3 Endpoints (cluster level; exact Clopper-Pearson)
* **PRIMARY** `any_forbidden_full_suite` — per source seed, ANY forbidden outcome over ALL
  shift kinds (covariate→`CONCEPT_SUSPECT` and visible-concept→`COVARIATE_COMPATIBLE` are
  forbidden too, not only must-abstain misses).
* secondary `any_false_positive_must_abstain`.

### 6.4 Freeze list — EXACT status of every parameter (`ProtocolConfig`)
| parameter | EXACT status |
|---|---|
| `tau_detect`, `tau_label` | **calibration RULE** (not a number): `training_only_pseudotarget_quantile` at `tau_quantile`, target-size + (real) subject-block matched; recomputed per source |
| `tau_margin`, `tau_resid` | **fixed pre-registered constants** (dominance margins); swept on source, frozen before the real run |
| `oracle_eps_concept_ce`, `oracle_eps_stable_ce` | **pre-registered equivalence bands** (CE/nats), frozen INDEPENDENTLY (§6.2) |
| `cov_loading_margin_kappa` (κ) | **pre-registered negligibility multiplier** (default 1.5); margin = `κ × cov_loading_null_scale` where the null scale = h0-null `1−α` quantile of the cov-subspace loading |
| `consensus`, `target_n_boot` | swept on source, frozen before the real run |
| `n_boot`, `n_dir_boot`, `oracle_boot`, `source_cv_folds`, `tau_n_pseudotargets`, `var_keep`, `C`, `MIN_PRINCIPAL_ANGLE_DEG`, `alpha`, `quantile_convention`, `group_aware` | recorded in the manifest hash |

`cov_stable` is a full-estimator equivalence test: `U_cov` is the **cluster-bootstrap** upper
CI of the cov-subspace loading (each replicate resamples whole clusters and RE-ESTIMATES cell
means, pooled means, `(A,R)` and the cov subspace), compared to `κ × cov_loading_null_scale`.
Source-side `cov_loading_*` (Z-units) are named distinctly from the oracle `eps_stable_ce`
(CE/nats) to avoid unit confusion.

### Selection rule (LEXICOGRAPHIC — never accuracy or aggregate agreement)
1. **first** satisfy the cluster-level false-certification upper bound ≤ `α`;
2. then **maximise the worst-case** oracle-confirmed visible-concept power;
3. then maximise covariate-compatible coverage on stable folds.

### Cluster-level inference (§3 honesty, enforced)
The unit is an **independent source–target cluster** (a seed / a real source pool), NOT a
per-target certificate: the must-abstain targets share one source and are correlated. The
exact one-sided Clopper-Pearson bound "≥ N zero-failure trials" refers to N **independent
clusters**.

### 6.5 CSC-P1.4 hardening (executable manifest, unified path, cluster-valid inference)
* **Executable manifest.** `ProtocolConfig.validate()` rejects unsupported field values and
  the executor **fails closed** (`group_aware=True` ⇒ both source and target group ids are
  mandatory; never silently IID). Every field drives behaviour (`quantile_convention`,
  `tau_target_size_matched`, `tau_group_resampling`, `analysis_unit`, rng/seed derivation).
  The manifest hash is the **FULL SHA-256** of the canonical manifest and includes the rng
  algorithm + `master_seed` + seed-derivation rule.
* **One executor.** `csc.protocol.execute_protocol` is the only certification path;
  `run_frozen_protocol`, `nested_lodo`, `ood_power_bank`, `synthetic_null_bank`, the sweep and
  the audit ALL call it (no LODO parameter drift). Tests that call `certify()`/`analyze_source`
  directly (`test_power`, `test_null_calibration`, `test_validity_gate`, `test_design_and_pairs`)
  are **component** tests; only `test_protocol` exercises the frozen path.
* **Three banks, distinct roles, each with a one-sided exact Clopper-Pearson bound at the
  INDEPENDENT-cluster level.** `SYNTHETIC_NULL_BANK` (generator-truth-stable targets) validates
  **false-concept control** directly and reports per-source-cluster false-concept failures + a
  **CP UPPER bound**; `CALIBRATION_NULL_BANK` (LODO + oracle) is an **estimator sanity** check
  only; `OOD_POWER_BANK` validates **power** on generator-truth OOD targets with an
  **unconditional** denominator (atlas failures = power misses; `atlas_availability` reported),
  a **CP LOWER bound** on concept power, and a **binding-failure decomposition**
  (`residual_T_not_sig` / `geometric_maxstat_not_sig` / `support_invalid` / `signature_overlap`
  / `not_dominant_or_robust_consensus_abstain`).
* **Cluster-valid inference, ONE analysis unit throughout.** The **source atlas itself** (cell
  means `a_d`/`r_{d,y}`, pooled mean, all bootstraps), source CV (StratifiedGroupKFold),
  support-gate cell counts, target mean (one **cluster vote** per subject), calibration
  pseudo-targets (whole-subject blocks), and the oracle OOB ALL use the **subject** vote — not
  merely "all receive `group_ids`". (Synthetic epochs are i.i.d. within subject, so this
  exercises the convention; genuine within-subject correlation matters only on real EEG.)
* **Residual decoder IS the concept gate** (THEORY §4): `concept_evidenced` = global
  max-statistic AND residual-`T` significant → honest, lower power (~0.5–0.7), not the inflated
  geometric 1.0. **Difficulty envelope** (concept effect size × cluster count × principal-angle
  separation × covariate-leakage × class imbalance × mechanism family) is a freeze-sweep
  requirement for a confirmatory power claim — framework present, full envelope deferred.
* **Commit separation by HOOK, not gitignore.** A `.gitignore` only blocks *untracked* files;
  it does NOT stop already-tracked audit files from being re-staged by `git add csc/`. The real
  guard is `csc/tools/check_commit_separation.sh` (install as `.git/hooks/pre-commit` or run in
  CI): it rejects any commit that mixes `csc/results/**` with other files.

### 6.6 CSC-P1.4.1 (cluster-inference, executable RNG, concept-detection redesign)
* **Subject-level decoder (estimand + inference).** Epoch fit with one-vote-per-subject weights
  `1/n_s` (mean 1); OOF loss aggregated within subject (`ℓ_s`) then `T = mean_s(ℓ_{s,h0}−ℓ_{s,h})`;
  group-CV by subject; cluster-consistent null `Y*_s ~ q_s` (per-subject geometric mean of `p̂0`)
  broadcast to epochs; invalid null replicates COUNTED not dropped. Support gate adds
  min-subjects-per-class + grouped-fold feasibility.
* **Executable RNG.** `master_seed`/`seed_derivation` REMOVED from the manifest hash (they did
  not drive computation — they faked the method id). The runtime root seed lives in an
  `ExecutionContext` and drives every stage via a named hash (`_stage_seed(root, stage)`).
  `validate()` now also enforces numeric ranges (fail-closed).
* **Concept-FIRST geometry** (THEORY §4): concept estimated from the class-conditional residual
  `R` first, `cov = a_d ⊥ concept`. Fixes the visible-concept (asymmetric `s_y`) leak that a
  cov-first order absorbed into `cov_dirs`, emptying `concept_dirs`.
* **Concept gate = geometric max-stat AND decoder, with type-I controlled by the geometric gate**
  (THEORY §3.3/§4): `concept_evidenced = (p_global ≤ α) AND decoder-T significant`. Empirically the
  geometric `p_global` has correct type-I on a covariate-only source (~0.8, never fires) while a
  decoder-ONLY gate over-fires ~50% (the subject random effect is confounded with the subject's
  single label → finite-sample class-conditional noise). A magnitude-only direction gate
  (`concept_top ≥ κ·cov_noise_scale`) was tried and REJECTED (uncalibrated — passes on pure noise);
  it is a diagnostic only. Honest subject-level power is LOW (cluster-consistent null is
  conservative) — an envelope quantity, not a flaw.
* **Relative (dominance) label gate** (THEORY §4): abstain for label only when `n_label` is NOT
  dominated by concept/cov by `tau_margin`; the old absolute `n_label ≥ tau_label` over-abstained
  genuine boundary shifts that carried a small finite-target label byproduct.
* **Banks: evaluable vs control_pass; audit FAIL-CLOSED.** `evaluable` = enough independent
  clusters to compute a bound; `control_pass` = the cluster-level CP bound meets `α` (DISTINCT).
  The audit records `provenance_ok` and exits nonzero (sbatch propagates `$AUDIT_RC`) on failed
  tests / dirty tree / `audited_code_commit ≠ HEAD`.
* **Working DEV regime (ONE difficulty-envelope point, NOT confirmatory):** `sep=1.2`,
  `subject_tau=0.2`, source `concept_scale=4`, 22 subjects/domain; target `concept_scale=14`,
  `cov_target_scale=10` along a fixed source nuisance axis, 30 subjects. boundary→`CONCEPT_SUSPECT`
  ≈3/4, covariate→`COVARIATE_COMPATIBLE` 4/4, clean/pure/label/label_cov→`UNIDENTIFIABLE`, **0
  false certifications**. These seeds are DEVELOPMENT.

### 6.7a CSC-P1.4.2 (null accounting, paired-unit semantics, operational separability)
* **Invalid null replicates are CONSERVATIVE, never dropped (#1).** Both the residual-decoder null
  and the geometric max-stat null charge a degenerate/exception replicate as the MOST adverse:
  `p = (1 + N_extreme_valid + N_invalid)/(1 + B)` (it can only RAISE `p`); the geometric null runs
  the SAME support-validity check; > 20% invalid ⇒ the null is not estimable ⇒ source `INVALID`.
  `validate()` enforces `B ≥ ⌈1/α⌉−1` so the bootstrap `p` can actually reach `α`.
* **Subject-CONDITION decoder estimand (#2).** `ℓ_s = (1/|U_s|) Σ_u (1/n_su) Σ_e ℓ_sue`,
  `u = (subject, condition/domain)`; fit weights `1/(|U_s| n_su)`. Inner mean ⇒ duplicating epochs
  within a (subject,condition) cell leaves `T` invariant; outer mean ⇒ unequal ON/OFF epoch counts
  do not reweight conditions. CV and every bootstrap cluster is the BIOLOGICAL subject; a paired
  subject is drawn whole (both conditions) and never split.
* **cov-stability bootstrap preserves domain-class support (#4) + paired integrity (#2b).** Whole
  biological subjects are resampled WITHIN each support-signature stratum (the set of (domain,class)
  cells the subject occupies), so no occupied cell can silently disappear and paired conditions stay
  together; the support gate is re-run per replicate (conservative invalid policy).
* **SUBJECT-level oracle (#3).** `oracle_boundary_effect` takes `group_tr`+`group_g`, fits pooled and
  group-specific boundaries SUBJECT-weighted, aggregates the effect per biological subject then
  averages over subjects, and resamples whole subjects (paired-safe) — the SAME estimand as the
  decoder.
* **Operational separability (#5).** The within-split principal angle was an algorithmic artifact
  (cov ⊥ concept by construction). Replaced by a CROSS-SPLIT diagnostic: from one subject-half take
  the RAW nuisance offsets `A_cov`, from the other the concept direction, and measure
  `arcsin(||A_cov − A_cov P_concept|| / ||A_cov||)`. This is forced-orthogonality-free and REACTS as
  the true cov/concept angle shrinks (→ the principal-angle difficulty axis genuinely moves it).
* **Executable manifest (#6).** ONLY `analysis_unit=subject ∧ group_aware ∧ tau_group_resampling`
  is supported; all other combinations FAIL CLOSED. Independent NAMED stage seeds for
  residual-CV / residual-null / geometry-null / cov-bootstrap / calibration / oracle / target-
  bootstrap / separability-split (no `seed+1/+7/+11/+13`); `quantile_convention` is passed to
  `np.quantile(method=...)`.
* **No "structural zero false-positive" (#7).** What is STRUCTURAL is exact-pair indistinguishability
  (byte-identical clean vs pure → SAME output). Controlling clean/pure/label false-certification at
  `α` is a FINITE-SAMPLE statistical claim (a chance clean marginal can exceed `tau_detect` + align
  with the atlas), measured by the exact-CP endpoint. `0/6` covariate-only false concept-evidence has
  a one-sided 95% CP upper bound ≈ 0.393 — NOT proof of type-I correctness.

### 6.7c CSC-P1.4.3 / P1.4.4 (subject-condition estimand, full-T invariance, attribution stability)
* **Full residual-test invariance (P1.4.3 #1 / P1.4.4 #5).** Duplicating epochs within a
  (subject,condition) cell leaves `T`, `p` and `status` invariant for `label_unit ∈ {subject,
  subject_condition}`: per-fold subject-condition-WEIGHTED standardisation + subject-level fold
  assignment (stratified by CLASS PROFILE, shuffled by the named CV seed) make the scaler and the
  folds row-multiplicity-independent. The audit records `T/p/status/fold-hash` before & after.
* **FROZEN, data-VALIDATED `label_unit` (P1.4.4 #1).** `subject` ⇒ `Y` constant per subject;
  `subject_condition` ⇒ constant per `(subject,domain)` cell; `trial` ⇒ one row per trial. A wrong
  declaration FAILS CLOSED. Mixed-label subjects are CV-stratified by class profile so every training
  fold keeps all-class support.
* **Concept-ATTRIBUTION stability (P1.4.4 #2).** The RETAINED leading concept direction (top right
  singular vector of the class residual) must (a) be well-defined (relative eigengap ≥
  `concept_eigengap_min`) and (b) REPRODUCE across independent subject-halves within
  `concept_stability_max_deg` (sign-invariant angle, upper quantile over splits). An UNASSESSABLE or
  UNSTABLE attribution of a real concept SIGNAL (geometric + decoder evidence) abstains BOTH
  definite states (`UNASSESSED_CONCEPT_ATTRIBUTION` / `UNSTABLE_CONCEPT_ATTRIBUTION`) — the cov_dirs
  depend on the concept projection. A covariate-only source (no signal) is unaffected and still
  certifies covariate. (`min_principal_angle_deg` was REMOVED from the manifest: it was hashed but
  drove no decision.)
* **Conservative cov-bootstrap (P1.4.4 #3).** An invalid cov replicate is charged `+∞` and KEPT in
  the (1−α) quantile (fixed `B`) — it can only RAISE `cov_ub`, never lower it (the v0 dropped it,
  shrinking `cov_ub`). `cov_ub` non-finite ⇒ `cov_stable=False`.
* **Single-condition target + profile calibration (P1.4.3 #5 / P1.4.4 #4).** One certificate covers
  ONE condition (a multi-condition target FAILS CLOSED; paired ON/OFF = two batches). The calibrator
  matches the target CLUSTER-SIZE PROFILE (epochs per subject). `visibility_statistic` uses the SAME
  subject-vote `cluster_mean` aggregator as the certifier (not a raw row mean).
* **Honesty.** The geometric gate is a parametric-bootstrap p-value calibrated under the FITTED `h0`,
  NOT a proof of finite-sample type-I control (the null bank is 0/4, CP UB 0.527). New manifest fields
  in the hash: `label_unit`, `concept_stability_max_deg`, `concept_eigengap_min`,
  `invalid_null_frac_max`. Inference procedure changed each round ⇒ numbers are not poolable.

### 6.7d CSC-P1.4.5 (epoch-duplication invariance of the WHOLE protocol, condition-first end-to-end)
* **Regularisation scale is duplication-invariant (P1.4.5 #1).** sklearn lbfgs applies L2 =
  `1/(C · Σ_i w_i)`, so the EFFECTIVE penalty depends on the TOTAL sample-weight. The subject-condition
  weights `1/(|U_s| n_su)` are RAW (NOT renormalised to mean 1), hence `Σ_i w_i = #training subjects`
  regardless of epochs/subject. Duplicating a subject-condition's epochs leaves the penalty — and
  therefore `T` — invariant to machine epsilon (verified `|ΔT| ≤ 1e-15`, not optimiser noise).
* **FULL-PROTOCOL invariance (P1.4.5 #2).** Duplicating a whole source subject 1×/5× leaves EVERY
  downstream output invariant: `T`, the atlas subspaces (cov/concept/label/pooled hash), `source_status`,
  `tau_detect`, `tau_label` AND the final certificate. The calibrator draws each pseudo-subject mean
  from a Gaussian sampling model `N(μ_s, σ²_s/n_i)` with `μ_s` the subject's condition-first mean and
  `σ²_s` its per-dim epoch variance (both duplication-invariant), so `tau` is EXACTLY invariant — an
  empirical row-resample would depend on the source row pool (which grows under duplication). The audit
  records the full before/after comparison.
* **Condition-first estimand end-to-end (P1.4.5 #2).** ONE shared subject-vote primitive
  (`= cluster_mean` over the masked cell) computes the atlas class-POOL means, the per-(domain,class)
  cell means, the `h0` null `q_s` (`exp[(1/|U_s|) Σ_u (1/n_su) Σ_e log p̂_0]`), and the oracle (fits +
  WEIGHTED class priors `_wprior`). A paired subject's high-epoch condition cannot dominate any moment.
* **Concept SIGNAL vs ATTRIBUTABILITY are separate gates (P1.4.5 #3).** A geometric+decoder concept
  SIGNAL whose leading direction is not identifiable (near-tied eigengap, multi-axis / tied-spectrum
  shift) is `UNASSESSED_CONCEPT_ATTRIBUTION` — it BLOCKS both definite states (safe abstain), and is
  NOT reinterpreted as "no candidate ⇒ certify covariate".
* **Per-replicate source validity (P1.4.5 #4).** Each residual-NULL replicate runs the SAME validity
  pipeline as the observed source (label-unit + support graph + grouped-fold all-class coverage); an
  invalid replicate is charged as extreme and bounded by `invalid_null_frac_max`.
* **Cov-loading null keeps a value per replicate (P1.4.5 #5).** An empty cov subspace or invalid
  replicate contributes `0` (NEVER dropped) to the noise floor `q0`. Since a LARGER `q0` would make
  `cov_stable=(cov_ub<κ·q0)` EASIER, inserting `0`s can only TIGHTEN stability — dropping them (the bug)
  would loosen it.
* **Mandatory target condition ids (P1.4.5 #6).** `tgt_condition_ids` is a REQUIRED contract on BOTH
  public entry points (`execute_protocol`, `run_frozen_protocol`); a MISSING declaration FAILS CLOSED
  (not silently assumed single-condition), a multi-condition target is rejected.

### 6.7 Pre-registered DIFFICULTY ENVELOPE (required before any freeze/confirmatory)
Power is NOT a single number; it is a surface. Before a confirmatory claim we will sweep, on
UNSEEN seeds, a grid over: concept effect size (target `concept_scale`), independent cluster count
(`subjects_per_domain` × `n_domains`), within-subject correlation (`subject_tau`), class separation
(`sep`), covariate-leakage / principal-angle separation, class imbalance, and mechanism family. The
deliverable is the boundary of the region where boundary→`CONCEPT_SUSPECT` power is non-trivial WHILE
false-certification stays controlled — the partial-identification operating region, reported as a
surface with exact cluster-level CP bounds, not one tuned regime.

---

## 7. Termination / kill criteria (falsify-first)

The direction is **dead** — write it up as a negative result and stop — if, on the frozen
real-data run:

* **No false-certification control.** After CSC-P1 calibration, the false-certification rate
  on pure conditional + pure label + permuted-`D` null is not controlled at `α` (with `N`
  large enough that the Rule-of-Three bound reaches `α`) — i.e. abstention is not protective.
* **No power.** On the PD ON/OFF positive control *that the oracle confirms is a real
  boundary move*, the certificate cannot reach `CONCEPT_SUSPECT` stably — it can only abstain
  → the certificate is vacuous.

A PASS is **not** "higher accuracy"; it is: the false-certification guarantee holds on real
invisible/label/null shift, **and** there is direction-linked power on an oracle-confirmed
real positive control. Anything weaker is reported as the (still publishable) partial-
identification boundary itself.
