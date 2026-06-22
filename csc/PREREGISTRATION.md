# PRE-REGISTRATION — Partial-Identification Concept-Shift Certificates (csc)

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
> (a) **never** falsely certifies an unidentifiable shift (pure conditional, pure label, or
> out-of-atlas) as compatible/suspect — it abstains, as THEORY §1 proves it must; and (b) has
> **stable, direction-linked power** to flag a real, marginally-visible concept change.

This is a statement about the certificate's **error profile**, not accuracy.
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
Rule-of-Three "≥ N zero-failure trials" refers to N **independent clusters**.

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
