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

## 3. Synthetic validation obtained (CSC-P0, this commit)

`conda run -n icml python -m csc.run_synthetic --seeds 10 --n_boot 50` (CPU):

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
`tests/test_design_and_pairs` (reference coding full rank; identical-Z → identical
certificate), `tests/test_null_calibration` (clean/invisible/label-shift abstain; covariate-
only source shows no concept evidence), `tests/test_power` (covariate → COMPATIBLE; visible
concept → SUSPECT; concept evidence detected).

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

## 6. Calibration & freeze protocol (CSC-P1.1, `nested_lodo`)

For each held-out **domain group** `g` (mechanism-group-out, not just one domain): build
atlas+evidence on the others (no leakage); calibrate the certifier's thresholds on the
TRAINING domains using the **exact certificate statistic** `visibility_statistic =
max(n_cov,n_concept,n_resid)` over block-resampled pseudo-targets; `dataclasses.replace`
them into the config and call **`certify_robust`** on `g`. Then compute an **oracle
boundary-effect** on `g` with `g`'s labels — bootstrapped WITH REFITTING (resample `g`,
refit the group boundary, evaluate OOB), label-shift corrected — as a **two-sided
equivalence test**:
```
oracle_lb > eps_concept                  -> VISIBLE_CONCEPT
-eps_stable < oracle_lb and oracle_ub < eps_stable  -> COVARIATE_STABLE (stability PROVEN)
otherwise                                -> AMBIGUOUS (excluded from the forced binary)
```
with pre-registered `0 < eps_stable < eps_concept`. Report **separately**: concept power on
*fair* visible folds (those whose training pool still had a concept atlas), false-concept
rate on stable folds, compatible coverage on stable folds, abstention. **No aggregate
agreement when the bank lacks either class** (`valid_bank`).

**Power is NOT validated on in-distribution LODO folds.** A held-out source domain's marginal
shift is within normal source spread, and leave-ALL-concept-out leaves no training atlas — so
LODO validates *false-concept control* + *threshold calibration*; deployment **power** is
validated on out-of-distribution synthetic targets (`run_synthetic`) and on the oracle-
confirmed real PD ON/OFF control (§2). PD ON/OFF qualifies as a positive control only if its
oracle verdict is `VISIBLE_CONCEPT`.

### Freeze list (ALL of these are frozen before the real run; none tuned on the target)
| parameter | EXACT status |
|---|---|
| `tau_detect`, `tau_label` | **nested-LODO-calibrated** — `1−α` quantile of the *exact certifier statistic* over training-only, target-size-matched (and block/subject-matched on real data) pseudo-targets |
| `tau_margin` | **fixed pre-registered constant** (structural dominance ratio between concept and covariate components); swept on source LODO, frozen before the real run |
| `tau_resid` | **fixed pre-registered constant** (out-of-atlas dominance margin); swept on source LODO, frozen before the real run |
| `eps_concept`, `eps_stable` (oracle, `0<eps_stable<eps_concept`) | **pre-registered equivalence bands** (CE/nats) |
| `cov_stable_margin` (κ) | **pre-registered negligibility multiplier** (default 1.5), swept in freeze; the equivalence margin is `κ × noise_scale` where `noise_scale` = the data-driven `1−α` quantile of the cov-subspace loading under the h0 NULL |
| `consensus` (`certify_robust`) | swept on source LODO, frozen before the real run |
| `MIN_PRINCIPAL_ANGLE_DEG` (signature separability) | fixed pre-registered constant |
| `alpha` | fixed pre-registered constant |

`cov_stable` semantics: the covariate-subspace boundary loading is a **non-negative** norm, so
stability is the explicit TOST-style one-sided test `U_cov < eps_stable` where `U_cov` is the
loading's data-bootstrap upper CI and `eps_stable = κ × noise_scale` (κ = `cov_stable_margin`,
pre-registered; `noise_scale` = the dimensionally-matched h0-null `1−α` quantile of the SAME
loading). It is an AFFIRMATIVE equivalence test ("cov-boundary ≤ κ× noise = negligible"), not a
`< 0` test and not a "failed to reject". Empirically the h0-null quantile ≈ the observed point
(a well-calibrated noise floor), and κ>1 provides the negligibility headroom.

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
