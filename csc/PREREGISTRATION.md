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
source concept evidence valid: 9/10
                                COVARI  CONCEP  UNIDEN  forbid
clean            (NONE)              0       0      10       0
covariate        (COVARIATE)         7       0       3       0
boundary_coupled (CONCEPT_VISIBLE)   0       9       1       0
pure_conditional (CONCEPT_INVISIBLE) 0       0      10       0
label_shift      (LABEL_SHIFT)       0       0      10       0
label_covariate_mixed (LABEL_COV)    0       0      10       0

false certifications, total                 : 0  (across 60 certificates)
false 'safe/suspect' on must-abstain shifts : 0/40  -> 95% upper bound on rate = 0.072
power on VISIBLE concept (boundary_coupled) : 0.900
covariate -> COVARIATE_COMPATIBLE           : 0.700
```

**Honesty on the headline.** Zero observed false certifications does NOT establish a rate
≤ α. With 0 failures in `N` trials the one-sided 95% upper bound is `1−0.05^(1/N) ≈ 3/N`
(Rule of Three): 0/40 ⇒ ≤ 0.072 here; controlling at 0.05 needs ≥ 59 zero-failure trials.
So the correct statement is: **"simulator smoke passed; false-certification control and
stable power are not yet statistically established."** The full frozen run scales `N` until
the bound reaches the pre-registered level.

The previous (v0) label-shift counterexample (31–53% false `CONCEPT_SUSPECT` at skewed
target priors) is now **0/12** and a regression test (`tests/test_null_calibration.py`).

Covariate power 0.70 is the current soft spot — conservative abstention on a benign
covariate shift (allowed, not forbidden); CSC-P1 calibration (§6) is the route to tightening
it without trading away the abstention guarantee.

---

## 4. Self-tests (all passing)

`tests/test_validity_gate` (single-class + disconnected + ill-posed → INVALID → abstain),
`tests/test_design_and_pairs` (reference coding full rank; identical-Z → identical
certificate), `tests/test_null_calibration` (clean/invisible/label-shift abstain; covariate-
only source shows no concept evidence), `tests/test_power` (covariate → COMPATIBLE; visible
concept → SUSPECT; concept evidence detected).

---

## 5. Open before real-data freeze

1. Lock the CSC-P1 calibration (`csc.calibration.lodo`): `tau_detect` from block-resampled
   pseudo-targets; equivalence bands `eps_concept, eps_stable`; the `certify_robust`
   consensus level — all chosen on source LODO, never on the target.
2. **Atlas beyond the mean** — add covariance / higher-moment signatures, or keep returning
   `UNIDENTIFIABLE` for shape-only concept shifts (safe but low power).
3. Domain/subject-**clustered** inference and a permuted-`D` real null.
4. Scale synthetic `N` until the Rule-of-Three bound reaches α for the must-abstain families.

---

## 6. Calibration protocol (CSC-P1, `nested_lodo`)

For each held-out source domain `d`: build atlas+evidence on the others (no leakage),
certify `d` label-blind, then compute an **oracle boundary-effect** on `d` with `d`'s labels
(label-shift-corrected, bootstrap CI) as an **equivalence test**:
`oracle_lb > eps_concept → VISIBLE_CONCEPT`; `oracle_ub < eps_stable → COVARIATE_STABLE`;
else `AMBIGUOUS` (excluded from the forced binary). Score the label-blind certificate against
the oracle verdict on non-ambiguous domains. *Not rejecting a boundary shift ≠ proving
stability* → two-sided bands. PD ON/OFF runs through this same oracle gate to qualify as a
positive control.

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
