# csc — Partial-Identification Concept-Shift Certificates with Abstention

**When can unlabeled EEG reveal concept shift? Partial-identification certificates with
abstention.**

A self-contained research package (it does **not** import or mutate the AAAI `cmi/`,
`h2cmi/`, or `oaci/` packages — they coexist). It runs entirely on a controllable shift
simulator, so every component is exercised by a self-test.

This is the **identifiability / counterexample phase** the Evidence Ledger named as the way
forward after the gate line closed
([`notes/EVIDENCE_LEDGER.md`](../notes/EVIDENCE_LEDGER.md)). A0 proved density / CMI scores
are *anti-aligned* with adaptation harm
([`notes/A0_FALSIFICATION_FROZEN.md`](../notes/A0_FALSIFICATION_FROZEN.md)); csc turns that
negative result into a positive, falsifiable object — a certificate that **abstains where
detection is provably impossible**.

## The one idea

For **any** observed target marginal `Q_Z` there are two joint laws with that same marginal
but different `P(Y|Z)` (one equal to the source posterior, one not). So no `Z`-only
certificate can identify concept shift — *not just* in the pure-conditional case
([`THEORY.md`](THEORY.md) §1, Proposition 1). The honest object is therefore a **three-state
partial-identification certificate**:

| state | meaning |
|---|---|
| `COVARIATE_COMPATIBLE` | visible, label-free, in-atlas shift in the covariate subspace where the source boundary is stable. A **compatibility** claim — NOT a guarantee any adaptation lowers risk (THEORY §5). |
| `CONCEPT_SUSPECT` | visible shift aligned with a **direction-linked** evidenced concept direction (a source boundary actually moved there). |
| `UNIDENTIFIABLE` | **abstain** — invisible, label-confounded, out-of-atlas, ambiguous, or no valid concept atlas. |

It never issues a positive verdict for a shift it cannot see. Over-abstaining (low power) is
allowed; **false certification is not.**

## What changed from the v0 scaffold (review fixes)

* **Stronger impossibility result** (partial identification for *any* marginal) + an explicit
  **transportability/separability assumption** named for what `CONCEPT_SUSPECT` requires.
* **Label shift handled.** The v0 fired `CONCEPT_SUSPECT` 31–53 % of the time under a skewed
  target prior; csc now carries a label subspace `U_π` and **abstains** (regression test).
* **`COVARIATE_ADAPTABLE` → `COVARIATE_COMPATIBLE`** (no adaptation operator / risk bound is
  claimed).
* **Valid null**: parametric bootstrap under fitted `h0` (conditions on `(Z,D)`), replacing
  the non-exchangeable within-`Y` permutation.
* **Reference-coded designs** (no rank deficiency) + **direction-linked** concept evidence.
* **Support gate** checks bipartite connectivity + cell counts + design conditioning (not
  just degree).
* **Clean must abstain** (scoring fixed); **nested oracle-labeled LODO** calibration with
  equivalence tests; **confidence-region** decision (`certify_robust`).

### CSC-P1.1 (second review — calibration correctness)
* `_orthonormal_complement` uses **SVD rank with an input-scaled tolerance** (the QR-norm
  version kept spurious columns, silently invalidating the leakage fix) + a
  **signature-overlap** abstention on small principal angles.
* the rank gate is an **exact `rank([1,X])==ncols`** check (the condition-number gate dropped
  exact-zero singular values and passed e.g. a duplicated feature).
* the calibrated `tau_detect`/`tau_label` now **actually enter** `certify_robust` per fold
  (`dataclasses.replace`), calibrated on the **exact certifier statistic** from training-only
  block resamples (no leakage).
* concept evidence is a **full-bootstrap max-statistic / step-down** (re-estimates subspaces
  each replicate — fixes post-selection bias).
* `COVARIATE_COMPATIBLE` requires positive **`cov_stable` equivalence** evidence, not
  absence-of-concept.
* LODO does **mechanism-group-out** with a diagnostic scorecard (no agreement when the oracle
  bank lacks a class), two-sided equivalence bands + refit-OOB oracle CI.
* the false-cert bound is **cluster-level** (per source seed), and all thresholds are in the
  freeze list with a **lexicographic** selection rule.

## Layout

```
csc/
  THEORY.md              partial-identification framework, taxonomy, T-statistic, certifier, calibration
  PREREGISTRATION.md     data design + Rule-of-Three honesty + kill criteria
  sim/shift_simulator.py clean / covariate / boundary_coupled / pure_conditional / label_shift / label_covariate_mixed
  certificate/
    residual_test.py     reference-coded T, parametric-bootstrap null, identifiability support gate
    atlas.py             3 orthogonal subspaces (cov / concept / label U_pi) + analyze_source (direction-linked evidence)
    certifier.py         certify (3-state) + certify_robust (confidence region) + ACCEPTABLE/FORBIDDEN
  calibration/lodo.py    nested oracle-labeled LODO + equivalence tests + tau_detect calibration
  run_synthetic.py       multi-seed eval with Rule-of-Three upper bounds
  tests/                 validity_gate / null_calibration / power / design_and_pairs
```

## Quickstart

```bash
# runnable components:
conda run -n icml python -m csc.sim.shift_simulator
conda run -n icml python -m csc.certificate.residual_test
conda run -n icml python -m csc.certificate.atlas
conda run -n icml python -m csc.certificate.certifier
conda run -n icml python -m csc.calibration.lodo

# self-tests:
conda run -n icml python -m csc.tests.test_validity_gate
conda run -n icml python -m csc.tests.test_null_calibration
conda run -n icml python -m csc.tests.test_power
conda run -n icml python -m csc.tests.test_design_and_pairs

# multi-seed evaluation (prints Rule-of-Three bounds, not "rate<=0.05"):
conda run -n icml python -m csc.run_synthetic --seeds 30 --n_boot 80
```

## Synthetic result so far (10 seeds; PREREGISTRATION §3)

```
                                COVARI  CONCEP  UNIDEN  forbid
clean            (NONE)              0       0      10       0
covariate        (COVARIATE)         5       0       5       0
boundary_coupled (CONCEPT_VISIBLE)   0      10       0       0
pure_conditional (CONCEPT_INVISIBLE) 0       0      10       0
label_shift      (LABEL_SHIFT)       0       0      10       0
label_covariate_mixed (LABEL_COV)    0       0      10       0

false certifications, total : 0 / 60
per-SEED clusters w/ a miss : 0 / 10   (95% cluster-level upper bound = 0.259)
power on VISIBLE concept    : 1.00
covariate -> COMPATIBLE     : 0.50     (conservative: now needs cov_stable evidence)
```

**Honesty:** 0 observed false certifications does **not** prove the rate ≤ 0.05. The unit is
an *independent source seed* (the 4 must-abstain targets share a source), so 0/10 → 95% upper
bound ≈ 0.259 (Rule of Three) — **not** the 0.072 a wrong 40-independent-trials count gives;
reaching 0.05 needs ≥ 59 independent clusters. The headline is *"simulator smoke passed;
control & power not yet statistically established."* Covariate→COMPATIBLE at 0.50 is the soft
spot — conservative abstention now that it requires positive `cov_stable` equivalence evidence
(a safe miss, not a false cert); the freeze sweep is the route to tightening it.

## Scaling to real EEG

`analyze_source(Z, Y, D)`, `certify(analysis, Z_target)`, `certify_robust(...)`, and
`nested_lodo(Z, Y, D)` take ordinary `(Z[n,d], y, D)` arrays. Point them at the AAAI loaders
(`cmi/data/*.py`) or the audited deployment-encoder dumps (A0 `erm:0 = CITA-no-LPC`) to run
PREREGISTRATION §2 — **PD medication ON/OFF** (a positive-control *candidate*, oracle-gated)
and **SCZ/PD cross-site** (the real null). Those, not the simulator, are the actual test.

## Status / honesty

A **research implementation on a simulator**. It demonstrates the framework is correct,
calibrated, label-deconfounded, and composes, and that the impossibility-driven abstention
works. It is **not** a real-EEG result. Equivalence bands (`eps_concept, eps_stable`) need
domain values; the atlas is mean/low-moment (shape-only concept shifts return
`UNIDENTIFIABLE` — safe, low power). The direction is **killed** if it cannot control
false-certification on real invisible/label/null shift, or has no power on an oracle-confirmed
PD ON/OFF positive control — see PREREGISTRATION §7.

Naming discipline (binding, Evidence Ledger): the source statistic is a residual-decoder
concept-evidence test / "extractable conditional domain information" — **not** "precise CMI";
the certificate is a diagnostic with abstention, **not** a "safety gate".
