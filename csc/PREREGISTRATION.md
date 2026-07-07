# PRE-REGISTRATION — Falsifiable Concept-Shift Certificates (csc)

Status: **DRAFT v0** (synthetic harness validated; real-data run NOT yet frozen).
This follows the project's freeze-before-run discipline (cf. `notes/FREEZE_PROTOCOL.md`,
`notes/A0_FALSIFICATION_FROZEN.md`). Numbers under §3 are the *synthetic* validation that
the method and its self-tests exist and behave as specified; they are **not** the result.
The result is the real-data run in §2, which must be frozen (thresholds, splits, seeds)
before execution.

---

## 1. Hypothesis and output

The deployable object is a three-state certificate over an unlabeled target batch:
`COVARIATE_ADAPTABLE` / `CONCEPT_SUSPECT` / `UNIDENTIFIABLE`. Claim:

> Reading only unlabeled target `Z` (no target labels, no source examples at scoring), the
> certificate (a) **never** falsely certifies a pure conditional shift as safe/suspect
> (it abstains, as §1.1 of THEORY proves it must), and (b) has **stable power** to flag a
> real, marginally-visible concept change.

This is a statement about *the certificate's error profile*, not about accuracy.

---

## 2. Data design (real-data validation — to be frozen)

| condition | source | target | expected certificate | role |
|---|---|---|---|---|
| **PD medication ON/OFF** | PD ON/OFF paired within subject | held-out subjects' OFF (or ON) | `CONCEPT_SUSPECT` | **positive control** (real paired concept change) |
| **SCZ / PD cross-site, same disease** | site A | site B (same disease) | mostly `COVARIATE_ADAPTABLE` / `UNIDENTIFIABLE` | real null / weak signal |
| **synthetic covariate** | sim | nuisance-only `P(Z)` shift | `COVARIATE_ADAPTABLE` | power on covariate |
| **synthetic boundary-coupled** | sim | concept + visible signature | `CONCEPT_SUSPECT` | power on visible concept |
| **synthetic pure conditional** | sim | relabel-only, `Z` identical | `UNIDENTIFIABLE` | **false-certification guard** |
| **domain-label permutation** | sim/real, `D` permuted within `Y` | — | residual test NON-significant | null calibration of `T` |
| **random-label-noise** | sim/real, labels noised | — | no spurious concept evidence | confound control |
| **single-class subject-domain** | one domain = one class | — | `INVALID` → `UNIDENTIFIABLE` | invalid case that MUST be rejected |

Substrate reuse: real `Z` comes from the existing AAAI loaders (`cmi/data/*.py`) /
deployment encoder dumps (the A0 `erm:0 = CITA-no-LPC` embeddings), so this rides on the
already-audited representation rather than introducing a new one. The certifier API takes
ordinary `(Z[n,d], y, D)` arrays — see README "Scaling to real EEG".

---

## 3. Synthetic validation already obtained (this commit)

`conda run -n icml python -m csc.run_synthetic --seeds 12 --n_perm 60` (CPU):

```
source residual test significant (valid concept atlas): 12/12
                                COVARIATE  CONCEPT  UNIDENTIFIABLE
clean            (NONE)                 0        0        12
covariate        (COVARIATE)          11        1         0
boundary_coupled (CONCEPT_VISIBLE)     0       11         1
pure_conditional (CONCEPT_INVISIBLE)   0        0        12

(A) false-certification on INVISIBLE concept shift : 0.000   (strict/forbidden 0.000)
(B) power on VISIBLE concept positive control       : 0.917
    covariate correctly ADAPTABLE                   : 0.917
    false concept alarm on CLEAN                    : 0.000
SYNTHETIC VERDICT: PASS
```

Known v0 imperfection: covariate → `CONCEPT_SUSPECT` once in 12 (a benign "cry wolf"; in
the FORBIDDEN map for COVARIATE). Mitigation = the LODO threshold calibration in §4. This
is the only forbidden-direction event observed and it is on the *covariate*, never on the
invisible-shift guard.

---

## 4. Open before real-data freeze

1. **Threshold calibration.** Replace hand-set `τ_detect, τ_margin` with leave-one-
   source-domain-out calibration: choose thresholds so a held-out *source* domain is
   certified `UNIDENTIFIABLE`/`COVARIATE_ADAPTABLE` (never `CONCEPT_SUSPECT`) at the target
   false-alarm rate `α`. This makes the covariate false-suspect rate a *controlled* quantity.
2. **Atlas beyond the mean.** Add covariance/higher-moment signatures to `cov_dirs` /
   `concept_dirs` so a concept shift that moves shape (not mean) is still seen — or
   correctly returns `UNIDENTIFIABLE`.
3. **Domain-clustered inference** on the real run (errors clustered by cohort, per A0 §5),
   and a permuted-`D` real null.

---

## 5. Termination / kill criteria (falsify-first)

The direction is **dead** — write it up as a negative result and stop — if either holds on
the frozen real-data run:

* **No false-certification control.** The certificate's false-certification rate on pure
  invisible conditional shift (and its domain-permutation real null) is not controlled at
  `α` after LODO calibration — i.e. abstention is not actually protective.
* **No power.** On the PD medication ON/OFF positive control the certificate does not
  reach `CONCEPT_SUSPECT` stably (across subjects/seeds), i.e. it can only ever abstain →
  the certificate is vacuous.

A PASS is **not** "higher accuracy"; it is: the false-certification guarantee holds on
real invisible/null shift **and** there is stable power on the one real positive control.
Anything weaker is reported as the (still publishable) identifiability boundary itself.
