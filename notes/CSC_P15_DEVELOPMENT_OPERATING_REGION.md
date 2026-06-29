# CSC-P1.5 — DEVELOPMENT difficulty-envelope: operating-region map

**STATUS: DEVELOPMENT-ONLY. This map MAY NOT be used to select thresholds, define the final
operating region, or seed a confirmatory claim. NO FREEZE / NO CONFIRMATORY / NO P2.** Every rate
below is over **12 independent source–target clusters per cell**; a `0/12` cell has a one-sided 95%
Clopper–Pearson upper bound of only `1 − 0.05^(1/12) ≈ 0.221`, which **locates an operating
boundary — it is not an error-control guarantee.**

---

## 1. Provenance

| field | value |
|---|---|
| code commit (run) | `b4d1f1e` (csc CSC-P1.5-parallel) |
| artifact commit | `3e5bcf5` (origin/csc) |
| audit baseline | `4ea423d` (P1.4.5a DEV AUDIT, provenance_ok) |
| protocol manifest hash | `da2c0f4309847a4e790843b9ece68010a90c33bdb9404097aee72dcbefbb2632` |
| canary job | SLURM `875529` — `CANARY_ONLY_PASSED`, preflight 5/5, validator_ok, 0 problems (base_seed 500000, ×2) |
| full job | SLURM `875553` on `nodecpu08` — `FULL_COMPLETE`, `validator_ok=True`, 0 problems, `canary_ref_verified=True` |
| base_seed (full) | `600000` (distinct from the 0–9 audit smoke set and the 500000 canary set) |
| parallelism | joblib `n_jobs=64`, BLAS threads pinned to 1; `test_parallel_matches_serial` proves the parallel grid is **bit-identical** to serial |
| protocol calls | 1728 (24 cells × 12 clusters × 6 target kinds) |
| wall-clock | 2026-06-29 12:53:51 → 13:19:03 (~25 min; serial would be ~6 h) |

The certifier path is the **frozen, audited** `execute_protocol`; the harness only reuses it +
`_cp_bound` + `_concept_failure_reason`. No certifier parameter (`tau_margin`, `cov_loading_margin_
kappa`, `consensus`, …) was changed by this sweep.

---

## 2. Grid

- **24 star-grid cells**: a baseline `EnvelopePoint`, then one cell per `(axis, level)` that differs
  from it (one axis varied at a time — NOT a full Cartesian product).
- **12 independent source–target clusters per cell.** Each cluster = one fresh source seed +
  one target per kind drawn from that source's geometry. Every cluster contributes **exactly one
  Bernoulli** to each cluster-level endpoint (`any_forbidden`, `fired`, `any_false_concept`).
- 9 difficulty axes → simulator knobs: `concept_effect_size`, `subjects_per_domain`,
  `epochs_per_subject` (epochs_max), `within_subject_corr`, `class_imbalance` (prior_alpha),
  `concept_eigengap_sep` (**PROXY** via concept_domains), `covariate_leakage`, `target_subjects`,
  `mechanism_family`.
- ⚠️ **`eigengap_axis_is_proxy=True`**: the simulator has a single `w_concept`, so the
  concept_domains axis is a *proxy* for eigengap separation, **not** a true multi-axis eigengap
  stress test. A genuine multi-concept-axis simulator extension is deferred.

---

## 3. Primary false-certification map (forbidden / 12)

**21 / 24 cells: `0/12` forbidden.** Three boundary cells show forbidden certificates:

| cell | forbidden/12 | CP upper | must-abstain FP | false-concept (stable) |
|---|---|---|---|---|
| `prior_alpha=0.5` (heavy class imbalance) | **2/12** | 0.438 | 2 | 0 |
| `epochs_max=12` (few epochs/subject) | **1/12** | 0.339 | 1 | 0 |
| `covariate_leakage=16` (strong leakage) | **1/12** | 0.339 | 1 | 0 |
| all other 21 cells | 0/12 | 0.221 | 0 | 0 |

**Key asymmetry — `false_concept_on_synthetic_null = 0` in ALL 24 cells.** The certifier **never**
over-claimed `CONCEPT_SUSPECT` on a generator-stable (covariate / clean) target, anywhere in the
envelope. All **4** forbidden events are **must-abstain false-positives** (a definite certificate
on a `clean / pure_conditional / label_shift / label_covariate_mixed` shift). So the observed
failure direction is *over-claiming structure/compatibility on hard unidentifiable shifts under
imbalance / leakage / data-starvation* — **not** over-claiming concept.

> Limitation: the artifact records cluster-level counts (`any_forbidden`, `must_abstain_FP`,
> `false_concept`), not the per-(cluster,kind) confusion, so the exact offending kind/state in the
> 3 boundary cells is not recoverable from this run. Logging the confusion matrix is a next-run
> enhancement.

---

## 4. Visible-concept power map (power, CP lower bound)

Power rises monotonically with the favourable axes and collapses in the data-starved / imbalanced /
fully-correlated regimes:

| axis sweep | power (CP-LB) by level |
|---|---|
| `concept_effect_size` | 6→0.17(0.03) · 10→0.42(0.18) · **14(base)→0.83(0.56)** · 20→0.92(0.66) |
| `subjects_per_domain` | 8→0.00(0.00) · 14→0.17(0.03) · **22(base)→0.83(0.56)** · 30→0.75(0.47) |
| `target_subjects` | 10→0.17(0.03) · 20→0.50(0.25) · **30(base)→0.83** · 50→0.75(0.47) |
| `within_subject_corr` | 0.0→0.58(0.32) · **0.2(base)→0.83** · 0.5→0.58(0.32) · 1.0→0.08(0.004) |
| `class_imbalance` (prior_alpha) | 0.5→0.00(0.00) · 1.0→0.17(0.03) · **4.0(base)→0.83** |
| `concept_eigengap_sep` (PROXY) | 1→0.25(0.07) · **3(base)→0.83** · 5→0.33(0.12) |
| `covariate_leakage` | 2→0.83 · 6→0.83 · **10(base)→0.83** · 16→0.83 |
| `mechanism_family` | **0(base)→0.83** · 1→0.42(0.18) · 2→0.42(0.18) |

Notes:
- **Covariate leakage does NOT reduce power** (flat 0.83 across 2/6/10/16) — the cross-fitted
  decoder is robust to nuisance movement; leakage's only adverse effect is the single must-abstain
  false-cert at the extreme (16).
- **Atlas availability** tracks power: high (~0.92) in the favourable region; collapses where power
  collapses — `subjects_per_domain=8`→0.00, `prior_alpha=0.5`→0.08, `concept_domains=1`→0.25,
  `subjects_per_domain=14`→0.17, `prior_alpha=1.0`→0.17, `within_subject_corr=1.0`→0.42.
- **Abstention rate** (all cells × kinds) is high everywhere (0.81–1.00), as designed: the
  certificate abstains by default and fires only on the visible-concept kind.

---

## 5. Gate-failure decomposition (why visible-concept clusters did NOT fire)

| regime | dominant binding gate |
|---|---|
| favourable, high-atlas (e.g. effect=6, target=10) | `not_dominant_or_robust_consensus_abstain` — atlas present, but the certifier conservatively abstains (consensus / dominance not met) |
| low-atlas / imbalanced (subjects=8, prior_alpha=0.5, concept_domains=1/5) | `geometric_maxstat_not_sig` — no concept atlas estimable → no concept evidence |
| medium data / high correlation (subjects=14, corr=1.0) | `residual_T_not_sig` — cross-fitted decoder gate not significant |
| corr=1.0, epochs=40, prior_alpha=1.0, concept_domains=1, mechanism_family=1/2 | a single `unstable_concept_attribution` cluster (source_invalid_rate = 0.083 = 1/12) |

- `support_invalid_rate = 0` in **all** cells (no `INVALID_SUPPORT` ever triggered).
- `source_invalid_rate` is 0 except six cells at 0.083 (one cluster), all from
  `UNSTABLE_CONCEPT_ATTRIBUTION` — i.e. the fail-closed attribution gate firing exactly as designed.

---

## 6. Identifiable core (DEVELOPMENT-observed — descriptive ONLY)

There **is** a DEVELOPMENT-observed favourable region where the method gives non-zero power and **no
forbidden certificate was observed** (`0/12`):

```
strong concept effect (concept_effect_size ≥ 14)
  AND  adequate source (subjects_per_domain ≥ 22)
  AND  balanced classes (prior_alpha = 4.0)
  AND  adequate target (target_subjects ≥ 20–30)
  AND  covariate_leakage ≤ 10
  AND  within_subject_corr ≤ 0.5
```

Representative cells in this core (all `0/12` forbidden):
`concept_effect_size=20` (power 0.92, CP-LB 0.66), `baseline` (0.83, 0.56),
`covariate_leakage=2/6` (0.83, 0.56), `subjects_per_domain=30` (0.75, 0.47),
`target_subjects=50` (0.75, 0.47).

**What may be claimed:** *a DEVELOPMENT-observed identifiable core exists — in it the method yields
non-zero visible-concept power and no forbidden certificate was observed this round.*

**What may NOT be claimed:** *the core is false-certification controlled.* With 12 clusters/cell a
`0/12` result only bounds the per-cell forbidden rate at ≤ 0.221 (95% CP). This is a boundary
locator, not error control. Thresholds are **not** frozen and the operating region is **not**
defined here.

---

## 7. Negative boundary (where Z-only certification must refuse)

These cells are the paper's *negative* contribution — the regimes where the certificate loses power,
loses its atlas, or (worse) emits a forbidden certificate:

| boundary | observed effect |
|---|---|
| **class imbalance** (`prior_alpha=0.5`) | power 0.00, atlas 0.08, **2/12 forbidden** (must-abstain FP) — the worst cell |
| **data-starved source** (`subjects_per_domain=8`) | power 0.00, atlas 0.00 (no atlas estimable at all) |
| **full within-subject correlation** (`corr=1.0`) | power 0.08, atlas 0.42, residual gate fails |
| **too few epochs/subject** (`epochs_max=12`) | **1/12 forbidden**, power 0.33 |
| **strong covariate leakage** (`covariate_leakage=16`) | power preserved (0.83) but **1/12 forbidden** |
| **single concept domain** (eigengap proxy, `concept_domains=1`) | power 0.25, atlas 0.25 |

This is the direction's core value: **not a universal concept detector, but a certificate that says
when concept shift is identifiable from unlabeled Z and when it must abstain** — and the boundary is
empirically mapped here.

---

## 8. Next gate (NOT authorised by this run)

- A **freeze** requires: a *pre-declared* identifiable core (chosen WITHOUT looking at this map's
  per-cell outcomes as a selection criterion), a frozen manifest, and a *separate, previously
  UNSEEN* cluster set.
- A **confirmatory** claim requires an independent seed list with a pre-registered CP-upper /
  power-lower criterion and enough independent clusters (≈ ≥ 59 zero-failure clusters for a 0.05
  bound).
- **P2 (real EEG)** remains gated behind both.

Until then: this artifact is **accepted for descriptive review only**. Do not tune any threshold,
do not delete any cell, do not pool cells to claim global control, and do not treat the 12-cluster
CP bounds as confirmatory.
