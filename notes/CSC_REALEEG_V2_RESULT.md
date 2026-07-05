# CSC real-EEG v2 validation — RESULT (scientific FAIL; C6-confirmed)

**Frozen, pre-registered, semi-synthetic-on-REAL-Lee2019-features validation. Final verdict: package = FAIL
(TIER1 B3 real-feature safety fails). Independently re-derived and confirmed by C6 (4 lenses, 0 issues).**
No real-EEG clinical / PD claim. Synthetic tags dee8958 / 0595f64 untouched.

## Provenance
| field | value |
|---|---|
| tag | `csc-realeeg-v2` |
| tag commit / git_head | `5cce744` (HEAD == tag, tree clean) |
| SLURM job / node | `881756` / nodecpu01, cpu-high, 24 cores |
| execution | cohort_parallel, n_jobs=24, resumed=false, joblib/loky; BLAS all=1 |
| runtime | 6913 s ≈ **1 h 55 min** (801/801 cohorts, 0 worker errors) |
| artifact | `csc/results/realeeg_validation_v2.final.json` |
| artifact sha256 | `e39bdfa9109c65e696a91abbcbbd788d9dfd9497cbb34ba3fd1414e7ec8b7b36` |
| cache sha256 | `5196b6d6460b31965f4a3bde4af40770f0419df1ecaaa15ffd0651cca0c22a68` |
| cache_metadata sha256 | `68cd8c95c8d8bc61c54c51d77f0f018a42a39906df546fba070eb56e3dc529b9` |
| bank manifest sha256 | `214b97de04492fdc47e634347e3e51107f4b0d3a23a82954766ab670151f9492` |
| routeA manifest sha256 | `7c395af1f5ae4d4bbaddef48070d274f6f8eca470c0c1857e09d70c46b762787` |
| routeB3 manifest sha256 | `5b0faeb8bc3bc96d7573759db4026f498e5c849d00a479dafe48dd9c5cc61650` |
| engine sha256 | `d5aaaf788970e18337169230cdeb80e3dfffae307bee4cf5ab4816ca84ae03cc` |
| runner sha256 | `7ed9d3ff949f0b613dbec6ff8b018cc12c980a53da1efc7c9884fe7e72ae3ac2` |
| task_table sha256 | `33847a9e44048f79…` |
| n_tasks | **801** (8 multi-cohort ×100 + 1 genuine; all unique, none missing/duplicated) |
| base_seed | `20000000` |

## TIER 1 — B3 real-feature SAFETY (GATING) → **FAIL**
DECIDED-only denominator; per-condition PASS iff cohort-bootstrap upper (B=2000) ≤ 0.05.

| gating control | ground truth | n_valid | confirmed | rate | boot_upper | invalid_frac | status |
|---|---|---|---|---|---|---|---|
| **NULL_cov** (primary) | NO_CONCEPT, real covariate drift | 100 | **15** | 0.15 | **0.21** | 0.00 | **FAIL** |
| NULL_label | NO_CONCEPT (label-shift trap) | 100 | 0 | 0.00 | 0.00 | 0.00 | PASS |
| **NULL_cov_plus_label** | NO_CONCEPT, cov+label | 100 | **6** | 0.06 | **0.10** | 0.00 | **FAIL** |
| random_label_control | NO_CONCEPT (sanity null) | 100 | 0 | 0.00 | 0.00 | 0.00 | PASS |

**Package = FAIL** (≥1 gating FAIL). The primary type-I gate **NULL_cov** false-confirms concept on 15/100
real-covariate-drift NO_CONCEPT cohorts, bootstrap upper **0.21 ≈ 4.2× the 0.05 family target**; the
combined-nuisance NULL_cov_plus_label also fails (0.10 ≈ 2×). Label-only and random-label controls are clean
(0/100), so this is a **specific** failure mode, not a machinery that confirms everything.

## TIER 2 — B3 POWER (reported, non-gating)
| condition | n_valid | confirmed | rate | boot_upper | invalid_frac |
|---|---|---|---|---|---|
| POS_concept | 95 | 32 | 0.34 | 0.42 | 0.05 |
| POS_concept_plus_cov | 97 | 25 | 0.26 | 0.33 | 0.03 |

Modest power; **cannot** change the package verdict (non-gating).

## TIER 3 — Route A trial-label MI diagnostic (reported, non-gating)
Route A **abstains (UNIDENTIFIABLE) 100/100 on every injected condition** (incl. POS_concept) and NOT_APPLICABLE
on the within-session control — zero false-confirm, zero power. Consistent with the A-negative line: on a
real trial-label MI substrate Route A never confirms. Transfer diagnostic only; NOT a revalidation of the
subject-label synthetic A.

## Genuine session contrast (DESCRIPTIVE only — NOT ground truth)
One cohort over all eligible subjects: B3 = `NO_CONCEPT_EVIDENCE_AFTER_PAIR_AUDIT`, A = `UNIDENTIFIABLE`.
Cannot affect PASS/FAIL; a real CONCEPT_CONFIRMED here would not be validated truth.

## Invalid / abstain / INCONCLUSIVE
Gating conditions: invalid_frac = 0.00 (all 400 records DECIDED) → no INCONCLUSIVE, no denominator padding.
Only abstain state anywhere = NEED_MORE_LABELS (29 total, all in POS/power conditions, never gating). B3 states
across all 801: NO_CONCEPT 672 / CONCEPT_CONFIRMED 100 / NEED_MORE_LABELS 29; 0 INVALID/ENGINE_ERROR/sampler
failures/boot-invalid.

## C6 red-team re-aggregation → CONFIRMS FAIL
4 independent adversarial lenses (re-aggregation, denominator-integrity, interpretation-honesty,
provenance-completeness) each re-derived the verdict from the raw 801 records: **all recompute FAIL, all match
the reported verdict byte-exactly (bootstrap uppers 0.21/0.10/0.0/0.0 to |diff|=0), 0 serious issues.**
Seed-robust (alternative seed-index interpretation gives the same 0.21/0.10). Anti-gaming: under a naive
all-cohorts denominator the result is identical (0 abstentions in gating). See CSC_REALEEG_V2_C6_REDTEAM.md.

## FINAL VERDICT
**The frozen paired B3 concept-shift certificate does NOT control type-I error on real Lee2019/OpenBMI features:
its primary safety gate (false-confirmation under real cross-session covariate drift) FAILS (upper 0.21).**
B3's synthetic PASS does **not** transfer to real EEG covariate structure. This is a real-feature NEGATIVE,
consistent with the whole CSC arc (no positive source-free/minimal-label detector survives a frozen test).
Honest scope: this is a semi-synthetic-on-real-features safety result on SM16_no_FCz log-bandpower MI features;
**not** a real clinical concept-shift / PD ON-OFF validation.

## Runtime correction (honest disclosure)
The pre-run abort of v1 was justified with a benchmark of ~681 s/cohort measured on a **contended** node
(nodecpu04, my own session loading it) → "~6.3 days serial > 5-day wall → infeasible." The v2 run shows the real
clean-core cost is **~207 s/cohort** (6913 s / effective 24 cores), so the benchmark was **~3.3× inflated** and
v1 serial would have been **~1.9 days — within the 5-day wall**. The accurate label on v1 is *slow + fragile
(single-core, no checkpoint), not strictly impossible.* This does not affect the result (v2 ≡ v1 byte-identical)
or the value of v2 (24× faster + checkpointed), but the "infeasible" framing was overstated; recorded here for
the record.

## Per protocol
Result preserved; **NOT rerun** (scientific FAIL is a valid result); thresholds/seeds/manifests/tag unchanged;
no code change in this result commit; genuine contrast descriptive-only; no clinical/PD claim.
