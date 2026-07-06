# B8.2 multi-seed stability replication (diagnostic-only) — B8.1 decision-level stability FALSIFIED

Reuses the **B8.1 engine byte-identical** (module `realeeg_b8_1.py` sha `dae229e39d89c940`, worker `realeeg_b8_1_canary.py`
sha `86f8dd0bb6da4ec9` — the exact committed B8.1 code) across **6 fresh disjoint seed blocks** (bases
500/520/540/560/580/600e6) × 12 conditions × 50 = **3600 cohorts**, to answer one question: is B8.1's decision-level
(both-gate) prior control + mixed-cell control + POS signal **stable** across seeds? No logic/gate/statistic/feature change;
mean-T-alone is diagnostic only; mean-T recalibration was **explicitly rejected** by the reviewer.

development-only / NOT confirmatory / NOT deployable / **NOT validation (Lee2019 SM16 emulator)** / no tag. Protocol
pre-registered + committed `d2aaf5f` (sha `62cf256e08d62551`) **before** the run.

## RESULT — B8.1's decision-level stability is FALSIFIED (Case C **and** Case D)

Red-team `w3u5q3x10`: accounting/isolation **PASS** (3600 reproduced, engine byte-identical, seeds disjoint, both-gate
recompute matches, no fabrication); science **MINOR_ISSUE** (negative genuine + honestly framed; wording refined).

| condition | B8.2 both-gate /300 | rate | screen ≤7/300 | mean-T-alone | B8.1 (n=50) |
|---|---|---|---|---|---|
| `CONTRACT_NULL_balanced` | 5 | 1.7% | ✓ | 8 | — |
| **`CONTRACT_NULL_prior_only`** | **18** | **6.0%** | **FAIL** (CP95 [3.6,9.3]%, > nominal 2.5%) | **46** | 2/50 |
| **`CONTRACT_NULL_cov_plus_prior`** | **13** | **4.3%** | **FAIL** (binom p=0.041; CI touches nominal) | **49** | 1/50 |
| `CONTRACT_random_label` | 5 | 1.7% | ✓ | 7 | — |
| `CONTRACT_POS_boundary` | 37 | 12.3% | ≥20 ✓ | 174 | 8/50 |
| `CONTRACT_POS_boundary_plus_prior` | 39 | 13% | ≥15 ✓ | 175 | — |
| all 6 `VIOLATION_*` | **0** | 0% | ✓ (300/300 refused, every block 0) | — | — |

per-seed `B8_ALERT`: prior_only **[2,2,3,3,3,5]**, cov_plus_prior **[3,1,3,2,2,2]** (fail across *all* blocks, not one outlier).

## The honest verdict (do NOT read as "B8 works" — nor as "B8 dead")

**B8.1's decision-level "meets targets" is FALSIFIED as unstable.** Specifically:

1. **prior_only and mixed both fail the ≤7/300 screen across all 6 blocks.** prior_only 18/300 = 6.0% is CI-excluded from
   nominal 2.5% (binom p=6.6e-4); cov_plus_prior 13/300 = 4.3% fails the screen (binom p=0.041), its 95% CI just touching
   nominal. **B8.1's 2/50 and 1/50 were TYPICAL single-block draws from a true ~4–6% rate** — the n=50 screen was
   **underpowered** to resolve ~5% from the 2.5% nominal (block 0 reproduced 2/50 exactly). B8.1 never had genuine control;
   it lacked the power to detect that it didn't.
2. **The studentized both-gate is anti-conservative on prior-bearing nulls.** Clean nulls (`balanced`, `random`) sit at
   ~1.7% (well-calibrated), but the prior/mixed nulls run 2–2.4× nominal. mean-T-alone is 46/49 (~15-16%); the studentized
   gate masks ~60–75% of that collider but **leaks a seed-dependent residual** through the both-gate. This is exactly what
   B8.1's result red-team predicted (masking-dependent, single-seed fragile).
3. **Survivors — the direction is NOT dead, only its null-control stability is falsified.** Violations refuse rock-solid
   (0/300 alerts, 300/300 refused, every block 0 — but **by construction** via the H3 schedule-adherence gate, which
   correctly refuses `quiet_cov_plus_concept` despite its true concept). POS is a **genuine but modest** survivor (37/300
   ≈ 12.3%, ≥20 strong; Fisher p=0.010 vs the failing null, but CP95 [8.8,16.6]% overlaps prior_only's at the margin — not
   a clean detector).

## Next (reviewer decision, NOT authorized)

Per the pre-registered decision tree this is **Case C (prior_only fails) AND Case D (mixed fails)**: **contract redesign /
narrow the estimand** and **diagnose the contract construction** — **NOT** mean-T gate recalibration (explicitly rejected;
would be another post-hoc null repair), **NOT** p-tuning, **NOT** a budget frontier yet (power isn't the blocker; null
control is). Provenance: 3600 unique seeds, disjoint from all prior runs; engine byte-identical; decision = B8.1's locked
both-gate; NULL kinds reported separately (no pooling). Emulator, semi-synthetic, single 6-block replication.

See `b8_stage2_protocol.json`, `b8_stage2_redteam_checks.json`, `b8_stage2_seed_tables.json`, `b8_stage2_mechanism_tables.json`,
`notes/b8_1_class_balanced_contract.md`. Related: `b8_stage1_class_balanced_contract/` (B8.1), `b8_stage0_contract/` (B8.0).
