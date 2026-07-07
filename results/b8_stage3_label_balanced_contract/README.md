# B8.3 label-balanced case-control audit contract (diagnostic-only) — INSUFFICIENT: halves but does not control the prior-collider residual

Contract redesign after B8.2 falsified B8.1's decision-level stability. **Removes label-prior via the audit sampling
contract:** a predeclared, Z-blind, deterministic **case-control selector** `A(C,Y,G,Block,seed)` balances P(Y|C) by
construction (within each (subject,block,label) stratum, select k=min(#HI,#LO) per condition); the **exact null re-applies
A under every randomized C\***. **mean-T-alone is now a PRIMARY screen** (B8.1/B8.2's lesson: the studentized both-gate only
*masks* the prior collider). Reuses the B8.1 engine byte-frozen (build/provenance/randomization); only the selector +
certifier are new. **Narrowed estimand:** label-balanced audit-population boundary evidence — NOT natural-prevalence
certification.

development-only / NOT confirmatory / NOT deployable / **NOT validation (Lee2019 SM16 emulator)** / no tag. Protocol
pre-registered + committed `d99ab79` (module sha `fa59a34134d4e645`, protocol sha `73e8f847`) and design-red-teamed
(`wrznv3lin`, no blockers) **before** the Phase-B run. Phase B = 6 fresh disjoint blocks (620-720e6) × 12 × 50 = 3600.

## RESULT — INSUFFICIENT (pre-registered): halves the residual, does not control it

Red-team `w1urtx7hm`: accounting/isolation **PASS** (3600 reproduced, engine byte-identical, seeds disjoint, exact C×Y
balance verified, 13-field bit-for-bit re-run, no fabrication); science **MINOR_ISSUE** (INSUFFICIENT genuine + decisive +
honestly framed).

| condition | both-gate /300 | **mean-T-alone /300** (PRIMARY) | screen | B8.2 mean-T |
|---|---|---|---|---|
| `CONTRACT_NULL_balanced` | 3 | 6 | both ≤7 ✓ | — |
| **`CONTRACT_NULL_prior_only`** | 2 | **28** | **mean-T FAILS** (CP95 [6.3,13.2]%) | 46 |
| **`CONTRACT_NULL_cov_plus_prior`** | 10 | **22** | **both FAIL** (mean-T CP95 [4.7,10.9]%) | 49 |
| `CONTRACT_random_label` | 4 | 5 | ≤7 ✓ (pref ≤3 missed) | — |
| `CONTRACT_POS_boundary` | **19** | 93 | **missed ≥20 strong bar** (met >0) | 37 |
| `CONTRACT_POS_boundary_plus_prior` | 30 | 105 | ≥15 ✓ | — |
| all 6 `VIOLATION_*` | **0** | — | 300/300 refused ✓ | — |

## The honest verdict (neither over- nor under-claimed)

**B8.3 is INSUFFICIENT — but it is a genuine partial result, not inert.**

1. **The label-balancing worked as designed, and halved the residual.** Count-balancing removed the first-order P(Y|C) main
   effect *exactly* (C×Y imbalance=0 everywhere), cutting the prior-collider mean-T residual **46→28** (prior_only, Fisher
   p=0.034, −39%) and **49→22** (cov_plus_prior, p=9e-4, −55%). This is a real, statistically-significant reduction.
2. **But it does not control the residual to nominal.** The mean-T PRIMARY screen requires ≤7/300 (~2.5% nominal); observed
   **28/300 and 22/300**, both CP95 lower bounds (6.3%, 4.7%) far above the bar (binom p=3.8e-9, 9.1e-6). The screen is
   decisively failed.
3. **The failure is the mean-T primary screen, NOT the masked both-gate.** prior_only's both-gate is 2/300 and would
   *falsely pass* — which is exactly why mean-T was pre-registered as primary. **Do not read 2/300 as success.**
4. **Mechanism confirmed** (as the design red-team predicted): channel (a) **selection-intensity asymmetry** — prior/mixed
   worlds discard ~390 more trials under observed-C than under randomized C\* (asymmetry **−390/−388** vs balanced **+2**),
   co-located with the breaches → the surviving second-order collider residual, not a coding fault. Channel (b) within-Y C-Z
   is design-asserted (marginal AUC is correctly zeroed by balancing, so not freshly re-measured here).
5. **POS survives but POS_boundary 19/300 missed the ≥20 strong bar** — this is **power** (the balanced audit sample is
   smaller: selN 3377 vs 4188), **not absorption**: mean-T 93/300 (Fisher vs balanced-null p=2e-24) shows the boundary
   signal is intact and the studentized gate is the bottleneck. POS+prior 30/300 met ≥15.
6. **Violations refuse 300/300** (0 alerts, incl. `quiet_cov_plus_concept` despite its real concept) — but **by
   construction** (H3 schedule-adherence refuses before selection).

## Next (reviewer decision, NOT authorized)

Per the pre-registered `if_b8_3_fails` branch, a mean-T breach on prior-bearing worlds → **B9 (genuinely randomized audit
acquisition) OR estimand-narrowing** (declare prior-bearing worlds outside the target). **NOT** mean-T recalibration, **NOT**
p-tuning, **NOT** altering the selector/statistic/null to match intensities (all pre-committed as out-of-scope; would be
post-hoc null repair). The information-contract line (B8.0→B8.3) has been pushed to its emulator limit without genuinely
randomized acquisition: the second-order collider residual is inherent to case-control-on-a-collider and count-balancing
cannot remove it.

Provenance: engine byte-reused from committed B8.1; seeds 620e6+idx·1e6+cohort, 0 intersection with all prior CSC runs;
decision = B8.1 both-gate (mean-T-alone reported + primary, never the decision gate); NULL kinds reported separately.
See `b8_stage3_protocol.json`, `b8_stage3_redteam_checks.json`, `b8_stage3_selector_tables.json`, `notes/b8_1_class_balanced_contract.md`.
Related: `b8_stage2_multiseed_stability/` (B8.2), `b8_stage1_class_balanced_contract/` (B8.1).
