# CSC confirmatory result — frozen `csc-confirmatory-v1` / `dee8958` — **scientific FAIL**

**Authorized single confirmatory run** (K=1, `P_baseline` only, pointwise). The frozen,
pre-registered, development-informed identifiable core **does not survive unseen confirmatory
testing**: both endpoints fail. Per protocol this is a **valid scientific outcome** — committed, not
rerun; no threshold/seed/tag change.

## Provenance (verified)

| field | value |
|---|---|
| SLURM job | 876329 (nodecpu03, 64 cores) |
| frozen tag → commit | `csc-confirmatory-v1` → `dee8958` (`git_head == expected_code_commit`, tree clean) |
| manifest hash | `da2c0f4309847a4e790843b9ece68010a90c33bdb9404097aee72dcbefbb2632` |
| artifact | `csc/results/confirmatory.json`, sha256 `8b07524ecc3b…` (freshness-verified: fresh product of job 876329) |
| source seeds → target seeds | `900000..900065` → `1800000..1800065` (disjoint) |
| scientific RC | 1 (FAIL); `headline_core_pass = false` |

## Result — `P_baseline`, **FAIL**

| metric | value | endpoint |
|---|---|---|
| G / n_valid / source_invalid | 66 / **65** / 1 (`UNSTABLE_CONCEPT_ATTRIBUTION`, frac 0.0152 ≤ cap 0.10 → evaluable) | — |
| **forbidden** | **1 / 65**, CP-upper **0.0709 > α=0.05** | **FAIL** (false-cert control NOT confirmed) |
| **fired** | **28 / 65** | **FAIL** (power) |
| min_fired (cond / uncond / for_pass) | 40 / 41 / **41** | 28 < 41 |
| power conditional (CP-lower) | 0.431 (**0.326**) | < bar 0.50 |
| power unconditional (CP-lower) | 0.424 (**0.321**) | < bar 0.50 |
| gate-failure decomposition | FIRED 28 · not_dominant_or_robust_consensus_abstain 15 · geometric_maxstat_not_sig 15 · residual_T_not_sig 7 · unstable_concept_attribution 1 | (sums to 66) |

## Interpretation — development → confirmatory generalization gap

On the **same operating point**, the CSC-P1.5 DEVELOPMENT map reported `P_baseline` power **0.83**
(CP-LB 0.56) with **0/12 forbidden**. On **unseen** clusters (frozen manifest, `base_seed=900000`)
the same point yields power **0.43** (CP-LB 0.33) and **1 forbidden** (CP-UB 0.071). The
development-informed core therefore:

- **does NOT control false-certification** at α=0.05 (a forbidden certificate appeared on an unseen
  cluster; CP-UB 0.071), and
- **does NOT retain ≥0.50 power** (well below the screening bar; 28 < 41 needed).

The development numbers were optimistic — an informed-selection / generalization gap that the frozen,
pre-registered confirmatory test was designed to expose, and did.

## Consequence (no action taken here — for reviewer decision)

This is an honest **negative confirmatory result**. The defensible paper contribution is the
**identifiability/abstention boundary** and this **negative operating-region theorem-with-evidence**
(a frozen, pre-registered Z-only concept-shift certificate does not confirm both control and power on
unseen synthetic clusters even inside a development-favourable core) — **not** a positive
concept-detection method. Consistent with the broader project finding that no positive source-free
method survived adversarial audit.

Per the frozen protocol's non-selection rules: thresholds, seeds, manifest, and tag are **unchanged**;
the run is **not** rerun; `P_strong` remains secondary descriptive (not run here). Any next step
(e.g. revising the estimand/method, or framing the paper around the abstention boundary + this
negative result) is a **new** design round, not a tweak of this frozen tag.

```
FREEZE: used (csc-confirmatory-v1 / dee8958)
CONFIRMATORY RUN: COMPLETE — scientific FAIL (valid, recorded)
P2 real EEG: NOT AUTHORIZED
```
