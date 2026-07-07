# Phase 1.3 — unsafe-acceptance search + estimator diagnosis

## Setup
Inject the Bayes-unsafe synergy "safe" span `[w1,w2]` DIRECTLY into the gate
(`candidate_mode=oracle_nuisance`), task_protect=True, oracle-T. Score every prefix by the gate
(`probe_task_gain_ucb`) AND the exact Bayes oracle (`Δ*` + CI → SAFE/UNSAFE). UNSAFE_ACCEPT =
gate accepts (risk_feasible) a deletion the Bayes oracle classifies UNSAFE.

## Finding 1 — the selector was hiding the weakness
`test_bayes_calibration` showed "no unsafe acceptance" only because the **selector never routed
the dangerous candidate into the gate**. When injected directly, the gate accepts it: the smoke
(n=3000, seed 0) produced **6 UNSAFE_ACCEPT** (k=1/2, probe_task_gain_ucb 0.008–0.021 while Bayes
Δ* = 0.075–0.153). The deployed pipeline is safe by *avoidance*, not by gate *power*.

## Finding 2 — the under-detection is FINITE-SAMPLE + CAPACITY, not structural
Sweep n × (hidden, epochs) on the failing config (synergy, oracle_nuisance, oracle-T, tm=2.0,
dm=2.6, seed 0):

| n | h,ep | k | probe_ucb | bayes | gap | accepts? |
|---|------|---|-----------|-------|-----|----------|
| 2000 | 64,200 | 1 | 0.0062 | 0.0986 | 0.0924 | **UNSAFE_ACCEPT** |
| 2000 | 64,200 | 2 | 0.0536 | 0.1428 | 0.0892 | reject |
| 2000 | 256,600 | 1 | 0.0331 | 0.0986 | 0.0655 | reject |
| 2000 | 256,600 | 2 | 0.0580 | 0.1428 | 0.0848 | reject |
| 6000 | 64,200 | 1 | 0.0427 | 0.0994 | 0.0568 | reject |
| 6000 | 64,200 | 2 | 0.0601 | 0.1446 | 0.0845 | reject |
| 6000 | 256,600 | 1 | 0.0475 | 0.0994 | 0.0519 | reject |
| 6000 | 256,600 | 2 | 0.1040 | 0.1446 | 0.0407 | reject |
| 18000 | 64,200 | 1 | 0.0596 | 0.0981 | 0.0385 | reject |
| 18000 | 64,200 | 2 | 0.0931 | 0.1432 | 0.0500 | reject |
| 18000 | 256,600 | 1 | 0.0570 | 0.0981 | 0.0411 | reject |
| 18000 | 256,600 | 2 | 0.1204 | 0.1432 | 0.0228 | reject |

- `probe_ucb` rises monotonically toward Bayes Δ* with BOTH n and capacity. **No structural
  blindness was observed; estimation improves with sample size and capacity on the tested synergy
  generator.** (This limited n×capacity sweep -- largely one failing configuration -- does NOT
  prove statistical consistency; it only refutes structural blindness on this family.)
- UNSAFE_ACCEPT count: 6 (smoke n=3000) → **1** (only n=2000, h=64). With h=256 even n=2000
  rejects (k=1 probe 0.0331 > δ_Y=0.03); at n≥6000 every cell rejects.
- The **k=1 single-direction** explaining-away is the slowest to detect (gap 0.04 even at
  n=18k/h=256) — weakest interaction signal.

## Consequences
- `task_protect` stays default **OFF**: the default-on exit condition (zero UNSAFE_ACCEPT, no
  region with Bayes ≫ δ_Y but probe ≪ δ_Y) FAILS at small n / low capacity.
- Targeted fix (per diagnosis, NOT generic depth): (a) raise the gate TASK critic capacity
  (hidden 64→~256, more epochs) — closes most of the gap and removes UNSAFE_ACCEPT across the
  sweep; (b) finite-sample ABSTENTION guard — refuse to ACCEPT a deletion when the gate sample is
  too small for the estimator to certify (power floor), since probe_ucb under-detects at small n.
  Recommended: both (capacity + power-floor abstention).
- The residual critic's magnitude stays biased-low for k=1 even at large n; for the SAFETY
  decision (reject when probe_ucb > δ_Y) capacity suffices, but accurate Δ* magnitude would need
  a stronger conditional-MI estimator (future work, not required for safety).
