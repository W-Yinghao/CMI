# C11a — OACI leakage-control line CLOSED (kill decision)

**In-project kill-decision record.** This exists to prevent re-entering the "maybe just tune the selector /
the split" loop. It closes OACI's *leakage-minimization-as-downstream-benefit-mechanism* hypothesis.

## Evidence

- **C8** (BNCI2014-001 LOSO seeds [0,1,2], native K1/K2; report `C8_BNCI001_LOSO_SEEDS012_K1K2`): NEGATIVE.
  K1 held-out audit leakage reduction does not survive multiplicity (Bonferroni & BH-FDR survivors = 0); K2
  frozen sweep = `stop_no_reproducible_gain` (worst_domain_bacc harmed 4/6 units).
- **C10a** (artifact-only diagnostics; report `C10_OACI_FAILURE_DIAGNOSTICS`): selection→audit optimism is
  total (OACI cuts SELECTION leakage 54/54 but audit transfer corr ≈ 0); audit leakage change is orthogonal
  to target worst-domain bAcc/NLL/ECE (all |pearson| < 0.14); larger λ costs target accuracy (corr −0.29).
- **C10b** (epoch-level counterfactual selector replay; identity 216/216, S0_current reproduces C8's K2
  EXACTLY): **case `C_oracle_also_fails`.** No source-only guard selector (S1–S4) recovers reproducible
  worst-domain K2 gain, AND the source_audit ORACLE (S5) — which picks an OACI checkpoint 51/54 by best
  held-out source worst-domain bAcc — ALSO fails. A better OACI checkpoint does not exist in the trajectory
  even by held-out source signal.

## Decision

**Stop investing in conditional-domain leakage control as a downstream-benefit mechanism.** The two fixable
diagnoses are ruled out: it is not a selector problem (case A) and not a selection-split problem (case B).

## Keep (still valuable)

- OACI support graph (per-class identifiable/estimable cells, fixed reference prior).
- Extractable-leakage probe (`L_Q^ov` grouped cross-fit) — as a MEASUREMENT / falsification instrument.
- K1 grouped-permutation machinery — as measurement.
- Artifact / provenance / staged-executor / target-isolation stack.

## Do NOT continue (under the OACI control hypothesis)

- OACI-v2 constrained selector (that was case A — ruled out).
- source-audit / selection-split redesign for OACI (case B — ruled out).
- seeds [3,4] of the OACI control sweep.
- BNCI2014_004 under the current OACI control hypothesis.
- treating C8/C10's weak nominal K1 signal as evidence the method is salvageable by tuning.

## Next (separate hypothesis — C11b/C11c)

Pivot to **endpoint-aligned source-robust** training (SRC / RF-WDC): directly optimize a smooth source-side
worst-domain endpoint surrogate under the same risk-feasibility constraint, selected by a source-train-only
endpoint guard; keep K1/K2 as measurement (SRC is not driven by K1). This tests a DIFFERENT hypothesis, not
a repair of OACI.
