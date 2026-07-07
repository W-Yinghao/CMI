# Figure Plan

Planned figures for the manuscript. Each entry names the concept, the source material, and the intended
takeaway. No figure is rendered yet.

## Figure 1 — Information regimes ladder
- `R0 (source-only) ⊑ R1 (target-unlabeled) ⊑ R2 (minimal-paired)` as a refinement chain.
- Annotate what becomes observable at each step (source law → + target marginal `p_T(X)` → + anchors).
- Source: `01_information_regimes.md`. Takeaway: observation, not the harness, sets the regime.

## Figure 2 — OACI compatibility sets
- Nested compatibility sets `K_{R,C}(o_R)` shrinking as observation refines (MONO-1) and as contracts
  strengthen (contract-strength monotonicity), with a functional `T` constant vs non-constant on `K`.
- Source: `06_oaci_identifiability.md §4–9`. Takeaway: identifiability = constancy of `T` on `K`.

## Figure 3 — Counterexample worlds
- The binary G/B world pair sharing the same source law (or same R1 observation) but disagreeing on the
  target functional — risk 0 vs 1, gain sign −1 vs +1, prior (0.2,0.8) vs (0.8,0.2).
- Source: `03 §4`, `07_counterexample_catalog.md`. Takeaway: one exact picture of non-identifiability.

## Figure 4 — Audit pipeline
- `run_three_settings → eval_bridge → ObservabilityReport → validator → tracked digest`, with the
  Claim → Verdict (allowed / reportable / identifiable / diagnostic) decision inset.
- Source: `h2cmi/observability/`, `08 §6`. Takeaway: the boundary is machine-enforced.

## Figure 5 — MOABB normalized results
- Per-dataset chance-normalized strict excess and offline-TTA gain-norm, plus offline-TTA harm-rate,
  for BNCI2014_001 (K=4) and BNCI2014_004 (K=2); BNCI2015_001 shown as legal-skip / not-applicable.
- Two panels: (a) raw bAcc (visually misleading across K) vs (b) chance-normalized excess (comparable).
- Source: `06_results_digest.md`, tracked summaries. Takeaway: normalize before comparing; TTA harms.

## Figure 6 — Claim boundary matrix
- Regime (R0/R1/R2) × estimand (target risk / prior / concept / transport / leakage) heat-matrix with
  cells: allowed · rejected · oracle-only · diagnostic · TU-1/MP-1-gated.
- Source: `04_contract_table.md`, `03_theorem_table.md`, audit registry. Takeaway: the reusable contract.

## Rendering notes
- Figures 1–4 and 6 are schematic (vector). Figure 5 is data-driven from the tracked summary JSONs
  (chance-normalized fields already present). All should be theme-neutral and colorblind-safe.
