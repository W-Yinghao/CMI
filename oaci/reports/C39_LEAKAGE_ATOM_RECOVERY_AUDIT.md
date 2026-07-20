# C39 - Leakage Atom Recovery / Support-Cell Conflict Audit (frozen C19 `664007686afb520f`)

> Read-only CPU replay over Phase-A source-train/source-audit frozen features. No training, no selector repair, no selected-checkpoint method artifact, and no per-atom UCL summation.

- **cases: `A9_atom_decomposition_irrecoverable, A10_ucl_quantile_atom_limit`**
- C37 exact-UCL pairs imported: **114**.

## Identity Gates

- selection point identity: **48 / 76** candidates.
- source-audit additive identity: **76 / 76** candidates.
- max selection point diff: **0.000**; max atom additive diff: **0.000**.

## Point Atom Diagnostics (Blocked)

- Persisted aggregate identity did not pass the frozen gate; the following numbers are diagnostic replay summaries, not elevated atom contribution claims.
- concentrated pairs: **0 / 114**; broad pairs: **108 / 114**.
- mean top-3 positive atom share: **0.368**; mean HHI: **0.086**.

## Audit And Support

- mean selection-to-audit atom sign preservation: **0.519**.
- selection-to-audit aggregate inversion: **0.447**.
- support-artifact pair fraction: **0.000**; dominant low-mass fraction: **0.000**.

## Target Gauge

- atom-target gauge conflict: **105 / 114**.
- target gauge prefers better / selected: **105 / 9**.

## Boundary

- Atom sums are exact for the recomputed point, but persisted aggregate point identity is required before atom contribution claims.
- UCL is a bootstrap quantile; C39 does not define or sum per-atom UCLs.
- Target endpoints and target gauge remain diagnostic-only and non-source-only.

## Bottom Line

> C39 does not elevate atom contribution claims: additive decomposition is exact for the recomputed point, but persisted C37 aggregate point identity is not bit-exact under the frozen tolerance, so A9 blocks atom-mechanism conclusions.
