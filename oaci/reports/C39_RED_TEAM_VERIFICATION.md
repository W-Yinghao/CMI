# C39 Red-Team Verification

Scope: C39 leakage atom recovery artifacts and reports.

## Verdict

- Red-team result: **pass with A9 boundary enforced**.
- C39 is a negative recoverability result: `A9_atom_decomposition_irrecoverable + A10_ucl_quantile_atom_limit`.
- Atom contribution summaries are retained only as diagnostic replay tables because persisted aggregate point identity failed the frozen `1e-9` gate.

## Checks

- Aggregate identity gate: **48 / 76** selection candidates passed; max persisted-point drift was `2.1521578246364026e-4`.
- Additive atom identity: recomputed point atom sums were exact within floating error; max additive diff was `4.440892098500626e-16`.
- Source-audit replay: additive identity passed **76 / 76** candidates and is not used as a selection proxy.
- UCL boundary: no per-atom UCLs are summed; UCL remains a bootstrap quantile aggregate diagnostic only.
- Support-cell audit: low-mass/support-edge flags no longer treat equal-mass cells as low-mass; support artifact fraction is `0.0`.
- Artifact hygiene: compact JSON is under 10 KB; largest C39 table is about 1.1 MB; no large monolithic payload.
- Output scan: no checkpoint hash, deployment claim, rescue claim, source-only detector claim, or selected-checkpoint method artifact claim.

## Blocked Claims

- No class/domain/support-cell atom mechanism claim is elevated.
- No broad-vs-concentrated leakage advantage claim is elevated.
- No atom-target gauge conflict claim is elevated as a mechanism conclusion.
- No selector repair, OACI-v2, deployment, target-unlabeled DG success, or source-only detector is claimed.
