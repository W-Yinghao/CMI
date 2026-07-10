# C75 - Factorization Non-Identifiability Note

## Equivalence

Let a linear classifier head satisfy `logits = Wz + b`. For any invertible matrix `A`, define:

```text
z' = A z
W' = W A^{-1}
```

Then `W'z' = W A^{-1} A z = Wz`, so logits, probabilities, margins, predictions, and every metric computed from them are unchanged.

## Identifiability Boundary

`Wz` is function-level identifiable once persisted `b` is known because `Wz = logits - b`. The separate origin of an effect in `W` versus coordinates of `z` is not identifiable from the prediction function alone. General invertible reparameterizations change z norms/covariances, W row geometry, and normalized z/W alignment; orthogonal transforms preserve only the registered orthogonal invariants.

The actual frozen architecture fixes one reproducible coordinate system, so coordinate-tied descriptors remain measurable. Reproducibility does not turn them into a unique causal origin: that requires extra architectural assumptions or an intervention that breaks the equivalence class.

## Synthetic Audit

Across identity, orthogonal, diagonal-scale, and well-conditioned non-orthogonal transforms, the maximum Wz error is `8.881784197e-15` and every prediction function is invariant. Diagonal and non-orthogonal transforms change coordinate geometry.

The construct-validity benchmark gives detection rates:

- stable endpoint-irrelevant descriptor: `0.026`
- functionally redundant descriptor: `0.044`
- truly incremental representation descriptor: `1.000`

## C75 Consequence

C75 can identify exact Wz/logit redundancy and test whether registered coordinate-tied blocks add held-out information. It cannot uniquely name W or z as the origin of an association. The significant nonlinear proxy is therefore an architecture-tied association only, not representation causality, target gauge, or a selector.
