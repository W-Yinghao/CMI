"""C29 — Representation-Head Origin of Target Class-Conditioned Confidence. Read-only mechanism audit of WHERE
the C27 carrier originates in the model: the linear head (logit = W.z + b) decomposes the target logits into a
parameter head-bias b and a representation projection W.z = (logit - b), so the offset-relevant representation
contribution is available read-only. Tests parameter-bias (R1) vs representation-projection / target shift
(R2/R3/R7) vs interaction (R5). DIAGNOSTIC-ONLY; not a selector / DG / rescue."""
