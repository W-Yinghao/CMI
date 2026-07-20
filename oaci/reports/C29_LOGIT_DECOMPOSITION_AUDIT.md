# C29 — logit decomposition audit (head-bias b vs representation projection W·z)

> removing the parameter head-bias b (-> W.z) PRESERVES the carrier (0.510 vs full 0.524) while removing the per-class EFFECTIVE mean DESTROYS it -> the offset-carrying effective class bias is the representation-projection mean mean(W.z), NOT the parameter head-bias b (R2). The carrier is a NONLINEAR softmax confidence, so the LINEAR b/projection-mean 4-vec gauges do not isolate it (b-gauge -0.943, projmean-gauge -0.085) -- evidence is the counterfactual on the actual carrier, not the linear gauges.

- full carrier gap +0.524 (survives True)
- parameter-bias-removed (W·z) gap +0.510
- effective-mean-removed (C27) gap -0.313
- effective-bias 4-vec gap -0.055
- parameter head-bias b 4-vec gap -0.943
- representation-projection mean W·z 4-vec gap -0.085
