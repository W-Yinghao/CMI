# C58 - Rank-Gauge Synthetic Model Spec

A toy model consistent with C30-C55 is:

```text
U_t(c) = R(c) + G_t(c) + epsilon_t(c)
```

`R(c)` is a weak source-visible rank axis. `G_t(c)` is a target-specific gauge/offset or interaction field. A source-only rule observes noisy functions of `R(c)` but not candidate-specific `G_t(c)`; endpoint diagnostics observe a target-label-derived function close to `R(c)+G_t(c)`.

Empirical anchors: source rank hit 0.506, key-only hit 0.488, label-diagnostic hit 0.813, endpoint scalar hit 0.944, and cross-target q10 divergence 0.937.

This is a theorem scaffold and simulation language only. It is not claimed as a proved EEG lower-bound model.
