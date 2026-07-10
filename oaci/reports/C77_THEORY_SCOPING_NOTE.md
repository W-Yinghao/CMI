# C77 Theory Scoping Note

## Object

Let `Y_{s,r,t,l,c}` be a held-out target utility for seed `s`, regime `r`, target `t`, trajectory/level `l`, and checkpoint candidate `c`. Let `X` be one of the prospectively registered strict-source, target-unlabeled, or split-label measurement blocks. C77 scopes, but does not prove, a heterogeneous model

```text
Y_{s,r,t,l,c} = a_{s,r,t,l} + f_{s,r,t,l}(X_c) + epsilon_{s,r,t,l,c}.
```

A nonzero within-group dependence between `X` and `Y` only establishes local measurement. Transport additionally requires stability of `f` across held-out targets, trajectories, regimes, and the seed-4 field. Control further requires enough separation at the extreme order statistic to improve top-k choice or regret.

## Extreme action

For candidate field size `M`, top-1 recovery depends on the best-versus-near-best gaps and effective near-tie multiplicity, not raw `M` alone. Even reliable bulk ordering can fail when many candidates occupy an epsilon-optimal set. C78/C79 must therefore report top gaps, effective multiplicity, random top-k baselines, and regret with association.

## Multi-regime transport

The synthetic benchmark varies local association, regime/target coefficient heterogeneity, candidate count, effective multiplicity, top-gap scale, and label budget. It is a design-calibration instrument. It demonstrates that the registered analysis can detect a stable signal and that heterogeneity can separate local association from transport. It is not evidence that real EEG follows this model.

## Identifiability boundary

Function-preserving latent reparameterizations remain valid. Orbit robustness cannot identify a W-versus-z causal origin. Strict-source, target-unlabeled, construction-label, evaluation-label, and same-label-oracle views remain separate information classes.

## Status

No theorem, EEG minimax bound, target-population claim, selector, or deployability result is asserted. Seed 3 is protocol-debug evidence; seed 4 is the future locked confirmation field. An external dataset remains a later, separately authorized stage.
