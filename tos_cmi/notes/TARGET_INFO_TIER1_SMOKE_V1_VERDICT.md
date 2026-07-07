# Tier-1 smoke v1 (hardened) — VERDICT (frozen)

Semi-synthetic; NOT a final paper claim. Jobs 888028 (v0) + 888437 (v1 hardened). Commits: v0 result 5c4d6d6,
hardening 7e7e4a3, v1 result 9c87377.

```
v0 (bootstrap LCB) showed a target-label signal but it was OVERCONFIDENT: accepts rose with k, but the
  small-k bootstrap LCB false-accepted ~23% (100% false at k=1).
v1 (hardened finite-sample bounded LCB: empirical-Bernstein, Bonferroni over classes x candidate interventions,
  underpowered -> abstain) ELIMINATED false accepts: deployable false-accept rate 0.000, B2/B3 accepts 0 at every
  k in {1,2,4,8,16}.
BUT true accepts also vanished (0 deployable accepts anywhere).
B4 ORACLE (all target labels; diagnostic) shows the target benefit EXISTS: audit ΔbAcc mean +0.018
  (v2_source_invisible) / +0.021 (source_rich), max +0.080.
=> The bottleneck is CERTIFICATION POWER, not absence of effect. With k<=16 labels/class, a valid finite-sample
  bound cannot certify a ~+0.02 target benefit. This empirically instantiates CEILING_THEORY Proposition 3:
  the source-only ceiling can be broken by target labels, but SAFELY breaking it needs enough target-label budget
  or a sharper estimator.
```

Conclusion: proceed to the **label-budget frontier** (extend k on the SAME data + hardened estimator), NOT seed
robustness and NOT Tier-2. Forbidden framing: "few-shot target labels safely solve the ceiling" /
"target-informed gate is validated". Correct framing: **signal real; safe certification not achieved at k<=16.**
The uploaded PDF is a stale 2a-only snapshot; do NOT write this into the manuscript.
