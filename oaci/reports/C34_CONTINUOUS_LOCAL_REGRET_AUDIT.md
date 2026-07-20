# C34 - Continuous Local Regret / Source-Objective Direction Audit (frozen C19 `664007686afb520f`)

> Diagnostic-only read-only audit. C34 asks whether C33 local misses remain real continuous endpoint regret after removing the binary joint-good threshold as the primary object. No training, no re-inference, no selector, no selected-checkpoint artifact.

- **cases: `M2_continuous_source_active_misranking, M7_target_unlabeled_pooled_only_reconfirmed, M8_continuous_endpoint_tradeoff_local`**

## Endpoint vectors first

Selected -> nearest continuous-better mean raw endpoint deltas:
- Δtarget bAcc: **+0.020**
- Δtarget NLL improvement: **+0.055**
- Δtarget ECE improvement: **+0.022**

Fixed scalar summaries after the vector:
- mean / median Δjoint-min-margin: **+0.390 / +0.325**.
- mean / median endpoint-norm regret reduction: **+0.569 / +0.490**.
- real continuous-regret fraction among selected->continuous-better pairs: **+0.941**.
- raw Pareto-nonworse / raw endpoint-backward / negative joint-min counts among nearest continuous-better: **72 / 81 / 33** of 153. Thus `real_endpoint_regret` means regret under the fixed C34 scalar/norm summaries, not pure Pareto dominance.

## Binary vs continuous boundary

- threshold-only fraction among binary misses: **+0.000**.
- binary misses: tiny-threshold / endpoint-tradeoff / scalar-or-norm-worse counts **0 / 15 / 3** of 81. `threshold-only` is the strict tiny-difference artifact, not every broader binary-label tradeoff.
- boundary status counts: **{'real_endpoint_regret': 144, 'no_near_continuous_better': 9, 'continuous_tiny_or_weak': 9}**.
- endpoint tradeoff fraction: **+0.431**.

## Source-objective direction

- source pairwise AUC against continuous target utility: **+0.534** (local random baseline **+0.500**).
- source wrong-direction / flat fractions: **+0.391 / +0.150**.
- selected-pair source active misranking fraction: **+0.285**.
- M2 is therefore read as a substantial selected-pair minority, not as a global source objective pointing mostly backward.
- component conflicts: leakage wrong **+0.040**, risk wrong **+0.255**.

## Gauge-locality and target-unlabeled rung

- meaningful-regret gauge-jump / gauge-unseen fractions: **+0.658 / +0.053**. C34 does not treat gauge jumps alone as M6; the source-insensitivity gate must also fire.
- target-unlabeled pm1 regret delta vs source: **+0.038** (positive means worse local continuous regret than source).

## Bottom line

> C34 is margin-free relative to C33's binary label: endpoint vectors show whether selected OACI is worse than nearby alternatives under fixed continuous summaries, and where that scalar/norm summary hides endpoint tradeoffs. The taxonomy is determined by continuous regret, source-objective direction, gauge-locality, target-unlabeled local regret, and explicit threshold-artifact flags. M8 means many nearest continuous-better pairs are endpoint tradeoffs under the fixed scalar summary; it is not a replacement for the endpoint-vector table. Target endpoint labels, target gauge, and target-unlabeled factors remain diagnostic-only and non-source-only.
