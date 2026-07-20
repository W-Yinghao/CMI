# C31 — Endpoint-Axis / Accuracy-Calibration Geometry Audit (frozen C19 `664007686afb520f`)

> C16 read as an accuracy↔calibration trade-off: target-accuracy-good checkpoints exist, the selected OACI was calibration-improved / accuracy-flat, and joint accuracy+calibration did not reproduce. C31 asks whether the rank/gauge mechanism (C22-C30) is ENDPOINT-SPECIFIC. Read-only over the C10 replay + C22 sidecar; no training / tuning / feature-selection / selector. Endpoint metrics DIAGNOSTIC-ONLY; oracles non-deployable.

- **cases: `E3_joint_good_exists_but_source_unobservable, E7_gauge_general_endpoint_offset`**

## Gate #11 — endpoint base rates & imbalance (reported BEFORE the taxonomy)

- merged in-regime candidates: **3804** (primary margin = any improvement `1e-09`, matching the frozen C22 competence label; robust margin `0.02` reported as sensitivity)

| endpoint label | base rate | count |
|---|---:|---:|
| accuracy_good | +0.473 | 1800 |
| nll_good | +0.576 | 2193 |
| ece_good | +0.617 | 2346 |
| calibration_good | +0.684 | 2601 |
| joint_good | +0.424 | 1614 |
| joint_strict_good | +0.346 | 1317 |
| pareto_good | +0.136 | 519 |
| accuracy_good_calibration_bad | +0.049 | 186 |
| calibration_good_accuracy_flat | +0.259 | 987 |

- imbalance flags: **none (all base rates in [0.05,0.95])**
- P(calibration-good | accuracy-good) = **+0.897** — joint-good is **NOT rare** (rate +0.424), so E2 is not the story.

## Q1/E1 — accuracy-calibration overlap vs trade-off

- Jaccard(accuracy, calibration) **+0.579**; accuracy_good_calibration_bad rate **+0.049**.
- bAcc-improvement ↔ calibration-improvement correlation: NLL **+0.590**, ECE **+0.605** (POSITIVE = they coincide, not conflict).
- **Epoch-confound control** (within-trajectory residualized on epoch): NLL **+0.504**, ECE **+0.554** → coupling survives epoch control: **True** (not merely training progress).
- **RED-TEAM robustness**: the shared per-trajectory ERM reference is provably inert (constant within each trajectory, max spread +0.000 → absorbed by the epoch-residual intercept, so the coupling is recovered from RAW metrics with no ERM subtraction); per-target signs **9/9** positive; target cluster-bootstrap mean **+0.607**, 95% CI **[+0.448, +0.755]** (strictly positive: True).
- **trade-off confirmed: False**. accuracy improvement is NOT negatively correlated with calibration improvement (mean corr 0.597); accuracy-good checkpoints largely overlap calibration-good -> trade-off NOT confirmed at the population level

## Q2/E2-E3 — do joint-good checkpoints exist, and are they source-observable?

- joint-good rate **+0.424**; joint-good Pareto points exist in **+0.944** of trajectories.
- source score ranks joint-good WITHIN-target AUC **+0.672** (outside a within-target permutation null, p≈0.002, sign-consistent 9/9 targets) but POOLED (cross-target, deployable) AUC **+0.541** (pooled strength **+0.041** < 0.05 deployability bar) → **E3** (joint-good is common + within-target visible but the pooled/deployable transport is GAUGE-BROKEN, same rank/gauge split as C30), NOT E2.
- **RED-TEAM caveat**: pooled 0.541 is a heavily-COLLAPSED / NON-DEPLOYABLE residual, **not literally at chance** at the primary margin — it sits just above a global-shuffle null (p≈0.002) because within-target rank leaks into the pool (the ~10% same-target pairs); it reaches literal chance (pooled 0.489, inside null) only under the 0.02 robustness margin. State pooled as 'collapsed / non-deployable', not '≈ chance'.

## Q3/E4-E5 — is the source rank accuracy- or calibration-specific?

| endpoint | within-target AUC | rank strength | sign-consistency |
|---|---:|---:|---:|
| accuracy_good | +0.659 | +0.159 | +1.000 |
| nll_good | +0.613 | +0.113 | +0.889 |
| ece_good | +0.565 | +0.065 | +0.667 |
| calibration_good | +0.620 | +0.120 | +1.000 |
| joint_good | +0.672 | +0.172 | +1.000 |
| pareto_good | +0.574 | +0.074 | +0.667 |

- accuracy strength **+0.159** vs calibration_good strength **+0.120** (ECE strength **+0.065** weakest); joint strength **+0.172**.
- **RED-TEAM DOWNGRADE (E4 not established)**: the probe SCORE is trained on the accuracy label (label==accuracy_good, **0** mismatches) so 'ranks accuracy best' is partly BY CONSTRUCTION. A 9-target cluster-bootstrap of the accuracy−calibration strength gap = **+0.039**, 95% CI **[-0.028, +0.100]** — **INCLUDES 0** (excludes 0: False). The ONLY distinguishable contrast is accuracy vs **ECE** (gap **+0.093**, CI **[+0.013, +0.160]**, excludes 0: **True**). Verdict: **endpoint-NONSPECIFIC, accuracy-aligned-by-construction** (not E4, not E5) — which reinforces the 'same object' reading. the source rank is accuracy-ALIGNED BY CONSTRUCTION (probe trained on label==accuracy_good, 0/3804 mismatches) and NOT distinguishably accuracy-specific vs calibration: acc strength 0.159 vs calibration_good 0.120, gap 0.039 95% CI [-0.028, 0.100] INCLUDES 0; the only distinguishable contrast is accuracy vs ECE (gap 0.093, CI [0.013, 0.160], excludes 0: True)

## Q4/E6-E7 — is the gauge accuracy- or calibration-specific?

- between-target variance fraction: bAcc **+0.882**, NLL **+0.842**, ECE **+0.834** (near-equal → GENERAL per-target offset). accuracy pooled-vs-within gap **+0.115** vs calibration **+0.085**. **E7_gauge_general_endpoint_offset**. the gauge is a GENERAL per-target offset: between-target variance fraction is near-equal across endpoints (bAcc 0.88 / NLL 0.84 / ECE 0.83); accuracy pooled-vs-within gap 0.115 is only mildly above calibration 0.085 (tilt inherited from the accuracy-aligned rank, not a distinct calibration gauge)

## Q5/E8 — is the C16 barrier a Pareto trade-off?

- mean Pareto front **+9.611**/traj; dominated fraction **+0.862**.
- accuracy-oracle (max target bAcc, non-deployable) is calibration-BAD in only **+0.037** of trajectories → **no Pareto wall**. **E8_pareto_geometry_explains_c16_barrier NOT established**.
- **RED-TEAM caveat (definition-favorable)**: 3.7% uses the frozen OR-calibration (any NLL *or* ECE improvement). Under STRICT both-NLL-and-ECE calibration the accuracy-oracle is calibration-bad in **+0.222** — higher, but still SUB-MAJORITY, so 'no wall' survives; do not over-weight the 3.7%.
- **Asymmetry, not a wall**: the CALIBRATION-oracle (min NLL / min ECE) is accuracy-flat in **+0.167** / **+0.296** of trajectories — a base-rate effect (calibration-good +0.684 > accuracy-good +0.473) that MATCHES C16's calibration-improved/accuracy-flat outcome rather than a symmetric trade-off. mean Pareto front 9.6/traj, dominated fraction 0.86; accuracy-oracle is calibration-BAD in 4% of trajectories; joint-good Pareto points exist in 94% of trajectories; source score ranks pareto/joint within-target AUC 0.574/0.672

## Margin sensitivity (robust margin 0.02)

- under the 0.02 robustness margin: joint-good rate **+0.278**, accuracy_good_calibration_bad **+0.019**, trade-off confirmed **False**, source-rank accuracy-specific **True**, accuracy-oracle calibration-bad **+0.111** → cases **E3_joint_good_exists_but_source_unobservable, E4_source_rank_accuracy_specific, E7_gauge_general_endpoint_offset** (verdict margin-robust).

## Bottom line — are accuracy rank, calibration rank, and joint Pareto point the same object?

> **Largely yes, at the within-target / diagnostic level (NOT deployable)** — and the E4 downgrade STRENGTHENS this. (1) E1: accuracy and calibration IMPROVE TOGETHER (+0.60, 9/9 targets, survives the epoch control and the shared-ERM check) — no trade-off separating them. (2) E4-downgraded: the source rank orders accuracy (0.159) and overall calibration (0.120) INDISTINGUISHABLY within per-target noise (bootstrap gap CI includes 0), with 90% candidate overlap — it is ranking ONE shared object, not two (an accuracy-SPECIFIC rank would have split them). ECE is the one partial exception (accuracy orders ECE distinguishably better). (3) E3/E8: the joint accuracy+calibration set is common (42%; a joint Pareto point in 94% of trajectories) and within-target rankable by the same score (AUC 0.67).
> **Three honest boundaries**: (a) the shared object is source-rankable WITHIN a target but its pooled/cross-target transport is GAUGE-BROKEN and non-deployable (E3 pooled 0.54 collapsed — not literally chance; E7 per-target offset anti-aligned) — same rank/gauge split as C22-C30, now shown endpoint-GENERAL; (b) 'largely', not identical — a base-rate asymmetry (calibration-good commoner 0.68 vs 0.47; calibration-oracle sacrifices accuracy 17-46%) and ECE as a partial exception; (c) this RECONCILES C16, it does not overturn it: the common joint-good set exists but the deployed selector cannot localize it across targets — a source-observability / gauge failure, NOT a checkpoint-space trade-off. DIAGNOSTIC-ONLY: the accuracy/calibration/joint oracles are non-deployable and no endpoint or Pareto selector is claimed.