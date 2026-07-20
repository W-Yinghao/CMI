# C46 Red-Team Verification

## Scope

C46 was checked as a read-only diagnostic audit over the inherited C45/C43 source-objective space. It did not train, fine-tune, re-infer, use GPU, tune source distances, add objectives, run feature selection, emit selected-checkpoint artifacts, or change the frozen C19 config hash `664007686afb520f`.

## Red-Team Findings

- **Conditioning boundary is real but specific.** Within-target q10 target-divergent rate is 0.00484 and within-trajectory q10 is 0.133, while cross-target q10 is 0.937. This supports conditioning-sensitive non-identifiability, not universal within-trajectory chaos.
- **Regime is not the main break.** Cross-regime q10 target-divergent rate is only 0.00403 because same-target cross-regime neighbors remain source-near and target-homogeneous. The high within-regime q10 rate, 0.299, means conditioning only on regime still mixes targets and remains ambiguous.
- **Target identity is not variance-dominant.** Target eta^2 for target utility is 0.247, while trajectory eta^2 is 0.615 and residual within-trajectory fraction is 0.385. Therefore CB5 remains inactive; the target boundary is a nearest-neighbor/global-comparability failure, not a claim that target identity alone explains most variance.
- **Source distance has local, not global, meaning.** Source-distance/target-gap Spearman is 0.391 within trajectory but only 0.016 cross-target. This matches C45: source geometry has local ordering information but loses cross-target comparability.
- **Seed/level apparent homogeneity is not a new method.** Within-seed and within-level nearest neighbors are homogeneous because the source space can often find same-target/same-trajectory-like counterparts inside those broad scopes. This is diagnostic grouping, not a selector or rescue path.
- **No metric shopping.** C46 inherits the C45 source distance unchanged: within-trajectory z-scored Euclidean over the C43 source objective registry. Pair-sample diagnostics use fixed seed `46046` and fixed cap `100000`.
- **No method artifact emitted.** Tables use trajectory/order coordinates and diagnostic outcomes only; they do not include model hashes, checkpoint hashes, or selected-candidate identifiers.

## Verification

- `py_compile`: passed for `oaci/source_conditioning_boundary/*.py` and `oaci/tests/test_c46_conditioning_boundary.py`.
- Slurm generation job `890180` on `cpu-high`: completed and wrote final C46 artifacts.
- Slurm focused job `890181` on `cpu-high`: `9 passed in 0.42s`.
- Slurm regression job `890182` on `cpu-high`: `221 passed in 32.22s` for C23-C46.

## Conservative Taxonomy

Accepted:

```text
CB1_source_space_informative_after_target_or_trajectory_conditioning
CB2_cross_target_grouping_breaks_source_equivalence
CB3_within_trajectory_neighborhoods_relatively_homogeneous
CB4_source_only_global_comparability_nonidentifiable
CB6_regime_conditioning_partial_not_sufficient
```

Not active:

```text
CB5_target_identity_component_explains_divergence
CB7_inconclusive_due_to_artifact_availability
```
