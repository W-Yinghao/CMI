# C45 Red-Team Verification

## Scope

C45 was checked as a read-only diagnostic audit over inherited C43/C44 source objectives and existing C42/C41/C35/C32R/C30/C29-era artifacts. It did not train, fine-tune, re-infer, use GPU, add objectives, tune scores, run feature selection, emit selected-checkpoint artifacts, or change the frozen C19 config hash `664007686afb520f`.

## Red-Team Findings

- **Nearest-neighbor base-rate check changed the interpretation.** Full all-source within-trajectory q10 neighborhoods are comparatively homogeneous: q10 target-divergent rate is 0.133, q05 target-utility variance / trajectory baseline is 0.000256, and q10 joint-label entropy / baseline is 0.0356. Therefore N2, N3, N4, and N8 remain inactive.
- **N1 is scope-conditioned.** Source-equivalent target-divergent witnesses are strong across target and same-regime source-neighborhoods: cross-target q10 target-divergent rate is 0.937 and same-regime q10 is 0.299. Within-target is near-identical (0.00484), and within-trajectory all-source q10 is modest (0.133).
- **Gauge over-credit was fixed before finalization.** The first generated taxonomy over-counted target-gauge witnesses by including joint-label disagreements. The final C45 gauge witness count requires a large target-gauge gap: 17 / 632 source-equivalent divergent pairs, fraction 0.0269. Although gauge gap and target-utility gap correlate at 0.726, this is not enough to activate N5.
- **Family-reduced ambiguity remains real.** All reduced source spaces retain q10 target-divergent rates above 0.2699. Rank-only reduces joint-good disagreement relative to the trajectory-conditioned baseline (0.238 vs 0.311) but leaves high q10 target-divergence (0.595), so N6 and N7 are active.
- **No metric shopping.** Primary distance is frozen as within-trajectory z-scored Euclidean over inherited C43 source objectives; rank-normalized L1 and family-block normalized distance are secondary and pre-registered in `schema.py`.
- **No source/target leakage in construction.** Target endpoints, joint-good, Pareto-good, preference-robust labels, and target gauge are used only after source-neighborhood construction as diagnostic outcomes.
- **No method artifact emitted.** Witness rows use trajectory/order coordinates only and do not include model hashes, checkpoint hashes, or selected-candidate identifiers.

## Verification

- `py_compile`: passed for `oaci/source_nonidentifiability/*.py` and `oaci/tests/test_c45_source_nonidentifiability.py`.
- Slurm generation job `890170` on `cpu-high`: completed and wrote final C45 artifacts.
- Slurm focused job `890171` on `cpu-high`: `10 passed in 0.22s`.
- Slurm regression job `890172` on `cpu-high`: `212 passed in 32.54s` for C23-C45.

## Conservative Taxonomy

Accepted:

```text
N1_source_equivalent_target_divergent_witnesses
N6_family_reduced_space_not_sufficient
N7_rank_space_reduces_but_does_not_close_ambiguity
```

Not active:

```text
N2_within_trajectory_nonidentifiability
N3_source_radius_target_variance_persists
N4_source_metric_neighborhood_not_discriminative
N5_target_gauge_residual_drives_divergence
N8_empirical_selector_lower_bound_supported
N9_inconclusive_due_to_objective_availability
```
