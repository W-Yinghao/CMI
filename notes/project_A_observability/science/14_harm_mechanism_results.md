# Step 18 — Harm-Mechanism Results

Scope: oracle/evaluation-only mechanism decomposition of the Step-13 raw per-trial predictions
(BNCI2014_001 4-class + BNCI2014_004 binary, 54 runs). **Not SOTA. No new datasets, no retraining.**
Oracle target labels are used only to decompose the identity→adapt change; nothing here is identifiable
under R0/R1 and no adaptation claim is made.

## Is TTA harm lost-correct > gained-correct? — Yes

Averaged over 54 runs:

| quantity | value |
|---|---:|
| mean lost-correct rate `P(identity✓ ∧ adapt✗)` | **0.1211** |
| mean gained-correct rate `P(identity✗ ∧ adapt✓)` | **0.0789** |
| mean net gain (`gained − lost`) | **−0.0423** |

The net accuracy change equals `gained_correct − lost_correct` exactly (a tested identity). Offline TTA
destroys more correct trials than it creates: the harm is a real prediction churn, not a scoring
artifact.

## Are class effects mixed? — Almost always

**52 / 54 runs (0.963)** have *mixed* class effects: at least one true class gains recall while another
loses recall after adaptation. TTA is not a uniform up/down shift; it redistributes recall across classes.

## Which classes lose, and which transitions dominate?

Most-common worst class (largest recall drop) by dataset:

| dataset | most-common worst class | worst-class histogram (class → #runs) |
|---|---|---|
| BNCI2014_001 (4-class) | **1** | `{0:6, 1:9, 2:3, 3:9}` |
| BNCI2014_004 (binary) | **0** | `{0:22, 1:5}` |

Dominant identity→adapt *wrong* transitions (true class → predicted), aggregated over runs:

```
0->1 ×35    1->0 ×33    1->2 ×18    3->2 ×17
0->2 ×15    2->0 ×8     2->1 ×7     3->0 ×7
```

The harm is concentrated in a few confusable class pairs (notably the 0↔1 channel), consistent with
adaptation over-aligning to the majority-confusion direction rather than failing uniformly.

## Do beneficial cells differ? — They are the mirror image

Because 96% of runs are mixed and the net is negative, "beneficial under uniform prior" cells are those
whose *positive* class deltas outweigh their negative ones. The same decomposition feeds the prior-stress
analysis ([15_prior_stress_results.md](15_prior_stress_results.md)): once class effects are mixed, the
sign of the aggregate gain depends on how the deployment prior weights the winning vs losing classes.

## Claim boundary

Every quantity uses oracle per-trial target labels. This is a mechanism decomposition, not an
identifiable target metric; `oracle_labels_used_only_for_mechanism_and_evaluation = true`; no R0/R1
claim; no SOTA.
