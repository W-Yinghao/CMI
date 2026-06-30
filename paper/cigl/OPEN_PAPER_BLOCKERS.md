# CIGL Open Paper Blockers (Phase 4E / v0.3)

> Candidate gaps, **classified only** — none are run here (all Phase 4x writing stages are writing-only; any experiment needs
> separate reviewer authorization). Classes: **MUST-FIX** (before draft submission) · **OPTIONAL-ABLATION**
> (strengthens, reviewer-gated) · **FUTURE-WORK** (named in Limitations/Future) · **DO-NOT-DO** (out of
> scope / would invite overclaim).

| Candidate | Class | Rationale |
|---|---|---|
| Related-work **citation verification** (replace all `TODO: verify citation`) | **MUST-FIX** | No fabricated bibliography may ship; every reference must be checked before submission. |
| **Baseline-table clarity** (state ERM ≈0.46 (2a) / ≈0.70 (2015); chance 0.25/0.50; what "task retained" means) | **MUST-FIX** | Prevents the "modest baseline" objection and any implied accuracy claim. |
| **Statistical summary across folds** (mean ± CI / per-fold counts for reduction & retention; already in JSON) | **MUST-FIX** (writing only) | Reviewers expect aggregate stats; computable from existing summary JSON, no new runs. |
| **Graph-only vs node-only at λ=0.010** (isolate which term drives reduction) | **OPTIONAL-ABLATION** | Clarifies mechanism; the pilot already ran graph_001/node_001/graph_003/node_003 (CIGL_27) — may suffice from existing data, else a small gated run. |
| **Marginal vs conditional regularizer** (show conditional avoids label erasure) | **OPTIONAL-ABLATION** | Strengthens the conditional-invariance positioning; only if the draft needs it. |
| **Node-map visualization** (electrode leakage map figure F3) | **OPTIONAL-ABLATION** (plot only) | Interpretability; uses existing audit output, no training. |
| **Third MI dataset** stress test | **FUTURE-WORK** | Two datasets support the bounded claim; a third is generalization evidence, not a precondition. |
| **Dynamic-edge constrained backbone** (low-rank / regularized `A(x)` that generalizes → enables edge audit) | **FUTURE-WORK** | A new design question; dynamic-edge currently overfits (negative result). Out of this paper's scope. |
| **Larger λ-grid / λ-robustness curve** | **DO-NOT-DO** (for this paper) | Fixed-candidate confirmation is the design; a grid would invite a "tuned/λ-robust" overclaim and post-hoc selection risk. |
| **SOTA / accuracy leaderboard table** | **DO-NOT-DO** | The claim is leakage-reduction-at-retention; an accuracy table would misframe it. |
| **Edge-CMI / dynamic-edge method results** | **DO-NOT-DO** | Unsupported (overfitting negative result); would be a false claim. |
| **Beyond-MI (SEED/DEAP) or cross-architecture claims** | **DO-NOT-DO** | No evidence; explicitly disclaimed in scope. |

## Reading

- The three **MUST-FIX** items are all **writing/analysis** tasks (citation checks, baseline clarity,
  aggregate stats from existing JSON) — **no GPU**. They can be done in manuscript polish without new runs.
- The **OPTIONAL-ABLATION** items may be partially answerable from **existing** Phase 3A-I data (the pilot
  λ-ladder already contains graph-only/node-only at small λ); a new GPU run should be requested only if the
  draft genuinely cannot stand without it (reviewer's "write first, experiment only if necessary" rule).
- **DO-NOT-DO** items are scope guards, not gaps.
