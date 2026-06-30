# CIGL_34 — Reviewer Risk Register

> Phase 4A consolidation (docs only). Anticipated reviewer objections and concise, evidence-backed,
> non-overclaiming responses. Each response points to the tracked evidence in `CIGL_33`.

| # | Likely objection | Response (bounded, evidence-backed) |
|---|---|---|
| 1 | "This is not real CMI." | Correct — and we never claim it is. `R_g`/`R_n` and the audit are a **posterior-KL plug-in proxy** for label-conditional domain leakage `E KL(q(D\|·,Y) ‖ π_y(D))`, judged against a **retrained within-label permutation null**. We say "leakage proxy / audit," not "unbiased CMI estimator" (CIGL_32). |
| 2 | "Why no edge-CMI?" | The dynamic-edge backbones (`shallow_graph_stem`, `eegnet_graph_stem`) **overfit** (train ≈ 1.0, source ≈ chance) and were rejected (Phase 3A-G). The task-capable backbone (DGCNN) has a **static/shared** adjacency → **no per-sample edge object** (`edge_logits=None`); the edge audit is explicitly skipped, never faked. Edge-CMI is out of scope here. |
| 3 | "Why DGCNN only?" | GraphCMINet under-fits the task (Phase 3A-R: src ≈ 0.33 < 0.45) and cannot host a meaningful task-vs-leakage tradeoff. A known-good-decoder sanity check (Phase 3A-S) showed the protocol is learnable; the graph-backbone redesign (Phase 3A-G) found the **static DGCNN adapter is the only graph-compatible backbone that learns the task** (no CNN bypass, verified). |
| 4 | "Is this SOTA?" | No. The claim is **leakage reduction at task retention**, not leaderboard accuracy. Baselines are modest (2a ERM ≈ 0.46; 2015 ERM ≈ 0.70). We make no SOTA/benchmark claim (CIGL_32 "not claimed"). |
| 5 | "Did you use target labels?" | No for training/selection. Target labels are **evaluation-only**: `used_target_labels_for_training=false`, `used_target_labels_for_selection=false`, `used_target_covariates=false`, `selection_uses_target_eval=false`, `confirmation_label_selection_uses_target_eval=false`. The target enters only a final **reported guardrail** (drop ≤ 0.05), and a unit test asserts that corrupting target labels changes neither source metrics nor the source-only confirmation. |
| 6 | "Why λ = 0.010?" | Selected once on the **BNCI2014_001 fold-0 pilot** (Phase 3A-I, the only place λ was searched, over a small ladder 0.001–0.010), then **frozen** for the BNCI2014_001 confirmation folds (3A-J) and BNCI2015_001 (3A-K). Fold-0 is reported as a development fold and excluded from the 2a primary aggregate. |
| 7 | "Why no large λ-grid?" | To avoid post-hoc tuning / selection bias: the confirmation is a **fixed-candidate** replication, not a hyperparameter search. Only the pilot ladder existed; everything after is the single frozen config. |
| 8 | "Does reduced leakage still clear the null?" | **Yes** — this is **partial controllability**, stated up front. Across all confirmation folds the regularized `graph_z`/`node_z` still clear the null (~40–65% reduction). We claim **reduction**, not **removal**. |
| 9 | "Does this generalize beyond MI?" | Not claimed. Evidence is two **motor-imagery** datasets; no claim for other paradigms (SEED/DEAP/etc.). |
| 10 | "Why are two datasets enough?" | Enough for a **bounded** cross-dataset method signal (different subjects, channels, class counts, chance levels: 2a 4-class/0.25, 2015 binary/0.50), **not** a universal claim. The blueprint (CIGL_35) frames a first paper at exactly this scope; a third dataset / ablation is an optional next step, not a precondition for the bounded claim. |

## Additional defensive notes

- **Selection firewall is testable, not just asserted:** target-label-corruption tests in
  `tests/test_phase3a_dgcnn_gn_*` show source metrics, leakage reductions, and the source-only verdict are
  invariant to target corruption; only the reported guardrail can move.
- **Negative results are part of the contribution** (CIGL_35 §6): GraphCMINet failure, dynamic-edge
  overfitting, and edge-CMI being unsupported are honestly reported, not hidden.
- **Data provenance:** all data from the read-only datalake; BNCI2015_001 via a readable symlink mirror
  (symlinks only, no download/copy), documented in CIGL_30/CIGL_36.
- **Estimator honesty:** the proxy can over- or under-state true CMI; we therefore frame the metric as a
  leakage *audit/proxy* and never invert it into an information-theoretic guarantee.
