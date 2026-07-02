# C8a — native K1/K2 decision call-site wiring (acceptance)

The C7 decision machinery is now wired natively into the staged runner + result ABI + artifact writer,
behind an explicit flag. Code/test milestone only: **no C6 re-score, no BNCI launch**, and no change to
training/selection/audit/prediction/metrics SEMANTICS. Commit **`19a727a`**. CPU CI: job **879247**, exit
**0**, **716** tests (`test_decision_callsite` **8**, incl. an end-to-end real-decision write→deep-verify).

## K1

- **call-site** — `finalize_level_run` (runner/finalize.py), AFTER selection lock + source-audit feature
  extraction + the source-audit leakage audit; K1 cannot run before the lock (finalize is post-AUDIT).
- **split** — `source_audit` only. K1 reuses the SELECTED per-method source-audit features retained by
  `run_post_selection_audit` (`audit_feature_items`, deduped by model hash), so there is **no re-forward**.
- **target access** — **false**. K1 receives only the ERM/OACI source-audit features + the fixed AuditScope
  (support graph / fold plan / probe config); the test `test_k1_uses_source_audit_population_not_target`
  asserts the K1 population hash equals the source-audit design population.
- **permutation plan** — deterministic per the manifest (seed 707, `paired_swap_within_y_recording_group`);
  the plan hash binds seed + n_permutations + strata + bits and is stored in `k1.json`.
- **degenerate** — when OACI selected the SAME checkpoint as ERM, the features are identical, so K1 is a
  recorded zero (`observed_delta=0`, `estimable_degenerate_same_checkpoint`, `same_checkpoint=true`) with NO
  permutation pass (`test_k1_same_checkpoint_degenerate_case_is_handled`).
- **fake/synthetic result** — the fake fixture (`fake_runner_v1.yaml`, n_permutations=10) and the
  `test_decision_callsite` end-to-end run produce a valid `k1.json` (`split_role=source_audit`, a status in
  {detected, stop, degenerate, skipped}) that deep-verifies inside the artifact.

## K2

- **single-seed behavior** — a single fold/seed artifact does NOT decide reproducibility: it stores
  `abstain_insufficient_seeds` with `available_seeds=1`, `required_min_seeds=<manifest.k2.min_seeds>` (=3).
  The real K2 decision is the later multi-seed aggregation.
- **endpoints** — `worst_domain_bacc` (↑) + `worst_domain_nll` (↓), read from the target-audit
  EvaluationMetrics (`worst_domain_reference_bacc` / `worst_domain_nll`); the per-seed Δ = OACI−ERM is
  recorded for the aggregation. **thresholds** (endpoints, min_seeds, level_policy, margins) all come from
  the manifest — none is hard-coded in code.

## Artifacts

- **paths** — `levels/<level>/decisions/{k1.json, k1.npz, k2.json}`, written THROUGH the writer index
  (`level_decisions` derived from `LevelRunResult.decision`); the null lives in `k1.npz`.
- **verifier** — deep `verify_artifact_tree` verifies the decision files like any indexed file (sha256 +
  logical hash) AND recomputes `level_result_hash` from the level payload, which now includes the decision
  binding hashes (`level_result_logical_payload` reconstructs them) — so a missing/tampered decision fails.
  `verify_decisions(require=True)` additionally enforces presence + `k1_status`/`k2_status`/null.
- **legacy compatibility** — a no-decision (legacy/disabled) artifact has no `decisions/` subtree and
  verifies whole; `verify_decisions(require=False)` tolerates it.
- **hash impact** — `level_result_hash` binds the decision hashes ONLY when enabled (empty ⇒ payload
  byte-identical), so `fold_result_hash` / `artifact_pure_science_hash` change only for decision runs.
  Training / checkpoint / selection / audit-leakage / prediction / metrics hashes **do not change**
  (`test_decision_wiring_does_not_change_training_or_selection_hashes`; staged bit-identity + fake-artifact
  + scientific-hash + runner-artifacts all remained green with decisions off).

## Enable signal

`DecisionContext(enabled=…)` threads the flag (NOT via `execution_config_hash`, so enabling never changes
the training/audit identity). `staged_demo phase-b --compute-decisions` builds it from the manifest K1/K2
specs + the leakage-parallel config; set ONLY for C8. Existing smoke / C6 / legacy runs+tests pay nothing.

## C6 / C7

**No C6 re-score.** The C7 decision machinery (`leakage/permutation`, `decision/*`, `artifacts/
decision_codec`) is unchanged — C8a only adds the runner/finalize/writer wiring + the result ABI/hash
binding around it.

## Next

**C8b** — print the multi-seed dry-run job graph (9 targets × seeds [0,1,2] = 27 Phase-A + 27 Phase-B + 1
aggregation) and review it BEFORE launching the BNCI2014-001 minimum-seed K1/K2 run.
