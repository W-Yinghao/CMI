# P13 Analyzer-Output Red Team

Review time: `2026-07-13T23:59:02+02:00`  
Review state: GPU execution still active; `142/162` raw units existed and no P13 aggregate had been computed or inspected.

## Scope

This review covers the post-freeze analyzer changes that preserve execution provenance and render the execution audit. It does not reopen the P13 runner, config, manifest, checkpoints, q values, methods, endpoints, comparator set, bootstrap, or claim rule.

## Findings

1. **Output/provenance work could conceal a statistical change.** The diff from amendment commit `a703619` was reviewed by function. It changes excluded-attempt validation, job-record consistency gates, submission-record snapshotting, execution-audit classification, and tests. It does not change `flatten`, `seed_averaged_subjects`, `bootstrap_endpoints`, `geometry_summary`, endpoint formulas, 10,000-replicate policy, bootstrap seed, or support rule. Status: `MITIGATED`.

2. **A mutable external submission record is not durable provenance.** The analyzer now validates its launch commit, runner/config/manifest hashes, expected units, clean launch, concurrency, and monitoring policy against accepted payloads and frozen constants. It writes a canonical JSON snapshot into the result packet and records both the input and snapshot checksums. Status: `MITIGATED`.

3. **Recorded failed attempts could exist only on paper.** Every `excluded_attempt` entry is now joined to its job/task directory under `excluded/`. Raw JSON, stdout, and stderr hashes must match; the raw payload must be `status=failed`, contain no result rows, match the target/seed and failure reason, and all three artifact paths are written to an excluded-artifact manifest. The six current attempts and 18 files pass this gate. Status: `MITIGATED`.

4. **Canceled pending jobs could silently contribute rows.** Every job marked `canceled_pending_zero_result` must be disjoint from the accepted unit provenance. The audit lists those launches separately from accepted result-carrying jobs. Status: `MITIGATED`.

5. **Hard-coding current retry counts would overfit execution history.** Validation compares verified excluded attempts and verified scheduler handoffs to the corresponding submission-record counts instead of hard-coding six and one. Any new event must be recorded and independently validated. Status: `MITIGATED`.

6. **The packet remains incomplete while jobs run.** Snapshot, merge, aggregation, and final claims remain forbidden until every recorded job is absent from `squeue` and all 162 unique target-seed units pass. Status: `OPEN_UNTIL_FINAL_EXECUTION_GATE`.

## Verdict

`PASS_FOR_OUTPUT_AND_PROVENANCE_AMENDMENT_ONLY`. Analyzer SHA-256 changes from `93e52ca3ecb30131ecf18c91b7d7009dad6f6b48a08ccffdf73b2c7a585facdf` to `0baca1664a8c47d50a46e07705fff538fac55f1a4bc62d74b2d088c743855bc7`; the analyzer/test diff SHA-256 is `39994a7304994c397495433b22c88c995c68e89dd4d3654314b4bda559229253`. Eight P13 regression tests pass.
