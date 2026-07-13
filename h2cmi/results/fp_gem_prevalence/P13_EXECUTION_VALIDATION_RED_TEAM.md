# P13 Execution-Validation Red Team

Review time: `2026-07-13T23:52:24+02:00`  
Review state: GPU execution still active; `141/162` raw units existed and no P13 aggregate had been computed or inspected.

## Scope

This red team reviews only the proposed validation treatment of the nonempty stderr from Slurm job `894879_4`. It does not reopen the frozen method, q grid, endpoints, comparator set, bootstrap, runner, config, manifest, checkpoints, or source models.

## Findings

1. **A post-freeze analyzer change can create a result-dependent-analysis path.** The P13A analyzer SHA-256 was `9986d890a24483dc6b3dfcd365eb0e0acdb8709fc63a7e396e73c93bcdd7da53`. The proposed analyzer SHA-256 is `93e52ca3ecb30131ecf18c91b7d7009dad6f6b48a08ccffdf73b2c7a585facdf`. Diff review found changes only in stderr/provenance validation and its regression test; result flattening, endpoint definitions, method comparisons, seed averaging, 10,000-replicate bootstrap, bootstrap seed, and claim rule are unchanged. The amendment was made before aggregation and while 21 units remained. Status: `MITIGATED_WITH_DISCLOSURE`.

2. **Calling a Slurm cancellation harmless could admit an incomplete unit.** The new status is deliberately not called harmless. It is accepted only as `verified_post_artifact_scheduler_handoff` when one exact cancellation line matches the payload job and node, the job record names only that accepted unit and exact array task, the payload is `status=ok`, stdout names that raw path with 18 rows and ok status, and filesystem times satisfy `raw < stdout < cancellation stderr`. Any failed check remains `real_or_unexpected_failure`. Status: `MITIGATED_FAIL_CLOSED`.

3. **The canceled process had begun the next group.** Stdout shows target 2/seed 1 completed atomically, followed by `unit_begin group=V100 group_index=10`; cancellation occurred before a second accepted raw unit from that process. The resumed frozen task must supply the group-index-10 unit. Final audit must prove complete unique keys and disclose zero accepted partial rows from the interrupted next group. Status: `OPEN_UNTIL_FINAL_KEY_AUDIT`.

4. **The external submission record is mutable.** The exception depends on the exact `completed_repair_then_canceled_for_scheduler_handoff` record and accepted key `Lee2019_MI:2:1`. The final packet must preserve a checksummed snapshot of the submission record and list every retry, failed artifact, and canceled zero-result launch. Status: `OPEN_UNTIL_FINAL_PACKET`.

5. **Other warnings or cancellations must not pass through this exception.** The regex accepts exactly one Slurm cancellation line, and the record/unit/node/time checks are mandatory. Unsupported architecture, missing kernels, CUDA initialization failures, runtime errors, OOM, mixed stderr, and ordinary cancellations remain failures. Mutation tests for wrong node, wrong record status, wrong unit, failed payload, and reversed artifact time all reject. Status: `MITIGATED`.

## Verdict

`CONDITIONAL_PASS_FOR_EXECUTION_VALIDATION_ONLY`. The amendment does not authorize analysis until all result-carrying jobs are absent from `squeue`, all `162` units pass raw gates, the submission record is snapshotted, and final red-team independently recomputes the scientific endpoints.
