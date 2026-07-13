# P13 Execution-Validation Amendment

Status: `EXECUTION_ONLY_AMENDMENT_BEFORE_AGGREGATION`  
Preceding review: `P13_EXECUTION_VALIDATION_RED_TEAM.md`

## Reason

Slurm job `894879_4` completed and atomically persisted the accepted target-2/seed-1 unit, then began the next group and was canceled 42 seconds later to release node11 for an exact-hardware repair. Its stderr contains one cancellation line (SHA-256 `5b75487a0e7ec4c7725d3854c963f56428b30ebe1bc3326de705055f25b1d826`). Treating that line as a generic harmless warning would be incorrect; rejecting the already completed unit without checking event order would also discard valid frozen-protocol output.

## Narrow Rule

The analyzer now emits `verified_post_artifact_scheduler_handoff` only when all of the following hold:

- stderr is exactly one syntactically valid Slurm cancellation line;
- job id and node match the raw payload;
- the submission record has one matching job marked `completed_repair_then_canceled_for_scheduler_handoff`;
- that record names only the exact dataset/target/seed key and array task;
- the raw payload has `status=ok`;
- stdout names the exact raw path with 18 rows and ok status;
- timestamps satisfy `raw < stdout < cancellation stderr`.

All other nonempty unexpected stderr remains a failure. The interrupted next group contributes no accepted row from `894879`; its result must come from the resumed frozen task and pass the ordinary complete-key gate.

## Frozen-Scope Check

- P13A analyzer SHA-256: `9986d890a24483dc6b3dfcd365eb0e0acdb8709fc63a7e396e73c93bcdd7da53`
- amended analyzer SHA-256: `93e52ca3ecb30131ecf18c91b7d7009dad6f6b48a08ccffdf73b2c7a585facdf`
- analyzer/test diff SHA-256: `520762dd25d1d9949091533fc68d799597fb6ac41a984bd28d0ea6971e11ab79`
- runner/config/manifest/checkpoints/method parameters: unchanged
- endpoints/comparators/bootstrap/claim rule: unchanged
- aggregate results inspected before amendment: none

Seven P13 regression tests pass, including positive and negative scheduler-handoff cases. This amendment changes execution-artifact acceptance only and does not alter the scientific protocol.
