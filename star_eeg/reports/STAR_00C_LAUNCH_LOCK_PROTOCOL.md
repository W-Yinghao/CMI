# STAR_00C Launch Lock and Persistent Provenance Protocol

STAR_00C is a launch-hardening patch on the accepted `d63bd3b` STAR_00B
baseline. It cannot alter the six-cell matrix, 3,750-step schedule, optimizer,
learning rate, batch size, anchor ratio, labels, layer scope, task head, or any
scientific gate.

Each formal cell writes to a fresh `<cell>/attempt_NN/` directory. The runner
atomically claims the attempt, appends and flushes one canonical JSON telemetry
row per optimizer step, publishes checkpoints through a no-overwrite atomic
link, persists run and execution manifests plus a final summary, and creates
`completion.json` only after all integrity checks pass. It then removes all
write bits from the completed attempt. A retry must use a new attempt ID.

Formal execution requires a regular read-only
`star01a_approval.<canonical-hash>.json` outside the Git tree. The lock binds the
clean STAR execution commit and branch, exact six cells, 3,750/final-step
contract, target-scoring block, immutable/data/compute artifacts, runner and
persistence sources, both Slurm sources, submitter, and final closure source.
A one-field `APPROVED` JSON cannot unlock training.

The formal runner additionally requires Slurm array geometry `0-5` with count
six, binds each array task ID to its one approved cell, and requires the runtime
attempt ID to equal `STAR_ATTEMPT_ID`. A valid approval file therefore cannot be
used to execute a single cell outside the blind array.

The current PM gate binds a single `attempt_01` array launch. An infrastructure
retry must use `attempt_02` and a newly issued approval lock; it may never
overwrite or silently reuse the first attempt.

All six cells are submitted together as one array. A CPU closure job is
submitted with `afterok:<array-job-id>`. It accepts only six valid immutable
completed attempts, revalidates telemetry and final checkpoints, copies each
step-3750 payload to a SHA-named read-only path, and produces the final manifest,
completion matrix, and go/no-go record. It does not score target data.

B/C/D are optimizer-update-, batch-count-, and scheduler-step-matched, not
strict FLOP-matched. C-D is the clean task-semantics contrast. C-B compares
task-anchor allocation with extra SSL continuation. Runtime accounting must
report wall time, encoder updates, full-reconstruction updates, and anchor
updates.

The post-closure source-val task gate controls whether L4/L5/L6 may be
interpreted for a cell. It cannot remove a completed cell from the future
single all-cell target scoring job. Only an integrity or firewall failure may
block that later stage. Target scoring remains blocked throughout STAR_00C and
STAR_01A training.
