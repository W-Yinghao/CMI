# C78 Authorized P1 Pre-Execution Red-Team

Final status: `PASS_FOR_LOCKED_82_UNIT_SUBMISSION`

## Authorization and provenance

- Exact CLI token digest matches `authorization.exact_token` in the committed C78 protocol.
- Generic authorization prose and whitespace variants remain rejected.
- The C78 protocol anchor predates both authorization and real execution.
- The authorized execution lock binds the protocol hash, no-auth result commit, worker/script hashes, 82-unit scope, physical process order, and forbidden expansions.

## Scope adversary

```text
dataset:             BNCI2014_001 only
target:              4 only
seed:                3 only
levels:              0 and 1
ERM anchors:         2
OACI checkpoints:    80
SRC:                 0
full seed-3 field:   forbidden
seed 4:              forbidden
BNCI2014_004:        forbidden
```

The worker resolves units from the committed 82-row manifest. It does not consume the protocol's future 1,458-unit execution matrix.

## Target-isolation adversary

- The training command checks authorization and implementation hashes before importing the EEG loader or training engine.
- The training loader receives exactly subjects `[1,2,3,7,8,9]`; target 4 and source-audit subjects 5/6 are absent.
- Target/source-audit provisioning is gated on a self-hashed `FIELD_FROZEN.json` containing all 82 checkpoint and sidecar hashes.
- Target-unlabeled inference receives a primary NPZ descriptor with no target labels and no label/oracle path.
- Construction, evaluation, and same-label-oracle views are post-freeze and physically separate.

## Historical-path and retention adversary

- Current ERM, OACI, training-engine, and manifest blobs replay byte-exact against the historical identities.
- The worker calls the historical `run_stage1_once`, `make_objective("OACI")`, and `train_stage2` paths.
- Retention is one ERM final anchor plus OACI epochs `4,9,...,199` per level.
- Optimizer state is captured passively by wrapping instantiated optimizers; the historical engine file is unchanged.
- Interrupted/requeued attempts are append-only and cannot inspect target outcomes.

## Dry-run evidence

Slurm job `892829`: `10 passed, 3 expected skips in 8.02s`, stderr empty. The skips are the not-yet-created frozen-field gate, instrumentation-complete gate, and authorized final report.

## Remaining runtime gates

GPU ABI/determinism, real source-only loader identity, exact 82-checkpoint cadence/genealogy, optimizer snapshot round trips, post-freeze view hashes, all-row `W·z+b=logits`, repeat inference, measured runtime/storage, and zero target-label visibility remain blocking runtime gates. Any failure stops the campaign before a positive report.
