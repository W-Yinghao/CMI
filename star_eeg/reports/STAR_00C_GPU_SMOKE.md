# STAR_00C Final-Code Persistent GPU Smoke

- Authoritative Slurm job: `893028`
- State / exit: `COMPLETED`, `0:0`
- Node / GPU: node22, NVIDIA A40
- Runtime: `00:02:04`
- Cells: H200_SSL_CONT, H200_STAR_TRUE, H200_STAR_SHUFFLED
- Scope: ten optimizer steps per variant from immutable H200_s0

The smoke persisted ten flushed telemetry rows, run/execution manifests,
run-summary, atomic final checkpoint, and completion marker per cell. It then
froze each attempt tree read-only and demonstrated that reuse of the same
attempt directory hard-fails. All model/data-path checks from STAR_00B remained
true, all values were finite, all checkpoint and manifest hashes verified, and
the H200 source SHA remained unchanged.

Update accounting was B = 10 encoder / 10 reconstruction / 0 anchor and C/D =
10 encoder / 8 reconstruction / 2 anchor. This confirms update and batch-count
matching without making a strict FLOP-matching claim.

Jobs 893017 and 893023 were preliminary persistent smokes before the formal
array/single-use attempt lock was complete. Job 893028 is the authoritative
final-source record. No 3,750-step cell or target metric was run in any smoke.
