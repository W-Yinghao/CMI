# Slurm Monitoring Policy

Do not use `sacct` on this server.

Use `squeue` for job-state monitoring. A job is considered finished with respect
to the scheduler only after it is absent from `squeue`.

Use artifact-level validation for completion. Scheduler disappearance alone is
not a result gate.

Accepted completion rule:

```text
job absent from squeue + artifact parse/count/checksum validation passed
```

Required validation fields for Slurm-backed artifacts:

- job id
- submit command
- final `squeue` absence
- stderr status
- stdout status
- artifact paths
- row counts
- checksums

