# Bounded CodeBrain Preflight Launch Hold

```text
time: 2026-07-11T16:40:04+02:00
execution_location_recorded: false
git_commit: 11fc4da8225038c0fc2a8774293c838796418893
entry: s2p/slurm/codebrain_bounded_preflight.sbatch
job_submitted: false
training_submitted: false
fine_tuning_submitted: false
```

The submit-time controller check flapped. `sinfo` first returned an available accelerator partition, but the
immediately following `squeue` failed with:

```text
getaddrinfo() failed: Name or service not known
Unable to resolve the configured SLURM controller
Unable to establish control machine address
slurm_load_jobs error: Resource temporarily unavailable
```

An explicit `scontrol ping` then reported:

```text
Slurmctld(primary) is DOWN
```

Disposition: fail closed before `sbatch`. The metadata dry-run used only `/tmp` and is not promoted as the SLURM
preflight package. The frozen preflight must be submitted from the pinned commit after scheduler stability is
re-established. No Stage-2 training or downstream job is authorized or running from this attempt.

A later recheck on 2026-07-11, after the ISRUC local-asset audit, again returned a controller DNS-resolution
failure and `Slurmctld(primary) is DOWN`. No stability gate or submission was attempted after that failure.
