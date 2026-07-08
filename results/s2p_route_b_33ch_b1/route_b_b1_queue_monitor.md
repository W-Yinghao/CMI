## 2026-07-08T20:20:34+02:00

git_head=b2aed019e262c1e2cb293207e28d5033fabaa5a8
job_id=889864

### squeue
```text
             JOBID PARTITION    STATE       TIME     NODELIST(REASON)
      889864_[0-7]       A40  PENDING       0:00  (QOSMaxGRESPerUser)
```

### sacct
```text
```

## 2026-07-08T20:21:04+02:00

git_head=b2aed019e262c1e2cb293207e28d5033fabaa5a8
job_id=889864

### squeue compact
```text
             JOBID PARTITION    STATE       TIME     NODELIST(REASON)
      889864_[0-7]       A40  PENDING       0:00  (QOSMaxGRESPerUser)
```

### squeue expanded
```text
             JOBID PARTITION    STATE       TIME     NODELIST(REASON)
          889864_0       A40  PENDING       0:00  (QOSMaxGRESPerUser)
          889864_1       A40  PENDING       0:00  (QOSMaxGRESPerUser)
          889864_2       A40  PENDING       0:00  (QOSMaxGRESPerUser)
          889864_3       A40  PENDING       0:00  (QOSMaxGRESPerUser)
          889864_4       A40  PENDING       0:00  (QOSMaxGRESPerUser)
          889864_5       A40  PENDING       0:00  (QOSMaxGRESPerUser)
          889864_6       A40  PENDING       0:00  (QOSMaxGRESPerUser)
          889864_7       A40  PENDING       0:00  (QOSMaxGRESPerUser)
```

### sacct
```text
sacct: error: slurm_persist_conn_open_without_init: failed to open persistent connection to host:localhost:6819: Connection refused
sacct: error: Sending PersistInit msg: Connection refused
sacct: error: Problem talking to the database: Connection refused
```

### summary
```text
pending_reason=QOSMaxGRESPerUser
started_tasks=0
completed_tasks=0
failed_tasks=0
note=sacct accounting query currently unavailable; squeue monitoring remains interpretable
```


## Monitor Snapshot - 2026-07-08T20:22:15+02:00
git_head=b2aed019e262c1e2cb293207e28d5033fabaa5a8

### squeue compact
             JOBID PARTITION    STATE       TIME     NODELIST(REASON)
      889864_[0-7]       A40  PENDING       0:00  (QOSMaxGRESPerUser)

### squeue expanded
          889864_0       A40  PENDING       0:00  (QOSMaxGRESPerUser)
          889864_1       A40  PENDING       0:00  (QOSMaxGRESPerUser)
          889864_2       A40  PENDING       0:00  (QOSMaxGRESPerUser)
          889864_3       A40  PENDING       0:00  (QOSMaxGRESPerUser)
          889864_4       A40  PENDING       0:00  (QOSMaxGRESPerUser)
          889864_5       A40  PENDING       0:00  (QOSMaxGRESPerUser)
          889864_6       A40  PENDING       0:00  (QOSMaxGRESPerUser)
          889864_7       A40  PENDING       0:00  (QOSMaxGRESPerUser)
squeue_expanded_rc=0

### sacct
sacct: error: slurm_persist_conn_open_without_init: failed to open persistent connection to host:localhost:6819: Connection refused
sacct: error: Sending PersistInit msg: Connection refused
sacct: error: Problem talking to the database: Connection refused
sacct_rc=1

### summary
pending_reason=(QOSMaxGRESPerUser)
started_tasks=0
completed_tasks=0
failed_tasks=0
pending_tasks=8
note=sacct accounting query unavailable; squeue remains the authoritative live monitor for this snapshot

## Disappearance Diagnosis - 2026-07-08T22:23:19+02:00

### live SLURM query
```text
squeue -j 889864: slurm_load_jobs error: Invalid job id specified
scontrol show job 889864: slurm_load_jobs error: Invalid job id specified
sacct -j 889864: accounting DB connection refused
```

### local task logs
```text
logs_created=results/s2p_route_b_33ch_b1/logs/train-889864_{0..7}.{out,err}
stderr_all_tasks=mkdir: cannot create directory '/var/spool/results': Permission denied
stdout_all_tasks=empty
```

### interpretation
```text
The array disappeared from squeue because the tasks left the live queue after immediate batch-wrapper failure.
No CBraMod training step started; the failure happened before trainer stdout, conda/python output, checkpointing, or model logs.
Root cause: the sbatch wrapper derives REPO_ROOT from BASH_SOURCE[0]. SLURM executes a copied script from its spool area, so BASH_SOURCE[0] resolved under /var/spool, making ROOT=/var/spool/results/s2p_route_b_33ch_b1. mkdir then failed with Permission denied.
Stop-rule status: wrong runtime path / wrapper startup failure; training is not complete and downstream remains held.
Actions taken: inspected only; no job submitted, no requeue, no partition/QoS change, no downstream launch.
```

## Relaunch Monitor Snapshot - 2026-07-08T22:37:37+02:00
git_head=c3a6265079de1645a35af482f8e2b11d7564dc7a
previous_job=889864_[0-7] FAILED_AT_WRAPPER_STARTUP
path_canary_job=890114_[0-7] PASS
relaunch_job=890125_[0-7]

### squeue expanded
```text
             JOBID PARTITION    STATE       TIME     NODELIST(REASON)
          890125_1       A40  PENDING       0:00  (QOSMaxGRESPerUser)
          890125_2       A40  PENDING       0:00  (QOSMaxGRESPerUser)
          890125_3       A40  PENDING       0:00  (QOSMaxGRESPerUser)
          890125_4       A40  PENDING       0:00  (QOSMaxGRESPerUser)
          890125_5       A40  PENDING       0:00  (QOSMaxGRESPerUser)
          890125_6       A40  PENDING       0:00  (QOSMaxGRESPerUser)
          890125_7       A40  PENDING       0:00  (QOSMaxGRESPerUser)
          890125_0       A40  RUNNING       0:37               node34
```

### summary
```text
running_tasks=1
pending_tasks=7
pending_reason=QOSMaxGRESPerUser
downstream_launched=false
```
