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
