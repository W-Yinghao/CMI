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

## Relaunch Startup Check - 2026-07-08T22:39:15+02:00
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
          890125_0       A40  RUNNING       2:15               node34
```

### startup evidence
```text
running_task=890125_0
cell=H200_s0
host=node34
training_git_head=c3a6265079de1645a35af482f8e2b11d7564dc7a
stderr_empty=true
trainer_entered=true
data_event_seen=true
model_event_seen=true
target_labels_used=false
checkpoint_seen=false
downstream_launched=false
```

## Scheduler Adjustment - 2026-07-08T23:12:15+02:00
reason=non-H2000 tasks allowed on A100,V100,H100,L40S,A40; H2000 remains A40-only

### actions
```text
kept_running=890125_0 H200_s0 A40
cancelled_pending=890125_1..890125_7
failed_submission_attempt=non-H2000 flexible with 96h rejected by partition time limit; no job created
submitted_non_h2000_flexible=890147_[1-5] partitions=A100,V100,H100,L40S,A40 time=1-00:00:00
submitted_h2000_a40=890151_[6-7] partition=A40 time=96:00:00
```

### squeue
```text
             JOBID         NAME PARTITION    STATE       TIME     NODELIST(REASON)
          890147_4    s2pB1flex      A100  RUNNING       1:07        nodeaudible01
          890147_2    s2pB1flex      A100  RUNNING       2:11               node04
          890147_3    s2pB1flex      A100  RUNNING       2:11               node04
          890147_1    s2pB1flex      A100  RUNNING       2:12               node03
          890147_5    s2pB1flex A100,V100  PENDING       0:00               (None)
          890151_6     s2pB1h2k       A40  PENDING       0:00               (None)
          890151_7     s2pB1h2k       A40  PENDING       0:00               (None)
          890125_0        s2pB1       A40  RUNNING      35:15               node34
```

### startup evidence
```text
890147_1 H200_s1 entered_trainer=true gpu=A100 stderr_empty=true target_labels_used=false
890147_2 H500_s0 entered_trainer=true gpu=A100 stderr_empty=true target_labels_used=false
890147_3 H500_s1 entered_trainer=true gpu=A100 stderr_empty=true target_labels_used=false
890147_4 H1000_s0 pending=true
890147_5 H1000_s1 pending=true
890151_6 H2000_s0 pending_a40=true
890151_7 H2000_s1 pending_a40=true
```

## Scheduler Adjustment Startup Update - 2026-07-08T23:12:59+02:00

### squeue
```text
             JOBID         NAME PARTITION    STATE       TIME     NODELIST(REASON)
          890147_4    s2pB1flex      A100  RUNNING       1:51        nodeaudible01
          890147_2    s2pB1flex      A100  RUNNING       2:55               node04
          890147_3    s2pB1flex      A100  RUNNING       2:55               node04
          890147_1    s2pB1flex      A100  RUNNING       2:56               node03
          890147_5    s2pB1flex A100,V100  PENDING       0:00  (QOSMaxGRESPerUser)
          890151_6     s2pB1h2k       A40  PENDING       0:00  (QOSMaxGRESPerUser)
          890151_7     s2pB1h2k       A40  PENDING       0:00  (QOSMaxGRESPerUser)
          890125_0        s2pB1       A40  RUNNING      35:59               node34
```

### updated startup evidence
```text
890147_4 H1000_s0 entered_trainer=true gpu=A100 stderr_empty=true target_labels_used=false
890147_5 H1000_s1 pending=true reason=QOSMaxGRESPerUser
890151_6 H2000_s0 pending_a40=true
890151_7 H2000_s1 pending_a40=true
```

## Partial Pretraining Completion Check - 2026-07-09T09:24:30+02:00

### squeue
```text
             JOBID         NAME PARTITION    STATE       TIME     NODELIST(REASON)
          890147_4    s2pB1flex      A100  RUNNING   10:13:22        nodeaudible01
          890151_6     s2pB1h2k       A40  RUNNING    8:08:07               node32
          890151_7     s2pB1h2k       A40  RUNNING    8:08:07               node33
          890147_5    s2pB1flex      V100  RUNNING    9:56:07               node13
```

### completed runs
```text
H200_s0 job=890125_0 epochs=50 done=true best_epoch=49 best_val=0.3162865489721298 checkpoint_reload=true nan_inf=false stderr_empty=true
H200_s1 job=890147_1 epochs=50 done=true best_epoch=50 best_val=0.32015835121273994 checkpoint_reload=true nan_inf=false stderr_empty=true
H500_s0 job=890147_2 epochs=50 done=true best_epoch=49 best_val=0.25837720558047295 checkpoint_reload=true nan_inf=false stderr_empty=true
H500_s1 job=890147_3 epochs=50 done=true best_epoch=46 best_val=0.25497831621517736 checkpoint_reload=true nan_inf=false stderr_empty=true
```

### running runs
```text
H1000_s0 job=890147_4 epochs_seen=33 best_val=0.23592040625711283 nan_inf=false
H1000_s1 job=890147_5 epochs_seen=20 best_val=0.25056258713205654 nan_inf=false
H2000_s0 job=890151_6 epochs_seen=9 best_val=0.26146836796154577 nan_inf=false
H2000_s1 job=890151_7 epochs_seen=9 best_val=0.2608661533643802 nan_inf=false
```

### policy
```text
downstream_launched=false
pretraining_gate_deferred_until_8_of_8_runs_complete=true
sacct_unavailable=true
```
