# SLURM operational hygiene

The interactive shell (and any `Bash`-tool command) runs on a **compute node** (e.g. `nodecpu04`) that also
hosts this project's SLURM jobs. So `ps -u <me>` lists the **live worker processes of running SLURM jobs**, not
just login-node processes.

**Rule:**
- **Never** use `ps` + `kill`/`kill -9` to clean up "suspected orphan workers" on a shared SLURM node. You
  cannot distinguish an orphan from a live job worker this way, and killing them takes the whole job down.
- To stop a job, use `scancel <jobid>` for the **specific** job id only.
- To inspect a job, use `squeue -j <id>` / `sacct -j <id>`.
- If unsure whether a process belongs to a live job, **do not kill it**.
- To kill a background *Bash-tool* process you started, use its exact PID from the launch and verify the full
  cmdline is that exact command before killing — never pattern-match `python`.

**Incident (2026-07-05):** the R1-hardened job `883257` was accidentally killed when its live workers on
`nodecpu04` were misread as orphaned schema-test processes and `kill -9`'d. No evidence was corrupted (R1 only
reads saved `.audit.npz`); the run was resubmitted chunked and completed cleanly. This note exists so it does
not recur.
