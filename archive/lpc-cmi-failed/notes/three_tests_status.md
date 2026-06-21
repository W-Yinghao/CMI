# Three dual-CMI tests — setup & submission status

Snapshot: 2026-06-11 ~18:35. Queue read from `squeue -u yinwang`. NOTE: SLURM
accounting (`sacct`) DB was down, so historical job state / submit-lines could not be
retrieved; pending-job argv is not surfaced by `scontrol` on this cluster, so for the
pending jobs the dataset/seed mapping is inferred from the route note + job names, not
read back from the scheduler.

| Test | What was built | Job ids / status |
|---|---|---|
| **(1) Route-B reweighted-dual, multi-seed ADFTD** | `--reweight_dual` flag honoured for `method=dual`; GLS weight `w_i=pi*(y)/pi_d(y)` applied to BOTH CMI estimators (encoder KL + decoder CE) and the Step-A posterior fit. Touched `cmi/methods/regularizers.py`, `cmi/train/trainer.py`, `cmi/run_loso.py`, `cmi/run_scps_crossdataset.py`. CPU-smoke-tested, backward-compatible (naive `dual` bit-identical when flag off). | **SUBMITTED & RUNNING.** 848130 RUNNING (confirmed from .out: `rwdual_ADFTD_seed1`, `dual:0.3:0.3 --reweight_dual --seed 1`, ~9 LOSO folds in). 848131 RUNNING; 848132, 848134 PENDING (Priority). 4 `cmi-loso` A40 jobs total → consistent with the seed1–4 ADFTD rwdual sweep. No `rwdual_ADFTD_seed*.json` yet (jobs still running). Single-shot `rwdual_ADFTD.json` (16:36) and `rwdual_MUMTAZ.json` (14:32) already on disk. |
| **(2) Route-A GLS-VAE, ADFTD + MUMTAZ** | `cmi/run_glsvae.py` EXISTS (17072 B, mtime Jun 11 17:02). DIVA-style partitioned latent (z_y/z_d), GLS reference-prior decode, `delta_d` concept-shift test with domain-permutation null (`--n_perm`). argparse supports `--dataset {ADFTD,ADFTD_bin,MUMTAZ,TUAB}`. Synthetic precursor (`synthetic/gls_vae.py`) already validated; write-up in `notes/route_A_gls_vae.md`. | **SUBMITTED (pending).** 848142 `glsvae-ADFTD` PENDING, 848143 `glsvae-MUMTAZ` PENDING (both A40, Reason=Priority). Logs empty (not started). No `glsvae_*.json` yet. |
| **(3) SCZ resting-FEP loader fix + cache rebuild + ladder resubmit** | Loader fix IS in `cmi/data/bids_data.py` (mtime Jun 11 17:02): COHORTS registry now includes the resting SCZ cohorts ds003944 (task=`Rest`) and ds003947 (task=`rest`) alongside ds004000/ds004367. Cache builder `scripts/build_scps_cache.py` + `scripts/build_scps_cache.slurm` present. | **CACHE REBUILD DONE; LADDER RESUBMIT MISSING.** scz-cache job 848159 COMPLETED: `SCZ.npz` rebuilt 18:35 (324 MB) with all 4 cohorts `[ds003944, ds003947, ds004000, ds004367]`, X=(9000,19,512), y=[4360,4640]. **But NO ladder job was resubmitted** — no `scz_resting_ladder` job in queue, no `scz_resting_ladder.json`, and no ladder script / `scz_resting_ladder` reference anywhere in the repo. |

## Honest gaps / caveats

- **Test 3 is incomplete.** The loader fix landed and the 4-cohort cache rebuilt
  cleanly, but the downstream ladder run was **not** resubmitted. The only SCZ
  cross-dataset results on disk (`dual_scps_SCZ.json` 12:32, `lc_scps_SCZ.json` 12:31)
  predate the fix and used only the 2 task-based cohorts (`cohorts:[ds004000,
  ds004367]`) — i.e. they ran on the buggy/partial loader, missing the two resting
  cohorts. They need to be re-run against the new cache. The expected output
  `results/scz_resting_ladder.json` does not exist and nothing is queued to produce it.
  ACTION NEEDED: submit the SCZ ladder (e.g. via `cmi.run_scps_crossdataset --condition
  SCZ ...`) now that `SCZ.npz` contains all 4 cohorts.
- **Test 1 seed coverage is inferred, not verified.** seed1 confirmed running from its
  .out; seeds 2–4 inferred from the 4 pending/running `cmi-loso` jobs + the route plan.
  `sacct`/`scontrol` could not return the pending jobs' argv to confirm seed numbers.
- **Tests 1 & 2 GPU jobs are pending behind Priority** on A40 — submitted, not yet
  producing results. No failures observed.
