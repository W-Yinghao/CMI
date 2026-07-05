# ACAR V5 — Stage-1B Real Substrate Build Result (run `acar-v5-stage1b-c4412b4-r1`)

```
STAGE1B_BUILT / ADMITTED
run_id = acar-v5-stage1b-c4412b4-r1
implementation_base_sha = c4412b40cb8218ed39c586ff2a4e48247648aa07
registry_sha256 = 2bbe55f4cdb4f1a18cee3b2c9e7583dba9fe9e84b9c563fb37781e98ebcbb76d
n_registered = 30
admit_run = ADMITTED
feature_dump_schema = ACAR_V5_STAGE1B_FEAT_DUMP_V5
Stage-2 not started
external/lockbox not touched
```

**This is a NOTES record only.** The package artifacts themselves — `registry.json`, `FINALIZED.json`, the 30 feature
dumps, encoder checkpoints, source-state files, and SLURM logs — live on the compute cluster under
`output_root` (below) and are **NOT committed to the repo**. This note records the outcome and its independent
verification so the result is provenance-tracked without importing any package bytes into git.

## 1. Run identity

| Field | Value |
|---|---|
| status | `STAGE1B_BUILT` |
| run_id | `acar-v5-stage1b-c4412b4-r1` |
| implementation_base_sha | `c4412b40cb8218ed39c586ff2a4e48247648aa07` |
| protocol_tag / target_sha | `acar-v5-protocol` / `4278435975a72b1127803dd2cffab420c083e430` |
| device_kind | cpu (single-threaded EEGNet backend; `torch_threads=1`) |
| SLURM job | `881227` (nodecpu09), partition CPU |
| runtime | 2026-07-04 14:39:24 → 2026-07-05 18:29:38 (RunTime 1d 03h50m), `ExitCode 0:0`, `Restarts 0` |
| env | acar-v4-regen (py3.13, torch 2.6.0, mne 1.12.1) |
| output_root | `/projects/EEG-foundation-model/yinghao/acar_v5_stage1b_out` |
| repair_staging_root | `/projects/EEG-foundation-model/yinghao/acar_v5_stage1b_repair/acar-v5-stage1b-c4412b4-r1` |

## 2. Independent verification (re-run after completion, not trusting the launch driver alone)

- `admit_run(output_root, run_id)` **→ ADMITTED** (re-invoked independently; returned a valid `SubstrateRegistry`).
- `registry.json`: **30 entries** (`n_refs = 30`); recomputed `sha256(registry.json)` =
  `2bbe55f4cdb4f1a18cee3b2c9e7583dba9fe9e84b9c563fb37781e98ebcbb76d` — **matches** the launch report.
- `FINALIZED.json` **binds**: `registry_sha256` (same as above) + `n_refs=30` + `n_registered=30` +
  `git_commit=c4412b40cb8218ed39c586ff2a4e48247648aa07` + `env_lock_sha256=b5852f1b5bff7782bca628e4d17c4993e55c09d55828c6c7e7ff484d1edf3ab1` + `status=FINALIZED`.
- All 30 `feat_dump.npz`: schema = **`ACAR_V5_STAGE1B_FEAT_DUMP_V5`**, embedding_dim **256**, **label-free** (no exact
  forbidden field name; per-window rows carry `subject_key` / `split_role` / `window_id` / `embedding` plus label-free
  read/repair/montage provenance).
- Launch hygiene: `run_root` fresh at launch = **true**; `repair_staging_root` empty at launch = **true**; staging
  **cleaned (empty) after success**.

## 3. Per-ref completion — 30 / 30, no failed refs

| Disease | Folds × seeds | Count |
|---|---|---|
| PD | fold0–4 × {20260711, 20260712, 20260713} | 15 / 15 |
| SCZ | fold0–4 × {20260711, 20260712, 20260713} | 15 / 15 |

## 4. Repair / completion census (matches the reviewed Stage-1B10..1B15 policies)

- **PD — 230 subjects**: montage-completion Pz ×149 (ds004584). No BrainVision read-repair, no channel-name repair.
- **SCZ — 226 subjects**:
  - Broken-pointer rewrite **rescued `ds004000/sub-042`** → **226** admitted subjects (vs the v4-lineage 225 where it
    was excluded) — direct payoff of the Stage-1B12 pointer-rewrite repair.
  - Marker-synthesis + ordinal channel-name rename: ds003944 ×82 + ds003947 ×61 = **143 recordings** (rename subtypes:
    `pure_eeg_ordinal` 78 / `type_prefixed_ordinal` 65).
  - Montage-completion: F3/F4/P3/P4 ×43 (ds004000) + F7 ×15 (ds004367).

## 5. Compute profile

PD refs ~16 min each; SCZ refs ~1h32m steady-state (first ~2h15m, then faster as the OS page-cache warmed on
re-reads); finalize barrier ~43 min. Total ~27h50m — used ~29 % of the 4-day CPU walltime; never at walltime risk.
No requeue, no error.

## 6. Explicitly NOT in this commit / NOT touched

- **Not committed** (notes only): `registry.json`, `FINALIZED.json`, feature dumps, checkpoints, source-state files,
  SLURM logs, any package artifact.
- **Not touched by Stage-1B**: DEV label *selection*, Stage-2 candidate selection, S1/S2/S3 robustness, any
  external/held-out cohort, ASZED, lockbox. (Label VALUES were read only inside the FIT firewall during substrate
  training, per the reviewed label firewall; the feature dumps are label-free.)

## 7. Provenance & next gates

- Package is bound to `implementation_base_sha = c4412b40cb8218ed39c586ff2a4e48247648aa07` (also in `FINALIZED.json`).
  Superseded real-run authorizations `3fe8852` and `0ab40ec` remain superseded.
- The admitted Stage-1B package is the Stage-1B result. **Stage-2 (DEV candidate selection) is NOT started** and
  requires separate authorization.
- Efficiency finding for FUTURE builds only (never retrofit this run):
  `notes/ACAR_V5_STAGE1B_FUTURE_PREPROCESS_WINDOW_CACHE_DESIGN.md` (Lever-1 per-subject dedup cache — design only).
