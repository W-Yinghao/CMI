# ACAR v4 — Option A residual (out-of-scope) read-only encoder-checkpoint sweep **(NOT_FOUND)**

```
DATE            : 2026-06-29T22:57Z (machine UTC)
COMMIT          : ca96707 (branch acar, HEAD at search time)
SCOPE           : READ-ONLY inventory of LOCAL paths OUTSIDE the four roots already searched
                  (notes/ACAR_V4_ENCODER_CHECKPOINT_SEARCH.md). NO retrain, NO model run, NO external/held-out read,
                  NO deserialization of unknown pickles, NO new numbers, NO tag.
CONCLUSION      : NOT_FOUND — no DEV EEGNet (erm:0) encoder checkpoint exists in any out-of-scope local path reachable here.
```

## Why this step
The in-scope search (`ACAR_V4_ENCODER_CHECKPOINT_SEARCH.md`) was NOT_FOUND and left one small INCONCLUSIVE tail: SLURM
scratch / job dirs / other-project output dirs / a CUDA-node mirror were not inspected. This sweep closes that tail before
moving to Option B. Low cost, still no external/raw read, no retrain.

## Environment observed
```
SLURM_JOB_ID = 866446 ; SCRATCH / SLURM_SUBMIT_DIR / SLURM_TMPDIR / TMPDIR / LOCAL_SCRATCH / WORK / DATADIR = unset
scratch-like mounts present: /data (per-user dirs, no yinwang dir), /mnt (empty listing)
```

## Out-of-scope roots searched (by filename + log text only)
```
~/ACAR_V3_LOCKED_RUN_817b04f   ~/acar_v3_dev_run_002   ~/acar_v4_dev_exploration_001   ~/slurm_logs
~/slurm-752316.out  ~/slurm-752317.out
~/CMI_AAAI_cigl  ~/CMI_AAAI_csc  ~/CMI_AAAI_oaci  ~/CMI_AAAI_qxu  ~/CMI_AAAI_tos
~/HCL_EEG  ~/EEGPT-main  ~/denoiseNet  ~/jeanzay  ~/AAAI_2026  ~/ICML_2026
/data (top-level; checked for a yinwang-owned dir)   /mnt (top-level)
```

## NOT searched (forbidden / out of reach — recorded honestly)
```
~/mne_data                         -- possible raw EEG cache (skipped, not opened)
*/zenodo14808296/*  */ds007526/*   -- held-out external payload (forbidden)
*/heldout/*  */lockbox/*           -- forbidden
*/site-packages/*  conda envs      -- excluded (torch ships .pt test fixtures = noise; a real ckpt here is implausible)
/data/<other-users>                -- other people's dirs (not ours)
other machines / devices           -- e.g. the CUDA node that minted some B0 hashes (see [[h2cmi-b0-checkpoint-device]]);
                                      unreachable from this host
```

## Findings
- **`.pt/.pth/.ckpt/.pkl/.joblib` in the ACAR run dirs / siblings / jeanzay = NONE.** `ACAR_V3_LOCKED_RUN_817b04f`,
  `acar_v3_dev_run_002`, `acar_v4_dev_exploration_001`, `slurm_logs`, all `CMI_AAAI_*` siblings, `jeanzay` → zero weights.
- **The only weight files on the box belong to UNRELATED projects**, none an EEGNet erm:0 encoder for the ACAR PD/SCZ
  cohorts: `HCL_EEG/...` (SSL: ResNet-SEED / ContraWR / barlowtwins / simclr / simsiam — emotion/SEED, wrong arch+data),
  `ICML_2026/eegspdnet/results/checkpoint_*.pth` (SPDNet), `AAAI_2026/.../loadbearing_ablation_results.pkl` (results pickle).
- **No `/data/yinwang`** dir; `/mnt` empty.
- **ACAR run logs / manifests reference no saved-encoder path.** Matches were only: `.gitignore` `*.pt`/`*.pth` rules; doc
  text about epoch "checkpoint" counts / `source_checkpoint_hash` (h2cmi) / "re-embed with the FROZEN checkpoint" (a script
  describing intent); `h2cmi/run_shift_grid.py` `state_dict` (the h2cmi simulator, not the DEV EEGNet). No actual file.

## Conclusion
```
FOUND_CANDIDATE : NO
NOT_FOUND       : YES — confirmed across both in-scope and out-of-scope local paths reachable from this host.
INCONCLUSIVE    : only for genuinely out-of-reach locations (other machines/devices; per-user /data of others; raw caches
                  deliberately not opened). These are not pursuable read-only from here.
DECISION        : Option A (recover original) is FORECLOSED in practice. Proceed to Option B DESIGN ONLY
                  (notes/ACAR_V4_SUBSTRATE_REGEN_PLAN.md) — NO retrain until that plan is reviewed/approved. Do NOT pick C yet.
```
This sweep does NOT change `ACAR_FROZEN_v4.md` executable status (still NOT_YET_EXECUTABLE; both blockers open). No tag, no
external read, lockbox SEALED.
