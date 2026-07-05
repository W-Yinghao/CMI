# CIGL_57 — Official braindecode baseline preflight (non-GPU)

For reviewer-facing validation: can we run the OFFICIAL braindecode EEGNet/ShallowConvNet/Deep4Net (not our
minimal reimplementations) through the same run_loso pipeline? Preflight only — **NOT running the 189-job
official-baseline GPU now** (the Mini sidecar already sufficed to drop the SOTA claim). Probe result:

| env | torch | braindecode | moabb | mne | torchaudio | official EEGNetv4 builds | cmi imports |
|---|---|---|---|---|---|---|---|
| **acar-v4-regen** | 2.6.0+cu124 | **1.5.2** | 1.5.0 | 1.12.1 | 2.6.0 | **YES → (2,4)** | **YES** |
| icml | 2.8.0+cu128 | 0.8 (old) | 1.2.0 | 1.8.0 | MISSING | YES → (2,4) | not tested |

## Verdict

**`acar-v4-regen` is a fully-capable env for official braindecode baselines.** braindecode 1.5.2 + official
`EEGNetv4`/`ShallowFBCSPNet`/`Deep4Net` build and forward; moabb 1.5.0 + mne 1.12.1 + torchaudio present; **and
the `cmi` package imports there** → official nets can run through the SAME `run_loso` pipeline via the existing
`HookedBackbone` path (`--backbone EEGNet/ShallowConvNet/Deep4Net`, which lazily imports braindecode). `icml`
has braindecode 0.8 (older) + no torchaudio — usable but less clean; prefer `acar-v4-regen`.

## Remaining step before an official run (when PI approves)

1. Tiny env smoke IN acar-v4-regen: `build_backbone("EEGNet"/"ShallowConvNet"/"Deep4Net")` via HookedBackbone
   + a 2-epoch/3-subject run on 2a & 2015 (Deep4Net needs n_times≥~450 → `--resample 250`; the mini sidecar
   used 128 — note the preprocessing difference for a like-for-like table).
2. Confirm numpy 2.x / torch 2.6 give the same LOSO splits/preprocessing as eeg2025 (or document the diff).
3. Then (PI-gated) the official-baseline full-LOSO run mirroring the Mini sidecar scope.

**Recommendation:** hold. The Mini baselines already established D-is-not-SOTA; official baselines are a
reviewer-hardening step, worth doing but not before P10. When run, use **acar-v4-regen**, report OFFICIAL
names explicitly, and match preprocessing to the Mini sidecar (or disclose the resample/window difference).
No official-baseline GPU launched.
