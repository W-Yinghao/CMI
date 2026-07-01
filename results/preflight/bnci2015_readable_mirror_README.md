# BNCI2015_001 readable symlink mirror (Graph-DualCMI pilot)

The datalake copy MOABB reaches by default is **owner-locked** and unreadable:
`/projects/EEG-foundation-model/datalake/raw/MNE-bnci-data/~bci/database/001-2015/*.mat`
(`-rw-------`, owner `tmaye`) — this is what made the preflight fail (`PermissionError` on `S01A.mat`, the
same block that hit Phase-3A-K).

A **world-readable** copy of the same files exists at:
`/projects/EEG-foundation-model/datalake/raw/MNE-bnci-data/database/data-sets/001-2015/*.mat`
(`-rwxrwxrwx`).

## Mirror

- **Mirror root:** `/projects/EEG-foundation-model/yinghao/cigl_bnci_readable`
- `MNE-bnci-data/~bci/database/001-2015/SxxY.mat` → **28 symlinks** to the readable `data-sets/001-2015/`
  copies (subjects S01–S12; A/B sessions; some missing per the dataset, hence 28 not 24).
- `MNE-bnci-data/~bci/database/001-2014` → symlink to the readable datalake original (BNCI2014_001 / BCI-IV-2a).
- First/last (sorted): `S01A.mat` … `S12B.mat`.
- Permission check: `head -c1` succeeds through every symlink (verified S01A/S06B/S12B).
- No data copied; no code changed; download stays disabled (offline).

## How runs use it (env, not code)

`cmi.paths.configure_offline_moabb()` uses `os.environ.setdefault(...)`, so a **pre-set** env var wins.
Export the mirror root before launching (preflight and the GPU job):

```bash
export MNE_DATASETS_BNCI_PATH=/projects/EEG-foundation-model/yinghao/cigl_bnci_readable
export MNE_DATA=/projects/EEG-foundation-model/yinghao/cigl_bnci_readable
```

## Verified

```
python scripts/preflight_moabb_env.py --datasets BNCI2014_001 BNCI2015_001 \
    --backbones DGCNNGraph --subjects 1 --no-download
```
→ **PASS (exit 0)**: DGCNNGraph builds; BNCI2014_001 loads (576 trials, 22ch, 4 classes);
BNCI2015_001 loads (400 trials, 13ch, 2 classes: feet/right_hand). Log:
`results/preflight/graphdualcmi_bfb1314_eeg2025_graphonly_preflight.txt`.

## Rebuild (if the mirror is lost)

```bash
M=/projects/EEG-foundation-model/yinghao/cigl_bnci_readable
SRC=/projects/EEG-foundation-model/datalake/raw/MNE-bnci-data/database/data-sets/001-2015
mkdir -p "$M/MNE-bnci-data/~bci/database/001-2015"
for f in "$SRC"/*.mat; do ln -sfn "$f" "$M/MNE-bnci-data/~bci/database/001-2015/$(basename "$f")"; done
ln -sfn /projects/EEG-foundation-model/datalake/raw/MNE-bnci-data/~bci/database/001-2014 \
        "$M/MNE-bnci-data/~bci/database/001-2014"
```
