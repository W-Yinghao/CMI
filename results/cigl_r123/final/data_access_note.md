# Data-access note (reproducibility)

`BNCI2014_001` loads directly from the shared datalake. `BNCI2015_001` required a **read-only** workaround:

- **Original source:** `/projects/EEG-foundation-model/datalake/raw/MNE-bnci-data/~bci/database/001-2015/*.mat`
  are owner-only (`tmaye`, mode ~0600; not group-readable) — MOABB/pooch failed with `PermissionError`.
- **Readable copy (world-readable, `rwxrwxrwx`):** `.../MNE-bnci-data/database/data-sets/001-2015/*.mat` (28 files,
  subjects S01-S12, sessions A/B[/C]).
- **Staged tree (ours, read-only symlinks to the readable copy):** `/home/infres/yinwang/mne_stage_bnci/MNE-bnci-data/~bci/database/001-2015/`.
- **Override:** the 2015 sbatch exports `MNE_DATASETS_BNCI_PATH=$STAGE` and `MNE_DATA=$STAGE` (cmi.paths uses
  `os.environ.setdefault`, so a pre-set env wins). We did **not** copy or modify any `.mat` bytes; symlinks point
  at the world-readable copy.
- **Loaded shape (verified):** X (5600, 13, 384), 2 classes (feet vs right_hand), 12 subjects → 12 LOSO folds.
- **No trial / label / fold / split / normalization logic changed** — only the data-file path. The datalake perms
  should be fixed at source (`chmod g+r` the `~bci/001-2015` files) for clean reproduction.
