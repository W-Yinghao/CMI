# ACAR v4 — Option A step 1: read-only DEV EEGNet encoder-checkpoint search **(NOT_FOUND)**

```
DATE            : 2026-06-29T22:48Z (machine UTC)
COMMIT SEARCHED : a6cbc8b (branch acar, HEAD at search time)
SCOPE           : READ-ONLY inventory. NO retrain, NO model run, NO external/held-out read, NO new numbers, NO tag.
CONCLUSION      : NOT_FOUND — no archived DEV EEGNet encoder checkpoint exists in any searched repo/project artifact path.
```

## What this step is / is not
Goal: determine whether the original DEV EEGNet encoder (the one that produced the `feat_dump_v4` / erm_0 embeddings)
was left behind anywhere local, so Option A (recover the original checkpoint) could keep the DEV feature substrate
bit-identical. This is a filename/text/metadata inventory only — no deserialization of unknown pickles, no inference.

## Paths searched
```
/home/infres/yinwang/CMI_AAAI_acar          (acar worktree, incl. its cmi/ copy)
/home/infres/yinwang/CMI_AAAI               (main repo)
/home/infres/yinwang/CMI_AAAI/archive       (incl. archive/lpc-cmi-failed/results/feat_dump_v4 — the DEV erm_0 dumps)
/home/infres/yinwang/CMI_AAAI/results
```

## Paths EXCLUDED (not searched — possible external/held-out/raw payload, or VCS internals)
```
*/heldout/*  */lockbox/*  */zenodo14808296/*  */ds007526/*  */.git/*
```
No held-out raw EEG / OpenNeuro / Zenodo external download was opened or read.

## Patterns / methods used
```
git grep (acar repo): torch.save | state_dict() | load_state_dict | torch.load | save_checkpoint | *.pt/*.pth/*.ckpt
                      | save_backbone | save_encoder | save_model       (-- excluding *.npz, *.npy)
git grep (cmi/):       torch.save | state_dict | checkpoint | ckpt | save_   ;  np.savez | dump_audit | feat_hash_te | z_te
find (both repos):     *.pt *.pth *.ckpt *.pkl *.joblib  AND  *eegnet* *encoder* *state_dict* *checkpoint*
                       (excluding *.npz/*.npy and the excluded paths)
find catch-all:        any file > 3 MB that is NOT *.npz/*.npy/*.json/*.csv/*.pdf/*.png  (a mis-named weights blob)
metadata triage:       (would have been) sha256sum / ls -lh / stat on any candidate
```

## Findings
1. **Code-level: the DEV pipeline NEVER persisted the encoder.** `git grep "torch.save" -- cmi/**` → **no matches**.
   `cmi/run_scps_crossdataset.py` produces the dumps via `np.savez` of *embeddings* only (`_dump_audit_fold` at L121/L154;
   `dump_features` at L306–309 writes `z_te=embed(bb, Xte)`, `y_te`). The EEGNet backbone `bb` is built, trained, used to
   embed, then discarded — its `state_dict` is never written. The `[ckpt]` prints in `run_loso.py`/`run_scps_crossdataset.py`
   are per-fold RESULT-JSON checkpointing, not weight saving.
2. **No serialized-model files.** `find` for `*.pt/*.pth/*.ckpt/*.pkl/*.joblib` across both repos (minus excluded paths) →
   **empty**.
3. **Name-matched hits are not weights.** `*eegnet*/*encoder*/*checkpoint*/*state_dict*` → all `*EEGNet*.json` metric-result
   files from the broader CMI benchmark study (TUAB/ADFTD/BNCI/MUMTAZ/SEED/…), the EEGNet paper PDF, and this directory's own
   `ACAR_V4_ENCODER_ARTIFACT_DECISION.md`. None are model weights.
4. **DEV dump dirs are npz-only.** `archive/lpc-cmi-failed/results/feat_dump_v4` = 21 `.npz` (embeddings) + 1 `.json`
   (manifest); all other `feat_dump*` dirs are likewise `.npz` (+ optional `.json`). No weights.
5. **Catch-all large-binary sweep empty.** No >3 MB non-data binary (i.e. no mis-named weights blob) in the searched paths.

The acar repo's only `state_dict`/`load_state_dict` references are the v3 DeepSets CONFORMAL predictor + its training
(`acar/v3/predictors.py`, `acar/v3/training.py`) and the v4 test fake — none is the EEGNet encoder.

## Conclusion
```
FOUND_CANDIDATE : NO
NOT_FOUND       : YES — no archived DEV EEGNet encoder checkpoint in the searched repo/project artifact paths, and the DEV
                  code path (cmi/) structurally never saved one (no torch.save). The erm_0 dumps retained embeddings only.
INCONCLUSIVE    : only insofar as locations OUTSIDE the four searched roots were not inspected (e.g. SLURM scratch, a conda
                  env, another machine/device — cf. the B0 finding that some DEV hashes came from a different (CUDA) device).
```

## Next allowed decision (still gated; no retrain / no external read / no tag from this note)
- **A — recover original**: effectively foreclosed for the LOCAL artifact set (none found, and it was never saved). Only
  residual hope = an out-of-scope location (SLURM scratch / other device); worth a quick out-of-scope check before B, but do
  NOT assume recoverability.
- **B — regenerate + archive an all-DEV V4 external substrate**: the realistic path. HONESTY REQUIREMENT (binding): a newly
  trained encoder is a NEW V4 external representation substrate, NOT "the original encoder", UNLESS it bit-reproduces the
  archived DEV erm_0 embeddings (`feat_hash_te`). If it does not, `ACAR_FROZEN_v4.md` must be rewritten to declare
  "V4 external substrate = all-DEV frozen encoder/source-state produced by command X" with its own training command, input
  scope, seed, env lock, artifact hashes, and replay/compatibility checks. No retrain happens without that decision.
- **C — DEV-only fallback**: if B is declined, position V4 as a DEV-only exploratory candidate (no external confirmation).

This search does NOT change `ACAR_FROZEN_v4.md` executable status (already NOT_YET_EXECUTABLE; blocker 2 — the held-out
reader — is also still open). See [ACAR_V4_ENCODER_ARTIFACT_DECISION.md](ACAR_V4_ENCODER_ARTIFACT_DECISION.md).
