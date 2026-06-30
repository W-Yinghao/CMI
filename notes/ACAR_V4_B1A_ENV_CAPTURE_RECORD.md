# ACAR v4 — B1a env-capture record **(env PROVEN; first lock SUPERSEDED; operational recapture pending, external)**

```
DATE   : 2026-06-29/30 (machine UTC)
RESULT : acar-v4-regen built (R1-A/A2) and RUNTIME-VALIDATED (CPU import + A40 CUDA acceptance). The FIRST captured lock
         (A40, sha 589ceed…, commit 785d963) is a USEFUL DIAGNOSTIC but is SUPERSEDED — NOT an operational B1 lock —
         because it had: torch_interop_threads=20 (training must pin interop=1), no import-critical version fields
         (torchvision/torchaudio/moabb), and a commit self-reference (lock protocol_commit 785d963 ≠ record commit). It is
         removed from the repo; a corrected operational lock is recaptured EXTERNALLY on the correction commit (see below).
STATUS : B1A_ENV_PROVEN / OPERATIONAL_LOCK_PENDING_RECAPTURE
```

## Proven (still NO training / DEV-raw / held-out / source-state fit / tag)
- Isolated env `acar-v4-regen` built (eeg2025 untouched); exact pins: torch 2.6.0+cu124 / torchvision 0.21.0+cu124 /
  torchaudio 2.6.0+cu124 + braindecode 1.5.2 + moabb 1.5.0 (+ mne 1.12.1, skorch 1.4.0, numpy 2.4.4, scipy 1.18.0,
  sklearn 1.9.0). See `notes/ACAR_V4_REGEN_ENV_INSTALL_LOG.md`.
- Runtime imports PASS (CPU + A40): torch/torchaudio/torchvision; braindecode 1.5.2 + moabb 1.5.0; `EEGNetv4`;
  `cmi build_backbone`; `build_backbone("EEGNet",19,512)`. A40 (node30) acceptance: `cuda.is_available()==True`. ⇒ both
  eeg2025 blockers (torchaudio ABI + braindecode/moabb BNCI name) RESOLVED.

## Correction patch (this commit) — env-lock schema/capture hardened
- `regen_envlock`: lock now carries `torchvision_version`/`torchaudio_version`/`moabb_version`/`mne_version`/
  `skorch_version` (in the hash); a CAPTURED lock MUST have non-empty torchvision/torchaudio/moabb AND
  `torch_intraop==torch_interop==omp==1`.
- `capture_regen_envlock`: pins intra/inter/omp to 1 in the fresh capture process BEFORE any work, and records the new
  version fields. (Guards added; suites green.)
- The in-repo `notes/ACAR_V4_REGEN_ENV_LOCK.json` (the superseded 589ceed lock) is DELETED — the operational lock is
  repo-external (commit-consistency, per REGEN_COMMAND §4).

## Operational recapture (the correct B1a completion)
On a GPU node, against the CLEAN correction commit `H` (this patch), env introspection only — NO training/data:
```
PYTHONPATH=<repo> OMP_NUM_THREADS=1 .../acar-v4-regen/bin/python -m acar.v4.capture_regen_envlock \
    --output /abs/acar_v4_regen_env_lock_<H>.json --protocol-commit <H>
accept iff: status=CAPTURED_AND_VERIFIED · device_kind=cuda · torch_intraop=interop=omp=1 ·
            torchvision 0.21.0 / torchaudio 2.6.0 / moabb 1.5.0 non-empty · torch 2.6.0 · braindecode 1.5.2 ·
            pipeline_config_sha256=canonical · protocol_commit=H.
```
Then a record commit references ONLY the external lock path + `env_lock_sha256` (the lock stays out of the repo). After
that: build fixed PD/SCZ manifests (`env_lock_path`=external, `env_lock_sha256`, `protocol_commit=H`) → fail-closed
preflight (expect SubstrateTrainingNotAuthorizedError) → then ask for B1b.

(A CPU lock via `--allow-cpu` was NOT taken — separate substrate/runtime decision.) Boundaries unchanged; lockbox SEALED;
v2/v3 untouched.
