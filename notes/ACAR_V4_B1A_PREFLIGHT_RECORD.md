# ACAR v4 — B1a preflight record **(env capture = CAPTURE_FAILED; B1a BLOCKED on env; B1b NOT approvable)**

```
DATE        : 2026-06-29/30 (machine UTC)
NODE        : SLURM job 866446 (no GPU: torch.cuda.is_available() == False)
COMMIT      : 7e1c49e (probe target; protocol_commit pinned in the env lock)
AUTHORIZED  : B1a only — capture a real regen env lock + build fixed PD/SCZ input manifests + run the fail-closed preflight.
              NO training · NO held-out read · NO substrate generated · NO tag.
OUTCOME     : env-lock capture = CAPTURE_FAILED → per the pre-agreed rule ("若训练节点无法满足 lock,记录为 CAPTURE_FAILED,不要继续"),
              STOPPED before manifest-build / preflight-to-pass. DEV raw availability was verified (read-only) for the record.
```

## Step 1 — regen env-lock capture = CAPTURE_FAILED
Tool: `acar/v4/capture_regen_envlock.py` (env introspection ONLY — imports torch + import-probes the training stack; NO
training, NO data, NO model fit). Output: `notes/ACAR_V4_REGEN_ENV_LOCK.json`.
```
status                 = CAPTURE_FAILED
env_lock_sha256        = ee4c615ec996787719186bc8c7485b674cbca4752a1681db8846c29a625cce39
python_version         = 3.13.7
torch_version          = 2.6.0+cu124
cuda_version (compiled)= 12.4
device_kind            = cpu        (cuda.is_available() == False on this node)
braindecode_version    = ""         (import broken)
```
capture_note (the blockers that make this env NOT training-ready):
1. **No CUDA device** on this node (`torch.cuda.is_available() == False`).
2. **`import braindecode` fails** — `cannot import name 'BNCI2014001' from 'moabb.datasets'` (moabb/braindecode version clash).
3. **`from braindecode.models import EEGNetv4` fails** — `libtorchaudio.so: undefined symbol: _ZNK5torch8autograd4Node4name...`
   (torch/torchaudio ABI mismatch).
4. **`from cmi.models.backbones import build_backbone` fails** — same torchaudio ABI symbol error.

⇒ The EEGNet encoder cannot even be CONSTRUCTED in eeg2025 as installed, and there is no GPU here. A CAPTURED_AND_VERIFIED
lock would be a lie, so the capture tool emitted CAPTURE_FAILED. `run_regen_substrate` rejects any lock whose status is not
CAPTURED_AND_VERIFIED (unit-guarded for SCHEMA_ONLY and CAPTURE_FAILED), so this lock cannot pass the training preflight.

## Step 2/3 — NOT performed (stopped on CAPTURE_FAILED, as agreed)
No PD/SCZ input manifests were built and no `run_regen_substrate` preflight-to-pass was run, because a valid B1a end-state
requires a CAPTURED_AND_VERIFIED env lock (the manifest's `env_lock_sha256` must point at one). Building manifests now would
be "continuing" past the failure.

## Read-only DEV raw availability (the OTHER B1b prerequisite — verified, GOOD)
The all-DEV `source_kind = raw_bids` source exists locally under `ROOT = /projects/EEG-foundation-model/datalake/raw/scps`:
```
PD  ds002778  participants.tsv ✓  sub-dirs 31
    ds003490  participants.tsv ✓  sub-dirs 50
    ds004584  participants.tsv ✓  sub-dirs 149
SCZ ds003944  participants.tsv ✓  sub-dirs 82
    ds003947  participants.tsv ✓  sub-dirs 61
    ds004000  participants.tsv ✓  sub-dirs 43   (cohort task = "proposer", per cmi COHORTS — NOT resting)
    ds004367  participants.tsv ✓  sub-dirs 40   (cohort task = "rdk", per cmi COHORTS — NOT resting)
```
So the DEV raw is present (no `INPUT_MANIFEST_NOT_EVALUABLE` on availability grounds) — **the sole B1a/B1b blocker is the
training environment** (broken braindecode/backbone import + no GPU). (NB: the cmi COHORTS tasks above are how the DEV
substrate selects windows; the held-out reader's resting-only selection is a separate, external concern.)

## Conclusion + what unblocks B1
```
B1a status : INCOMPLETE — env CAPTURE_FAILED.
B1b status : NOT APPROVABLE (no captured env lock; no trained substrate).
To unblock (a separate, reviewed step — still NO training until then):
  1. Repair the training env: a Python env where  `from braindecode.models import EEGNetv4`  AND
     `from cmi.models.backbones import build_backbone`  both import (fix moabb/braindecode compat + torch/torchaudio ABI).
  2. Run on a GPU node (or explicitly choose CPU via capture --allow-cpu, recording device_kind=cpu — a deliberate decision,
     noting the DEV substrate likely used CUDA; cf. the B0 different-device finding).
  3. Re-run B1a: capture → CAPTURED_AND_VERIFIED → build fixed PD/SCZ input manifests (raw_bids; with subject-list /
     diagnosis-label / per-cohort raw-file-manifest hashes) → fail-closed preflight (expect SubstrateTrainingNotAuthorizedError).
```

## Constraints reaffirmed (unchanged)
NO training/GPU training · NO DEV raw signal loaded into a model · NO window extraction / embedding / source-state fit · NO
held-out raw read · NO external preprocessing · NO compatibility replay on real DEV · NO `acar-v4-protocol` tag · NO
ACAR_FROZEN_v4 finalization · NO external Arm-B run. External Arm B = NOT_YET_EXECUTABLE; lockbox SEALED; v2/v3 frozen
results+tags untouched. See `notes/ACAR_V4_SUBSTRATE_REGEN_COMMAND.md` (frozen command contract) and
`notes/ACAR_V4_ENCODER_ARTIFACT_DECISION.md` (A foreclosed; B design; C not chosen).
