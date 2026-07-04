# ACAR V5 — Stage-1B first real substrate build: LAUNCH BLOCKED at pre-launch readiness (production reader never wired to the reviewed repair)

```
STATUS: STOPPED BEFORE ANY REAL DATA WAS READ / ANY TRAINING STARTED.
authorization: Stage-1B real substrate build (pinned implementation_base_sha=3fe885245133e2bc141651955da33bb7fd7adeac).
outcome: pre-launch readiness preflight (read-only; no signal load, no DSP, no training, no artifacts) found ONE run-dooming blocker.
decision: did NOT start run_stage1b_real_build; did NOT patch production code. The real-run authorization is treated as SUPERSEDED by
          this structural readiness gap (same discipline as the 0ab40ec preflight-fail). 0ab40ec also remains superseded.
```

The Stage-1B real-run authorization was received (pinned to the reviewed Stage-1B14 commit `3fe8852`). Before invoking
`run_stage1b_real_build(...)` — a first, expensive, irreversible 30-substrate GPU run over real DEV EEG — I ran a **read-only
launch-readiness preflight** (no `get_data`/DSP/training/artifacts). It surfaced **one blocker** that would crash the SCZ arm on the
first SCZ subject. Everything else is ready.

## Blocker (run-dooming): the production DEV reader never invokes the reviewed BrainVision read-repair
`acar/v5/substrate/real_dev_reader.py` — both `RealBidsDevReader.read_subject_windows` (line 64) and
`WindowsOnlyReader.read_subject_windows` (line 34) call:
```python
RMR.preprocess_subject(disease, cohort, subject, subject_dir)      # <-- NO staging_dir
```
`real_mne_reader.preprocess_subject(..., staging_dir=None)` takes the **pre-Stage-1B12** branch (`build_manifest` + `_read_raw`
directly, **no repair**). The reviewed Stage-1B12/1B13/1B14 BrainVision read-repair (marker synth / pointer rewrite / channels.tsv
ordinal rename) is invoked **only** when a `staging_dir` is passed — and the ONLY callers that pass it are the preflight tools
(`stage1b12p/13p/14p_preflight`) and the integration test `test_stage1b_read_repair_preprocess_subject_staging_integration`. The
**production** read path passes nothing.

**Why Stage-1B14P still PASSed:** the preflight tool performs the repair itself (`BR.plan_repair` + `BR.apply_repair` + opens the
repaired header). It validated **data ↔ repair-policy compatibility** (all 539 recordings admissible), NOT the **production reader
wiring** — which was never connected to the repair.

### Verified consequences (both reproduced read-only; no signal loaded)
- **ds004000 / sub-042** (2 recordings): `RM.build_manifest(subject_dir)` fails immediately —
  `RawManifestError: declared sidecar missing: .../019_P1.dat` (the broken internal pointers the Stage-1B12 pointer-rewrite exists to
  fix). Reproduced.
- **ds003944 (82) + ds003947 (61)**: marker-less headers → `_read_raw` → `mne.io.read_raw_brainvision(..., preload=True)` raises
  `configparser.NoOptionError: No option 'markerfile'` (confirmed repeatedly since Stage-1B11P; the channels.tsv ordinal rename +
  marker synth exist to fix exactly this).

So the SCZ substrates (which train on + dump ds003944/ds003947/ds004000) crash at the first SCZ subject read. The PD substrates
(ds002778/ds003490 native; ds004584 Pz-completion, which `preprocess_subject` handles WITHOUT `staging_dir`) would build, but the run
is not admissible partial — per the authorization a failed run stops immediately with no resume/skip.

## Everything else is READY (read-only audit — no blocker)
- **Numeric backend is a REAL implementation, not a stub** (`torch_eegnet_backend.py`): `fit(...)` builds the pinned EEGNet
  (`eegnet_architecture.build_eegnet`), trains deterministically with Adam + CrossEntropy + early-stopping on FIT-val, encodes the
  train features, fits the class-conditional-Gaussian source state (`source_state.fit_source_state`), and returns the 4 model
  artifacts as canonical pickle-free bytes; `embed_from_artifacts(...)` reloads the FROZEN encoder from the checkpoint and encodes
  per-subject windows (rows == n_windows). End-to-end training + embedding is wired.
- **Trainer / dumper / orchestration wiring is correct**: `make_real_trainer` → `train_encoder_and_source_state` (passes
  `TRAINING_CONFIG` to `backend.fit`, writes the 5 model/config files); `make_real_embedding_dumper` → `dump_fold_embeddings`
  (FrozenSubstrateHandle bound to the same ref, label-free view); `run_stage1b_real_build` → `build_fold_raw` (Phase-A FIT training →
  Phase-B label-free dump) with no signature mismatch / None-deref found.
- **Label firewall intact** (embedding view fails closed unless handed a windows-only reader; `WindowsOnlyReader` has no
  `read_subject_label`); subject index + eligibility robust; config/training-config/artifact-schema complete; file-artifact writer
  receives the expected path keys.
- **SHA**: the only difference between the auth SHA `3fe8852` and branch HEAD `a196dee` is the report-only preflight tool + the 1B14P
  note; **no production substrate code differs** (`stage1b_build.py` byte-identical). (Moot given the stop, but a clean `3fe8852`
  worktree was the intended runtime checkout to avoid mixing auth-SHA and checkout-SHA.)

## Recommended reviewed fix (NOT implemented here — needs a new authorization)
A small, well-scoped **production substrate** change — proposed as a next reviewed stage (e.g. **Stage-1B15**), not done under a "run
the build" authorization:
1. Wire `real_dev_reader` (both `read_subject_windows` methods) to pass a **per-run ephemeral `staging_dir`** into
   `preprocess_subject(...)`, so the production read path uses the reviewed repair. The staging dir must be **outside the raw tree**
   (BR already enforces this and fail-closes otherwise) and **ephemeral scratch** (NOT a registered artifact under the run's
   `output_root`, so the hash-bound file package stays clean); created + cleaned up per run.
2. Decide where the staging root comes from (execution context field vs a `tempfile.mkdtemp` per reader) — a reviewed design choice.
3. Re-run a synthetic/fixture guard proving the production reader now repairs (a BIDS `sub-*/eeg` fixture with a marker-less /
   generic-ordinal / broken-pointer recording read through `RealBidsDevReader.read_subject_windows` → validated SubjectWindows with
   `read_repair` recorded), on both pythons; adversarial review; then re-issue the Stage-1B real-run authorization pinned to the fixed
   commit.

## Residual first-real-run note (not a blocker; honest disclosure)
The real build is the first time montage completion actually runs `interpolate_bads` on **real** signal (ds004584 Pz, ds004000
F3/F4/P3/P4, ds004367 F7) — the preflights only checked donor geometry, never loaded signal. This path is fail-closed (non-finite
interpolated output → `MontageCompletionError`), so a numerically bad interpolation would crash cleanly rather than silently corrupt;
it cannot be preflighted without loading signal (which is forbidden until the real run).

## Decision
Stopped and reported. **No** real DEV signal was read, **no** DSP/interpolation/training/embedding ran, **no** artifacts/registry were
written, **no** production code was changed, and the build was **not** started. The Stage-1B real-run authorization (pinned `3fe8852`)
is treated as **superseded by this structural readiness gap**; `0ab40ec` stays superseded. Awaiting a reviewed authorization to wire
the production reader to the repair (Stage-1B15), after which the real-run authorization can be re-issued against the fixed commit.
