# ACAR V5 — Stage-1B10 reviewed channel-alias + cohort-label adaptation layer (CODE + SYNTHETIC/FIXTURE TESTS ONLY; NO REAL RUN)

Protocol frozen at tag `acar-v5-protocol` @ 4278435. Stage-1B10 fixes the two preflight blockers that superseded the `0ab40ec`
real-run authorization (see `ACAR_V5_STAGE1B_PREFLIGHT_FAIL_0AB40EC.md`), WITHOUT restricting cohorts/subjects, without re-pinning the
montage to modern names, and without a broad global label alias. No real DEV read, no training, no embedding, no registry, no SLURM.

## 1. Pinned channel-alias layer (Blocker 2)
The OUTPUT montage stays the old-10-20 canonical order (`preprocessing_config.CHANNELS_19`). A PINNED input-alias layer
(`channel_aliases.py`) maps a recording's raw names to the canonical electrodes BEFORE the pick:
- input names stripped + case-normalized (casefold), so `Fp1/FP1/fp1` collapse and any-case canonical maps to itself;
- pinned modern→canonical aliases `T7→T3, T8→T4, P7→T5, P8→T6` (the same historical temporal electrodes);
- a name that is neither canonical nor a known alias is a NON-canonical extra → dropped;
- if two raw channels map to the SAME canonical channel → FAIL (e.g. both `T3` and `T7` present);
- if any of the 19 canonical channels is missing after aliasing → FAIL.
`preprocessing_config` gains the pinned policy (part of `preprocessing_config_sha256`): `logical_montage_policy =
old_10_20_canonical_with_reviewed_10_10_aliases`, `channel_alias_schema_version = acar_v5_channel_alias_v1`,
`input_channel_aliases = {T7:T3, T8:T4, P7:T5, P8:T6}`, `duplicate_logical_channel_policy = fail_closed`. `real_mne_reader`
resolves the source names in canonical order, picks them, runs the DSP, and reindexes the data ROWS to canonical order (robust to
mne pick ordering). Guards: channel-aliases-modern-to-canonical / duplicate-logical-fail / missing-canonical-fail /
extra-channels-dropped / fp-case-normalized.

## 2. Frozen per-cohort label spec (Blocker 1)
`cohort_label_spec.py` pins EXACTLY how each of the 7 DEV cohorts' labels are read (no disease-wide fallback, no path inference):
- ds002778: subject-id prefix — strip `sub-`, casefold, `hc`→control(0) / `pd`→case(1);
- ds003490: column `Group`  values `CTL`→0 / `PD`→1;
- ds004584: column `GROUP`  values `Control`→0 / `PD`→1;
- ds003944 / ds003947: column `type`  values `Control`→0 / `Psychosis`→1;
- ds004000: column `group`  values `HC`→0 / `P`→1;
- ds004367: column `Group`  values `Control`→0 / `Patient`→1.
Column matching is case-insensitive + whitespace-stripped; two columns collapsing to the same name → FAIL; a value not exactly the
pinned control/case (after strip+casefold) → FAIL; missing/duplicate subject rows → FAIL. `COHORT_LABEL_SPEC` covers EXACTLY the 7
frozen DEV cohorts. `real_dev_reader.read_subject_label` / `subject_label_resolvable` now delegate here — the label VALUE is produced
only through `AuthorizedFitDatasetView.read_label` (FIT training only); the eligibility resolver returns a BOOLEAN only; the embedding
view still has no label path (verified, incl. via closure `__self__`). Guards: cohort-label-spec-all-7-fixtures /
cohort-label-unknown-value-fails / cohort-label-wrong-column-fails / cohort-label-duplicate-casefold-column-fails /
ds002778-subject-prefix-label / embedding-view-still-has-no-label-path.

## Verification
Full v5 suite = **97 guard modules, green py3.9 + py3.13**; every `acar.v5.substrate` module imports with NO heavy dependency
(`channel_aliases` / `cohort_label_spec` are pure). The existing FakeRaw reader tests still pass (exact-canonical names alias to
themselves).

## Still forbidden in Stage-1B10 (unchanged)
real DEV read (beyond the metadata preflight already reported) · real training · embedding dump · registry population from real
artifacts · SLURM · Stage-2 selection · S1/S2/S3 · held-out/external · lockbox.

## Next gates (separate authorizations)
1. **Stage-1B10P** — a tiny metadata/channel preflight on the real cohorts: read `participants.tsv` + `channels.tsv`/raw header
   channel names ONLY (no signal load, no DSP, no training/embedding/artifacts/registry) to confirm the alias layer resolves all 19
   canonical channels for every subject and the cohort label spec resolves every subject's label.
2. **Stage-1B real run** — re-authorize against the Stage-1B10 commit (full 40-hex `implementation_base_sha`) + a captured runtime
   lock, then `run_stage1b_real_build(...)` with the three real factories on the 7 pinned DEV cohorts.
