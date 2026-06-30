# ACAR v4 — C1/C2 compatibility-replay READINESS **(executable, AUTHORIZATION-gated; NO replay run)**

```
STATUS : run_substrate_compatibility.py is an EXECUTABLE, authorization-gated fixed-candidate DEV substrate-compatibility
         replay. C1 = two-commit split + auth gate + taxonomy. C2 = REAL replay body under the FROZEN B1b source-state (NO
         refit): _load_frozen_substrate + _reembed_dev_under_substrate + _derive_under_frozen_source_state (never
         build_cohort_input) + REAL _extract_fixed_candidate_stats. C3 = REAL alignment vs the sha-pinned DEV feat-dump metadata
         + EXACT eval_L_harm_all (all-batch denominator; replaces the harm_rate proxy; harm_among_adapted descriptive only).
         C4 = REAL _load_subject_raw_windows (cmi DEV pipeline, single-subject allowlist; the dump metadata supplies keys+order+
         labels, reader supplies ordered signal paired by position with a hard COUNT check) — NO SubstrateReplayNotWiredError
         raise-frontier remains; the only DEV-raw read happens at the authorized C-run (under acar-v4-regen 3.13). NO replay was
         run: without a valid compatibility authorization manifest it
         fails closed (SubstrateCompatibilityNotAuthorizedError) before any torch/cmi import or DEV read. Synthetic-tested only.
         NO DEV raw read, NO re-embedding, NO held-out/external read, NO acar-v4-protocol tag. v2/v3 untouched; lockbox SEALED.
DATE   : 2026-06-30 (machine UTC)
```

## 1. Two-commit split (the key C1 design — avoids the dead-lock)
The substrate-compat manifest now pins TWO commits instead of one ambiguous `protocol_commit`:
```
substrate_protocol_commit     = b99fa4f…   (the FROZEN substrate-generation code that trained the PD/SCZ substrates)
compatibility_protocol_commit = <C1 commit>  (this executable-replay code)
```
The runner requires `HEAD == compatibility_protocol_commit` (NOT the substrate commit), so the b99fa4f substrates stay
authoritative artifacts while the replay runs under the C1 commit. `validate_substrate_manifest` REJECTS the retired bare
`protocol_commit`. The substrates are NOT retrained or modified.

## 2. Compatibility authorization manifest (schema only; NOT created)
`regen_substrate.validate_compat_authorization` + `COMPAT_AUTH_FIELDS` + `REQUIRED_COMPAT_STATEMENT`:
```json
{
  "compatibility_protocol_commit": "<C1 commit; HEAD must == this at run>",
  "substrate_protocol_commit": "b99fa4fcfb83c6ee60996c50dba6828d40561f26",
  "substrate_manifest_sha256": "<sha256 of the substrate-compat manifest FILE bytes>",
  "env_lock_sha256": "61e505b3…",
  "output_path": "<absolute new nonexistent output directory>",
  "authorized_by": "...", "authorization_time": "<ISO-8601 UTC>",
  "statement": "Authorize fixed-candidate DEV substrate compatibility replay exactly under ACAR_V4_SUBSTRATE_REGEN_COMMAND.md"
}
```
`run_substrate_compatibility._load_compat_authorization` BINDS it: compatibility_protocol_commit / substrate_protocol_commit /
substrate_manifest_sha256 (== the file sha of THIS manifest) / env_lock_sha256 / output_path must all match. Missing/invalid →
fail before any torch/cmi import, DEV read, or output claim.

## 3. Substrate-compat manifest schema additions (for the replay data source)
Per disease, in addition to the 4 unambiguous artifact hashes (H5): `dev_input_manifest_path` + `dev_input_manifest_sha256`
(the B1b regen input manifest — pins the EXACT eligible DEV universe to re-embed: PD 230 / SCZ 225, ds004000/sub-042 excluded,
FROZEN_PIPELINE, cohort-aware keys — so the replay can NOT rediscover a different subject universe). Top-level: `env_lock_path`
(the substrate env lock, re-verified at replay). The stdlib preflight verifies the `*_file_sha256`, the dev-input-manifest file
sha, and the env-lock file sha.

## 4. Executable replay body (gated; `_run_compatibility_replay`)
Reached ONLY with a valid, bound authorization, under an atomic `os.mkdir(output)` claim:
```
_verify_runtime_matches_lock(spec)              # cuda + threads=1 + versions == the substrate env lock         (REAL)
_verify_substrate_semantic_hashes(spec)         # canonical encoder_state_dict + source_state_artifact == record (REAL; safe
                                                #   weights_only load + acar.v3 self-verifying load_frozen; no unsafe pickle)
frozen = _load_frozen_substrate(spec)           # per disease: EEGNet(weights_only) + load_frozen_source_state_artifact (REQUIRED; no refit)
reembed = _reembed_dev_under_substrate(spec,frozen)  # eligible old-seven DEV → NEW-encoder z + v3 keys + labels; build_deployment_batches
records,v2 = _derive_under_frozen_source_state(reembed)   # SourceStateRegistry(frozen art) + disease_exec_cache + run_c0 +
                                                #   real_adapter._emit_records over the v3 CV folds — NEVER build_cohort_input (no refit)
per_disease = run_dev_exploration(records, EXACTLY ONE config 1x1x1, g3=v2_replay) → _extract_fixed_candidate_stats   (REAL)
authorized,reason = regen_substrate.compatibility_replay_pass(per_disease)   # FROZEN pre-registered pass-line   (REAL)
status = SUBSTRATE_COMPATIBILITY_PASS | _FAIL
```
- **No-refit (C2, blocker 2):** the replay loads the FROZEN B1b PD/SCZ source-state artifacts (`load_frozen_source_state_artifact`,
  REQUIRED) and EXECUTES them (`SourceStateRegistry`→`disease_exec_cache`→`SourceStateArtifact.execute`; `run_c0`;
  `real_adapter._emit_records`). It DELIBERATELY never calls `real_adapter.build_cohort_inputs` / v3 `build_cohort_input` (which
  RE-FIT a per-cohort source-state) — enforced by a source-grep guard test. So the verdict checks the B1b substrate, not an
  implicit re-fit.
- **FIXED candidate only:** the exploration is pinned to EXACTLY ONE config (1×1×1: `score_families=[shift_margin]`,
  `policy_families=(benefit_ranked,)`, `losses=(harm_indicator,)`, `budget_by_loss={harm_indicator:0.10}`); no grid to reselect.
- **`_extract_fixed_candidate_stats` is a REAL deterministic accessor**: filters `result.reports` to the single
  (disease, benefit_ranked, harm_indicator) cell, fail-closes if not exactly one per disease or a disease/v2 is missing, and maps
  lambda_certified=`g4_harm_control_pass`, coverage, red, **L_harm_all_eval=`eval_L_harm_all`** (C3), v2_replay_red=`c0_red`
  (== the v2_replay comparator since g3=v2_replay), v2_evaluable=finite(c0_red).
- **(C3) EXACT EVAL harm — no proxy.** The gate uses `eval_L_harm_all`, the EXACT all-batch-denominator EVAL harm_indicator loss
  (subject-macro over ALL eval batches incl. identity/fallback), added as an ADDITIVE field to `develop.V4CandidateReport`
  (`RC.subject_losses_from_policy(np.stack([calibrated]), dr_ev, subj_ev, loss="harm_indicator").mean()`, mirroring
  `external_adapter.evaluate_stratum`). It is REQUIRED + finite (None/NaN ⇒ fail-closed). The conditional `harm_rate` is carried
  DESCRIPTIVELY as `harm_among_adapted` (None when nothing adapted) and NEVER gates. Zero-adaptation: `eval_L_harm_all=0.0`,
  `harm_among_adapted=None`, compatibility FAILS via the coverage gate (not via harm). The `develop` change is additive — it does
  NOT alter coverage/red/harm_rate/g4 or any existing v2/v3 number (manifest assertions are relative; v3 uses a separate module).
- **The pass-line** is the FROZEN `compatibility_replay_pass` (v2_replay HARD; per disease: CAL LTT λ* certified, coverage≥0.15,
  red>0, EVAL L_harm_all≤0.10, v2 evaluable, red>v2_replay_red; macro red>macro v2).
- **(C3+C5) Raw-window↔v3-WindowKey alignment is REAL + synthetic-tested; the window reader is REAL + EXACT-KEYED (no frontier,
  no by-position trust).** The sha-pinned DEV feat-dump metadata (per-disease/cohort `dev_feat_dump_paths`+`dev_feat_dump_sha256`)
  is the alignment SOURCE OF TRUTH: it supplies the v3 WindowKeys (recording_id + window_index, VERBATIM) + the labels. The
  dump's `window_index_te` is a GLOBAL index in the producer's concatenated X (cmi.run_scps_crossdataset `load()` then
  `te_g=where(te_mask)`; `recording_id_te==subject`) — and that GLOBAL index IS the row index of the sha-pinned scps cache the
  dump was built from. So `_load_disease_cache` loads the cohort-filtered `{X,subject,cohort}` from `scps_cache_path`
  (`allow_pickle=False`), and `_load_subject_windows_and_keys` fetches each window at `cache["X"][window_index]` — an EXACT KEYED
  lookup, NOT a same-count by-position pairing — VERIFYING `cache.subject[gi]==subject` AND `cache.cohort[gi]==dataset_id` at every
  row. FAIL-CLOSED if the pinned cache sha mismatches (preflight), a global index is out of range, the subject/cohort-at-index
  disagrees (catches a same-count REORDER or wrong cache), a wrong shape, a non-finite window, or a duplicate
  (recording_id, window_index) — before any digest. `_check_reembed_universe` additionally asserts the re-embedded eligible-subject
  set == eligible AND counts EXACT. The live `load_cohort` raw reader is NOT called in the replay (grep-guarded).
  **No `SubstrateReplayNotWiredError` raise-frontier remains** — the whole replay path is real + executable; the only DEV
  read happens at the authorized C-run from the pinned cache (tests inject a synthetic keyed cache). Everything (frozen-substrate
  load, no-refit derive, keyed alignment, exact-keyed window reader, 1×1×1 exploration, exact eval_L_harm_all extraction,
  pass-line) is REAL.

> **C5 RESOLVED — read from the sha-pinned scps cache by EXACT KEYED lookup.** The window source is now the SAME per-condition
> scps cache (`{CACHE}/{condition}.npz`, X/subject/cohort) that produced the DEV feat dumps, sha-pinned per disease
> (`scps_cache_path`+`scps_cache_sha256`, verified in the preflight). The dump's `window_index_te` IS the cache row index
> (cmi.run_scps_crossdataset `load()` then `te_g=where(te_mask)`), so each window is fetched at `cache["X"][window_index]` —
> EXACT keyed lookup, NOT a same-count by-position guess — and `_load_subject_windows_and_keys` VERIFIES `cache.subject[gi]==subject`
> AND `cache.cohort[gi]==dataset_id` at every row. A same-count REORDER (or wrong cache) fails two ways: the pinned cache sha
> mismatches at preflight, and/or the subject/cohort-at-index check fails — so it can NEVER silently mispair. The live
> `load_cohort` raw reader is NOT used in the replay (grep-guarded); `_load_disease_cache` loads `allow_pickle=False`. Fail-closed:
> cache missing/sha-mismatch, out-of-range index, subject/cohort-at-index mismatch, wrong shape, non-finite, duplicate WindowKey,
> + `_check_reembed_universe` (re-embedded set==eligible & count==EXACT). Synthetic-tested incl. the same-count-reorder→FAIL case.
>
> **C5 adversarial review (4 lenses) = GO, no must-fix.** Empirically the real sha-pinned caches contain EXACTLY DEV_SCOPE and
> nothing else (PD 8523 = ds002778:1240 + ds003490:2000 + ds004584:5283; SCZ 9000 = ds003944:3280 + ds003947:2440 + ds004000:1680
> + ds004367:1600), so `np.isin(cohort, DEV_SCOPE)` is an all-True no-op and `X[keep]` is the full cache in producer order →
> `cache["X"][gi]` is byte-identical to the producer's `X[gi]`. The `np.isin` filter (not a hard "== DEV_SCOPE" assert) is
> DELIBERATE: it also stays correct if a cache is ever a SUPERSET of DEV_SCOPE with the producer having filtered to DEV_SCOPE
> (identical order-preserving mask reproduces the producer's subset index). Two residuals were judged ACCEPTABLE (closed by
> existing layered guards, NOT must-fix): (1) the producer's exact cohort set is not pinned as its own manifest field — a SUBSET
> producer would omit whole cohorts/subjects from the dump → `_check_reembed_universe` FAILS (missing != eligible 230/225); a
> SUPERSET producer → larger gi → out-of-range or subject/cohort-at-index mismatch. (2) a within-subject, same-cohort window
> reorder (a different window of the RIGHT subject at index `gi`) is the one thing the per-row subject/cohort check cannot see
> (the cache carries no per-window recording/content column) — but any such row reorder changes the cache BYTES → the pinned
> `scps_cache_sha256` mismatches at preflight BEFORE any read. Closing (2) at the code level would need a per-window raw-content
> hash in the dump = regenerating the DEV dumps, which is OUT of C5 scope; the sha-pin closes it as-is.
- **(C4) runtime:** the replay executes under `acar-v4-regen` (python 3.13). The exact replay-path modules are green under 3.13
  (regen_substrate + develop suites — RSC import, the 1×1×1 run_dev_exploration path, eval_L_harm_all extraction,
  compatibility_replay_pass, the alignment subset). The home v4 suite is 3.9 (10/10); the alignment exercise needs
  acar.v3.set_features (py3.10+) so it SKIPS on 3.9 and RUNS under 3.13. (A pre-existing 3.13 numpy diff in
  `frontiers.py::frontier_auc` makes the descriptive `frontiers_policies` AUC test red under 3.13 — NOT on the replay path, NOT
  touched by C1–C4; green on 3.9.)

## 5. Result taxonomy (no selection/external/binding vocabulary)
`regen_substrate.SUBSTRATE_COMPAT_STATUSES = (SUBSTRATE_COMPATIBILITY_PASS, SUBSTRATE_COMPATIBILITY_FAIL,
OPERATIONALLY_ABORTED_NO_VERDICT)`. PASS/FAIL are written to `compat_RESULT.json` (last; `allow_nan=False`) +
`compat_manifest.json`. An operational failure removes the claimed output (no partial; ABORT is NEVER read as a FAIL). This is
the old-seven-DEV-substrate compatibility check — NOT a new DEV selection run, NOT external evidence; no SELECT / DEV_STOP /
V4_EXTERNAL_CONFIRMED / external-G2 / coverage-theorem vocabulary is emitted.

## 6. Synthetic guards (no real raw/torch/GPU)
two-commit required + retired `protocol_commit` rejected; `env_lock_path` required; per-disease dev-input fields required
(incl. C3 `dev_feat_dump_paths`/`dev_feat_dump_sha256` AND C5 `scps_cache_path`/`scps_cache_sha256`); HEAD≠compatibility_protocol_commit
→ fail; reselection (candidate≠fixed) → fail; op-point drift → fail; missing/invalid compat authorization → fail before
import/read/output; authorization hash mismatch → fail; wrong statement / extra / missing field → fail; fake replay body called
exactly once only after authorization; PASS verdict & FAIL verdict both written; non-verdict status → abort+cleanup; replay
raising → output cleaned. **C5 keyed-alignment exercise (runs under 3.13):** keyed-cache happy path; same-count REORDERED cache
→ FAIL (subject-at-index mismatch); cohort-at-index mismatch → FAIL; out-of-range global index → FAIL; bad window shape → FAIL;
duplicate WindowKey → FAIL; grep-guard `load_cohort(` NOT in the replay source; `fit_source_state`/`build_cohort_input(`/
`real_adapter.derive` NOT in the replay source (no-refit). Tests monkeypatch `_run_compatibility_replay` and inject a synthetic keyed cache.

## Next sequence (separate decisions; NOT started)
1. (if the replay uses torch/GPU) capture/verify a runtime lock for the C1 commit;
2. rebuild the substrate-compat manifest with `compatibility_protocol_commit = <C1 commit>` (+ env_lock_path + per-disease
   dev_input_manifest fields);
3. run the fail-closed compatibility PREFLIGHT at detached C1 WITHOUT authorization → expect SubstrateCompatibilityNotAuthorizedError;
4. record; THEN ask for an explicit compatibility-replay authorization.
Only after that authorization may the old-seven DEV compatibility replay RUN (and the two frontiers be finalized + validated).
A PASS then unlocks ACAR_FROZEN_v4 + the acar-v4-protocol tag; a FAIL → DEV-only / new dated protocol, with NO post-hoc tuning
of candidate/score/loss/grid/comparator/thresholds.
