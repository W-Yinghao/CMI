# ACAR V5 — Stage-2A Selection-Readiness / Package-Intake / Runner-Preflight

```
CODE + SYNTHETIC TESTS ONLY
REAL PACKAGE-INTAKE PREFLIGHT IS READ-ONLY (registry.json / FINALIZED.json / feat_dump HEADERS)
NO DEV LABEL USE · NO CANDIDATE SCORING ON REAL DATA · NO THRESHOLD FITTING ON REAL DATA
NO G1–G6 EVALUATION · NO SELECTED CANDIDATE · NO S1/S2/S3 · NO EXTERNAL · NO LOCKBOX
NO SUBSTRATE REBUILD · NO REPAIR/MONTAGE/LABEL POLICY CHANGE
STAGE-2B (REAL DEV SELECTION) IS A SEPARATE AUTHORIZATION — NOT ISSUED HERE
```

Stage-2 is the **first label-consuming** step of ACAR V5. Before that run, Stage-2A proves the selection runner **consumes the
Stage-1B package correctly and cannot violate the protocol**: it admits only the finalized package, uses only the canonical
selection substrates and the exact 22-candidate manifest, keeps S1 robustness seeds / external sites / lockbox out of selection,
and reads no label. It is **dry-run only** — it selects nothing.

## Package intake

The admitted Stage-1B package is `acar-v5-stage1b-c4412b4-r1` (`registry_sha256 =
2bbe55f4cdb4f1a18cee3b2c9e7583dba9fe9e84b9c563fb37781e98ebcbb76d`, 30/30 refs, `admit_run = ADMITTED`; see
`ACAR_V5_STAGE1B_REALRUN_RESULT_C4412B4_R1.md`). The Stage-2A real preflight may READ, read-only:

```
registry.json          (via admit_run: FINALIZED + n_refs==30 + matching registry_sha256)
FINALIZED.json         (marker binding)
feat_dump HEADERS      (schema/provenance scalars only — the `embedding` array is NEVER materialized)
registry hashes / artifact metadata
```

It may **not** run candidate selection, read a DEV label, fit a threshold, or score on real data.

## Modules (new, `acar/v5/`)

- **`stage2_selection_manifest.py`** — binds + re-validates the frozen 22-row manifest from `protocol.py` (never redefines it):
  exactly 22 rows, `{P1:4,P2:4,P3:6,P4:2,P5:6}`, ids == `protocol.CANDIDATE_IDS`, canonical order, every `disease_scope='both'`
  (joint PD/SCZ). Fail-closed `Stage2ManifestError` on any drift.
- **`stage2_package_intake.py`** — the ONLY Stage-2 module that touches the real package. `admit_and_validate_registry(output_root,
  run_id)` → `admit_run` + re-validation (exactly the 30 canonical refs; no forbidden/foreign token; the 10 selection refs present;
  the 20 others are S1-only) → immutable `Stage2PackageView`. `read_feature_dump_header` / `validate_selection_feature_dumps` read
  headers only (numpy lazy; embeddings never loaded; a label-like field → fail-closed). Fail-closed `Stage2IntakeError`.
- **`stage2_selection_runner.py`** — `dry_run_selection_readiness(...)` proves readiness and returns a report with
  `selected_candidate = None`. `run_binding_selection(...)` **always raises** `Stage2BNotAuthorizedError` (`_STAGE2B_ENABLED =
  False`). Exposes `GATE_CERTIFIER = ltt.gate_disease` (the conditional-on-adapted G4) and `assert_label_free_routing()`.

## The 10 fail-closed properties Stage-2A proves (synthetic guards)

```
 1. admit_run(output_root, run_id) succeeds.                          test_stage2_intake_admits_stage1b_package
 2. registry has exactly 30 refs (= the canonical set).              test_stage2_intake_admits_stage1b_package
 3. selection consumes only the 10 canonical selection refs          test_stage2_uses_selection_seed_only
      {PD,SCZ} × folds 0..4 × seed 20260711.
 4. seeds 20260712/20260713 are S1-robustness only and cannot        test_stage2_rejects_s1_seed_for_selection
      influence candidate identity.
 5. candidate manifest is exactly the 22 rows P1=4,P2=4,P3=6,        test_stage2_exact_22_candidate_manifest
      P4=2,P5=6.
 6. candidate identity is selected JOINTLY across PD and SCZ.        test_stage2_joint_disease_selection_only
 7. no external/provisional/excluded/foreign token appears.          test_stage2_no_external_tokens
 8. no label is visible to routing/scalarization code.              test_stage2_no_label_in_routing
 9. G4 harm_among_adapted stays conditional-on-adapted;             test_stage2_g4_conditional_adapted_non_evaluable_fail
      no-subject-adapts = non-evaluable = fail.
10. the Stage-2 output path is dry-run only (no candidate           test_stage2_dryrun_no_candidate_selected
      selected) unless a later Stage-2B authorization exists.
```

Frozen pins used (from `acar/v5/protocol.py` @ tag `acar-v5-protocol` `4278435`): `SELECTION_SEED = 20260711`; `S1_SEEDS =
(20260711, 20260712, 20260713)`; 22-candidate manifest; DEV cohorts PD `{ds002778,ds003490,ds004584}` / SCZ
`{ds003944,ds003947,ds004000,ds004367}`; forbidden site tokens `{zenodo14808296, ds007526, zenodo14178398, ds007020}`; gates G1
coverage LCB≥0.15, G3 UCB≤0.10, G4 UCB≤0.30 (conditional-on-adapted), and Holm over H1(G3)/H2(G4)/H3(G1) at α=0.05 with G2
(ε=0.02 NLL) as the effect-size gate outside Holm.

## Forbidden in Stage-2A

```
no DEV label use · no candidate scoring on real data · no threshold fitting on real data
no G1–G6 evaluation · no selected candidate · no S1/S2/S3 robustness · no external / held-out read
no ASZED · no lockbox · no substrate rebuild · no repair/montage/label policy change
```

## Verification

Full v5 suite (now **163** guard modules incl. the 9 Stage-2A guards) green on py3.9 (home; mne 1.8, no torch) and py3.13
(acar-v4-regen; torch 2.6, mne 1.12.1). Every `acar.v5` Stage-2A module imports with **no** heavy dependency (numpy is imported
lazily, only inside the feat_dump header reader). Adversarial multi-agent review clean.

## Next gate (SEPARATE authorization)

**Stage-2B — real DEV candidate selection**: the first label-consuming V5 selection run, limited to the admitted Stage-1B package
only, the 10 canonical selection refs only, the exact 22 candidates only, joint PD/SCZ, gates G1–G5 only; no S1/S2/S3, no external,
no lockbox. It requires a new authorization pinned to a reviewed implementation SHA, which flips `_STAGE2B_ENABLED` and adds the
real selection logic + gates. Until then, `run_binding_selection` fails closed.
