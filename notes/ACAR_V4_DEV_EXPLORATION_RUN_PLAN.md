# ACAR v4 — DEV EXPLORATION RUN PLAN (first real old-seven run)

```
STATUS        : NON-BINDING / POST-V3 DEV_STOP / OLD-SEVEN DEV ONLY
LOCKBOX       : NO ACCESS
EXTERNAL ARM  : NOT AUTHORIZED
THIS FILE     : NOT ACAR_FROZEN_v4.md (a run plan; freezes nothing, selects nothing, ships nothing)
OUTPUT        : results/acar_v4_dev_exploration_001/
DATE          : 2026-06-29
```

This pins every choice for the **first real** V4 Phase-1 exploratory run on the seven old DEV cohorts **before** any
real record is derived. It is binding *as a pre-registration of the exploration* (no change after the run starts), but
the RESULT is **exploratory / model-selection only** — never external validation, never a deployable-router claim. All
choices below are fixed; if any must change, this file is edited and re-committed BEFORE a real read, and the run is
restarted.

## 0. Result taxonomy (the ONLY allowed verdicts)

```
run_status : V4_DEV_EXPLORATION_COMPLETE | OPERATIONALLY_ABORTED_NO_SCIENTIFIC_VERDICT
verdict    : V4_DEV_CANDIDATE_FOUND_FOR_POSSIBLE_FREEZE | V4_DEV_NEGATIVE_NO_LOCKBOX
NEVER      : SELECT · DEV_STOP · PROCEED_SAFE_ROUTER · UTILITY_ONLY · external G2 · coverage theorem · lockbox / Arm B
```

A killed/partial process = `OPERATIONALLY_ABORTED_NO_SCIENTIFIC_VERDICT` (no scientific verdict), per the v3 lesson.

## 1. Cohorts (old seven DEV; development-only)

```
PD  : ds002778, ds003490, ds004584        (230 subjects)
SCZ : ds003944, ds003947, ds004000, ds004367   (225 subjects)
```

Feature substrate: the archived `erm_0` (CITA-no-LPC) 16-d tangent dumps used by the A0/v2/v3 line
(`archive/lpc-cmi-failed/results/feat_dump_v4/audit_{cond}_{cohort}_erm_0.npz`; resolved via `acar.config.feat_dump_dir`
/ `$ACAR_FEAT_DUMP`). NO held-out / external / lockbox cohort is read.

## 2. Record derivation (V4OOFRecord from the v3 single-execution substrate)

`acar/v4/real_adapter.py` derives one `V4OOFRecord` per (disease, cohort, batch, fold, split) cell by REUSING the v3
machinery (import-only; v3 is never modified):

```
- per cohort: build the v3 source state (f_0) and execute each batch ONCE (identity + matched_coral/spdim/t3a) via the
  v3 execution path → ΔR_a(B) = NLL(f_a) − NLL(f_0) per non-identity action, and the captured p0/pa/z0/za/state.
- features_v2 [A,11] per non-identity action = bit-for-bit v2 acar.features.feature_vector on the SAME captured
  execution (the exact helper v3 uses for its C0/v2 replay).
- split = v3 S5 split-as-one-algorithm (subject-disjoint outer folds; non-EVAL → FIT/CAL): a batch is EVAL in its
  held-out outer fold and CAL in folds where its subject is in the CAL portion (cross-fit). FIT cells are emitted but
  not used by the V4 calibration.
- subject unit = cohort-aware cohort_id::subject_id; fallback batches (< MIN_BATCH) retained, forced identity.
```

**EXACT OOF coverage is required** (`run_dev_exploration(real_mode=True)`): every physical subject and every physical
batch must be EVAL in exactly one fold, else the run fails closed.

## 3. Score-family registry (frozen for this run)

`acar/v4/develop.py SCORE_FAMILY_REGISTRY` — 10 PRE-DECLARED label-free families, each depending ONLY on `features_v2`
(never ΔR). v2 paired-feature order: `d_entropy0 d_margin1 flip_rate2 js3 bures4 post_sep5 n_eff6 | g_unc7 s_support8
s_sep9 pr_cmi_proxy10`. Both fields "lower is safer/better".

```
shift_margin  : harm = benefit = +f[1]   (d_margin_pos)
js_flip       : harm = +f[3], benefit = +f[2]
d_entropy_pos : harm = benefit = +f[0]
d_entropy_neg : harm = benefit = −f[0]
d_margin_neg  : harm = benefit = −f[1]
flip_pos      : harm = benefit = +f[2]
js_pos        : harm = benefit = +f[3]
bures_pos     : harm = benefit = +f[4]
n_eff_neg     : harm = benefit = −f[6]
unc_pos       : harm = benefit = +f[7]
```

The run passes **all 10** names explicitly (`real_mode=True`). This is a Phase-1 minimal/representative registry, NOT a
comprehensive information ceiling. Pinned: `score_family_registry_sha256 = fe5a1f58986f7af1e8cb9db797ae9f08b46bfd749fc22ef8dbc8619005bc774e`
(must match the result manifest). Families are NOT chosen/changed after seeing the real run.

## 4. Calibration / gate config (frozen)

```
policy_families : safe_set (primary), benefit_ranked, direct_selective
losses          : mean, positive, harm_indicator   (subject-level; fallback realizes 0, stays in denominator)
budget_by_loss  : mean 0.0 · positive 0.05 · harm_indicator 0.10
risk control    : finite-grid LTT, method = ttest, correction = holm, alpha = 0.10
λ grid          : per family, grid_size = 12, derived from CAL label-free score statistics (no ΔR)
λ* selection    : most aggressive PASSING λ; PASS only if EVERY EVAL fold is certified (else NO_PASS)
weighting       : subject-macro (subject-equal); coverage/red/harm are weighted means
gate            : G0 provenance · G1 coverage ≥ 0.15/disease · G2 red > 0/disease · G3 disease-macro red > C0 ·
                  G4 CAL risk rule (rc PASS) · G5 fallback in denominator · G6 both diseases non-vacuous
```

## 5. C0 / comparator (G3)

```
g3_comparator  = v2_replay          (PRIMARY)
report BOTH    : v2_replay_red  AND  best_fixed_red   (best_fixed = best single fixed action; descriptive utility floor)
```

`v2_replay_red` is the deployed-NLL-reduction of the bit-for-bit v2 recipe (`acar.regressor.ActionRegressor`,
seed 0) on the SAME split, reused from the v3 C0 path. **Fallback (decided BEFORE any real read if v2 replay cannot be
produced reliably in the adapter):** edit this file to `g3_comparator = best_fixed` with `v2_replay_red =
descriptive/unavailable`, re-commit, then run. The comparator is NEVER switched after a real record is derived or the
run starts.

## 6. Output + provenance

```
output dir : results/acar_v4_dev_exploration_001/   (must NOT exist; atomic os.mkdir claim; RESULT.json sentinel)
manifest   : manifest.json with v4_oof_records_sha256, per (disease,cohort,fold,split) subject_list_sha256 + counts,
             config_sha256, score_family_registry_sha256, per-disease c0_best_fixed_red + c0_v2_replay_red,
             per-config G0–G6 + frontier gaps (ceiling + auc) + hierarchy B0/B1/B2 + per-config & global policy gaps,
             manifest_sha256.
```

## 7. Execution

Long non-interactive process (SLURM, NOT `nohup &`), conda env **eeg2025** (has torch; base lacks it), pinned to a node
whose kernel matches the env-lock platform (mirror the v3 run #002 sbatch: `nodecpu05`-class). Exit 0 before
interpreting; killed/partial ⇒ operationally aborted. Result commit includes this run plan,
`results/acar_v4_dev_exploration_001/` (RESULT.json + manifest.json), the console/scheduler log, and the input record
digest summary.

## 8. Acceptance + next step

```
completes  ⇒ verdict ∈ {V4_DEV_CANDIDATE_FOUND_FOR_POSSIBLE_FREEZE, V4_DEV_NEGATIVE_NO_LOCKBOX}
killed     ⇒ OPERATIONALLY_ABORTED_NO_SCIENTIFIC_VERDICT
candidate  ⇒ next is to DRAFT ACAR_FROZEN_v4.md, freeze ONE candidate, tag a new V4 protocol — NOT external Arm B.
             Whether any external/held-out data may be consumed is a SEPARATE later decision.
negative   ⇒ record the negative V4 DEV exploration; lockbox stays sealed.
```

No score/loss/grid/comparator/threshold/seed change after the run starts. Any continuation past this run is a new dated,
separately-tagged protocol. Never edits the frozen v2/v3 commits or tags.
