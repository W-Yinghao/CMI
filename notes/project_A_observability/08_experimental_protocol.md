# Project A — Experimental Protocol

> Experiments in Project A are **observability audits**, not just performance comparisons: each
> reports *observed information + contracts + estimator → allowed claim*. This protocol fixes the
> per-experiment claim ledger, the three experiment tiers, the required experiments, and the
> hard reporting rules that forbid reporting R0/R1 metrics as target guarantees. Uses
> `06_oaci_identifiability.md` (`OA-0`, checkable-vs-assumed) and `05_csc_shift_calculus.md`.
> Naming per `00_repo_audit.md §5`. No training-code changes are implied by this file.

## 0. Purpose

Every experiment answers: *given the observed regime `R`, the declared contracts `C`, and the
estimator, which claims are admissible under `OA-0`?* Performance numbers are inputs to that
question, not the answer.

## 1. Required claim ledger

Every experiment must report (machine-checkable fields in §5):
```
Regime:                 R0 | R1 | R2
Observed coordinates:   (which parts of O_R are actually used)
Contracts invoked:      (subset of C1..C12)
Checkable contracts:    (invoked ∩ checkable in R)
Uncheckable contracts:  (invoked \ checkable in R  — assumed, not evidenced)
Identifiable estimand:  (the T that (R,C) pins down under OA-0; or an identified set)
Estimator:              (f / critic; C5 fidelity diagnostics)
Failure certificates:   (which CE/P0 fires if a contract breaks)
Forbidden claims:       (the overclaims this design must NOT make — 05 §6)
```

## 2. Experiment tiers

- **Tier 0 — exact discrete certificates.** `counterexamples/run_counterexamples.py`; **no
  training**; validates the non-identifiability *arithmetic*. These are proofs.
- **Tier 1 — `h2cmi` simulator illustrations.** `h2cmi.data.eeg_simulator`; vary
  `prior/concept/montage/noise/label_mechanism`. **Oracle labels (`ystar`) may be used only for
  *validation*, never as R1 evidence.** Illustrations, not proofs.
- **Tier 2 — real EEG audit.** source-only LOSO → **source claim only**; target-unlabeled →
  **marginal / support / prior-under-contract only**; minimal-paired → **paired transport /
  bounded risk only if anchors satisfy C8 ∧ C11**.

## 3. Required experiments

- **E0 — exact certificates.** CE-R0-1/2/3, CE-R1-1/2, CE-C1-1, CE-MP-1, CE-C11-1, CE-MONO-1
  (all in Tier 0; all currently PASS).
- **E1 — `TOS-1` source-only audit.** Show source LOSO metrics **cannot** be reported as
  target-specific gain (regime R0; forbidden claim = target gain). Certificate: `TOS-1` / CE-R0-2.
- **E2 — R1 prior-identifiability stress.** Vary the rank of the mixture matrix `B` and the
  support overlap; show `π_T` recovery **only** when C1 ∧ C2 ∧ C3 hold (`TU-1`), and
  non-recovery at rank-deficiency (CE-R1-2) or support failure (CE-C1-1).
- **E3 — R1 concept non-identifiability.** Same target `X`/`Z`, different target labels; show any
  R1-only procedure sees the **same** observation (`TU-2` / CE-R1-1); forbidden claim = "concept
  detected from unlabeled target".
- **E4 — R2 transport-rank study.** Vary anchor count `k` and dimension `p`. If `k<p` (or
  anchors rank-deficient / invalid), the transform is non-identifiable (CE-MP-1 / CE-C11-1); if
  the anchor matrix is full-rank and the transform family low-dimensional, the transform is
  identifiable up to estimation error (`MP-1`).
- **E5 — safety-gate falsification.** Use CE-R0-2 and simulator analogues to show the source
  gain sign does **not** identify the target gain sign (C9 / measurement→control gap).

## 4. Reporting rules (hard)

**Balanced accuracy** may be reported as:
- a **source** validation metric under R0;
- an **oracle** simulator check, *explicitly marked oracle* (Tier 1);
- a **target-label** metric only under R2 / evaluation-only held-out labels.

> It must **never** be reported as identifiable from R1 unlabeled target data, nor may an R0
> source metric be reported as a target risk / gain / concept guarantee.

**Leakage metrics** (`I(Z;D|Y)`, `I(Y;D|Z)`) may be reported as **diagnostics only**, never as
risk or accuracy guarantees (CSC-R2; C5 governs whether the measured value is even the population
value). `I(Y;D|Z)` is reported as a predictive-insufficiency residual, never as "concept detected"
outside R2 + C4 ∧ C6 ∧ C5.

## 5. Minimal output format

Each run should save an observability report (`observability_report.json` + `.md`) with fields:
```
regime
contracts_invoked
checkable_contracts
uncheckable_contracts
identifiable_estimand          # or identified_set; null unless pinned down under OA-0
reportable_metric              # oracle/eval-only benchmark number: reportable without being identifiable
observation_used
estimator                      # incl. C5 fidelity diagnostics (stepA_dom_acc, probe gap)
certificate_passed             # which CE/P0 checks ran and their result
forbidden_claims_checked       # the 05 §6 list, each asserted NOT made
oracle_fields_used_for_validation_only   # e.g. ystar in Tier 1 — flagged, never as evidence
```
The report makes the *claim boundary* auditable independently of the accuracy table: a reviewer
can verify that every asserted estimand is licensed by `(regime, contracts)` under `OA-0`, and
that every forbidden claim was explicitly checked and not made.

## 6. Audited evaluation bridge

Real EEG benchmark labels are allowed for **evaluation**, but they are **not part of the R0/R1
adaptation observation operator** (`06 §2`). A metric may therefore be **reportable** (an
oracle/evaluation-only benchmark number) without being **identifiable** (a target functional
pinned down by the regime under `OA-0`). The audit layer separates these
(`Verdict.reportable` vs `Verdict.identifiable`); the bridge
(`h2cmi/observability/eval_bridge.py`) turns `h2cmi/eval/harness.py` outputs into audited claims:

- **strict-DG target bAcc** → `R0`, **oracle/evaluation-only** (reportable,
  `identifiable_estimand = null`);
- **offline / online TTA target gain / bAcc** → `R1`, **oracle/evaluation-only** (the measured
  gain is not an `R1`-identified gain);
- **offline-TTA target prior** → `R1`, identifiable **only under `TU-1` (C1∧C2∧C3)**, else rejected;
- **leakage** → a diagnostic, never a target-risk guarantee;
- the audit rejects any claim whose declared `regime` conflicts with its observed coordinates:
  target **labels** under `R0`/`R1` are allowed only with an oracle/eval mark, while target
  **data** and **anchors** under `R0`/`R1` are rejected **regardless** of the oracle mark
  (anchors are R2-only); an **unregistered** coordinate is itself a mismatch (deny-by-default);
- **no target metric may be reported without an `ObservabilityReport` entry.**

Executable: `h2cmi/tests/test_observability_eval_bridge.py` (acceptance tests) and
`notes/project_A_observability/examples/make_audited_eval_bridge_smoke.py` (smoke → JSON/MD).

## 7. Real EEG audited pilot

A real EEG number is **admissible only if accompanied by**:
- `raw_results.json` — the harness output (incl. `per_domain_pi_T` / TTA diagnostics);
- `observability_report.json` + `observability_report.md` — the audited claim ledger;
- `run_manifest.json` — dataset / subject / seed / config / environment provenance.

Rules for the pilot:
- Target metrics computed with held-out target labels are **evaluation-only** (oracle;
  `identifiable_estimand = null`) unless the regime is R2 and the labeled slice is declared.
- The offline-TTA **target prior** is **reported** (its `pi_T` estimate is evidence) but **not
  claimed identified** — no TU-1 contracts are asserted for a pilot, so the prior claim is
  rejected-but-flagged (`conclusion=false`), keeping the report clean *and* honest.
- The first pilot is **one dataset, one target subject, one seed** — an interface and
  claim-boundary validation, **not a performance claim**. It gracefully **skips** (legal skip
  artifact, exit 0) if the dataset load fails — cache unavailable **or** the dataset is invalid for
  the loader/paradigm (e.g. `BNCI2015_001` is not a left/right-hand task, so `LeftRightImagery`
  rejects it — a legal skip with that exact reason, not a missing cache).

Runner: `h2cmi/run_real_audited.py` (`--dataset synthetic` validates the whole path
self-contained; `--dataset <MOABB name>` loads offline). Wrapper:
`notes/project_A_observability/examples/run_real_eeg_audit_pilot.py`. Batch:
`scripts/project_A_real_eeg_audit_pilot.slurm`
(real GPU runs belong on SLURM, not the login node). Contract test:
`h2cmi/tests/test_eval_bridge_harness_contract.py`.

## 8. Audited real-EEG mini-grid

Step 8 expands the single pilot to a small **audited grid** — still **not a SOTA claim**:
- one dataset: **BNCI2014_001**;
- target subjects **1, 2, 3**; seeds **0, 1, 2**; fast training, modest epochs;
- **align factor: subject** (single-site MOABB → aligning on `site` is a no-op; the manifest
  flags `alignment_factor_degenerate`).

It validates:
1. repeated real runs each produce a **clean** `ObservabilityReport` (0 forbidden violations);
2. target metrics remain **oracle/evaluation-only** (`identifiable_estimand=null`);
3. prior estimates are **reported but not claimed identified** unless TU-1 contracts are declared
   (rejected, `conclusion=false`);
4. a **tracked summary digest** lets a reviewer check every run's claim boundary **without**
   committing raw training outputs.

Mechanics: `h2cmi/run_real_audited_grid.py` (loads once, runs each `target×seed`) →
`h2cmi/observability/validate_results.py` (validates each run's claim boundary +
`result_index.py` aggregation) → **tracked** digest at
`notes/project_A_observability/results_summaries/*.json/.md`. Raw run directories stay in
**gitignored** `notes/project_A_observability/results/`. Batch:
`scripts/project_A_bnci2014_001_minigrid.slurm`. Tests:
`h2cmi/tests/test_observability_result_index.py`.

## 9. Expanded audited BNCI grid

Step 9 expands Step 8 to a **full single-dataset grid** (still **not** a SOTA comparison — the
purpose is stability + claim-boundary compliance at scale):
- **BNCI2014_001** subjects **1–9**; target subjects **1–9**; seeds **0–2**; epochs **50**;
  align factor **subject** → **27 cells**.

Mechanics: `h2cmi/run_real_audited_grid.py` gains `--resume` / `--overwrite` /
`--shard-index` / `--num-shards` (deterministic sharding for a SLURM array, idempotent re-runs);
`validate_results.py` gains `--expected-targets` / `--expected-seeds` (grid-completeness:
`missing_cells` — a cell with no directory at all is **not** a legal skip). The digest gains a
descriptive **statistical layer** (per-target / per-seed / overall mean·std·min·max +
`offline_tta_harm_rate = P(gain<0)`) — no inferential statistics.

Batch: `scripts/project_A_bnci2014_001_expanded_array.slurm` (9-task array, one shard each) then
`scripts/project_A_bnci2014_001_expanded_validate.slurm` (a **separate** job — array tasks never
race to write the summary). Required tracked digest:
`results_summaries/step9_bnci2014_001_expanded_summary.{json,md}` with
`missing_cells=[]`, `all_forbidden_violations_empty=true`, `all_target_metrics_oracle_only=true`,
`all_target_metrics_identifiable_null=true`, `all_prior_claims_compliant=true`,
`no_unknown_estimands=true`. Tests: `test_real_audited_grid_plumbing.py`,
`test_observability_text_hygiene.py`.

## 10. Multi-dataset audited expansion

Step 10 adds additional MOABB motor-imagery datasets under the same audit discipline. It does **not**
chase a higher score on `BNCI2014_001`; its purpose is externality / dataset-transfer sanity, a
chance-normalized descriptive comparison, and claim-boundary compliance across **binary and 4-class**
settings simultaneously.

**Why normalize.** Raw balanced accuracy is not comparable across datasets with different class
counts (`BNCI2014_001` is 4-class; `BNCI2014_004` / `BNCI2015_001` are binary): a raw 0.40 is above
chance for 4-class and below chance for binary. Every ok run therefore records `n_classes` and the
**chance-normalized** metrics
`bAcc_excess = bAcc − 1/K`, `bAcc_excess_norm = (bAcc − 1/K)/(1 − 1/K)`,
`gain_norm = gain/(1 − 1/K)` (0 at chance, 1 at perfect, for any `K`).

**Reporting rules (hard).**
- Within-dataset tables MAY report raw bAcc.
- Cross-dataset aggregates MUST use normalized excess / normalized gain.
- It is forbidden to pool raw bAcc into one overall mean across datasets with different `n_classes`
  — the combiner refuses (`raw_bacc_overall_suppressed=true`).
- All R0/R1 target metrics remain oracle/evaluation-only with `identifiable_estimand=null`; a target
  prior without C1∧C2∧C3 stays `rejected_conclusion_false` (never a forbidden violation).
- No SOTA comparison.

**Mechanics.** `run_real_audited_grid.py` gains `--subjects all` / `--target-subjects all` (resolved
**after** load, so concrete ids are recorded) and writes a `grid_manifest.json` at the grid root
listing the resolved `expected_cells`. `validate_results.py` reads that manifest (or `--grid-manifest`)
to flag a **missing** cell (no directory at all) versus a legal skip (`status=skipped` + `skip_reason`),
and requires `n_classes` on every ok run. `combine_summaries.py` merges the per-dataset digests into one
chance-normalized `overall_normalized` block.

**Batch.** `scripts/project_A_step10_moabb_multidataset_gpu_array.slurm` (18-task array = 2 datasets ×
9 shards) then `scripts/project_A_step10_moabb_multidataset_validate.slurm` (a **separate**
`--dependency=afterok` job that validates each grid, regenerates the Step-9 reference digest under the
normalized schema, and combines). Required tracked digests: per-dataset
`results_summaries/step10_<slug>_summary.{json,md}` and the combined
`results_summaries/step10_moabb_multidataset_summary.{json,md}`.

**Acceptance.** ≥1 additional dataset with `n_ok>0` (priority `BNCI2014_004`); every dataset digest
valid; `missing_cells=[]` unless `--allow-missing`; overall cross-dataset numbers normalized (raw
suppressed); all claim-boundary flags true. Tests: `test_observability_result_index.py` (normalized +
missing-`n_classes`), `test_real_audited_grid_plumbing.py` (grid_manifest + all-resolution),
`test_observability_multidataset_summary.py` (combiner refusal + normalized pooling).

## 12. Scientific exploration before manuscript writing

Step 12 does **not** write the paper. It studies the audited behavior scientifically:
- **offline-TTA harm attribution** (`harm_attribution.py`): per-run diagnostics split strictly into
  R0 source-only, R1 target-unlabeled, and oracle evaluation-only groups; missing diagnostics are
  reason-coded, never faked;
- **retrospective harm prediction** (`harm_predictor.py`): can R0 diagnostics predict which cells
  offline-TTA harmed, and does R1 add value? Leave-one-(dataset,target)-out logistic regression,
  scored against the 0.5 majority baseline;
- **minimal-paired phase transition** (`minimal_paired.py`): on a controlled simulator, how k
  labeled target trials move harm/gain from non-identifiable (k=0, R1) to a labeled-slice estimate
  (k>0, R2) under an iid sampling contract.

Hard rules (enforced by tests):
- the oracle offline-TTA gain/harm is an **evaluation label only** — it is the prediction target,
  never a predictor feature (denylist in `harm_attribution.ORACLE_KEYS`);
- an R0/R1 harm predictor is an **empirical retrospective** predictor, not target-gain/harm
  identifiability (TOS-1 / TU-2 stand);
- k=0 is the R1 non-identifiability boundary; k>0 is an R2 **labeled slice under an iid sampling
  contract**, never "full target risk identified";
- no SOTA claim; no manuscript in this step.

Pipeline: `scripts/project_A_step12_science_cpu.slurm` writes tracked `step12_*` digests; the
`science_dashboard.py` combines them into "what we learned / what remains unknown". Tests:
`test_observability_harm_attribution.py`, `test_observability_harm_predictor.py`,
`test_minimal_paired_phase_transition.py`, `test_observability_science_dashboard.py`.

## 13. Rich R1 diagnostics and real minimal-label curves

Step 13 tests whether Step 12's null harm-prediction result was caused by missing R1 diagnostics. It
adds richer **label-free** target-unlabeled diagnostics to the real runner and a per-trial oracle
prediction payload for R2 minimal-label analysis.

- **Rich R1 diagnostics** (`harness.py` prediction diagnostics + `run_real_audited.py`
  representation/prior diagnostics) land in a `r1_diagnostics` block of `raw_results.json`; missing
  ones are reason-coded in `r1_diagnostics_missing`. `harm_attribution.py` prefers this block and falls
  back to the legacy per-domain computation for older runs.
- **Per-trial oracle predictions** (`per_trial_oracle_predictions`) are stored evaluation-only and read
  ONLY by `real_minimal_labels.py`.

Hard rules (tests enforce):
- R1 diagnostics use target X / predictions only, never target labels (label-free; a permutation of
  labels cannot change them).
- Oracle per-trial labels are evaluation-only; a per-trial key is never an R0/R1 feature.
- Real minimal-label curves: k=0 = R1 non-identifiable; k>0 = R2 labeled slice under an iid sampling
  contract, never full-target identification.
- Any R1 harm predictor remains retrospective empirical prediction, not identifiability. No SOTA.

Reruns the two ok datasets only (no new datasets): `BNCI2014_001` + `BNCI2014_004`, all subjects/all
targets × seeds 0-2 × 50 epochs → `results/step13_<DATASET>_diagnostics`. Pipeline:
`scripts/project_A_step13_diagnostics_gpu_array.slurm` then
`scripts/project_A_step13_science_cpu.slurm` (harm attribution → harm predictor → real minimal-label
curves → Step-13 dashboard). Tests: `test_observability_harm_attribution.py` (rich R1 + label-free),
`test_real_minimal_label_curves.py`, `test_observability_science_dashboard.py` (Step-13 mode).

## 14. Metric semantics and power

Step 14 repairs two Step-13 over-readable metrics (no rerun, no new data):
- **Real minimal-label curves** decompose the ambiguous "harm_sign_accuracy" into **coverage**
  (`decisive_rate`), **unconditional_correct_rate**, and **conditional_accuracy_given_decisive**.
  **k=0 accuracy is NULL** (no estimator licensed under R1), not 0.5. On real data the burden is
  coverage, not accuracy.
- **Harm predictor** gains a configurable `--step-label` (fixes a Step-12 provenance label), a larger
  `--n-perm` permutation null reported at **p90/p95/p99**, and a `--robust-margin` rule
  (`bAcc > perm_null_p95 + margin`). `harm_power.py` reports the minority-class limitation and the
  **minimum detectable bAcc** below which a result is indistinguishable from the overfitting null.

Rules (unchanged boundary): permutation-null significance is empirical retrospective evidence, NOT
target-gain identifiability; k>0 curves are R2 labeled slices under an iid sampling/coverage contract;
oracle labels never enter R0/R1 features; no SOTA; no manuscript. Pipeline reuses the Step-13 raw
diagnostics grids (CPU only). Tests: `test_observability_harm_predictor.py` (step-label, perm
percentiles, robust margin, power warning), `test_observability_harm_power.py`,
`test_real_minimal_label_curves.py` (decomposition, k=0 null), `test_observability_science_dashboard.py`.

## 15. Coverage-aware harm-control policies

Step 15 evaluates R2 minimal-label policies that choose per target among **adapt / identity / abstain**
(`harm_control.py`), exploiting the Step-14 finding that a small labeled slice gives high-precision but
low-coverage harm-sign calls. Policies: always_identity / always_adapt / plugin_sign /
ci_adapt_only_{abstain,identity} / ci_three_way, plus an evaluation-only `oracle_full_target` upper
bound that is **never deployable**.

Rules (tests enforce): k=0 is R1 non-identifiable → label-based policies abstain; k>0 is an R2 labeled
slice under an iid sampling/coverage contract, not full-target identification; the oracle policy is
never selected as best-deployable; oracle labels are used only inside the R2 slice / for evaluation.
The best deployable policy is chosen by a **predeclared** rule (harm-among-adapt ≤ 0.05, maximize
adaptation coverage, tie-break minimize missed benefit), not post-hoc. No SOTA, no manuscript, no new
datasets — reuses the Step-13 raw diagnostics grids (CPU). Tests:
`test_harm_control_policies.py`, `test_observability_science_dashboard.py` (Step-15 mode).

## 16. Benefit anatomy and sequential label acquisition

Step 16 analyzes oracle benefit rarity/stability and evaluates budgeted R2 sequential policies (Step-13
raw per-trial predictions only). `benefit_anatomy.py` reports where beneficial cells are (rarity,
per-(dataset,target) sign consistency, gain distribution) — **oracle/evaluation-only**, not R0/R1
observable. `sequential_harm_control.py` evaluates seq_ci_three_way / seq_ci_adapt_only /
seq_plugin_confirm that acquire labels batch by batch and stop when the paired-gain CI is decisive; the
predeclared best rule adds a minimum-labels objective. `policy_frontier.py` reports whether any
deployable policy meets harm thresholds 0.05 / 0.10 / 0.20 / 0.50. Rules (tests enforce): full-target
labels are evaluation-only; k>0 is an R2 labeled slice under an iid sampling contract, not full-target
identification; the oracle policy is never deployable; `budget=full` is a full-label calibration policy,
not the oracle. The Step-15 `best_deployable_ci_attempt` field is renamed `best_label_based_attempt`
(plugin_sign is not a CI policy). No SOTA, no manuscript, no new datasets. Tests:
`test_benefit_anatomy.py`, `test_sequential_harm_control.py`, `test_policy_frontier.py`,
`test_observability_science_dashboard.py` (Step-16 mode).

## 17. Estimand-consistent harm control

Step 17 evaluates policy frontiers **separately** for ordinary accuracy gain and balanced-accuracy
gain (the mismatch surfaced in Step 16: bAcc-benefit 0.0926 vs accuracy-benefit 0.1481). A policy must
declare which gain estimand it controls; **cross-estimand conclusions are forbidden** (an accuracy-gain
policy is never reported as controlling bAcc gain). `estimand_consistency.py` estimates both gains from
a labeled slice under two sampling contracts — iid (bAcc abstains / marks `missing_class` when a class
is absent) and class-balanced calibration (contract **C13**, abstains when `k < n_classes`).
`estimand_frontier.py` reports harm/coverage frontiers per (estimand, sampling), never mixed. Rules
(tests enforce): k=0 abstains (R1 non-identifiable); k>0 is an R2 labeled slice under a sampling
contract, not full-target identification; the oracle policy is never deployable; a bAcc/class-balanced
policy declares C13. No SOTA, no manuscript, no new datasets (reuses Step-13 raw). Tests:
`test_estimand_consistency.py`, `test_estimand_frontier.py`, `test_observability_science_dashboard.py`.

---

**Scope.** This protocol governs how results are *reported and bounded*. Tier 0 is live
(`run_counterexamples.py`); the audited evaluation bridge (§6), the real-EEG audited pilot
(§7, `run_real_audited.py`), the audited mini-grid + validator (§8), the expanded grid +
statistical digest (§9), the multi-dataset audited expansion + chance-normalized digest
(§10), the Step-12 scientific exploration (§12), the Step-13 rich-R1 diagnostics + real
minimal-label curves (§13), the Step-14 metric-semantics + power repair (§14), the Step-15
coverage-aware harm-control policies (§15), the Step-16 benefit anatomy + sequential
label-acquisition frontier (§16), and the Step-17 estimand-consistent harm control (§17) are live. A
full multi-dataset SOTA table and manuscript writing are out of scope.
