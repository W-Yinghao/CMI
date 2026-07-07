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
  artifact, exit 0) if the local MOABB cache is unavailable.

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

---

**Scope.** This protocol governs how results are *reported and bounded*. Tier 0 is live
(`run_counterexamples.py`); the audited evaluation bridge (§6), the real-EEG audited pilot
(§7, `run_real_audited.py`), the audited mini-grid + validator (§8), and the expanded grid +
statistical digest (§9) are live; a full multi-dataset table remains future work under this same
audited discipline.
