# Project A — Paper-Preparation Package

This is a **paper-preparation package, not the final manuscript.** It indexes Project A's theory,
audit code, counterexamples, and real-EEG audited digests into a reviewer-readable form so a
manuscript can be assembled from stable, cross-referenced parts.

Nothing here introduces a new claim. Every entry points to an existing artifact in the repository
(theory notes `00`–`08`, the `h2cmi/observability/` audit layer, the counterexample runner, or a
tracked digest under `results_summaries/`). Where a number appears it is copied from a tracked
summary JSON, not recomputed here.

## Contents

| File | Role |
|---|---|
| `00_claims_and_contributions.md` | Thesis, contributions C1–C6, and an explicit "what we do NOT claim" list |
| `01_outline.md` | Proposed manuscript skeleton (abstract + sections, with source files per section) |
| `02_related_work_map.md` | A *map* of the relevant literature areas and the gap Project A addresses (not a bibliography) |
| `03_theorem_table.md` | Theorem ledger OA-0 / MONO-1 / TOS-1 / TU-1 / TU-2 / MP-1 / PD-1 / ID-1 |
| `04_contract_table.md` | Canonical contract taxonomy C1–C12 with checkability and overclaim-blocked columns |
| `05_experiment_table.md` | Every tier-0 certificate and real-EEG audited grid, with its tracked file and claim boundary |
| `06_results_digest.md` | Reviewer-readable narrative of the audited digests (values copied from tracked summaries) |
| `07_limitations_and_claim_boundary.md` | Limitations + the allowed/forbidden claim boundary |
| `figures/figure_plan.md` | Planned figures |
| `tables/table_plan.md` | Planned tables |

## Provenance

- Theory: `notes/project_A_observability/00`–`08`.
- Audit layer: `h2cmi/observability/` (schema, registry, audit, eval_bridge, report, result_index,
  validate_results, combine_summaries).
- Counterexamples: `notes/project_A_observability/counterexamples/run_counterexamples.py`.
- Real-EEG digests: `notes/project_A_observability/results_summaries/` (raw run dirs are gitignored
  under `results/`).

## Scope

No SOTA claim. Target metrics are oracle/evaluation-only. Exact counterexamples are the proof
layer; the real-EEG grids illustrate the audit discipline and TTA fragility, they do not prove a
theorem.
