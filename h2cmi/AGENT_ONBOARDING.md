# H2CMI — agent onboarding map

Orientation for another agent (e.g. Codex) taking over or auditing this work. Fact-only; every claim is
tied to a commit or file you can open. **Not** manuscript prose — the paper is owner-controlled (see
"Role boundary" below).

## 1. What this is

`h2cmi/` is a self-contained EEG domain-adaptation / test-time-adaptation (TTA) package (isolated from the
AAAI `cmi/` code). Broad program map: `h2cmi/PROJECT_OVERVIEW.md`. The *recent* work (this branch)
algebraically decomposes and stress-tests one specific failure: whether an estimated class prior should be
used as the **decision** prior under **balanced accuracy** (BA). The four-branch protocol decodes
`{identity, joint-geometry} × {uniform, π_J}` and splits the "joint" delta as `full = G + P + I_int`
(geometry `G` @uniform, fit-prior `P` @identity, interaction). Definitions live in
`h2cmi/eval/p0_eval.py` (`eval_unit_p0`).

## 2. Repo coordinates

- remote `git@github.com:W-Yinghao/CMI.git`; work branch **`exp/h2cmi-wave0-mechanism`** (HEAD `f3c9dd3`,
  pushed).
- **Terminal baseline:** tag `h2cmi-review-p0-terminal` → commit `5bc9bf0` (the REVIEW_P0 result of
  record). Everything after `5bc9bf0` (38 commits) is Wave 0 + Wave 1, built *on top of* the frozen
  terminal — the terminal is read-only, never re-tagged.
- Datasets: sleep = Sleep-EDF Cassette (75 paired-night subjects); MI = BNCI2014_001, BNCI2014_004,
  Lee2019_MI (V2P cross-session pairs). Env: conda `icml` (`python -m h2cmi.<module>`); GPU via SLURM.

## 3. Program structure (each phase: frozen pre-reg → runner → analyzer → neutral results doc)

| phase | question | pre-reg | runner(s) | analyzer | results | one-line result |
|---|---|---|---|---|---|---|
| REVIEW_P0 (terminal) | fix 2 reviewer P0 mismatches (decision-prior confound; V2P pool) | `REVIEW_P0_FROZEN.md` | `run_w2_p0.py`, `run_w1_p0.py`, `run_v2p_weighted.py`, `p0_source.py` | `analyze_p0_final.py`, `audit_p0_provenance.py` | `REVIEW_P0_RESULTS.md` | W2 harm = decision prior (P=−0.144), geometry ~neutral; tagged `5bc9bf0` |
| W0.1 | close the W2 reproducibility hole + admissible confusion | `WAVE0_FROZEN.md` | `run_w2_wave0.py` (deterministic eval-only reuse) | `analyze_wave0_w2.py` | in `wave0_w2.report.json` + `WAVE0_MANUSCRIPT_NOTES.md` | 27/27 self-replay; G/P re-confirm to 4dp; per-stage confusion admissible |
| W0.2 | fixed-reservoir prevalence utility | `WAVE0_FROZEN.md` | `run_v2p_wave0.py` | `analyze_wave0_v2p.py` | `W0.2_RESULTS.md` | FRSC not prevalence-invariant; displacement ≠ utility |
| W0.3 | same-session decision-prior mechanism | `W0.3_MECH_APPENDUM.md` | `run_w2_wave0_null.py`, `run_prior_decomp.py` | `analyze_wave0_priordecomp.py` | `W0.3_RESULTS.md` | **metric-prior mismatch dominates** (oracle ρ_E wrong BA prior) |
| W0.4 | is the harm a small-batch artifact? | `W0.4_MECH_APPENDUM.md` | `run_w2_wave0_batchsweep.py` | `analyze_wave0_batchsweep.py` | `W0.4_RESULTS.md` | larger n **exposes** (not fixes) the harm (ΔP_J −0.044) |
| W0.5 | is the decision prior metric-dependent? | `WAVE0_FROZEN.md` | (analyzer-only, reuses W0.2 + prior-decomp) | `analyze_wave0_metricswitch.py` | `W0.5_RESULTS.md` | metric switch real but **specification-dependent** |
| W1.geometry | do diagonal-latent operators cover real sensor-geometry shifts? | `W1_GEOMETRY_FROZEN.md` | `run_v2p_geometry.py` | `analyze_wave1_geom.py` | `W1_GEOMETRY_RESULTS.md` | falsification **fails** → diagonal family adequate; FRSC/joint weak, pooled/latent_im_diag match full-cov |

Navigation shortcuts already in the repo: **`WAVE0_INDEX.md`** (per-wave result/insert commits + claims
checklist), **`WAVE0_EVIDENCE_PACKET.md`** (fact-only numbers/tables/checksums), and the memory file
`h2cmi-wave0-mechanism.md` (project narrative). Read those first.

## 4. Where the data lives

- **Committed** (readable now, on the branch, under `h2cmi/results/`): every `*_RESULTS.md`, `*_FROZEN.md`,
  `*_APPENDUM.md`, the `*.report.json` (aggregated stats), the `w0*/w1*_*.csv` tables, and `*.sha256`
  checksums (39 tracked artifacts).
- **NOT committed** (gitignored, on disk under `results/h2cmi/`): raw per-unit JSONL (`wave0_*/`,
  `wave1_geom/`, `p0_*_all.jsonl`), the source-model bundles (`p0_w2_bundles`, `v2_bundles`,
  `p0_v2pw_bundles`), and the sleep cache (`p0_sleep_cache`). To re-run you need these bundles + SLURM GPU;
  to *read results* you only need the committed reports.

## 5. Conventions / infrastructure (enforced; break these and results are untrustworthy)

- **Frozen pre-reg BEFORE compute.** Each `*_FROZEN.md` / `*_APPENDUM.md` is committed before its runner
  aggregate is viewed (W0.3/W0.4 appendums were frozen after a 1-unit probe, before the aggregate).
- **Fan-out gate:** subjects addressed by **real id**, never bench index (`wave0_fanout.py`, regression
  test `tests/test_wave0_fanout.py`, 5/5). `bench_index` is never an identity.
- **Probe-gate:** never launch a refilling babysitter without a 1-unit probe passing QC (determinism,
  invariants, "perturbation actually hurts", real-id). Coverage guards use a proper Python unique count
  (not `sort -un` on strings).
- **Two-commit discipline:** result-only commit first, then (only if asked) staged interpretation.
- **Provenance:** reuse frozen bundles only after strict validation (`p0_source.get_source_p0`);
  `ProvenanceError` → STOP the unit. Weighted-estimator tests `tests/test_weighted_tta.py` (10/10).
- **Determinism:** `eval_unit_p0` reproduces bit-identical predictions under
  `use_deterministic_algorithms` (W0.1 self-replay 27/27). Detach embeddings before any transform fit
  (frozen encoder must get no backward).

## 6. How to reproduce / verify a result

```bash
conda activate icml
cd <repo>
# re-derive any aggregate from the committed raw (if the gitignored JSONL/bundles are present):
python -m h2cmi.analyze_wave0_priordecomp --out /tmp/chk.json   # W0.3 (needs results/h2cmi/wave0_priordecomp/)
python -m h2cmi.analyze_wave1_geom       --out /tmp/chk.json    # W1  (needs results/h2cmi/wave1_geom/)
# verify committed artifacts unchanged:
cd h2cmi/results && sha256sum -c w03.sha256 w04.sha256 w02_w05.sha256 w1g.sha256
```
The `*.report.json` under `h2cmi/results/` are the authoritative aggregates; the `*_RESULTS.md` tables are
copied from them. Each results doc lists its exact QC (residuals ≤ 5.6e-17, invariance checks,
main-consistency 0).

## 7. Role boundary (important for a continuing agent)

This package is **experiments-only**: frozen pre-registrations, runners/analyzers, QC gates, neutral
`*_RESULTS.md`, and CSV/JSON/figure-ready artifacts. Do **NOT**: draft/edit the H2CMI manuscript, create
`h2cmi/paper/`, write abstract/intro/discussion/conclusion, or apply the staged inserts. The
`MANUSCRIPT_INSERTS_*.md` files are archived audit trail only — see
`MANUSCRIPT_INSERTS_ARCHIVED_DO_NOT_APPLY.md`. Manuscript narrative/positioning is owner-controlled.

## 8. Gotchas fixed along the way (so you don't repeat them)

- `_predict_transform` / `_predict_generative` return `[N,K]` **probabilities**; recall/ordinary-accuracy
  code must `.argmax(1)` first (bit us in W0.2 `60db118` and W1 `692aaf7`).
- `require_clean_git` ignores the whole `results/h2cmi` compute dir; a stray file outside it (or a bash
  `>`-redirect artifact from a `2>3`/`0>1` in an unquoted string) dirties the tree and blocks every job.
- MOABB offline `get_data(subjects=<subset>)` can silently return only 1 subject → always full-load, filter
  in-loop (`93ecfdc`).
- A babysitter whose "done" check is "file non-empty" will mark partial output done → check completeness
  (expected unit count), and never resubmit-loop a crashing job (probe-gate first; a 15h crash-loop
  happened once, `82e411a`).

## 9. Open state

Wave 0 (W0.1–W0.5) + W1.geometry are complete and pushed. Deferred (not started, would each need their own
frozen pre-reg → probe → babysitter → results doc): Lee cross-session, P300 panel. No manuscript work
pending on this side.
