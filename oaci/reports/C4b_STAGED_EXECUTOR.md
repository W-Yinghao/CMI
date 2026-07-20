# C4b — Resource-Aware Staged Executor (GPU-record / CPU-replay)

> **Execution-only optimization.** The staged executor is bit-identical to the monolithic pipeline; it
> moves the CPU-bound leakage scoring off the V100. This validation run is reduced-bootstrap pipeline
> validation, **not** confirmatory efficacy evidence.

C3 showed the full-bootstrap one-fold spent ~48 min training and **~5 h 38 m scoring leakage with the V100
idle** (88 % of wall-clock, CPU-bound). C4b splits a fold into a **GPU record stage** (train + extract every
feasible candidate's features/logits) and a **CPU replay stage** (select → lock → audit → finalize from the
stored artifacts), so the V100 is released before the long leakage phase.

## Design (all proven bit-exact on the CPU fixture)

| commit | piece |
|--------|-------|
| `e44ea1a` | **C4a** — optimization RNG decoupled from the bootstrap/eval config (a bootstrap change no longer perturbs training) |
| `e2b7e4f` | **C4b-1** — replay store: record the two keyed GPU forwards (`extract_frozen_features`, `predict_checkpoint`), replay on CPU |
| `5db8c92` | **C4b-2a** — role-segregated store (`feat:source_train` / `feat:source_audit` / `logits:*`) → target isolation |
| `957feb6` | **C4b-2b** — staged level executor (Stage A over-extracts every feasible candidate; Stage B resumes from store) = `run_level_complete` bit-for-bit |
| `7e8af63` | **C4b-2c** — two-phase persistence boundary (Stage A persist → Stage B rebuild + resume) |
| `180af5c` | **C4b-3** — two-job CLIs + SLURM (Phase A V100 records & chains Phase B CPU) |
| `38b540d` | **C4b-3 hotfix** — Phase B loads the EXACT Phase-A fold (never re-load the data) |

**Bit-exactness is enforced by the test suite** (CPU fixture, both runs CPU-extracted): the staged
`level_result_hash` and `fold_result_hash` equal the monolithic at both deletion levels; the selected
checkpoints, selection & audit leakage, predictions and metrics all match; Stage A extracts exactly ERM +
every feasible unique candidate with no capping; `target_fit_ids` stays empty (selection structurally only
requests `feat:source_train`, so it cannot open the audit or target stores). 16 staged tests; 648 total.

## Failure found & fixed (determinism mismatch)

The first staged run failed in Phase B: `"Phase B fold scope hash does not match Phase A"`. Phase B
re-loaded the BNCI data on a different (CPU) node and the offline MNE/scipy filter+resample is not
bit-reproducible across nodes → a different `target_tensor_hash` / `fold_scope_hash`. **Fix (`38b540d`):**
Phase A persists the fold (heavy `load_result` nulled); Phase B loads that exact fold instead of re-loading
the data. The re-run passed.

## Validation run (BNCI2014_001, target subject-001, seed 0, reduced bootstrap)

| item | value |
|------|-------|
| commit | `38b540d` |
| CPU CI | job `876815`, exit 0, **648 tests ALL-PASS** |
| Phase A (GPU record) | job `876816`, node09, Tesla V100S-PCIE-32GB, exit 0 |
| Phase B (CPU replay) | job `876980`, CPU partition, exit 0 |
| **V100-held wall-clock (Phase A)** | **≈ 1 h 40 m** (12:40 → 14:19) — train both levels + GPU-prefetch |
| **CPU-replay wall-clock (Phase B)** | **≈ 2 h 43 m** (14:19 → 17:03) — leakage scoring + artifact, **no GPU** |
| transient staging size | **3.10 GB** (`fold.pkl` 176 MB + per-level trained + stores) |
| feasible unique checkpoints | level 0 = **38**, level 1 = **60** (ERM + every feasible Stage-2, no cap) |
| stored feature/logit arrays | level 0 = **190**, level 1 = **300** (= candidates × 5 roles) |
| leakage workers | 16 (`process_bootstrap_replicate`, 1 thread each) |
| artifact deep verification | **OK** — 548 indexed files, 202 checkpoints, 14 plans |
| `target_fit_ids` | **∅** |
| summary matches memory | **True** |
| `artifact_scientific_hash` | `2ac2b7c18c3faeda…` |
| `artifact_pure_science_hash` | `43bd6edb25d40b7b…` |

### The resource win

| | monolithic (full bootstrap, 876133) | staged (this run, reduced) |
|--|--|--|
| V100-held | **6 h 26 m** | **≈ 1 h 40 m** (Phase A only) |
| leakage phase | on the V100 (idle GPU) | on CPU (Phase B), **no GPU held** |

Phase A is **bootstrap-independent** (it trains + extracts; it never scores), so its ~1 h 40 m V100 hold is
the same at full bootstrap — only Phase B (CPU) grows. So a **full-bootstrap** staged one-fold would hold
the V100 for ~1.7 h instead of 6.4 h, with the ~5 h leakage entirely on CPU.

### Monolithic-vs-staged equality

Bit-exactness is **proven by the CPU staged-executor tests** (`test_staged_level_result_hash_matches_monolithic`,
`test_staged_two_phase_fold_matches_monolithic`): identical `fold_result_hash` and every level/selection/
audit/prediction/metrics hash. On real data the staged path runs the *same* extraction code with the *same*
keys, and the artifact structure (548 files / 202 checkpoints / 14 plans) matches the monolithic. A direct
same-commit real-data monolithic↔staged hash comparison was **not** run here (it would be a second ~3 h+
V100 job); it can be added if a hardware-level cross-check is wanted. The C3 monolithic (876133) is a
different commit (pre-C4a), so its hashes deliberately do not reproduce.

## Status & next

- ✅ The staged two-job runs end-to-end on real data to a deep-verified artifact (`target_fit ∅`), with the
  V100 released during the leakage scoring. The hard part — a GPU/CPU split bit-identical to the monolithic,
  target kept forward-only — is done and test-proven.
- ⏭ Optional: a **full-bootstrap** staged one-fold for the real full-budget timing (Phase A ~1.7 h V100,
  Phase B ~5 h CPU). Then resume the milestone order: **BNCI2014_001 LOSO** (9 targets, seed 0, full
  bootstrap, staged) → k1/k2 aggregation + decision → BNCI2014_004 → multi-dataset sweep.
