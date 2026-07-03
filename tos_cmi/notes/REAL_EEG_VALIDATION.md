# Real-EEG multi-dataset validation of the TOS-CMI conclusions

Branch `tos`. Purpose: test whether the 2a (BCI-IV-2a, 9 subjects) frozen-feature conclusions replicate on
additional real EEG datasets — **not** on synthetic. Status: **in progress** (big-N dumps running). This file
is the validation-branch provenance / run manifest; the paper's claim contract stays in
`paper/claim_evidence_table.md` and is **not** edited until the big-N readout is in.

## Datasets (preflight-confirmed; offline from the datalake)

| dataset | MOABB name | channels | classes | chance bAcc | subjects | note |
|---|---|---|---|---|---|---|
| BCI-IV-2a | BNCI2014_001 | 22 | 4 | 0.25 | 9 | original paper dataset |
| BCI-IV-2b | BNCI2014_004 | 3 | 2 | 0.50 | 9 | very low channel count |
| OpenBMI | Lee2019_MI | 62 | 2 | 0.50 | **54** | big-N |
| GigaScience | Cho2017 | 64 | 2 | 0.50 | **52** | big-N |
| High-Gamma | Schirrmeister2017 | 128 | 4 | 0.25 | 14 | channel-rich |
| ~~Stieger2021~~ | Stieger2021 | var (59/60) | 4 | 0.25 | 62 | **EXCLUDED**: inconsistent per-subject channel counts (MOABB concat fails); would need cross-subject channel intersection that changes the feature space. |

Confirmed set = **129 subjects** (14× the 9 on 2a).

## Bug-fix provenance (high priority — these gate the validity of every multi-dataset number)

- **`ede201a` — fold-cap bug (P0).** `run_eeg_frozen_pilot` and `run_capacity_factorial` hardcoded the LOSO
  target list to `[1..9]` (a 2a-ism) for `--target-subjects all`, so **every non-9-subject dataset dumped only
  its first 9 LOSO target folds** (Lee 9/54, Cho 9/52, HGD 9/14). Fixed to read the real MOABB `subject_list`.
  The existing 9-fold dumps were verified to use the **full source pool** (Lee 53 / Cho 51 / HGD 13 source
  subjects per fold), so only the missing folds are topped up — no re-run of valid folds.
- **`10c22e9` — degenerate-metric robustness.** `_deploy_file` (and `erasure_baselines` per-file) now catch a
  `LinAlgError: metric ill-conditioned` and **skip that fold with a `[SKIP]`/`[FAIL]` marker** instead of
  crashing the dataset. This is the guard that makes TSMNet-on-2b an explicit skip (below), not a silent drop.
- **`99b767d` — idempotent skip-existing + group/array submission** (`%4` concurrency, ≤15 tasks/group). Every
  banked fold writes its own npz; re-submitted tasks skip existing folds.

## Pre-registered acceptance criteria (fixed BEFORE the big-N readout)

**Cross-dataset comparison uses the PAIRED delta vs the full-`Z` frozen baseline, never absolute bAcc**
(chance differs: 2a/HGD 4-class → 0.25; 2b/Lee/Cho 2-class → 0.50).

**"No meaningful target gain" thresholds** (applied to each eraser's ΔbAcc vs full `Z`, target-subject-cluster
95% CI):
- ΔbAcc **upper** 95% CI `< +0.01`  → *no practically meaningful gain*.
- `+0.01 ≤` upper `≤ +0.02`         → *no statistically-supported gain, but not powered to exclude a small benefit*.
- upper `> +0.02`                    → **cannot** write "no gain".

**Per (dataset, backbone) verdict labels:**
- **C7/C8 erasure profile** (source-side): `CONFIRM` if LEACE drives linear subject decode ≈ chance with task
  preserved and an MLP residual, INLP destroys task, TOS/RLACE partial (i.e. the 2a pattern); `MIXED` if
  partially; `OVERTURN` if the profile qualitatively differs.
- **C12 target deployment**: `CONFIRM` if **no** principled eraser (LEACE/TOS/RLACE — excluding INLP-collapse
  and random-k) has ΔbAcc upper CI `> +0.02`; `OVERTURN` if some principled eraser has ΔbAcc **lower** CI
  `> +0.01` (a supported target gain); `MIXED` otherwise (borderline / seed- or backbone-inconsistent).
- **TSMNet metric validity**: `VALID` / `PARTIAL` (some folds skipped) / `DEGENERATE` (all folds skipped).

CIs: report target-subject-cluster (LOSO fold = one held-out target subject) **and** paired per-fold CI. For
low-N (HGD 14), CIs are wide → down-weight. Do **not** report only the mean.

## Readout log (fill as datasets land)

### BNCI2014_004 / 2b — ACCEPTED (PM), dataset #2, EEGNet-only

- **EEGNet — CONFIRM.** Track G: full subject 0.60/0.66 → LEACE 0.125 lin (=chance) / 0.25 MLP (residual),
  task 0.74→0.72; INLP subject→chance but task 0.74→0.55 (destroyed); TOS/RLACE partial. C12 deployment
  (target chance 0.5, full bAcc 0.652): **no meaningful target-bAcc gain** — LEACE ΔbAcc −0.012 [−0.023,
  **+0.001**] (upper CI +0.001 < +0.01 → no practically meaningful gain; point estimate negative but not
  strictly-negative), TOS −0.000, RLACE −0.002, INLP −0.061 (task collapse), random −0.003. NLL non-specific.
- **TSMNet — DEGENERATE, not evaluated.** Exact note (ledger/paper):
  > On BNCI2014_004 / 2b, the EEGNet branch is valid and confirms the 2a erasure/deployment pattern. The
  > TSMNet-210 branch is not evaluated because the 3-channel montage yields an ill-conditioned 210-d tangent
  > metric; all TSMNet folds are skipped by the degeneracy guard. This is a configuration limitation of applying
  > high-dimensional SPD tangent diagnostics to a very low-channel dataset, not evidence against or for the
  > TSMNet conclusions.
  Parking lot (optional appendix only, must NOT enter main conclusions): if all big-N TSMNet jobs later fail,
  run a low-dim TSMNet (`subspacedims` → tangent ≤ 6) on 2b as a robustness check.

### Lee2019_MI / Cho2017 / Schirrmeister2017 — PENDING (dumps running)

Run order when dumps land: **Track G + C12 first (NOT LPC/factorial)** → EEGNet erasure → EEGNet deployment →
TSMNet erasure → TSMNet deployment → skipped/degenerate report → CONFIRM/MIXED/OVERTURN per the criteria above.
