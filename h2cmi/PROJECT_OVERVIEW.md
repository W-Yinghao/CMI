# H²-CMI — Complete Project Overview

*A single reference for the whole project: what it is, every phase we ran, what each found, where the
artifacts live, and the current state. Read this top-to-bottom to understand everything.*

Last updated: 2026-06 (W1/V2P/W2 P0-correction compute complete; P0 finalizer pending).

---

## 0. TL;DR (one paragraph)

H²-CMI is a redesigned EEG domain-adaptation / test-time-adaptation (TTA) method. Across a long
program of **negative-result-driven** experiments, the headline contribution turned out to be a
**measurement→control gap**, not a winning adapter: we can often *detect* when adaptation is unsafe,
but unlabeled, source-calibrated adaptation yields little reliable *benefit* on real EEG. Concretely:
(1) the claim "CMI improves TTA" was falsified; (2) the joint EM's harm localizes to its **prior
M-step**; (3) the deployable policy is **metadata-only operator routing + identity-by-default** (no
target-statistic eligibility gate, because the source-calibrated score lacks transferable power);
(4) on real EEG the **safety/abstention** half holds but the **utility** half does not reach the bar;
(5) the effect of joint geometry+prior adaptation is **prevalence-dependent** (helps on balanced MI,
harms under natural sleep-stage prevalence). A bounded "P0" correction phase is now separating the
**fit prior from the decision prior** and re-doing the prevalence test by **reweighting a fixed trial
reservoir**.

---

## 1. What the package is

- `h2cmi/` is a **self-contained** EEG DG/TTA package, isolated from the concurrent AAAI `cmi/`
  (Tri-CMI / LPC-CMI) work in the same repo — it never imports-with-side-effects or mutates `cmi/`.
- It runs end-to-end on a **controllable mechanism simulator** (`data/paired_simulator.py`) and on
  **real EEG** (offline MOABB from the read-only datalake `/projects/EEG-foundation-model/datalake/raw`;
  + Sleep-EDF).
- Core pieces: encoder (`models/encoder.py`, EEGNet-style filterbank + optional SPD), class-conditional
  density head (`density/student_t_mixture.py`), hierarchical CMI (`cmi/hierarchical.py`), and the
  class-conditional TTA operators (`tta/class_conditional.py`).
- **The "method" as frozen**: `fixed-prior geometry adaptation + metadata-only operator routing +
  identity-by-default`. Tag `H2CMI_METHOD_FREEZE`.

### The TTA operator vocabulary (used everywhere below)
- **identity** — no adaptation.
- **pooled (`pooled_empirical_diag`)** — classless diagonal moment-match to the source pooled moments
  (CORAL-diagonal); independent of `p(z|y)`.
- **fixed-reference one-shot (`gen_oneshot_diag`)** — class-conditional diagonal transform, soft
  responsibilities generated once on identity, prior pinned at source = the "canonical CC".
- **fixed-iterative geometry (`gen_iterative_diag`)** — responsibilities re-estimated each EM round,
  prior pinned (geometry-only).
- **current_joint (`joint_iterative_diag`)** — transform + **prior M-step** (the full joint EM).
- **Latent-IM-Diag** — source-free information-maximization recentering (renamed from "SPDIM"; it is
  NOT an official SPD-manifold SPDIM).
- **source-recolored EA** — Euclidean Alignment in raw channel space (recolor target→source reference).

---

## 2. Repository map

- **Remote**: `git@github.com:W-Yinghao/CMI.git`.
- **H²-CMI lives on ONE branch**: `exp/h2cmi-responsibility-qxu` (the consolidated line; all earlier
  `h2cmi-*` branches were ancestors and were deleted as redundant). Worktree:
  `/home/infres/yinwang/CMI_AAAI_qxu`.
- **Current correction branch**: `exp/h2cmi-review-p0-corrections` (forked from
  `exp/h2cmi-responsibility-qxu @ 09e9249`; HEAD `278fc85`).
- **Other branches are SEPARATE research directions / sessions** (NOT H²-CMI): `main` (AAAI Tri-CMI),
  `exp/lpc-cmi` (closed/failed line), `acar`, `csc`, `oaci`, `tos`.
- **Committed artifacts** live in `h2cmi/results/*.md` + `*.report.json` + checksums; **raw JSONL +
  model bundles are NOT in git** (large; under `results/h2cmi/`, with SHA-256 recorded in the docs).

### Tag timeline on `exp/h2cmi-responsibility-qxu` (the research history)
```
b1-infra-v1/v2 → b1a-code-v1/v2 → b1a-astar-terminal → b2a-canonical-freeze → b2a-terminal
→ b2b-source-power-fail → H2CMI_METHOD_FREEZE → stage-v-terminal → audit-corrected
→ v2p-terminal → w1w2-terminal → w1b-terminal
```

---

## 3. The research arc (chronological), with verdicts

### 3.1 Mechanism localization — Stage B1a + A* + B2a + B2b  (all CLOSED/terminal)
- **B1a factorial grid** (7 variants × responsibility × transform-family): decomposed the joint EM.
  **Finding (solid):** the joint's harm is the **prior M-step** — geometry-only / fixed-prior beats the
  joint on covariate-family shifts (CIs exclude 0). p(z|y) earns its keep on prior-shift; low-rank
  helps rotation. → `h2cmi-b1a-*` tags.
- **A\* nested-null falsification:** the eligibility GATE works at B1a scale (false-adapt 0.03), but
  target-only ACTION SELECTION fails (top-1 0.40). → `h2cmi-b1a-astar-terminal` (A_STAR_FAIL, pivot).
- **B2a metadata substrate** (two-axis: acquisition-geometry × prevalence-risk → frozen rule table):
  metadata ROUTING works and is safe *without* a gate; the frozen A* gate over-vetoes. → `b2a-terminal`
  (B2A_FAIL localized to the gate). Canonical CC frozen = `gen_oneshot_diag` (`b2a-canonical-freeze`).
- **B2b source-power futility checkpoint:** the frozen change-of-variable evidence score lacks
  route-conditioned power — ROC ceiling ≈ 0.16 < 0.25 retention at 10% false-adaptation, replicated on
  fresh seeds. → `b2b-source-power-fail`. **Conclusion: no source-calibrated target-only eligibility
  gate works.** Deployed policy frozen to metadata-only + identity-default (`H2CMI_METHOD_FREEZE`).

### 3.2 Stage V — confirmation + real-EEG external validation
- **V1 (fresh seeds 100–119, simulator):** all six load-bearing claims replicate (prior-coupling;
  feedback-not-the-harm; pooled↔class-cond split; metadata_only safe+modestly-useful; eligibility-score
  power limit; low-rank rotation). One honest refinement: `C_responsibility` becomes significant with
  20 seeds (oracle labels help modestly; doesn't touch the label-free method).
- **V2 (real EEG, offline MOABB, binary L/R):** two panels.
  - **A — out-of-support abstention** (6 cross-dataset pairs over BNCI2014_001/Cho2017/Lee2019, common
    21-ch grid): metadata_only **holds identity on all 230 routes** (exact identity-equivalence,
    binomial upper-95% 0.013). Blind adapters get a small +mean but **harm 21–27%** of subjects. →
    operator-support safety holds.
  - **B — supported-regime utility** (cross-session, DIAG→pooled): mean paired Δ **+0.001 (CI includes
    0)**, harm 0.37 → **utility NOT established**. `current_joint` harms (prior-M-step replicates on
    real EEG). → `stage-v-terminal`, `audit-corrected` (after fixing a merge bug that had under-reported
    A to 18 routes, and adding subject-clustered CIs).
- **V2P — controlled prevalence intervention:** holds real trials fixed, varies the unlabeled
  adaptation-pool class ratio. **Every adaptive operator's geometry moves with pool prevalence**
  (pooled −0.063, joint −0.042, **fixed-prior CC −0.016**, all CIs exclude 0; identity flat). →
  **fixed-prior CC is NOT prevalence-invariant** (revised bias theorem). → `v2p-terminal`.
  *(NOTE: this V2P used a contiguous-subset pool construction — superseded by the P0 weighted version,
  §4.)*

### 3.3 W1 / W2 — unseen-subject benchmark + natural-prevalence task
- **W1 (MI unseen-subject LOSO; same-backbone panel):** identity / EA / pooled / canonical-CC /
  current_joint / Latent-IM-Diag across BNCI2014_001/Cho2017/Lee2019. **`current_joint` is best on
  balanced MI** (overall Δ +0.090; Cho +0.180).
- **W2 (Sleep-EDF cross-subject, natural prevalence):** `current_joint` is **worst** (Δ −0.043, CI
  excludes 0, harm 0.85); the metadata route (DIAG×DIFFERENT→CC) is the safest adapter (harm 0.30) but
  modest utility (CI includes 0).
- **Headline (W1+W2):** the effect of joint geometry+prior adaptation is **prevalence-dependent** —
  estimating the target prior can help when class composition is stable but causes severe negative
  transfer when prevalence varies. Stated as a *prevalence-conditional failure mechanism* supported by
  V1+V2P+W1+W2 jointly (NOT a single causal proof; MI and sleep differ in task/classes/structure). →
  `w1w2-terminal`.
- **W1-B — native BTTA-DG external reference** (ICLR-2026 `luo-huan-123/BTTA-DG`, pinned `5932d026`,
  MIT): faithful LOSO reproduction → **BTTA-DG ≡ its own source-only ensemble, Δ +0.000 on all 35
  subjects**. Root cause (verified): the published `OnlineClustererGMM.add_sample()` (the only
  buffer-fill) is **never called**, so the GMM never fits and the calibration is an **exact no-op**.
  With the orphaned call restored it activates on ~2% of trials and flips 1 prediction total → still
  Δ≈0. → `w1b-terminal`. (Reported precisely as a property of the public snapshot, not a blanket
  dismissal.)

---

## 4. Current phase — "REVIEW_P0" bounded correction (branch `exp/h2cmi-review-p0-corrections`)

**Why:** a public review found two P0 implementation/paper mismatches (both audited + verified in code):
- **P0-1 decision-prior confound:** the old `current_joint` evaluated with the joint's *estimated*
  prior `π_J` as the **decision** prior, while every other operator used uniform → the "joint
  helps/harms" delta conflated **geometry** with **decision-prior**.
- **P0-2 V2P pool construction:** the old V2P drew *different* contiguous trial subsets per ratio
  (`_pool_indices` ignored its seed) → prevalence was confounded with trial-set differences.
- Plus: W1/W2 used a single source seed; W2 used the first-20 subjects.

**Frozen design (`h2cmi/results/REVIEW_P0_FROZEN.md`):**
- **A. Separate fit vs decision prior:** fit the joint EM **once** → `(T_J, π_J)`, then decode the four
  branches `{identity, joint-geometry} × {uniform, π_J}`. Balanced-accuracy primary always uses the
  **uniform** decision prior. Exact decomposition with a numerical identity check:
  `full = G + P + Interaction`, where `G` = geometry@uniform, `P` = prior@identity, `Interaction` =
  the rest. **The corrected mechanism contrast = `fixed_iterative_geometry_uniform −
  joint_geometry_uniform`** (W1 primary; W2 secondary).
- **B. W1:** 3 source seeds {0,1,2}, seed-0 frozen bundles reused only after strict provenance
  validation, seeds 1–2 trained; average seeds within subject, stratified-within-dataset bootstrap.
- **C. W2:** **all 75** valid paired-night Sleep-Cassette subjects (audited, not hard-coded), 3 seeds,
  two protocols (night1→night2 primary; night2-split secondary), full recordings (ρ's, four JS
  divergences, per-stage recall, confusion). W2 primary = `joint_geometry_uniform − identity_uniform`.
- **D. V2P_WEIGHTED:** same fixed reservoir, only **sample weights** change across class-0 mass
  q∈{0.50,0.75,0.25} (same trial IDs reweighted). Weighted pooled/one-shot/iterative/joint + an oracle
  diagnostic; 6 mandatory tests all pass (equal→unweighted exact; rational→replication; identical IDs;
  exact masses; labels never enter non-oracle fits; identity ratio-invariant).
- **E. Stats:** 10k percentile bootstrap; W1 stratified, W2 subject, V2P subject-cluster; Holm only
  within declared confirmatory families; "harm rate" → **"negative-change rate"** at 0/−0.01/−0.02.
- **F. Naming:** SPDIM→Latent-IM-Diag; EA→source-recolored EA. (No official SPDIM; no BTTA-DG rerun.)

**Compute status (locked at `278fc85`):** **COMPLETE.**
- W1: 3450 rows (115 units × 3 seeds × 10), 0 provenance failures, seed-0 reused + seeds 1–2 trained.
- V2P-weighted: 5670 rows (90 units × 3 seeds × 21).
- W2: primary **75/75** (2250 rows) + secondary **75/75**, both protocols, all 3 seeds.
  (Scheduling note: a per-user submit cap forced a refilling "babysitter" SLURM job to run W2 ≤24
  concurrent; this is purely scheduling — the protocol is unchanged.)

**Finalizer (pending) — to run now that compute is done, in this commit order, then push + tag:**
1. **Keyed merge + manifest** — expected-key manifest on `(protocol, subject, source_seed, branch)`
   (not filenames/mtime); exclude any residual shard from the earlier rejected W2 array; assert every
   row `commit==278fc85`, each key exactly once, each unit has seeds {0,1,2}; per-seed provenance
   assert (`seed==0 → seed0_validated`; `seed∈{1,2} → newly_trained`).
2. **Analyzer + provenance + tests** corrections (V2P cluster key=(dataset,subject); per-unit bootstrap
   CIs for G/P/I/G+P+I/fixed_iter−joint in W1 **and** W2; W1 subject-weighted + dataset-equal-macro +
   per-dataset CIs + leave-one-dataset-out + per-method negative-change rates; V2P FRSC-vs-oracle &
   vs-pooled, translation + embedding displacement, slope divisor `2·ln 3`; provenance audit of every
   reused bundle with retrain-on-fail; +3 weighted tests; supersession table). Report `runner_commit`,
   `analyzer_commit`, `diagnostic_replay_commit` separately.
3. **W2 confusion replay** (7 branches, eval-only, **must be hash-equivalent** to the primary run).
4. **`REVIEW_P0_RESULTS.md` + `review_p0.report.json` + `.sha256`**, then the **terminal tag** — only
   after all terminal conditions (keys/commit/seeds, provenance, replay hash-equivalence, full test
   suite green, report self-consistent, old claims explicitly superseded).
- **Do not interpret the interim W1 decomposition before the finalizer.** The first result to report is
  `fixed-iterative geometry − joint-fit geometry (uniform decision)` across datasets, and the W2
  **G / P / Interaction** split (geometry vs decision-prior vs interaction as the source of harm).
- The corrected phase **decides whether the old W1/W2/V2P conclusions are kept or retracted** — the goal
  is correctness, not rescuing the old numbers.

---

## 5. Consolidated findings (what is actually true)

1. **"CMI improves TTA" — falsified.** The factorial decomposition does not support it.
2. **The joint EM's harm = the prior M-step.** Geometry-only / fixed-prior beats the joint on the
   shift families where the joint hurts. (Being re-stated cleanly via the P0 G/P/Interaction split.)
3. **No source-calibrated target-only eligibility gate works** (B2b power ceiling). Deployed = metadata
   routing + identity-default.
4. **Real-EEG safety holds, utility does not.** Metadata abstains correctly under acquisition mismatch
   (V2-A, 0/230 unsafe adaptations); supported-regime utility is not statistically established (V2-B).
5. **fixed-prior CC is prevalence-robust-er, not prevalence-invariant** (V2P).
6. **Joint geometry+prior adaptation is prevalence-dependent** (W1 balanced MI: helps; W2 natural-
   prevalence sleep: harms).
7. **The published BTTA-DG snapshot is an inert no-op** (orphaned `add_sample`); reproduced faithfully.
8. **Central contribution = the measurement→control gap:** detecting *whether/when* to adapt is the
   binding problem; unlabeled source-calibrated adaptation is safe but marginal on real EEG.

---

## 6. Environment + how to reproduce

- **Env:** conda `icml` (`/home/infres/yinwang/anaconda3/envs/icml/bin/python`) — torch 2.8.0+cu128,
  moabb 1.2.0, mne 1.8.0, scikit-learn 1.5.2, scipy 1.13.1, numpy 1.26.4.
- **Data:** offline datalake `/projects/EEG-foundation-model/datalake/raw` (no network; moabb keeps old
  class-name aliases). Sleep-EDF Sleep-Cassette under `.../sleep-edf/sleep-cassette` (75 valid
  paired-night subjects).
- **Compute:** all GPU work via SLURM (never login-node GPU). GPU partitions `A40,A30,V100,V100-32GB,
  V100-16GB` (V100 max 2-day → use ≤47h walltime; A100/L40S are 1-day). Per-user **submit cap** limits
  concurrent jobs → use job arrays or a refilling babysitter (≤24). CPU partitions `CPU`, `cpu-high`.
- **Tests:** `for t in test_weighted_tta test_b1a test_grid_io ... ; do python -m h2cmi.tests.$t; done`.
- **Key runners:** `run_b1a_grid`, `run_b2a_grid`, `run_b2b_source_power`, `run_v2` (A/B), `run_v2p`,
  `run_w1_mi`, `run_w2_sleep`; **P0:** `run_w1_p0`, `run_w2_p0`, `run_v2p_weighted`, with
  `eval/p0_eval.py` (decomposition) + `tta/weighted_tta.py` (weighted estimators) + `p0_source.py`
  (validated 3-seed source loader). Analyzers: `analyze_*` + `analyze_p0`.
- **Result indices to read:** `results/METHOD_FREEZE.md`, `B1A_RESULTS.md`, `B2A_TERMINAL.md`,
  `B2B_TERMINAL.md`, `V2_FROZEN.md`/`V2_RESULTS.md`, `V2P_FROZEN.md`/`V2P_RESULTS.md`,
  `W1_W2_FROZEN.md`/`W1_W2_RESULTS.md`, `W1B_REPRODUCTION.md`, and (pending) `REVIEW_P0_FROZEN.md`/
  `REVIEW_P0_RESULTS.md`.

---

## 7. Hard process rules established in this project
- Pre-register every outcome-producing phase (a `*_FROZEN.md`) **before** running it.
- Never retune frozen gates/thresholds/acceptance clauses mid-evaluation.
- Provenance-bind every result (commit SHA, code signature, data hash, expected-key manifest); reuse a
  frozen bundle only after strict validation; on mismatch STOP the unit, never silently relax.
- Commit code / pre-registration / results as separate commits; push only on explicit instruction;
  terminal tags only after all terminal conditions hold.
- Report faithfully (negative results stand); separate predeclared *primary* from *secondary/
  exploratory*; rename "harm rate" → "negative-change rate".
