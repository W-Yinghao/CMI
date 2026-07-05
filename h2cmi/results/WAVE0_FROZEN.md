# WAVE 0 — Real-EEG Controlled-Mechanism Foundation (FROZEN PRE-REGISTRATION)

**Phase:** Wave 0 of the real-EEG controlled-mechanism program. **Branch:** `exp/h2cmi-wave0-mechanism`
(off the frozen terminal `5bc9bf0` / tag `h2cmi-review-p0-terminal`). **Freeze timestamp := the git
commit timestamp of this file.** This document is the pre-registration; the primary/secondary endpoints,
go/no-go criteria, determinism controls, and analysis choices below are fixed *before* any Wave 0 compute.

> **Relationship to the terminal.** Wave 0 is **additive**. It does **NOT** supersede or re-tag the
> REVIEW_P0 result-of-record (`278fc85` raw / analyzer `9a35cc9` / tag `h2cmi-review-p0-terminal`). The
> W2 deterministic rerun (W0.1) writes to a **new bundle root and a new tag**; the non-deterministic
> `278fc85` W2 primary remains the result-of-record for the P0 correction, with the deterministic run as
> its reproducible companion.

## 0. Why Wave 0 (contribution framing)

The paper's contribution is a **prior-decoupled diagnostic**, not a leaderboard: a single unlabeled
"joint TTA" number conflates **geometry fitting (G)**, **fit-prior (P)**, and **decision-prior**
effects, and must be decomposed. Wave 0 hardens the *foundation* of that claim on **real EEG** before any
dataset expansion, by (i) closing the acknowledged W2 reproducibility hole, (ii) adding real **negative
controls**, (iii) upgrading the prevalence intervention from "how much did it move" to "**what did the
movement help or harm**", (iv) tying the weak-identification theory to real data via a batch-size sweep,
and (v) making the **metric-switch** (which objective each prior serves) a first-class result. No new
external datasets are required; all Wave 0 compute reuses existing cached real EEG.

## 1. Shared protocol (all Wave 0 groups)

- **Four branches**, exactly as in REVIEW_P0: `(I, Unif)`, `(I, π_J)`, `(T_J, Unif)`, `(T_J, π_J)`,
  where `I` = identity geometry, `T_J` = joint-EM geometry, `Unif` = uniform decision prior,
  `π_J` = joint-EM estimated prior used as the decision prior.
- **Exact decomposition** of the conventional joint delta: `full = G + P + I_int`, with
  `G = BA(T_J,Unif) − BA(I,Unif)` (geometry @uniform), `P = BA(I,π_J) − BA(I,Unif)` (fit-prior @identity),
  `I_int` the interaction; identity `|full − (G+P+I_int)| < 1e-9` asserted per unit.
- **Balanced accuracy is always evaluated at the uniform decision prior** for the primary endpoint.
- **Comparator operators** carried through (renamed, per terminal): pooled, fixed_reference_oneshot
  (FRSC), fixed_iterative, joint, **Latent-IM-Diag**, **source-recolored EA**, identity; oracle
  (true-label) transform as a **diagnostic** only.
- **Seeds** averaged **within unit** (scalars) before any bootstrap; **never** average latent (a,b)
  vectors across seeds.
- **Cluster bootstraps** reused: W2 = subject; V2P = **(dataset, subject)**; 10 000 percentile
  replicates; Holm within any declared confirmatory family.

## 2. Experiment groups (FROZEN)

### W0.1 — W2 deterministic re-evaluation (closes the reproducibility hole)
- **AMENDMENT (design, 2026-07-06): eval-only reuse of the frozen terminal bundles, not a retrain.** The
  REVIEW_P0 replay non-determinism was in the **eval** (EM/Adam transform fit), not training; and the
  encoder's only non-deterministic op (`adaptive_avg_pool2d_backward`) is a *training* backward — its
  forward (inference) is deterministic. So W0.1 **reuses the frozen terminal bundles** (`p0_w2_bundles`,
  code_sig `763bf49d`, read-only) and re-runs a **deterministic eval**. The confusion is therefore for the
  **actual terminal models** (more faithful than a fresh retrain), and no encoder change is needed (which
  would change `code_sig` and break reuse). If a fresh retrain were ever required, its training-side
  non-determinism would be hardened separately under a new freeze.
- **Data:** existing `p0_sleep_cache` (Sleep-Cassette, all 75 paired-night subjects), both protocols
  (primary night1→night2; secondary within-night-2). No re-download.
- **Determinism controls (fixed):** `torch.use_deterministic_algorithms(True)`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `cudnn.deterministic=True` / `benchmark=False`, fixed per-unit
  seeds (`stable_hash_int`). If any op lacks a deterministic implementation it raises and that fold is
  recorded as `determinism_fail` — we do **not** silently fall back.
- **AMENDMENT (user-directed, 2026-07-06): per-fold GPU-type reproducibility, not a single global pin.**
  Jobs are scheduled by SLURM across **{A100, H100, A40, V100}** (throughput). Each fold **records its
  GPU type + CUDA/torch/library manifest**. The self-replay reproducibility gate re-runs a fold on **its
  own recorded GPU type** and requires identical prediction/logit hashes. Cross-architecture float
  differences are therefore never compared; each fold's confusion is admissible iff it is bit-reproducible
  on the architecture that produced it. Walltime honors the queue caps (A100/H100 ≤ 23:59:59; A40/V100 ≤
  2 days); **per-fold append + skip-if-done** guarantees no completed fold is lost on a walltime kill.
- **Full logging per (subject, seed, branch):** logits, hard predictions, confusion matrix, per-stage
  recall, `π_J`, transform params `T_J=(a,b)` and its norm, source-bundle SHA, adapt/eval split hash,
  optimizer seed, and a GPU/CUDA/library manifest.
- **Reporting:** the four-branch `G/P/I_int` decomposition (re-confirming the terminal within the
  previously-characterized optimization band) **plus** the newly-admissible per-stage recall / confusion.
- **New root + new tag** (`p0_w2_det_bundles`, tag `h2cmi-wave0-w2det-*`); terminal untouched.

### W0.2 — Fixed-reservoir prevalence UTILITY curves (movement → harm/benefit)
- **Data:** the existing fixed reservoir — same real EEG trials, **reweighted** (effective-count
  weights) at a **q-grid `q ∈ {0.1..0.9}` step 0.1**, anchor `q=0.5`. Trial identity + temporal position
  invariant across q (P0-2-correct). Per operator {pooled, FRSC, fixed_iterative, joint, oracle}.
- **Primary — displacement from q=0.5:** eval-embedding displacement, translation norm, log-scale norm.
- **Primary — utility under fixed evaluation:** BA, macro-F1, NLL, negative-change rate vs q=0.5.
- **Secondary — matched-prevalence ordinary accuracy:**
  `ordinary_acc_q = q·recall_class0 + (1−q)·recall_class1`; compare `π_dec = Unif`, `π_dec = π_J`, and the
  **oracle q decision prior** (diagnostic).
- **Three separated questions:** (A) *do prevalence-only weights induce geometry movement?* → displacement
  / translation / log-scale. (B) *is that movement useful for a balanced objective?* → BA / macro-F1 /
  negative-change. (C) *does the estimated prior have a legitimate use under an ordinary-accuracy
  objective?* → matched-prevalence ordinary accuracy. This keeps the paper from reading as "estimated
  prevalence is never useful": the claim is metric-conditional (π_dec is set by the metric; `π_J` is a
  diagnostic branch, not the default predictor).

### W0.3 — Same-session **consistency control** (within-session fake-split control)
- **Renamed (user, 2026-07-06): NOT an "identity-geometry null".** A real same-session split can still have
  a genuine subject-geometry gap, so `G ≈ 0` is **not** the pre-registered expectation and we do not claim
  "adaptation should have no effect."
- **Data:** within-subject, within-session stratified fake split (equal prevalence), both directions.
- **Primary endpoints:** (1) **directional asymmetry** `Δ(A adapt→B eval) − Δ(B adapt→A eval)` — should be
  ~0 (no systematic direction effect); (2) **branch stability** — distribution of `G/P/I_int` under
  stratified same-session fake splits; (3) **mechanistic comparison** — magnitude of minority-stage recall
  collapse here vs the W0.1 cross-night collapse (expected **weaker** because `π_J` is closer to the
  evaluation prevalence within a session).
- **What it tests:** that branch bookkeeping + real-subject addressing are stable with no true night shift,
  the stratified split does not manufacture a cross-night-like large `P` collapse, and the decision-prior
  harm is markedly smaller when `π_J ≈ ρ_eval`.

### W0.4 — Adaptation batch-size sweep (theory ↔ real EEG)
- **Design (frozen):** `batch_n ∈ {16, 32, 64, 128, 256}`; **draws_per_unit = fixed predeclared** count;
  **stratified** sampling where possible else natural-prevalence with recorded class counts; **evaluation
  set fixed across n**; source/eval hashes fixed; **average draws within the biological unit before**
  cluster bootstrap.
- **Three required curves** (not just mean-BA-vs-n): (1) `|π_J − ρ_eval|` vs n; (2) transform-norm /
  displacement vs n; (3) BA / negative-change-rate vs n.
- **Pre-registered prediction (weak identification):** at small n, prevalence/geometry are harder to
  separate → `π_J` less stable, `P` worse, geometry movement larger; the confound attenuates as n grows.

### W0.5 — Metric-switch (which objective each prior serves) — cross-cutting
- **Reuses W0.1/W0.2 branch logits** (independent endpoint, **no model recompute**; pre-declared as such).
- **Level 1 — natural W2 metric-switch:** BA → `π_dec = Unif` is metric-correct; ordinary accuracy →
  compare `π_dec ∈ {Unif, π_J, oracle ρ_eval(diagnostic)}`. Interpretation: `π_J` estimates the
  *adaptation-window* prevalence, not necessarily the *eval-night* prevalence.
- **Level 2 — fixed-reservoir matched metric-switch:** q controls **both** adaptation prevalence and the
  ordinary-accuracy evaluation weights; when the deployment objective truly matches q, prevalence-aware
  decisions may help.
- **Claim:** not "`π_J` is always bad" but "`π_J` must not be used **unconditionally** as a
  balanced-accuracy decision prior"; the correct `π_dec` is metric-dependent.

## 3. Falsification / expected-informative outcomes (pre-committed)

- **Geometry-only perturbations may falsify the latent-diagonal family — and that is a publishable
  bound.** A real re-reference / channel-gain / montage perturbation is linear in *sensor* space but need
  not become a *diagonal positive-affine latent* map after the nonlinear encoder. If diagonal-affine
  operators (FRSC/joint) underperform a full-covariance/CORAL operator on a geometry-only shift, we report
  it as **bounding where the method applies**, not as a failure. *(Geometry-only real perturbations are
  specified here for continuity but their confirmatory run is Wave 1; Wave 0 freezes only the null,
  prevalence, batch-size, deterministic-W2, and metric-switch groups on existing data.)*
- **W0.3 consistency control: directional asymmetry ≈ 0, not `G ≈ 0`.** A large `Δ(A→B) − Δ(B→A)`
  asymmetry, or a cross-night-sized `P` collapse under an equal-prevalence same-session split, falsifies
  the control (→ investigate leakage/model-mismatch/bookkeeping before any positive
  claim).
- **W0.1 must be bit-reproducible.** Acceptance = two back-to-back identical runs on the pinned GPU type
  produce **identical prediction hashes** for all branches; only then are per-stage recall / confusion
  admitted. If determinism cannot be achieved, confusion stays excluded and we report why.

## 4. Go / No-Go (frozen)

| group | PASS criterion |
|---|---|
| W0.1 | self-replay prediction hashes identical on pinned GPU; all branch artifacts saved + hash-manifested; `G/P/I_int` re-confirmed (aggregate within the characterized band); per-stage confusion then admitted |
| W0.2 | full displacement + utility panels over the 9-point q-grid, cluster-bootstrapped; metric-switch shown (BA vs ordinary-accuracy sign) |
| W0.3 | directional asymmetry `Δ(A→B)−Δ(B→A)` CI includes 0; `G/P/I_int` distribution reported; minority-stage recall collapse **weaker** than W0.1 cross-night (else flag leakage/mismatch/bookkeeping) |
| W0.4 | `G/P/I_int` + variance + reproducibility reported across all 5 batch sizes; monotone-attenuation trend stated (confirmed or refuted) |
| W0.5 | BA and ordinary-accuracy objectives reported side-by-side for W0.1/W0.2; π-usefulness statement supported or refuted |

## 5. Sleep-cohort scope clause (frozen)

> Primary sleep robustness in Wave 0 is **Sleep-EDF alternative split** using the existing processed
> cache. **External ISRUC replication is a separately-frozen extension, activated only if the full
> ISRUC dataset lands on the datalake before this document's freeze timestamp; otherwise it is
> journal-version future work.** MASS / NSRR (SHHS/MESA) are **not** in the current confirmatory scope
> and, if pursued, start only under a new freeze tag.

## 6. Provenance & commit discipline

- New branch `exp/h2cmi-wave0-mechanism`; this pre-reg is a **separate commit** from any code or results.
- Reuse of any frozen bundle requires the **strict** `get_source_p0` validation (null-field rejection);
  ProvenanceError → STOP that unit and report.
- GPU work via SLURM only (no login-node training); per-unit checkpointing; deterministic W2 pinned to a
  single GPU type.
- Terminal artifacts (`278fc85`, tag `h2cmi-review-p0-terminal`) are **read-only**; Wave 0 never
  modifies or re-tags them.

## 7. Fan-out addressing invariant (regression-gated) + GO/NO-GO

**Invariant (enforced by `h2cmi/wave0_fanout.py` + `test_wave0_fanout.py`):** `bench_index` is **never** an
identity — not a job key, output path, merge key, or CLI identity argument; only a non-identifying log
field. Units are addressed by the canonical key
`(dataset, real_subject_id, session/night, pair_id, source_seed, split_seed, wave, q|batch_n, branch)`.
Downstream joins use real ids / session / wave variables only.

**GO for the next fan-out wave (W0.2 …) only after ALL of:**
1. fan-out real-id fix committed **and** `test_wave0_fanout.py` passes (shuffle-invariant unit set; each
   real_subject covered once-and-only-once; no bench-index in identity; preflight catches dup);
2. `WAVE0_FROZEN.md` updated with that wave's endpoints (this file);
3. an **expected-unit coverage report** generated (`python -m h2cmi.wave0_fanout --wave … --coverage`)
   **before** job submission;
4. no W0.1 outputs modified after `e75a0b0` except manuscript-facing summaries.

**NO-GO if any of:** a job still addresses subjects by bench index; an output merge key uses bench index
as identity; q-grid coverage differs across operators without a pre-declared exclusion; the deterministic
hash manifest (logits/preds/confusion) is missing.
