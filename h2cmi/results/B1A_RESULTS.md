# Stage B1a / B1b-1 results

CMI-OFF, simulator, 3 development seeds (0–2) × 5 sites. All runs on A40 (torch 2.8.0+cu128,
cuDNN 91002) from the `exp/h2cmi-responsibility-qxu` worktree. Raw JSONL is **not** committed —
see `B1A_ARTIFACT_MANIFEST.json` for SHA-256 / row counts / SLURM jobs / exact commands, and the
release asset `h2cmi-b1a-results-v1.tar.zst` (183K) for the full rows. Report JSONs + checksums
in this directory are the committed index.

## 1. Mechanistic decomposition (standard grid, 735 rows)
The joint EM's harm localises to the **prior M-step**, not the responsibility loop:
- **C_prior_coupling** (geometry-only > joint): fires on the cov-family, CIs exclude 0
  (cov +0.047 [+0.011,+0.067]; cov_prior +0.021 [+0.012,+0.034]). Cleanest signal; confirms B0.
- **C_feedback** null/opposite → iterative responsibility feedback is *not* harmful.
- **C_responsibility** null → responsibility *quality* is not the bottleneck **for a transform
  shared across target subjects** (this is what the per-target-subject LOSO oracle measures —
  cross-subject transform transferability, NOT single-subject responsibility quality).
- **C_class_cond**: p(z|y) is load-bearing for **prior shift** (full +0.052, OOF +0.060, CIs
  exclude 0) but a pooled empirical moment-match wins on covariate/rotation. Reading: p(z|y)'s
  value is avoiding mistaking a label-prior difference for a geometry shift, not universal gain.
- **C_family** (low-rank > diagonal, **oracle** responsibilities): fires on rotation
  (cov_cond_rot +0.038 [+0.018,+0.057]). Supports off-diagonal *capacity* value under the
  responsibility ceiling — NOT that any unsupervised method already recovers rotation.

Development-grade (3 seeds; several CIs touch 0). Confirmatory needs fresh seeds.

## 2. Hard-null safety (105 rows; re-frozen preset 0.11/2.2, strict bAcc 0.763)
Held-out-safe but in-sample-unstable: identity is the OOF-best action on aggregate
(`identity_within_tol=True`), but the variants are not in-sample inert under low SNR (5–13%
prediction flips, OOF NLL worse than identity's 0.999), and **joint is worst** (occupancy
collapse 0.037) — the same prior-M-step culprit.

## 3. B1b-1 evidence router — NEGATIVE RESULT (the deployable router as defined FAILS)
Fixed rule: per unit pick argmax cross-fitted (LOSO) change-of-variable evidence gain over the
deployable label-free actions {identity, pooled_empirical_diag, gen_oneshot, gen_iterative},
fall back to identity if max ≤ 0. On the existing grids:
- **false-adaptation rate on nulls = 1.00** (standard and hard) — it adapts on every null unit.
- **harm rate 0.41 (standard) / 0.93 (hard)**; top-1 oracle agreement 0.12 / 0.00.
- Root cause: the change-of-variable marginal likelihood gain is large-positive for *any*
  volume-expanding transform regardless of class structure (gen evidence_gain +4.8 on the true
  null with ΔbAcc 0.000; +5.8 on prior-shift with ΔbAcc −0.034). It does not track accuracy.

The only label-free held-out signal recorded is this evidence gain, and it is exactly the one
that fails. So the "evidence-routed" mainline needs a **class-structure-aware, null-calibrated**
routing signal before any action-set expansion or confirmatory run.

## 4. B1a ≡ B0 bridge — bAcc-METRIC-EQUIVALENT (not "decision-equivalent")
For matching methods, B1a and B0 give **identical balanced accuracy per unit** (identity 105/105
exact; joint 105/105 exact, max|Δ|=0.0000); genuinely different methods differ
(gen_oneshot vs geometry_only, std 0.057). This is metric-equivalence on bAcc only — per-trial
prediction equality / logit differences were NOT verified (B0 stored no per-trial predictions),
so we do not claim decision-equivalence. The byte-level checkpoint-hash divergence (CPU `e56d`,
A40 `8c8d`, B0 `0f80`) is therefore at least bAcc-inert; `--b0-ref` ran in soft mode.

## 5. A* terminal target-only router test (nested canonical-site null) — DECOMPOSITION RESULT
Proper nested null (30 LOSO-site models/difficulty, cached by unordered pair; 60 null units),
other-seed empirical max-null calibration, C dropped, B as veto. Two routers, frozen criteria.

**N1 (nested-null gate) passes every clause EXCEPT action-selection:**
- WHEN-to-adapt is solved: false-adaptation **0.03** (std) / **0.00** (hard) [was 1.00 raw],
  coverage **0.36** (≥0.25 — not a trivial always-identity), shift ΔbAcc **+0.044** (>0),
  non-null harm 0.11 (≤0.20), hard-null harm 0.00, hard ΔbAcc ok, disagreement ok. **All PASS.**
- WHICH-action FAILS: top-1 oracle agreement **0.40** (<0.50) and action regret **0.038** (>0.02).
  The three diagonal actions {pooled_empirical, gen_oneshot, gen_iterative} are too similar for a
  target-only signal to rank (N1 shift selection: identity 38, pooled 20, gen_iter 12, gen_oneshot 5).
- N2 (+B veto) over-abstains (coverage 0.05) and is strictly worse.

Per the pre-registered rule (both routers fail ALL) the decision is **A_STAR_FAIL → pivot to B**.
But this is a *decomposition*, not a flat wall: A* VALIDATES the **source-calibrated abstention**
component of the B mainline (the nested-null gate is safe, covering and useful) and FALSIFIES
**target-only action selection** (top-1 0.40) — which is exactly why B routes the OPERATOR choice
through metadata, not target statistics. The earlier raw/prototype-null router (false-adapt 1.00,
coverage 0.01) failed because of the training-included null + dead C signals; the proper nested
null removed those confounds and isolated the residual, genuine limit: choosing *which* operator
from unlabeled target data alone.

Terminal: no further target-only score is developed. Next is the B mainline
(fixed-prior geometry adaptation + source-calibrated abstention + metadata-conditioned operator
selection), with the nested-null gate carried forward as the validated abstention rule.

## Provenance / code
Tags `h2cmi-b1a-code-v1`/`v2`; CUDA fix `1583878`; hard re-freeze `bb42d3a` (frozen on identity
ONLY, before any hard variant). Analysis: `analyze_b1a_grid.py` (5 contrasts + hard-null safety),
`analyze_b1a_router.py` (B1b-1 router). Source models trained @ `2b4c4f1`, runs @ `1583878`/`bb42d3a`.
