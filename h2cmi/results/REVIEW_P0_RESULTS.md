# REVIEW_P0 — Corrected Results

**Status: TERMINAL for the P0 correction phase.** This file supersedes the pre-P0 W1/W2/V2P
conclusions wherever they conflict (see the supersession table, §6).

| field | value |
|---|---|
| marker | `REVIEW_P0_RESULTS` |
| runner_commit (raw compute) | `278fc85` |
| analyzer_commit | `9a35cc9` |
| diagnostic_replay_commit (W2 confusion, excluded) | `9a35cc9` |
| rows analyzed | 13 620 (W1 3450 · V2P 5670 · W2 primary 2250 · W2 secondary 2250) |
| seeds | source {0,1,2}, averaged **within unit** (scalars) before any bootstrap |
| provenance | 196 reused seed-0 bundles strict-clean (code_sig `763bf49d`); 0 provenance-fails |
| decomposition identity | `full = G + P + Interaction`, |residual| ≤ 3.7e-17 (W1 exactly 0) |

The two audited P0 mismatches that motivated this phase:
- **P0-1 (decision-prior confound).** The old `current_joint` decoded with the joint EM's *estimated*
  prior `π_J` as the **decision** prior while every other operator used uniform — conflating
  **geometry** with **decision-prior** effects. Corrected by decoding four branches
  `{identity, joint-geometry} × {uniform, π_J}` with balanced-accuracy **always at uniform decision**,
  and the exact decomposition `G` (geometry@uniform) `+ P` (prior@identity) `+ Interaction`.
- **P0-2 (V2P pool construction).** The old V2P drew *different* contiguous trial subsets per prevalence
  ratio → prevalence confounded with trial identity. Corrected by `V2P_WEIGHTED`: one fixed reservoir,
  the **same** trials reweighted (effective-count weights) per ratio.

All deltas are balanced-accuracy (bAcc) at the **uniform** decision prior unless explicitly a P/decision
contrast. CIs are 10 000-replicate percentile bootstraps with the cluster scheme noted per panel.

---

## 1. W1 — motor imagery (DG, 115 LOSO units across 3 datasets)

Bootstrap: stratified-within-dataset (subject-weighted) and dataset-equal macro (resample subjects
within each dataset, average the 3 dataset means equally).

| contrast | subject-weighted [95% CI] | dataset-equal macro [95% CI] |
|---|---|---|
| **PRIMARY** `fixed_iterative − joint_geometry` (uniform) | +0.0022 [−0.0005, +0.0049] **NS** | +0.0047 [+0.0009, +0.0088] *sig* |
| `G` = `joint_geometry − identity` (uniform) | **+0.0604 [+0.043, +0.078]** | +0.0480 [+0.034, +0.063] |
| `P` = `identity@π_J − identity@uniform` | −0.0065 [−0.0147, +0.0018] **NS** | −0.0071 [−0.0142, −0.0001] *sig* |
| `joint_geometry − pooled` (uniform) | −0.0035 [−0.0065, −0.0005] | −0.0074 [−0.0127, −0.0023] |
| Interaction (mean) | +0.0043 | — |

Per-dataset `G`: Cho2017 **+0.123** [+0.087, +0.159], BNCI2014_001 +0.013 (NS), Lee2019_MI +0.008 (NS).
Per-dataset PRIMARY: BNCI2014_001 +0.011 *sig*, Cho2017 −0.0007 (NS), Lee2019_MI +0.0034 (NS).
Leave-one-dataset-out (PRIMARY): only **drop-Cho2017** stays significant (+0.0046); drop-BNCI and
drop-Lee are NS — the small positive PRIMARY effect is not robust to which dataset is removed.

Mean bAcc: identity 0.671 · joint_geom@uniform **0.732** · fixed_iterative **0.734** · pooled 0.735 ·
fixed_reference_oneshot 0.721 · Latent-IM-Diag 0.726 · joint_geom@π_J 0.729 · identity@π_J 0.665 ·
source-recolored-EA 0.698.

**W1 reading (per the decision grid).**
- `G > 0` clearly (+6.0 bAcc pts, robust to weighting): the **joint geometry @uniform genuinely helps**
  in W1, concentrated almost entirely in **Cho2017** (+12.3 pts); BNCI/Lee are individually flat.
- `P ≲ 0` and marginal (≈ −0.7 pts; significant only under macro weighting): using the joint-fit prior
  as the *decision* prior gives at most a small W1 penalty.
- PRIMARY `fixed_iterative − joint_geometry` ≈ 0 subject-weighted, tiny-positive macro: in W1 the
  iterative prior-M-step feedback does **not** materially degrade the geometry — fixed-reference and
  joint geometry are effectively tied. Note `joint_geometry` is itself slightly *below* the plain
  `pooled` baseline (−0.4 pts), so the W1 "geometry helps" is relative to identity, not to pooled.

---

## 2. W2 — sleep staging (Sleep-Cassette, 75 paired-night subjects)

Bootstrap: subject cluster. Primary protocol = night-1 (adapt) → night-2 (eval); secondary protocol =
within-night-2 split (same-night, marked secondary).

| contrast | **primary** [95% CI] | secondary [95% CI] |
|---|---|---|
| **PRIMARY** `G` = `joint_geometry − identity` (uniform) | −0.0200 [−0.0413, +0.0007] **NS** | −0.0230 [−0.0436, −0.0029] *sig* |
| `P` = `identity@π_J − identity@uniform` | **−0.1438 [−0.159, −0.128]** | −0.1300 [−0.150, −0.110] |
| decision-prior diagnostic `joint_geom@π_J − joint_geom@uniform` | −0.0846 [−0.097, −0.073] | −0.1061 [−0.121, −0.091] |
| **SECONDARY** `fixed_iterative − joint_geometry` (uniform) | +0.0185 [+0.0127, +0.0245] *sig* | +0.0234 [+0.0112, +0.0377] *sig* |
| Interaction (mean) | +0.0593 | +0.0238 |

Mean bAcc (primary): identity **0.657** · joint_geom@uniform 0.637 · fixed_iterative 0.656 ·
**fixed_reference_oneshot 0.695** · pooled 0.660 · Latent-IM-Diag 0.636 · joint_geom@π_J 0.553 ·
identity@π_J **0.513**.
Negative-change rate vs identity (primary, fraction of subjects worse): identity@π_J **98.7 %** ·
joint_geom@π_J 88.0 % · joint_geom@uniform 61.3 % · pooled 48.0 % · fixed_iterative 46.7 % ·
fixed_reference_oneshot **30.7 %**.

**W2 reading (per the decision grid).**
- The grid case that fires is **`P < 0` (large) and `G ≈ 0`**: `P = −14.4 bAcc pts` (huge, CI excludes 0)
  while `G = −2.0 pts` is **not significant**. **The harm in the original "joint" operator is dominated by
  the decision prior — the night-1 joint-fit prior `π_J` used as the night-2 decision prior — not by the
  geometry.** The decision-prior diagnostic confirms it: switching only the decision prior
  (uniform → `π_J`) at fixed joint geometry costs −8.5 pts.
- A **secondary, much smaller** effect is also real: `fixed_iterative − joint_geometry = +1.85 pts`
  (CI excludes 0). This supports the **prior-M-step-feedback** mechanism — the joint EM's prior M-step
  feeds back into and slightly degrades the geometry relative to a fixed reference. It is an order of
  magnitude smaller than the decision-prior effect.
- Net, the honest one-line W2 statement: **joint decision/prior coupling under prevalence variation is
  harmful, almost entirely through the decision prior; the geometry @uniform is roughly neutral
  (NS primary, ≈ −2 pts secondary), with a small additional prior-M-step degradation of the geometry.**
- The secondary same-night protocol agrees and is marginally stronger on `G` (−2.3 pts, just significant).
- `fixed_reference_oneshot` is the **best** W2 operator (0.695, only 30.7 % subjects harmed) — consistent
  with "fix the reference, don't let prevalence/prior feed back into geometry."

---

## 3. V2P_WEIGHTED — prevalence-stress (90 (pair,session) units, 72 (dataset,subject) clusters)

Fixed reservoir, same trials reweighted at q ∈ {0.25, 0.50, 0.75}. Bootstrap: **(dataset,subject)**
cluster. PRIMARY = eval-embedding displacement ‖T_q(U_eval) − T_0.5(U_eval)‖ and the
**FRSC-vs-pooled / FRSC-vs-oracle** Holm 2-contrast family (oracle = diagnostic). Slope normalized by
**2·ln 3** (corrected from the earlier ÷2). Log-scale and translation displacements separated.

| operator | embedding disp [95% CI] | log-scale disp | translation disp | slope ÷ 2ln3 |
|---|---|---|---|---|
| pooled | 0.049 [0.033, 0.069] | 0.205 | 0.158 | 0.222 |
| fixed_reference_oneshot (FRSC) | **0.314 [0.228, 0.409]** | 0.044 | 0.297 | 0.268 |
| fixed_iterative | 0.640 [0.461, 0.825] | 0.083 | 0.595 | 0.514 |
| joint | 0.640 [0.461, 0.824] | 0.082 | 0.595 | 0.514 |
| oracle_label_conditional | **1.960 [1.878, 2.036]** | 0.191 | 1.838 | 1.648 |

Confirmatory family (Holm): `FRSC − pooled` = **+0.265 [+0.191, +0.341]** (excludes 0);
`FRSC − oracle` = **−1.646 [−1.773, −1.512]** (excludes 0). Both survive Holm.

**V2P reading (per the decision grid).**
- FRSC's eval-embedding displacement is **significantly nonzero (0.314)** and **significantly larger than
  the prevalence-agnostic `pooled` baseline (+0.265)** → the fixed-reference one-shot operator is **not
  prevalence-invariant**: its soft assignments shift the transformed embedding as target prevalence changes.
- **The oracle also moves substantially (1.96).** Per the pre-registered red-line, because the
  label-conditional oracle itself responds strongly to prevalence, we **cannot attribute FRSC's movement
  purely to soft-assignment bias** — part of any class-conditional operator's displacement is the
  legitimate prevalence response. The clean, defensible statement is the **attenuation** one:
  `pooled (0.049) < FRSC (0.314) ≪ oracle (1.96)`, i.e. **FRSC exhibits a real prevalence-dependent
  displacement that is strongly attenuated relative to the true label-conditional response** (FRSC−oracle
  = −1.65). It under-responds rather than over-responds.
- Decomposition of *where* FRSC moves: its displacement is overwhelmingly **translation** (0.297) with a
  near-zero **log-scale** component (0.044) — the prevalence sensitivity enters through the mean shift,
  not the diagonal scaling. The iterative/joint operators move ~2× more than FRSC and are essentially
  identical to each other (0.640 vs 0.640).

---

## 4. W2_CONFUSION_REPLAY_AUDIT — excluded (strict)

`W2_CONFUSION_STATUS = EXCLUDED_STRICT`. The hash-equivalence gate (predictions must reproduce the
primary run before per-stage recall / confusion can enter the report) **was not met**, for a located,
non-bug reason:

- An eval-only replay (diagnostic_replay_commit `9a35cc9`) re-evaluated the 7 review-requested branches by
  **reusing the cached source bundles** (no retraining).
- **`pi_J` reproduced exactly on 148/225 units (median Δ = 0)** but **co-diverged on 34 units**
  (max |Δπ_J| = 0.022, 2 units > 0.01).
- **114 / 2025 prediction hashes differed; 13 units differed by more than 0.01 bAcc, with a maximum bAcc
  difference of about 0.10** (mean |ΔbAcc| = 2.7e-4).
- Cause: the iterative **EM/Adam geometry optimization is nondeterministic across GPU executions** (CUDA
  atomics, compounded over EM iterations) and occasionally reaches a different local solution; the prior,
  being co-estimated inside the same joint EM, inherits that nondeterminism.

> A replay intended to recover per-stage confusion did not reproduce the primary iterative geometry fits
> exactly. The prior estimates were identical for most units (148/225, median Δ = 0) but co-diverged on a
> minority; 114 / 2025 prediction hashes differed, 13 units by more than 0.01 bAcc, with a maximum bAcc
> difference of about 0.10. We therefore **do not report replay-derived confusion matrices or per-stage
> recall**.

**This does not supersede the primary W2 results.** The primary W2 metrics remain the recorded `278fc85`
run and are analyzed with seed-averaging and subject-cluster bootstrap. The replay audit shows additional
optimization nondeterminism for iterative geometry fits, so replay-derived branch-level confusion is not
used as confirmatory evidence. We did **not** rerun W2 deterministically and did **not** admit the replay
confusion as a secondary artifact (both are out of scope for this P0 correction; a deterministic sleep
per-stage confusion run, if ever needed, would be a separate pre-registered robustness study).

---

## 5. Provenance & tests

- **Reused seed-0 bundles**: W1 115 (`w1_bundles`) + V2P 81 (`v2_bundles`) = 196, all code_sig
  `763bf49d`, **strict-clean** — no null `code_sig`/`data_hash`/`epochs`/`n_chans`/`n_train`, all
  checkpoints present (`review_p0_provenance.json`). W2 used pre-registered **no-reuse** (all seeds
  trained into `p0_w2_bundles`); recorded as freshly-trained.
- **Validator hardened**: `get_source_p0` now **rejects null** sidecar fields (no permissive
  None-tolerance); seed-conditional reuse/training preserved.
- **Weighted-estimator tests (10/10 pass)**, including the 3 added: all-ones weighted joint == unweighted
  joint (exact), all-ones one-shot == unweighted (exact), integer-weight joint == replication (< 1e-7).
  Full P0 test suite green.

---

## 6. Supersession table

| superseded claim/artifact | status | replaced by |
|---|---|---|
| Old W1 "joint operator" single number | **superseded** | G/P/Interaction decomposition (§1): geometry helps (Cho2017-driven), prior-as-decision marginally hurts, fixed≈joint |
| Old W2 "joint harms" (geometry blamed) | **superseded** | §2 decomposition: harm is the **decision prior** (P = −14.4 pts), geometry @uniform ≈ neutral (G NS); small extra prior-M-step geometry degradation (+1.85 pts) |
| Old V2P prevalence test (different contiguous subsets per ratio) | **superseded** | `V2P_WEIGHTED` fixed-reservoir, same trials reweighted (§3); confirmatory FRSC displacement + FRSC−pooled, oracle diagnostic |
| "SPDIM" operator name | **renamed** | **Latent-IM-Diag** (source-free information-maximization diagonal recentering) |
| "EA" operator name | **renamed** | **source-recolored EA** |
| "harm rate" | **renamed** | **negative-change rate** |
| V2P slope ÷ 2 | **corrected** | slope ÷ **2·ln 3** |
| V2P cluster key = bare subject | **corrected** | cluster key = **(dataset, subject)** (90 units → 72 clusters) |
| Old external BTTA-DG comparison | **demoted** | reproducibility audit only (not a confirmatory baseline) |
| W2 per-stage recall / confusion | **excluded (strict)** | not reported — replay not hash-equivalent (§4); primary W2 numbers stand |

---

## 7. Headline (corrected)

The original W1-helps / W2-harms contrast for the "joint" operator was partly an artifact of the
decision-prior confound (P0-1). Corrected: **W1 — joint geometry genuinely helps (+6 bAcc pts,
Cho2017-driven); W2 — the apparent harm is almost entirely the night-1 joint-fit prior used as the
night-2 decision prior (−14 pts), with the geometry itself roughly neutral and only a small additional
degradation from the prior-M-step feedback (+1.85 pts removing it).** Under prevalence stress
(P0-2-corrected V2P), the fixed-reference one-shot operator is not prevalence-invariant but its
displacement is strongly attenuated relative to the label-conditional oracle. Numbers are reproducible
from `review_p0.report.json` (analyzer `9a35cc9` on raw compute `278fc85`); W2 per-unit confusion is
excluded because the iterative geometry fit is not bit-reproducible across GPUs.
