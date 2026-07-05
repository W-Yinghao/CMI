# CIGL_53 — P9: CSP-init FBCSP-LGG confirmation & baseline preflight (non-GPU)

Branch `project/cspinit-fbcsp-lgg-confirm` off P8 tip `ad066ba`. PI gate: **P8 PASS, promote D forward.** P9 is
NOT new-method invention — it turns the first real win into a writable, credible result.

**SOTA-track main line renamed:** `DualCMI` → **CSP-initialized FBCSP-LGG (+ spatial auxiliary supervision)**.
The load-bearing positive component is source-CSP initialization, not the CMI penalty. The old static-DGCNN
graph/node-leakage line stays a SEPARATE bounded audit/control paper ([[cigl-graphcmi-direction]]); do not mix
its audit claims with this accuracy-track.

## P9-A — paper-ready P8 evidence (DONE, this commit)

`results/p8_cspinit/`: `P8_GATE_REPORT_FINAL.md` (4-layer tables + conservative headline), `P8_AGGREGATE.csv`,
`P8_DECOMPOSITION.csv`, `P8_SUBJECT_TABLE.csv`, `P8_SUMMARY.json`. **Reported effect = collapse-robust
+0.06–0.07** on BNCI2014 CSP-decodable {1,3,8,9} (not the +0.096 raw mean). Claim hierarchy: D = primary
candidate; **CSP-init = primary mechanism** (B alone +0.062; D−C +0.068); spatial-aux = secondary/weaker
(D−B +0.034, ~0 at seed1) → **B is the conservative fallback**.

## P9-B — D main, B fallback (settled)

Main results table = A/B/C/D factorial. No re-running B/D single-term: the 2×2 factorial already decomposes
CSP-init (B−A), aux (C−A), interaction (D−A / D−B / D−C). Further single-term runs would just repeat spend.

## P9-C — NO in-dataset tuning (frozen)

Not authorized: aux_weight sweep, CSP-filter-count sweep, fusion_floor sweep, λ/CMI sweep, dec_scale sweep,
seed3+. We have full-LOSO × 3 seeds × 4 variants; more tuning turns a clean positive into a search result.
Next is confirmation / external validation / baselines, not internal score-squeezing.

## P9-D — baseline sidecar PREFLIGHT (done here; GPU NOT yet requested)

We finally have a positive → we must add strong baselines for credibility. **Env preflight result (eeg2025):**
- **braindecode: MISSING (ImportError), torchaudio: MISSING (OSError)** — the previously-flagged broken stack.
- **BUT faithful pure-torch baselines already exist in-repo** (`cmi/models/sanity_backbones.py`, Phase 3A-S):
  `EEGNetMini`, `ShallowConvNetMini`, `DeepConvNetMini` — all build + forward on (22ch/128/4cls) AND (2015)
  in-env, **no braindecode needed**. moabb 1.5.0 / skorch / sklearn 1.8 / scipy 1.17 present → classical
  **CSP+LDA** buildable from `cmi/models/csp_init.py` (source_csp_filters) + sklearn LDA.
- **Recommendation:** run baselines with the in-env pure-torch nets (no env repair) as the primary sidecar;
  optionally attempt a braindecode install later ONLY if a reviewer demands the official implementations.
  CAVEAT for the paper: these are faithful *minimal* reimplementations (built for sanity, reach 1.0 on
  learnable synthetic), not the official braindecode nets — disclose that.

**Proposed baseline sidecar run-spec (awaiting PI approval — NO GPU launched):**
```text
Backbones: EEGNetMini, ShallowConvNetMini, DeepConvNetMini (pure-torch, in-env) + CSP+LDA (classical)
Config: erm (deep nets) / fit (CSP+LDA)
Datasets: BNCI2014_001 all 9 folds + BNCI2015_001 all 12 folds
Seeds: 0/1/2 (deep nets); CSP+LDA deterministic (seed-invariant, report once)
Protocol: full-LOSO, source-only, --source_val_early_stop where applicable, same preprocessing/window as P8;
          target labels EVAL-ONLY. Report on the SAME 2a CSP-decodable {1,3,8,9} endpoint + full means.
Scale: 3 deep nets x 21 folds x 3 seeds = 189 GPU jobs + CSP+LDA (CPU). %8.
Purpose: position CSP-init FBCSP-LGG D vs standard deep MI baselines + classical CSP on the same folds.
```
Without these we can only say "better than our graph baselines / CSP-init helps", not "competitive with
standard MI decoders".

## P9-E — held-out dataset/site (later, non-GPU survey first)

For a deployable/general claim: non-GPU MOABB survey (available MI datasets, channel/class compatibility, trial
counts, preprocessing match) → pick ONE held-out site → full-LOSO seed0. NO new-dataset GPU until the survey +
loader are confirmed (recall the BNCI2015_001 LeftRightImagery→MotorImagery loader trap).

## Frozen
P6 (fragile null), P7a cov_tangent (FAIL/killed), P7b, decodability-adaptive gating, conditioning-first
covariance, PCMI-TIF — all frozen. CMI returns only as (a) an audit/regularization appendix (old bounded
story), or (b) a post-hoc stability regularizer on the *validated* D backbone — never mixed into P8-D
confirmation. No GPU beyond a PI-approved baseline run-spec; no sweeps.
