# CSC real-EEG cache FEASIBILITY REPORT (Lee2019/OpenBMI, pre-reg v4) — PASS

Status: **FEASIBILITY PASS.** This closes the authorized feasibility step (pre-reg v4 §13). No validation bank
was run, no tag created, no certifier executed. Next steps (package build) require a separate go.

## What was built
Isolated direct-`.mat` cache builder `csc/mininfo/build_lee2019_b3_cache.py` (NO `cmi`, NO `moabb`; `scipy.io`
only). Reads OpenBMI `.mat` per the canonical layout verified against moabb's own `Lee2019.py`:
`scipy.io.loadmat`; struct `EEG_MI_train[0,0]` with `x`[time×ch], `fs`=1000, `chan`, `t` (onsets), `y_dec`
(1=right\_hand, 2=left\_hand). Frozen pipeline: `SM16_no_FCz` (16 ch), band-pass 8–30 Hz (Butterworth-4,
zero-phase), window 0.5–3.5 s, resample 128, `normalize=None`, `Z=log var_t`, label map {left:0, right:1},
MI training run per session.

Feasibility run: SLURM job 880788 (nodecpu01, eeg2025), ~10 min. Output cache (not committed; rebuildable):
`/home/infres/yinwang/realeeg_feas/cache/LEE2019_B3.npz` (+ `LEE2019_B3_metadata.json`).

## Feasibility gates (all PRIMARY gates PASS)
| gate | result |
|---|---|
| G1 all 16 channels present (every file) | PASS (missing_channels: none) |
| G3 ≥20 eligible paired subjects | PASS — **54/54 eligible** |
| G4 eligible have both sessions | PASS |
| G5 each session both classes | PASS (every cell 50/50 L/R) |
| G6 ≥8 trials/cell | PASS (100 trials/session, 50/class) |
| G7 feature dim = 16 | PASS |
| G8 rank(Z) ≥ 3 | PASS — **rank 16 (full)** |
| G9 no NaN/inf | PASS (0/0) |
| G10 non-degenerate variance | PASS — std median **0.691** (min 0.653, max 0.753) |

Totals: 54 subjects × 2 sessions × 100 trials = **10,800 trials**; no missing subjects/sessions/channels.
The v1 log-bandpower degeneracy (from the loader's time z-score) is genuinely fixed by `normalize=None`
(per-trial std ~0.25–0.88 across subjects; the v1 degenerate value was ~7e-8).

## Sanity checks (both required, both PASS)
1. **Trial-count / balance:** every subject×session recovers 100 trials, exactly 50/50 L/R. PASS.
2. **Session-1 label cross-check vs moabb (job 880795):** for subjects 1–3 the moabb-exposed per-trial label
   sequence **exactly matches** the direct reader (100/100). This validates the label parse/mapping (a swap
   would survive the 50/50 count check but fail here). It also **explains the moabb gap**: moabb's exposed
   session `'1'` = physical **session 2** (`session_name = str(session−1)`); moabb was dropping session 1. The
   direct reader recovers **both** sessions — exactly the reason D2 (direct reader) was chosen.

## Conclusion
The real-feature bridge is **feasible**: on real Lee2019 EEG, the frozen 16-channel un-normalized
log-bandpower feature is non-degenerate and full-rank; both paired sessions are available for all 54 subjects
(≥20 eligible floor cleared with margin). Per authorization, work **STOPS here**. The next step — building the
isolated freeze package (manifests + Route A/B3 runners + semi-synthetic injected bank + subject-clustered
bounds + red-team) and, later, running it — requires a separate explicit go. Method locks remain byte-unchanged;
the frozen synthetic tags `dee8958` / `0595f64` are untouched.
