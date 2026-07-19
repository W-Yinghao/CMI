# Readout Prior LOCKBOX — execution freeze (pre-registered BEFORE target-utility results)

**Branch** `agent/cmi-trace-readout-prior-lockbox` (worktree `/home/infres/yinwang/CMI_AAAI_readout_prior`, base
`da835d8c`). PM-directed; manuscript FROZEN; **no amendment**; only the owner stops/redirects the line. This note is
committed **before** the 525-cell matrix produces any target-utility number, so the decision rule cannot be tuned to
the result. It inherits the arms / objective / τ-selection / firewall of
[READOUT_PRIOR_DECOMPOSITION_CONTRACT.md](READOUT_PRIOR_DECOMPOSITION_CONTRACT.md) unchanged.

## Why this round
The decomposition round found a matched-τ source-head CENTER effect (H2@τ vs H1@τ, low-shot) on the 4
method-development datasets, but those 4 were used to build/tune the whole line — so the effect could be dev-only. The
PM correction established the estimand is the **LOW-SHOT prior value** (a prior SHOULD peak at low k and wash out at
Full; `dU_center(Full)≈0` is expected shrinkage, not a confound — the earlier post-hoc "Full-discriminating" estimand
change was reverted). The decisive open question is **external replication on UNTOUCHED data**: does the matched-τ
center effect survive on natural multi-session MI datasets never touched by method development?

## The two lockboxes (chosen by the owner; Ma2020 unavailable)
- **Stieger2021 — PRIMARY.** Natural longitudinal 4-class MI (left/right/both-hand/rest), 62 subjects, 7 or 11
  sessions, ~250k trials, 60 common channels (uniform across all 62 — probed and pinned). Frozen protocol: epoch
  `[0,3]s`; cal = **session 1**; **primary query = sessions 2–7** (session-macro) — THE verdict query; **long-horizon
  = sessions 8–11** reported SEPARATELY and **EXCLUDED** from the primary L-A..L-D verdict (only 11-session subjects).
- **Shin2017A — CONFIRMATORY.** Natural 3-session binary L/R MI, 29 subjects, ~10 trials/class/session. Frozen
  protocol: epoch `[0.5,3.5]s`; cal = earliest session; query = the other two (generic split, unchanged).

The original 4 datasets are **re-run with the 8 protocol fixes** and reported as **context only** — they do not drive
the verdict.

## Matrix (525 cells = subjects × 3 seeds, ERM EEGNet lam0, 300 epochs, LOSO leave-one-SUBJECT-out)
`BNCI2014_001 27 + BNCI2015_001 36 + Lee2019_MI 162 + BNCI2014_004 27 + Stieger2021 186 + Shin2017A 87 = 525`.
Each cell: train ERM EEGNet on the source fold → dump frozen Z(16-d)+logits+session axis → readout-prior arms on the
held-out subject's sessions. The frozen-feature dump is the SAME validated `feature_dump.dump_fold` that produced the
existing 4 datasets (only the datasets are new).

## PRIMARY endpoint (unchanged from the reverted-to pre-reg estimand)
Low-shot **AULC over k∈{1,2,4,8}** of the **matched-τ** center contrasts. Full is a posterior-washout diagnostic, not
a failure.
1. `dU_center_t0(k) = U_H2@τ0 − U_H1@τ0` — pure center (both heads at the zero-centered τ). **PRIMARY.**
2. `dU_center_ts(k) = U_H2@τs − U_H1@τs` — pure center (both heads at the source-centered τ). **PRIMARY.**
3. `dU_center(k)   = U_H2@τs − U_H1@τ0` — policy (mixes center + shrinkage-strength selection).
4. `dU_MAP_frozen(k) = U_H2 − U_H0` — deployable adapt-vs-frozen.
5. `dU_gate_frozen(k) = U_H4 − U_H0` — safe source-gated policy.

Inference unit = **target subject** (draw→seed→subject); subject-cluster bootstrap + exact sign-flip; LCB95.

## FROZEN decision rule (verbatim in `scripts/aggregate_readout_prior_lockbox.py::_route`)
- **L-D** `SOURCE_ANCHORED_TARGET_READOUT_IMPROVES_LABEL_EFFICIENCY` — a lockbox has `LCB(dU_MAP_frozen)>0` with no
  clear harm on the other (`mean>−0.005`) AND a center effect holds. (strongest — deployable label-efficiency gain)
- **L-A** `SOURCE_HEAD_CENTER_IMPROVES_LOW_LABEL_ESTIMATION` — `LCB(dU_center_t0)>0` AND `LCB(dU_center_ts)>0` on
  **Stieger2021** AND same-direction no-harm on **Shin2017A** (mean>0 both, `LCB>−0.01` both). (real τ-isolated prior)
- **LONGITUDINAL_FEEDBACK_REGIME_DEPENDENT** — center holds on Stieger2021 but not Shin2017A.
- **L-C** `SOURCE_PRIOR_PREVENTS_LOW_SHOT_OVERFIT_NO_ADAPTATION_HEADROOM` — center exists but no lockbox clears frozen.
- **L-B** `SOURCE_CENTERED_POLICY_OUTPERFORMS_CENTER_NOT_ISOLATED` — policy `LCB>0` on primary but matched-τ not both.
- **NO_EXTERNAL_CENTER_EFFECT_LOW_SHOT_ADVANTAGE_NOT_STABLE** — neither lockbox shows a matched-τ center effect →
  the dev advantage is not externally stable; pivot to prior transportability / hierarchical or class-conditional
  prior, NOT B_cond erasure.

## Refuse / firewall
- Aggregator **REFUSES** (SystemExit) on `<525` cells or **any** `status=failed_solver` cell (P0.2 fail-loud).
- Firewall: target **QUERY** enters only the final utility; cal Y only adapts heads; τ + gate are **source-only**; NO
  re-inference. `long_horizon` is stored per-cell but never enters the primary verdict.

## Execution (autonomous, redundancy-controlled)
1. **Epoch once, in parallel** — one SLURM task per (dataset,subject) banks a per-subject cache; assembled + a single
   consolidated cache banked so the 186 LOSO folds don't re-epoch (fixed a `np.savez` tmp-naming bug that had silently
   broken every cache write; salvaged 20 already-epoched Stieger subjects rather than recompute). **DONE 91/91.**
2. **GPU dump array** — one task per (dataset,subject,seed), A100/H100/L40S first (`--time=1d`), skip-existing +
   done-marker, exclude-running self-healing driver. Reads the consolidated cache (no re-epoch).
3. **CPU matrix** — 525 cells, stable enumeration, launched only after all 273 dumps land + `enumerate_cells==525`.
4. **Aggregate → adversarially verify → report.** No post-hoc estimand change; verdict is whatever `_route` returns.

## Pre-results code red-team (adversarial; BEFORE any target-utility number)
A 5-lens adversarial review (firewall / session-split / matched-τ arm-curve / routing / cache-label-encoding) with
independent refutation of each finding was run before the matrix. Outcome:
- **Firewall = CLEAN** (2-agent CONFIRMED): no target-QUERY label/feature reaches the whitener, standardization, the
  source head, τ0/τs selection, the H4 gate, any cal-adapted head, the specificity MATCHING criterion, or `_route`.
  Query enters only the final `session_macro_bacc` utility + three reporting-only `_headroom` diagnostics (never routed).
- **Session-split & cache/label-encoding = REFUTED** (no bug): Stieger masks (numeric 1 | 2–7 | 8–11) mutually
  exclusive; generic Shin/existing-4 split unchanged; per-subject-assembly LabelEncoder consistent with `dump_fold`.
- **ONE verdict-changing bug FIXED** (commit 139b2b67): the aggregator's "refuse partial" gate counted `.done`
  markers, but skipped/failed cells also write a `.done` — a dump regression dropping the session axis could pass the
  525 gate and route on a biased subset (`_cluster_ci` has no min-n floor). Now: gate on `status==ok` count; REFUSE on
  any skip (exit 4), on `usable<expect` (exit 2), on solver-fail (exit 3), and on a lockbox below its subject floor
  (Stieger≥60 / Shin≥28, exit 5). All refuse paths unit-tested.
- **Deferred hygiene footgun** (not verdict-affecting): the Stieger `(0.0,3.0)` epoch window is not centralized in
  `DATASET_DEFAULTS`; the committed dumper `gpu_dump_one.py` passes it explicitly so no on-disk cache/result is wrong,
  but a generic runner could silently re-epoch a different window. To be centralized as post-fleet hygiene (touching
  the loader mid-fleet is avoided per the stale-code discipline).

## Holds (unchanged)
New erasure / target-X selector / source proxy / mechanism-consistency loss / learned projector / TTE — all PARKED.
Manuscript FROZEN.
