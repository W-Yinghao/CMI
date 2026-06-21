# ACAR — Action-Conditional Counterfactual Adaptation-Risk Router (PRE-REGISTERED, frozen 2026-06-21)

Working title: *Predicting Negative Transfer, Not Distribution Shift, for Safe EEG Test-Time Adaptation.*

This is **Direction 2**, the leak-proof successor to the CLOSED gate-falsification line (A0 / A0′ / A0′-R /
A0-PILOT, see `notes/EVIDENCE_LEDGER.md`). It is **TTA, not strict DG** — we do touch the unlabeled target batch.
This file is frozen **before** running. Estimand, features, leakage guards, decision rule, and the GO/NO-GO kill
criterion below are NOT to be changed post-hoc. The one and only permitted post-hoc action is **STOP**.

---

## 0. Why the predecessor died, and what is structurally different here

From the autopsy (`notes/EVIDENCE_LEDGER.md`, claims #3–#6):

1. **Wrong target.** A0 predicted a *source-free static scalar* (density / CMI-proxy / Mahalanobis support) and
   asked "is this batch shifted / atypical?". Those scores are **anti-aligned with adaptation harm** (harm AUROC
   0.40–0.47): OOD-ness predicted base *difficulty*, not whether *adapting* makes things worse.
2. **Outcome-conditioned leakage.** The one apparent batch-level "win" (A0′, ρ≈+0.3–0.5) was fabricated by
   aggregating the score over the `base_correct` subset — i.e. conditioning on true target labels. Whole-batch
   label-blind aggregation (A0′-R) collapsed it to noise (AUROC≈0.5).
3. **Measurement→control gap.** Even the surviving sample score (`s_sep`, harm AUROC 0.69) did **not** reduce
   deployed loss in closed loop (A0-PILOT: retained NLL *worse* than random). Ranking ≠ loss reduction.

ACAR changes **three** things, each targeting one failure:

| Failure | A0 (dead) | ACAR (this file) |
|---|---|---|
| Wrong target | predict *shift magnitude* / absolute risk | predict **paired incremental harm** `ΔR_a(B)=R_B(f_a)−R_B(f_0)` — *does acting beat not acting on THIS batch?* |
| One score for all | a single source-free scalar gates every adapter | **action-conditional**: a separate predictor per candidate action `a` |
| Ranking ≠ control | thresholded a raw score | **conformal one-sided upper bound**: act iff `U_a(B) < −δ`, calibrated leave-one-source-cohort-out; validated by **closed-loop deployed loss**, not AUROC alone |

The features are also new in kind: A0 used static functions of the target alone. ACAR uses **paired pre/post
observables** — what *changed* when action `a` was actually executed on the batch (flip rate, JS, Δmargin, Bures
transport). These are counterfactual ("what did `a` do?"), not descriptive ("is the batch weird?").

CMI and density are demoted to **background context only**. They may enter `φ_a` as raw coordinates fed to the
learned regressor, but NO fixed "higher = more dangerous" direction is asserted for them — A0 proved that sign
is unreliable.

---

## 1. Substrate (frozen, GPU-free)

- Dumps: `archive/lpc-cmi-failed/results/feat_dump_v4/audit_{cond}_{coh}_erm_0.npz` — the **CITA-no-LPC deployment
  encoder** (`erm:0`, P1.5-closed), 16-dim tangent embeddings `z`. Same artifacts the A0 line was bound to, so
  results are directly comparable.
- Cohorts (each is a held-out pseudo-target): PD = {ds002778, ds003490, ds004584}; SCZ = {ds003944, ds003947,
  ds004000, ds004367}.
- Per cohort: source readout state is fit on `(z_ev, y_ev)` via `cmi.eval.source_state.fit_source_state` (frozen
  LR probe + class-conditional moments + source prior; serialized, no raw source at scoring). The pseudo-target is
  `z_te`; its labels `y_te` are used **only** in Phase-2 for the estimand.
- **Natural batches** (primary): `z_te` sorted by `window_index_te`, grouped by `recording_id_te` (one recording =
  one deployment session), chunked into batches of **B = 32** in recording order. No label balancing. This is the
  deployment regime — harm is the tail of an on-average-mildly-helpful adapter.
- Synthetic generators (A0's `lowmargin_rot` / `highmargin_cbw` / `covariate_shift_beneficial`) are a **secondary
  stress sensitivity only** (`--stress`), never the go/no-go endpoint.

## 2. Candidate actions (frozen set)

Each action maps an unlabeled batch `B` (its features `z`) to post-predictions `p_a` and, where geometric,
post-features `z̃_a`, reading ONLY the serialized source state — source-free, label-free, gradient-allowed.

| key | what | post-features `z̃_a`? |
|---|---|---|
| `identity` | frozen readout `f_0` = `clf.predict_proba(z)`; the no-op reference | yes (`z̃ = z`) |
| `matched_coral` | `pmct_predict_serialized(..., ref="pooled", tmap="wc")` — the deployed transductive lever (≡ CITA-no-LPC on accuracy) | yes |
| `spdim` | IM recentering bias on `z` (`tta_baselines.spdim_predict`) | shift-only (`z̃ = z+b`) |
| `t3a` | test-time classifier adjustment (`tta_baselines.t3a_predict`) | no (prob-only) |

Signal-level **raw EA** and full **CITA-with-LPC** require re-embedding from raw `X` and are **deferred to the
backbone run** if and only if the go/no-go passes. The z-space set already gives ≥2 non-trivial actions to route
between, which is sufficient to test action-conditional routing. `identity` is always an available fallback.

## 3. Estimand — paired incremental risk (NOT shift, NOT absolute accuracy)

For batch `B` and action `a`, with frozen reference `f_0` = `identity`:
```
ΔR_a(B) = R_B(f_a) − R_B(f_0)
```
- **Primary risk** `R`: mean per-sample NLL, `R_B(f) = mean_{i∈B} −log p_f(y_i | x_i)`.
- **Secondary risk**: balanced 0–1 error on `B`.
- `ΔR_a(B) > 0` ⇒ action `a` **HARMED** this batch (negative transfer). `ΔR_a(B) < 0` ⇒ beneficial.
- Computed offline on source cohorts where `y` is available; this is the only place `y_target` is touched, and only
  AFTER the label-free scoring path has been recorded (Phase-2, §5).

The harm label for AUROC is the **sign**: `harm = 1[ΔR_a(B) > 0]`, evaluated **whole-batch** (no subsetting).

## 4. Features `φ_a(B)` — all label-free, all paired pre→post

Computed in Phase-1 from `p_0` (identity) vs `p_a` (action) and the feature batches, with NO access to `y_target`:

| feature | definition | reads `y`? |
|---|---|---|
| `d_entropy` | `mean H(p_a) − mean H(p_0)` | no |
| `d_margin` | `mean(top1−top2)(p_a) − mean(top1−top2)(p_0)` (confidence margin, label-free) | no |
| `flip_rate` | `mean 1[argmax p_a ≠ argmax p_0]` | no |
| `js` | `mean_i JSD(p_0^i ‖ p_a^i)` | no |
| `bures` | Bures–W2 between `z̃_a` and `z` batches (transport size); `nan` for prob-only actions | no |
| `post_sep` | label-free Fisher trace-ratio of `z̃_a` under `p_a` **pseudo-labels** (separability the adapter induced) | no |
| `n_eff` | effective sample size of `B` under `p_a` responsibilities (min over classes) | no |

Background context (raw coordinates only, NO asserted direction): `g_unc`, `s_support`, `s_sep`, `pr_cmi_proxy`
(the A0 source-free scores, recomputed identically). They may improve the learned regressor; they are NOT gates.

## 5. Leak-proofing (the five non-negotiables; metamorphically enforced in code)

The deployment scoring API is `route(state, z_batch) → {action, U_a}` and MUST satisfy ALL of:

1. **No `y_target` in scoring.** `route()` and every `φ_a` has no labels argument; labels are reachable only in the
   Phase-2 evaluator.
2. **Whole-batch aggregation.** Every batch statistic is a mean/quantity over the *entire* batch — never over a
   label-defined subset (this is exactly the A0′ leak).
3. **Label-permutation invariance (bit-identical).** Permuting `y_target` must leave `φ_a`, `U_a`, and the routed
   action **bit-identical**. Asserted per batch; any violation ⇒ `SystemExit` (mirrors A0′-R's metamorphic guard).
4. **Serializable source state.** Scoring consumes only `fit_source_state(...)` output + unlabeled target; no raw
   source examples. Verified against the deployed transductive predictor bit-exactly before scoring.
5. **No class-conditional batch deletion.** The only fallback is label-blind (`len(B) < 8` → forced `identity`);
   never drop/keep a batch by its true classes.

Determinism: SHA-256 seeds, double-run canonical-hash equality (abort on mismatch).

## 6. Router — learned per-action predictor + clustered conformal one-sided bound

- For each action `a`, train a regressor `ĝ_a: φ_a(B) ↦ ΔR̂_a(B)` (monotone gradient-boosting on the batch-summary
  vector; a DeepSets set-encoder over per-sample features is the drop-in upgrade, deferred unless go/no-go passes).
- **Leave-one-source-cohort-out** within disease: to deploy on cohort `c`, train `ĝ_a` and calibrate on the OTHER
  same-disease cohorts only. Conformal residuals are grouped **by cohort** (clustered split-conformal) → the
  guarantee is a clinical-cohort-level one-sided bound, not i.i.d.-per-batch.
- One-sided upper bound: `U_a(B) = ĝ_a(φ_a(B)) + q_{1−α}`, `q_{1−α}` the cohort-clustered (1−α) quantile of
  calibration residuals `ΔR_a − ĝ_a`. **Default α = 0.1, δ = 0.0** (act only when confidently non-harmful).
- **Decision rule**: among actions with `U_a(B) < −δ`, execute the one with the smallest `U_a`; else **`identity`**
  (abstain from adaptation). Sample-level abstention, if added later, uses an **independent** model (per Direction-2
  spec) — out of scope for the go/no-go.

## 7. GO / NO-GO (FROZEN kill criterion — both gates required, on BOTH diseases)

This direction CONTINUES iff **both** hold; otherwise it **TERMINATES** (no score/coverage/seed re-search):

- **G1 — Signal exists.** For at least one candidate action `a ∈ {matched_coral, spdim, t3a}`, on BOTH PD and SCZ,
  the **cohort-macro AUROC** of at least one paired feature in `φ_a` (or the learned `ĝ_a`) for discriminating
  `harm = 1[ΔR_a(B) > 0]` is **≥ 0.60**, and is **stable** (per-cohort AUROC > 0.5 in ≥ all-but-one cohort, same
  sign across PD/SCZ — no direction reversal).
- **G2 — Control follows.** The conformal router (§6) on held-out cohorts **reduces deployed NLL vs both
  `always_adapt` and the matched-coverage `random_abstain` baseline**, while retaining **≥ 50 %** of the
  beneficial alignment captured by `always_adapt` (i.e. it abstains on harm, not on benefit). Measured closed-loop,
  per disease — AUROC alone is explicitly insufficient (this is the A0-PILOT lesson).

Outcomes:
- **G1 ∧ G2 → `PROCEED`**: lift to the backbone run (add raw-EA + CITA actions, DeepSets, full conformal paper).
- **G1 ∧ ¬G2 → `MEASUREMENT_ONLY`**: report as a negative-transfer *predictor* with the measurement→control gap
  made explicit; do not claim a deployable router. STOP coding new scores.
- **¬G1 → `TERMINATE`**: paired counterfactual observables do not predict negative transfer here either. Close the
  direction. Do **not** swap features and retry.

## 8. Output (immutable schema)

```
results/acar_gonogo/<freeze_a1_hash16>/
  acar_gonogo_summary.json   # per action: per-feature cohort-macro AUROC (PD/SCZ) + per-cohort spread;
                             # learned-regressor AUROC; router closed-loop NLL vs always/random/never; G1,G2,decision
  run_manifest.json          # dump hashes, source-state verify residual, seed map, this file's path, git sha, double-run hash
```
Guards enforced in code before any endpoint: serialized-state verify gate, metamorphic label-permutation guard,
double-run determinism. Unit tests (`acar/tests/`) run the guards on synthetic data before the real run.
