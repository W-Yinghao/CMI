# ACAR v2 — post-result AUDIT NOTES (non-analytic; 2026-06-22)

Scope: corrections to the **reporting/provenance** of the binding v2 run. This file does **NOT** modify the frozen
result files (`results/acar_gonogo/9b2f0c1_057f437d6615/{acar_gonogo_summary.json,run_manifest.json}`), the tagged
pre-run code (`acar-v2-protocol` @ `9b2f0c1`), or the scientific endpoint. The endpoint **stands**:
G1=True, G2=False, double-run hash `217099da054e96a8` matched, 10 hard guards pass (result commit `1528a94`).

## 1. Corrected status string (use verbatim)

> **MEASUREMENT_ONLY — SCZ empirical coverage diagnostic missed nominal; formal coverage remains conditional on
> exchangeability.**

EVAL subject-event coverage (frozen v2 §A5 requires ≥ 1−α = 0.90):
- **PD = 0.900** exactly (207 / 230 subject clusters) — meets nominal.
- **SCZ = 0.8933** (201 / 225 subject clusters) — **below** nominal: a literal pre-registered **diagnostic miss**.

Exact, copy-safe wording (use verbatim; do NOT write "misses by 24/225" — 24/225 is the *uncovered-subject* count,
not the shortfall in coverage):
> **SCZ coverage was 201/225 = 0.8933, with 24 subjects uncovered; this is 0.67 percentage points below 0.90 and
> two covered subjects short of the integer pass threshold of 203/225.**

Caveats (do not over- or under-claim): the implementation **reports** the A5 coverage diagnostic but does **not
enforce** it in the decision function. A realized test proportion can fall below 0.90 even when the marginal
conformal theorem holds (finite-sample sampling variation), so SCZ=0.8933 is **not by itself** evidence that the
conformal calibration is mathematically broken — it is nevertheless a literal pre-registered diagnostic miss.
**Do not** round 0.893 → 0.90, rerun, or amend the threshold post hoc.

## 2. Mechanistic interpretation (broaden: NOT "the valid bound was merely over-conservative")

The paired measurement signal is **real** (registered same-feature witnesses, PD/SCZ AUROC, out-of-fold):
- matched-CORAL Δentropy ≈ 0.641 / 0.790
- matched-CORAL Δmargin ≈ 0.650 / 0.790
- SPDIM flip-rate ≈ 0.677 / 0.686

But the **learned PD risk-regressors are weak** (ĝ AUROC ≈ 0.526–0.561). The large conformal `q` (≈ 3–5 ≫ |ΔR|≈0.3)
therefore reflects **both** (a) residual prediction error in ĝ_a **and** (b) the subject/action/batch
**joint-maximum** nonconformity construction — **not** conformal calibration alone. The correct framing of the gap is

> **measurement → risk-regression → calibrated-control gap**

(a three-link chain), not merely "measurement → conformal-control." The predictor sees the signal; the regression
does not yet turn it into a tight per-batch ΔR estimate; and the simultaneous bound is consequently loose.

## 3. Closed-loop failure is disease-specific

- **PD:** router NLL reduction 0.090 **<** best-fixed (T3A) 0.128; oracle-benefit retention 0.134; abstention ≈ **97.0%**.
- **SCZ:** router 0.291 **>** best-fixed 0.137, but **cohort-macro fails** and retention is only 0.320; abstention ≈ **94.7%**.

So G2 fails for **different reasons** per disease (PD: loses to best-fixed + low retention; SCZ: cohort-macro + retention).

## 4. Reporting units (do not conflate)

- **455 subject clusters** = the independent coverage/calibration units (230 PD + 225 SCZ).
- **856 batch records** = the per-batch evaluation rows.
`n = 856` is the batch count and **must not** be presented as the number of independent coverage units. The
conformal `m` per fold is the **subject** count (`n_cal`), and that is what `onesided_quantile` used (correct).

## 5. Provenance / reporting DEFECTS (in the manifest; correct only in a future v3 amendment, never in-place)

These do **not** affect the `MEASUREMENT_ONLY` endpoint (the conformal `m` was computed on subjects correctly):
1. **Inconsistent units in `run_manifest.json` `folds[*]`:** `n_fit` and `n_eval` are **batch** counts, while
   `n_cal` is a **subject** count (= the calibration unit `m`). Example PD fold0: `n_fit=200, n_cal=53, n_eval=93`.
   Future fix: label them explicitly (`n_fit_batches`, `n_cal_subjects`, `n_eval_batches`) and additionally log
   `n_fit_subjects` / `n_eval_subjects`.
2. **Truncated dump hashes:** `dump_sha256` stores only the **first 16 hex chars**, not the full 64-char SHA-256.
   Future fix: store the full digest (keep the field name accurate).

## 6. Suggested paper framing

> **Predicting Negative Transfer Is Not Enough: The Measurement–Control Gap in EEG Test-Time Adaptation.**

Defensible claim = a *label-free, action-conditional paired-harm **predictor** of negative transfer* (G1, validated
out-of-fold on PD & SCZ) with a *subject-clustered conformal coverage construction* (PD meets nominal; SCZ
201/225 = 0.8933, 0.67 pp below 0.90). **Not** a deployable conservative router (G2 fails; three-link gap above).

## 7. Disposition

Endpoint `MEASUREMENT_ONLY` is final under the tagged v2 protocol. Any follow-up — operating-point exploration
(δ<0, larger α, per-action `q`, stronger ĝ), the provenance fixes in §5, or the coverage-diagnostic enforcement —
must be a **new dated amendment (v3)**, pre-registered, explicitly exploratory, and validated on **held-out
cohorts**; never an in-place patch to the v2 protocol, code tag, or frozen result.
