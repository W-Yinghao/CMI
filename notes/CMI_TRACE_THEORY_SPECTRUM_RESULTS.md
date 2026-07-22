# CMI-Trace Theory-Spectrum — RESULTS (neutral)

Fleet `906373` (CPU, EXP=all, 8-way), branch `agent/cmi-trace-theory-spectrum`. Coverage FULL and firewall
clean before aggregation: E1 63/63 fold-cells (firewall_ok 63/63), E2 27+27 dumps, E3 15/15. Aggregation on
the full matrix only. Subject-cluster bootstrap (n_boot=10000). Nothing pushed.

## E1 — subject-info × exact-head reliance spectrum (Theorem 2) — PRIMARY PREDICTION FALSIFIED

Prediction: amount-only CMI control (CIGL) strips λ from LOW-τ (task-orthogonal) subject directions first →
`corr(τ_erm, Δλ) > 0`, top-direction reliance rises, effective rank falls.

| endpoint | BNCI2014_001 | BNCI2015_001 | prediction | outcome |
|---|---|---|---|---|
| corr(τ, Δλ) — cosine pairing | **−0.319 [−0.389, −0.251]** | **−0.130 [−0.199, −0.054]** | >0 | **opposite sign, CI excludes 0** |
| corr(τ, Δλ) — rank pairing | −0.265 [−0.325, −0.202] | −0.129 [−0.198, −0.059] | >0 | opposite sign (robust to pairing) |
| Δ top-dir reliance (CIGL−ERM) | −0.0033 [−0.0044, −0.0022] | −0.0003 [−0.0009, +0.0002] | >0 (rises) | falls / null |
| Δ effective rank (CIGL−ERM) | +0.376 [+0.130, +0.588] | −1.161 [−1.351, −0.965] | <0 (falls) | dataset-split (rises / falls) |
| top-dir reliance ERM→CIGL | 0.0035 → 0.0002 | 0.0001 → −0.0002 | rise | tiny, → ~0 |
| top-2 energy concentration | 0.263 → 0.272 | 0.249 → 0.305 | rise | up (both) |
| top-dir head alignment | 0.0042 → 0.0018 | 0.0026 → 0.0050 | rise | split |

**Grid outcome:** the primary endpoint `corr(τ,Δλ)` is significantly **negative** on both datasets, both
pairings (not >0, not merely null); top-direction reliance does **not** rise; effective-rank change is
dataset-split. → Theorem 2's predicted spectral mechanism is **NOT supported on real EEG.** Falsifier fired.

Caveats on the negative sign (do NOT over-read as a new mechanism): (i) top-direction CE reliance is tiny
(~0.003 nats) — the label-conditional subject directions are nearly UNUSED by the head (consistent with the
prior finding that removable subject leakage sits in the head's near-nullspace); (ii) `Δλ = λ_cigl − λ_erm`
is mechanically anti-correlated with `λ_erm` (high-λ directions have more room to fall), so a negative
`corr(τ, Δλ)` is partly a regression artifact if high-τ directions also carry high λ_erm. The defensible
claim is the FALSIFICATION of `corr>0`, not a positive claim about the negative direction.

## E2 — removability rank r_D + head geometry (Theorem 1) — rank structure holds; exact-head safety does NOT

| endpoint | EEGNet (d_z=16, probe-head) | TSMNet (d_z=210, exact head) |
|---|---|---|
| r_D = k_mean_complete (analytic) | 13.48 [13.30, 13.70] | 23.81 [23.48, 24.15] |
| k_probe_chance (weaker) | 13.67 [13.37, 13.96] | 17.19 [17.00, 17.37] |
| redundancy_rank (r_D − probe) | −0.19 [−0.56, 0.11] (≈0) | **+6.63 [+6.30, +6.96]** |
| subject effective rank | 8.13 [8.02, 8.24] | 15.67 [15.54, 15.81] |
| dim(S_D ∩ row Wᵀ̃), 5° | — (head-free) | 0.00 [0.00, 0.00] |
| min principal angle S_D↔row(Wᵀ̃) | — | 45.0° [43.8, 46.3] |
| logit change removing S_D (rel) | — | **0.53 [0.52, 0.55]** |

**Grid outcome:**
- **Rank threshold — supported (as refined).** r_D is well-defined; `k_mean_complete = r_D` by construction.
  The over-completeness/redundancy story is clean and dimension-dependent: `redundancy_rank ≈ 0` at d_z=16
  (probe reaches chance right at r_D) vs `≈ 6.6` at d_z=210 (probe reaches chance well before mean-completion).
  This replaces the latent-dimension capacity proxy with the theorem's r_D and shows redundancy grows with
  dimension. (`k_probe_chance < r_D` on TSMNet is expected redundancy, NOT a violation — per the owner-steered
  reinterpretation.)
- **Exact-head safety — NOT satisfied on TSMNet EEG.** The condition is `S_D ⊆ ker(WΣ^{1/2})`. Although no S_D
  direction is within 5° of the head row space (dim=0), the **min principal angle is 45°** (cos 45°≈0.71 =
  substantial overlap) and removing S_D changes logits by **53%** of their magnitude. So the label-conditional
  subject subspace is **entangled** with the head's used directions, not in its kernel — subject-subspace
  removal is not logit-safe. (logit change is L2 logit magnitude, not task-loss; a task-loss safety statement
  needs the CE/accuracy version, but the geometric safety condition clearly fails.)

## E3 — K* on beneficial vs legitimate-use worlds (Proposition 2) — VERIFIED

15/15 cells. Exact squared-loss identity holds everywhere (max |Gain⋆ − Gain_direct| = 4.4e-16). Worlds
separate in every cell: beneficial K*<1 & gain>0 (K* mean −3.82, gain [+0.62,+0.76]); legitimate K*>1 &
gain<0 (K* mean +5.24, gain [−0.47,−0.38]); `sign(1−K*)=sign(Gain⋆)`. Proposition 2's sign law is confirmed.

## Firming-up analyses (owner-steered 2026-07-22) — mechanism nailed

**E1 partial correlation (control the λ_erm regression confound).** `corr(τ, Δλ | λ_erm)`:
BNCI2014_001 **+0.064 [−0.024, +0.152]**, BNCI2015_001 **+0.039 [−0.047, +0.122]** (both CIs include 0;
same under rank pairing). Confound legs: corr(τ,λ_erm)=+0.47/+0.19, corr(λ_erm,Δλ)=−0.76/−0.80. → The raw
negative `corr(τ,Δλ)` is **entirely** the mechanical regression (high-λ directions fall more); the partial
correlation is **null**. Clean E1 statement: the λ-ordering carries NO reliance information — no support for
the predicted `corr>0`, and no reverse mechanism. Consistent with entanglement (reliance ⊥ λ-ordering).

**E2 task-loss exact-head safety + free/entangled decomposition (TSMNet, full matrix, subject-cluster CI).**
Removing the label-conditional subject subspace `S_D`:
- **accuracy drop = 0.141 [0.135, 0.147]** (~14 points), **CE increase = 0.313 [0.297, 0.332] nats** — the
  exact-head-safety failure is real TASK damage, not just a 53% logit-magnitude move.
- **logit change from the KERNEL (free) part of S_D = 0.000 [0.000, 0.000]** — removing only the head-null
  part is exactly safe.
- **frac(S_D energy in head rowspace) = 0.050 [0.048, 0.052]** — only ~5% of S_D is entangled; ~95% is
  head-null/free.
→ **Free/entangled decomposition confirmed and quantified:** total label-conditional subject subspace =
~95% head-null (removal-safe; reconciles with the prior 71–78% head-nullspace "free" leakage) + ~5%
entangled with the head's used directions, and that ~5% sliver carries a ~14-point accuracy cost. Removing
the free part alone is a no-op; removing S_D as a whole hurts because of the entangled part.

## Synthesis (tentative — owner-controlled framing)
E3 (an algebraic identity) holds. The two theorems whose EEG mechanism depends on the subject subspace being
task-ORTHOGONAL / head-SAFE are NOT borne out on real EEG: E1's predicted spectrum fails (corr negative,
reliance not rising) and E2's exact-head-safety condition fails (S_D at 45° to the head, 53% logit change).
Both point the same way — **on real EEG the label-conditional subject subspace is task-entangled, not
orthogonal to the head** — contradicting the stylized shared-covariance premise. This is consistent with the
project's measurement→control theme (safe erasure ≠ free). Manuscript implications (Fig 2B / the corr
sentence / the exact-head-safety claim) are for the owner to decide.
