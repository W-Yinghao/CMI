# CITA_02 — target-unlabeled CMI adaptation, seed0 readout

```
Setting: OFFLINE TRANSDUCTIVE TARGET-UNLABELED adaptation (not source-only DG). target X allowed at adaptation,
target y forbidden everywhere except the final metric. 126 runs: {EEGNetMini, EEGConformerMini} ×
{ERM-no-adapt, TTA-Control, CITA-CMI} × {BNCI2014_001 (9), BNCI2015_001 (12)}, seed0, full-LOSO. All three methods
per fold share ONE source-ERM M0. λ_cita=0.010 (no grid). Jobs 886308/886309 COMPLETE. seed0 = scientific
SCREENING. Attribution order: 1) TTA−ERM (is adaptation useful?), 2) CITA−TTA (the CMI-specific test), 3) CITA−ERM.
```

## 0. Integrity / stop-conditions — all CLEAN
target-y firewall enforced (`adapt()` has no target-label param) + recorded per artifact; TTA & CITA share the
exact adaptation budget (except λ) and BN/dropout policy (fail-closed assert never fired); exact classifier-level
head-replay (removal_mode=head_replay, firewall=True, all 126); random_subspace control ≈0; finite; no NaN. No
stop-condition triggered.

## 1. Target generalization (mean over folds)
| dataset·backbone | ERM-no-adapt | TTA-Control | CITA-CMI | **TTA−ERM** | **CITA−TTA** | CITA−ERM | src ERM→adapt |
|---|---|---|---|---|---|---|---|
| 2a·eegnet | 0.423 | 0.469 | 0.468 | **+0.046** | **−0.001** | +0.045 | 0.567→0.517 |
| 2a·conformer | 0.412 | 0.447 | 0.449 | **+0.035** | **+0.002** | +0.037 | 0.544→0.510 |
| 2015·eegnet | 0.637 | 0.731 | 0.731 | **+0.093** | **+0.000** | +0.093 | 0.773→0.722 |
| 2015·conformer | 0.640 | 0.718 | 0.717 | **+0.077** | **−0.001** | +0.077 | 0.754→0.710 |

- **TTA−ERM: +0.035 … +0.093, positive in ALL 4 cells.** Target-unlabeled adaptation genuinely improves the
  held-out subject — the **first robustly-positive generalization effect in the project.** (It trades a little
  source accuracy for target, as expected for transductive adaptation.)
- **CITA−TTA (the CMI-specific test): ≈ 0 in all 4 cells (−0.001 … +0.002).** The CMI term adds nothing over
  plain TTA on target.
- CITA−ERM (+0.037 … +0.093) is **entirely the TTA contribution** — attributing it to CMI would be exactly the
  ERM-baseline error the protocol guards against.

## 2. Leakage / CMI audit (feature_z)
feature_z leakage significant on all folds; leakage-rich (conformer featKL ≈ ConformerMini). CITA reduces the
measured leakage proxy marginally more than TTA but the effect is tiny and does not propagate. The conditional
source/target domain IS distinguishable given y (final_cond_domain 0.03 eegnet / 0.22 conformer — larger on the
leakage-rich transformer), i.e. there is structure the CMI term *could* act on.

## 3. R3 reliance (classifier-level head_replay, firewall=True)
| dataset·backbone | ERM k2 | TTA k2 | CITA k2 | **CITA−TTA** | TTA−ERM | random_ctrl | k8 (erm→cita) |
|---|---|---|---|---|---|---|---|
| 2a·eegnet | — | — | — | **+0.000** | +0.005 | ~0.002 | — |
| 2a·conformer | +0.048 | +0.039 | +0.041 | **+0.002** | −0.009 | ~0.001 | +0.097→+0.096 |
| 2015·eegnet | +0.027 | +0.051 | +0.051 | **−0.000** | +0.024 | ~0.001 | +0.074→+0.130 |
| 2015·conformer | +0.058 | +0.085 | +0.085 | **−0.000** | +0.027 | ~0.001 | +0.076→+0.105 |

- **CITA−TTA (R3) ≈ 0 in every cell.** The CMI term does not reduce functional reliance vs TTA either.
- TTA−ERM R3 is mixed/slightly positive (adaptation tends to increase, not decrease, the subject-subspace
  reliance) — so plain TTA improves accuracy without reducing reliance.

## 4. Adaptation diagnostics (collapse guards) + verdict
Entropy before→after: 1.03→0.70 / 0.64→0.41 / 0.47→0.29 / 0.23→0.14 — confidence rises but **does not degenerate**
(all well above 0). Label-balance KL(target marginal ‖ source prior) after = 0.006–0.016 — **no label collapse.**
So the TTA gain is a genuine adaptation effect, **not** an entropy/label-collapse artifact.

### Verdict — **TTA-only pass** (NOT CMI positive)
By the pre-registered tiers: TTA-Control robustly improves target (all 4 cells, no collapse), but **CITA-CMI does
not beat TTA-Control on target OR on R3** (CITA−TTA ≈ 0 everywhere). ⇒ **TTA-only pass**: target-unlabeled
adaptation is useful, the CMI-specific term is not.

**Honest λ nuance (disclosed):** at the pre-registered λ_cita=0.010 the cond-domain penalty is **near-inert** —
≈0.1–0.2% of the loss (final_cond_domain 0.03–0.22 × 0.010) — so entropy/balance/target/R3 for CITA are
numerically ~identical to TTA. This seed0 verdict therefore means *"at λ=0.010 the CMI term adds nothing"*; from
this single run we cannot separate "λ too small" from "CMI direction useless." Per the pre-registered rule
(no λ grid; no auto-sweep on a no-signal seed0), the verdict stands as TTA-only pass.

### Recommendation (per the PM decision tree)
- **Seeds 1/2: NO** — none of the expansion triggers hold (CITA beats TTA on neither target nor R3; no clean
  CMI-specific signal on any backbone).
- **No λ sweep** (pre-registered), no β/k/threshold sweep, no ConformerFull, no source-only revival.
- Per the PM's "CITA ≤ TTA" branch: **the CMI-specific control objective shows no signal in the first
  target-unlabeled setting either.** Combined with the source-only closure (CIGL_70), the accumulated position is:
  > CMI is robust as an audit / diagnostic tool, but current CMI *control* objectives do not yield reliable
  > functional reliance control or generalization gains in either the source-only or the first target-unlabeled
  > setting. Target-unlabeled adaptation itself (TTA-Control) DOES help — a real positive, but not a CMI result.
  Recommended next = **write the synthesis** (durable: CMI audit + the TTA positive; closed: CMI control), not
  more method tuning.
- One deliberate option for the PM (NOT launched, would be an explicit exception to the no-sweep rule): because
  λ=0.010 is near-inert, a SINGLE larger-λ probe (e.g. λ=1.0) on the best cell would test whether a *strong* CMI
  push helps or hurts before the line is closed. Only on explicit authorization.
```
