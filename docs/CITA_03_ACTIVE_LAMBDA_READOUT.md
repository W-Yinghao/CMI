# CITA_03 — active-CMI probe (λ_cita=1.0) readout: FAIR FAILURE

```
The ONE PM-approved active-CMI probe (NOT a grid). 126 runs: {EEGNetMini, EEGConformerMini} ×
{ERM-no-adapt, TTA-Control, CITA-CMI λ=1.0} × {2a (9), 2015 (12)}, seed0, full-LOSO, offline transductive
target-unlabeled. All three methods per fold adapt the IDENTICAL source-ERM M0 (matched attribution). Isolated
gate_lambda1/ (the λ=0.010 gate untouched). Jobs 886893/886894 COMPLETE. Purpose: at λ=0.010 the CMI term was
near-inert (~0.1-0.2% of loss) so its no-effect was not a fair test; λ=1.0 makes it ACTIVE.
```

## 0. Activation check (the PM's gate) — CMI term is ACTIVE at λ=1.0
| dataset·backbone | cond_domain_fraction_of_total_loss | λ·cond | grad_norm |
|---|---|---|---|
| 2a·eegnet | **1.8%** | 0.035 | 0.22 |
| 2a·conformer | **17.8%** | 0.197 | 0.86 |
| 2015·eegnet | **4.1%** | 0.032 | 0.15 |
| 2015·conformer | **29.3%** | 0.171 | 1.02 |

At λ=1.0 the conditional-domain term is a substantial, active part of the loss — strongly so on the leakage-rich
conformer (17.8% / 29.3%, grad-norm ~1.0), meaningfully on eegnet (1.8-4.1%, grad-norm 0.15-0.22). **This is a
real active-CMI test**, not the near-inert λ=0.010 regime. (Verified pre-launch: λ=1.0 produces a different model
than λ=0.010 and than TTA-Control.)

## 1. Target generalization (matched, same M0)
| dataset·backbone | ERM | TTA | CITA λ1.0 | TTA−ERM | **CITA(λ1)−TTA** | CITA(λ1)−ERM |
|---|---|---|---|---|---|---|
| 2a·eegnet | 0.423 | 0.467 | 0.467 | +0.043 | **−0.000** | +0.043 |
| 2a·conformer | 0.411 | 0.448 | 0.446 | +0.037 | **−0.002** | +0.035 |
| 2015·eegnet | 0.637 | 0.730 | 0.730 | +0.092 | **+0.000** | +0.093 |
| 2015·conformer | 0.640 | 0.715 | 0.713 | +0.075 | **−0.003** | +0.073 |

- **TTA−ERM +0.037…+0.092, positive all 4 cells** — reproduces the λ=0.010 run; target-unlabeled adaptation works.
- **CITA(λ1.0)−TTA ≈ 0 (−0.003…+0.000), even where the term is MOST active** (2015·conformer 29.3% → −0.003,
  marginally worse). The active CMI term adds nothing to target — if anything slightly hurts on the most-active cell.

## 2. R3 reliance (classifier-level head_replay, firewall=True)
| dataset·backbone | ERM k2 | TTA k2 | CITA λ1 k2 | **CITA(λ1)−TTA** | random_ctrl | k8 (tta→cita) |
|---|---|---|---|---|---|---|
| 2a·eegnet | +0.032 | +0.035 | +0.032 | **−0.003** | ~0.000 | +0.137→+0.137 |
| 2a·conformer | +0.047 | +0.040 | +0.040 | **+0.000** | ~0.000 | +0.096→+0.095 |
| 2015·eegnet | +0.027 | +0.049 | +0.050 | **+0.000** | ~0.000 | +0.128→+0.125 |
| 2015·conformer | +0.057 | +0.082 | +0.076 | **−0.006** | ~0.000 | +0.102→+0.099 |

- **CITA(λ1.0)−TTA (R3) ≈ 0** everywhere; the largest is −0.006 on 2015·conformer (the most-active cell) but that
  is within the fold-sd (~0.07). No clear functional-reliance reduction, even with the term active.
- random_subspace control ≈ 0 (valid); removal_mode=head_replay, firewall=True on all folds.

## 3. Collapse guards — no collapse; CITA ≈ TTA
| dataset·backbone | entropy_after (TTA / CITA) | label-balance KL (TTA / CITA) |
|---|---|---|
| 2a·eegnet | 0.699 / 0.703 | 0.008 / 0.008 |
| 2a·conformer | 0.412 / 0.422 | 0.017 / 0.018 |
| 2015·eegnet | 0.290 / 0.295 | 0.006 / 0.006 |
| 2015·conformer | 0.142 / 0.153 | 0.011 / 0.013 |

Entropy stays non-degenerate; label-balance KL small. Strikingly, even though the CMI term is 29% of the loss on
conformer, CITA's adaptation state is barely different from TTA's — the conditional-domain-confusion gradient does
not move the model in a way that changes target, reliance, entropy, or balance.

## 4. Verdict — **CITA-CMI fails fairly** (λ=1.0 NEGATIVE)
By the pre-registered tiers: the CMI term is ACTIVE (activation gate passed), target is retained, no collapse,
random control valid — but **CITA(λ1.0) does not beat TTA-Control on target (≈0, slightly negative where most
active) nor on R3 (≈0, −0.006 max within noise).** This is the "TTA-only again" branch: TTA improves over ERM,
CITA does not beat TTA → **CITA-CMI fails fairly.** The λ=0.010 near-inertness loophole is now closed: the
CMI-specific control objective adds no value over plain target-unlabeled adaptation *even when active*.

### Terminal decision (per the PM's "λ=1.0 negative" branch)
**Close CMI method development.** No more λ sweep, no new CITA objectives, no new backbones, no source-only
revival, no ConformerFull, no external-dataset CMI method search. TTA-Control is retained as a **separate,
non-CMI** positive. Proceed to the project synthesis (see CMI_SYNTHESIS).

**Final synthesis position:**
> CMI is robust as an audit / diagnostic tool. Source-only CMI control (CIGL / FCIGL / dCIGL / MetaCMI) fails to
> deliver functional reliance reduction or target generalization. Target-unlabeled adaptation (TTA-Control)
> improves target performance in all four cells. But CMI-CITA — inactive at λ=0.010 and active at λ=1.0 — does not
> add value over TTA-Control on target or reliance. The CMI *control* objective is closed across both the
> source-only and the target-unlabeled information regimes; the durable contributions are the CMI *audit* and the
> (non-CMI) TTA positive.
```
