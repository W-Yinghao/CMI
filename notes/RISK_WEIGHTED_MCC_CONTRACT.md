# Risk-Weighted MCC (RW-MCC) — execution contract (SPEC; NOT a prereg amendment; manuscript FROZEN)

PM 2026-07-18 fixed the weighting signal (no open exploration): **source-LOSO excess pairwise predictive risk**.
Branch `agent/cmi-trace-risk-weighted-mcc`, base 2e1a3eb2. After the estimator audit rejected E1 (K=4 not
variance-limited; EMA deprioritized), the next hypothesis: weight the MCC consistency by SOURCE-ONLY predictive
instability so the objective targets only the mechanism disagreement tied to source generalization failure. Only
the project owner may stop a scientific line.

## Weight definition (per outer target fold; SOURCE continuation-train split ONLY; source-val = ckpt selection only)
Source subjects S, class pair p=(a,b). Features = Z at the ERM WARM-UP checkpoint (weights computed once, frozen).
1. **Pseudo-target readout** h_{-d}: for each source subject d, fit a FRESH linear classifier on the OTHER source
   subjects (D_i ≠ d): StandardScaler.fit(D≠d) + LogisticRegression(L2, C=1, class_weight=balanced, solver=lbfgs,
   max_iter=500). Weight-only; never deployed.
2. **Pairwise balanced log-loss**: restrict + renormalize the multiclass probs to the pair,
   p̃_a = p_a/(p_a+p_b), p̃_b = p_b/(p_a+p_b). Held-out loss on subject d (class-balanced):
   ℓ^hold_{d,p} = −½[ (1/n_{d,a}) Σ_{D=d,Y=a} log p̃_a + (1/n_{d,b}) Σ_{D=d,Y=b} log p̃_b ].
   Reference loss with the SAME h_{-d} on the TRAINING source subjects (subject-balanced):
   ℓ^ref_{-d,p} = (1/(|S|−1)) Σ_{e≠d} ℓ_{e,p}(h_{-d}).
   **Transfer-specific excess risk** r_{d,p} = [ ℓ^hold_{d,p} − ℓ^ref_{-d,p} ]_+ . The subtraction prevents a
   pair that is hard for EVERYONE from being mistaken for subject-specific instability.

## Weight normalization (per outer fold)
1. Winsorize positive r_{d,p} at the 90th percentile (of the positive values).
2. Mean-normalize over all subject-pair cells: w̃_{d,p} = r^win_{d,p} / (mean_{d,p} r^win_{d,p} + ε).
3. Clip: w_{d,p} = min(w̃_{d,p}, 4).
4. Do NOT re-force mean=1 after clipping — when only a few cells are unstable, total regularization strength drops
   naturally instead of blowing up one cell.
If ALL r_{d,p}=0 → status `NO_POSITIVE_SOURCE_TRANSFER_GAP`, all weights 0, RW-MCC a NO-OP for that bundle
(do NOT fall back to uniform MCC — that would disguise "no source-risk signal" as the new objective).
Save per fold: positive_weight_fraction, effective_weight_support=(Σw)²/Σw², max_weight, weight_entropy,
raw/normalized weights, LOSO hold/ref losses, weight_hash.

## Objective
ℓ^MCC_{d,p} = 1 − ⟨u_{d,p}, sg(ū_{-d,p})⟩ (direction-normalized, LOSO consensus, stop-grad, as before).
**L_RW-MCC = (1/(|S||P|)) Σ_{d,p} w_{d,p} ℓ^MCC_{d,p}** ; total L = L_task + λ_RW L_RW-MCC, **λ_RW = 1.0 (fixed, no
sweep** — global MCC at λ=1 had no source damage, so it is the reasonable single strength). Weights computed once
at the warm-up checkpoint and FROZEN for the whole continuation; NO dynamic reweighting this round.

## PRIMARY matched control = Weight-permuted MCC (Arm C)
For each pair p, apply ONE fixed permutation of {w_{d,p}} across source subjects → w^π_{d,p}. Keeps the same weight
multiset, per-pair total weight, MCC formula, batch, gradient budget, sparsity, max weight; breaks ONLY which
source subject's contrast is more constrained. Fairer than a fresh random weight set.

## Arms (3, from the SAME ERM warm-up hash) — 63 bundles × 3 = 189 GPU arms
A = ERM continuation ; B = true RW-MCC ; C = pairwise weight-permuted RW-MCC. NO 4th uniform-MCC arm (the λ=1
global-MCC round is a FROZEN secondary comparator, not a primary gate). Fixed: EEGNet, 20 continuation epochs, LR
1e-4, K=4, same balanced sampler, same source-val checkpoint selection, same warm-up hashes, full encoder trainable.

## Execution order
- **RW0** (this contract + config + weight builder + weight-permute + tests): compute weights on all 63 warm-up
  cells (NO training) → risk_weight_rows.csv / risk_weight_fold_summary.csv / risk_weight_completeness.csv. Audit:
  all-zero bundle fraction, effective support, single-subject domination, weight distribution per dataset, true-vs-
  permuted gradient numerically different, seed stability. Source-only characterization — do NOT use target accuracy
  to decide the weight definition.
- **RW1**: one real GPU bundle — A/B/C same warm-up hash; weights use NO target; B/C same weight multiset; B/C
  gradients differ; source task drop ≤ 0.02; effective rank / contrast norm no collapse; artifacts reconstructable.
  No scientific weight.
- **RW2**: full 189-arm GPU fleet. PRE-AUTHORIZED after RW0/RW1 engineering pass (no further PM decision needed).
  Aggregate only after 189/189.

## Primary endpoints
1. Source mechanism target moved: ΔR^{B−A}_source-LOSO and Δ mean_{d,p} r_{d,p} (does the new objective achieve its
   own goal).
2. Weighted vs unweighted geometry: WSCI_w = Σ w cos(u,ū)/Σ w, and plain unweighted WSCI (did it improve only the
   risk-selected mechanism, or overall geometry).
3. DG utility: primary ΔU_RW−ERM = U_B − U_A; **risk-specific ΔU_RW−WPerm = U_B − U_C — success MUST depend on B−C,
   not just B−A.** Inference unit = target subject; 3 seeds subject-cluster bootstrap + exact sign-flip.
4. Frozen historical: U_RW − U_{global-MCC, λ=1} (warm-up-hash-matched; secondary, not the gate).

## Routing
- **A**: source-LOSO risk↓ AND DG beats BOTH ERM and weight-permuted → source-only instability localizes valuable
  mechanism → next: seeds 3,4 + C-CORAL/IRM same-protocol + DGCNN replication + CMI posterior audit.
- **B**: source-LOSO risk↓ but DG still null → source meta-generalization failure ≠ future target failure → next
  round build weights from source-only cross-SESSION (early→later) instability, not more loss strength.
- **C**: RW-MCC no better than weight-permuted at cutting source risk / moving weighted geometry → failure layer
  `STATIC_LOSO_WEIGHTS_DO_NOT_TARGET_TRAINABLE_MECHANISM` → analyze weight instability / risk-contrast cell mismatch
  / train pairwise margins directly. Do NOT revert to EMA.
- **D**: source-LOSO↑ but target damaged → downweight unstable subjects, or shared+residual decomposition.

## Deliverables (this branch; NOT a prereg amendment)
`tos_cmi/train/risk_weighted_mcc.py` (weight builder + weight-permute + rw_mcc_loss), `scripts/run_risk_weight_audit.py`,
`scripts/aggregate_risk_weight_audit.py`, RW training runner + arms + GPU sbatch, config, this contract, tests. NO
manuscript edit; M2 / projector / TTE / CMI HELD.
