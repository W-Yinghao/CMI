# CIGL_69A — Phase-1 ERM preflight readout (seed0, full LOSO, source-only)

```
42 fold-records: {EEGNetMini, EEGConformerMini} × ERM × {BNCI2014_001 (2a, 9 subj, 4-class),
BNCI2015_001 (12 subj, 2-class)}, seed 0, strict source-only LOSO, source-val early-stop, target eval-only,
feature_z audit (head-replay verified). Jobs 885120/885121 complete. Scientific screening tier = seed0.
```

## Results (mean ± sd over folds)
| dataset | backbone | n | target_bacc | source_bacc | feature_KL | perm_p<0.05 | replay_ok (max\|Δ\|) |
|---|---|---|---|---|---|---|---|
| 2a (4-cls) | EEGNetMini | 9 | **0.423 ± 0.144** | 0.567 | 1.088 | 9/9 | ✓ (1.6e-6) |
| 2a (4-cls) | ConformerMini | 9 | **0.411 ± 0.118** | 0.543 | 1.456 | 9/9 | ✓ (2.3e-6) |
| 2015 (2-cls) | EEGNetMini | 12 | **0.637 ± 0.102** | 0.773 | 0.676 | 12/12 | ✓ (1.4e-6) |
| 2015 (2-cls) | ConformerMini | 12 | **0.640 ± 0.097** | 0.754 | 1.485 | 12/12 | ✓ (1.8e-6) |

Δ(Conformer − EEGNet) target = **−0.012** (2a), **+0.003** (2015).

## 1. Integrity
- **Head-replay exact on real EEG** for both backbones, all 42 folds: max |logits − (feature_z·Wᵀ+b)| ≤ 2.3e-6
  ⇒ the classifier-reliance (head-replay) R3 claim is valid on feature_z, not only a probe claim.
- **No collapse.** Source bacc 0.54–0.77; target above chance everywhere (2a 0.41 vs 0.25 chance; 2015 0.64 vs
  0.50). Conformer source (0.54/0.75) is not degenerate vs EEGNet (0.57/0.77).
- Firewall metadata on every record: source-only training, source-val selection, target eval-only,
  projector-source-train-only. No target labels used for fit/selection.

## 2. Backbone comparison — Conformer is at PARITY, not stronger
- ConformerMini ≈ EEGNetMini: **Δ = −0.012 (2a), +0.003 (2015)** — both **within one fold-sd** (0.10–0.14),
  i.e. statistically indistinguishable at seed0.
- The high-capacity arm did **not** beat the anchor at ERM. This is the honest worst cell: on 2a Conformer is
  0.012 *below* EEGNet — just past the −0.01 healthy tolerance.
- **This does NOT mean "Conformer doesn't help."** ConformerMini is the internal *faithful-minimal* transformer
  (emb 32, depth 2, 4 heads); parity-at-ERM is consistent with Mini-capacity being the limiter. Per the
  guardrail, a full/official (or equal-param) Conformer is required before any Conformer-family statement →
  motivates **CIGL_69A2 full-Conformer preflight**.

## 3. Auditability (the point of the preflight)
- **feature_z leakage is live and significant on real EEG:** permutation p < 0.05 on **all 42 folds** (9/9,
  12/12, both backbones). The R3 / head-replay / reliance machinery runs unchanged on feature_z.
- **Conformer carries MORE measured subject leakage** than EEGNet (feature_KL 1.46 vs 1.09 on 2a; 1.49 vs 0.68
  on 2015). So the high-capacity arm is the *more* interesting MetaCMI target: more label-conditional subject
  structure to (attempt to) control, at equal task accuracy.

## 4. Decision
**CONDITIONAL PASS.** Not a healthy_pass (Conformer did not beat EEGNet; 2a is −0.012, past the −0.01 tol), not
a fail (nothing collapses; auditable everywhere). ConformerMini is a **valid, auditable, non-degenerate**
high-capacity arm at parity accuracy with more leakage to control — usable for the Phase-2 MetaCMI mechanism
test — **but Mini parity is not a Conformer-family verdict.**

### Recommended next (for PM approval; NO GPU launched yet)
1. **CIGL_69A2 full-Conformer preflight** (CPU + import check): official braindecode `EEGConformer` if the env
   imports it, else an in-repo equal-param `EEGConformerFull` (official arch: emb 40, depth 6, 10 heads, MLP
   head). Check: feature_z extraction, probe-compatible head (MLP head ⇒ R3 falls back to source-fit probe,
   replay_ok=False — allowed), source-only LOSO path, no target leakage, R3-consumable artifact, one-fold CPU
   smoke. THEN decide full-Conformer LOSO seed0.
2. **Phase-2 MetaCMI seed0** on {EEGNetMini, ConformerMini} × {2a, 2015} (ERM comparators reused, so +metace +
   metacmi_direct β0.1 only). If any positive signal appears, it must be **re-validated on the full/official
   Conformer** before a "CMI produces a stronger model" claim.

Held: no GPU for Phase 2 / A2 until PM approves; no β=0.5 first gate; no external data; static-DGCNN route stays
frozen.
```
```
