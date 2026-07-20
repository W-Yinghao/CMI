# C6 — BNCI2014_001 LOSO seed-0: interpretation (independently verified)

> Companion to the descriptive `C6_BNCI001_LOSO_SEED0.md`. Verdict cross-checked by an independent 3-lens
> review (leakage semantics from code / numerical replication from the raw JSON / adversarial refutation).
> **Single seed, single dataset — the report header's "not the final multi-seed, multi-dataset efficacy
> result" applies. All conclusions are provisional and descriptive (no significance test yet → C7).**

## Verdict: no detectable held-out benefit from OACI (STANDS-with-caveats)

**The selection-time leakage reduction does not transfer to the held-out audit.**
- SELECTION-time extractable-leakage UCL (the quantity OACI's selector minimises, on `source_train`):
  Δ(OACI−ERM) = **−0.335 (L0) / −0.290 (L1), 9/9 folds reduced.**
- HELD-OUT AUDIT UCL (on `source_audit`, disjoint from all training/selection — the honest split):
  Δ(OACI−ERM) = **+0.012 (L0) / +0.005 (L1), only 3–4/9 folds reduced.**
- The reduction is **overfitting to the selection criterion** (same functional, same probe family/budget —
  capacities 32/128/512, α=0.1, n_bootstrap=200 — re-scored on a held-out split; the codebase itself
  annotates the selection UCL "post-selection; optimistic … recompute on audit split"). This is the same
  selection-optimism the survivor audit found sank the LPC line, reproduced by OACI's own audit split.

**Honesty corrections (the audit is underpowered — do NOT over-read):**
- Frame the audit result as **"no *detectable* held-out leakage reduction," not "OACI leaks more."** The
  held-out audit is a 2-domain problem on ~1152 windows; |Δ_audit| < the bootstrap width in 9/9 folds at
  both levels; a sign test on the Δ_audit direction gives **p=0.51 (L0) / p=1.00 (L1) — a statistical
  null.** The design cannot resolve a true reduction smaller than ~0.05–0.10.
- **Do not present −0.335 vs +0.012 as a like-for-like effect size.** Selection UCL is a 6-domain problem
  (ERM ~1.67); audit UCL is a 2-domain problem (ERM ~0.70) — a ~2.4× scale/ceiling difference. The
  overfitting evidence is the **within-split Δ contrast** (large-negative on selection, ≈0 on audit), not
  the cross-split magnitude.
- **target-001** is the worst audit fold (Δ_audit +0.086/+0.083, wrong direction) but also the
  highest-variance fold (~2× mean bootstrap width) — its increase is within its own noise. Report
  separately; do not read as demonstrated harm. Excluding it leaves Δ_audit ≈ 0 (median +0.007).

**No accuracy benefit; OACI is risk-feasible; no ERM-collapse.**
- Δ target bAcc (OACI−ERM) = −0.009 (L0) / −0.016 (L1); 4/9 and 2/9 folds improved.
- OACI stays risk-feasible (selected_risk ≈ ERM, 0/9 infeasible) and did **not** collapse to ERM (distinct
  checkpoints, selected epochs 9–199) — it genuinely trained a leakage-penalised model whose penalty does
  not generalise to the held-out audit.

**OACI is not distinguished from the trivial alignment baselines on any held-out metric** (level 0, Δ vs ERM):

| method | Δ audit_ucl (leakage) | Δ bAcc | Δ ECE |
|---|---|---|---|
| OACI | +0.012 (3/9 reduce) | −0.009 | −0.023 (8/9) |
| global_lpc | +0.004 (3/9) | −0.001 | **−0.049** (8/9) |
| uniform | +0.004 (3/9) | −0.001 | **−0.049** (8/9) |

- The **level-0 ECE improvement is a generic Stage-2 conditional-alignment side effect, not an OACI win** —
  `global_lpc`/`uniform` achieve ~2× the ECE gain, and OACI is the *weakest* of the three (and worse still
  at level 1: OACI 4/9 vs uniform 7/9). Consistent with the survivor-audit finding that calibration wins
  are an alignment/temperature side-effect, not method-specific.

## Alternative reading that does NOT exonerate the method
The selection (6 source domains) vs audit (2 held-out source domains) gap could partly be train/audit
**domain shift** rather than pure criterion-overfitting. Either way the operational conclusion is identical:
**selection-time leakage control does not transfer to held-out domains.**

## Pending (C7)
A pre-registered paired test / across-fold CI on the aggregate Δ (none exists in the report yet), then
multi-seed / multi-dataset (BNCI2014_004) before any efficacy claim.
