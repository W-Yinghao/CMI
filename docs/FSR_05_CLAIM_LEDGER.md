# FSR_05 — Claim Ledger

**Project FSR — Step 2B.** Paper-ready status for every candidate claim, with the exact allowed wording and the forbidden wording. Statuses are **derived from the RQ result JSONs** by `scripts/fsr/build_claim_readiness.py` (not hand-set) and cross-checked against the PM's pre-stated expectation — all 10 match. Machine table: `results/fsr_phase2b/claim_readiness_table.csv`.

Status vocabulary: `READY` (state as a finding) · `READY_WITH_CAVEAT` (state with the caveat attached) · `SUPPORT_ONLY` (context, not a headline) · `NOT_READY` · `FORBIDDEN`.

| id | status | claim |
|---|---|---|
| C1 | READY_WITH_CAVEAT | Measured leakage magnitude does not certify reliance. |
| C2 | READY_WITH_CAVEAT | Task-head alignment is closer to reliance than raw leakage in frozen CIGL R3. |
| C3 | READY | Subject signal is erasable. |
| C4 | READY | Erasure strength does not certify target benefit. |
| C5 | READY_WITH_CAVEAT | Random-k falsifies non-specific NLL movement. |
| C6 | READY | Spatial branch is load-bearing. |
| C7 | READY | Branch-local leakage/reliance is missing. |
| C8 | READY | CMI-control remains closed. |
| C9 | SUPPORT_ONLY | TTA-Control is positive but non-CMI. |
| C10 | READY | FSR is an audit framework, not a new DG method. |

---

### C1 — Measured leakage magnitude does not certify reliance. — **READY_WITH_CAVEAT**
- **Ladder:** L1 → L5. **Evidence:** CIGL (RQ1B, RQ1C).
- **Allowed:** "In the recomputable seed0 slice, graph leakage is *negatively* associated with functional reliance (ρ=−0.42, n=42); the frozen pooled association is also negative (ρ=−0.342)." Both contradict the old 'more leakage ⇒ more reliance' direction.
- **Caveat (mandatory):** the pooled n=126 leakage correlation is `FROZEN_NOT_RECOMPUTABLE` (seeds 1/2 per-fold graph_kl were pruned); state it as a frozen-verified support result, not a full recomputation.
- **Forbidden:** "we fully reproduced the pooled leakage correlation"; "high leakage means the model relies on it."

### C2 — Task-head alignment is closer to reliance than raw leakage. — **READY_WITH_CAVEAT**
- **Ladder:** L4 vs L1 → L5. **Evidence:** CIGL (RQ1A, RQ3).
- **Allowed:** "Task-head alignment is positively associated with reliance (ρ=+0.338, n=126) and is the correctly-signed predictor; the Spearman difference align−graph_kl excludes zero (+0.816, seed0), so alignment is more positively associated with reliance than raw leakage."
- **Caveat (mandatory):** dataset-heterogeneous (2a significant, 2015 ns); within-group partial betas are not individually significant; alignment is **not** a validated estimator.
- **Forbidden:** "align_k2 is a validated reliance estimator"; "alignment has larger effect magnitude than leakage" (graph_kl has larger |β|, wrong sign).

### C3 — Subject signal is erasable. — **READY**
- **Ladder:** L3. **Evidence:** TOS_LEACE / mean_scatter / INLP / RLACE.
- **Allowed:** "LEACE drives linear subject decodability to chance on both backbones; INLP drives it to chance."
- **Caveat:** a nonlinear MLP residual persists after linear erasure (erasable ≠ fully removed).
- **Forbidden:** "LEACE removes subject leakage" (nonlinear residual remains).

### C4 — Erasure strength does not certify target benefit. — **READY**
- **Ladder:** L3 → L6. **Evidence:** the 5 TOS erasers (RQ2).
- **Allowed:** "Across 40 cells, erasure strength is *negatively* associated with target bAcc (ρ=−0.42 all, −0.44 clean; both exclude 0); no eraser certifies a proven target benefit (0/40)." More removal → worse target.
- **Forbidden:** "erasing subject signal improves DG"; counting a positive point estimate whose CI includes 0 as a benefit.

### C5 — Random-k falsifies non-specific NLL movement. — **READY_WITH_CAVEAT**
- **Ladder:** L3 control → L6. **Evidence:** TOS_random_k vs TOS_LEACE (RQ2).
- **Allowed:** "On 2a-TSMNet, LEACE's NLL drop (−0.031) is matched by same-k random removal (−0.034) while the subject stays decodable (0.998) → the NLL movement is non-specific regularization, not domain removal."
- **Caveat (mandatory):** flagged non-specific in 1/8 LEACE-vs-random_k cells — the claim is scoped to the flagged (canonical 2a-TSMNet) cell, not universal.
- **Forbidden:** "LEACE improves target NLL" as a DG claim.

### C6 — Spatial branch is load-bearing. — **READY**
- **Ladder:** L4. **Evidence:** FBCSP_LGG_branch_ablation (RQ4 descriptive).
- **Allowed:** "The spatial branch is load-bearing (zero_spatial ablation drop +0.074 on 2a / +0.088 on 2015; gate weight 0.489 / 0.572, highest of the three branches)."
- **Forbidden:** "spatial leakage is harmful" (no per-branch leakage probe).

### C7 — Branch-local leakage/reliance is missing. — **READY**
- **Ladder:** L1/L5 per branch (absent). **Evidence:** RQ4 (blocked).
- **Allowed:** "RQ4 is blocked, not failed: per-branch leakage probe (L1) and per-branch functional reliance (L5) do not exist on disk (0 frozen embeddings, 0 per-branch probe); status BLOCKED_MISSING_METRIC for every branch."
- **Forbidden:** "per-branch CMI predicts reliance."

### C8 — CMI-control remains closed. — **READY**
- **Ladder:** frozen premise. **Evidence:** CIGL_70, CMI_SYNTHESIS_01, CITA_03.
- **Allowed:** "Source-only and target-unlabeled CMI *control* are closed; FSR builds on the audit/measurement, not a revived control objective."
- **Forbidden:** any CIGL/FCIGL/dCIGL/MetaCMI/CITA-CMI positive-method claim.

### C9 — TTA-Control is positive but non-CMI. — **SUPPORT_ONLY**
- **Ladder:** L6. **Evidence:** TTA_Control_non_CMI (CITA_02/03).
- **Allowed (context only):** "Target-unlabeled adaptation (TTA-Control) improves target +0.037…+0.093 in all four cells — a genuine but **non-CMI** positive."
- **Caveat (mandatory):** seed0 only (no seeds 1/2); must be walled off from CMI-control; it is support/context for the FSR thesis, not an FSR headline.
- **Forbidden:** presenting it as evidence CMI-control works; treating the +gain as a CMI effect.

### C10 — FSR is an audit framework, not a new DG method. — **READY**
- **Ladder:** L1–L6 relationships. **Evidence:** the whole ledger.
- **Allowed:** "The contribution is the measurement→reliance→control audit ladder and the two boundary findings — *measurable ≠ relied-upon* and *erasable ≠ beneficial* — not a new DG method."
- **Forbidden:** any SOTA / new-DG-method framing.

---

## Wording discipline (applies to every write-up)

- Never state a leakage or erasure *measurement* as a reliance or benefit *conclusion* (C1/C4 are the guardrails).
- Always attach the mandatory caveat to C1, C2, C5, C9.
- The raw `improves_target` flag is `raw_improves_target_flag`; only `benefit_claimable=YES` (proven bAcc gain, no collapse, no harm, specific) licenses a benefit claim — and it is 0/40.
- Provenance tiers are load-bearing: `RECOMPUTED` (align n=126) ≠ `RECOMPUTED_SIGN_ONLY` (graph_kl seed0) ≠ `FROZEN_NOT_RECOMPUTABLE` (graph_kl pooled).
