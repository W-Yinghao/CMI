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
| C11 | READY | FSR detects, localizes, and attributes controlled injected harmful shortcuts. |
| C12 | READY_WITH_CAVEAT | E4 repairs controlled first-moment constant-offset shortcuts. |
| C13 | FORBIDDEN | E4 repairs general or natural shortcuts. |
| C14 | NOT_READY (PENDING_PHASE4G) | E4b repairs controlled second-moment shortcuts. |
| C15 | NOT_READY (GPU_PAUSED) | PC2 learned-reliance repair works. |

> C11–C15 are the **repair-line** claims (Phases 4C–4G + PC2), derived from `pc1_verdict.json`,
> `phase4d/4e/4f_verdict.json` (hand-set from the frozen verdicts, each adversarially verified). C1–C10 above are
> the audit-line claims from the Step-2B RQ JSONs.

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

### C4 — Erasure strength does not certify target benefit. — **READY** (main claim)
- **Ladder:** L3 → L6. **Evidence:** the 5 TOS erasers (RQ2 + RQ2 sensitivity, Step 2C).
- **Allowed (main, robust):** "No eraser certifies a proven target benefit — `benefit_claimable = 0/40` (positive bAcc point **and** 95% CI lower > 0, no task-collapse, no binary-harm, not random-k-matched)." This is the robust basis and is independent of the eraser subset.
- **Supporting negative-association result — `NOT_ROBUST_DO_NOT_HEADLINE`.** The all-cells correlation `corr(E_subject_removed, target_bAcc)` is −0.42 [−0.68,−0.11], but Step 2C shows it is **not robust**: it flips to **+0.54 (excludes 0)** on the principled-eraser subset (LEACE/RLACE only) and is ns when INLP and/or random_k are dropped. It is driven by INLP over-erasure and the random-k anchor, not a real "more removal → worse target" law. **Do not report the negative correlation as a finding.**
- **Forbidden:** "erasing subject signal improves DG"; "more subject removal harms the target" (the negative correlation is a confound artifact); counting a positive point estimate whose CI includes 0 as a benefit.

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

### C11 — FSR detects, localizes, and attributes controlled injected harmful shortcuts. — **READY**
- **Ladder:** L1–L6 on an injected positive control. **Evidence:** PC1-S (`pc1_verdict.json`, FSR_19/20).
- **Allowed:** "On an injected subject-token positive control, FSR detects the induced target harm (+0.041/+0.066 bAcc at α=1/2), localizes it to the injected spatial branch (0.041 > 0.021 graph > 0.020 temporal), and attributes it exactly (token subtraction recovers 1.0); `harm_localization_attribution_pass = true`."
- **Caveat:** injected positive control, not a natural shortcut (4B natural = `NO_VERIFIED_HARMFUL_BRANCH_SHORTCUT`).
- **Forbidden:** "FSR found a natural harmful shortcut"; framing PC1 as evidence natural prevalence creates the shortcut.

### C12 — E4 repairs controlled first-moment constant-offset shortcuts. — **READY_WITH_CAVEAT**
- **Ladder:** L6 repair on an injected first-moment positive control. **Evidence:** Phase 4F (`phase4f_verdict.json`, FSR_26/27); Phase 4D/4E `none` (erasure + counterfactual head fail).
- **Allowed:** "E4 first-moment mean alignment provides a target-X-only repair for a controlled first-moment constant-offset injection, with a small shortcut-specific gain (absolute +0.033 bAcc, clustered CI [0.009, 0.058]) after netting the generic TTA benefit and beating a random-direction control." `repair_claim_level = strong`, qualified `strong_within_controlled_first_moment_scope`.
- **Caveat (mandatory, all):** 73% of the recovery is a **mechanical identity** (first-moment aligner inverts a first-moment offset by algebra; non-identity netted 0.68); it **fails leave-one-dataset-out** (BNCI2014 harm/specificity not established — BNCI2015-carried); the pre-registered LOSO-seed axis is near-inert; E4 **overshoots** orig and helps clean models → generic domain-mean TTA, not surgical removal.
- **Forbidden:** "E4 is a general shortcut repair method"; "E4 surgically removes shortcut information"; "E4 repairs natural EEG subject shortcuts"; leading with the 0.93 ratio.

### C13 — E4 repairs general or natural shortcuts. — **FORBIDDEN**
- **Why:** E4's scope is a construction-matched first-moment positive control that fails leave-one-dataset-out; 4B/4D natural = `none`. No evidence licenses general/natural repair.
- **Forbidden:** any statement that E4 (or FSR) repairs natural, learned, or general shortcuts / harmful subject leakage; any DG/SOTA framing.

### C14 — E4b repairs controlled second-moment shortcuts. — **NOT_READY (PENDING_PHASE4G)**
- **Status:** the Phase-4G controlled second-moment positive control (CPU) has not run. Do not claim; the entry exists so a 4G pass is scoped to `controlled_second_moment_only` and a 4G fail is reportable as "repair is first-moment-specific."
- **Forbidden (until 4G):** any claim about second-moment / covariance / learned repair.

### C15 — PC2 learned-reliance repair works. — **NOT_READY (GPU_PAUSED)**
- **Status:** PC2 GPU is **paused** (`pc2_gpu_run_authorized = false`); the PC2-E4 readiness preflight (FSR_28/31) is design-only. Learned reliance ≠ clean first/second-moment offset; guarded/possibly-negative expectation.
- **Forbidden:** any statement that PC2 / learned-reliance repair works or is being run; presenting PC2 in results (future-work/preflight only).

---

## Wording discipline (applies to every write-up)

- Never state a leakage or erasure *measurement* as a reliance or benefit *conclusion* (C1/C4 are the guardrails).
- Always attach the mandatory caveat to C1, C2, C5, C9.
- The raw `improves_target` flag is `raw_improves_target_flag`; only `benefit_claimable=YES` (proven bAcc gain, no collapse, no harm, specific) licenses a benefit claim — and it is 0/40.
- Provenance tiers are load-bearing: `RECOMPUTED` (align n=126) ≠ `RECOMPUTED_SIGN_ONLY` (graph_kl seed0) ≠ `FROZEN_NOT_RECOMPUTABLE` (graph_kl pooled).
