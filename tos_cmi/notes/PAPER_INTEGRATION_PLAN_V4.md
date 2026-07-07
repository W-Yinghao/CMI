# Paper integration plan V4 — full restructure after Fork 1/Fork 2 closeout

Branch `paper-v4-integration` (from tag `tos-cmi-science-final-v1` @ a147817). **PLANNING DOC ONLY — no `.tex`
edits until explicit PM go.** Supersedes `PAPER_INTEGRATION_PLAN_V3.md`. The uploaded PDF is a STALE 2a-only
snapshot (Methods/Results discuss only BCI-IV-2a + two backbones; limitations still say "Single dataset"; no
multi-dataset C12, Track B, V2, or Fork 1/Fork 2). **V4 is a substantive restructure, not a patch.** Venue: TMLR.

## 1. New thesis
> Selective conditional invariance in EEG is a SOURCE-ONLY DECISION problem: conditional domain leakage is
> measurable, erasure is not control, refusal is often the correct action, and acceptance requires either a
> source-visible shift or enough target information — both of which have nontrivial limits.

## 2. Title
Recommended:
> **Selective Conditional Invariance in EEG: Measurement, Refusal, and the Source-Only Acceptance Ceiling**

Alternative:
> From Leakage Measurement to Certified Refusal: Selective Conditional Invariance in EEG

Recommend the first (puts the final theoretical finding — the source-only acceptance ceiling — in the title). The
old title ("Selective Conditional Invariance in Task-Orthogonal EEG Subspaces") is too narrow: the projector/
subspace is NOT the contribution (Track G showed LEACE dominates TOS_VD deletion at equal task cost).

## 3. Main-text structure (8 sections)
```
1. Introduction
2. Framework: measurement -> intervention -> certificate -> refusal
3. Source-only theory: non-identifiability (P1) and source-rich sufficiency (P2)
4. Synthetic controls: why geometry and weak gates unsafe-accept
5. Real EEG validation: erasure is not target-DG control
6. Source-only controller: Track B refusal and V2 acceptance ceiling
7. Escaping the ceiling: source-rich environments and target-information limits
8. Discussion
```

## 4. Core results that MUST appear in the main text (compact tables / summary figures only)
1. **Concept-erasure controls** (Track G): LEACE/RLACE/INLP/TOS/random on 2a; the score-Fisher projector is NOT
   the contribution (LEACE dominates TOS_VD deletion at equal task cost); INLP over-erases. -> §5.
2. **Capacity x architecture factorial** (C11, 3-seed): the TSMNet-vs-EEGNet removability gap is LARGELY latent
   dimension (~2/3), with a RESIDUAL architecture interaction at high d_z. Do NOT write "resolved". -> §5.
3. **Multi-dataset C12/C13 validation** (2a/2b/Lee/Cho/HGD; 9/9 valid cells): no source-fitted eraser gives a
   practically meaningful target-bAcc gain; task-safety heterogeneous (binary Lee/Cho EEGNet driven to chance). -> §5.
4. **Track B source-only gate** (C14): rejects harmful/useless real EEG erasures; 0 false accepts in the sampled
   Lee/Cho pilot; naive leakage/domain-gain controllers false-accept. Caveat: acceptance power untested on real EEG. -> §6.
5. **V2 source-only acceptance ceiling** (C14/C15): EEGNet World A = clean ceiling (Stage 2, 72k tasks, 0 principled
   accept, oracle-supported); World B/C robust refusal both backbones; TSMNet World A NOT cleanly demonstrable
   (state the caveat). P1 non-identifiability + P2 source-rich sufficiency + P3 sample-complexity bridge. -> §3,§6.
6. **Fork 2 source-rich partial positive** (C16): source-rich environments make a target-beneficial shift
   source-visible in semi-synthetic EEGNet Lee->Cho (E_oracle accepts, covariance env recovers); but discovery is
   representation-dependent and FAILS on high-dim TSMNet (covariance-regime AMI 0.365 EEGNet vs -0.011 TSMNet). -> §7.
7. **Fork 1 target-information sample-complexity limit** (C17): the B4 oracle confirms target-beneficial cells
   exist, but a hardened finite-sample bounded LCB gives 0 deployable accepts up to the full 50 labels/class
   calibration budget (best cal-LCB -0.53/-0.59, clipped). Few-shot safe certification is sample-complexity
   limited. -> §7.

## 5. Appendix (full detail; keep out of main)
```
full multi-dataset C12/C13 tables (per dataset-backbone cell)
full Track B table + sampled-pilot provenance
full V2 Stage-2 table (source_subject_counts x seeds)
Fork 2 Lee/Cho EEGNet + TSMNet source-rich tables + covariance-regime AMI
Fork 1 label-budget frontier tables (k=1..50) + bounded-LCB (raw+clipped) + B4 oracle
proof sketches for P1 / P2 / P3
concept-erasure baseline table (Track G) + capacity factorial (C11)
artifact / provenance / manifest-and-token discipline notes
```

## 6. Page budget (control scope; do not over-expand)
```
TMLR main            : 16-18 pages
Theory box (P1/P2/P3): <= 1 page
Real EEG validation  : <= 2 pages
Track B + V2         : <= 2 pages
Fork 1 + Fork 2      : <= 1.5 pages combined
Appendix             : carries all full tables / proofs / provenance
```
Main text: compact summary tables + summary figures only. Do NOT paste raw CSV result tables into the main text.

## 7. Forbidden claims (grep-guard before submission)
```
source-only gate has acceptance power
few-shot target labels solve the ceiling
source-rich environments solve source-only acceptance
conditional invariance never helps
target labels are useless
oracle selector is deployable
TSMNet World A cleanly confirms the ceiling
this is a real EEG target-gain method
the dim<->type confound is resolved / not architecture-type
erasure improves target domain generalization
```

## 8. Allowed framings
```
the source-only gate has refusal power (not acceptance power)
target-beneficial erasure can be source-uncertifiable (the acceptance ceiling)
source-rich environments can make benefit source-visible in a semi-synthetic EEGNet setting, but environment
  discovery is representation-dependent and fragile
target labels reveal that source-invisible beneficial interventions can exist, but under a valid finite-sample
  certificate k<=50 labels/class is insufficient to safely license deployment
the removability contrast is LARGELY latent-dimension with a residual architecture interaction at high d_z
```

## 9. Discipline
Science is CLOSED (tag tos-cmi-science-final-v1). No more experiments / Track E / Tier-2 / estimator R&D /
source-rich redesign / new datasets. This branch is manuscript-integration only. `.tex` edits await explicit PM go
(this doc is the plan, not the rewrite). Every main-text number must trace to a `claim_evidence_table.md` row
(C1-C17) and a `CLAIMS_LEDGER.md` block. See [[tos-paper-writeup]], [[tos-cmi-method-deepen-v2]].
