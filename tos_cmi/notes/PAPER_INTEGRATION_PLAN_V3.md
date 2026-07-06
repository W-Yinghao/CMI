# Paper integration plan V3 (method-deepen -> manuscript) --- PLAN ONLY, no .tex edits yet

Prereq: method-deepen line FROZEN (tag tos-cmi-method-deepen-v2-final @ 0ef7be3); theory spine locked
(notes/CEILING_THEORY.md @ ea88f5e). Venue = TMLR (two-build main.tex + tmlr_main.tex). Claim contract =
claim_evidence_table.md C14 (method-deepen results) + C15 (ceiling theory) + CLAIMS_LEDGER. This is a PLAN; the
manuscript (`tos_cmi/paper/sections/*.tex`) is NOT edited until the PM approves "start manuscript rewrite".
The uploaded PDF is a STALE 2a-only snapshot (limitation still says "Single dataset"; no C12/Track-B/V2) -->
the rewrite is a STRUCTURAL rearrangement to V3, NOT a small diff on the old PDF.

## 1. New paper thesis (put at the top of the manuscript)
> Measurement works; erasure is not control; source-only selective invariance is refusal-first; deployment-shift
> benefit may be source-uncertifiable unless the shift is source-visible or target information is available.

Do NOT position the paper as "TOS-CMI finds a better task-orthogonal eraser" (the projector is not the
contribution) or as "conditional invariance never helps" (too strong). The contribution is the certified
DECISION (accept / reject / abstain) and its source-only ceiling.

## 2. Title options (primary first)
1. **Selective Conditional Invariance in EEG: Measurement, Refusal, and the Source-Only Acceptance Ceiling** (primary)
2. From Leakage Measurement to Certified Refusal: Selective Conditional Invariance in EEG
3. (fallback) When to Refuse to Erase: A Source-Only Decision Theory for Conditional Invariance in EEG
Old title "Task-Orthogonal EEG Subspaces" is too narrow (projector is no longer the core; source-only decision
theory is).

## 3. Main-text section outline (V3)
```
1. Introduction
2. Framework: measurement -> intervention -> certification -> refusal
3. Source-only ceiling theory (Theory Box)
4. Synthetic certification: safety requires refusal
5. Real EEG results
6. Discussion / implications
```
* **S2 Framework (downplay projector):** localization + deletion are DIAGNOSTIC / candidate-intervention
  construction; the contribution is the certified decision, not the eraser. Keep: M-metric deletion diagnostic
  (frozen EEG), the certified task-protected direct-sum variant as a framework component, concept-erasure
  methods as BASELINES/controls (not defeated competitors).
* **S3 Theory Box (<=0.75 pg):** Proposition 1 (source-only non-identifiability) + Proposition 2 (source-rich
  sufficiency) + a one-sentence bridge to target labels ("Target information can break the ceiling; we leave
  target-information budget experiments to future work."). P3 detail + proof sketches -> appendix.
* **S4 Synthetic certification (rename "Synthetic certification: safety requires refusal"):** keep the synthetic
  necessity result = geometry alone != safety; the gate must be allowed to refuse. Supports "gate should not
  default-accept". Compress figures.
* **S5 Real EEG (LOGICAL order, not historical):**
  1. 2a mechanism (TSMNet collapse/redundancy + EEGNet removable-but-no-benefit)
  2. Strong erasure controls (LEACE/RLACE/INLP/TOS/random: projector not the contribution; linear erasure
     stronger but nonlinear residual / task cost remain)
  3. Capacity x architecture factorial (latent dimension dominant; residual high-d_z architecture effect)
  4. Multi-dataset deployment validation (C12: no source-fitted eraser yields practically meaningful target-bAcc
     gain in valid cells; task-safety heterogeneous -- Lee/Cho EEGNet LEACE/RLACE drive task to chance)
  5. Track B source-only gate (benefit+safety gate rejects/abstains; 0 false accepts in sampled Lee/Cho pilot;
     naive leakage controllers false-accept harmful/useless cells)
  6. V2 source-only acceptance ceiling (EEGNet World A clean ceiling robust across n_source; B/C refusal robust
     both backbones; TSMNet World A not clean under appended-nuisance construction)
* **S6 Discussion:** measurement->control gap; refusal-first control; ceiling + source-rich sufficiency;
  target-information as the next paper.

## 4. Main-text vs appendix result allocation
**MAIN TEXT must contain:**
* one COMPACT multi-dataset C12 table (tab:bigN_compact);
* one COMPACT Track B / V2 decision-outcomes table (accept/reject/abstain + 0 false-accept + naive contrast);
* one Theory Box (P1 + P2 + bridge sentence);
* one ceiling schematic/figure: source-visible benefit -> possible accept; source-invisible target benefit ->
  abstain/reject; target information / source-rich environments can break the ceiling.

**APPENDIX:**
* full multi-dataset C12 table (tab:bigN_full);
* full concept-erasure table;
* full Track B table + naive-controller baselines;
* Phase 2 task-preserving erasure dry-run;
* V2 Stage-2 full table (World A/EEGNet by n_source; B/C by backbone x n_source) + ceiling scatter;
* TSMNet World A caveat + Stage-1B nuisance_fraction robustness probe;
* proof sketches for P1/P2/P3;
* artifact / provenance tables.

**NOT in the current paper (future work / parked):**
target-information budget experiment; active calibration; target-informed branch; Track E; World-A redesign for
high-dimensional latents; full big-N LPC / capacity Wave 2.

## 5. Limitations to rewrite (old -> new mapping)
OLD (stale PDF): "Single dataset." / "Two backbones only." / "EEGNet dim/type confound." / "No end-to-end TOS
training." / "Certified deletion mostly abstains."
NEW (verbatim intent):
* Primary mechanism figures are on 2a, but the target-deployment conclusion is validated across additional
  datasets (NOT "single dataset").
* Two backbone families remain a limitation.
* Capacity/factorial analysis partially decomposes dimension vs architecture, with a residual high-d_z
  architecture effect.
* No end-to-end selective training; current work is frozen-representation intervention and decision.
* Refusal is not a failure mode but a designed safe outcome; however, source-only acceptance power is limited.
* TSMNet V2 World A is not cleanly demonstrated under the appended-nuisance construction.
* Target-information frontier is future work, not current evidence.

## 6. Claim contract to obey (grep before/after rewrite)
Reference C15 (theory) + C14 (results). ALLOWED: "Strict source-only certification cannot license
deployment-shift benefit that is not represented in source-domain variation." / "Refusal is the safe action when
benefit is source-invisible." / "Source-rich environments or target information can break the ceiling." /
"the source-only gate has safe refusal power and exposes an acceptance ceiling." FORBIDDEN: "Source-only
acceptance is impossible in general." / "The gate has acceptance power." / "V2 proves no erasure can ever help." /
"V2 shows the gate accepts genuine target-beneficial erasure." / "World A succeeds on all backbones." / "TSMNet
World A cleanly confirms the ceiling." / "Conditional invariance never helps." / "Target-informed oracle is a
deployable method." The manuscript rewrite MUST grep these (only allowed in forbidden-lists/caveats/negations).

## 7. Page budget (TMLR)
```
Main paper target        : 15-17 pages
Theory box               : <= 0.75 page
Track B + V2 combined     : <= 1.25 pages
Multi-dataset validation : <= 0.75 page
Appendix                 : carries full tables
```
Compressions: synthetic + 2a-mechanism figures compressed; Fig.4/Fig.5 merged or moved to appendix; Table 1 ->
updated claim summary; full erasure tables to appendix (not main text).

## 8. Execution order once .tex rewrite is approved
1. Draft S3 Theory Box + the two NEW result blocks (Track B compact + V2 ceiling) against the claim contract.
2. Reorder S5 into the logical sequence (2a mechanism -> controls -> factorial -> C12 -> Track B -> V2).
3. Rewrite limitations + scope-of-validation per SS5 mapping; update title + abstract to the V3 thesis.
4. Move full tables/proofs to appendix; enforce the page budget.
5. Run check_claims.sh / anonymity / forbidden-wording gate; two-build main.tex + tmlr_main.tex = 0/0/0.
DO NOT edit tos_cmi/paper/sections/*.tex until the PM approves this plan.
