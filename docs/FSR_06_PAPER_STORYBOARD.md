# FSR_06 — Paper Storyboard

**Project FSR — Phase 3A (writing package, no new experiments).** Turns the frozen Step-1/2 results into a manuscript structure. Every claim here is bounded by `FSR_05_CLAIM_LEDGER.md`; no result is created in this document.

## Working title
> **Measurable Is Not Reliance: Functional Shortcut Auditing in EEG Representations**

Alt: *Subject Leakage Is Not Necessarily a Shortcut: Functional Reliance Audits for EEG Representations.*

## One-sentence thesis
Measurable subject/domain leakage in an EEG representation is **not** sufficient evidence of a harmful functional shortcut; a harmful shortcut requires **task-coupled, functionally-relied-upon** domain information, and current observables do not certify it — so the contribution is an **audit framework**, not a new DG method.

## Positioning (what this paper is / is not)
- **Is:** a measurement→reliance→control audit ladder for EEG representations, plus two boundary findings (*measurable ≠ relied-upon*, *erasable ≠ beneficial*) and one honestly-blocked question (branch-locality).
- **Is not:** a new domain-generalization method, a CMI regularizer, or a SOTA accuracy claim. (CMI-control is a closed premise, C8; TTA-Control is non-CMI support, C9.)

## The five-beat story
1. **Problem.** EEG representations carry strong subject/domain information; a linear/MLP probe recovers subject identity. The field often treats "leakage is decodable" as "the model relies on a harmful shortcut." That inference is unlicensed (encoded ≠ used).
2. **Audit ladder.** We define a six-level ladder — L1 detectability, L2 reducibility, L3 erasability, L4 task coupling, L5 functional reliance, L6 target consequence — and require any shortcut claim to be a *relationship between levels*, never a leap from L1 to L5/L6.
3. **Finding 1 (RQ1/RQ3 — measurable ≠ relied-upon).** In a frozen graph-CMI diagnostic (CIGL), task-head alignment is positively associated with functional reliance (align→R3 ρ=+0.34, n=126), while recomputable graph-leakage magnitude carries the *wrong* (negative) sign (graph_kl→R3 ρ=−0.42 seed0); the signed difference excludes zero. Alignment is closer to reliance than raw leakage — but not a validated estimator (dataset-heterogeneous).
4. **Finding 2 (RQ2 — erasable ≠ beneficial).** In a frozen erasure study (TOS: LEACE/INLP/RLACE/mean-scatter/random-k on TSMNet+EEGNet across 4 datasets), subject signal is erasable (LEACE linear→chance), but **no eraser certifies a proven target benefit** (0/40 cells); erasure can even harm the target (task-collapse, binary-EEGNet harm). The naïve "more removal → better target" hypothesis fails.
5. **Finding 3 (RQ4 — branch load matters, but is unmeasured).** In FBCSP-LGG, the spatial branch is load-bearing (ablation + gate); but per-branch leakage and per-branch reliance are not measured, so we **cannot** say "spatial leakage is harmful" or "graph leakage is benign." This is a blocked, not failed, question — and a scoped call for the right instrument.

## Section map
| § | Title | Content | Backing |
|---|---|---|---|
| 1 | Introduction | thesis; encoded≠used; the L1→L5/L6 fallacy | FSR_00, FSR_09 |
| 2 | The FSR audit ladder | L1–L6 definitions, direction signs, inclusion gate | FSR_02, FSR_08 |
| 3 | Setup & provenance discipline | datasets, backbones, frozen artifacts, claim-strength tiers, target-label firewall | FSR_01, FSR_03 |
| 4 | RQ1/RQ3 — measurable ≠ relied-upon | align vs leakage → reliance; provenance tiers | FSR_04 §RQ1/RQ3 |
| 5 | RQ2 — erasable ≠ beneficial | erasure strength vs target; benefit_claimable=0; sensitivity | FSR_04 §RQ2, FSR (2C) |
| 6 | RQ4 — branch-locality (blocked) | spatial load-bearing; missing per-branch metrics | FSR_04 §RQ4, preflight |
| 7 | Related work | probing limits, concept erasure, DG selection, EEG identity | FSR_09 |
| 8 | Limitations & claim hygiene | provenance limits, non-reproducible pooled leakage, single-dataset caveats | FSR_10 |
| 9 | Conclusion | the audit framework; what a shortcut claim requires | FSR_05 |

## Claim → section placement (from FSR_05)
- C1 (leakage doesn't certify reliance) → §4, headline with provenance caveat.
- C2 (alignment closer than leakage) → §4, with the "not a validated estimator" caveat.
- C3 (erasable) + C4 (no certified benefit) → §5; C4 headline is `benefit_claimable=0/40`, NOT the negative correlation (2C: not robust).
- C5 (random-k falsifies non-specific NLL) → §5, scoped to the flagged 2a-TSMNet cell.
- C6 (spatial load-bearing) + C7 (branch-local missing) → §6.
- C8 (CMI-control closed) → §3 premise; C9 (TTA non-CMI) → §3/§8 support only; C10 (audit framework) → §1/§9.

## Venue framing
A boundary/audit contribution — fits a venue that values honest negative/boundary results and methodology (measurement discipline) over leaderboard gains. The provenance-tiered reproduction and the explicit claim ledger are selling points, not weaknesses.
