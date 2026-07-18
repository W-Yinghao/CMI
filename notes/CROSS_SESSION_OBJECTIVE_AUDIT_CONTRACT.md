# Cross-Session Objective Audit — exact-gradient discriminator (SPEC; NOT an amendment; manuscript FROZEN)

PM 2026-07-18: after RW-MCC DG-null (source cross-subject LOSO risk trainable but DG-inert), do NOT jump to a full
cross-session RW-MCC 189-arm fleet. First a NO-TRAINING, 63-cell real-EEG audit that separates: (1) does source
CROSS-SESSION instability carry a signal different from LOSO risk? (2) does its training gradient point closer to
the held-out target's descent direction than LOSO-RW-MCC? (3) is the bottleneck the MCC geometry MEDIATION, or does
cross-session shift itself fail to predict unseen-subject failure? Branch
`agent/cmi-trace-cross-session-objective-audit`, base f13a9c1e (RW head incl. source-risk reaggregation fix). Only
the project owner may stop a scientific line.

## Cross-session risk weight (source-only)
Natural session ordering: BNCI2014 `0train→1test`; BNCI2015 `0A→1B/2C`. For source subject d, class pair p=(a,b):
fit `h^early_{-d}` on the EARLIEST session of the OTHER source subjects only. On subject d compute pairwise-balanced
log-losses `l^early_{d,p}` (its early session) and `l^late_{d,p}` (its later session). Then
`r^sess_{d,p} = [ l^late_{d,p} − l^early_{d,p} ]_+` — same classifier for early/late (early controls the subject's
own cross-subject difficulty; the difference isolates the added session drift; all info from outer-source subjects).
Weights use the IDENTICAL winsor-p90 / mean-normalize / clip-4 / no-remean as RW-MCC (fair comparison).

## Two candidate objectives (one proxy, TWO intervention points)
1. **CS-RW-MCC** — geometry mediation: `L = Σ_{d,p} w^sess_{d,p} [1 − cos(u_{d,p}, ū_{-d,p})]`. Does a
   deployment-closer weight finally make MCC point at target utility?
2. **CS-Risk** — direct: `L = Σ_{d,p} w^sess_{d,p} l^late_{d,p}(h_ω∘f_θ)` (weighted later-session task loss, no
   cosine geometry). Is the cross-session signal valuable but the MCC mediator wrong?

## Matched controls
per-pair subject-weight permutation; source session-order permutation; the current LOSO-RW-MCC gradient; the
ordinary source task gradient.

## Exact-gradient audit (NOT training) — 63 real-EEG cells at the existing warm-up checkpoints
Per cell compute the encoder-param gradients `g_CS-RW`, `g_CS-Risk`, `g_LOSO-RW`, `g_perm` (and the control
gradients), plus a NON-DEPLOYABLE held-out FUTURE-SESSION task gradient `g_target`. TARGET LABELS NEVER enter any
source objective or weight — used ONLY to evaluate whether a source objective points the right way.

### Primary endpoint — target-gradient alignment
`A_o = cos(g_o, g_target)`. Compare `A_CS-RW − A_perm`, `A_CS-Risk − A_perm`, `A_CS − A_LOSO`. Gradient descent
updates by −g_o, so a positive `g_oᵀ g_target` means the source objective reduces target loss to first order.

### Secondary endpoint — normalized one-step target loss
All objectives take the SAME-norm parameter step `θ'_o = θ − α g_o/‖g_o‖`, then evaluate ΔL_target on the held-out
future session ONLY; also check the source later-session loss moves as intended.

## Gate to a full 189-arm training round (a candidate needs ALL)
1. vs its matched permuted control: `LCB95(ΔA) > 0` on ≥1 dataset;
2. other dataset not clearly reversed: `UCB95(ΔA) > −0.05`;
3. normalized one-step target loss better than the control;
4. ≥60% of target subjects directionally consistent;
5. source later-session risk drops as expected;
6. not driven by a single seed or subject.
Inference unit = target subject (3 seeds aggregated per subject first), subject-cluster bootstrap + exact sign-flip.

## Routing
| audit result | next |
|---|---|
| CS-RW-MCC passes, CS-Risk fails | one cross-session RW-MCC training round |
| CS-Risk passes, CS-RW-MCC fails | train cross-session risk directly (drop MCC) |
| both pass | prefer the more direct CS-Risk |
| neither passes | NO cross-session GPU fleet → pivot to the target-conditioned information regime |
Any positive is EXPLORATORY method selection; a method claim requires confirmation on a THIRD EEG dataset.

## If cross-session also fails — the next scientific question (not a stop)
Shift from "how to predict the unseen subject FROM SOURCE geometry" to "does improvement REQUIRE target-specific
information, and what is the MINIMAL kind?" → a strict information-regime ladder
`source-only → target-X → few-shot target labels` comparing zero-shot DG / unlabeled target adaptation / 1·2·4
labeled-trials-per-class calibration — turning the earlier `TARGET_HINDSIGHT_ONLY` result into a testable
sample-complexity question instead of inventing another source proxy.

## Deliverables (this branch; NOT a prereg amendment)
`tos_cmi/train/cross_session_objective.py` (cross-session weight builder + CS-RW-MCC / CS-Risk gradients + controls),
`scripts/run_cross_session_audit.py`, `scripts/aggregate_cross_session_audit.py`, `scripts/sbatch_cross_session_audit.sh`,
config, this contract, tests.

## Tests (pinned)
cross-session weight uses only source EARLY/LATE (no target); early-loss controls baseline (subtraction); CS-Risk
gradient != CS-RW-MCC gradient (different intervention point); target labels never in any source objective/weight
(signature + run); two-pass exact gradient == full-batch for the MCC path; g_target uses future-session only;
alignment/one-step finite; 63-cell completeness fail-closed; warm-up hash consistent; not-driven-by-1-subject
surfaced.

## Allowed / forbidden (PM)
Allowed: source-risk reaggregation (done), cross-session weight construction, CS-RW-MCC/CS-Risk exact-gradient
audit, 63/63 cells, matched controls, target-labels audit-only firewall, gate verdict. FORBIDDEN: full cross-session
training, new λ sweep, EMA, M2, projector, TTE, CMI loss, manuscript edit.
