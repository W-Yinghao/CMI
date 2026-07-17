# Mechanism-Contrast Consistency (MCC) training — internal execution plan (SPEC; NOT an amendment; manuscript FROZEN)

Bounded viability study authorized by PM 2026-07-17 after M1-P Result C. Branch
`agent/cmi-trace-mechanism-consistency-training`, base cd9c0025. This is NOT a new prereg amendment and does NOT
touch the manuscript. Only the project owner may explicitly stop a scientific line.

## Reframed objective (corrected from the post-hoc line)
NOT "create a deletable subspace". The intervention point moves from *delete an existing direction* to *shape how
task information is encoded*: during training, encourage different SOURCE subjects to express the same class
contrast along consistent DIRECTIONS, and test whether that makes a FUTURE subject's task mechanism closer to the
source consensus AND yields DG gain over a matched shuffled-subject control. M1-P already falsified the post-hoc
form (top-G_dis directions are well-identified but NOT future-harm-enriched vs shared-null random). MCC is a
different hypothesis, not the same failed experiment plus an optimizer. HONEST: M1-P gives NO positive evidence
that consistency training works — so this must be fast, falsifiable, no method stacking.

## The loss (direction-normalized; NOT raw Frobenius)
Raw `Σ_d ‖C_d − C̄‖²_F` is rejected — it mixes contrast direction + magnitude + latent scale and its cheapest
minimizer shrinks all contrasts (C_d→0). We care only about DIRECTION agreement. In the EEGNet 16-d bottleneck,
for each source subject d and unordered class pair (a<b):
    c_d^{a,b} = μ_{d,a} − μ_{d,b}                    # class-mean contrast (batch, per subject-class cell)
    u_d^{a,b} = c_d^{a,b} / (‖c_d^{a,b}‖_2 + ε)      # unit direction
    ū_{-d}^{a,b} = norm( Σ_{d'≠d} u_{d'}^{a,b} )     # LEAVE-ONE-SUBJECT-OUT consensus (excludes d)
    L_MCC = (1 / (m·|P|)) Σ_d Σ_{a<b} [ 1 − ⟨ u_d^{a,b}, sg(ū_{-d}^{a,b}) ⟩ ]
`sg` = stop-gradient on the consensus (each subject is pulled toward the others' consensus, not a moving self-ref).
Total: `L = L_task + λ_MCC · L_MCC`. 4-class → all 6 pairs; binary → the 1 contrast. Scale-insensitive (task CE
holds the margin). Monitor feature norm / contrast norm / effective rank as collapse guards.

## Balanced episodic sampler (subject × class × K)
Random mini-batches miss subject-class cells → unstable c_d^{a,b}. Each batch = all source subjects × all task
classes × K trials/cell. Primary K=4 → BNCI2014 8×4×4=128, BNCI2015 11×2×4=88 samples/batch. Source-side
replacement sampling ONLY if a cell is short; never skip hard subjects; NEVER target trials in training.

## Three arms (each forked from the SAME ERM warm-up checkpoint)
- **A — ERM continuation**: `L = L_task`. Controls extra epochs / LR continuation / checkpoint selection.
- **B — MCC-true**: L_MCC with the REAL source-subject grouping.
- **C — MCC-shuffle**: within each class, independently permute the subject assignment per batch, then the SAME
  L_MCC formula (same batch, class counts, gradient budget) — breaks real within-subject contrasts. Isolates
  whether MCC's benefit is real cross-subject mechanism alignment vs a generic smoothing regularizer.
No C-CORAL / IRM / PCGrad this round. C-CORAL enters the SAME protocol only AFTER MCC shows a utility signal.

## Fixed settings (NO hyperparameter engineering; ONE λ)
backbone EEGNet; latent = current 16-d bottleneck; warm-up = the existing ERM exactly; continuation 20 epochs;
λ_MCC 0.25; λ ramp linear 0→0.25 over the first 5 continuation epochs; optimizer = the ERM optimizer;
continuation LR = 0.1 × warm-up initial LR; checkpoint SELECTION = source-only validation; target X/Y = evaluation
ONLY. If no ERM warm-up checkpoint exists in the repo, retrain ONE warm-up per (dataset, subject, seed) and cache
it, then fork all three arms from it (NEVER a per-arm warm-up).

## Real-EEG GPU matrix (encoder retraining → GPU)
Primary tranche: 2 datasets × 21 LOSO subjects × 3 seeds × 3 arms = **189 continuation cells**, submitted as 63
paired bundles (dataset × target subject × seed → one warm-up → 3 arms). Result asset must show 189/189 arms
complete. Do NOT substitute the existing ERM target score for Arm A (continuation must be controlled).

## Metrics (all logged per arm)
1. Objective movement (source-whitened frozen features): ΔWSCI_source; Δ tr(G_dis)/tr(G_shared);
   Δ target-to-source contrast alignment `A_t = (1/|P|) Σ_{a<b} cos(c_t^{a,b}, c̄_s^{a,b})` (target labels
   EVAL-only).
2. Collapse/damage: source task bAcc; source-heldout bAcc; feature norm; effective rank; class-contrast norm; task
   margin; selected checkpoint epoch.
3. DG utility (inference unit = outer target subject; 3 seeds bootstrapped with the subject cluster):
   ΔU_MCC−ERM = bAcc_MCC − bAcc_ERM_continue; mechanism-specific control ΔU_MCC−shuffle = bAcc_MCC − bAcc_MCC_shuffle.
4. M1-P RE-AUDIT (diagnostic, NOT a success condition): re-run the SAME frozen oracle on ERM-continue / MCC-true /
   MCC-shuffle features — did G_dis rank drop? did the informed/random capture gap change? is residual disagreement
   still not future-harm-enriched? did training merely rotate the representation?
5. CMI: NOT run this round (posterior ruler only; used later only if MCC shows DG utility).

## Result routing (PM A–E)
- **A** geometry↑ + DG↑ + beats shuffle (source dir-consistency↑, A_t↑, source drop ≤0.02, ≥1 dataset
  LCB95(ΔU_MCC−ERM)>0, other not harmful, MCC>shuffle) → seeds 3,4 + C-CORAL same-protocol + DGCNN replication →
  only then learned subspace / TTE.
- **B** geometry↑ but DG flat → analyze target shared class signal / covariance / whether A_t followed source WSCI;
  do NOT just raise λ.
- **C** geometry doesn't move → update/loss too weak; ONE bounded fix: λ=1.0 OR unfreeze full-encoder-vs-top-block
  (one, not both).
- **D** geometry↑ but source/target damaged → hard consensus too rigid; go to `C_d = C_shared + R_d`, penalize only
  residual cross-subject variance, or a capacity-limited shared subspace.
- **E** MCC ≡ MCC-shuffle → generic regularization not real mechanism consistency; analyze true/shuffle gradient
  cosine, margin, effective rank, feature-smoothing equivalence.

## Deliverables (this branch)
`tos_cmi/train/mechanism_consistency.py` (MCC loss + balanced subject-class sampler + true/shuffle grouping) +
EEGNet trainer integration (3 arms from one warm-up) + one YAML config + this plan + tests + real-EEG engineering
smoke + the 189-arm GPU job matrix. Engineering smoke checks only: MCC grad nonzero; task grad dominant; contrast
norm not →0; feature effective rank not collapsed; true≠shuffle loss. If those pass, run full EEG WITHOUT tuning λ
from smoke target scores.

## Tests (10, pinned)
1 identical contrasts → L_MCC≈0; 2 contrast scaling doesn't change the direction loss; 3 LOSO consensus excludes
self; 4 true vs shuffled grouping → different loss; 5 gradient flows into the encoder; 6 target arrays never enter
training; 7 missing subject-class cell FAILS LOUD; 8 three arms fork from the SAME warm-up hash; 9 feature/contrast
collapse guard fires; 10 artifacts fully recoverable.

## HELD / forbidden this round
M2 source selector, learned oblique deletion projector, TTE, CMI loss, PCGrad, dynamic projector, new backbone, new
prereg amendment, any manuscript edit. Post-hoc secondary families (DGCNN contrast, EEGNet rule/grad) deferred.
