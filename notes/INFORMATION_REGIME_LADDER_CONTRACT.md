# Information-Regime Ladder — pre-registration (Track B; PM-directed new PRIMARY line; manuscript FROZEN)

**Branch** `agent/cmi-trace-information-regime-ladder` (worktree `/home/infres/yinwang/CMI_AAAI_infoladder`, base
`de170ede`). Only the project owner stops/redirects a scientific line. This turns the earlier `TARGET_HINDSIGHT_ONLY`
result into a strict **sample-complexity** question: *from zero target information → unlabeled target X → 1/2/4
labeled trials per class → all calibration labels, at what information threshold does a useful subspace-deletion
action first become IDENTIFIABLE (selectable), and is that threshold subspace-SPECIFIC?*

## Encoder & features (frozen; NO re-inference)
Frozen ERM EEGNet dumps `tos_cmi/results/tos_cmi_eeg_frozen/{DS}_EEGNet_LOSO/sub{S}_erm_lam0_seed{K}.npz` (the SAME
features the M1-P mechanism-subspace oracle used → this line directly extends that result). 63 cells = BNCI2014_001
(9 subj × 3 seed) + BNCI2015_001 (12 subj × 3 seed). Loader `feat_from_tos_dump` → `Z_source, y_source, subj_source,
Z_target, y_target, session_target`. Read the dumps by ABSOLUTE path from the main worktree (untracked frozen data,
stable across branch switches).

## Fixed target calibration / query protocol (session-based, deterministic)
`session_split(session_target, y_target)` (targetx_observability): cal = earliest session, query = later session(s).
- BNCI2014_001: cal = `0train`, query = `1test`.
- BNCI2015_001: cal = `0A`, query = `1B`/`2C`; PRIMARY outcome = **session-macro** bAcc (mean over query sessions).
Calibration (X and Y) never enters the query outcome; query (X,Y) is used ONLY for the final outcome (and, audit-only,
for the hindsight ceiling). Whitening `source_whitener(Z_source)` (Ledoit-Wolf A_s); all deletions in whitened metric,
affine-mapped back to raw (`whitened_delete_fn`).

## PRIMARY = SELECTION-ONLY ladder (target labels SELECT an action; they NEVER retrain the encoder or the head)
This isolates "does target information help IDENTIFY the right subspace?" from "does ordinary supervised calibration
raise accuracy?". The readout for an action S is a fresh source-fitted `LogisticRegression` on `delete_S(Z_source)`
(the deployable-DG readout used line-wide, `dg_identifiability._bacc`); it is refit per action but NEVER on target
labels.

### Action families (whitened rows; exhaustive subsets rank ≤ 3, identity included)
- **PRIMARY informed dictionary** = `B_cond` (label-conditional subject subspace, `whitened_cond_basis`), rank r
  capped so `build_exhaustive_action_family(r, max_subset_rank=3)` is tractable; identity = empty subset.
- **Matched specificity control** = `n_random` random dictionaries of EQUAL rank r and the SAME exhaustive action
  budget (`build_ambient_random_dictionaries(D, r, n_random, seed)` → one action family each). A larger dictionary must
  NOT get a larger selection advantage — rank and action count are matched.
- **SECONDARY informed dictionary** = mechanism-disagreement subspace (`build_shared_null_contrast_basis` /
  class-conditional gradient disagreement), reported separately.

### Regimes and their SELECTORS (all pick one action from the SAME family; no gates beyond source-safety where noted)
| regime | target information | selector |
|---|---|---|
| R0 | none | source-meta: argmax action by source-LOSO held-out bAcc gain (frozen) |
| RX | all calibration X, NO labels | frozen target-X selector: argmax `observable_G1` = ‖U·A_s(µ_s−µ_tcal)‖² |
| R1 | 1 labeled trial / class | argmin_S CE on the k cal trials |
| R2 | 2 labeled trials / class | argmin_S CE on the k cal trials |
| R4 | 4 labeled trials / class | argmin_S CE on the k cal trials |
| RF | all calibration labels | argmin_S CE on ALL cal trials |

- CE (not bAcc) for label-based selection: defined at k=1, uses margin, no per-draw head refit (head is source-fit;
  CE = cross-entropy of the source-fitted `predict_proba` on `delete_S(X_cal,k)` against the k cal labels).
- R1/R2/R4: ≥ **20 deterministic, class-balanced** label draws per (subject, seed); draws never chosen by query
  outcome; the SAME draw indices are used for the informed dictionary AND every matched-random control.

### PRIMARY endpoints (inference unit = target subject; 3 seeds → subject first; subject-cluster bootstrap + exact sign-flip)
- **Query gain** `ΔU_k = U_query(S*_k) − U_query(identity)` = `score_on_target_query(Zs_w, ys, U_{S*_k}, Xq_w, yq, sq)`
  (session-macro; identity → 0). For R1/R2/R4, averaged over the ≥20 draws per cell first.
- **Dictionary specificity** `ΔU_k^specific = ΔU_k^{B_cond} − E_R[ΔU_k^{random}]`.
- **Hindsight recovery** `Recovery_k = ΔU_k / ΔU_RF`, reported ONLY if `LCB95(ΔU_RF) > 0`. Also report the absolute
  `crossfit_target_oracle` ceiling (select on disjoint query labels) as context.
- **Minimal information level** `k* = min{ k : LCB95(ΔU_k) > 0 }`; **subspace-specific threshold** additionally
  requires `LCB95(ΔU_k^specific) > 0`.

## SECONDARY = practical calibration ladder (independent; head-only few-shot)
For each R1/R2/R4/RF: start from the source head, update ONLY a linear head on the k cal labels (encoder frozen),
compare query bAcc of {native representation, selected-subspace representation, matched-random-subspace}. Distinguishes:
- **A** selection-only already positive, head-calibration adds nothing → target labels mainly IDENTIFY the subspace.
- **B** selection-only zero but head-calibration positive → need a target-specific decision boundary, not a deletion.
- **C** both positive, combination best → labels solve geometry selection AND readout calibration.
- **D** even RF gives no gain → the action family is wrong; earlier target-hindsight was selection optimism / generic
  low-rank effect.

## Result routing (owner decides next; nothing built unilaterally)
- **IL-A** few labels make the informed dictionary beat random (query gain + specificity gate at R1/R2/R4):
  `SUBSPACE EXISTS / SOURCE+TARGET-X OBSERVABILITY FAILS / FEW-SHOT LABELS RESOLVE IDENTIFIABILITY` → THEN a few-shot
  subspace selector or TTE is worth building.
- **IL-B** few labels raise utility but informed == random → generic low-rank / action-search / calibration effect,
  not subject-subspace → ordinary few-shot adaptation, not more basis surgery.
- **IL-C** only head-calibration works → the DG bottleneck is the target-specific `P(Y|Z)` readout, not a deletable
  subspace.
- **IL-D** RX works, R0 does not → unlabeled target geometry suffices; source-only does not → transductive adaptation.
- **IL-E** only RF works → task is strongly subject-specific; large calibration needed; not near-calibration-free DG.

## Discipline / HOLD
Pre-reg BEFORE running; adversarial-verify AFTER. All compute via SLURM (CPU array — frozen features + sklearn, no
neural forward/backward). No new source-only proxy, no new MCC weighting, EMA, M2, learned projector, TTE, CMI loss,
or manuscript edit until the ladder identifies a threshold and the OWNER chooses the follow-up. No CLOSED without a
third-EEG-dataset confirm; every candidate positive must survive BOTH datasets (they have historically reversed sign).

## Deliverables
`tos_cmi/eval/information_ladder.py` (regime selectors + per-cell ladder), `scripts/run_information_ladder.py`,
`scripts/aggregate_information_ladder.py`, `scripts/sbatch_information_ladder.sh` + driver, this contract, tests.
