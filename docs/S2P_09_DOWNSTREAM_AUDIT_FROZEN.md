# S2P_09 — P1 Downstream Frontier Audit (FROZEN pre-registration)

**Status: FROZEN before any downstream compute. Do not edit after the pre-reg commit; new analyses go in a dated `*_APPENDUM.md` frozen BEFORE viewing the aggregate.**

Frozen: 2026-07-08. Branch `project/s2p-subject-scaling`. Predecessor: P1 pretraining (S2P_08, 15/15 clean,
checkpoints `results/s2p_p1_cbramod/N{N}_s{S}/best.pth`). Reuses the FSR-hardened L1/L4/L5/L6 probe
(`CMI_AAAI_rq4/scripts/cb_cbm_8b_{audit,f1_audit,feature_dump}.py`, FSR_46), self-contained copy in the S2P runner.

---

## 0. Question (frozen; not to be re-scoped)

> Along the fixed 200 h pretraining allocation frontier (T = 200 h, N ∈ {128,256,512,1024,2048}), do the CBraMod
> representation's **SHU-MI transfer**, **subject separability (L1)**, **task coupling (L4)**, and
> **subject-subspace reliance (L5/L6)** systematically change as we move from fewer-deeper to more-shallower
> subjects?

**Estimand = the BUNDLED net fixed-budget allocation trend** (coverage + depth + population + redundancy vary
together along N; NOT population-adjusted at any N). Inherits the S2P_06 v4 framing verbatim: **descriptive +
identifiability lesson, NOT a coverage-vs-depth causal effect, NOT a pure-diversity effect.** The T = N·e triangle
makes pure subject-diversity unidentifiable (established across P1 red-team rounds); this audit does not attempt it.

---

## 1. Frozen design decisions (PM-approved 2026-07-08)

### 1.1 Channel mapping — SHU 32ch → COMMON19 canonical (pretraining) order
The P1 checkpoints were pretrained on the 19-common 10–20 montage `19ch-703de0`. Downstream SHU is fed the **same
19 channels in the same canonical order** the encoder was trained on (`tueg_subject_loader.COMMON19`, "-LE" stripped):

```
FP1, FP2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, FZ, CZ, PZ
```

SHU-MI 32ch (`FP1,FP2,FZ,F3,F4,F7,F8,FC1,FC2,FC5,FC6,CZ,C3,C4,T3,T4,A1,A2,CP1,CP2,CP5,CP6,PZ,P3,P4,T5,T6,PO3,PO4,OZ,O1,O2`)
**contains all 19** (uppercase match; SHU uses legacy `T3/T4/T5/T6` naming → **no alias needed**). The mapping table
(19 SHU-index → canonical-position pairs) is **hashed into the manifest**; ad-hoc mapping forbidden.
**STOP** if any of the 19 is missing/ambiguous (verified present: 19/19).

### 1.2 Windowing — native SHU 4 s trials (4 patches @ 200 Hz), NO padding
SHU trial = 1000 samples @ 250 Hz = 4 s → resample to 800 @ 200 Hz → **4 patches × 200**. Fed natively; **not**
padded to 30 patches. Architecture check PASSED (random-init CBraMod `bb((2,19,4,200))` → `(2,19,4,200)`, finite,
F1 dim 3800). **STOP** if native 4-patch forward fails; padding is a *later sensitivity only*, never primary.

### 1.3 Normalization — per-PATCH per-channel z-score (MATCH pretraining)
Pretraining loader z-scores each 1 s patch over its 200 samples, per channel (`tueg_subject_loader.py:203`,
`(x - x.mean(-1)) / (x.std(-1)+1e-6)`, no `/100`, HBN normalizer neutralized). Downstream uses the **identical**
per-patch z-score. **This corrects the FSR-8B feature dump**, which z-scored per whole trial; per-patch is required
so the encoder sees in-distribution inputs. Same normalization for **every** cell (stop-rule 5).

### 1.4 Feature — F1 spatial primary, F0 for the beats-F0 gate
F1 = per-channel encoder feature (mean over 4 patches, keep 19 channels → **19·200 = 3800**); preserves MI C3/C4
lateralization. F0 = pooled (mean over channels+patches → 200), used only for the F0-target-bAcc reference in the
task gate.

### 1.5 Probe protocol — source-only, fixed subject-disjoint split
SHU has **25 subjects × 5 sessions × 100 trials** (2-class MI, labels {1,2}). Fixed split:

```
source train : subjects 1–15   (fit PCA, head, subject subspace, rank)
source val    : subjects 16–20  (head selection, L5 reliance eval)
target test   : subjects 21–25  (final scoring ONLY)
```

- PCA (95 % var, cap 128) **fit on source-train only**.
- LDA task head **trained on source-train**, **selected on source-val only**.
- Subject subspace + removed-variance rank **fit on source-train only**.
- **Target task labels used for final bAcc / macro-F1 / NLL / Δ scoring ONLY** — never in PCA / head / subspace /
  rank / checkpoint selection.

### 1.6 Random-init control
One random-init CBraMod (init_seed 0, no pretraining load), same forward/probe. It is a **representation-quality
floor** ("did P1 learn anything more useful than random init?"), **NOT** a competing pretraining cell and **NOT**
used for checkpoint selection. Checkpoint selection = pretrain-val loss only (already done: the 15 `best.pth`).

---

## 2. Units

16 forward-cells: **15 P1 checkpoints** (`best.pth`, N∈{128,256,512,1024,2048} × seed∈{0,1,2}) + **1 random-init**.
Each cell: one frozen forward over all 25 SHU subjects → F1/F0 dump → the fixed-split probe.

---

## 3. Endpoints (frozen)

**Per cell (transfer + reliance):**
- **Transfer**: target bAcc (PRIMARY), macro-F1, NLL — fixed head, source-val-selected.
- **Task gate**: `source_val_bAcc ≥ 0.60 AND target_bAcc ≥ 0.58`, OR `target_bAcc − F0_target_bAcc ≥ 0.04`.
  Fail → `WEAK_TASK_NOT_INTERPRETED` (L4/L5/L6 not promoted; L1 + transfer still reported).
- **L1 PRIMARY**: source-side subject decodability WITHIN source-train {1–15}, session-held-out LDA on PCA(F1),
  marginal + class-conditional, 1000-perm null → bAcc, null-mean, p, effect.
- **L1 SECONDARY (diagnostic only, not for any selection)**: subject separability among source-val {16–20} and
  among target {21–25}, same probe.
- **L4**: task-head ↔ source-fitted label-conditional subject subspace alignment (k-curve {1,2,4,8}; **k=2 primary**).
- **L5**: erase source-fitted subject subspace vs **variance-matched null** vs oracle-task, on source-val; report
  drop_subject, drop_variance, var_removed. **PRIMARY L5 = subject-drop CI-lower > variance-null mean.**
- **L6**: target bAcc / macro-F1 / NLL delta after erasing the source-fitted subject subspace (final scoring),
  with the variance-null delta alongside. Conservative-null caveat retained (source-estimated subspace on novel
  target → null L6 is NOT proof subject info is task-irrelevant).

**Frontier (across the 15 P1 cells; random-init excluded from the slope):**
- **Slope** of each of {target bAcc, L1 effect, L4 k2, L5 drop_subject, L6 delta_subject} vs **log₂N**;
  cluster-bootstrap CI over seeds. **Straddling CI ⇒ null** (MJ-15).
- **Curvature**: quadratic term + peak-N (MJ-12 pre-registered; endpoint may be monotone OR peaked).
- **Leave-one-N-out sign stability** of each slope (descriptive robustness, MJ-6 — slopes are DESCRIPTIVE not
  inferential given VIF/power limits).
- **Positive-control floor**: each pretrained cell's transfer & metrics vs random-init.

---

## 4. Aggregation rule

- Endpoint definitions are FROZEN by this doc. **D0 (one-checkpoint probe gate) sees only QC**, never the
  15-cell aggregate. If D0 surfaces a needed new analysis, freeze it in `S2P_09_APPENDUM.md` BEFORE viewing the
  aggregate, and disclose that the probe was seen.
- Per-cell CIs: L1 by session-fold; L5 by source-val subject; L6 by target subject (cluster bootstrap, 2000).
- Frontier inference over 15 points = 5 N-levels × 3 seeds; x = log₂N; seed = cluster. Random-init is NOT a
  frontier point (it is the floor).
- Freeze-before-aggregate: the neutral `S2P_10_..._RESULTS.md` is written, THEN framing is proposed and the PM
  steers before any interpretation doc.

---

## 5. Go / No-Go

### 5.1 D0 probe gate (ONE checkpoint = N512_s0 + random-init) — all 8 must PASS
1. channel mapping exact (19/19, table hash matches manifest).
2. native 4-patch forward works (no shape error, finite F1/F0).
3. embeddings deterministic (F1 & F0 repeat-max < 1e-5; batch-group-max < 1e-5).
4. PCA / head / subspace source-only (fit populations ⊆ source-train; asserted, logged).
5. L1 / L4 / L5 / L6 all compute (non-None).
6. variance-matched null computable & consistent (var_removed matched to subject-k; drop_variance finite).
7. target-label firewall clean (`p1_target_label_firewall.json`: target labels only in final scoring).
8. output schemas correct (all required files below present, columns as specified).

**Any fail → STOP and report. Never launch the D1 fleet without a passing D0.**

### 5.2 D1 → analysis
- Coverage: 16/16 cells dumped, verified by **proper unique count** of `(N,seed)`+random-init, not a string sort.
- Determinism QC pass per cell.
- Task gate reported per cell. If ALL 15 P1 cells are `WEAK_TASK` → interpret only L1 + transfer, not L4/L5/L6.
- **If SHU is near-chance for ALL cells → STOP, return for review; do NOT auto-add PhysioNetMI/BNCI** (stop-rule 8).

---

## 6. Pre-committed interpretation grid (written before seeing results)

| Observed along the frontier | Frozen reading |
|---|---|
| transfer ↑ with logN, **L1 flat**, **L5 null** | **role-change-not-erasure** (strongest S2P thesis): improvement does not require removing subject-identifiable structure; measured subject subspace stays non-reliant under the head. |
| transfer ↑, **L1 ↓** | subject separability drops along the frontier — bundled coverage/diversity co-varies (NOT pure-diversity). |
| transfer ↑, **L5 reliant** (subject beats variance-null, growing) | subject-subspace reliance changes along the frontier. |
| transfer **flat** (slope CI straddles 0) | "this 200 h CBraMod frontier does not produce a detectable downstream allocation effect." |
| random-init ≈ or > most pretrained cells | training-quality floor NOT cleared → stop-rule 6, return for review. |
| curvature significant, monotone rejected | report peak-N; frontier is non-monotone (allocation optimum interior). |

Report honest nuance when data lands between cells; do not force a clause the data does not support.

---

## 7. Claims checklist

**REQUIRED / allowed wording (only the matching one):**
- positive: *"Along a fixed 200 h pretraining allocation frontier, transfer changes as we move from fewer deeper
  subjects to more shallower subjects."*
- L1 flat + transfer up: *"Improvement does not require removing subject-identifiable structure."*
- L5 null: *"The measured subject subspace remains non-reliant under the task head."*
- flat: *"This 200 h CBraMod frontier does not produce a detectable downstream allocation effect."*

**FORBIDDEN (permanent + this phase):** "subject diversity causes improvement", "pure diversity effect",
"population-adjusted effect", "diversity removes leakage", "foundation subject-invariant", "TUEG solves
cross-subject", growing-hours-read-as-diversity, SOTA / full-FT, "X converges to Y" without the supporting metric,
any deployable-selector / oracle-as-feature claim.

---

## 8. Stop rules (verbatim, PM)

1. any 19-common channel missing or ambiguously mapped.
2. native 4 s / 4-patch forward fails.
3. padding becomes necessary.
4. target labels enter PCA / head / subspace / rank / checkpoint selection.
5. feature extraction differs across N cells except checkpoint weights.
6. random-init control outperforms most pretrained cells (training-failure signal).
7. variance-matched null cannot be computed consistently.
8. SHU task near chance for all cells → return for review (do NOT auto-add another downstream dataset).

---

## 9. Provenance

Per cell records: checkpoint path + **sha256**, git sha, channel-map-table hash, split hash, `per_patch_zscore=true`,
`target_labels_used=false`, `pretrain_val_loss` (from `run_summary.json`). Deterministic forward (eval + no_grad +
masking off, `CUBLAS_WORKSPACE_CONFIG=:4096:8`). Two-commit discipline: **(a) this pre-reg**, **(b) runner + D0
gate**, **(c) D1 results-only**, **(d) interpretation after PM steer**. Push only on explicit PM instruction.

---

## 10. Required outputs — `results/s2p_p1_downstream/`

```
p1_downstream_run_manifest.csv        p1_task_performance.csv           p1_l6_target_consequence.csv
p1_channel_mapping_manifest.csv       p1_pairwise_subject_separability.csv   p1_random_init_control.csv
p1_windowing_manifest.csv             p1_l4_task_alignment.csv          p1_frontier_summary.json
p1_feature_dump_manifest.csv          p1_l5_subject_subspace_replay.csv p1_target_label_firewall.json
```

`p1_frontier_summary.json` carries at least: primary_estimand=`bundled_fixed_budget_allocation_frontier`,
downstream_dataset=`SHU-MI`, channel_mapping=`19-common`, windowing=`native_4s_trials`,
target_labels_used_for_selection=`false`, checkpoint_selection=`pretrain_val_loss_only`, primary_metric=`target_bAcc`,
frontier_slope, curvature_or_peak, leave_one_N_stability, l1_frontier_trend, l5_frontier_trend, p2_recommended.

---

## 11. Execution order

**D0 one-checkpoint probe gate** (N512_s0 + random-init, 8-item QC) → report gate-checklist → **WAIT for explicit
PM go** → **D1 15-checkpoint + random-init fleet** (QOS-aware babysitter, real-id addressing, per-cell append +
skip-if-done) → neutral `S2P_10_..._RESULTS.md` → propose framing → WAIT. **No CodeBrain, no specialist baselines,
no new downstream dataset.**
