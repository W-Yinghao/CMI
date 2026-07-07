# FSR_46 — CodeBrain + CBraMod 8B Encoder-Audit Protocol (Phase 8B; pre-registration)

**Project FSR — Phase 8B.** Pre-registration of the **frozen-encoder FSR audit** of two EEG foundation encoders.
PM-approved; **CBraMod included from the start as a clean single-stage encoder CONTROL** (not just backup) so we
can tell whether a finding is CodeBrain-specific. Frozen encoders, **no fine-tuning**, target labels for **final
scoring only**. **Design-red-teamed (agent abc90a55) — 2 BLOCKERs + 8 MAJORs fixed below**; adversarially verified
after the run.

## Scientific question (Phase 8)
> When cross-subject generalization improves with more source subjects or foundation pretraining, does subject
> leakage **decrease**, or does its **functional role** change (harmful shortcut → task-useful structure)?

8B = **encoder-audit sanity** (not scaling, not a leaderboard): confirm the FSR L1/L4/L5/L6 ladder is measurable on
frozen embeddings of **both** encoders, establish the baseline subject-leakage / task-coupling picture, and test
whether it is architecture-general. 8C (scaling on PhysioNetMI) runs **only if 8B passes the gate**.

## Models (frozen; `model_load_manifest.csv`)
- **CodeBrain (PRIMARY)** — EEGSSM encoder `CodeBrain.pth` sha256 `d9714b87…`; TFDual tokenizer
  `CodeBrain_Tokenizer.pth` sha256 `e9560b67…`. GPU (SSSM hardcodes `.cuda()`).
- **CBraMod (CONTROL)** — `Cbramod_pretrained_weights.pth` sha256 `0792cb80…`. CPU-runnable; single-stage, **no
  tokenizer / no VQ-collapse failure mode**. Isolates whether findings are CodeBrain-tokenizer/SSM-specific.
- Audit feature is the **encoder pooled embedding** for both (CodeBrain downstream uses the encoder, not codes).

## Datasets (8B-0; per-dataset only — see MAJ-7)
- **SHU-MI (PRIMARY, CodeBrain-native MI):** 25 subj × 5 sessions, 32 ch → canonical-19, 250→200 Hz, 4 s = 4 patch,
  2-class. **Primary for any subject-subspace claim** (enough subjects).
- **BNCI2014_001 (ALIGNMENT SANITY):** 9 subj × 2 sessions, 22 ch → canonical-19, 250→200 Hz, 4 s, 4-class.
  **Sanity only** — 8 source subjects ⇒ subject-subspace rank ≤ K·(n_src−1); no subspace headline off it.
- BNCI2015_001 optional (12 subj, 512→200 Hz); must not block 8B; if its resample path differs, disclose, do not
  compare across it.

## Channel montage (BLK-2 fix — load-bearing; native-per-dataset)
BLK-2 forbids arbitrary **first-N** channel selection. A *shared* canonical-19 10-20 set is **rejected** because
BNCI2014_001 (2a) is an FC/C/CP-centric **motor** montage that overlaps the 10-20 canonical in only ~5 electrodes —
forcing canonical-19 would drop 2a's most informative motor channels. Since we report **per-dataset only** (MAJ-7,
no cross-dataset magnitude comparison) and the encoders are **conv-over-channel** (CBraMod's own `model_for_bciciv2a`
uses all 22 of 2a; CodeBrain SSSM PatchEmbedding is a channel conv), the pinned choice is: **use ALL native
channels of each dataset, in the dataset's documented native order** (SHU-MI 32, 2a 22), resampled to 200 Hz —
**no subsetting, no first-N**. The exact ordered channel-name list per dataset is recorded in
`feature_dump_manifest.csv` (from SHU-MI BIDS `channels.tsv` / MOABB montage) and pinned; `input_chans=range(C+1)`
for the CodeBrain tokenizer indexes embedding slots `0..C` over that fixed order. No channel is silently
reindexed or dropped. (Cross-dataset comparisons remain forbidden, so identical electrodes across datasets is not
required — each dataset is audited against its own chance lines.)

## Determinism / QC (MAJ-6, STOP-1)
Frozen inference must be **deterministic**: `eval()` + `torch.no_grad()` + **masking disabled** (`mask_ratio=0` /
deterministic full mask), `torch.use_deterministic_algorithms(True)`, `cudnn.benchmark=False`, fixed seed.
`encoder_embedding_qc.csv` verifies **F0 and token ids are identical** (within 1e-5) across (i) repeated identical
passes and (ii) **different batch groupings** of the same trials. Token determinism, not just F0, must pass or STOP-1.

## Splits + firewall
- **Task metrics (L4/L5/L6):** subject-disjoint **LOSO** — source subjects train head/subspace, target subject
  scores only. Source-val (nested, source-only) for any selection.
- **L1 subject probe (BLK-1 fix):** the subject-ID probe needs every source subject in train AND test → it uses a
  **within-subject, SESSION-held-out split** (train subject's session A, test session B; both datasets have ≥2
  sessions), **not** LOSO. Chance = **1/n_source_subjects**, reported as **balanced accuracy**. Session-held-out
  attributes decodability to the encoder, not same-session drift.
- **Firewall:** target labels never touch encoder extraction, subspace, head, probe, or (8C) source-subset
  selection. z-score is **per-trial within-window** (MIN-2) — never statistics pooled over target trials.
  `target_label_firewall.json` logs every target-label read (final scoring only).

## Metrics (red-team-hardened; per-dataset; balanced accuracy + printed chance line)
- **L1 — subject leakage (Q8B-1):** within-subject session-held-out subject probe on `F0` (and F1 channel-pooled,
  F2 patch-pooled, and CodeBrain **frequency-token** histogram). **Two estimands, two nulls (≥1000 perms, effect
  size+CI)** (MAJ-1): marginal I(Z;D) with a **subject-permutation (permute d)** null; label-conditional I(Z;D|Y)
  with a **label-permutation (permute y)** null.
- **L4 — task coupling (Q8B-2):** cosine / principal-angle overlap of the source-trained linear **task head** with
  the **label-conditional subject subspace** (top-k of `√n·(mean(z|y,d)−mean(z|y))`, MAJ-4). Pin **PRIMARY_K=2**,
  curve k∈{1,2,4,8}; **disclose rank ceiling** (≤K·(n_src−1)); subspace claims **SHU-MI-primary**.
- **L5 — subject-subspace reliance (Q8B-3):** task-head bAcc drop after erasing the subject subspace, measured on
  **held-out source trials** (nested CV — not in-sample, MAJ-3), against **three nulls** (MAJ-2): (a) **variance-
  matched** subspace (top-k PCA of marginal F0), (b) **oracle task-subspace** (top-k task-discriminative dirs =
  upper bound of any k-erase cost), and reporting the **removed-variance fraction** of every erased subspace.
  Reliance credible only if subject-erase drop **exceeds the variance-matched control**.
- **L6 — target consequence:** held-out-**target** true-label bAcc before/after the source-estimated subject-
  subspace erase (final scoring only). **Interpretive limit (MAJ-5):** this erases a *source*-estimated subspace on
  a *novel* target subject → **conservative**; a **null L6 is NOT evidence that subject info is task-irrelevant**.
  Same variance-matched + oracle controls reported. **Diagnostic intervention, NOT repair.**

## Temporal-token side-check (one-time; then downgrade)
`temporal_token_sidecheck.csv`: unique count + entropy under at most (1) canonical 30-patch window, (2) /100 vs
z-score vs SHU-native norm, (3) an alternate LOCAL checkpoint if present. **No** retrain / codebook / temperature /
post-hoc rescale (STOP-6). Still 1-unique → `temporal_token_status="collapsed"`, excluded from main analysis.
Frequency tokens remain a secondary diagnostic (`frequency_token_status="usable"` if diverse).

## Wording (fixed)
- **Allowed:** "encoder retains decodable subject information"; "subject info is / is not task-coupled";
  "subject-subspace removal changes / does not change target behavior"; "finding is / is not architecture-general
  (holds on CBraMod too)."
- **Forbidden:** "CodeBrain/CBraMod repairs subject shortcuts"; "solves cross-subject generalization"; "encoder X
  leaks more/less than Y" (cross-encoder magnitude — needs 8C); any "foundation pretraining reduces/increases
  leakage" (needs 8C scaling); SOTA/leaderboard; transporting CodeBrain's temporal-token interpretability claims.

## Outputs (`results/fsr_codebrain_cbramod_8b/`)
```
model_load_manifest.csv          # ckpt hashes, params, missing/unexpected, device
feature_dump_manifest.csv        # per model/dataset/subject/trial: channel map (canonical-19 + index), n_patch, rate, QC
encoder_embedding_qc.csv         # F0 + token determinism across repeats + batch groupings (STOP-1)
dataset_split_manifest.csv       # LOSO folds + within-subject session split; source/target; class balance; chance lines
l1_subject_probe.csv             # subject bAcc vs chance; marginal(perm-d) + label-cond(perm-y) nulls; per feature + freq-token
l4_task_head_alignment.csv       # task-head vs label-conditional subject subspace; k-curve; rank ceiling
l5_subject_subspace_replay.csv   # held-out-source bAcc drop; subject vs variance-matched vs oracle-task; removed-var frac
l6_target_consequence.csv        # target true-label bAcc before/after erase (final scoring only) + controls
frequency_token_diagnostic.csv   # freq-token entropy/unique/subject-probe/label-probe (CodeBrain)
temporal_token_sidecheck.csv     # temporal collapse side-check + status
target_label_firewall.json
codebrain_cbramod_8b_verdict.json
```
`codebrain_cbramod_8b_verdict.json`:
```json
{"codebrain_encoder_audit_pass": null, "cbramod_encoder_audit_pass": null,
 "temporal_token_status": "collapsed|rescued|not_tested", "frequency_token_status": "usable|not_usable",
 "l1_per_dataset": null, "l4_per_dataset": null, "l5_per_dataset": null, "l6_per_dataset": null,
 "target_labels_used_for_fit": false, "target_labels_used_for_selection": false, "proceed_to_8c": null}
```

## Encoder-audit gate (MAJ-8; measurability, NEVER a result direction)
`{codebrain,cbramod}_encoder_audit_pass` = **all**: (1) embeddings finite + non-degenerate + **deterministic**
(STOP-1); (2) ≥1 encoder gives **above-chance task** on SHU-MI with **source-only** head selection (STOP-5/8);
(3) L1/L4/L5/L6 tables produced **without target-label leakage**; (4) subject-subspace **stability** across
folds/seeds reported (principal angles) and **SHU-MI subject floor** met (BNCI = sanity only); (5) ≥1000-perm
nulls with CIs. **A NULL result (no decodable/coupled/relied-upon subject info) is a PASS** — the gate is
measurability, not a direction. `proceed_to_8c` gated on measurability only (MIN-1: no cross-encoder / cross-N
claims in 8B).

## STOP rules (PM)
```text
1  either encoder's embeddings not deterministic across repeated inference / batch groupings.
2  SHU-MI labels or subject splits not reproducible.
3  BNCI2014_001 preprocessing silently changes prior FSR labels/splits.
4  target labels enter encoder extraction / head / probe / source-subset selection.
5  BOTH encoders at chance on SHU-MI under frozen linear/shallow heads -> STOP, return (do NOT decide fine-tuning in 8B).
6  temporal-token rescue requires training/tuning.
7  frequency-token diagnostics become the ONLY positive while the encoder audit fails.
8  fixed-trial subject-scaling cannot be matched (8C gate).
9  PhysioNetMI channel mapping not reproducible (8C-0 gate).
```

## After 8B
Return the verdict for PM review **before** 8C and **before** any specialist baseline. PC2 paused; Paper 1
unaffected; **Paper 2 stays frozen** — Phase 8 is a separate empirical axis until 8B/8C results are in.
