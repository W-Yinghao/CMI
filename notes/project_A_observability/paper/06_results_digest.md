# Results Digest

Reviewer-readable narrative of the audited real-EEG digests. **Every number here is copied from a
tracked summary JSON under `results_summaries/`** (checked by
`h2cmi/tests/test_project_A_paper_package.py`). Nothing is recomputed in this file.

## Step 8 — BNCI2014_001 mini-grid

- Dataset BNCI2014_001 (4-class), 9 runs (targets 1/2/3 × seeds 0/1/2).
- Raw mean strict-DG bAcc **0.3383**, mean offline-TTA gain **−0.0926**, mean online-TTA bAcc 0.3206.
- All target metrics oracle/evaluation-only; validator clean. Interface + claim-boundary validation,
  **not** a performance claim. (Older digest schema, before the chance-normalized layer.)

## Step 9 — BNCI2014_001 expanded (4-class)

- 27 runs (subjects 1–9 × target 1–9 × seed 0–2, 50 epochs).
- Mean strict-DG bAcc **0.3946** (chance 0.25), mean offline-TTA gain **−0.0502**, mean online-TTA
  bAcc 0.3753.
- Chance-normalized strict excess **0.1928**; offline-TTA **harm-rate 0.8148** (22/27 cells harmed).
- `missing_cells=[]`; all boundary flags true; target prior `rejected_conclusion_false`.

## Step 10 — BNCI2014_004 (binary)

- 27 runs (`--subjects all --target-subjects all` × seeds 0–2, 50 epochs), K = 2.
- Raw mean strict-DG bAcc **0.6282** (chance 0.5) — **not** comparable to the 4-class 0.3946.
- Chance-normalized strict excess **0.2563**; offline-TTA gain-norm −0.0639; **harm-rate 0.8519**.
- `missing_cells=[]`; all boundary flags true.

## Step 10 — BNCI2015_001 (legal skip)

- 0 ok / 36 skipped. The dataset is not a left/right-hand task, so the binary `LeftRightImagery`
  paradigm rejects it (`AssertionError: Dataset BNCI2015-001 is not valid for paradigm`) — a **legal
  skip**, reported verbatim. Not a missing cache.
- `claim_boundary_status = not_applicable_all_skipped`; boundary flags **null** (it asserts no target
  metric); `missing_cells=[]`; still counted **valid**.

## Step 10 — combined MOABB digest

- 54 ok runs across the 2 ok datasets (BNCI2014_001 + BNCI2014_004); BNCI2015_001 all-skipped and
  excluded from the boundary roll-up but reported (`datasets_all_skipped=["BNCI2015_001"]`).
- Overall chance-normalized strict excess **0.2246**, offline-TTA gain-norm **−0.0654**, offline-TTA
  **harm-rate 0.8333**.
- Raw bAcc is **never** pooled across the differing class counts (`raw_bacc_overall_suppressed=true`,
  `mixed_n_classes=true`).

## Interpretation (bounded)

- **Normalization matters.** BNCI2014_004's raw 0.6282 looks far above BNCI2014_001's 0.3946, but that
  is a chance artifact (binary vs 4-class). After normalization the two are comparable
  (excess 0.2563 vs 0.1928).
- **TTA often hurts here.** Offline-TTA harms the oracle target metric in most cells (pooled harm-rate
  0.8333). This is evidence of **measurement/adaptation fragility**, not a theorem and not a SOTA
  claim.
- **The metrics stay oracle/evaluation-only.** Every target bAcc/gain above is computed with oracle
  target labels for *evaluation*; none is claimed identifiable under the R0/R1 deployment regime
  (`identifiable_estimand=null`).
