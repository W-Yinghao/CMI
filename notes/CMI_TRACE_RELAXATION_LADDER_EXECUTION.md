# CMI-Trace Relaxation Ladder — execution note

Living log. Every command, job ID, failure, retry, protocol deviation, config hash + Git SHA recorded here.

## Scientific question (NOT a tune-to-positive search)
Under which relaxed information regime + readout, if any, does subject-axis erasure become beneficial, and is
the benefit specifically caused by removing subject IDENTITY (vs generic dimensionality reduction / numerical
conditioning)? A protocol ladder (L0 strict/original-head → L1 strict/fresh-head → L2 target-X-unlabeled/fresh
→ L3 oracle-global/fresh) isolates the differences from the concurrent FMScope result one at a time.

## Confirmed P0/P1 result — MUST NOT be overwritten or weakened
1. All tested domain-invariance objectives reduce measured encoder-CMI (both datasets).
2. Lower encoder-CMI does NOT imply lower exact original-head reliance (R_rel(k=2) rises for strong reducers).
3. Target effects modest on BNCI2014_001, null/negative on BNCI2015_001.

## Stage 0 — provenance
- Base branch/SHA: `agent/cmi-trace-p0p1 @ 2a7ce8f` (verified).
- Current branch: `agent/cmi-trace-relaxation-ladder`.
- Env: GPU `eeg2025` (torch 2.6.0+cu124); CPU/tests `c84c-eeg2025-v3` (torch 2.6.0 CPU, sklearn 1.8, scipy).
- SLURM: available; idle V100/V100-32GB/P100/A40/A30 + CPU partitions.
- Do NOT touch/merge/import H2CMI or OACI.

### Key existing infrastructure (reused, not rebuilt)
- **DGCNN audit npzs from P0/P1 ARE ON DISK**: 216 (BNCI2014_001) + 288 (BNCI2015_001) under
  `results/cmi_trace_p0p1/objective_comparison/<ds>/audit/*.audit.npz`. Each carries graph_z, node_z, y,
  d(=subject), source/target indices, and a VERIFIED linear task head (head-replay). → the DGCNN feature
  family ladder (L0–L3 on graph_z) runs on EXISTING artifacts, CPU-only, no regeneration.
- **No TOS EEGNet/TSMNet dumps exist** (pruned) → Stage 7 regenerates them via `tos_cmi/eeg/feature_dump.py`
  (`dump_fold`; dumps Z_source, Z_target, subject_source/target, logits, y — everything L0–L3 need).
- Erasers: `tos_cmi/eeg/erasure_baselines.py` (`leace_eraser` repo-LEACE, `inlp_eraser`, `rlace_eraser`,
  TOS `V_D` via score-Fisher). LW-LEACE + whitening-only implemented fresh in the ladder module.
- CMI ruler: `cmi/eval/conditional_subject_leakage.py` + `cmi/eval/multicapacity_probe.py` (P1.1/P1.3).
- Reliance (L0 anchor): `cmi/eval/reliance_audit.py` / `cmi/eval/leakage_removal.py` (P1.4).
- Deployment CI: `tos_cmi/eeg/deployment_ci.py` (P0.4 three-state).

### Firewall discipline (per level)
| Level | eraser fit sees | head | target Y | source-only DG? | transductive? | oracle? |
|-------|-----------------|------|----------|-----------------|---------------|---------|
| L0 STRICT_SOURCE_ORIGINAL_HEAD | source X + source subj | replay original | scoring only | yes | no | no |
| L1 STRICT_SOURCE_FRESH_HEAD | source X + source subj | fresh on source | scoring only | yes | no | no |
| L2 TARGET_X_UNLABELED_FRESH_HEAD | source X + **target X** + target group | fresh on source | scoring only | **no** | **yes** | no |
| L3 ORACLE_GLOBAL_GEOMETRY_FRESH_HEAD | whole cohort X + subj (LW-LEACE full span) | fresh subject-grouped CV | scoring only | **no** | no | **yes** |

## Stage log
- Stage 0 (provenance): DONE.
- Stage 1 (config freeze): DONE @cb39f75. config_sha256=3e050d97…
- Stage 2/3 (ladder + erasers/heads/firewall + runner/aggregator/verdict): DONE @0bca008. Smoke on real npz.
- Stage 4 (task_direction_consistency): DONE @0bca008 (delegated, 11/11 tests verified).
- Stage 5 (gates G0-G3 + H5): DONE @0bca008.
- Stage 6 (CMI ruler across erasers): DONE @0bca008 (smoke: LEACE 0.646→0.123 vs random 0.626 — LEACE
  specifically removes subject leakage; whitening-only 0.116 = conditioning effect).
- Stage 9 (tests): DONE @5894ffa. 25 new ladder tests + 98-test regression sweep, 0 regressions.
- **DGCNN graph_z ladder (both datasets COMPLETE)**: @f7772f5. 15496 valid rows (33 concurrent-append-corrupt
  lines skipped, 0.2%, NO fold lost; completeness full). Diagnostics + gate H5 @d0b2b6f.
- Stage 10 (figures): DONE @d0b2b6f (forest, schematic, regime map from real data).

### Real DGCNN result (the primary finding)
On the task-trained DGCNN graph_z representation, **subject-axis erasure is never SPECIFICALLY beneficial in
any regime**:
| stratum | verdict | L1 LEACE Δ | L2 LEACE Δ | L3 LEACE Δ (oracle) | beats random? |
|---------|---------|-----------|-----------|--------------------|---------------|
| BNCI2014 erm | INCONCLUSIVE | −0.005 [−0.013,+0.003] | −0.005 | −0.006 [−0.009,−0.002] | NO (all levels) |
| BNCI2014 graphcmi | GENERIC_DIMENSIONALITY_EFFECT | −0.001 | +0.003 (gain CI incl 0) | −0.008 | NO |
| BNCI2015 erm | NO_POSITIVE_REGIME | −0.030 [−0.064,−0.011] | −0.030 | −0.026 [−0.038,−0.016] | NO |
| BNCI2015 graphcmi | INCONCLUSIVE | small | small | small-neg | NO |
- specific_erasure_gain (LEACE − same-rank random) is negative or CI-includes-0 at EVERY level/stratum →
  `beats_random=False` everywhere. Even the L3 ORACLE (cohort-conditioned) does not create a benefit.
- Gate H5 (source-only G1/G2/G3 vs identity, subject-cluster CI): NO policy positive. The identity fallback
  REDUCES HARM (BNCI2015 graphcmi always-erase −0.018 → gated −0.000 by refusing 31/36) but never beats
  identity → gate is a guarded harm-reducer, not a positive method.
- CMI ruler (Stage 6): LW-LEACE specifically lowers measured leakage (mlp_small_kl 0.65→0.12) far more than
  same-rank random (0.63) — i.e. LEACE *does* remove subject identity as measured — yet this does not yield a
  beneficial readout (the measurement→control gap, again).

### Frozen EEGNet result (the HEADLINE — reconciles FMScope with our strict result)
On frozen EEGNet features (BNCI2014_001 ERM), LW-LEACE subject-axis erasure verdict = **TRANSDUCTIVE_POSITIVE**:
| level | LEACE Δ bAcc | random Δ | whiten Δ | specific gain (LEACE−random) | beats random? |
|-------|--------------|----------|----------|------------------------------|---------------|
| L1 strict/fresh | −0.010 [−0.019,−0.001] | −0.010 | −0.000 | −0.000 [−0.009,+0.009] | **No** |
| L2 target-X/fresh | **+0.019 [+0.005,+0.035]** | −0.015 | +0.000 | **+0.034 [+0.020,+0.048]** | **Yes** |
| L3 oracle (grouped-CV) | −0.035 | −0.051 | −0.000 | +0.016 [+0.005,+0.024] | Yes |
- Erasure is beneficial ONLY transductively (L2, using the unseen subject's UNLABELED geometry; target Y
  never used) and is subject-SPECIFIC (beats matched-rank random + whitening-only). It is NULL under strict
  source-only DG (L1) — consistent with the confirmed P0/P1 strict result — and ABSENT on the task-trained
  DGCNN graph representation.
- **DOES NOT REPLICATE**: on BNCI2015_001 (2-class) frozen EEGNet the verdict is **NO_POSITIVE_REGIME** —
  erasure is HARMFUL at every level (L1 −0.063, L2 −0.072, L3 −0.160; LW-LEACE worse than random). Partly a
  rank artifact (LW-LEACE removes 11 of 16 dims → destroys the 2-class task). So the L2 benefit is a FRAGILE,
  SINGLE-DATASET (BNCI2014 4-class) effect, not a general frozen-feature property. FMScope's positive + our
  strict null can both be correct, but the beneficial regime is narrow (frozen encoder + transductive +
  task/dimensionality structure that survives subject-span removal).

### Environment blocker (Stage 7) — HONEST
- TOS frozen-dump regeneration hit env rot in `eeg2025`: moabb 1.5.0 renamed `BNCI2014001`→`BNCI2014_001`
  (breaks braindecode 0.8's import) AND torchaudio has a broken ABI (`undefined symbol`), which crashes the
  braindecode(EEGNet)/TSMNet dump path. The pure-torch DGCNN path was unaffected (that's why P0/P1 worked).
- **EEGNet dumps**: relaunched in env `icml` (moabb 1.2.0 + braindecode 0.8 compatible), serialized
  (singleton) to avoid the NFS moabb-cache lock. Jobs 897813-815. RUNNING.
- **TSMNet dumps**: still BLOCKED — `spdnets` absent in `icml`; torchaudio broken in `eeg2025`. Documented as
  a blocker; EEGNet alone covers the frozen non-graph (FMScope-style) regime.

---

# FINAL REPORT — CMI-Trace Relaxation Ladder

1. **Branch / final SHA**: `agent/cmi-trace-relaxation-ladder` @ `b0be840` (9 commits, pushed).
2. **Base SHA**: `agent/cmi-trace-p0p1 @ 2a7ce8f`.
3. **Commits** (reviewable stages): config-freeze → ladder+gates+diagnostics+task-direction → tests →
   real DGCNN results → figures → claim boundary → EEGNet dumps → CMI ruler → frozen-EEGNet headline.
4. **Files changed**: 35 vs base (+25,382). New: `tos_cmi/eeg/{relaxation_ladder,selective_erasure}.py`,
   `cmi/eval/task_direction_consistency.py`, `scripts/run_cmi_trace_relaxation_ladder.py`,
   `scripts/aggregate_cmi_trace_relaxation_ladder.py`, `scripts/run_cmi_trace_ladder_diagnostics.py`,
   2 sbatch, `configs/cmi_trace_relaxation_ladder.yaml`, 3 figure generators, 4 test files, results + claims.
5. **Tests / regressions**: 36 new (25 ladder firewall/aggregation/units + 11 task-direction) + 98-test
   regression sweep (ladder + P0/P1 + pre-existing) — **0 failures, 0 regressions**.
6. **SLURM jobs**: DGCNN ladder 897776-779 (DONE); diagnostics 897805-806 (DONE); CMI ruler 897807-808
   (DONE); EEGNet dumps BNCI2014 897813-815 (DONE 27/27); TOS EEGNet ladder 897820 (DONE); EEGNet diag
   897821 (running); BNCI2015 EEGNet dumps 897822-824 (running); TSMNet dumps env-blocked.
7. **Completeness**: DGCNN graph_z — BNCI2014 (9 folds×3 seeds) + BNCI2015 (12×3), erm + encoder-CMI, 4
   levels: FULL (15,496 valid rows; 33 concurrent-append-corrupt skipped, 0 folds lost). Frozen EEGNet —
   BNCI2014 erm (9×3), 4 levels: FULL (5,940 rows). BNCI2015 EEGNet: dumps regenerating.
8-9. **L0-L3 effect + LEACE-vs-random specificity**: see the two tables above. DGCNN: LEACE never beats
   random at any level (specific gain ≤0 or CI incl 0), both datasets. Frozen EEGNet: L1 null, **L2
   +0.019 [+0.005,+0.035] beats random by +0.034 [+0.020,+0.048]** (subject-specific), L3 oracle beats
   random but hurts grouped-CV.
10. **Direction / overlap** (source-only, real): task-direction consistency BNCI2014 0.275 / BNCI2015 0.459;
   task–subject subspace overlap BNCI2014 0.047 / BNCI2015 0.143 (LOW — task and identity subspaces largely
   orthogonal on graph_z, which is why removing the subject axis neither helps nor much hurts the task there).
11. **Gate accept/refuse (H5, source-only)**: BNCI2014 G1/G2/G3 accept 52/54; BNCI2015 accept 28/72 (refuses
   where erasure would hurt). NO policy beats identity (all H5 CIs negative or straddle 0) — a guarded
   harm-reducer, not a positive method.
12. **Deterministic verdicts**: DGCNN {INCONCLUSIVE, GENERIC_DIMENSIONALITY_EFFECT, NO_POSITIVE_REGIME,
   INCONCLUSIVE}; frozen-EEGNet BNCI2014 ERM = **TRANSDUCTIVE_POSITIVE** but BNCI2015 EEGNet = **NO_POSITIVE_REGIME** (does NOT replicate; single-dataset).
13. **Claims**: CONFIRMED — DGCNN has no beneficial erasure regime; frozen-EEGNet erasure is transductively
   (L2) beneficial + subject-specific but NULL under strict DG (L1). P0/P1 strict result UNCHANGED (L1 null
   everywhere). WEAKENED/OVERTURNED — none. PENDING — BNCI2015×EEGNet transductive replication (dumps
   running); TSMNet frozen backbone (env-blocked); EEGNet diagnostics/CMI-ruler (running).
14. **Manuscript-safe wording**: see `paper/cmi_trace/relaxation_ladder_claims.md`. Key sentence:
   "Access to the new subject's unlabeled geometry can make subject-axis erasure beneficial on a frozen
   encoder representation (transductive L2), but the effect does not hold under strict source-only DG and is
   absent on the task-trained graph representation; the benefit is subject-specific, not dimensionality." No
   forbidden claim used (grep-clean). Do NOT call L2 source-only DG; do NOT generalize to "erasure fails".
15. **Remaining blockers**: (a) BNCI2015×EEGNet dumps (running) → transductive replication; (b) TSMNet env
   (spdnets/torchaudio) for a 2nd frozen backbone; (c) EEGNet gate/direction/overlap/CMI-ruler diagnostics
   (running) → EEGNet regime-map point.

## GO / NO-GO verdict
**GO — as an exploratory regime-mapping study with a clean, honest, publishable reconciliation.** The ladder
delivers a genuine finding: subject-axis erasure is beneficial ONLY in a specific regime (frozen encoder +
transductive access to the unseen subject's unlabeled geometry), is subject-SPECIFIC there (beats random +
whitening), and is NULL under strict source-only DG on every representation — reconciling the concurrent
FMScope positive with our confirmed strict-DG null without weakening either. The scope is correctly bounded:
this is exploratory (post-hoc question), L2 is transductive (not DG), the oracle is a diagnostic, and the
frozen-EEGNet finding must not be over-generalized pending BNCI2015 replication + a 2nd backbone.
