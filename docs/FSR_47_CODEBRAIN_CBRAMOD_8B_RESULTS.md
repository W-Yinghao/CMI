# FSR_47 — CodeBrain + CBraMod 8B Encoder-Audit Results (Phase 8B; PRELIMINARY)

**Project FSR — Phase 8B.** First results of the frozen-encoder FSR audit (protocol FSR_46). SHU-MI (25 subj, 5
sessions, 2-class MI); frozen **CodeBrain** (EEGSSM, GPU) + **CBraMod** (single-stage, control) pooled window
embedding `F0∈R^200`; L1/L4/L5/L6 with permutation / variance / oracle nulls, per-dataset, target labels final
scoring only. **PRELIMINARY** — two open issues below gate the task-coupling conclusions and the 8C go.

## Headline (robust part) — subject is strongly ENCODED but not RELIED-UPON, and it is architecture-general
| metric (SHU-MI, 25 subj, chance in ⟨⟩) | CodeBrain | CBraMod |
|---|---|---|
| **L1 subject decodability** ⟨0.04⟩, session-held-out, perm-null p | **0.591** (eff +0.55, p=0) | **0.555** (eff +0.52, p=0) |
| L1 **class-conditional** (subject info beyond the task label) | 0.591 | 0.556 |
| **L4** task-head ↔ subject-subspace alignment (k=2) | **0.000** | 0.001 |
| **L5** reliance = held-out-source bAcc drop after subject-erase; beats variance null? | +0.0008, **No** | −0.0002, **No** |
| **L6** target consequence Δ (before−after erase) | −0.0017 (CI∋0) | −0.0006 (CI∋0) |

**Solid finding (L1):** both frozen foundation encoders **strongly and class-conditionally encode subject
identity** (≈0.55–0.59 balanced accuracy at 25-way, vs 0.04 chance, permutation p=0) — decodable subject structure
survives large-scale pretraining. **And it is not used by a linear task head** (L4≈0, L5 does not beat a variance-
matched erase, L6≈0). The pattern **replicates across two different architectures** (EEGSSM/SGConv vs criss-cross
transformer) ⇒ not CodeBrain-specific. This is the FSR thesis *at the frozen-foundation-encoder level*: **subject
encoding ≠ harmful subject reliance** (measurable ≠ relied-upon).

## Open issue 1 — task decodability is WEAK on the pooled embedding (near STOP-5); L4/L5/L6 are PRELIMINARY
Frozen `F0` + linear head decodes SHU-MI 2-class MI at only **0.530 (CodeBrain) / 0.548 (CBraMod)** (chance 0.5).
`F0` **mean-pools over channels**, which washes out MI's **C3/C4 mu/beta lateralization** — the discriminative
signal MI needs. So "subject not task-coupled" (L4/L5/L6≈0) is *confounded* by "the pooled embedding barely
represents the task." **L1 is unaffected** (subject decodability doesn't need the task). **Recommended fix before
trusting L4/L5/L6:** re-dump a **spatially-preserving** feature (per-channel: mean over patches only, keep
channels → C×200; PCA-reduced in the audit) so MI decodes strongly, then re-run L4/L5/L6. This converts
"not-task-coupled on a weak task (ambiguous)" into a solid statement (or triggers STOP-5 if the task stays weak).

## Open issue 2 — CodeBrain determinism (STOP-1)
CodeBrain `F0` is **repeat-exact (max-abs-diff 0.0)** but shows a **≤2e-3 batch-size numerical path** (the SGConv
FFT-conv reduces slightly differently across batch sizes; persists under `use_deterministic_algorithms`). It is
**immaterial to the L1 effect (0.55 ≫ 2e-3)** but is not bit-deterministic across batch groupings. **CBraMod is
fully deterministic (3e-7).** Options: dump CodeBrain at fixed batch (reproducible-by-scheme, current) + prove
audit-invariance, or per-trial (bs=1) for strict bit-determinism.

## Firewall / discipline (clean)
`target_label_firewall.json`: L6 is the only target-label read (final scoring). Subject probe, task head, and all
subspaces fit on source / within-source session split; z-score per-trial within-window. No target label in feature
extraction / head / probe / subspace / selection. Determinism, per-dataset reporting (2-class only here), and
permutation/variance/oracle nulls per FSR_46. Temporal-token side-check still pending (`collapsed` from preflight);
frequency-token diagnostic pending.

## Gate status
`measurability = PASS` (L1/L4/L5/L6 all produced with meaningful nulls; L1 strongly non-null). **`proceed_to_8c` =
HELD** pending (a) resolution of the weak-task/STOP-5 issue via the spatial-feature re-dump (or an explicit PM
decision to accept the pooled-feature limitation), and (b) the determinism disposition. **Not proceeding to 8C
until the PM rules on issue 1 (STOP-5-adjacent).** BNCI2014_001 alignment-sanity + temporal side-check to follow.

## Deliverables (`results/fsr_codebrain_cbramod_8b/`)
`{codebrain,cbramod}_shu_F0.npz`, `feature_dump_manifest_*.json`, `audit_summary_*.json`, `l1_subject_probe.csv`,
`l4_task_head_alignment.csv`, `l5_subject_subspace_replay.csv`, `l6_target_consequence.csv`,
`codebrain_cbramod_8b_verdict.json`, `target_label_firewall.json`. PC2 paused; Paper 1 unaffected; Paper 2 frozen.
