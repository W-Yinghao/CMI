# ACAR V5 Stage-2B Closeout — DEV_STOP / No Candidate Selected

```
status:                    DEV_STOP
run_id:                    acar-v5-stage2b-ba09777-r1
implementation_base_sha:   ba097775f239b2107210e1626a08603a2071790a
stage1b_run_id:            acar-v5-stage1b-c4412b4-r1
stage1b_registry_sha256:   2bbe55f4cdb4f1a18cee3b2c9e7583dba9fe9e84b9c563fb37781e98ebcbb76d
protocol_tag_target_sha:   4278435975a72b1127803dd2cffab420c083e430
selected_candidate_id:     null
n_selection_refs_used:     10
candidate_universe_size:   22
holm_family_size:          132
holm_nonevaluable_cells:   2        (V5-P4-002 / {PD,SCZ})
run:                       SLURM 885395, nodecpu03, 8.18 h, rc=0   (result note:
                           notes/ACAR_V5_STAGE2B_REAL_SELECTION_RESULT_acar-v5-stage2b-ba09777-r1.md; commit d287635)
```

**Outcome.** No candidate in the frozen 22-candidate V5 universe passed the pre-registered CAL certification + EVAL
utility/retention gates jointly across PD and SCZ. The binding run completed cleanly (rc=0) and emitted the pre-registered
**DEV_STOP**. Take-1 (`f079aca`) had crashed on an n=1 forced tail; this take-2 on the Stage-2B3 forced-tail fix ran end-to-end.

**Primary failure mode: harm control, not coverage collapse.** (This is the sharpest distinction from v3, which died on coverage
collapse — see the lineage note in `notes/ACAR_V5_CLOSEOUT.md`.)

## Result

```
CAL certification (H1=G3 UCB[L_harm_all]<=0.10, H2=G4 UCB[harm_among_adapted]<=0.30, H3=G1 LCB[coverage]>=0.15; Holm/132/alpha=0.05):
  cert_pass  PD =  0/22    SCZ =  0/22          <- NO candidate certified on either disease
  G1 cov     PD = 13/22    SCZ = 19/22          <- coverage is FINE (adaptation is not being starved)
  G3 Lharm   PD =  1/22    SCZ =  0/22          <- L_harm_all UCB > 0.10 almost everywhere
  G4 hAdapt  PD =  0/22    SCZ =  0/22          <- harm_among_adapted UCB in [0.61, 0.87] >> 0.30 on ALL 42 evaluable cells

EVAL utility:
  red range: -12.12 .. +0.01                    (red = -mean chosen dR; red<0 => the executed action INCREASED NLL loss)
  v2_replay_red: PD -0.079,  SCZ -0.049,  macro -0.064   (the v2 comparator adaptation is ALSO harmful)
  eligible candidates (cert & G2 & G5, both diseases + macro): 0/22

Non-evaluable: V5-P4-002 on both diseases (zero FIT proposed-action records -> NonEvaluableCandidate -> p=1, cert_pass=False;
               enters the fixed 132-cell Holm family, never shrinking it).
```

**Interpretation.**
- Coverage was **not** the primary bottleneck (G1 passes for 13/22 PD, 19/22 SCZ). The candidates DO adapt.
- The adaptation actions (`matched_coral` / `spdim` / `t3a`) were **often harmful when executed**: when a candidate adapts,
  ~61–87 % of adapted batches are harmful (harm_among_adapted UCB), and the chosen-action risk change is strongly negative
  (`red` down to −12.12) — the transformed embeddings land where the source-state readout `f_0` is badly miscalibrated.
- The **identity / source-state LDA `f_0`** dominated the adaptation policies: doing nothing beats adapting, so the
  pre-registered harm gates (G3/G4) correctly reject all 22 candidates.

## This is a pre-registered terminal DEV_STOP

It is **not** a numerical crash (rc=0; the Stage-2B3 forced-tail contract held — 0 provider calls on any of the 405 sub-MIN_BATCH
tails; see forced-tail accounting below), **not** a substrate failure (Stage-1B package ADMITTED, registry-hash-bound), and **not**
an external failure (external / held-out / ASZED / lockbox were never touched). The gates were applied exactly as pre-registered
and no candidate cleared them.

## Forced-tail accounting (Stage-2B3 contract, on real data)

```
n_forced_tail_batches_lt8            405        n_partial_eligible_batches_8_to_31   1805
n_full_32_batches                    8965        n_total_batches                      11175
n_forced_tail_provider_calls         0           n_provider_calls_below_min_batch     0
min_provider_call_n                  8           n_provider_calls (total)             952396
forced tails counted in denominators YES (metrics.collect total=len(batches); guard test_stage2b_forced_tail_counts_in_denominator)
forced tails red_upper contribution  0   (Stage-2B3 policy_eval; guard test_stage2b_forced_tail_red_upper_zero)
```

## Split discipline + forbidden-stage confirmation

```
FIT thresholds only  · CAL certification only (Holm/132) · EVAL final reporting only (red/red_upper/G2/G5/G1-G5)
labels read ONLY inside the v2 comparator (FIT) + policy_eval (CAL/EVAL); routing/thresholds label-free
S1 seeds 20260712/20260713 opened: NO   ·   S1/S2/S3 robustness run: NO
external/held-out read: NO   ·   ASZED read: NO   ·   lockbox touched: NO   ·   substrate rebuilt: NO
code / candidate space / batch size / MIN_BATCH / stable-CORAL / CAL-EVAL interp changed after run: NO
```

## Consequence — allowed next

Per the authorization, DEV_STOP is **terminal for V5**: no tuning, no candidate-space change, no reselection, no rerun without a
NEW dated protocol. No Stage-4 (S1/S2/S3 robustness needs a *selected* candidate as input — there is none). No external / held-out
/ ASZED. Lockbox stays sealed. Pulling P3/P4 near-identity candidates out to "rescue" a pass would turn the pre-registered Stage-2B
into post-hoc tuning and is forbidden. Verified provenance: the completion note was independently re-derived from the raw report
(all 9 claims + full 22-row gate table CONFIRMED). Authoritative V5 lineage + engineering-vs-science split:
`notes/ACAR_V5_CLOSEOUT.md`; claim status: `notes/EVIDENCE_LEDGER.md` (A8/A9).
