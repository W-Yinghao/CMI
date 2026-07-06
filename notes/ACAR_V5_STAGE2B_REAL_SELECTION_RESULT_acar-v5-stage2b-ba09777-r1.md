# ACAR V5 — Stage-2B real DEV candidate selection (RESULT: DEV_STOP)

```
status: DEV_STOP

run_id:                    acar-v5-stage2b-ba09777-r1
implementation_base_sha:   ba097775f239b2107210e1626a08603a2071790a
stage1b_run_id:            acar-v5-stage1b-c4412b4-r1
stage1b_registry_sha256:   2bbe55f4cdb4f1a18cee3b2c9e7583dba9fe9e84b9c563fb37781e98ebcbb76d
protocol_tag_target_sha:   4278435975a72b1127803dd2cffab420c083e430

n_selection_refs_used:     10       (PD/SCZ x fold0..4 x seed20260711)
candidate_universe_size:   22
holm_family_size:          132      (22 candidates x 2 diseases x {H1,H2,H3})
holm_nonevaluable_cells:   2        (V5-P4-002 / {PD,SCZ}: NonEvaluableCandidate -> p=1, cert_pass=False)
selected_candidate_id:     null
objective:                 maximize min_disease(red - v2_replay_red)
elapsed:                   8.18 h   (SLURM job 885395, nodecpu03; ~952k action-provider calls — 22x recomputation + v2 replay)
dev_stop_reason:           no candidate passed G1-G5 (both diseases + macro)
```

The binding run COMPLETED (rc=0) and emitted a pre-registered **DEV_STOP**: no candidate cleared the certification + utility gates.
This is take-2, pinned to the Stage-2B3 forced-tail fix (`ba09777`); take-1 (`f079aca`) crashed on an n=1 forced tail. This run ran
end-to-end without crashing — the forced-tail contract held on real labels (see accounting below). Per the authorization,
DEV_STOP => stop: NO tuning, NO candidate-space change, NO reselection, NO rerun.

## Pre-flight (both passed)

- No-label `--guard` PASS: worktree HEAD == ba09777, ADMITTED, registry sha matched, exactly the 10 seed-711 refs + 22 candidates,
  S1 seeds not opened, no forbidden site token, auth valid+bound, labels not read.
- Adversarial 4-lens launcher review (spy-correctness / pin+guard / label-firewall / discipline): 0 findings (verified the agents
  inspected the launcher, incl. the new pass-through spy).

## Why DEV_STOP — the adaptations are net harmful on real DEV data

CAL certification (H1=G3 UCB[L_harm_all]<=0.10, H2=G4 UCB[harm_among_adapted]<=0.30, H3=G1 LCB[coverage]>=0.15; Holm over the
fixed 132-cell family at alpha=0.05):

```
gate pass counts (of 22, per disease):
  cert_pass  PD= 0/22   SCZ= 0/22       <- NO candidate is CAL-certified on either disease
  G1 cov     PD=13/22   SCZ=19/22       <- coverage is fine (the problem is NOT too little adaptation)
  G3 Lharm   PD= 1/22   SCZ= 0/22       <- L_harm_all UCB > 0.10 almost everywhere
  G4 hAdapt  PD= 0/22   SCZ= 0/22       <- harm_among_adapted UCB in [0.61, 0.87] >> 0.30 EVERYWHERE
  G2 red     PD= 2/22   SCZ= 3/22
  G5         PD= 0/22   SCZ= 0/22
  eligible (cert & G2 & G5, both diseases + macro): 0/22
```

`red` (= -mean chosen ΔR, NLL) is strongly NEGATIVE across candidates (range **-12.12 .. +0.01**): the routed actions
(matched_coral/spdim/t3a) produce much higher NLL than the identity LDA f_0 — i.e. adapting HURTS. When a candidate does adapt,
~61-87% of adapted batches are harmful (harm_among_adapted UCB), so the pre-registered harm gates (G3/G4) reject all 22. The few
near-zero-`red` candidates (P3-001/002, P4-001) abstain to near-identity but still fail G4 and G5. Conclusion: on the real PD/SCZ
DEV target, no action policy in the frozen 22-candidate universe is safe/beneficial; the identity (no-adapt) baseline dominates.

## Per-candidate table (CAL cert + EVAL gates; label-based, DEV split)

```
cid          fam | PD: cert G1 G3 G4 G2 G5  cov  Lharm hAdapt   red   | SCZ: cert G1 G3 G4 G2 G5  cov  Lharm hAdapt   red
V5-P1-001    P1  |  0    1  0  0  0  0  0.17 0.21  0.73   -2.82  |   0    1  0  0  0  0  0.47 0.42  0.70   -8.69
V5-P1-002    P1  |  0    1  0  0  0  0  0.20 0.23  0.71   -3.31  |   0    1  0  0  0  0  0.50 0.44  0.70   -9.06
V5-P1-003    P1  |  0    0  0  0  0  0  0.07 0.14  0.79   -1.42  |   0    1  0  0  0  0  0.18 0.24  0.73   -4.28
V5-P1-004    P1  |  0    0  0  0  0  0  0.07 0.15  0.78   -1.48  |   0    1  0  0  0  0  0.18 0.24  0.73   -4.32
V5-P2-001    P2  |  0    1  0  0  0  0  0.17 0.21  0.73   -2.82  |   0    1  0  0  0  0  0.47 0.42  0.70   -8.69
V5-P2-002    P2  |  0    1  0  0  0  0  0.20 0.23  0.71   -3.31  |   0    1  0  0  0  0  0.50 0.44  0.70   -9.06
V5-P2-003    P2  |  0    0  0  0  0  0  0.07 0.14  0.79   -1.42  |   0    1  0  0  0  0  0.18 0.24  0.73   -4.28
V5-P2-004    P2  |  0    0  0  0  0  0  0.07 0.15  0.78   -1.48  |   0    1  0  0  0  0  0.18 0.24  0.73   -4.32
V5-P3-001    P3  |  0    1  0  0  1  0  0.55 0.54  0.79   +0.00  |   0    1  0  0  1  0  0.64 0.47  0.61   +0.01
V5-P3-002    P3  |  0    1  0  0  1  0  0.66 0.61  0.77   +0.00  |   0    1  0  0  1  0  0.82 0.56  0.62   +0.01
V5-P3-003    P3  |  0    1  0  0  0  0  0.60 0.58  0.80   -4.08  |   0    1  0  0  0  0  0.74 0.57  0.69  -10.45
V5-P3-004    P3  |  0    1  0  0  0  0  0.70 0.61  0.77   -5.47  |   0    1  0  0  0  0  0.87 0.64  0.69  -12.12
V5-P3-005    P3  |  0    1  0  0  0  0  0.52 0.54  0.86   -1.13  |   0    1  0  0  0  0  0.46 0.49  0.85   -0.84
V5-P3-006    P3  |  0    1  0  0  0  0  0.66 0.67  0.87   -1.28  |   0    1  0  0  0  0  0.65 0.63  0.85   -1.14
V5-P4-001    P4  |  0    1  0  0  0  0  0.36 0.40  0.76   -0.00  |   0    1  0  0  1  0  0.72 0.50  0.62   +0.01
V5-P4-002    P4  |  NON-EVALUABLE (zero FIT proposed-action records)  |  NON-EVALUABLE
V5-P5-001    P5  |  0    1  0  0  0  0  0.28 0.29  0.72   -4.49  |   0    1  0  0  0  0  0.60 0.51  0.70  -10.61
V5-P5-002    P5  |  0    1  0  0  0  0  0.20 0.23  0.71   -3.31  |   0    1  0  0  0  0  0.50 0.44  0.70   -9.06
V5-P5-003    P5  |  0    0  0  0  0  0  0.14 0.19  0.74   -2.35  |   0    1  0  0  0  0  0.35 0.35  0.70   -6.46
V5-P5-004    P5  |  0    0  0  0  0  0  0.07 0.15  0.78   -1.48  |   0    1  0  0  0  0  0.18 0.24  0.73   -4.32
V5-P5-005    P5  |  0    0  0  0  0  0  0.03 0.11  0.76   -0.97  |   0    0  0  0  0  0  0.11 0.19  0.76   -3.04
V5-P5-006    P5  |  0    0  1  0  0  0  0.00 0.09  0.87   -0.56  |   0    0  0  0  0  0  0.04 0.14  0.83   -2.10
```

(cov = coverage_lcb (G1 vs 0.15); Lharm = l_harm_all_ucb (G3 vs 0.10); hAdapt = harm_among_adapted_ucb (G4 vs 0.30);
red = -mean chosen ΔR (G2/G5 utility). cert = CAL H1-H3 Holm certification. reason rejected: EVERY candidate fails CAL cert
(G3/G4 harm) and EVAL G5; none is eligible. reason selected: none.)

## Forced-tail accounting (Stage-2B3 contract, on real data)

```
n_forced_tail_batches_lt8               405
n_partial_eligible_batches_8_to_31      1805
n_full_32_batches                       8965
n_total_batches                         11175
n_forced_tail_provider_calls            0          (no forced tail routed through any non-identity action)
n_provider_calls_below_min_batch        0          (the provider was NEVER called on a sub-MIN_BATCH batch — not even identity)
min_provider_call_n                     8
n_provider_calls (total)                952396
forced tails counted in denominators    YES  (Stage-2B3 metrics.collect total=len(batches); guard test_stage2b_forced_tail_counts_in_denominator)
forced tails red_upper contribution = 0 YES  (Stage-2B3 policy_eval; guard test_stage2b_forced_tail_red_upper_zero)
```

The run processed all 405 forced tails without routing any to matched_coral/spdim/t3a and without a single non-finite crash — the
exact failure mode that killed take-1 is resolved on the real package.

## v2 replay comparator

```
PD  v2_replay_red    -0.07914164385821819
SCZ v2_replay_red    -0.04881504960645789
macro_v2_replay_red  -0.06397834673233804
```

The v2 comparator adaptation is ALSO harmful (negative red) on real DEV data — consistent with the DEV_STOP.

## Split discipline

```
FIT thresholds only        : thresholds fit on FIT (train u val) eligible batches only
CAL certification only      : H1-H3 Holm(132) computed on CAL
EVAL final reporting only    : red / red_upper / G2 / G5 / final G1-G5 on EVAL
labels                       : read ONLY inside the v2 comparator (FIT) + policy_eval (CAL/EVAL); routing/thresholds label-free
```

## Forbidden-stage confirmation

```
S1 seeds 20260712 / 20260713 opened:  NO (only seed20260711 refs loaded)
S1 / S2 / S3 robustness run:          NO
external / held-out read:             NO
ASZED read:                           NO
lockbox touched:                      NO
substrate rebuilt:                    NO
code changed / candidate space / batch size / MIN_BATCH / stable-CORAL / CAL-EVAL interp altered:  NO
```

## Status

Stage-2B real DEV candidate selection is COMPLETE with outcome **DEV_STOP** (no selected candidate). Per the authorization, this is
a terminal DEV outcome: no tuning, no candidate-space change, no reselection, no rerun without a new authorization. No Stage-4 /
S1-S3 robustness (separate later authorization). The scientific finding: on the real PD/SCZ DEV target, no action policy in the
frozen 22-candidate universe is safe or beneficial — the identity (source-state LDA f_0) baseline dominates, and the pre-registered
harm-control gates (G3/G4) reject every candidate.
