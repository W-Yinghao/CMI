# V2 Stage-2 (scoped) verdict --- PASS

Job array 885746 (5 shards world x backbone; C/EEGNet + C/TSMNet finished on cpu-high), merge 885747.
Config b8e24e34fc84; 72,000 tasks, 0 fail, 0 degenerate; all 12 dataset/backbone/seed cells VALID 15/15.
All 5 shards: TARGET_LEAK_STRUCTURAL_PASS. Global: STAGE2_MERGE_CLEAN. Merge integrity verified
(4800 cells = 960 x 5 shards, 0 duplicate keys, disjoint union; independent recompute of stop-conditions).

## World A / EEGNet clean ceiling --- robust across source_subject_count
| n_source | clean positives | principled ACCEPT | oracle_supported | oracle max ΔbAcc | random max | best deployable |
|---|---|---|---|---|---|---|
| 8   | 51 | 0 | yes | +0.060 | +0.007 | leace +0.100 [+0.074,+0.131] |
| 16  | 67 | 0 | yes | +0.056 | +0.005 | rlace +0.089 [+0.070,+0.108] |
| 32  | 65 | 0 | yes | +0.066 | +0.013(mean) | rlace +0.113 [+0.082,+0.145] |
| all | 53 | 0 | yes | +0.065 | +0.003 | rlace +0.108 [+0.083,+0.134] |
Clean positives exist at EVERY n_source (none disappear); oracle-supported at every n_source; 0 principled
ACCEPT; random-k never reproduces (max random_k dtgt LCB = +0.0057 < +0.01). => CLEAN PASS, not MIXED.

## World B / C refusal --- robust across backbone x source_subject_count
* World B: 0 principled ACCEPT (EEGNet + TSMNet). Task-entangled erasers rejected: unsafe (task-drop UCB>0.02)
  = inlp/leace/rlace (96/96/89 EEGNet ; 96/96/58 TSMNet). Action dist ABSTAIN 391/422, REJECT 281/250.
* World C: 0 principled ACCEPT (both backbones); 530 (EEGNet) / 546 (TSMNet) high-domain-gain-useless cells.

## Naive controllers (all deployable cells) --- our gate 0 false accepts
domain-gain-only 2089 acc / 1807 FALSE ; safety-only 3604 / 3368 FALSE ; always-domain-gain 2324 / 2040 FALSE ;
OUR GATE 0 / 0 / 0 ; ORACLE target-informed selector 284 true / 0 false (diagnostic; uses target labels).

## Verdict
Stage 2 confirms, across source_subject_counts {8,16,32,all} and seeds {0,1,2}:
* the EEGNet source-only acceptance CEILING is robust (clean target-beneficial-but-source-uncertifiable cells
  at every n_source; gate never accepts; oracle-supported; random does not reproduce);
* B/C refusal / no-false-accept is robust on BOTH backbones;
* our source-only gate has 0 false accepts across 72,000 cells while naive source-only rules false-accept 1807-3368.
Combined with Stage 1B (TSMNet World A not cleanly demonstrable under the appended-nuisance construction), the
honest final V2 result stands: EEGNet clean ceiling; TSMNet refusal + no-false-accept; source-only acceptance is
provably conservative and cannot certify deployment-shift benefit.
