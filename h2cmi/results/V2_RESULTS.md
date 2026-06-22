# V2 real-EEG external validation — results (frozen run; 702 rows; raw v2_all.jsonl sha256=8be5437334b1eff0)

Offline datalake, binary L/R, 21-channel sensorimotor grid (sha 9bac2f43), target labels eval-only.
Separate A and B verdicts (NO single headline mean), per the pre-registration.

## A — out-of-support abstention audit (operator-support safety). N=18 target subjects, 6 cross-dataset pairs
metadata_only: unsupported-route adaptations = 0/18 (rate 0.000, binomial upper95 = 0.153), EXACT
prediction-equivalence with identity = 1.000, all geometry = UNSUPPORTED. mean Δ +0.000, harm 0.00.
Comparators (which adapt blind where geometry is UNSUPPORTED):
    always_pooled       mean Δ +0.032  harm 0.11  worst-quartile -0.014
    always_canonical_CC mean Δ +0.031  harm 0.11  worst-quartile -0.014
    euclidean_alignment mean Δ +0.029  harm 0.33  worst-quartile -0.049
READ: the metadata rule holds identity everywhere on cross-dataset acquisition mismatch (guaranteed
0 harm). Blind adaptation has a POSITIVE mean here (these real cross-dataset pairs are less adversarial
than the simulator's UNSUPPORTED regime) BUT carries 11-33% subject-level harm with negative tails
(EA worst, 33%). Abstention buys tail safety at the cost of a modest forgone average gain -- reported
honestly, not as free safety.
A-severe (descriptive, BNCI2014_001-LR -> BNCI2014_004 C3/Cz/C4): 0/9 adaptations (identity).

## B — supported-regime utility test. N=90 subject-sessions, dataset-stratified
metadata_only (= pooled; DIAG_COMPATIBLE x SAME; coverage 1.00):
    mean paired Δ(metadata_only - identity) = +0.001  CI95 [-0.007,+0.009] (INCLUDES 0)
    harm rate 0.37 (> 0.20 target)  worst-quartile -0.048
    per dataset: BNCI2014_001 +0.012 | BNCI2014_004 +0.005 | Lee2019_MI -0.002
Comparators: always_pooled +0.001 | always_canonical_CC +0.002 | euclidean_alignment +0.002 (harm 0.44)
    current_joint -0.005 (harm 0.49, worst-quartile -0.069)  <-- DIAGNOSTIC
B verdict (4 preregistered criteria): mean>0 TRUE; CI-excludes-0 FALSE; harm<=0.20 FALSE; coverage~1 TRUE.
READ: even in the supported regime the frozen diagonal operator yields NO reliable utility on real
cross-session MI (+0.001, CI includes 0, harm 0.37). Utility is NOT established. Dataset-heterogeneous.

## Cross-cutting (mechanism replication on real EEG)
current_joint HARMS on real EEG in BOTH A-context and B (B: -0.005, harm 0.49) -- the V1 prior-M-step
finding (geometry-only/fixed-prior beats the joint) REPLICATES on real data. The deployable
architecture's two halves land differently on real MI: the SAFETY half (metadata abstains, no harm;
joint's harm avoided) holds; the UTILITY half (positive gain when supported) does NOT reach the
preregistered bar. Consistent with the program's measurement->control gap: source-calibrated /
geometry-only adaptation is safe but its real-EEG utility is marginal.

## AUDIT CORRECTION (post-hoc; supersedes the numbers above)
Two bugs found in the public audit, both in MERGE/ANALYSIS only -- the raw per-pair JSONL was complete
on disk:
(1) V2-A merge: the orchestration monitor's `squeue -n` job names omitted underscores, so the merge
    ran before the 4 slow pairs (Cho/Lee targets, 52-54 subjects each) wrote. The first report had only
    2 pairs / 18 routes. CORRECTED full merge = all 6 directed pairs, 230 subject-routes.
(2) V2-B CI: bootstrapped 90 rows directly; now SUBJECT-CLUSTERED (BNCI2014_004 = 3 cross-session
    units/subject; 72 subject clusters).
CORRECTED A (N=230, 6 pairs): metadata_only 0/230 unsupported-route adaptations (binomial upper95 =
0.013, MUCH tighter), exact identity-equivalence 1.000. Comparators: always_pooled +0.097 (harm 0.22,
wq -0.057); always_canonical_CC +0.070 (harm 0.21, wq -0.046); EA +0.070 (harm 0.27, wq -0.090).
-> operator-support safety STRONGER (230/230, upper95 0.013); blind adaptation has a larger positive
mean but ~21-27% subject-level harm with negative tails.
CORRECTED B (subject-clustered, 72 clusters): metadata_only +0.001, harm 0.37, clustered CI95
[-0.007,+0.009] (includes 0) -> utility STILL NOT established (DESCRIPTIVE interval). current_joint
-0.005, harm 0.49, clustered CI [-0.015,+0.006] -> "real EEG confirms prior-M-step harm" is TIGHTENED
to: in V2-B the joint is directionally consistent (negative) and has the HIGHEST harm rate, but its
clustered CI includes 0.
