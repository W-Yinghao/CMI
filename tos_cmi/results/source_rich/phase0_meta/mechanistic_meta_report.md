# Fork 2 Phase 0 --- mechanistic meta-analysis (REAL EEG primary; E0 subject environment)

Joined 20 real-EEG cells (Track B source-only gate x erasure_report x deployment). Target = audited outcome only. Datasets: ['Cho2017', 'Lee2019_MI'] ; backbones: ['EEGNet', 'TSMNet'].

## Source-only signal vs actual target ΔbAcc (Pearson / Spearman / dataset-cluster 95%% CI)
- 1 Δsubject LINEAR decode vs target ΔbAcc             : r +0.042 | rho -0.166 | 95% CI [+0.026,+0.051]
- 2 Δsubject MLP decode vs target ΔbAcc                : r -0.175 | rho -0.222 | 95% CI [-0.238,-0.140]
- 3 domain-gain vs target ΔbAcc                        : r -0.137 | rho -0.413 | 95% CI [-0.151,-0.133]
- 4 source task-drop UCB vs target ΔbAcc               : r -0.886 | rho -0.769 | 95% CI [-0.917,-0.857]
- 5 source-LOSO predicted ΔbAcc (mean) vs target ΔbAcc : r +0.867 | rho +0.743 | 95% CI [+0.838,+0.905]
- 7 source-LOSO benefit LCB vs target ΔbAcc            : r +0.871 | rho +0.756 | 95% CI [+0.844,+0.907]

## 6 source-LOSO sign agreement with target: 18/20 (90%)

## 8 random-k control: mean target ΔbAcc -0.004 (vs principled -0.080)

## 9 per-dataset breakdown
- Cho2017          n=10 : source-LOSO-vs-target r +0.838 ; target ΔbAcc mean -0.056 [-0.150,-0.000]
- Lee2019_MI       n=10 : source-LOSO-vs-target r +0.905 ; target ΔbAcc mean -0.072 [-0.185,+0.000]

## 10 per-backbone breakdown
- EEGNet           n=10 : source-LOSO-vs-target r +0.990 ; target ΔbAcc mean -0.102 [-0.185,-0.001]
- TSMNet           n=10 : source-LOSO-vs-target r +0.955 ; target ΔbAcc mean -0.027 [-0.154,+0.000]

## 11 per-method breakdown
- INLP             n= 4 : source-LOSO-vs-target r -0.627 ; target ΔbAcc mean -0.149 [-0.185,-0.105]
- LEACE            n= 4 : source-LOSO-vs-target r +0.994 ; target ΔbAcc mean -0.085 [-0.185,-0.001]
- RLACE            n= 4 : source-LOSO-vs-target r +0.985 ; target ΔbAcc mean -0.085 [-0.185,-0.000]
- TOS_VD           n= 4 : source-LOSO-vs-target r -0.645 ; target ΔbAcc mean -0.000 [-0.001,-0.000]
- random_k         n= 4 : source-LOSO-vs-target r -0.822 ; target ΔbAcc mean -0.004 [-0.007,+0.000]

## 12 gate-action vs actual target outcome
- %-8s : (none)
- REJECT   n=16 : target ΔbAcc mean -0.080 ; harmful cells 8 ; beneficial cells 0
- ABSTAIN  n= 4 : target ΔbAcc mean -0.000 ; harmful cells 0 ; beneficial cells 0

## V2 semi-synthetic panel (SEPARATE; mechanism control, NOT mixed with real EEG)
- V2 stage2 deployable cells: source-LOSO benefit LCB vs target ΔbAcc Pearson r +0.983 (n=4320) -- semi-synthetic; injected nuisance; kept apart from the real-EEG conclusion.

## Reading (E0 baseline)
- If the source-LOSO<->target correlation (#5/#7) is ~0 / CI spans 0 under E0 subject environments, the E0 baseline does NOT make benefit source-visible -> E1-E5 must materially improve source-target alignment for Fork 2 to have a positive route.
- If leakage reduction (#1/#2) is uncorrelated with target ΔbAcc, this supports 'leakage removal is not a benefit certificate' (do NOT overstate as 'leakage can never matter').
- Any target-positive subgroup is a CANDIDATE only -> must be reproduced under the pre-registered dev/confirm split with random-k / task-drop / leakage audits before any claim.
