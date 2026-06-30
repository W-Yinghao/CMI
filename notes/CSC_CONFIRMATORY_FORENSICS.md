# CSC confirmatory failure forensics (B-P0; analysis only, no new runs)

Forensics of the frozen confirmatory FAIL (`csc-confirmatory-v1`/`dee8958`, SLURM 876329) from the
**committed** artifact `csc/results/confirmatory.json`. No reruns, no tag/protocol/threshold/seed change.

## What the artifact supports

**Power collapse (visible-concept / `boundary_coupled` target, 66 clusters):** fired **28/66 (42%)**;
the 38 non-fires decompose as:

| binding reason | count | share | bottleneck |
|---|---|---|---|
| `geometric_maxstat_not_sig` | 15 | 23% | **source evidence**: no estimable concept geometry on the unseen source |
| `not_dominant_or_robust_consensus_abstain` | 15 | 23% | **certifier**: evidence present but dominance/consensus gate abstained |
| `residual_T_not_sig` | 7 | 11% | **source evidence**: cross-fitted decoder gate not significant |
| `unstable_concept_attribution` | 1 | 2% | source attribution unstable (also the 1 source-invalid cluster) |

So the power loss is **two roughly-equal bottlenecks**: ≈ **36%** of clusters the *source side* failed
to produce concept evidence at all (geometry + decoder + attribution), and ≈ **23%** had evidence but the
*certifier* refused to commit (dominance/robust-consensus). It is **not** a single fixable gate — both
the evidence extractor (mean/low-moment atlas + full-interaction residual decoder + geometric max-stat)
**and** the conservative certifier are fragile on unseen sources.

**False-certification:** `forbidden = 1/65` → CP-UB 0.0709 > α. **One source-invalid** cluster
(`UNSTABLE_CONCEPT_ATTRIBUTION`), correctly excluded (frac 0.0152 ≤ 0.10).

## What the artifact does NOT contain (forensic gap)

The committed run logged **aggregates only** (per-point counts + the decomposition). It does **not**
record, per cluster: the forbidden cluster's **kind/state** (false `CONCEPT_SUSPECT` on a stable target
vs false `COVARIATE_COMPATIBLE` on a must-abstain shift), the components `n_cov/n_concept/n_label/n_resid`,
the source `residual-p / p_global / attribution status`, or the realized `tau_detect/tau_label`. Answering
those (e.g. *which* failure mode the single forbidden event was) requires per-cluster logging — which is a
**B-version runner enhancement** (log per-cluster from the start) or a separately-authorized deterministic
reproduction of the frozen tag (same seeds → identical aggregates, extra detail). Per the B-P0 rule
("analyze existing, no new runs"), it is **not** done here.

## Implication for Route B (method change, not a tag tweak)

The decomposition argues **against** a pure Z-only B1 as the primary bet and **for** minimal-information
B2/B3: both observed bottlenecks are intrinsic to *source-only marginal* evidence —
- the ≈36% source-evidence failures reflect that a mean/low-moment atlas + full-interaction decoder is a
  weak, high-variance concept-evidence extractor on unseen subjects (subject random-effect confound), and
- the ≈23% certifier abstentions show even when evidence exists it is too marginal to clear a
  non-vacuous dominance/consensus bar without inflating the forbidden risk (the control↔power tradeoff
  the FAIL sits on).

A better Z-only extractor (B1: score-space / low-rank boundary) might lift the first bucket, but still
fights the same tradeoff on the second and cannot cross pure-conditional unidentifiability. **Minimal
extra information** (B2 few target labels; B3 paired within-subject ON/OFF differencing, which cancels the
subject random effect) attacks the root cause directly and is the higher-probability path to a confirmable
positive result. Any B prototype starts a **new** freeze→audit→unseen-confirmatory cycle under a **new**
tag — it does not reuse `csc-confirmatory-v1`.
