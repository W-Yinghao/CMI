# CMI Synthesis — audit durable, control closed, TTA a separate non-CMI positive

```
Project-level synthesis after the active-CMI probe (CITA λ=1.0) closed the last open CMI-control question.
Spans BOTH information regimes: source-only (CIGL_70) and target-unlabeled (CITA_02/03). This closes the CMI
CONTROL research line; it does not close the CMI project's durable contributions (audit) nor the newly-found
non-CMI positive (target-unlabeled adaptation). Evidence: results/cmi_synthesis/synthesis_route_summary.csv.
```

## One-paragraph result
Conditional mutual information (label-conditional domain leakage, `I(Z;D|Y)`) is a **robust, verifiable audit
signal** for EEG decoders — measurable, permutation-significant, and consumable by a classifier-level
head-replay reliance test with a valid random-subspace control. But **no CMI *control* objective we built
converts that measurement into functional control.** Across four source-only routes (global-leakage CIGL,
task-head-alignment FCIGL, direct-counterfactual dCIGL, source-episodic MetaCMI) and the target-unlabeled CITA
route at both an inactive (λ=0.010) and an **active** (λ=1.0, 17.8–29.3% of the loss) setting, the CMI term never
reduced functional reliance or improved held-out-subject generalization beyond the appropriate non-CMI baseline.
Separately, **target-unlabeled adaptation itself (TTA-Control) robustly improves target accuracy** (+0.035…+0.093
across all four dataset×backbone cells, no collapse) — the project's first robust generalization gain, but a
non-CMI result.

## The three claims (frozen)
```
CMI audit works.
CMI proxy control works partially (measured leakage is reducible without task collapse).
CMI control does NOT deliver functional reliance control or generalization gains — in source-only OR target-unlabeled.
Target-unlabeled adaptation (TTA-Control) improves target performance — a separate, non-CMI positive.
```

## Evidence map (see route_summary.csv for the full table)
### Durable positive — CMI audit
- Label-conditional domain leakage is significant on real EEG (permutation p<0.05) on the graph backbone
  (CIGL_65) and both feature_z Minis (CIGL_69A/C, CITA); leakage-rich especially on the transformer.
- Reliance is measured **classifier-level** (exact head-replay, max|Δ|≤~9e-6) with a **valid random-subspace
  control** (≈0 throughout). This audit machinery is the durable methodological contribution.

### Partial — CMI proxy control
- CIGL reduces measured graph/node leakage ~40–65% on two datasets without task collapse (CIGL_65/69). MetaCMI /
  CITA reduce the feature_z leakage proxy a little more than their non-CMI baselines. So the proxy is *reducible*
  — the failure is specifically at the proxy→function step.

### Closed negative — CMI control (both regimes)
| regime | route | CMI active? | CITA/CMI vs correct baseline | verdict |
|---|---|---|---|---|
| source-only | CIGL / FCIGL / dCIGL | yes | R3 reliance not reduced (or seed0-only) | negative |
| source-only | MetaCMI (episodic) | yes | ≤ MetaCE on target; R3 within noise | FAIL |
| target-unlabeled | CITA λ=0.010 | **no** (near-inert) | ≈ TTA on target & R3 | TTA-only pass |
| target-unlabeled | CITA λ=1.0 | **yes** (17.8–29.3%) | ≈ TTA on target & R3; slightly worse where most active | **fair failure** |

The λ=1.0 active probe is decisive: it removes the "the term was never active" objection — the CMI-control
objective adds nothing **even when it dominates a third of the adaptation loss.**

### Separate positive — target-unlabeled adaptation (non-CMI)
TTA-Control (`CE(source_replay)+τ·H(p_target)+μ·KL(mean p_target‖source_prior)`) improves target_bacc in all four
cells (2a +0.037…+0.043; 2015 +0.075…+0.093), trading a little source accuracy, with no entropy/label collapse
(entropy non-degenerate, balance-KL small). This is the first robust generalization gain in the project and is
**not** a CMI effect (CITA−TTA ≈ 0 at both λ). It should be reported and (if pursued) built on as a
target-unlabeled adaptation result, never as evidence that CMI control works.

## The unifying mechanism — measurement→control gap
A CMI proxy/objective can be measured and even reduced, but the *functional nuisance structure* it targets is not
controllable from the available information: source subjects do not carry enough to control the held-out
subject's reliance (source-only), and even unlabeled target X + an active conditional-domain-confusion objective
does not (target-unlabeled) — while a plain confidence/balance adaptation on the same target X does help accuracy.
The gap is **CMI-control-specific**, holds across graph and feature_z backbones, and across both information
regimes.

## Frozen prohibitions (CMI control line closed)
No further λ/β/η/α/k sweeps; no new CITA objective; no new backbone for CMI control; no source-only revival
(static-DGCNN CIGL/FCIGL/dCIGL, source-only MetaCMI); no ConformerFull CMI arm; no external-dataset CMI method
search. These are scientifically closed.

## What remains open (non-CMI, only if the PM directs)
Target-unlabeled adaptation (TTA-Control) is a genuine positive — a possible standalone line (stronger
adaptation objectives, more datasets, seeds 0/1/2), explicitly framed as **non-CMI**. Any write-up must keep the
boundary: *"target-unlabeled adaptation improves target performance; CMI control does not add to it."*

## Pointers
- Source-only closure: `docs/CIGL_70_SOURCE_ONLY_CMI_CLOSURE.md`, `results/cigl_source_only_closure/`
- Target-unlabeled: `docs/CITA_01_TARGET_UNLABELED_PROTOCOL.md`, `docs/CITA_02_SEED0_READOUT.md`,
  `docs/CITA_03_ACTIVE_LAMBDA_READOUT.md`, `results/cita/gate/` (λ=0.010), `results/cita/gate_lambda1/` (λ=1.0)
- Audit machinery: `cmi/eval/{graph_leakage,head_export,leakage_removal,audit_npz}.py`
- This synthesis: `results/cmi_synthesis/{synthesis_route_summary.csv, MANIFEST.yaml}`
```
