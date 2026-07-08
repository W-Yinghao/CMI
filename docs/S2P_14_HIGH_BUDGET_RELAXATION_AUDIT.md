# S2P_14 - High-Budget Feasibility Relaxation Audit

**Phase:** S2P 9D.

**Status:** metadata-only relaxation audit plus Route-B 33ch CBraMod smoke complete. No training, downstream audit,
fine-tuning, CodeBrain science run, or auto-launch chain was submitted.

## Frozen 9C v2 conclusion

9C v2 high-budget floor calibration is a **NO-GO** under the strict launch contract:

```text
corpus = TUEG 19-common
allocation = exact high-coverage
min_exposure_per_subject = 0.25h
no oversampling / no window reuse
subject-disjoint pretrain-val
H_high must be >= 2000h
```

Feasibility supports 200/500/1000h, but no exact high-coverage endpoint >=2000h. Therefore 9C-1 training was not
launched.

Allowed claim:

```text
The current processed 19-common corpus is not sufficient for a strict 2000h+ exact high-coverage calibration under
the 9C v2 allocation contract.
```

Forbidden claims:

```text
CBraMod cannot scale.
2000h training would fail.
The full TUEG corpus lacks enough data.
Subject scaling is infeasible in general.
```

## 9D question

> What minimal protocol relaxation would make a >=2000h high-budget calibration feasible, and what scientific claim
> would that relaxation license?

The main relaxation audit reads TUEG metadata/window counts. The Route-B smoke additionally runs a synthetic
`B x 33 x 30 x 200` tensor through native CBraMod. No target labels are read or used.

## CBraMod channel-flexibility update

Local source audit confirms that CBraMod's backbone reads `ch_num` and `patch_num` at runtime in `forward`, native
`generate_mask` creates `(B, C, P)`, and criss-cross attention reshapes using runtime `ch_num`. This makes a
33-channel CBraMod-only route technically plausible.

This does **not** mean CBraMod is channel-invariant. A 33-channel result depends on channel order, reference scheme,
and montage grouping. It must be reported as:

```text
CBraMod 33-channel full-corpus high-budget calibration
```

not as:

```text
CodeBrain-compatible 19-common calibration
channel-invariant CBraMod evidence
```

## Routes audited

| route | relaxation | feasible metadata/compute budgets | claim allowed | claim forbidden |
|---|---|---:|---|---|
| A | keep 19-common, relax exact equal windows to bounded imbalance | 500, 1000, 2000 | bounded-imbalance 19-common high-coverage calibration | exact high-coverage claim; pure diversity/depth claim |
| B | switch to fixed 33ch CBraMod-only substrate | 500, 1000, 2000 | CBraMod 33-channel full-corpus budget calibration | direct CodeBrain 19-common comparison; patching 19-common curve |
| C | keep 19-common, switch from high-coverage to data-volume sampling | 500, 1000, 2000, 3000 | 19-common total data-volume scaling test of whether 200h was too short | subject-coverage causal claim; exact high-coverage claim |

The updated audit recommendation is **Route B**, because CBraMod's adaptive-channel architecture makes a fixed 33ch
CBraMod-only high-budget substrate technically plausible, and the 33ch smoke passes. Training still requires explicit
PM approval.

## Route notes

### Route A - bounded imbalance within 19-common

Route A keeps maximum subject coverage and allows bounded contribution imbalance:

```text
min contribution = 30 windows (0.25h)
max/min windows ratio <= 2
subject_contribution_gini <= 0.05 or 0.10
```

Result:

```text
500h: feasible, N=2000, contribution_gini=0.0
1000h: feasible, N=4000, contribution_gini=0.0
2000h: feasible, N=6343, contribution_gini=0.003766
3000h: infeasible under max-coverage N=6343, min_w=30, max/min<=2
4000h: infeasible and exceeds 96h compute estimate
```

This is scientifically usable only as bounded-imbalance high-coverage, not exact high-coverage.

### Route B - fixed 33ch CBraMod-only substrate

Route B uses the fixed `n_channels == 33` processed TUEG subset, not the mixed variable-channel full metadata table.
The route is CBraMod-only. It is not CodeBrain-compatible and not a 19-common curve.

Metadata result:

```text
n_recordings = 8279
n_subjects = 2641
total_usable_hours = 4168.35h
train_hours_after_val = 4021.525h
train_windows_after_val = 482583
eligible_subjects_min_exposure_after_val = 2470
unique_channel_orders = 6
reference_scheme_counts = LE:3745, REF:4534
```

Feasibility result:

```text
data-volume exact-window feasibility:
  500 / 1000 / 2000 / 4000h all feasible without window reuse

metadata + current 96h compute cap:
  500 / 1000 / 2000h feasible
  4000h data-feasible but compute estimate = 179.76h > 96h

exact high-coverage feasibility:
  only 500h feasible
```

The 33ch substrate is not channel-order uniform. Any Route-B training design must pin a channel-order grouping or
canonicalization rule before launch.

33ch CBraMod smoke result:

```text
input_shape = [1, 33, 30, 200]
mask_shape = [1, 33, 30]
output_shape = [1, 33, 30, 200]
loss finite = true
gradients finite = true
checkpoint save/reload = true
feature dump shape = [1, 200]
summary/ptflops 19ch reporting hardcode detected and bypassed = true
smoke_passed = true
```

This licenses only a CBraMod 33-channel technical route. It does not license channel-invariance or direct comparison
to CodeBrain's 19-common scaling curve.

### Route C - 19-common data-volume scaling

Route C keeps the 19-common corpus but drops high-coverage subject allocation. It asks only whether the no-reuse
19-common train pool after fixed val exclusion can support a target total data volume.

Result:

```text
available train windows after val exclusion = 403165 (~3359.7h)
1000h: feasible
2000h: feasible
3000h: feasible
4000h: infeasible without reuse, and exceeds 96h compute estimate
```

This route best matches the published data-volume scaling language, but it does not license a subject-coverage claim.

## Artifacts

```text
results/s2p_budget_floor_calibration_v2/relaxation_audit/relaxation_route_feasibility.csv
results/s2p_budget_floor_calibration_v2/relaxation_audit/relaxation_claim_matrix.csv
results/s2p_budget_floor_calibration_v2/relaxation_audit/relaxation_population_shift_diagnostics.csv
results/s2p_budget_floor_calibration_v2/relaxation_audit/relaxation_recommendation.json
results/s2p_budget_floor_calibration_v2/relaxation_audit/route_b_33ch_feasibility.csv
results/s2p_budget_floor_calibration_v2/relaxation_audit/route_b_33ch_channel_order_diagnostics.csv
results/s2p_budget_floor_calibration_v2/relaxation_audit/route_b_33ch_population_diagnostics.json
results/s2p_budget_floor_calibration_v2/relaxation_audit/route_b_33ch_smoke.json
results/s2p_budget_floor_calibration_v2/relaxation_audit/route_b_33ch_feasibility_smoke_summary.json
```

`relaxation_recommendation.json` records:

```json
{
  "strict_19common_exact_highcoverage_feasible_ge2000": false,
  "route_A_bounded_imbalance_feasible": true,
  "route_B_33ch_full_corpus_feasible": true,
  "route_B_33ch_smoke_passed": true,
  "route_C_19common_data_volume_feasible": true,
  "recommended_next_training_design": "B",
  "training_requires_pm_approval": true,
  "training_launched": false,
  "target_labels_used": false
}
```

## Current launch policy

```text
9C-1 training:
  not approved.

9D:
  accepted as metadata feasibility plus 33ch smoke preparation.

Next training:
  not approved.
  Route B is the recommended candidate design for PM review.
```
