# V2 full-lite (Stage 1) --- audit + World-A taxonomy reclassification

The full-lite verdict is **not** "World A passes across both backbones". Introducing a stricter World-A
taxonomy (below) shows the honest picture: EEGNet demonstrates the source-only acceptance ceiling cleanly;
TSMNet confirms B/C + no-false-accept but its World A is MIXED because the injected-nuisance oracle is not
reliably target-beneficial.

## World-A taxonomy (now enforced in report_v2.py::_worldA_taxonomy)
Per (dataset, backbone):
* **target_beneficial** cell = deployable principled eraser that is SAFE (source task-drop UCB <= 0.02) AND has
  actual target dbAcc LCB > +0.01.
* **oracle_supported** = the injected-nuisance ORACLE eraser (ground-truth removal of exactly the injected
  block) is itself target-beneficial (LCB > +0.01) AND random_k does NOT reproduce it.
* **clean_worldA_positive** = target_beneficial AND oracle_supported (the ceiling is cleanly tied to the
  injected deployment-shift nuisance, not to an eraser removing broader structure).
* status = CLEAN (>=1 clean positive, 0 accept) / MIXED (target-beneficial exists but oracle-unsupported) / NONE.

World-A PASS now requires `clean_worldA_positive` (not merely "safe target-beneficial cells exist").

## Full-lite reclassification (job 883340, commit 1383635, config 6b4d2622bdde)
| dataset | backbone | target_beneficial cells | oracle best ΔbAcc | oracle_supported | clean positives | ACCEPT | carriers | status |
|---|---|---|---|---|---|---|---|---|
| Lee2019_MI | EEGNet | 24 | +0.063 | Y | 24 | 0 | fair/leace/rlace | **CLEAN** |
| Cho2017 | EEGNet | 29 | +0.065 | Y | 29 | 0 | fair/leace/rlace | **CLEAN** |
| Lee2019_MI | TSMNet | 7 | +0.009 | n | 0 | 0 | rlace only | **MIXED** |
| Cho2017 | TSMNet | 6 | +0.006 | n | 0 | 0 | rlace only | **MIXED** |

* World B: 0 unsafe accept (leace/inlp/rlace REJECTed); action dist ABSTAIN 206 / REJECT 130. **PASS.**
* World C: 0 accept; 270 high-domain-gain-useless cells. **PASS.**
* Naive controllers: domain-gain-only 556 false-accepts, safety-only 1009, always-domain-gain 621; OUR GATE
  0/0/0; ORACLE target-informed selector 105 true / 0 false (diagnostic). **Our gate 0 false accept.**

## Formal status of the full-lite (use this wording; do NOT overclaim)
```
Stage 1 full-lite result:
  B/C pass across EEGNet and TSMNet.
  EEGNet World A cleanly demonstrates the source-only acceptance ceiling.
  TSMNet World A is MIXED: target-beneficial RLACE cells exist and are NOT accepted (source-LOSO benefit
  LCB negative), but the injected-nuisance oracle is not reliably target-beneficial (best +0.006..+0.009,
  LCB < +0.01), so the TSMNet World A ceiling is not yet cleanly established.
```
Forbidden: "World A passes across both backbones" / "V2 full-lite fully confirms the ceiling across capacity".
Correct headline: "V2 full-lite confirms the safety/refusal behavior and a clean EEGNet ceiling; TSMNet World A
requires robustness because oracle support is marginal."

## Why TSMNet is MIXED (mechanism)
On TSMNet the oracle (removing exactly the injected m=42 block) barely moves the target (+0.006..+0.009), while
RLACE moves it more (+0.03..+0.04). So RLACE's TSMNet target gain is NOT injected-nuisance-specific -- it likely
removes broader z-correlated structure. The injected deployment-shift nuisance at fraction 0.20 is too weak a
target-harm on a 210-d latent to be the clean carrier of the ceiling.

## Stage 1B result (TSMNet World-A robustness probe) --- NEGATIVE
Sweep of world-gen `nuisance_fraction in {0.15, 0.20, 0.25, 0.30}` (TSMNet, Lee+Cho, seed0, first-5, World A
only, all interventions, alpha grid, gate thresholds FROZEN; job 885017). Full grid (no cherry-pick):

| fraction | m (TSMNet) | oracle best ΔbAcc [CI] | oracle_supported | clean positives | target_beneficial | ACCEPT | random_k |
|---|---|---|---|---|---|---|---|
| 0.15 | 32 | +0.008 [+0.001,+0.015] | no | 0 | 6 | 0 | +0.002 |
| 0.20 | 42 | +0.009 [+0.001,+0.018] | no | 0 | 5 | 0 | +0.003 |
| 0.25 | 52 | +0.009 [+0.003,+0.016] | no | 0 | 3 | 0 | +0.001 |
| 0.30 | 63 | +0.008 [+0.001,+0.017] | no | 0 | 4 | 0 | +0.002 |

**NO fraction makes the TSMNet oracle target-beneficial** (best +0.009, LCB never > +0.01), and the oracle gain
does NOT increase with m (flat ~+0.008..+0.009, even dips at 0.30). This is NOT a fraction-tuning problem.

**Mechanism:** the nuisance is APPENDED as an orthogonal block, and on a 210-d high-quality latent the real-Z
task signal dominates the head; the injected deployment-shift nuisance -- at any block size 15-30% of the
latent -- is too weak a target-harm for its removal (oracle) to be clearly beneficial. Adding more nuisance
dimensions of the same weak subject-gated shortcut does not increase the head's reliance proportionally.

**Conclusion (per the pre-registered fallback):** TSMNet World A is **not cleanly demonstrable** under the
appended-nuisance construction via fraction tuning. We keep the honest asymmetric result:
```
EEGNet: source-only acceptance ceiling cleanly demonstrated (oracle-supported, multi-method).
TSMNet: B/C + no-false-accept confirmed; World A target-beneficial RLACE cells exist and are NOT accepted,
        but the injected-nuisance oracle is not target-beneficial at any fraction -> World A ceiling NOT
        cleanly established for high-dimensional latents under this construction.
```
Per the ban on tuning after a negative fraction probe: NO alpha expansion, NO threshold changes, NO f_align
tuning without explicit approval. Decision (redesign World A vs accept-as-not-demonstrable vs Stage-2-for-
EEGNet/B/C-only) deferred to the PM.
