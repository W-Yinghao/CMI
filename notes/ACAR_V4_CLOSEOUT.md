# ACAR v4 — CLOSEOUT

```
substrate commit   : b99fa4fcfb83c6ee60996c50dba6828d40561f26   (B1b all-DEV substrate regeneration / H5)
compatibility commit : 5237378a7c8e4cd5c35aff8efb59701f148575c0  (C5b — keyed-cache replay alignment, subject-namespacing fix)
readiness record   : b539809   (notes/ACAR_V4_C5B_COMPAT_PREFLIGHT_RECORD.md)
result record      : c605e24   (notes/ACAR_V4_C5B_COMPAT_REPLAY_RESULT.md)
compat runtime lock : 523b44bc9480551c96b9f6d50bbadc7fccf027ebf2689d04379324f1dc3b767c
substrate manifest  : 6c6ab457aed9f9925380214d154c407037e002cf52c8954976e98367d1b2e605
authorization       : b385411f6abb41a311dfb88216d90b8c93d84d79c517dbd9e0d114bd7c85c344
result hashes       : compat_RESULT a30ba74f… · compat_manifest 28a889ef… (/home/infres/yinwang/acar_v4_compat_replay_001_5237378)
VERDICT            : SUBSTRATE_COMPATIBILITY_FAIL / NO_EXTERNAL / NO_LOCKBOX_CONSUMED
external Arm B      : NOT RUN (unauthorized — the FAIL forecloses it)
lockbox            : NOT CONSUMED
acar-v4-protocol tag : NOT CREATED (was conditioned on a PASS)
allowed future     : NEW dated protocol only (ACAR v5); NO post-replay tuning of candidate / score family / policy / loss /
                     λ grid / comparator / thresholds; never edit b99fa4f / 5237378 / this result
```

ACAR v4 is **terminated at the substrate-compatibility gate — before external Arm B**. This is NOT an "external validation
failure": the lockbox was never consumed and no held-out/external data was approached. The fixed v4 candidate, selected on the
old seven-DEV leave-one-cohort-out (LOSO) substrate, **does not transfer to the regenerated all-DEV substrate** that an external
run would have to use, and so the external path is foreclosed under the pre-registered no-tuning rule.

## What failed
The single authorized fixed-candidate compatibility replay (`shift_margin + benefit_ranked + harm_indicator`) ran ONCE on the
B1b regenerated all-DEV substrate (new EEGNet encoder + frozen source-state), at detached HEAD==5237378, under the compat
runtime lock + hash-bound authorization (SLURM A40 job 877665, node27, REPLAY_EXIT=0). At the pinned operating point
(α=0.10, budget=0.10, coverage_min=0.15, v2_replay HARD) it failed every gate on BOTH diseases:

| disease | coverage | red | v2_replay_red | L_harm_all_eval | harm_among_adapted | λ cert | v2 eval |
|---------|----------|-----|---------------|-----------------|--------------------|--------|---------|
| PD  | 0.0239 | **−0.00217** | −0.0     | 0.0174 | **0.727** | ✓ | ✓ |
| SCZ | 0.0489 | **−0.01916** | −0.00081 | 0.0489 | **1.000** | ✓ | ✓ |

`reason = coverage<0.15 (both); red≤0 (both); red≤v2_replay (both); disease-macro red ≤ disease-macro v2_replay`. λ was certified
and v2 was evaluable for both diseases, so the HARD v2 comparator was applied and the candidate lost on every criterion. The
candidate barely adapts (≈2–5 % coverage) and is **net-harmful where it does** (negative deployed NLL reduction; conditional
adapted-harm 0.73 PD / 1.00 SCZ). Acceptance: all 14 post-run checks passed (status exactly FAIL; fixed candidate + pinned
pass-line unchanged; v2 evaluable both; commits/manifest/env-lock hashes match; no SELECT/DEV_STOP/external/lockbox/held-out
vocabulary; outputs allow_nan-clean). Details: `notes/ACAR_V4_C5B_COMPAT_REPLAY_RESULT.md`.

## Why it failed — methodological, not a bug
1. **Representation-substrate instability.** The v4 candidate was selected on per-fold LOSO embeddings, but external execution
   needs a single all-DEV encoder (which was never archived and never existed as one object). Re-creating an all-DEV substrate
   changes the geometry of signed scores like `d_margin`, so the candidate's thresholds + action directionality no longer hold.
   The candidate is a local empirical regularity of the old OOF substrate, not a substrate-invariant label-free risk law.
2. **Low-coverage degeneracy of the safety endpoint.** v4 certified `L_harm_all` (all-batch subject-mean harmful risk, with
   non-adapted batches contributing 0). At very low coverage this stays small EVEN IF nearly every adapted batch is harmful —
   exactly what the replay shows (PD adapted-harm 0.73, SCZ 1.00, yet `L_harm_all_eval` ≤ 0.05). Controlling `L_harm_all` alone
   lets a policy "adapt rarely but badly".
3. **Utility-control mismatch.** v4's finite-grid LTT fixed v3's conformal coverage collapse, but it certifies the SAFETY loss,
   not utility. On the new substrate it can certify a low-harm-all threshold whose adaptation set has negative utility. The core
   ACAR problem is still open: the label-free features are MEASURABLE but do not yet translate into stable deployment-benefit control.

## Direction-2 lineage (manuscript Table 1)
| stage | tag / commit | status | one-line |
|-------|--------------|--------|----------|
| A0 / A0′ gate-falsification | (exp/lpc-cmi) | **DIAGNOSTIC_ONLY (closed)** | no source-free harm controller reduces deployed loss; density/CMI wrong-signed; rollback was label leakage |
| ACAR v2 | `acar-v2-protocol @ 9b2f0c1`; result `1528a94` | **MEASUREMENT_ONLY** | label-free action-conditional features predict negative transfer (G1✓); router not deployable (G2✗) |
| ACAR v3 (HSCR) | `acar-v3-dev-design-v1 @ 817b04f`; result `9f4e83f` | **DEV_STOP / NO_LOCKBOX_CONSUMED** | stricter redesign fails the DEV S2/S4 gate (all-action conformal coverage collapse ~0.6–1.1 %; weak PD center) |
| ACAR v4 (CURB) | DEV `e9760e6`; substrate `b99fa4f`; compat `5237378`; result `c605e24` | **SUBSTRATE_COMPATIBILITY_FAIL / NO_EXTERNAL / NO_LOCKBOX_CONSUMED** | direct executed-policy LTT found a DEV-only candidate (14/90, non-binding), but it does NOT transfer to the regenerated all-DEV (external-compatible) substrate — coverage ≪0.15, deployed red<0 both diseases, fails v2-HARD |

## Reader's note (avoid the mid-state trap)
The repo contains earlier v4 artifacts at intermediate states — `notes/ACAR_V4_DEV_EXPLORATION_001_RESULT.md`
("V4_DEV_CANDIDATE_FOUND_FOR_POSSIBLE_FREEZE", **NON-BINDING / selection over 90 configs**), `notes/ACAR_FROZEN_v4.md`
(**DRAFT, never tagged**), and the encoder/reader blocker notes. **This closeout is the authoritative final v4 status.** The
DEV #001 candidate is exploratory-only and is NOT confirmatory evidence; it must be read as a NEGATIVE PRIOR for any successor.

## Consequence / what is allowed next
Per the pre-registered FAIL rule: **no post-replay tuning** of the candidate / score family / policy / loss / λ grid /
comparator / thresholds. ACAR_FROZEN_v4 substrate hashes are NOT promoted; the `acar-v4-protocol` tag is NOT created; external
Arm B stays unauthorized; the lockbox stays SEALED. v2 (`MEASUREMENT_ONLY`) and v3 (`DEV_STOP`) are untouched. Any continuation
is a **NEW dated, separately-pre-registered protocol (ACAR v5)** — a substrate-robust, constrained-utility router whose protocol
puts substrate robustness and a conditional-adapted-harm gate at the center (the two failure modes above), NOT a re-tuning of
this gate.
