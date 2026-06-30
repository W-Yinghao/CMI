# ACAR v4 — C5b fixed-candidate DEV substrate-compatibility REPLAY result **= `SUBSTRATE_COMPATIBILITY_FAIL`** (recorded; NO tag / NO external Arm-B)

```
RESULT : The single authorized fixed-candidate (shift_margin + benefit_ranked + harm_indicator) old-seven-DEV
         substrate-compatibility replay RAN ONCE under the regenerated all-DEV B1b substrate (new EEGNet encoder + frozen
         source-state), at detached HEAD==5237378 (C5b), under the compat runtime lock + the hash-bound authorization.
         VERDICT = SUBSTRATE_COMPATIBILITY_FAIL. The candidate does NOT transfer to the regenerated substrate: it barely
         adapts (coverage well below 0.15) and is net-harmful where it does (red < 0 on BOTH diseases), and it does NOT beat
         the v2-replay comparator (v2 was evaluable for both). This is a FAIL of the substrate-compatibility check — NOT a new
         DEV selection, NOT external Arm-B, NOT a lockbox run. Per the pre-registered FAIL rule: V4's external path stops /
         moves to a new dated protocol; the candidate / score family / policy / loss / lambda grid / comparator / thresholds
         must NOT be tuned after seeing this new-substrate replay.
DATE   : 2026-06-30 (machine UTC)
SUBSTRATE COMMIT (B1b/H5)   : b99fa4fcfb83c6ee60996c50dba6828d40561f26
COMPATIBILITY COMMIT (C5b)  : 5237378a7c8e4cd5c35aff8efb59701f148575c0
RAN AT                      : detached clean worktree /home/infres/yinwang/ACAR_V4_COMPAT_PREFLIGHT_5237378 (HEAD==5237378, clean)
ENV / NODE                  : acar-v4-regen (py 3.13); SLURM A40 job 877665, node27, driver 610.43.02 (== env lock); REPLAY_EXIT=0
```

## Verdict (FAIL) + per-disease numbers
```
status = SUBSTRATE_COMPATIBILITY_FAIL
reason = NOT AUTHORIZED: PD: coverage<0.15; PD: red<=0; PD: red<=v2_replay; SCZ: coverage<0.15; SCZ: red<=0;
         SCZ: red<=v2_replay; disease-macro red <= disease-macro v2_replay
                 coverage   red         v2_replay_red   L_harm_all_eval   harm_among_adapted   lambda_certified   v2_evaluable
   PD            0.0239     -0.00217    -0.000000        0.0174            0.727                True               True
   SCZ           0.0489     -0.01916    -0.000813        0.0489            1.000                True               True
```
Interpretation: at the pinned operating point (alpha=0.10, budget=0.10, coverage_min=0.15, v2_replay HARD), the fixed candidate
under the NEW substrate adapts only ~2–5% of subject-batches and those adaptations REDUCE nothing (negative red) — consistent
with the broader project finding that no source-free harm controller beats no-op (see notes/EVIDENCE_LEDGER.md + the gate-
falsification line). lambda was certified and v2 was evaluable for BOTH diseases, so the HARD v2 comparator was applied and the
candidate failed it on every criterion.

## Provenance (preserve)
```
authorization manifest : /home/infres/yinwang/acar_v4_compat_authorizations/compat_auth_5237378.json
                         sha256 = b385411f6abb41a311dfb88216d90b8c93d84d79c517dbd9e0d114bd7c85c344
substrate-compat manifest : /home/infres/yinwang/acar_v4_compat_manifests/acar_v4_substrate_compat_manifest_5237378.json
                         substrate_manifest_sha256 = 6c6ab457aed9f9925380214d154c407037e002cf52c8954976e98367d1b2e605
compat runtime lock    : /home/infres/yinwang/acar_v4_compat_capture/acar_v4_compat_env_lock_5237378.json
                         env_lock_sha256 = 523b44bc9480551c96b9f6d50bbadc7fccf027ebf2689d04379324f1dc3b767c
output dir             : /home/infres/yinwang/acar_v4_compat_replay_001_5237378   (created atomically by the run; was absent before)
   compat_RESULT.json   sha256 = a30ba74f0d272a6d90f91dc0c91d0bf7f161f533020b9842f5a89c6df784c937
   compat_manifest.json sha256 = 28a889ef755540b11ccbbb6d4d2e8f96917680cba66bd5a873f233496837d218
SLURM logs             : /home/infres/yinwang/acar_v4_compat_replay_logs/replay_5237378_877665.out (+ .err: only the benign
                         braindecode EEGNetv4 deprecation-alias warning)
command                : python -m acar.v4.run_substrate_compatibility --substrate-manifest <manifest> --output
                         /home/infres/yinwang/acar_v4_compat_replay_001_5237378 --compat-authorization <auth>
```

## Acceptance checks (ALL PASS — 14/14)
output dir + compat_RESULT + compat_manifest exist · RESULT/manifest parse allow_nan-clean (no NaN/Inf) · status is EXACTLY
SUBSTRATE_COMPATIBILITY_FAIL · fixed candidate unchanged (RESULT + manifest) · pass_line pinned (alpha 0.10 / budget 0.10 /
coverage_min 0.15 / v2_replay HARD) · v2_replay evaluable BOTH diseases · substrate_protocol_commit==b99fa4f ·
compatibility_protocol_commit==5237378 · substrate_manifest_sha256==6c6ab457… · env_lock_sha256==523b44bc… · NO
SELECT/DEV_STOP/external/lockbox/held-out vocabulary in the outputs · NO external/held-out input paths in the command.

## Confirmations
single replay run (exit 0; not an operational abort — a real PASS/FAIL verdict) · reads only the old-seven DEV scps cache +
the sha-pinned DEV feat-dump metadata + the FROZEN B1b substrate (no source-state refit) · NO held-out/external read · NO
external preprocessing · NO ACAR_FROZEN_v4 update · NO acar-v4-protocol tag · NO lockbox access. v2 (`MEASUREMENT_ONLY`) /
v3 (`DEV_STOP / NO_LOCKBOX_CONSUMED`) untouched; lockbox SEALED. The `.pt`/`.npz`/cache binaries are NOT committed (recorded by hash).

## Consequence (pre-registered FAIL rule — NO post-hoc tuning)
SUBSTRATE_COMPATIBILITY_FAIL ⇒ the V4 external path STOPS here (no external Arm-B authorized) or moves to a NEW dated protocol.
The candidate / score family / policy / loss / lambda grid / comparator / thresholds must NOT be tuned after seeing this
new-substrate replay. ACAR_FROZEN_v4.md substrate hashes are NOT promoted and the acar-v4-protocol tag is NOT created (those
were conditioned on a PASS). Any continuation is a separately pre-registered, dated decision. STOP for review.
