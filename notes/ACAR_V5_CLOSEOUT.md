# ACAR v5 — CLOSEOUT

```
protocol tag        : acar-v5-protocol @ 4278435975a72b1127803dd2cffab420c083e430   (frozen pre-registration)
Stage-1B package    : run acar-v5-stage1b-c4412b4-r1  (30/30 refs admitted; registry_sha256 2bbe55f4…bbcbb76d)
Stage-2B fix chain  : matched_coral numeric amendment (f079aca) · forced-tail eligibility (ba09777)
Stage-2B run        : acar-v5-stage2b-ba09777-r1  (SLURM 885395; 8.18 h; rc=0)  -> DEV_STOP / no candidate  (commit d287635)
VERDICT             : ENGINEERING/PROTOCOL RECOVERY = SUCCESS ; SCIENTIFIC ROUTER HYPOTHESIS = FAIL / REFUTED ON DEV
Stage-4 S1/S2/S3    : NOT AUTHORIZED / NOT RUN  (needs a selected candidate; there is none)
external / held-out : NOT TOUCHED   ·   ASZED : NOT TOUCHED   ·   lockbox : SEALED
allowed future      : NEW dated protocol only (a genuinely new hypothesis, NOT a v5b re-tune); NO rerun / tuning / reselection
```

ACAR v5 is **closed**. It must be read as two separate results — do not conflate them:

```
ACAR-V5 succeeded in constructing an auditable, external-compatible, hash-bound DEV substrate
and a guarded, label-firewalled Stage-2 selection pipeline that runs to a clean, pre-registered verdict.

ACAR-V5 failed to produce a safe/useful DEV-selected routing candidate.
The Stage-2B run ended in DEV_STOP with zero eligible candidates.

Therefore:
  no Stage-4 robustness is authorized;
  no external/held-out evaluation is authorized;
  lockbox remains sealed;
  the V5 router hypothesis is REFUTED on DEV.
```

## What v5 recovered (the engineering/protocol success — real, but scope-limited)

v5's mandate was to fix the two failure modes that killed v3 (coverage collapse) and v4 (substrate incompatibility): a
**substrate-robust, constrained-utility router** with a conditional-adapted-harm gate at the center. On the machinery, it delivered:

- **A hash-bound, external-compatible Stage-1B DEV substrate.** 30/30 refs admitted; registry / FINALIZED package verified; a
  single external-compatible source-state per fold; repair/montage/channel provenance recorded (`notes/ACAR_V5_STAGE1B*.md`). This
  is the object v4 never had (v4's fixed candidate had no all-DEV encoder to transfer to). **This is the one confirmed positive**,
  and it is ENGINEERING scope only — not efficacy.
- **A guarded, label-firewalled Stage-2 selection pipeline** (FIT thresholds / CAL Holm-132 certification / EVAL G1–G5), with the
  numeric + eligibility hardening the real substrate demanded: the `stable_matched_coral_v1` amendment (bounded, rank-aware CORAL;
  `notes/ACAR_V5_STAGE2B2_*`) and the forced-tail identity-only correction (`notes/ACAR_V5_STAGE2B3_*`), each adversarially
  reviewed and stress-tested label-free on the real package across two distinct nodes (`…STAGE2B2P_*`, `…STAGE2B3P_*`).
- **A clean, pre-registered binding run** (rc=0), whose verdict is scientifically interpretable rather than an artifact.

## What v5 refuted (the scientific failure — the actual result)

On the admitted real PD/SCZ DEV substrate, the frozen ACAR-V5 label-free routing/action policy class does **not** produce a
candidate that is both safe and beneficial under the pre-registered G1–G5 gates. The sharpest, cleanest form of the finding:

> On the admitted real PD/SCZ DEV substrate, the frozen ACAR-V5 label-free routing/action policy class does not produce a
> candidate that is both safe and beneficial under the pre-registered G1–G5 gates.

The failure is **harm control, not coverage collapse** (unlike v3): coverage passes for most candidates (G1 PD 13/22, SCZ 19/22),
but the harm gates fail universally (G4 harm_among_adapted UCB ∈ [0.61, 0.87] ≫ 0.30 on all 42 evaluable cells; G3 L_harm_all UCB
> 0.10 almost everywhere), and EVAL `red` is strongly negative (−12.12 .. +0.01) — the actions increase loss. The identity /
source-state LDA `f_0` dominates. Details + the full 22-row gate table: `notes/ACAR_V5_STAGE2B_CLOSEOUT.md`.

## Direction-2 lineage (authoritative; supersedes the mid-state drafts)

| stage | tag / commit | status | one-line |
|-------|--------------|--------|----------|
| A0 / A0′ gate-falsification | (exp/lpc-cmi) | **DIAGNOSTIC_ONLY (closed)** | no source-free harm controller reduces deployed loss; density/CMI wrong-signed; rollback was label leakage |
| ACAR v2 | `acar-v2-protocol @ 9b2f0c1`; result `1528a94` | **MEASUREMENT_ONLY** | label-free action-conditional features predict negative transfer (G1✓); router not deployable (G2✗) |
| ACAR v3 (HSCR) | `acar-v3-dev-design-v1 @ 817b04f`; result `9f4e83f` | **DEV_STOP / NO_LOCKBOX_CONSUMED** | stricter conformal redesign fails the DEV S2/S4 gate — **coverage collapse** (all-action conformal coverage ~0.6–1.1 %); weak PD center |
| ACAR v4 (CURB) | DEV `e9760e6`; substrate `b99fa4f`; compat `5237378`; result `c605e24` | **SUBSTRATE_COMPATIBILITY_FAIL / NO_EXTERNAL** | executed-policy LTT found a DEV-only candidate (14/90, non-binding) that does **not transfer** to the regenerated all-DEV substrate — dies before a clean selection |
| ACAR v5 (substrate-robust constrained-utility router) | `acar-v5-protocol @ 4278435`; substrate `acar-v5-stage1b-c4412b4-r1`; run `acar-v5-stage2b-ba09777-r1` @ `ba09777`; result `d287635` | **substrate/protocol recovery SUCCESS ; Stage-2B DEV_STOP — no candidate** | first CLEAN run on a hash-bound external-compatible substrate; DEV_STOP driven by **harm control** (G3/G4), not coverage — coverage is fine, adaptation HURTS, identity `f_0` dominates |

The through-line across v2→v5: the **measurement→control gap is NOT closed**. v2 showed the harm signal exists but isn't a
deployable router; v3 (coverage collapse), v4 (substrate incompatibility), and v5 (harm — adaptation is net-harmful on a clean
substrate) are three distinct pre-registered ways the label-free routing/action class fails to become a safe, beneficial
controller. v5 is the strongest form: given a properly-admitted external-compatible substrate and proper harm gates, no policy in
the frozen 22-candidate universe is safe or useful.

## Overall project judgment

```
As rigorous protocol + negative-result research:   SUCCESS
As "ACAR finally closes measurement -> deployment": FAIL
Overall ACAR deployment gap:                        NOT CLOSED
```

## What is NOT allowed (binding)

Do **not**: enter Stage-4 (no selected candidate exists); read external / held-out / ASZED; open the lockbox; re-tune G3/G4, the
candidate universe, `matched_coral`, batch size, MIN_BATCH, the threshold grid, the CAL/EVAL interpretation, or the v2-replay
comparator; or pull P3/P4 near-identity candidates out for a rescue pass — each of these turns the pre-registered Stage-2B into
post-hoc tuning. Do **not** frame the outcome as an "action-provider engineering problem": the forced-tail / stable-CORAL numeric
issues were fixed and stress-verified; the final run was a clean rc=0 DEV_STOP.

## If a v6 is ever opened

Only as a **new hypothesis** with a **new dated protocol** and a clean confirmation route — never a v5b re-tune. Note the
identifiability caveat: the same DEV labels have now been used to observe that the frozen V5 action/policy class is harmful; any
v6 designed using this observation is **exploratory** unless it carries a new dated protocol and a clean confirmation route
(fresh substrate / held-out). A useful (diagnostic-only, separately-authorized) pre-v6 question set — do NOT reverse-engineer a
new policy from it and then claim confirmatory success: *why do the adaptation actions hurt?* (is `f_0` already strong; are
post-action distributions overconfident; does CORAL/spdim/t3a destroy class separation; are harms concentrated in the repaired /
completed cohorts ds004584 / ds004000 / ds004367; are harms action-specific or universal). These are diagnostics unless
separately authorized.
