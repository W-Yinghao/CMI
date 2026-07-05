# CIGL_55 — Cho2017 held-out preflight (non-GPU, metadata-grounded)

Preflight for the primary external-validation candidate from CIGL_54. Metadata from a MOABB probe (no
download). Machine-readable: `results/heldout_preflight/cho2017_preflight.json`. **No new-dataset GPU** — this
just enumerates what must be resolved before a Cho2017 full-LOSO seed0 run.

## Findings

| check | result |
|---|---|
| repo loader supports Cho2017 | **YES** — `DATASET_DEFAULTS["Cho2017"]=dict(binary=True, fmin=8, fmax=30)` |
| paradigm routing | binary→**LeftRightImagery** — CORRECT (events ARE left/right hand); no BNCI2015-style trap |
| n_subjects | **52** (largest MI pool available) |
| classes | 2: `{left_hand, right_hand}` |
| MOABB interval | **[0, 3] s** |
| window 0.5–3.5s compatible? | **NO** — overruns the [0,3] interval by 1.0s → must use `tmax ≤ 3.0` (e.g. window [0.5,3.0]=2.5s) |
| channels / sfreq | need a 1-subject load (literature: ~64 ch, 512 Hz) — verify on load |
| data cached / in datalake mirror | **NO** → **download required** (~GB, 52×64ch) |
| central_strip_v1 preset | **NONE** for Cho2017 — `FBCSPLGGGraph` graph branch needs a new 64-ch montage preset OR a sensorimotor channel-subset map |
| CSP-init source-only feasible | YES (binary CSP; `csp_init.source_csp_filters` handles n_cls=2) |
| decodable {1,3,8,9} subset applies? | **NO** (2a-specific 4-class) → report full-LOSO **mean + worst** on Cho2017 |
| firewall | LOSO source-only + target eval-only — feasible, same protocol |

## Verdict

Cho2017 is a **viable primary external-validation target** (loader supported, correct paradigm, 52 subjects,
clean firewall), but it is **not drop-in**. Blockers to resolve BEFORE any GPU run:

1. **Download** Cho2017 (not in the datalake readable mirror) or locate/build a readable mirror.
2. **Window fix:** `tmax ≤ 3.0` for the [0,3] interval (the 0.5–3.5s default overruns).
3. **Channel/sfreq verification** on a 1-subject load (expect ~64 ch / 512 Hz).
4. **Montage adaptation:** add a `central_strip_v1` preset for the Cho2017 64-ch montage OR map to a
   sensorimotor subset so `FBCSPLGGGraph` (and the D CSP-init) run on it.
5. **CSP-init source-only smoke** on the Cho2017 montage.

Endpoint on Cho2017 = full-LOSO **mean/worst** (2-class); the 2a CSP-decodable-subset story does not transfer.
A positive on Cho2017 would support a *cross-site generalization* claim for CSP-init FBCSP-LGG; it is a
different (weaker) claim than the 4-class 2a-decodable result.

**Recommendation:** hold for PI gate. The two real costs are the **download** and the **montage/central_strip
adaptation**. If approved, next non-GPU step = download + 1-subject load verify + montage map + CSP-init smoke;
only then a full-LOSO seed0 GPU run. No Cho2017 GPU now.
