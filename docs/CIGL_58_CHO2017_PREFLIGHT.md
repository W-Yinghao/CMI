# CIGL_58 — Cho2017 held-out preflight, finalized (non-GPU)

Supersedes/finalizes CIGL_55 with verified-on-load facts. Machine-readable:
`results/heldout_preflight/cho2017_preflight.json`. **No Cho2017 GPU** — this confirms feasibility + the
adaptations needed before any external-validation run. Decision deferred until P10 seed0 returns.

## Verified (via datalake load)

| check | result |
|---|---|
| data location | **ALREADY in datalake** `/projects/EEG-foundation-model/datalake/raw/MNE-gigadb-data/gigadb-datasets/` — **no download** (all **52/52** subjects readable) |
| repo loader | supported; `DATASET_DEFAULTS["Cho2017"]` binary→**LeftRightImagery** (CORRECT for left/right; no BNCI2015 trap) |
| classes | 2: `{left_hand, right_hand}` |
| n_subjects | 52 (largest MI pool) |
| channels / sfreq | **69 ch / 512 Hz** (verified on subj-1 load) |
| MOABB interval | **[0, 3] s** → default 0.5–3.5s window overruns 1.0s → use **tmax ≤ 3.0** |
| sensorimotor coverage | **23 sensorimotor electrodes present** (FC5/3/1/z/2/4/6, C5/3/1/z/2/4/6, CP5/3/1/z/2/4/6, T7/T8), **incl. C3/Cz/C4** → a central-strip subset IS feasible |
| CSP-init source-only | feasible (binary CSP via `csp_init.source_csp_filters`, n_cls=2) |
| firewall | LOSO source-only + target eval-only — feasible, same protocol |
| endpoint | full-LOSO **mean/worst** (2-class; the 2a {1,3,8,9} 4-class decodable subset does NOT transfer) |

## Adaptations required before a Cho2017 run (no GPU now)

1. **`tmax ≤ 3.0`** for the [0,3] interval.
2. **`central_strip_v1` preset for Cho2017's 69-ch montage** — map to the 23-ch sensorimotor subset (or a
   central-strip subset) so `FBCSPLGGGraph`/graph branch runs; the CNN baselines (EEGNetMini/CSPInit) can take
   the full 69 ch or the subset — decide for a fair comparison.
3. Resample choice (512 Hz native → match the sidecar's 128 Hz or justify).
4. CSP-init source-only smoke on the Cho2017 montage.

## Verdict

Cho2017 is a **viable external-validation target** (data present, correct paradigm, 52 subjects, rich
sensorimotor montage, clean firewall). It is **not drop-in** — needs the tmax fix + a montage/central-strip
map. A positive on Cho2017 supports a **cross-site generalization** claim (weaker/different than the 2a
4-class decodable result). **Recommendation:** hold for the PI; revisit after P10 seed0. No Cho2017 GPU.
