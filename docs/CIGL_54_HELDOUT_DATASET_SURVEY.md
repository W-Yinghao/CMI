# CIGL_54 — Held-out dataset survey for CSP-init FBCSP-LGG external validation (non-GPU)

Goal: pick ONE held-out MI site/dataset for a later full-LOSO seed0 external check of the promoted D model.
**No new-dataset GPU now** — survey + loader/preprocessing verification first (the BNCI2015_001
LeftRightImagery→MotorImagery loader trap showed why). Metadata below from a MOABB metadata probe (no download);
channel counts flagged "verify on load".

## Candidate table (MOABB)

| dataset | n_subj | classes | sess | ~n_ch | montage vs 2a(22ch) | fit CSP-init source-only? | recommend |
|---|---|---|---|---|---|---|---|
| BNCI2014_001 (2a) | 9 | 4: L/R/feet/tongue | 2 | 22 | — (USED) | yes | in use |
| BNCI2015_001 | 12 | 2: R/feet | 2 | 13 | subset | yes | in use |
| BNCI2014_004 (2b) | 9 | 2: L/R | 5 | **3** (C3/Cz/C4) | **too few ch** | CSP degenerate on 3ch | **no** |
| Cho2017 | **52** | 2: L/R | 1 | ~64 | different (need channel map) | yes (after map) | **candidate (2-class, largest)** |
| Lee2019_MI | **54** | 2: L/R | 2 | ~62 | different (need channel map) | yes (after map) | candidate (2-class) |
| Zhou2016 | 4 | 3: L/R/feet | 3 | ~14 | subset | too few subjects | no |
| Weibo2014 | 10 | **7-class** | 1 | ~60 | complex | class mismatch | no |
| PhysionetMI | 109 | L/R/feet/hands/rest | 1 | ~64 | movement+imagery mix | paradigm mismatch risk | maybe (careful) |
| Schirrmeister2017 (HGD) | 14 | 4: R/L/rest/feet | 1 | ~128 | high-density subset | yes (drop rest, subset ch) | **candidate (only other 4-class)** |

## Key finding

**There is no drop-in 4-class, ~22-channel held-out dataset.** External validation requires adaptation:
- **2-class route (cleanest loader):** Cho2017 (52 subj) or Lee2019_MI (54 subj), both **L/R** (note: different
  contrast from 2015's R/feet). Needs a channel mapping to a common sensorimotor montage before CSP-init;
  the CSP-decodable-subset story is 2-class so the {1,3,8,9} 4-class endpoint does NOT transfer — we'd report
  full-LOSO mean + worst instead.
- **4-class route (keeps the endpoint type):** Schirrmeister2017 HGD (14 subj, 4-class incl. `rest`) — needs
  dropping `rest`/mapping to L/R/feet/tongue-analog + a 22-ch (or central-strip) subset from its 128 ch. More
  preprocessing work; closest to the 2a paradigm.
- **BNCI2014_004 (2b) is out** — 3 bipolar channels make CSP-init/central_strip degenerate.

## Recommendation (for PI gate; NO GPU yet)

1. **Primary held-out target: Cho2017** — largest subject pool (52), simplest binary loader, MOABB-native.
   Report full-LOSO mean/worst (not the {1,3,8,9} subset, which is 2a-specific). Requires: (a) MOABB loader +
   readable-mirror check, (b) channel-map to a sensorimotor subset compatible with central_strip_v1, (c)
   CSP-init source-only verification on its montage.
2. **If a 4-class external check is wanted: Schirrmeister2017 HGD** — keeps the 4-class decodable-subset flavor
   but needs rest-drop + 128→~22 channel subsetting.
3. **Do NOT** attempt BNCI2014_004 (channels), Weibo2014 (7-class), Zhou2016 (4 subj).

**Next (non-GPU, on PI approval):** a preflight for the chosen dataset — MOABB load via readable mirror,
paradigm/class/channel/shape verification, channel-map to central_strip, CSP-init source-only smoke — mirroring
the BNCI2015_001 preflight discipline. Only then a full-LOSO seed0 GPU run. Channel counts above are from
memory/typical montages and MUST be verified on load.
