# CIGL_46 Graph-DualCMI — Pilot 1 (G0/G1/G2) result summary

**Verdict: static `DGCNNGraph` fails the SOTA scale-up gate → PI Rule B + Rule D → close the
static-DGCNN SOTA track, pivot the main line to FB-LGG-DualCMI.** This is a clean negative result;
the CIGL v0.6 bounded graph/node leakage-audit contribution is unaffected (it never claimed SOTA,
edge-CMI, cross-architecture, or beyond-MI results).

## Provenance

| field | value |
|---|---|
| Branch / SHA | `project/graph-dualcmi-sota-roadmap` @ `30a97ec` (tracked tree clean at run time) |
| Environment | conda `eeg2025`; torch `2.6.0+cu124`; moabb `1.5.0`; mne `1.11.0` |
| braindecode | import fails in this env → **sidecar only, not used in pilot 1** (graph-only) |
| SLURM partitions | `A100,V100,V100-32GB,A40`, default QOS, no `--time` (fail-closed if CUDA unavailable) |
| MNE data path | readable mirror `/projects/EEG-foundation-model/yinghao/cigl_bnci_readable` |
| MNE concurrency mitigation | per-job private `HOME=/tmp/mne_home_<jobid>` (fixes the config-lock stale-file-handle fault) |
| Runner | `python -m cmi.run_loso --backbone DGCNNGraph` |
| Train hyperparams | `--epochs 300 --bs 64 --warmup 40 --n_inner 2 --device cuda` |

## Datasets / folds / seeds / configs

| dataset | target folds | classes | chance |
|---|---|---|---|
| BNCI2014_001 | 0, 1 | 4 | 0.25 |
| BNCI2015_001 | 0, 9 (fold 9 = prior source-retention boundary case) | 2 | 0.50 |
| seeds | 0, 1, 2 | — | — |

Configs (config strings): `erm:0`, `graphcmi:0.010:0.010:0.000`,
`graphdualpc:0.000:0.000:0.000:0.100` (decoder-only), `graphdualpc:0.010:0.010:0.000:0.100` (dual),
`cdann:1` (conditional-adversarial baseline; appendix).

## Job IDs

- **Seed 0** (G0/G1): 2a `878176` (5-config); 2015 split config-jobs `878182`–`878186`, `878270`.
- **Seeds 1/2 mainline** (G2): `878452` (2a s1), `878453` (2a s2), `878454` (2015 s1), `878455` (2015 s2)
  — 4 mainline configs each, MNE private-HOME.
- **CDANN side-jobs** (G2, state-gated after all 4 mainline passed data-load): `878483`, `878484`,
  `878485`, `878486` → written to separate `*_cdann_side.json`, never merged into mainline JSON.

## Aggregate (across seeds 0/1/2; 6 cells per config = 2 folds × 3 seeds). Authoritative CSV: `G2_AGGREGATE.csv`.

### BNCI2014_001 (4-class, chance 0.25)
| config | mean tgt bAcc | worst bAcc | source bAcc | Δ vs ERM | worst Δ |
|---|---|---|---|---|---|
| ERM | 0.349 | 0.271 | 1.000 | — | — |
| graphcmi | 0.331 | 0.285 | 1.000 | −0.018 | +0.014 |
| graphdualpc decoder-only | 0.341 | 0.278 | 1.000 | −0.008 | +0.007 |
| **graphdualpc dual** | 0.329 | 0.259 | 1.000 | **−0.020** | −0.012 |
| cdann *(appendix)* | 0.337 | 0.250 | 0.748 | −0.012 | −0.021 |

### BNCI2015_001 (binary, chance 0.50)
| config | mean tgt bAcc | worst bAcc | source bAcc | Δ vs ERM | worst Δ |
|---|---|---|---|---|---|
| ERM | 0.594 | 0.538 | 1.000 | — | — |
| graphcmi | 0.604 | 0.532 | 1.000 | +0.011 | −0.007 |
| graphdualpc decoder-only | 0.580 | 0.543 | 1.000 | −0.014 | +0.005 |
| **graphdualpc dual** | 0.570 | 0.528 | 1.000 | **−0.024** | −0.010 |
| cdann *(appendix)* | 0.541 | 0.475 | 0.685 | −0.053 | −0.063 |

## Operational checks (all pass)

- **8/8 G2 jobs `rc=0`**; MNE private-HOME held → **0 stale-file-handle**, 0 MNE config-lock faults.
- **0 NaN/inf** across all 60 result cells.
- **Graph branch is load-bearing:** `ablate_zero_graph` → exactly chance (0.250 / 0.500);
  `ablate_permute_nodes` ≈ chance (0.241–0.243 / 0.505–0.506). The negative result is **not** a
  graph-ignored artifact — the graph is used, it just does not transfer across subjects.
- **BNCI2015 loaded via the mirror** in every job (fold 9 included); no unreadable-datalake fallback.
- No braindecode / EEGNet import in any job log.

## Decision (PI-precommitted rules)

- **Rule B fires (BNCI2015):** the seed-0 dual **+2.9pp** gain **did not survive** seeds 0/1/2 — dual is
  now **−2.4pp** mean / **−1.0pp** worst vs ERM. It was seed noise. `graphcmi` is only +1.1pp mean and
  worse on worst-subject → insufficient to scale to 12 folds. → *stop static DGCNNGraph SOTA track.*
- **Rule D fires (BNCI2014):** all configs near chance (0.33–0.35 vs 0.25) with **source bAcc = 1.000**
  (total source overfit / cross-subject transfer failure, method-independent). → *static DGCNNGraph =
  diagnostic backbone only; main line → FB-LGG-DualCMI.*
- **Rule A** (dual ≥ +2pp → scale to 12 folds): ✗ dual negative on both.
- **Rule C** (decoder-only *stably* > ERM → objective ablation): ✗ decoder-only −0.8pp mean on 2a.

## Caveats (do not over-read the negative)

1. **The decoder-residual thesis was never actually exercised.** `dec_js_res ≈ 3e-4`,
   `dec_ce_res ≈ 0` on both datasets → the `[JS−τ]_+` gate never fired; at `γdec=0.100` the decoder
   term is numerically negligible vs CE. So `I(Y;D|Z)` is **dormant, not falsified** — decoder-only ≈
   ERM by construction. Activating it needs a τ / `dec_scale` design (→ P3-D on the new track).
2. **Encoder GLS terms were active** (`reg_graph_gls ≈ 2.0`, `reg_node_gls ≈ 1.0`) and at λ=0.01 they
   mildly **hurt** target bAcc — consistent with "encoder penalty too strong / backbone too weak".
3. **source bAcc = 1.000 everywhere** is a red flag: 300 fixed epochs with no source-domain validation
   over-memorizes source. The next track adds **source-only early stopping** (→ P3-E), never touching
   target labels.

## Conclusion / next

Static `DGCNNGraph` SOTA track **closed** (diagnostic backbone retained). Main line pivots to
**FB-LGG-DualCMI** (stronger filterbank temporal + local–global graph backbone, active decoder
residual, source-only early stopping). Design + scaffolding are **non-GPU**; GPU stays frozen until a
fresh run-spec is approved. Roadmap: `docs/CIGL_47_FB_LGG_DUALCMI_ROADMAP.md`. CIGL_45 audit-hardening
fallback remains available.
