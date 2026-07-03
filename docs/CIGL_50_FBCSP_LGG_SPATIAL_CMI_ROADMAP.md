# CIGL_50 — FBCSP-LGG spatial-CMI + fusion-balance (P6, non-GPU)

## 0. Motivation (from FBCSP F0 / CIGL_49)

FBCSP-LGG F0 (full-LOSO ERM) showed the spatial branch is the load-bearing one on 4-class
(`zero_spatial` biggest ablation, gate_spatial highest), closing the P4 CSP gap on the decodable fold
(subj1 0.306→0.474 ≈ CSP 0.483). But two problems remained:

1. The existing dual-CMI objective (`graphcmi` / `graphdualpc`) penalizes **graph_z / node_z**, not the
   **spatial_z** branch that actually carries the 4-class signal — the objective is misaligned.
2. The 3-way fusion gate collapsed toward spatial (gate_spatial ~0.5–0.6) and **starved the graph
   branch** (gate_graph ~0.24, `zero_graph` neutral) — which caused the small BNCI2015 regression.

P6 fixes both, **non-GPU**. F1/GPU frozen; this is scaffold + tests only. Implemented on an isolated git
worktree so it never touches the checkout the running F1a/reference GPU jobs read.

## 1. P6-A — spatial encoder CMI (new `fbdualpc` method)

Adds a spatial posterior head and penalizes the spatial branch's conditional domain leakage:

```text
post_spatial : q(D | spatial_z, Y)        # DomainPosteriors on spatial_z_dim, independent params
R_spatial    = I~(spatial_z ; D | Y)      # GLS posterior-KL, reference = marginal (same as graph/node)
```

New method `fbdualpc` = `graphdualpc` + the spatial term (it shares the whole Graph-DualCMI branch, so
head-split / decoder residual / GLS / firewall are unchanged). New grammar:

```text
fbdualpc:<lambda_g>:<lambda_node>:<lambda_spatial>:<lambda_edge>:<gamma_dec>[:<dec_scale>]
```

Examples:
```text
fbdualpc:0.000:0.000:0.010:0.000:0.100:300   # spatial-only encoder CMI + decoder residual
fbdualpc:0.005:0.005:0.010:0.000:0.100:300   # graph + node + spatial + decoder
```

Requires a backbone with a spatial branch (`meta['has_spatial_branch']`) exposing a grad-carrying
`last_spatial_z`; fails closed otherwise. Diagnostics returned/recorded: `reg_graph_gls`, `reg_node_gls`,
**`reg_spatial_gls`**, `lambda_spatial`, **`loss_spatial`**, `stepA_spatial_loss_gls`, plus the existing
decoder-activation set (`dec_js_res_raw/scaled`, `loss_dec_over_ce`, ...).

## 2. P6-B — fusion balance (gate floor)

A light, deterministic floor so no branch is fully starved (chosen over a gate-entropy regularizer for
simplicity, per PI):

```text
gate = softmax(gate3(...))                     # [B,3]
gate = (1 - 3*eps) * gate + eps                # still sums to 1; each weight >= eps
```

Config: `--fusion_floor 0.0 / 0.05 / 0.10` (0 = plain softmax, off; only affects FBCSPLGGGraph).
Does NOT force the graph branch to dominate — only prevents starvation. New diagnostic
`gate_entropy_mean` (per-batch gate entropy; low = collapsed, high = balanced; max ln 3 ≈ 1.099) added to
`gate_summary` alongside the per-branch gate mean/std.

## 3. Contract / scope

- `FBCSPLGGGraph.__init__(..., fusion_floor=0.0)`; `forward_graph` now stores `last_spatial_z`
  (grad-carrying) for the spatial CMI. 5-tuple contract, ablations, gate instrumentation unchanged.
- `run_loso`: `fbdualpc` grammar (9-tuple parse_config with a new `lam_spatial` slot), `--fusion_floor`
  passthrough, and `reg_spatial_gls`/`loss_spatial`/`gate_entropy_mean` recorded per fold.
- `FBLGGGraph` / `DGCNNGraph` / `graphdualpc` / `graphcmi` behavior unchanged (fusion_floor and
  lam_spatial default to off; graphdualpc parse is 9-tuple with lam_spatial=0).

## 4. Status / tests

- 7 new P6 tests + 81 regression tests pass (grammar 9-tuple; fbdualpc runs + `reg_spatial_gls` live;
  fail-closed on non-spatial backbone; gate floor bounds each gate ≥ eps + raises entropy; fusion_floor=0
  is plain softmax; firewall determinism; graphdualpc unaffected).
- Real-EEG CPU smoke via SLURM is pending queue capacity (the 147-job GPU run + other-project jobs
  saturate the QOS submit cap); it will run before any P6 GPU screening.

## 5. Proposed P6 GPU screening — full-LOSO seed0 (NOT approved; submit-only)

Per the "full-LOSO for science" rule (no 2-fold pilots):

```text
Backbone : FBCSPLGGGraph
Datasets : BNCI2014_001 folds 0-8  +  BNCI2015_001 folds 0-11   Seed: 0
Configs  :
  erm:0                                   --fusion_floor 0.05   # does the floor recover the graph branch?
  erm:0                                   --fusion_floor 0.10
  fbdualpc:0.000:0.000:0.010:0.000:0.100:300   --fusion_floor 0.05   # spatial-only encoder CMI + decoder
  fbdualpc:0.005:0.005:0.010:0.000:0.100:300   --fusion_floor 0.05   # graph+node+spatial + decoder
Flags    : --source_val_early_stop
Gate     : does aligning the CMI to the spatial branch (+ un-starving graph) beat FBCSP-LGG ERM
           (2a full-LOSO 0.349 / 2015 0.608)? Report gate means + gate_entropy + all 4 ablations +
           reg_spatial_gls activation (loss_spatial ratio). dec_scale=300 (recalibrate only if diagnostics
           show it dormant/dominating).
```

## 6. Frozen
No GPU until the PI approves a run-spec. No old `graphcmi` / old `graphdualpc dual` on FBCSPLGGGraph
(they ignore the load-bearing spatial branch). CMI otherwise frozen.
