# RQ4 representation-dump plan (gated on checkpoint availability)

**Not executed.** Plan only; runs only after the PM approves Option A (located checkpoints) or Option B (deterministic ERM re-fit) in `FSR_11`.

## What to dump
Per fold/seed, from a fixed FBCSP-LGG checkpoint in `eval()` mode, dump the **source-subject** frozen latents for the 3 fusion branches (+ optional node_z):

| latent | source | shape | role |
|---|---|---|---|
| `graph_z` | `self.last_aux['graph_z']` | [B, z_dim] | fusion branch (probe L1 + reliance L5) |
| `temporal_z` | `self.last_aux['temporal_z']` | [B, z_dim] | fusion branch |
| `spatial_z` | `self.last_aux['spatial_z']` | [B, z_dim] | fusion branch (the load-bearing one) |
| `fused_z` | `self.last_aux['fused_z']` | [B, fused_z_dim] | reference (whole-representation) |
| `node_z` (optional) | `forward_graph` return | [B, C, node_z_dim] | per-channel latent — NOT a fusion branch; label clearly |
| `gate` | `self.last_gate` | [B, 3] | branch weights (graph/temporal/spatial) |

Also dump `y` (task) and `d` (subject/domain) aligned to `B`, plus `source_indices`.

## Mechanics
1. Load checkpoint (`state_dict`), `backbone.eval()`, pin seed, disable dropout/BN-update.
2. Forward each source batch through `forward_graph(x)`; read `last_aux` + `last_gate` (already detached).
3. Concatenate per fold into `[N_source, ...]` arrays; save one `.npz` per (dataset, fold, seed, branch) with `z`, `y`, `d`, `source_indices`, `checkpoint_sha`.
4. **Determinism check:** dump twice; assert max|Δz| ≤ 1e-5 (as ACAR/H2CMI byte/│Δ│ self-replay). Fail → STOP.
5. Firewall: only source subjects enter the dump used for probe training; the held-out target subject's latents are dumped separately and used **only** for the L6 eval, never for probe fit.

## Determinism prerequisites
- Fixed checkpoint SHA + fixed seed + `eval()` mode + identical MOABB preprocessing (`central_strip_v1` montage, same filter-bank config as F0).
- No data augmentation, no shuffling in the dump loader.
- Record the exact config hash + checkpoint SHA in every `.npz` and in the manifest.

## What this plan does NOT do
No training, no fine-tuning, no CMI loss, no architecture change. If no checkpoint exists (current state), this plan does not run — it is blocked on `FSR_11` Option A/B.
