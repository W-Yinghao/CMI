# CIGL Phase 3A-K — Second-Dataset Confirmation Results (BNCI2015_001)

> **EXPLORATORY confirmation — NOT a benchmark / SOTA result.** Replication of ONE fixed source-only
> candidate (`graph_node_010`) on a SECOND MI dataset. No λ grid, no new configs, no edge term. Decision A
> here means the effect **replicates across two datasets**; it is a measurement→control result on the
> DGCNN backbone, not a general/cross-architecture or SOTA claim.

## 1–6. Run provenance

```bash
sbatch scripts/sbatch_cigl_phase3a_dgcnn_gn_second_dataset_confirmation_bnci2015_001.sh
# -> python scripts/run_cigl_phase3a_dgcnn_gn_second_dataset_confirmation.py --dataset BNCI2015_001 \
#      --device cuda --seeds 0 1 2 --epochs 80 --probe_epochs 100 --n_perm 50 --gate_alpha 0.05
```

| field | value |
|---|---|
| SLURM job id | **877369** |
| partition / node | multi-partition `A100,V100,V100-32GB,A40` (default QOS) → **node12** |
| runtime | ~65 min (12 folds × 2 configs × 3 seeds × 80 ep + n_perm=50 graph+node audits; log mtime 18:18) |
| branch / commit_hash | `project/cigl-phase3a-dgcnn-gn-second-dataset-confirmation` / `f1908742869de1f70abf5c1cc86916f994123a2c` |
| config_hash | `bc247f1f68c2` |
| environment | conda `eeg2025`, torch 2.6.0+cu124 (CUDA) |
| dataset / folds | BNCI2015_001 / **12 LOSO folds** (subjects 1–12; all are confirmation — no dev fold) |
| n_classes / chance | **2** / **0.50** |

## 7. Preprocessing + datalake-mirror provenance

`meta.preprocessing`: `moabb_paradigm=MotorImagery`, `events=["right_hand","feet"]`, `resample=128`
(rationale: match BNCI2014_001 confirmation protocol; 250 Hz a known note, not changed), `tmin=0.5`,
`tmax=3.5`, `dataset_interval=[0,5]`, `window_inside_declared_interval=true`.

**Data sourced entirely from `/projects/EEG-foundation-model/datalake`.** MOABB's `bnci_2015` lampx mirror
(`~bci/database/001-2015/`) is owner-locked/unreadable; the sbatch builds a **readable symlink mirror**
(symlinks only, **no download, no data copied**) pointing `~bci/database/001-2015/` at the datalake's
readable bnci-horizon copy `database/data-sets/001-2015/` (other datasets resolve to the datalake
unchanged). Preflight verified identical load (12 subj, 13 ch, n_times 384).

## 8. Fixed configs (no λ grid was run)

`erm_fixed` (λ_g=λ_node=0) and `graph_node_010` (λ_g=λ_node=0.010, λ_edge=0). Two FIXED configs only.

## 9. Per-fold table (ERM vs graph_node_010; `d`=drop vs ERM; tgt evaluation-only; clears = null-clearance g/n seeds)

| fold (subj) | ERM src | reg src (d) | ERM tgt | reg tgt (d) | graph KL e→r (red) | node KL e→r (red) | reg clears | PASS |
|---|---|---|---|---|---|---|---|---|
| 0 (s1) | 0.693 | 0.689 (+0.003) | 0.602 | 0.568 (+0.035) | 1.05→0.24 (77%) | 0.55→0.23 (57%) | 3/3 | ✓ |
| 1 (s2) | 0.694 | 0.676 (+0.018) | 0.799 | 0.780 (+0.019) | 1.18→0.46 (62%) | 0.59→0.31 (48%) | 3/3 | ✓ |
| 2 (s3) | 0.689 | 0.683 (+0.006) | 0.644 | 0.642 (+0.002) | 1.15→0.34 (71%) | 0.60→0.29 (51%) | 3/3 | ✓ |
| 3 (s4) | 0.682 | 0.695 (−0.014) | 0.699 | 0.719 (−0.020) | 1.22→0.54 (56%) | 0.61→0.36 (41%) | 3/3 | ✓ |
| 4 (s5) | 0.701 | 0.696 (+0.006) | 0.568 | 0.583 (−0.016) | 1.14→0.29 (74%) | 0.58→0.25 (57%) | 3/3 | ✓ |
| 5 (s6) | 0.715 | 0.710 (+0.004) | 0.632 | 0.632 (+0.000) | 1.19→0.69 (43%) | 0.66→0.42 (37%) | 3/3 | ✓ |
| 6 (s7) | 0.710 | 0.703 (+0.008) | 0.502 | 0.506 (−0.004) | 1.16→0.31 (74%) | 0.58→0.25 (56%) | 3/3 | ✓ |
| 7 (s8) | 0.714 | 0.712 (+0.002) | 0.535 | 0.520 (+0.015) | 1.16→0.33 (71%) | 0.53→0.23 (57%) | 3/3 | ✓ |
| 8 (s9) | 0.707 | 0.696 (+0.012) | 0.534 | 0.538 (−0.004) | 1.11→0.28 (75%) | 0.52→0.20 (61%) | 3/3 | ✓ |
| 9 (s10) | 0.734 | 0.710 (**+0.024**) | 0.543 | 0.546 (−0.002) | 1.18→0.41 (65%) | 0.62→0.32 (49%) | 3/3 | ✗ (src drop >0.02) |
| 10 (s11) | 0.722 | 0.708 (+0.014) | 0.496 | 0.506 (−0.009) | 1.29→0.47 (64%) | 0.66→0.31 (53%) | 3/3 | ✓ |
| 11 (s12) | 0.717 | 0.722 (−0.005) | 0.516 | 0.475 (+0.041) | 1.17→0.42 (65%) | 0.58→0.26 (56%) | 3/3 | ✓ |

ERM clears the null g3/n3 in every fold too. **Only fold9 fails** — source drop +0.024 (>0.02); its leakage
still reduces (65%) and its target guardrail holds.

## 10–12. Pass/fail criteria + three-layer verdict (PRIMARY = all 12 folds; no dev fold)

| criterion | threshold | result |
|---|---|---|
| ERM adequacy (src ≥0.60 mean, ≥2/3 seeds ≥0.55) | ≥8/12 | **12/12** ✓ |
| ERM leakage target exists (graph or node clears null) | ≥8/12 | **12/12** ✓ |
| regularizer reduces graph or node KL ≥30% (≥2/3 seeds) | ≥7/12 | **12/12** ✓ |
| source retained (reg src ≥0.60, drop ≤0.02) | ≥7/12 | **11/12** ✓ |
| target guardrail (target drop ≤0.05, eval-only) | ≥7/12 | **12/12** ✓ |

- **`source_only_confirmed = true`** (criteria 1–4).
- **`target_guardrail_pass = true`** (12/12 ≥ need 7).
- **`confirmed_with_target_guardrail = true` → Decision A.**

## 13. Edge — explicitly skipped (not faked)

`edge_regularization_used=false`, `edge_logits_dynamic=false`, `edge_audit_skipped=true`,
`edge_skip_reason="static/shared adjacency: edge_logits=None; no per-sample edge object"`.

## 14. Firewall flags

`used_target_labels_for_training=false`, `used_target_labels_for_selection=false`,
`used_target_covariates=false`, `target_eval_is_evaluation_only=true`, `selection_uses_target_eval=false`,
`confirmation_label_selection_uses_target_eval=false`.

## 15. Recommended decision — **A: confirmed on the second dataset** *(pending reviewer)*

The fixed `graph_node_010` **replicates on BNCI2015_001**: it reduces graph KL **43–77%** (mean ≈ 65%) and
node KL **37–61%** while **retaining the (binary) source task** (11/12 folds; ERM ~0.68–0.73 ≫ 0.60) and
holding the target guardrail (12/12). Combined with Phase 3A-J (BNCI2014_001, 4-class, ~40% reduction at
no task cost across 9 folds), the task-preserving graph/node leakage reduction now holds on **two MI
datasets** with different subjects, channels, class counts, and chance levels. Per the reviewer's rule
(`confirmed_with_target_guardrail=true`) → **Decision A: cross-dataset method framing may begin.**

**Honest caveats (keep the claim bounded):**
- **Partial controllability, not erasure — consistently.** Reduced leakage **still clears the null in every
  fold** (`reg clears g3/n3` throughout); ~65% reduction, the subject fingerprint is strongly dented, not
  removed. The leakage metric is a posterior-KL **proxy**, not an unbiased CMI.
- **Two MI datasets, one fixed config, graph/node only.** This is the DGCNN static-adjacency backbone with
  `graph_node_010`; **no edge-CMI** (no per-sample edge object), no λ exploration. Not a cross-architecture
  or SOTA claim.
- **One fold (fold9) missed source retention** (drop +0.024); 11/12, well above the 7/12 bar.
- BNCI2015_001 is more comfortable (binary, ERM ~0.70) than BNCI2014_001 (4-class, ~0.46); the effect holds
  in both regimes, but absolute task levels differ.

**Next (reviewer-gated):** with two-dataset confirmation in hand, method framing may begin — but it should
state the bounded scope (graph/node CMI on a task-capable static-adjacency graph backbone; partial,
significant, task-preserving leakage reduction; two MI datasets). No edge-CMI, no λ grid, no SEED/DEAP,
no Lee2019, no SOTA table without explicit authorization. Generated per-fold JSON are gitignored; this doc
is the tracked record.
