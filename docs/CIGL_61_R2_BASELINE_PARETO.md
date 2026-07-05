# CIGL_61 (R2a) — Baseline registry + same-backbone contract + Pareto schema (non-GPU scaffold)

```
Status:
  Engineering scaffold complete (R2a subset).
  Scientific evidence NOT yet complete.

Validated by tests/synthetic fixtures:
  all R2a methods run on the SAME DGCNN adapter; config registry well-formed; CDAN wired;
  within-label permutation firewall; Pareto domination rule; .audit.npz round-trip.

NOT validated (needs real EEG full-LOSO):
  whether CIGL sits on the leakage-vs-task Pareto frontier; whether a plain conditional-DANN matches it.
```

Branch `project/cigl-r123-scaffold`. PM R2a scope: **scaffold + integration + report schema only** — not the
full baseline zoo, not a leaderboard. Answers (once run): *does CMI regularization have INDEPENDENT value on
the task/leakage Pareto front, vs adversarial/conditional-adversarial baselines on the SAME backbone?*

## R2a — active method set (all on the SAME static-adjacency DGCNN adapter)
`cmi/eval/baseline_registry.py`: **erm · cigl_graph · cigl_node · cigl_graph_node · dann · cond_dann · cdan**.
Deferred (registry placeholders, NOT run in R2a): coral/label-coral, mmd/label-mmd, irm, vrex, groupdro,
nodedat, eeg_dg. `SAME_BACKBONE_CONTRACT`: `dgcnn_forward_graph_adapter`, source-only LOSO, source-val
early-stop, target eval-only firewall, graph_z+node_z audit, within-label permutation null.

**CDAN added (additive, `cmi/methods/dg_penalties.py`):** `make_cdan_discriminator` + `cdan_penalty`
(multilinear map z⊗softmax(ŷ), gradient-reversed) + trainer dispatch + `ADV_METHODS += cdan`. DANN/CDANN/
graphcmi semantics unchanged. (Also fixed: braindecode is now imported LAZILY in `backbones.py`, so the
pure-torch DGCNN path is importable in eeg2025.)

## R2b — Pareto report schema
`cmi/eval/pareto_report.py`: unified per-run `ROW_SCHEMA` (dataset/fold/target/seed/method/λ_g/λ_node/source&
target bAcc/graph&node KL proxy/perm-p/FDR-q/multiprobe count/task-retention/leakage-reduction). Domination:
**a method is dominated iff another has ≥ target task AND ≤ graph leakage AND ≤ node leakage, with ≥1 strict
improvement.** Consumes dummy/prior JSON without breaking. **NOT an accuracy leaderboard.**

## R2->R3 — `.audit.npz` sidecar
`cmi/eval/audit_npz.py`: per-fold export of `graph_z, node_z, y, d, model_logits, probe_logits,
probe_predictions, node_leakage_map` (+ fold/seed/target/method/dataset) so R3 (subspace removal, node masking,
topomaps) can start immediately. Save/load/validate + tests.

## Tests (CPU, pass): registry valid + scoped; all 7 methods run on the one adapter; firewall (within-label
perm preserves Y, permutes D within label); Pareto domination + frontier + dummy-JSON; audit.npz round-trip +
validation.

## Gated GPU (NOT launched — PM will pick the first real-EEG gate)
Initial gate (per PM): 2a + 2015, methods ERM / CIGL graph+node / DANN / cond-DANN / CDAN, **full-LOSO seed0
first** (no method-level judgment). Primary question: does CMI-driven CIGL improve the leakage/task Pareto
frontier on real EEG? Expand to seeds 1/2 only on a scientifically meaningful seed0 signal. **10-seed is the
final confirmatory stage, not next.** No λ-curve / full zoo / P10 expansion until the next checkpoint.
