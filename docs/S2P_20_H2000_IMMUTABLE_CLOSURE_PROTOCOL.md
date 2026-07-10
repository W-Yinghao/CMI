# S2P_20 - H2000 Immutable Closure Protocol

**Phase:** A. **Status:** FROZEN before H2000 completion. **Training action:** none; existing jobs finish naturally.

## Inputs

```text
training tasks:
  890151_6 = H2000_s0
  890151_7 = H2000_s1

mutable source root:
  /home/infres/yinwang/CMI_AAAI_s2p_b1_launch/results/s2p_route_b_33ch_b1

immutable destination root:
  /home/infres/yinwang/CMI_AAAI_s2p_b1_launch/results/s2p_route_b_33ch_b1_immutable
```

The closure job is submitted with `afterok:890151_6:890151_7`. It is CPU-only and cannot call a trainer or a
downstream script.

## Per-cell closure gate

All conditions are required:

1. both parent jobs are absent from `squeue` when closure executes;
2. `train_log.jsonl`, `run_summary.json`, and `best.pth` exist;
3. log epochs are exactly 1 through 50, finite, with one final done event;
4. summary reports route B, budget 2000 h, matching subset/init seed, 50 epochs, non-smoke, and no target labels;
5. summary reports checkpoint strict reload success and a train/val-disjoint route manifest;
6. `best.pth` epoch and val loss exactly match `best_epoch` and `best_val_loss` selected by pretrain-val loss;
7. CBraMod loads the source checkpoint with `strict=True`;
8. source size, mtime, and SHA256 remain unchanged across validation and copy;
9. destination SHA256 equals source SHA256 and destination strict reload passes;
10. immutable target is SHA-named, read-only, and `best.pth` is a stable symlink to that target.

Any failure writes a NO-GO result and exits non-zero. Existing immutable files are never overwritten with different
content.

## Outputs

Git-tracked metadata:

```text
results/s2p_route_b_h2000_immutable_closure/
  h2000_immutable_checkpoint_manifest.csv
  h2000_immutable_checkpoint_manifest.json
  h2000_immutable_closure_go_nogo.json
  h2000_immutable_closure_job_id.txt
  logs/
```

Untracked immutable model payloads:

```text
s2p_route_b_33ch_b1_immutable/H2000_s{0,1}/
  best.<sha256>.pth
  best.pth -> best.<sha256>.pth
  run_summary.json
  train_log.jsonl
```

The manifest records source/destination path, full SHA256, bytes, checkpoint epoch, best val loss, config hash,
route-manifest hash, selected-subject hash, source git commit, and strict-reload result.

## Downstream unlock

Closure PASS does not itself change a scientific result. After reviewing the closure package, rerun the identical
FACED audit on the immutable paths:

- task Kappa, balanced accuracy, weighted F1;
- L1;
- source-val task gate and L4/L5/L6;
- 5000 paired target-subject bootstrap replicates;
- source-val-energy-matched random null and Holm correction;
- random and released references under the unchanged pipeline.

Only a new full-scope verifier that exactly reproduces the immutable H2000 rows may restore H2000 to the claim
ledger. H4000 and every later phase remain held during closure.
