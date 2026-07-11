# S2P_22 - Phase B Provenance Closure Results

**Status:** PASS. **Job:** 892980. **B1:** recommended but not authorized.

## Closed scope

The seven approved writable sources were copied without overwrite into direct full-SHA256 paths:

```text
released
H200_s0, H200_s1
H500_s0, H500_s1
H1000_s0, H1000_s1
```

Random initialization remains a deterministic content-addressed code/seed contract. H2000_s0/s1 directly use the
existing Phase-A SHA-named payloads rather than their logical `best.pth` symlinks.

## Integrity result

```text
representation objects:                 10
physical checkpoint payloads:            9
deterministic random contracts:           1
full SHA256 pinned:                    10/10
strict reload:                         10/10
zero missing/unexpected keys:          10/10
parameter tensors exactly equal:       10/10
source repeat feature max_abs = 0:      10/10
source/immutable feature max_abs = 0:   10/10
mutable B1 checkpoint path consumed:    false
target labels read or used:             false
```

The seven new payloads are mode 0444 in a mode-0555 artifact directory and have no symlink aliases. Destination
filenames contain the full content SHA. The copy report records two consecutive stable source hashes, unchanged
source stat signatures, matching destination size/hash, and the no-overwrite creation contract.

## Selection provenance

All eight Route-B checkpoints retain their original `pretrain_val_loss_only` selection, selected epoch, validation
loss, training task, config hash, full data-manifest hash, and checkpoint-recorded code commit. No FACED metric was
used for selection.

Released CBraMod is explicitly recorded as `externally_released_locally_unverified`. Its unknown training
provenance is not repaired or inferred and it remains a path-validity reference only.

## Independent red-team

`route_b_phase_b_provenance_verify.py` independently rehashed all nine physical payloads and cross-checked the
closure manifest, strict-reload table, parameter hashes, feature-equivalence table, copy report, and post-closure
B0 go/no-go. The final verdict is `PASS` with an empty failure list.

## Authority boundary

The post-closure B0 package now reports:

```text
phase_b0_design_pass:                       true
checkpoint_count_immutable:                 10
all_checkpoint_sha256_pinned:               true
all_checkpoint_strict_reload_pass:          true
all_checkpoint_feature_equivalence_pass:    true
immutable_blocking_tags:                    []
mutable_checkpoint_path_used_by_b1:         false
phase_b1_go_recommended:                    true
phase_b1_compute_authorized:                false
```

This closure provides no scientific Phase-B endpoint. It launches no probe, geometry analysis, variance analysis,
training, fine-tuning, H4000, CodeBrain run, new downstream dataset, layer hook, or writing workflow. B1 remains
held until an explicit PM decision.

