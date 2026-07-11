# S2P_21 - Phase B Immutable Checkpoint Closure

**Status:** provenance-only execution approved. **B1 scientific compute:** held.

## Purpose

This closure converts the seven remaining writable checkpoint sources into content-addressed Phase-B artifacts
and verifies the complete ten-object representation set. It changes no selected epoch, model parameter, metric,
rank, probe, dataset split, or scientific result.

Approved new copies:

```text
released
H200_s0, H200_s1
H500_s0, H500_s1
H1000_s0, H1000_s1
```

Random initialization remains a deterministic code-and-seed contract. H2000_s0/s1 reuse their existing Phase-A
SHA-named payloads directly; their `best.pth` symlinks are not used by B1.

## Content-addressed destination

New payloads are written under:

```text
/home/infres/yinwang/CMI_AAAI_s2p_b1_launch/results/s2p_route_b_phase_b_checkpoints/
  sha256_<FULL_SHA256>.pt
```

The destination is flat and has no tag-based mutable alias. An existing destination may be reused only when its
size and full SHA256 match. Creation uses an exclusive temporary file and an atomic no-overwrite hard link. Every
payload is mode 0444 and the artifact directory is mode 0555 after successful closure. No B1 script may resolve a
source path or a logical symlink.

`chattr +i` is not required and is not attempted. Scientific immutability is the conjunction of a content-addressed
path, no-overwrite creation, a read-only payload, a committed full hash, and runtime hash verification.

## Selection provenance

For each Route-B run, closure verifies the selected checkpoint epoch and validation loss against
`run_summary.json`. Selection remains `pretrain_val_loss_only`; FACED labels and performance are absent from the
closure. The manifest records the run tag, SLURM task, training code commit when present in the checkpoint, config
hash, and full Route-B data-manifest hash.

Training task mapping is frozen as:

```text
H200_s0=890125_0    H200_s1=890147_1
H500_s0=890147_2    H500_s1=890147_3
H1000_s0=890147_4  H1000_s1=890147_5
H2000_s0=890151_6  H2000_s1=890151_7
```

Released CBraMod is recorded as `externally_released_locally_unverified`; its selection epoch, training data,
config, and code provenance remain `NA`. This does not block its use as a path-validity reference and does block
any released-training or equivalence claim.

## Integrity gates

### P1: byte integrity

For every newly copied source:

1. capture source size and mtime;
2. compute two consecutive source SHA256 values;
3. require identical stat signatures and hashes;
4. copy without overwrite;
5. require source size/hash = destination size/hash;
6. hash the source again after copy and require no change.

Existing H2000 payloads are checked against the Phase-A immutable manifest.

### P2: model integrity

Source and immutable objects are independently loaded on CPU. State-dict wrappers and `module.` prefixes are
normalized, then both objects must load into native CBraMod with `strict=True`. Missing and unexpected key counts
must both be zero. Every state tensor must be `torch.equal`, and canonical tensor-state hashes must match.

Random initialization is instantiated twice from torch seed 0. Its source-code hashes, architecture arguments,
torch version, parameter-state hash, and seed form its immutable logical contract. A fresh model must accept that
state with `strict=True`.

### P3: functional integrity

The checksum batch contains 18 source-train FACED samples: subjects 1 and 80, one fixed clip from each of the nine
classes, and a class-index-derived segment. It reads EEG samples but never reads labels. Preprocessing and final
feature extraction exactly match Phase A:

```text
per-channel/per-patch z-score
patch_embedding
encoder
mean over patch dimension
flatten 32 x 200
```

On one GPU and in eval mode, require:

```text
source repeat max_abs_diff = 0
source vs immutable max_abs_diff = 0
source feature SHA256 = immutable feature SHA256
```

There is no tolerance relaxation. Any nonzero difference is NO-GO.

## Outputs

```text
results/s2p_route_b_phase_b_checkpoint_closure/
  phase_b_checkpoint_immutable_manifest.csv
  phase_b_checkpoint_copy_verification.json
  phase_b_checkpoint_reload_verification.csv
  phase_b_feature_equivalence_rerun.csv
  phase_b_provenance_closure.json
  logs/
```

After closure, the metadata-only B0 precheck reruns in `--post-closure` mode. It directly hashes every physical B1
path, rejects symlinks and writable files, checks all ten reload/parameter/feature gates, and updates:

```text
results/s2p_route_b_representation_emergence_b0/
  phase_b0_checkpoint_provenance.csv
  phase_b0_precheck.json
  phase_b0_go_nogo.json
```

## Success condition

Closure PASS requires all ten representation objects to be pinned, all nine physical checkpoint payloads to be
direct and read-only, all strict reloads and parameter comparisons to pass, all feature differences to equal zero,
and no mutable path to be consumable by B1.

A PASS changes `phase_b1_go_recommended` to true but keeps `phase_b1_compute_authorized=false`. B1 still requires a
new PM decision after the committed closure package and final provenance red-team verdict are reviewed.

## Prohibited actions

This job cannot call a trainer, probe, task scorer, geometry analysis, variance decomposition, layer hook,
fine-tuning path, H4000 launcher, CodeBrain run, or manuscript workflow. It does not auto-launch B1.

