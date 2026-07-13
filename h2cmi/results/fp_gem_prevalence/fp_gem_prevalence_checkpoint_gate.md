# FP-GEM Prevalence Checkpoint-Reuse Gate

- status: `PASS`
- job: `894726`
- final job state: absent from `squeue`
- unit: `Lee2019_MI`, target 1, source seed 0
- performance metrics computed: `false`
- evaluation labels accessed: `false`
- target labels passed to adaptation: `false`
- fresh source training performed: `false`
- frozen fleet approved: `true`

## Exact Reproduction

- checkpoint file SHA-256: `8ae4e3d0dd15dbe22a6133e36f4672ac6ebec4923f4239feae95c7916548d36d`
- loaded model-state SHA-256: `9d088de5208bfb276919392d1167804faa89eab84b30b7053a96351251bf2d1d`
- source-density state SHA-256: `62dd6b7b2713f2fc222c271b29210d7a966c87a81fed621b8f1a7361dd8ad3a2`
- source-only prediction SHA-256: `eef2221a3c8a95b2631ab98ac334624163144564ae89a3cb143844fd409d8964`
- source-only logits SHA-256: `1c9035576ae7389fc04d7803b75c038893f3302f2c5b56d99b7780faabd4ed78`
- all six q=0.5 prediction/logits references match: `true`
- both Joint-GEM and FP-GEM geometry references match: `true`
- pre-classifier replay maximum absolute error: `0.0`

The official SPD modules initialize train/test running buffers with shared storage. The loader separated 216 registered running buffers before copying the persisted state. This is a state-loading repair, not a model or method change; the resulting state, prediction, logits, density, and geometry hashes all reproduce P12.

## Runtime And Artifacts

- actual GPU: `Tesla V100-PCIE-32GB`, compute capability `7.0`
- interpreter: `/home/infres/yinwang/anaconda3/envs/icml/bin/python`
- Python: `3.9.25`
- PyTorch: `2.8.0+cu128`
- PyTorch CUDA: `12.8`
- MOABB: `1.2.0`
- MNE: `1.8.0`
- launch commit: `7b48813e1430a700a8246604e9def58772e09a4f`
- clean worktree: `true`
- elapsed: `1835.9236` seconds
- gate JSON SHA-256: `4c8ef4631d6bcfe09d2d13443238c146b206f6bd1c07c8933b72be855f8b1535`
- stdout SHA-256: `5f46999284fd88b2fed62c46d1615384319fd8423d25b440a9f464905ebb50e8`
- stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- stderr status: `empty`

The independent red-team verdict is `PASS_FOR_FROZEN_FLEET`. This gate approves only the already frozen P13 fleet; it does not approve parameter changes or a performance claim.
