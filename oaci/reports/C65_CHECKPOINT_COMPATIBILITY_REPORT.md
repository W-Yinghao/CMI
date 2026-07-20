# C65 - Checkpoint Compatibility Report

Checkpoint ABI code and sidecar metadata are present. Sidecars show ShallowConvNet-compatible state keys/shapes including classifier `[4,800]` and BatchNorm buffers.

C65 did not load real checkpoint state_dicts and did not run EEG forward passes.
