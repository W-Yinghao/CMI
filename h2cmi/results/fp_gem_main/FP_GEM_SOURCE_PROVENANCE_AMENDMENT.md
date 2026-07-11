# FP-GEM Source-Provenance Amendment

Status: **FROZEN BEFORE REPLACEMENT SMOKE; NO TARGET PERFORMANCE OBSERVED**.

## Trigger

P9 persisted source-state hashes but no TSMNet checkpoint files. Job `893416` reproduced the exact P9 BNCI2014-001 target-1 seed-0 training configuration on a Tesla V100. Its epoch 0 and 10 trace matched the committed P9 stdout to printed precision; epoch 19 differed by only `0.0002` in printed loss, while the full byte-level state SHA differed. The runner stopped before RCT, GEM, evaluation-label access, or any target metric. Job `893415` was an earlier zero-result workdir visibility failure.

## Frozen Correction

For each dataset x target x seed unit, P12 reproduces the exact P9 source-training configuration on the recorded GPU family, persists the actual checkpoint, and runs source-only TSMNet, RCT, SPDIM geodesic, SPDIM bias, Joint-GEM, and FP-GEM from that same state. The four official controls are unchanged methods with unchanged P9 adaptation settings; rerunning them is required because the committed P9 checkpoint weights cannot be loaded. Direct P9 row reuse is therefore zero.

No dataset, split, source seed, source-training setting, adaptation hyperparameter, FP-GEM definition, Joint-GEM definition, bootstrap rule, or interpretation boundary changed.
