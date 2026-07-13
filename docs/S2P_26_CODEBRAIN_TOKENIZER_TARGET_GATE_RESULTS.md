# S2P_26 - CodeBrain Tokenizer Target Gate Results

**Status:** CLOSED / FAIL. Bounded CodeBrain Stage-2 training remains prohibited.

## Authority

```text
S2P commit: b933029a9301107a8edd82feaf73f726b909c085
CodeBrain source commit: 071f8d7a45dfe57d5a26c12b73549afa3c24386a
tokenizer SHA256: e9560b670d64ea4712fd99a48dc2131326b919744c7d3eb504cf57b1ef3af999
tokenizer role: released frozen target generator
```

The scheduler stability gate passed 30/30 probes, and a submit/query/cancel canary passed before this preflight.
Only the frozen tokenizer target-distribution and native shape canary ran. No optimizer, backward pass, trainer,
Stage-2 checkpoint, downstream probe, or fine-tuning job was constructed.

The native tokenizer loader calls `model.load_state_dict(weights)` without overriding PyTorch's default
`strict=True`. Model construction completed against the pinned checkpoint, so missing or unexpected model keys did
not occur. The source-file hash and this loader contract are recorded separately in the result package.

## Passed contracts

```text
source provenance: PASS
H200/H1000/H2000 nested exact-window metadata: PASS
FACED / SEED-V / ISRUC_S3 asset contracts: PASS
released tokenizer frozen: PASS
tokenizer deterministic canary: PASS
native input shape [8,19,30,200]: PASS
native mask shape [8,19,30]: PASS
temporal/frequency target shape [8,570]: PASS
temporal/frequency logit shape [2279,4096]: PASS
finite temporal CE: PASS (7.675994873046875)
finite frequency CE: PASS (8.465712547302246)
target-label firewall: PASS
```

## Target utilization

Each stratum contains 256 independently sampled TUEG windows and 145,920 token positions per stream.

| Stratum | Stream | Unique codes | Effective perplexity | Dominant fraction | Unique sequences | Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| H200 | temporal | 2 | 1.000315 | 0.999973 | 2 | FAIL |
| H1000 increment | temporal | 6 | 1.006710 | 0.999212 | 8 | FAIL |
| H2000 increment | temporal | 2 | 1.000920 | 0.999911 | 5 | FAIL |
| pretrain validation | temporal | 1 | 1.000000 | 1.000000 | 1 | FAIL |
| H200 | frequency | 4069 | 2652.703 | 0.006853 | 256 | PASS |
| H1000 increment | frequency | 4067 | 2608.114 | 0.008176 | 256 | PASS |
| H2000 increment | frequency | 4063 | 2613.590 | 0.007436 | 256 | PASS |
| pretrain validation | frequency | 4064 | 2517.421 | 0.006860 | 256 | PASS |

The temporal stream fails every frozen non-degeneracy threshold in every stratum. The frequency stream passes every
threshold in every stratum. This is a dual-target compatibility failure, not a target-shape, finite-loss, asset,
checkpoint-load, or scheduler failure.

## Decision

```text
tokenizer_target_gate_pass: false
temporal_target_non_degenerate: false
frequency_target_non_degenerate: true
native_shape_canary_pass: true
scientific_preflight_pass: false
launch_bounded_stage2: false
```

The bounded CodeBrain test stops here under the frozen protocol. It must not continue as an effective
frequency-only Stage-2 experiment, and no threshold or normalization may be changed using this result.

This result licenses only the following claim:

> The released TFDual tokenizer's temporal target stream is effectively collapsed on the current processed TUEG
> 19-common substrate under the frozen native amplitude path, while its frequency target stream remains diverse.

It does not establish that CodeBrain's original raw-corpus pipeline, a retrained tokenizer, or CodeBrain in general
has a collapsed temporal target stream.
