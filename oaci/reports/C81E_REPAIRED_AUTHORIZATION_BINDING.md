# C81E Repaired Authorization Binding

The PI directly stated:

```text
授权修复后的 C81E
```

Under policy commit `3d9dd76`, no token or repeated hash recital is required.
The statement is bound to the unique repaired execution objects:

```text
base protocol:         16a0d2e / cbdb42f54956b685c27a1718c37d7c56c513084817a5c69fb29f06bfb67ad3ee
repair protocol:       6371b22 / ba0434b4ea7965691dafaf506547af64f851c57bdca330a0a5c88e4fa7ba1b15
implementation:        5703163 / b922139e77265d11135982d494832e63204c22ff2a29b7619055ad4259463f02
final repaired lock:   bad8db4 / 3093201d3f2959d828cb9debb8a4aeb9252f5385b9e8f806445cff05307a8b1c
manifest digest:       6180275dcef26bdda4ae4b291d1ef6dc83434462ecacee0350fa94ae9c6a7fef
```

The initial authorization bound to `541651c` is retained in history but is not
used by the repaired runtime. Failed selection attempt `894878` remains
preserved and produced zero selection manifests, evaluation-label reads,
baseline statistics, oracle accesses, or target-4 primary rows.

This authorization covers only the repaired C81E existing-field baseline
comparison. It does not authorize training, forward/re-inference, GPU, target 4
primary use, same-label oracle, seed 5, BNCI2014_004, active acquisition, new
methods/features/kernels/models, C82, or manuscript experiments.
