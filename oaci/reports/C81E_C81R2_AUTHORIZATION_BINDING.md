# C81E C81R2 Authorization Binding

The PI directly stated:

```text
授权 C81R2 修复后的 C81E 继续执行
```

Policy commit `3d9dd76` requires no token or repeated hash recital. The server
binds the statement to the unique current objects:

```text
base protocol:                 16a0d2e / cbdb42f54956b685c27a1718c37d7c56c513084817a5c69fb29f06bfb67ad3ee
source-schema repair:          6371b22 / ba0434b4ea7965691dafaf506547af64f851c57bdca330a0a5c88e4fa7ba1b15
selection-descriptor repair:   5062f5a / 2acf6ecc179c739f73845d430f9eac9e9e83a83015370b1125dbe447b8b59272
repaired implementation:       225df1c / d5d8825e9c06994970de87728f73c6c8fef56af0cdd0f734746f1bd4863bf701
C81R2 execution lock:          f82ffa4 / 13414dde0a88eb8a1a0810b3b36f25c718669d4cfe3178b871239eff6e292705
field/view manifest digest:     6180275dcef26bdda4ae4b291d1ef6dc83434462ecacee0350fa94ae9c6a7fef
frozen selection manifest:     4677ed3aba7758ea0008c2093b44d6fb81d425930727e5941950179737ebd519
frozen selection payload:      1ed893acd9190914eb4cb122f3ef26bc1e2355c4103894b816894bd264669257
```

Selection recomputation is forbidden. This authorization covers exact frozen
selection replay, physically disjoint held-evaluation scoring, locked Q1/Q2,
max-T, LOTO, taxonomy, reporting, red-team, and regression only.

It does not authorize training, forward/re-inference, GPU, target 4 primary use,
same-label oracle access, seed 5, BNCI2014_004, active acquisition, new methods,
features, kernels or models, C82, or manuscript experiments.
