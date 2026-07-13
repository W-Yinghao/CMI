# C80E Direct PI Authorization Policy

## Operative rule

A direct PI statement in the current execution conversation that explicitly
authorizes a named milestone is sufficient authorization for that milestone's
current unique operative scope.

The PI is not required to repeat a token, protocol hash, lock hash, manifest
digest, or CLI flag. Those identifiers are repository audit objects, not
authorization secrets. The executor resolves and records them automatically
from the single operative lock before restricted access.

## Current authorization

The direct PI statement `我明确授权C80E了` authorizes C80E under the current
operative C80R protocol, analysis lock, and field/view manifest set. The full
message also instructs that direct PI language controls and that no repeated
token is required.

The machine-readable authorization record binds that direct statement to:

```text
protocol commit:  e88a24484590636f87d0f22798401a762875046a
protocol SHA-256: 2d72eb5119056a6520fd33fc0ac14ee6270bfd573b59c36b74be6aa3dc25fe39
analysis lock:    f19acd8775f9b0ddf60401739741bec0019d021c
lock SHA-256:     e18f2b5f1d79b6fcd96207339c5842e30b7aecb5bc22b8939a475487068b1b82
manifest digest:  6180275dcef26bdda4ae4b291d1ef6dc83434462ecacee0350fa94ae9c6a7fef
```

This automatic binding is an audit action, not a second authorization gate.
It does not expand C80E beyond the locked existing-field P1/P2/S1/S2/S3
analysis. Training, forward/re-inference, GPU, target 4 primary use,
same-label-oracle access, BNCI2014_004, seed 5, active acquisition, new
feature/kernel/model search, C81, and manuscript drafting remain outside scope.

If repository state contains no operative lock or more than one competing
operative lock, execution must stop for repository repair or ambiguity. The
executor must not ask the PI for a magic token to compensate for ambiguous
repository state.
