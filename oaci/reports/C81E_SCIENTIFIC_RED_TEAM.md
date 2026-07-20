# C81E Scientific-Result Red Team

## Result

```text
36 / 36 PASS
final scientific status: BLOCKED
```

The red team replayed blocker result commit `8801b1c` and SHA-256
`e98bc3f1f0228a40508459a8fe5d4dd3bbc4cf646727fb5ff5e7c5c7084e00f7`.

Selection job `894915` remained frozen and was not recomputed. Authorized job
`894958` opened the registered construction/evaluation views after selection
freeze and computed all 32 contexts in memory. It then failed before writing the
first result table because selector/oracle rows and the analytic random-control
rows contain the same eight fields in different dictionary orders. The strict
CSV writer treats key order as schema identity.

The process read 4,746 evaluation-label rows and 4,470 construction-label rows
across the two seed routes. No scientific row was persisted: method-context,
primary comparison, max-T, LOTO, and nonblocker taxonomy outputs are all absent.
No value from the failed process was printed or inspected.

Because evaluation outcomes were read, the locked policy forbids patching and
rerunning under the same protocol identity. Authorization is consumed and the
runtime now fails closed.

Protected boundaries remain intact:

```text
same-label oracle accesses: 0
target4 primary rows:       0
training / forward:         0 / 0
re-inference / GPU:         0 / 0
selection recomputation:    0
```

No C81-A/B/C/D decision is available. The only valid taxonomy is:

```text
C81-E_protocol_input_implementation_or_provenance_blocker
```
