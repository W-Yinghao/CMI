# C80E Selection Descriptor Repair Replay

The additive repair protocol `c19ef34` preceded implementation commit
`37e38d0` and replacement lock `0797599`. The direct PI authorization remained
authoritative and was automatically rebound to the single repaired operative
lock without asking the PI for a token or repeated identifiers.

```text
failed job:                       894641
failure before evaluation open:  yes
selection freeze reused:         yes
selection values inspected:      no
evaluation-label reads:          0
evaluation outcomes:             0
repair protocol SHA-256:         03b7a00a06e18c8e7a9d7b42adba5954cebadac681d2eec08bfe07ad1df543ce
repaired adapter SHA-256:         dae306166227ba588c5a74af1b5a009464ee614dfbb0efea76a2a006a21128b7
replacement lock SHA-256:        2149895865bd44b4ab8358c76848bb6774abb59d4a203b261864be0ec599ff62
```

The verifier now binds exact registered fields, dtypes, and shapes. The frozen
selection payload hash remains
`a0bf1ae048afcff5ab78d79ea7a4bd6243ec05dd373d987b98af83994d8beda6`;
it will not be recomputed. No scientific registry entry, selector, RNG stream,
budget, threshold, dependence rule, taxonomy, or report schema changed.

Gate: `C80E_SELECTION_DESCRIPTOR_REPAIR_RELOCKED_RESUME_ALLOWED`.
