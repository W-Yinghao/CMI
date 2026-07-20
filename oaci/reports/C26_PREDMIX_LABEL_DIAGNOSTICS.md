# C26 Q5 — predicted-class-mix label diagnostics (LABEL-DIAGNOSTIC-ONLY)

> predmix deviates from the true (balanced) class prior and tracks per-class recall -> reflects the frozen model's target decision-boundary occupancy / error geometry, not the true label prior itself

- predmix vs true class prior corr n/a
- predmix vs per-class recall corr +0.881
- mix distance from true (balanced) prior +0.373
- tracks target error geometry: True
