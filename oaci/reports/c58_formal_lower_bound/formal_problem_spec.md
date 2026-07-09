# C58 - Formal Problem Specification

Let `C` be the finite candidate set in the frozen audit universe and `Y(c)` the evaluated target joint-good event. Each selector family is constrained to be measurable with respect to a sigma-field `G` in the C58 ladder.

For a registered finite partition `Pi_G`, C58 uses:

```text
H*_G = mean_cell max_block mean_{c in block} Y(c)
M_G = 1 - H*_G
R_G_to_EO = H*_endpoint_oracle - H*_G
```

This is an empirical finite-population statement. It is exact only for the registered partitions and rule families listed in `finite_population_bound_summary.csv`.

Le Cam and Fano templates are recorded as proof attempts, but C58 does not claim distributional worlds, KL/TV control, stable mutual information estimates, or minimax impossibility.
