# C58 - Synthetic Lower-Bound Derivation Sketch

If two candidates have the same observed source rank `R` but target gauges differ, a source-measurable selector cannot distinguish them. In a balanced two-candidate cell, any source-only selector has hit at most `1/2` while an endpoint oracle has hit `1`.

The finite-population version replaces the synthetic balance assumption with registered empirical partitions. C58 therefore reports `M_G = 1 - H*_G` for observed partitions rather than a universal theorem.

To turn this into a theorem one would need a distribution over target gauges, independence or exchangeability assumptions, and traceable training artifacts showing how `R` and `G_t` are generated. Those assumptions are currently too strong for the committed artifacts.
