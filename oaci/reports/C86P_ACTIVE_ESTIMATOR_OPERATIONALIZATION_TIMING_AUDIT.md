# C86P Active-Estimator Operationalization Timing Audit

The main C86P protocol was committed before implementation. During subsequent
shadow-contract review, the fixed total-query budget exposed a semantic edge:
at budget 4, uniform or active sampling can observe only one class. Treating
that event as a failed balanced-accuracy denominator would make complete
2,048-chain coverage effectively unattainable.

No EEG, label, candidate output, registered synthetic stream, or C86 scientific
result had been opened. This additive protocol therefore fixes a symmetric,
outcome-independent Jeffreys `1/2` smoothing rule before the implementation is
committed. It also removes ambiguity in the A2/A3 candidate sets and fixes the
A4 NLL-based plausible-set diagnostic. None of these objects changes the held
evaluation metric, target population, query grid, primary comparator,
inference threshold, taxonomy, or claim boundary.
