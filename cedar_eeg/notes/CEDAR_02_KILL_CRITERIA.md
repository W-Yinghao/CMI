# CEDAR Kill Criteria

## P0 Continue

- leakage drop is at least 30 percent;
- source balanced-accuracy drop is at most one point;
- target balanced-accuracy drop, if evaluated, is at most one point;
- R3 does not increase;
- matched random-subspace control is approximately zero.

## P0 Redirect

If leakage drops but target/R3 do not improve, continue only as
privacy-preserving compression or leakage-atlas diagnostics. Do not make a
generalization claim.

## P0 Stop

If leakage reduction requires source or target collapse, stop structured
pruning and report a negative diagnostic.

## P1/P2

P1 structured pruning and P2 TTA preconditioning are blocked until P0 has a
reviewed accepted candidate.
