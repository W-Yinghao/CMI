# C72 - Theory Note: Multi-Candidate Measurement-to-Control Gap

## Object

For one frozen target, let candidates be c=1,...,M, held-out utility U(c), and a construction measurement S_b(c) from b class-stratified shared trials. Reliability concerns the ordering association between S_b and U. Control concerns whether argmax S_b intersects argmax U. These are different functionals: a global rank statistic averages O(M^2) pairs, while top-1 fails when any candidate crosses the extreme boundary.

## Proposition 1: Common-Offset Invariance

For every scalar a, argmax_c [U(c)+a] = argmax_c U(c). For logits, adding one scalar to every class leaves softmax probabilities exactly unchanged in real arithmetic. C72 checks both identities numerically; a shared class-dependent vector is deliberately excluded from this proposition because nonlinear metrics can reorder candidates.

## Proposition 2: Exact Finite-Population Pair Error

For a candidate pair, paired correctness contrasts on a class stratum take values in {-1,0,1}. If the stratum contains n+, n-, n0 values and b trials are sampled uniformly without replacement, the sampled contrast distribution is multivariate hypergeometric. C72 computes each class-mean contrast mass, convolves the four equally weighted class contributions, and reports P(bAcc contrast<=0). The top-1 lower bound is one minus the Bonferroni sum over the held-out best candidate versus every competitor. This statement is conditional on the frozen construction population and the registered class-stratified sampling design; it does not solve construction-to-evaluation gauge mismatch.

## Proposition 3: Stylized Gaussian Rank-Gauge Bound

Assume U_t(c)=R(c)+W_t(c)+epsilon_t(c), centered candidate-specific W and finite-label epsilon are Gaussian with registered variances, and the source-rank best has pair margin Delta_j over competitor j. Then pair reversal probability is Phi[-Delta_j/sqrt(2(sigma_W^2+sigma_epsilon^2/b))], and the sum over j is a valid union bound under these assumptions. C72 labels this a stylized model bound. It is neither an EEG population theorem nor a minimax lower bound.

## Extreme-Order Consequence

Even when most candidate pairs are ordered correctly, increasing M adds extreme competitors and the minimum top margin contracts. The probability of at least one crossing can therefore rise while Spearman stays high. C72's candidate-count intervention and synthetic grid test this implication directly.

## Empirical Scope

On T3-HO, the median target-universe top-two bAcc gap is `0.012916`. The source-plus-construction residual variance fraction is `0.431578`. Exact finite-population and stylized union bounds are tabulated separately so finite-label sampling uncertainty is not conflated with candidate-specific target gauge.
