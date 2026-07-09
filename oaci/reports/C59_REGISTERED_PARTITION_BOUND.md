# C59 - Registered Partition Bound

Theorem. Given a finite universe Omega, a registered partition Pi, and a binary label Y, any selector that is constant within Pi-cells has hit at most H*(Pi)=|Omega|^{-1} sum_B max_y n(B,y). Therefore empirical error is at least 1-H*(Pi).

Proof. In each block B, a Pi-measurable selector cannot distinguish candidates with the same Pi atom; its best possible constant decision captures at most the majority label count max_y n(B,y). Summing over blocks and dividing by |Omega| gives the bound.

Limitation: this covers registered partitions and does not cover arbitrary nonlinear source-measurable functions unless the registered partition generates that sigma-field.
