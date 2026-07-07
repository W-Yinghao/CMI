# §3 Synthetic certification — why the safeguards are necessary (draft)

*Organizing figure: Fig 2. Framing: certification discipline, not a positive default-on result.*
Claim tags: [C1]–[C3] in [claim_evidence_table.md](claim_evidence_table.md). Keep this section short — it
justifies the safeguards that make §4 credible.

## 3.0 Opening
Before applying TOS-CMI to EEG, we construct synthetic controls in which the true leakage and task
information are known (Gaussian-mixture generators with a computable Bayes oracle). These controls are not
designed to maximize benchmark accuracy; they are designed to test whether each safety layer prevents a
specific failure mode — first-moment blindness, geometry-only deletion, weak conditional task gates, and
underpowered certification. The main outcome of this line is not default deletion, but **calibrated refusal
when deletion cannot be certified**.

## 3.1 Why geometry alone is not enough
A selective-invariance method that only deletes "domain-rich" directions implicitly assumes (i) that those
directions can be *found*, and (ii) that removing them is *safe* for the task. Neither is automatic: a
direction can be domain-rich yet still carry conditional task information, and a deletion that is exactly
task-preserving *in the geometry* (R T = T) can still raise conditional task risk. We therefore treat
localization and deletion as steps that must be **measured and certified**, and use the synthetic suite to
show that each unguarded step has a concrete failure mode.

## 3.2 Score-Fisher detects leakage missed by first-moment scatter
The candidate subspace is read from the conditional score-Fisher operator (§2.2), which is second-order in
the log-likelihood scores. By construction it responds to covariance- and synergy-structured leakage, where
a first-moment (mean-scatter) statistic is identically blind: on a covariance-only generator the
mean-scatter direction is a no-op while the score-Fisher direction recovers the leakage subspace [C1]. This
is why localization uses the score-Fisher operator rather than class-mean differences. *(Established on the
covariance-only synthetic family; reported qualitatively, not as a Fig 2 panel.)*

## 3.3 Direct-sum deletion can still be conditionally unsafe
On a synergy ("explaining-away") generator, the domain-rich subspace is task-light by first- and
second-moment geometry, so a direct-sum projector deletes it without any *apparent* task cost — yet the
Bayes oracle shows the deletion removes real conditional task information (Δ_Y > 0). Geometry alone would
accept this deletion. This is the concrete sense in which **direct-sum geometry is necessary but not
sufficient for conditional task safety** [C2], and it motivates an explicit conditional task-risk gate.

## 3.4 Weak task gates unsafe-accept; log-ratio gates reduce the failure
A naive task gate — a small nested critic estimating the conditional task risk — *unsafe-accepts* such
deletions: its probe task-risk upper bound sits far below the true Bayes gap, so the deletion passes (Fig
2B). Replacing it with a cross-fitted one-step plug-in log-ratio estimator shrinks the gap between the
estimated bound and the oracle, and the gap continues to close with calibration sample size (Fig 2C); the
unsafe-accept occurs only in the small-sample, weak-critic regime [C3]. The estimator is the bottleneck —
not the geometry — so the gate must use a calibrated estimator, not a convenience critic.

## 3.5 Power certification turns unsafe deletion into refusal
Even a calibrated estimator can be *underpowered*: at a given sample size it may be unable to distinguish a
safe deletion from an unsafe one. We therefore attach a power certificate (a minimum-detectable-effect
floor) and require both task safety (an upper bound on Δ_Y) and a domain-gain lower bound before deleting.
With this in place the certified gate produces **zero unsafe-accepts** across the synthetic phase diagram,
but is deliberately conservative — it abstains or rejects in most cells (Fig 2D). Abstention is the intended
behavior when deletion cannot be certified, and is reported as such (not as a missing result).

## 3.6 Takeaway
> The synthetic suite does not show that selective deletion should be enabled by default; it shows why a
> selective-invariance method needs explicit measurement, conditional task-risk testing, and abstention.

This is the discipline we carry into the EEG study (§4): localize, test conditional task risk, and refuse
when a safe, useful deletion cannot be certified.
