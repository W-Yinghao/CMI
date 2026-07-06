# W0.4 — proposed manuscript inserts (STAGED, NOT APPLIED)

Based on `W0.4_RESULTS.md` (75 subjects, QC green, Δ P_J = −0.0445 significant). **Appendix-first with a
compact main-text bridge** — W0.3/W0.5 remain the conceptual pair; W0.4 answers the "small-batch artifact"
reviewer objection. **Do not edit the manuscript source until all Wave-0 inserts are reviewed together.**

**Wording guards (enforced):**
- Do **not** say "larger batches better identify prevalence" or "π_J converges to ρ_A" — `TV(π_J,ρ_A_full)`
  does not support that (flat/slightly increasing).
- Use "larger batches de-regularize / sharpen the fitted prior" and "the BA decision effect moves toward
  the oracle-prevalence penalty."

## Main-text bridge (one paragraph, after the W0.3/W0.5 mechanism)

```latex
A batch-size audit rules out the explanation that the W2 decision-prior harm is a small-adaptation-batch artifact. Increasing the adaptation batch from \(n=16\) to \(n=256\) made the fitted-prior branch more negative:
\(P_J\) changed from \(-0.092\) to \(-0.137\), with endpoint contrast
\(-0.0445\) and CI \([-0.054,-0.036]\). The balanced-accuracy metric-mismatch anchor was invariant at \(-0.162\); the change came from shrinkage of the positive \(\pi_J\)-deviation offset, from \(+0.080\) to \(+0.030\). Thus small batches partially mask, rather than create, the prior-branch harm.
```

## Appendix caveat sentence (with the by-n table + figure)

```latex
This should not be interpreted as pointwise convergence of \(\pi_J\) to the full adaptation prevalence: \(TV(\pi_J,\rho_{A,\mathrm{full}})\) was flat or slightly increasing. The batch-size effect is instead a decision-effect and sharpening result: entropy decreased from \(1.442\) to \(1.188\), and the minimum fitted prior mass decreased from \(0.085\) to \(0.014\).
```

## Figure / table placement (appendix)

- **Appendix figure (compact):** x-axis `n` spaced by `log2(n)`; three curves — `P_J`, the constant
  metric-mismatch anchor, the `π_J`-deviation offset.
- **Appendix table:** `H(π_J)`, `min π_J`, `TV(π_J,ρ_A_full)`, missing-stage rate by `n` (see `w04_by_n.csv`).
- **Main text:** only the endpoint contrast + the one-paragraph bridge, unless page budget allows a small
  inset.

## Wave-0 conceptual ordering (for the Discussion)

> W0.3: oracle prevalence is the wrong BA decision prior.
> W0.5: prevalence-aware decisions can help ordinary accuracy only under the right metric/specification.
> W0.4: more adaptation data does not fix the BA prior mismatch; it can expose it.

## Supersession / cross-reference

- Strengthens the existing rule (BA fixes the decision prior to uniform; fitted-prior branches are
  diagnostic). W0.4 adds: **more adaptation data does not cure the balanced-metric prior harm.**
- Note the transfer/noise term `B_E(ρ_A^draw)−B_E(ρ_E)` (≈ −0.005..−0.010) is included in the appendix
  decomposition table so the rows sum to `P_J` (residual 5.6e-17).
