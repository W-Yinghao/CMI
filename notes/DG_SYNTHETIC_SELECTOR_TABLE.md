# Synthetic DGP x selector table (P0.5 closeout — resolves the majority-shortcut apparent contradiction)

cond/full basis, spurious-task DGP (n_domains=12, per_domain=250, seed=1, inv 0.5 / spur 2.5 / id 3.0).
The unit test asserts the *nested-prefix* selector REFUSES a majority-sign shortcut; the audit preview
showed the *greedy source* selector RECOVERS a STRONG majority shortcut. Both are correct — different
selectors. Greedy source is strictly more expressive (arbitrary coordinates + no no-harm gate), so it
can exploit a strong majority shortcut that helps the source average; the prefix+no-harm nested rule
cannot. This is exactly why the real-EEG negative is meaningful: the *stronger* greedy source selector
also fails there, so the failure is not a too-weak selector.

| DGP config | greedy target oracle (existence) | nested prefix mean_1SE | nested prefix CVaR25 | greedy source audit |
|---|---|---|---|---|
| majority-sign (nmin=3) | +0.152 (rand -0.160) | k*=0 Δ=+0.000 | k*=0 Δ=+0.000 | k=2 Δ=+0.150 (rand -0.054, align 1.00) |
| balanced-sign (nmin=5) | +0.034 (rand -0.036) | k*=1 Δ=+0.031 | k*=3 Δ=+0.031 | k=2 Δ=+0.032 (rand -0.033, align 0.50) |

Reading: BALANCED shortcut is recovered by BOTH selectors (source-visible instability).
MAJORITY shortcut: nested-prefix REFUSES (correct under its no-harm gate — deletion hurts the source
majority), but greedy-source RECOVERS it because with a STRONG shortcut the mis-signed minority is
hurt enough that deleting improves the source-LOSO average. Neither is a bug. On real EEG neither
selector recovers the confirmed ticket -> genuine source-unobservability, not selector weakness.
