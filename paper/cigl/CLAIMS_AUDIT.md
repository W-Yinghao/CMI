# CIGL Claims Audit (Phase 4E / v0.3)

> Every manuscript claim, its evidence source, the allowed wording, the forbidden wording, and status.
> Bounded language per `docs/CIGL_32`. "Status": READY (evidence in hand) / WORDING (phrase carefully) /
> NEEDS-TABLE (depends on a generated/plotted table).

| Claim | Evidence source | Allowed wording | Forbidden wording | Status |
|---|---|---|---|---|
| Graph/node leakage **exists** on a task-capable backbone | CIGL_25 (3A-H): graph ≈8× / node ≈15× perm, p=0.020, 3/3 seeds, node-map corr 0.945 | "significant label-conditional domain leakage in graph_z/node_z, clearing a retrained within-label permutation null" | "the representation *is* the subject ID"; "unbounded leakage" | READY |
| Regularizer **reduces** leakage | CIGL_29 (2a: graph −35..58%, node −31..45%), CIGL_31 (2015: graph −43..77%, node −37..61%) | "partially reduces graph/node leakage (~40–65%)" | "eliminates / removes leakage"; "drives leakage to zero" | READY |
| Reduction is **partial, not elimination** | CIGL_29/31: regularized leakage still `clears_null` every fold | "partial; regularized leakage still clears the null" | "fully controlled"; "leakage-free" | READY |
| **Task retained** under the regularizer | CIGL_29 (src drop ≤0.02 primary folds), CIGL_31 (src retained 11/12; target guardrail 12/12) | "meets the pre-specified source-task retention gate; one BNCI2015_001 fold misses the per-fold retention threshold but the dataset-level gate passes" | "without harming performance"; "improves accuracy"; "no cost ever"; ignore fold9 miss | READY |
| **Two-dataset** confirmation | CIGL_29 (BNCI2014_001), CIGL_31 (BNCI2015_001) | "confirmed on two MI datasets, source-only, fixed λ" | "generalizes across datasets/paradigms"; "universal" | READY |
| Metric is a **posterior-KL proxy** | CIGL_32; `audit_graph_node_objects` | "posterior-KL plug-in proxy for label-conditional domain leakage" | "unbiased CMI estimator"; "information-theoretic guarantee" | READY |
| **No edge-CMI** | CIGL_23 (3A-G: dynamic-edge overfit), DGCNN static adjacency (edge_logits=None) | "graph/node only; static adjacency; edge object absent" | "edge-CMI works"; "dynamic-edge CIGL"; "I(A;D\|Y) controlled" | READY |
| **No SOTA** | CIGL_32; ERM baselines 2a ≈0.46, 2015 ≈0.70 | "leakage reduction at task retention, not a leaderboard result" | "state-of-the-art"; "best accuracy" | READY |
| **Source-only firewall** | CIGL_36; firewall flags + target-corruption tests | "target labels evaluation-only; selection/confirmation source-only" | "target-informed"; "uses target labels for selection" | READY |
| **Negative results** are evidence | CIGL_18 (3A-R), CIGL_21 (3A-S), CIGL_23 (3A-G) | "GraphCMINet failure, dynamic-edge overfitting reported as method-shaping evidence" | hide/omit negatives; "everything worked" | READY |
| **Limitations** stated | CIGL_32/35 §7 | "proxy, partial, two MI datasets, one backbone, one λ, modest baselines" | downplay or omit limitations | READY |
| **Dynamic-edge** overfitting is *consistent with* a subject-fingerprint risk but **not causal proof** | CIGL_23 (3A-G) | "the tested dynamic-adjacency designs overfit and carry leakage; consistent with `A(x)` as a fingerprint channel, but we do not causally isolate `A(x)`" | "`A(x)` is *the* cause"; "dynamic edge is task-harmful"; "edge-CMI fails because of A(x)" | READY |
| **Source-task retention is gate-based**, not zero-cost in every fold | CIGL_29 (9/9 ≤0.02), CIGL_31 (11/12; fold9 +0.024) | "meets the pre-registered retention gate (≤0.02 drop) in the large majority of folds" | the "zero-cost"/"no-task-cost" phrasing; "never costs accuracy"; hide fold9 | READY |

## Global wording rules

- Never write "eliminates", "removes", or "leakage-free" — only "partially reduces".
- Never write "CMI" without "proxy/plug-in"; never "unbiased".
- Never write "edge-CMI works" — only "out of scope; static adjacency; future work".
- Never write "SOTA"/"best"/"outperforms"; the result is leakage-reduction-at-retention.
- Always note: target labels evaluation-only; λ fixed (no grid); one backbone; two MI datasets.
- Report fold9 (BNCI2015_001) source-retention miss honestly (11/12), not silently.
