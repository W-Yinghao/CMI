# Track B — Subspace candidate-span stability audit (FROZEN spec; replaces the revoked FFW full-EEG matrix)

PM strategic pivot: the "learn what to keep/delete" idea is kept, but as a low-rank **subspace in
representation space**, never weights/neurons. Internal name for the method line = **Subspace Ticket
Selection** (L1 subset on a fixed candidate span → L2 learned oblique rotation `P=UUᵀ`, `U⊆span(B)`, rank≤3, in
the source-whitened metric with the task-head functional anchor → L3 Train-Through-Erasure). The FFW
weight/neuron line is STOPPED; its pilot is archived as a coordinate-mask negative control
(`notes/FFW_ARCHIVE_COORDINATE_MASK_NEGATIVE_CONTROL.md`), NOT extended to a 63-cell matrix.

Track B **prepares** the subspace method by characterizing the candidate span on real EEG — it does NOT learn
any mask, train any selector, or make a method GO/NO-GO. It runs in parallel with F2.1a (Track A). Manuscript
FROZEN.

## Base models (only two anchors; no TSMNet / foundation / new backbones)
- **Primary: EEGNet compact latent** (d=16): near-exhaustive low-rank subspaces, session-aware cal/query, cheap
  multi-seed. No stored replayable head → task-head anchor uses a fresh source-fit linear head.
- **Secondary: DGCNN graph representation**: has the stored replayable task head → exact-head nullspace is
  well-defined; consistent with the CMI measurement chain; lets us separate head-null vs contested subspaces.

## Candidate span (source-only; source-whitened metric)
Work in the whitened coordinates `Z̃ = Σ_s^{-1/2}(Z − μ_s)` (kills coordinate scale / variance dominance).
Build three source-only bases, then their orthonormalized union `B = orth[B_cond, B_rule, B_grad]`:
- `B_cond`: label-conditional subject directions (μ_{d,y} − μ_y), SVD.
- `B_rule`: per-subject decision-rule disagreement (class-centered head Wₐ,c − W̄c), SVD.
- `B_grad`: per-subject task-gradient disagreement (mean representation gradient g_d − ḡ), SVD.
The exact-head-null basis is kept SEPARATE as the safe-cleaning control; it is NOT mixed into the DG span.

## Reported quantities (full LOSO × 3 seeds, subject/fold-cluster CI; NO mask learning)
Per (dataset, backbone, fold, seed):
1. **span rank** of each basis and of the union (effective rank at tol).
2. **principal angles** between the three bases (are cond/rule/grad the same subspace or complementary?).
3. **fold/seed stability**: principal-angle / Jaccard similarity of each basis across folds and seeds.
4. **task-head overlap**: fraction of each basis inside row(W_c) (contested) vs ker(W_c) (free); how much of the
   span is functionally used.
5. **CMI captured**: posterior-KL leakage removed by deleting the top-k (k≤3) of each basis (vs matched-rank
   ambient random) — screening only.
6. **target-hindsight utility ceiling**: cross-fitted greedy target oracle Δ (select on T_select, score on
   T_query) per basis — the existence upper bound (reuses the F2.1a hindsight machinery).

## Deliverable
A read-only characterization table + cluster-CI intervals answering: *is there a small, stable, functionally-
anchored subspace whose deletion has a hindsight utility ceiling > 0, and is it source-whitened-stable across
folds/seeds?* This gates whether L2 (learned oblique selector) is even worth attempting. If the span is
unstable or the hindsight ceiling is ~0 on both backbones, the subspace line closes at Track B (no L2/L3).

## Firewall & discipline
Source-only basis + mask/rank selection; target labels only in final hindsight scoring. Full EEG only for any
characterization claim; synthetic/smoke = engineering. No learned selector, no TTE, no manuscript. Reports use
the standard provenance/completeness/firewall/negative-control format.
