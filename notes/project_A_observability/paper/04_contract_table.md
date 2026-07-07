# Contract Taxonomy (C1–C12)

Canonical registry from `02_contract_taxonomy.md` (the authoritative source; this is a paper-facing
condensation). "Checkable" columns mark whether the contract is data-supported in each regime
(✓ checkable · ~ partial · ✗ assumed-only). Every contract carries a failure certificate that fires
when it is violated.

| ID | Name | Needed for | R0 | R1 | R2 | Failure certificate | Overclaim it blocks |
|---|---|---|:--:|:--:|:--:|---|---|
| **C1** | class support overlap `supp p_T(z\|y) ⊆ supp p_S(z\|y)` | TU-1, MP-1, risk transfer | ✗ | ~ | ✓ | **CE-C1-1** | target risk/prior transfer off source support |
| **C2** | shared class-conditional geometry `p_T(z\|y)=p_ref(z\|y)` | TU-1, PD-1 residual | ✗ | ✗ | ✓ | **CE-R1-1** | prior/concept change read as invariant |
| **C3** | mixture full-column-rank `B` (confusion invertible) | **TU-1** | ✓ | ✓ | ✓ | **CE-R1-2** | prior "identified" under rank-deficient mixture |
| **C4** | stable label mechanism (no target concept shift) `p_T(Y\|X)=p_S(Y\|X)` | target-risk transfer, concept reading | ✗ | ✗ | ~ | **CE-R1-1** | concept shift claimed absent from unlabeled target |
| **C5** | critic/estimator sufficiency `q_ψ→p(D\|Z,Y)` | measured = population leakage | ~ | ~ | ~ | **P0-2** | posterior-KL treated as a CMI upper bound |
| **C6** | representation sufficiency `I(Y;X)=I(Y;Z)` (span ≥2 classes/domain) | interpreting `I(Y;D\|Z)` as concept | ~ | ~ | ~ | **P0-4** | Z-insufficiency read as concept shift |
| **C7** | reference-prior / GLS reweighting `w_d(y)=π*(y)/π_d(y)` | **PD-1** | src ✓ / tgt ✗ | tgt via TU-1 | ✓ | **CE-R1-2** | source-side prior decoupling read as target prior |
| **C8** | low-dim invertible transport (near-identity affine) | **MP-1** | ✗ | weak | ✓ | **CE-MP-1** | transport "identified" from too few anchors |
| **C9** | source→target safety transfer (inner-LOSO gain transfers) | safety gate | src proxy | weak | calib. w/ labels | **CE-R0-2 / TOS-1** | source gain read as target safety |
| **C10** | zero-Bayes / `D⊥Y\|Z` escape (H(Y\|Z)=0 sufficient **not** necessary) | legacy escape | src diag | src diag | anchored | **P0-3** | "joint CMI=0 ⇔ zero Bayes error" |
| **C11** | anchor validity (no fake pairing / label leakage) | **MP-1** | ✗ | ✗ | ~ | **CE-C11-1** | fake pairing read as true transport |
| **C12** | domain-factor separability (`determines_label` correct) | OACI/CSC invariance legitimacy | src `D_j⇒Y` test | +tgt metadata | anchored | **P0-4** | `D=subject⇒Y=g(D)` invariance illusion |

## The three "concept" preconditions (do not conflate)

- **C2** — shared class-conditional *feature geometry* `p_T(z\|y)=p_S(z\|y)`.
- **C4** — stable *label mechanism* `p_T(Y\|X)=p_S(Y\|X)`.
- **C6** — encoder *sufficiency* `I(Y;X)=I(Y;Z)`; a lossy `Z` inflates `I(Y;D\|Z)` even under C2∧C4.

Only under **C2 ∧ C4 ∧ C6 ∧ C5** is a positive `I(Y;D\|Z)` reading admissible; otherwise it stays a
P0-4 predictive-insufficiency diagnostic.

## Monotone checkability

Checkability is monotone R0 ⊑ R1 ⊑ R2 (MONO-1): a contract checkable in a coarser regime stays
checkable in a finer one. The audit registry encodes this and self-tests it.
