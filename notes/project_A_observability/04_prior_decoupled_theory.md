# Project A — Prior-Decoupled Theory

> Prior shift must be decoupled from conditional transport **before** any conditional-domain-
> leakage number is interpreted. This file is **information accounting, not a target-accuracy
> theorem.** It imports the safe algebra from the legacy pillars (`notes/theory/01_tension.md`,
> `02_resolution.md`) and re-states its *semantics* through the authoritative `h2cmi/THEORY.md`
> corrections (P0-2…P0-5). Naming per `00_repo_audit.md §5`.

## 0. Purpose

This file imports and connects:
- **`ID-1`** — the legacy chain-rule identity;
- **`PD-1`** — the reference-prior additive relation (what `02_resolution.md` calls Corollary 2.2 / the (★) relation);
- **`TU-1`** — target-prior identifiability under the mixture / full-rank contract (`C1∧C2∧C3`);
- **`CE-R1-2`** — the failure certificate when full-rank (`C3`) fails.

It produces **no new theorem**. It extracts the *safe* algebra and quarantines the retracted
overclaims.

## 1. Legacy boundary (what is safe to import, what is not)

**Safe (pure algebra, assumption-light — import as-is):**
```
ID-1:   I(Z;D|Y) − I(Y;D|Z) = I(Z;D) − I(Y;D)
```

**Unsafe (retracted by `h2cmi/THEORY.md`; do NOT re-assert):**
- the posterior-KL **upper-bound** claim on `I(Z;D|Y)` — **retracted by P0-2** (`R=E KL(q_ψ‖π_y)`
  is a plug-in, not a bound: `q_ψ=π_y ⇒ R=0` while true CMI can be `>0`);
- "both CMIs zero ⇔ `Y=f(Z)` / zero-Bayes-error" (legacy A3) — **retracted by P0-3** (correct
  condition is `D⊥Y|Z`);
- reading `I(Y;D|Z)` as **certified / genuine concept-shift** — **retracted by P0-4** (it is a
  predictive-insufficiency diagnostic).

`h2cmi/THEORY.md` P0-2/P0-3/P0-4/P0-5 are authoritative wherever the legacy pillars conflict.

## 2. Source-side prior decoupling

All quantities in this section are **source-side** (source labels observed, so every source
`π_d(y)` is known). Let `π_d(y)=P(Y=y|D=d)` and let `π*(y)>0` be a reference prior. Define the
per-sample GLS weight and the reweighted law:
```
w_d(y) = π*(y) / π_d(y),        p̃(z,y,d) ∝ w_d(y) · p(z,y,d).
```

> **Lemma (prior removal).** Assume **class support** `π_d(y)>0` for all `(y,d)` (every class
> observed in every source domain, so `w_d(y)` is finite — the source-side analogue of the
> single-class target failure in §8). Then under `p̃`, `p̃(Y=y|D=d)=π*(y)` for all `d`, hence
> `Y ⟂ D` and `Ĩ(Y;D)=0`.

*Proof.* `p(z,y,d)=p(d)·π_d(y)·p(z|y,d)`, so `w_d(y)·p(z,y,d)=p(d)·π*(y)·p(z|y,d)`; the label
factor `π*(y)` no longer depends on `d`, so `Y⟂D` under `p̃` and `Ĩ(Y;D)=0`. ∎ (This is exactly
Lemma 2.1 of `02_resolution.md`, verified there to `≤8.3e-17`.)

## 3. `PD-1` theorem

Apply `ID-1` under the reweighted law `p̃` and substitute `Ĩ(Y;D)=0`:
```
Ĩ(Z;D|Y) − Ĩ(Y;D|Z) = Ĩ(Z;D) − Ĩ(Y;D) = Ĩ(Z;D)
⟹   Ĩ(Z;D|Y) = Ĩ(Y;D|Z) + Ĩ(Z;D)          (PD-1, all terms ≥ 0)
```

> **`PD-1` (prior-decoupled additive relation).** After prior decoupling (`Ĩ(Y;D)=0`), the three
> information terms are non-negative and additive: the conditional **encoder** leakage
> `Ĩ(Z;D|Y)` upper-bounds **both** the decoder residual `Ĩ(Y;D|Z)` and the marginal domain
> leakage `Ĩ(Z;D)` under `p̃`. Consequently driving `Ĩ(Z;D|Y)→0` forces both other terms to `0`
> — no forced encoder-vs-decoder trade-off (contrast the pre-decoupling regime, where
> `ID-1` with `I(Y;D)>0` makes them fight).

**What `PD-1` does NOT say** (state explicitly to avoid re-import of retracted claims):
- it does **not** prove target accuracy;
- it does **not** certify concept shift (`I(Y;D|Z)` remains a P0-4 diagnostic even under `p̃`);
- it is **not** a source-only target-adaptation result — it is a statement about the
  **source** law reweighted to a reference prior (`C7` source-side; see §6).

## 4. Relation to the `h2cmi` implementation (P0-5)

The reweighted **reference domain marginal** must be the GLS-weighted bincount (P0-5), not
`mean_y p(d|y)`:
```
p_ref(d) = normalise( Σ_i w_i · 1[d_i=d] ),     w_i = π*(y_i)/p(y_i|d_i).
```
Marginal alignment is under the fixed reference prior `p_d*(z)=Σ_y π*(y) p_d(z|y)` (align
class-conditionals, never raw `p_d(z)`). Code: `h2cmi/align/reference_marginal.py`
(`gls_weights`, `gls_reference_domain_marginal`, `ReferenceMarginalAlignment`).

## 5. `TU-1` — target-prior identifiability under R1

Now move to regime **R1** (target-unlabeled): we observe `p_T(z)` for a fixed encoder `f`.

> **`TU-1`.** Under **C1** (support overlap), **C2** (shared class-conditional geometry
> `p_T(z|y)=p_ref(z|y)`), and **C3** (mixture identifiability), the target prior `π_T` is
> identifiable from the mixture `p_T(z)=Σ_y π_T(y) p_ref(z|y)`.

**Discrete statement.** Let `B_{z,y}=p_ref(z|y)`, so `p_T = B π_T` (a `|Z|×|Y|` mixture matrix
times the prior vector).
- If `B` has **full column rank** (`rank B = |Y|`, contract C3), then `B` is injective in `π_T`,
  so `p_T=B π_T` has a unique solution ⇒ **`π_T` identifiable** (equivalently `w=C⁻¹μ` well-posed).
- If `B` is **rank-deficient** (columns linearly dependent — e.g. two classes share a
  class-conditional), then `B π_T = B π_T'` for `π_T ≠ π_T'` ⇒ **`π_T` non-identifiable**. This
  is **`CE-R1-2`** (`07_counterexample_catalog.md §5`, executed in `run_counterexamples.py`:
  rank-1 `B` ⇒ distinct priors give identical `p_T(z)`).

`TU-1` is an **R1** result: it consumes the observed target marginal `p_T(z)`. It is **not** a
source-only target-prior estimator (see §6).

## 6. R0 boundary — the asymmetry that must not be blurred

Under strict **R0** (source-only):
- source-side `π_d(y)` **is** observed ⇒ source-side prior decoupling (§2) and `PD-1` (§3) are
  legitimate **source** operations;
- target `π_T` is **not** observed ⇒ **target prior is not identifiable** (`03_…`, `TOS-1`;
  `CE-R0-3`);
- target adaptation gain is **not** identifiable (`TOS-1`; `CE-R0-2`).

> **Therefore `PD-1` is a source-side information-accounting result, NOT a source-only
> target-prior estimator.** Writing "GLS gives us the target prior" as if source-only is the
> `C7` overclaim the taxonomy (`02_contract_taxonomy.md §3`) exists to block: target-prior
> estimation needs **R1 + C1∧C2∧C3** (`TU-1`), and even then yields an *estimate*, not a
> guarantee.

## 7. P0-4 diagnostic boundary

After reweighting, `Ĩ(Y;D|Z)` is still a **predictive-insufficiency residual**, not certified
concept-shift. It is *evidence* of concept shift **only** under the joint contract:
- **C7** correct reweighting (so label shift is actually removed),
- **C6** sufficient `Z` (`I(Y;X)=I(Y;Z)`; else Z-loss inflates the residual),
- a flexible predictor,
- **C1** support overlap,
- and, to *confirm*, target labels or anchors (**R2**).

Absent these, a positive `Ĩ(Y;D|Z)` is a diagnostic to investigate, not a concept-shift
certificate. Under the clinical degeneracy `D=subject ⇒ Y=g(D)` it collapses to `H(Y|Z)` (P0-4).

## 8. Failure modes

| failure | consequence | certificate |
|---|---|---|
| **C3** rank failure | `π_T` non-identifiable under R1 | `CE-R1-2` |
| single-class target | class-conditional transport non-identifiable | TTA identity-fallback (P0-3 / audit red-flag 6) |
| misspecified `f` | the `p_T(z)=Bπ_T` mixture statement may fail (wrong `B`) | — (encoder-dependence; `01_… §1.1`) |
| poor critic (**C5**) | measured leakage ≠ population leakage | P0-2 |
| finite sample | estimator instability of `w=C⁻¹μ` | statistics layer (`01_… §7`) |

---

**One-line summary.** `PD-1` cleanly separates prior shift from conditional transport **on the
source law**; `TU-1` recovers the target prior **only under R1 + C1∧C2∧C3**; neither is a
target-accuracy or concept-shift certificate, and both are quarantined from the retracted
posterior-KL / zero-Bayes-error / concept-shift overclaims by the P0 corrections.
