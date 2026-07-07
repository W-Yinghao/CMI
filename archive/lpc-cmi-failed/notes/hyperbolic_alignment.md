# Hyperbolic Alignment (HA) — exploration note

Status: **exploratory** (user-requested, 2026-06-08). EA is the mainline; RA is the principled curved-manifold
variant; HA asks whether an *explicit constant-curvature* embedding adds anything.

## Geometric motivation
Cross-subject EEG shift is largely a **covariance shift**: subject *s* applies a roughly subject-specific
linear mixing, so trial covariances are congruence-translated. Alignment = removing that translation.

- **EA** (He & Wu 2020): recenter with the *arithmetic* mean covariance, `X̃ = R̄^{-1/2} X`. Signal-level, cheap.
- **RA** (Zanini 2018): recenter with the *geometric* (AIRM Fréchet) mean — respects that SPD matrices live on a
  **Hadamard (non-positively-curved) manifold**. Affine-invariant: `d(WAWᵀ,WBWᵀ)=d(A,B)`, exactly the invariance
  EEG needs. **SPD(n)+AIRM is already hyperbolic-LIKE**: `SPD(2) ≅ ℝ × ℍ²` (log-det = flat ℝ; unit-det shape =
  hyperbolic plane). So RA is alignment in a hyperbolic-flavored geometry.

## What HA does here (and its honest caveats)
`hyperbolic_align(F, d)` (in `cmi/data/alignment.py`): take LogCov tangent features `F`, embed into the Poincaré
ball (`expmap0`), per-domain compute the tangent-space Fréchet centroid, Möbius-translate it to the origin, map
back (`logmap0`). I.e. the EA/RA recentering idea, executed in a *constant-curvature* Poincaré ball instead of
on SPD-AIRM. Feature-level → pairs with the LogCov arm (`--align ha --backbone LogCov`).

**Why this may NOT beat RA (state upfront):**
1. For `n>2` channels, `SPD(n)` is a rank-`(n-1)` symmetric space with *varying* curvature — strictly richer than
   constant-curvature ℍⁿ. Forcing covariances into a Poincaré ball *loses* that structure.
2. Hyperbolic's superpower is low-distortion embedding of **hierarchies/trees**, not removing a congruence shift —
   which is what alignment needs and AIRM already provides.
3. Manifold optimization is fragile (our TSMNet/SPDNet collapsed); hyperbolic ops are unstable near the boundary.

So the expected outcome is "HA ≈ or < RA for *trial* alignment" — and that negative result is itself informative
(it confirms SPD-AIRM is the right geometry for covariances).

## The genuinely promising hyperbolic angle: DOMAIN HIERARCHY (separate from alignment)
The *domain structure* is a tree: `dataset ⊃ subject ⊃ session ⊃ trial`. Hyperbolic space embeds trees with low
distortion — so the natural place for hyperbolic geometry in this project is **not trial alignment but the domain
model**: embed domains hyperbolically and use that geometry in the CMI estimator — e.g. a hyperbolic prior `π(D)`
or a hyperbolic-distance-structured domain posterior `q(D|Z,Y)`, so that "leakage toward a *nearby* subject"
(same session/site) is penalized differently from leakage toward a far one. This is hyperbolic-for-hierarchy
(what it's good at), and would be a geometric contribution orthogonal to (and composable with) LPC-CMI.

## Experiments to run (when comparing)
- `--align {none,ea,ra}` on a few MI datasets (2a/2b/Lee2019), all backbones — EA vs RA on accuracy + (raw-probe)
  leakage. Expect EA≈RA, both > none on accuracy; both REDUCE measured leakage (so report leakage on the
  un-aligned raw signal, per the preprocessing policy).
- `--align ha --backbone LogCov` vs `--align ra` on the covariance arm — does explicit hyperbolic recentering help?
- (later) domain-hierarchy: embed subjects/sessions in ℍ², inspect whether leakage concentrates on near domains.
