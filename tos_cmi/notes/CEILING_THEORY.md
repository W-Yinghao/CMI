# Source-only acceptance ceiling --- theory spine (for the current source-only paper)

Purpose: lock the theoretical spine that turns the method-deepening empirical result (Track B refusal + V2
source-only acceptance ceiling) into a principled decision-theoretic statement. NOT a full math paper; a spine
the manuscript can cite. Three propositions: (1) non-identifiability [limit], (2) source-rich sufficiency
[constructive], (3) target-label sample complexity [bridge to the NEXT paper, not a current-paper result].

Setup. An intervention `a` (e.g. an erasure) maps a representation Z to a(Z) (identity = no-op). Deployment
risk on the target domain is `R_T(a)`. The intervention gain is
    Delta_T(a) = R_T(identity) - R_T(a)     (positive = beneficial at the target).
A **source-only** controller sees only source data `S` and outputs a policy `pi(S) in {accept, reject, abstain}`.

---

## Proposition 1 (source-only non-identifiability of deployment-shift benefit)
Let `S` denote all source data available to a gate, and let `pi(S) in {accept, reject, abstain}` be any
source-only intervention policy. There exist two worlds `W+` and `W-` such that:
1. `P_S^{+} = P_S^{-}`                         (identical source law)
2. all source diagnostics are identical under `W+` and `W-`
3. an intervention `a` improves target risk in `W+`     (Delta_T^{+}(a) > 0)
4. the same intervention is neutral or harmful on target in `W-`  (Delta_T^{-}(a) <= 0)

Then `pi` has the same action distribution in `W+` and `W-`. Hence any source-only policy that accepts `a` in
`W+` with nonzero probability also accepts it in `W-` with the same probability. To control false accepts
uniformly over `{W+, W-}`, `pi` must abstain/reject in `W+` as well.

**Proof sketch.** Because `P_S^{+} = P_S^{-}`, any measurable function of `S` has the same distribution under
both worlds. The gate input is source-only, so `pi(S)` cannot distinguish `W+` from `W-`. The target-risk
contrast `Delta_T(a)` changes only through `P_T`, which `pi` never observes. Therefore
`Pr[pi(S)=accept | W+] = Pr[pi(S)=accept | W-]`; a policy with false-accept rate 0 in `W-` accepts in `W+`
with probability 0. QED.

**Construction (existence).** V2 World A is a concrete witness: real EEG latents + an injected deployment-shift
nuisance whose target-harm is invisible in the source marginals; the injected-nuisance oracle improves the
target while every source-only signal (safety, source-LOSO benefit) is identical to a no-benefit construction.

**Wording discipline (do NOT overclaim).**
* This is NOT an impossibility of all source-only learning.
* It is a non-identifiability result for target benefits that are **not represented in source-domain variation**.
* Write: *"source-only certification requires either source-visible shift, target information, or additional
  structural assumptions."*  Do NOT write: *"source-only benefit is impossible in general."*

---

## Proposition 2 (source-rich sufficiency condition --- the constructive counterpart)
Idea: source-only benefit certification CAN be valid when the target shift is represented by source-domain
variation.

Assume the target domain is exchangeable with, or lies in the convex hull of, the source-domain **environment**
distribution under a bounded loss class. Let `G_hat_src-LOEO(a)` be a leave-one-source-ENVIRONMENT-out estimate
of the intervention gain for action `a`. If the source environments cover the deployment-relevant shift and the
confidence bound is calibrated, then a lower confidence bound on `G_hat_src-LOEO(a)` is a conservative
certificate for target gain up to a coverage error term:

    Delta_T(a)  >=  LCB_{1-alpha}( G_hat_src-LOEO(a) )  -  eps_coverage

where `eps_coverage` measures how poorly the source environments cover the target shift (0 under exact
exchangeability / convex-hull coverage; grows as the target shift leaves the source-environment support).

**Definitions used downstream.**
* **source-visible benefit** = the deployment target shift IS represented by held-out source-environment variation.
* **source-invisible benefit** = the deployment shift is NOT represented in source domains.

**This explains the two empirical results.**
* Track B (real EEG): source-LOSO IS the leave-one-source-environment-out estimate (environment = subject). If
  it shows benefit (LCB > +0.01) and safety holds, accept is licensed; on real EEG it never does -> refuse.
  0 false accepts. => the gate acts under Prop 2 with environment = subject.
* V2 World A: deliberately constructs a target-beneficial shift that is source-INVISIBLE (`eps_coverage` large
  by design), so the source-only LCB is <= 0 and the gate correctly ABSTAINS -- consistent with Prop 1.

---

## Proposition 3 (target-label sample-complexity bridge --- NEXT PAPER, not a current-paper result)
Included only as the intellectual bridge to "From Refusal to Control: How Much Target Information Is Needed?".
With even a small labeled target calibration set, the Prop-1 non-identifiability can be broken because the gate
can estimate the target risk contrast directly.

For bounded per-trial loss differences `d_i(a) in [-1,1]` between identity and intervention `a`, a
target-labeled calibration set of size `n` yields a confidence interval for `Delta_T(a) = E_T[d_i(a)]`. A
sufficient condition for accepting with error probability `delta` is

    n = O( eps^{-2} * log(1/delta) )

to distinguish `Delta_T(a) > eps` from `Delta_T(a) <= 0` (Hoeffding / empirical-Bernstein on bounded `d_i`).

**Balanced-accuracy note.** For balanced accuracy the bound applies PER CLASS and combines across classes, so
the effective sample size is the **minimum labeled target count per class** (stratified).

**Scope.** Prop 3 is a BRIDGE statement only. The target-information budget experiment (0 / unlabeled /
1,2,4,8,16 labels per class / active calibration / oracle) is the NEXT paper and is NOT run in the current line.

---

## One-paragraph spine for the manuscript
Measurement works; erasure is not control; source-only certified control is refusal-first. By Prop 1, a
source-only gate cannot certify deployment-shift benefit that is not represented in source-domain variation, so
refusal is the safe action (V2 World A is the witness; Track B is the real-EEG instantiation). By Prop 2, when
source environments cover the deployment shift, a calibrated source-LOEO lower bound is a conservative benefit
certificate -- so acceptance is licensed exactly in the source-rich regime. By Prop 3, a small amount of target
information breaks the ceiling with per-class sample complexity `O(eps^{-2} log(1/delta))` -- the subject of the
next paper. Crossing the ceiling therefore requires source-rich environments or target information.
