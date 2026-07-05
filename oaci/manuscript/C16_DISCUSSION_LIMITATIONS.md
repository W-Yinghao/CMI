# Discussion

Our results are best read not as "OACI underperformed ERM" but as a **localization of where a source-side
control hypothesis breaks** under a domain-generalization protocol with strict target isolation. The battery
converts a negative outcome into a set of specific, reproducible falsifications, and — equally important — into
a set of instruments that are *retained* precisely because they made the failure visible.

**What is closed (under this protocol).**
- OACI as a *leakage-control* method for downstream target benefit: selection-time leakage reductions do not
  survive audit (G1), audit signals are not multiplicity-stable (G2), and they do not convert to reproducible
  target worst-domain gains (G3) — a result that a source-audit oracle cannot overturn (G4).
- SRC as a *source-endpoint-control* method: improving the source worst-domain endpoint anti-transferred to
  the target under the tested configuration (G5).

**What is retained.**
- Support-aware extractable leakage `L_Q^ov` as a **measurement object** (estimable-cell restriction,
  grouped cross-fit, held-out source audit).
- The pre-registered K1 (multiplicity-controlled held-out leakage) and K2 (reproducible worst-domain endpoint)
  gates as **decision instruments**.
- **Source-audit oracle replay** as an escape-hatch diagnostic that separates "the control signal is
  uninformative" from "the selector or split was bad."
- **Source→target anti-transfer analysis** as a falsification diagnostic for source-side endpoint objectives.

**What is not claimed.** We do not claim that domain generalization fails, that EEG transfer is impossible,
that support-aware invariance is useless, or that support mismatch is empirically established by BNCI2014-001
alone. We do not claim generality across datasets or backbones, and we do not claim that every DG penalty (or
every well-regularized source-robust objective) must fail. The contribution is the falsification battery and
the two case studies it localizes, not a universal negative.

**Why a measurement-first, falsification-first stance is useful.** Source-side control signals are attractive
because they are computable without target labels; that is exactly why they are easy to trust prematurely. A
battery that audits such signals against held-out target endpoints — before a method is believed — turns a
plausible-but-untransferring penalty into a reported, reproducible failure rather than a published gain. The
same instruments would, in principle, *certify* a genuinely transferring method; demonstrating that
(discriminative validity) is the natural next step.

# Limitations

We foreground the limitations that an adversarial review of our own claims surfaced; each is scoped so the
main text does not lean on it.

1. **One dataset family, one backbone.** All results are BNCI2014-001 LOSO with ShallowConvNet. We do not
   claim the measurements or verdicts are dataset- or backbone-invariant.
2. **Support-mismatch existence is not quantified here.** BNCI2014-001 is balanced 4-class motor imagery; we do
   not report a count of estimable vs. unsupported cells, so we do not claim that a support-mismatch regime is
   empirically exercised on this data. The support-aware construction is motivation and measurement apparatus,
   not a demonstrated property of this dataset.
3. **No naive-vs-support-aware contrast.** We do not present a naive (ungrouped / support-agnostic) diagnostic
   giving a spurious answer that the support-aware `L_Q^ov` corrects; "ill-posed under mismatch" is a premise.
4. **Probe-relativity.** `L_Q^ov` is defined relative to a fixed probe-capacity family and reference prior;
   we do not vary them, so the measurement's sensitivity to that choice is unquantified.
5. **The oracle is a source-audit oracle.** G4 rules out rescue *from held-out source signal*; it is not a
   target oracle, so it does not establish that no rescuing checkpoint exists in an absolute sense.
6. **SRC anti-transfer is single-seed and un-swept.** The anti-transfer result is seed-0 only, n = 6, with no
   confidence intervals, and only the smoothing temperature was varied. The source-guard NLL collapse to
   ≈ 0.09 indicates guard memorization, so anti-transfer under a *well-regularized* source objective is
   untested — the most important open question for this result.
7. **No positive control / discriminative validity.** The battery has only ever returned `falsified`; we have
   not run an ERM-beating method through it, so its ability to certify success (not merely flag failure) is
   unshown.
8. **Minimum-seed / paused protocol.** K1/K2 use seeds [0,1,2] (a minimum-seed configuration, not a full
   5-seed manifest), and BNCI2014-004 / additional seeds are deliberately not run under a pre-registered pause.

The reviewer-objection matrix (supplementary) lists these together with the specific committed numbers that
do and do not answer each objection.
