# Project B Reviewer Checklist

## Q1. Does the router use target labels to decide?
Answer: No. Target labels are used only **post-hoc** for metrics, after every RouterDecision. No
threshold, diagnostic, or action reads target labels.

## Q2. Why does the router miss R2's raw TTA benefit?
Answer: Because **ACAR-harm** is degenerate; v1 refuses to adapt without usable harm calibration. The
forgone benefit is a knowing **missed benefit**, not a bug.

## Q3. Why does HF3 concept-degraded identity pass?
Answer: Source-only support diagnostics do not identify concept-shift accuracy loss; a
**concept-degraded identity** can be support-valid yet inaccurate.

## Q4. Why does H-OOD density support clear under the nested threshold?
Answer: The nested excess widens the support threshold; **LOW_ESS** remains the active support signal
for low-effective-sample domains, so part of H-OOD is still refused.

## Q5. Is Project B a new TTA optimizer?
Answer: No. It is a refusal-first deployment router on top of existing TTA.

## Q6. What is the real-EEG evidence?
Answer: A BNCI2014_004 LOSO **bridge smoke** plus a **bounded real benchmark expansion** (not a full
benchmark): raw TTA harmful; the router blocks TTA and accepts support-valid identity.

## Q7. What remains for benchmark expansion?
Answer: More subjects, more targets, additional datasets (BNCI2014_001 / Lee2019_MI, GPU run).

## Q8. What changed from the real bridge smoke to the bounded real benchmark?
Answer: Step-3A showed the bridge can run on two BNCI2014_004 targets. Step-3C expands to four targets,
subject- and session-level routing, and both source-only support modes.

## Q9. Does the bounded real benchmark prove accuracy improvement?
Answer: No. Raw offline TTA was harmful in the evaluated BNCI2014_004 targets. The result supports harm
avoidance and refusal/identity routing, not accuracy improvement over identity.

## Q10. Why is nested support calibration inert on real BNCI2014_004?
Answer: Nested source-subject excess was near zero, meaning held-out source subjects were not above the
in-source support boundary under the normalized excess criterion. Thus nested and baseline support
thresholds made the same decisions.
