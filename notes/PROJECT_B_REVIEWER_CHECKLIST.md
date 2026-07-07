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
Answer: A BNCI2014_004 LOSO **bridge smoke** (not a full benchmark): raw TTA harmful on two targets;
the router blocks TTA and accepts support-valid identity.

## Q7. What remains for benchmark expansion?
Answer: More subjects, more targets, session-level routing, and additional datasets.
