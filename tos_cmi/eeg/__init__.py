"""Phase 2 -- frozen-feature EEG pilot for TOS-CMI (measurement / diagnostic only).

Trains/loads a frozen backbone (TSMNet LogEig tangent latent) on real EEG (BNCI2014_001 / 2a, LOSO)
for ERM and global-LPC, dumps Z/logits/labels/domain, and runs the score-Fisher DIAGNOSTIC offline.
It does NOT train the encoder with a selective penalty, does NOT enable task_protect, and does NOT
promise deletion -- the certified gate's role here is a safety diagnostic + refuse-to-delete
(the task-protected certification line closed as an honest negative; see PHASE131_CERTIFICATION.md)."""
