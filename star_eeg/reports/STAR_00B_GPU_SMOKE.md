# STAR_00B Real CBraMod GPU Smoke

## Execution

- Authoritative final-source Slurm job: 893001
- State / exit: `COMPLETED`, `0:0`
- Node / GPU: node35, NVIDIA A40
- Wall allocation: 00:00:48
- Start: immutable H200_s0 SHA `64977656005c6ac848af317caa48215eb50c780c869e8cebc930cc6bc5c15e63`
- Cells: H200_SSL_CONT, H200_STAR_TRUE, H200_STAR_SHUFFLED
- Steps: ten per variant, batch size 64

## Gate result

All smoke checks passed:

- identical B/C/D starting model-state and model-update-scope hashes;
- identical common SSL batch-ID and normalized-tensor hashes;
- identical C/D anchor-X IDs and normalized tensors;
- different true versus frozen shuffled label streams;
- two B replacement SSL slots and two C/D anchor slots;
- finite loss, encoder/model/head gradients, clipping, and parameter deltas;
- B temporary head unchanged; C/D temporary heads updated;
- all diagnostic checkpoints strict reload;
- immutable H200 source SHA unchanged;
- zero source_val/test reads and tensors.

Peak allocated GPU memory was 14.745 GB for B and 14.780 GB for C/D (decimal bytes; approximately 13.76 GiB maximum). The sequential ten-step wall times are cold-cache/setup confounded and cannot compare variants. The planning artifact retains a conservative 24-hour A40 template and marks all runtime estimates as non-selection diagnostics.

Job 893001 repeated the complete smoke after the fail-closed 20-step preflight
ceiling was added; its summary freezes the SHA-256 of every active runner,
stream, and schedule source file. Jobs 892998 and 892999 are retained as
preliminary provenance but are not the authoritative launch-preflight record.

This was a bounded real-path integrity smoke. It did not run a 3,750-step cell, calculate a downstream metric, or authorize STAR_01.
