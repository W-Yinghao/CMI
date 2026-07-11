# STAR_00C Artifact Inventory

`results/star/star00c_preflight/` contains only small launch-hardening evidence:

- persistence, approval-lock, comparison-accounting, final-closure, source-task-audit, and dependency contracts;
- final-code smoke summary, persistence index, Slurm job record, stdout/stderr;
- independent red-team, preflight summary, and go/no-go records.

The bounded smoke attempt trees are external under
`/home/infres/yinwang/CMI_AAAI_star_runtime/results/star00c_realpath_smoke/`.
They contain real diagnostic `.pth` payloads and per-step JSONL telemetry and
are deliberately not committed.

The actual post-commit approval lock, launch receipt, six formal attempt trees,
and final immutable closure outputs are operational artifacts under
`/home/infres/yinwang/CMI_AAAI_star_runtime/results/`. They are created only
after the clean STAR_00C commit and are not part of that commit.

No raw EEG, feature dump, `.pth`, `.pt`, `.npz`, target prediction, or target
metric is committed.
