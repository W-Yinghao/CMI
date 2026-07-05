"""C17 — Source-Signal Identifiability Audit. C16 showed target-accuracy-good OACI checkpoints EXIST but the
source-audit oracle cannot find them. C17 asks the sharper question: is that information ABSENT from
source-only checkpoint observables, or did C10's oracle use the wrong projection? Diagnostic study only —
target labels are used POST HOC as diagnostic labels (marked diagnostic_only_non_deployable), never as a
deployable selector. Reads only committed C10 replay artifacts + C16 labels. No GPU, no new penalty, no OACI/
SRC tuning, no deployable selector. Imports only within `oaci`."""
