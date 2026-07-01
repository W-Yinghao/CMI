#!/usr/bin/env bash
# Artifact-level anonymity + completeness gate. Run over the assembled reviewer artifact tree.
#   bash scripts/check_artifact.sh <artifact_root>
# Exits non-zero on any identity leak, forbidden legacy wording, or missing required file.
set -uo pipefail
ROOT="${1:?usage: check_artifact.sh <artifact_root>}"
fail=0
SELF="$(basename "$0")"

# identity / site strings that must not appear anywhere in the artifact (exclude this checker + .git)
LEAK='W-Yinghao|yinghao|yinwang|/home/infres|CMI_AAAI|/projects/EEG-foundation|github\.com/W-'
hits=$(grep -rInE "$LEAK" "$ROOT" --exclude="$SELF" --exclude-dir=.git 2>/dev/null || true)
if [ -n "$hits" ]; then echo "FAIL [identity leak]:"; echo "$hits" | head -40 | sed 's/^/    /'; fail=1
else echo "ok   [no identity/site leaks]"; fi

# forbidden legacy framing (the artifact must read as TOS-CMI only)
LEG='Tri-CMI|LINE CLOSED|AAAI-27|LPC-CMI failed'
lhits=$(grep -rInE "$LEG" "$ROOT" --exclude="$SELF" --exclude-dir=.git 2>/dev/null || true)
if [ -n "$lhits" ]; then echo "FAIL [legacy Tri-CMI framing]:"; echo "$lhits" | head -20 | sed 's/^/    /'; fail=1
else echo "ok   [no legacy Tri-CMI framing]"; fi

# no submodule pointer (we vendor instead)
if [ -f "$ROOT/.gitmodules" ]; then echo "FAIL [.gitmodules present -- vendor TSMNet instead]"; fail=1
else echo "ok   [no .gitmodules]"; fi

# required files
req=( "README.md" "repos/TSMNet/VENDORED.md" "repos/TSMNet/LICENSE"
      "tos_cmi/results/tos_cmi_eeg_frozen/erasure_target_deploy/erasure_target_deploy_summary.json"
      "tos_cmi/results/tos_cmi_eeg_frozen/factorial/factorial_multiseed.json" )
for f in "${req[@]}"; do
  if [ -e "$ROOT/$f" ]; then echo "ok   [present: $f]"; else echo "FAIL [missing: $f]"; fail=1; fi
done
# at least one built PDF
if ls "$ROOT"/tos_cmi/paper/*.pdf >/dev/null 2>&1; then echo "ok   [PDF present]"; else echo "FAIL [no built PDF]"; fail=1; fi

[ "$fail" = 0 ] && echo "CHECK_ARTIFACT_PASS" || { echo "CHECK_ARTIFACT_FAIL"; exit 1; }
