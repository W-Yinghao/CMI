#!/bin/bash
# CMI-Trace P1.5 forbidden-wording gate. FAILS (exit 1) if the LaTeX makes an ASSERTIVE use of a forbidden
# phrase. Explicit disclaimers ("not an unbiased CMI", "we do not claim ... unbiased") are allowed and
# expected. Usage: bash paper/cmi_trace/tables/check_forbidden_wording.sh [latex_dir]
set -uo pipefail
DIR="${1:-paper/cigl_latex}"
# assertive patterns: a forbidden phrase NOT immediately preceded by a negation on the same line.
PATTERNS='exact CMI|calibrated bits|universal upper bound|Barber.?Agakov'
hits=$(grep -rniE "$PATTERNS" "$DIR" 2>/dev/null \
  | grep -viE "not (an? )?(exact|unbiased|calibrated|universal)|do not claim|never (an? )?(exact|unbiased)|disclaim|NOT unbiased|not a universal" \
  || true)
if [ -n "$hits" ]; then
  echo "FORBIDDEN WORDING (assertive) found:"
  echo "$hits"
  exit 1
fi
echo "forbidden-wording gate PASS ($DIR): no assertive use of: $PATTERNS"
