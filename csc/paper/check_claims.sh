#!/usr/bin/env bash
# Wording / scope / build gate for the CSC manuscript. Non-zero exit on any failure.
#   bash csc/paper/check_claims.sh
set -uo pipefail
cd "$(dirname "$0")"
fail=0
chk() { # name  pattern  -- FAILS if the (forbidden) pattern matches in sections/
  local name="$1" pat="$2"; shift 2
  local hits; hits="$(grep -rniE "$pat" sections/ "$@" 2>/dev/null || true)"
  if [ -n "$hits" ]; then echo "FAIL [$name]:"; echo "$hits" | sed 's/^/    /'; fail=1
  else echo "ok   [$name]"; fi
}

echo "== forbidden over-claims (must be absent) =="
chk "working detector"        'working (concept-shift )?detector|we have a (working )?detector'
chk "solved concept shift"    'solv(e|ed|es) concept[- ]shift'
chk "no Z-only ever"          'no Z-only.{0,20}(can )?ever (work|succeed)'
chk "real EEG validated"      'real[- ]eeg.{0,30}\b(validated|confirmed|solved|works)\b|validated on real'
chk "general solution"        'general (solution|detector) (to|for) concept[- ]shift'
# C2 is pointwise: forbid only lines that CLAIM familywise/simultaneous control (allow the "not familywise" disclaimer)
fw="$(grep -rniE 'familywise|simultaneous.{0,20}confidence' sections/ 2>/dev/null | grep -viE 'not (a )?(simultaneous|familywise)|pointwise, not|not familywise|not.{0,20}simultaneous' || true)"
if [ -n "$fw" ]; then echo "FAIL [familywise-on-C2]:"; echo "$fw" | sed 's/^/    /'; fail=1; else echo "ok   [familywise-on-C2]"; fi

echo "== required scope phrases (must be PRESENT somewhere in sections/) =="
present() { local name="$1" pat="$2"; if grep -rniqE "$pat" sections/ 2>/dev/null; then echo "ok   [$name]"; else echo "FAIL [$name] missing"; fail=1; fi; }
present "synthetic-only scope" 'synthetic'
present "abstention necessary" 'abstention|abstain'
present "envelope declared"    'envelope'

echo "== stray literal claim tags [Cx] (use \\claimtag) =="
if grep -rnoE '\\claimtag\{[^}]*\}' sections/ >/dev/null 2>&1; then
  if grep -rnoE '(^|[^{])\[C-[a-z]' sections/ 2>/dev/null | grep -vE '\\claimtag' >/dev/null; then
    echo "FAIL: stray [C-...] tags outside \\claimtag"; fail=1
  else echo "ok   [claim tags via macro]"; fi
fi

echo "== neutral build compiles (0 undefined references) =="
if bash compile.sh >/tmp/csc_cc.log 2>&1; then
  u=$(grep -ciE 'undefined (citation|reference)' main.log 2>/dev/null || true); u=${u:-0}
  o=$(grep -c 'Overfull' main.log 2>/dev/null || true); o=${o:-0}
  echo "   [main] undefined=$u overfull=$o (overfull informational during skeleton)"
  [ "$u" = 0 ] || { echo "FAIL: undefined refs/citations"; fail=1; }
else
  echo "FAIL: compile.sh did not exit 0 (see /tmp/csc_cc.log)"; fail=1
fi

[ "$fail" = 0 ] && echo "CHECK_CLAIMS_PASS" || { echo "CHECK_CLAIMS_FAIL"; exit 1; }
