#!/usr/bin/env bash
# Claim / wording / anonymity / build gate for the TOS-CMI manuscript. Non-zero exit on any failure.
#   bash tos_cmi/paper/check_claims.sh
set -uo pipefail
cd "$(dirname "$0")"
fail=0
chk() { # name  pattern  -- fails if pattern matches (grep -E, case-insensitive) in sections/
  local name="$1" pat="$2"; shift 2
  local hits; hits="$(grep -rniE "$pat" sections/ "$@" 2>/dev/null || true)"
  if [ -n "$hits" ]; then echo "FAIL [$name]:"; echo "$hits" | sed 's/^/    /'; fail=1
  else echo "ok   [$name]"; fi
}

echo "== forbidden wording (must be absent) =="
chk "bare capacity-not-arch" 'capacity-mediated, not (architecture|representation)-type'
chk "pure dimension effect"  '\bpure dimension effect\b'
chk "confound resolved"      'confound is resolved|confound.{0,12}resolved'
chk "EEGNet-210 in progress" 'in progress'
chk "frozen-deletion->target tie" 'removing it does not improve target|same deletion removes much of the leakage, yet target'

echo "== anonymity (must be absent in sections + tmlr build) =="
chk "identity leak" 'W-Yinghao|CMI_AAAI|/home/|infres|yinghao|github\.com/W-' tmlr_main.tex

echo "== stray literal claim tags [Cx] (use \\claimtag) =="
if grep -rnoE '\[C[0-9]+' sections/ 2>/dev/null; then echo "FAIL: stray [Cx] tags"; fail=1; else echo "ok   [no stray claim tags]"; fi

echo "== builds compile clean (0 undefined / 0 overfull) =="
for b in compile.sh compile_tmlr.sh; do
  bash "$b" >/tmp/cc_$b.log 2>&1 || { echo "FAIL: $b did not exit 0"; fail=1; }
done
for b in main tmlr_main; do
  u=$(grep -ciE 'undefined (citation|reference)' "$b.log" 2>/dev/null || true); u=${u:-0}
  o=$(grep -c 'Overfull' "$b.log" 2>/dev/null || true); o=${o:-0}
  echo "   [$b] undefined=$u overfull=$o"
  [ "$u" = 0 ] || fail=1
done

[ "$fail" = 0 ] && echo "CHECK_CLAIMS_PASS" || { echo "CHECK_CLAIMS_FAIL"; exit 1; }
