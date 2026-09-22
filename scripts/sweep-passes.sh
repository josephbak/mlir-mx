#!/usr/bin/env bash
# sweep-passes.sh: run every pass registered in mlir-opt on one input file,
# one pass at a time, and report which passes remove a given text pattern.
#
# Usage:  scripts/sweep-passes.sh <input.mlir> [pattern] > sweep.txt
#         pattern defaults to "d2 floordiv 32", the tiled scale map's block index.
#
# Env:    MLIR_OPT  path to mlir-opt (default: mlir-opt on PATH)
#         TIMEOUT   per-pass limit in seconds (default 30). Uses `timeout` on
#                   Linux, `gtimeout` on macOS (brew install coreutils); runs
#                   without a limit if neither is installed.
#
# Output: one line per pass:
#           unchanged  pattern still present
#           CHANGED    pattern gone; inspect by hand, since a pass that lowers
#                      the op away also removes the text
#           error      pass did not run on this input (needs options, or
#                      expects other ops); not evidence either way
#           timeout    hit the per-pass limit
#         followed by "# "-prefixed summary lines, so `grep -c ^unchanged` etc.
#         still count only per-pass results.
#
# Scope:  passes run individually, not in combination. Transform ops and
#         downstream projects are not covered.

set -uo pipefail

input=${1:?usage: sweep-passes.sh <input.mlir> [pattern]}
pattern=${2:-"d2 floordiv 32"}
mlir_opt=${MLIR_OPT:-mlir-opt}
limit=${TIMEOUT:-30}

[[ -f $input ]] || { echo "no such file: $input" >&2; exit 1; }
command -v "$mlir_opt" >/dev/null || { echo "mlir-opt not found: $mlir_opt" >&2; exit 1; }
grep -q -- "$pattern" "$input" || { echo "pattern not in input: $pattern" >&2; exit 1; }

runner=()
if command -v timeout >/dev/null; then
  runner=(timeout "$limit")
elif command -v gtimeout >/dev/null; then
  runner=(gtimeout "$limit")
fi

# Pass and pipeline names sit at exactly six spaces of indentation in
# `mlir-opt --help`; their own options are indented further and are skipped.
passes=$("$mlir_opt" --help | grep -oE '^      --[a-z0-9-]+' | sed 's/^ *//')

n_unchanged=0 n_changed=0 n_error=0 n_timeout=0
while read -r p; do
  [[ -z $p ]] && continue
  # ${runner[@]+...} keeps an empty array safe under `set -u` on bash 3.2 (macOS).
  out=$(${runner[@]+"${runner[@]}"} "$mlir_opt" "$input" "$p" 2>&1 </dev/null)
  status=$?
  if (( status == 124 )); then
    echo "timeout   $p"; n_timeout=$((n_timeout + 1))
  elif (( status != 0 )); then
    echo "error     $p"; n_error=$((n_error + 1))
  elif grep -q -- "$pattern" <<<"$out"; then
    echo "unchanged $p"; n_unchanged=$((n_unchanged + 1))
  else
    echo "CHANGED   $p"; n_changed=$((n_changed + 1))
  fi
done <<<"$passes"

echo "# input:    $input"
echo "# pattern:  $pattern"
echo "# mlir-opt: $("$mlir_opt" --version 2>/dev/null | grep -i 'llvm version' | head -1 | sed 's/^ *//')"
echo "# unchanged $n_unchanged, CHANGED $n_changed, error $n_error, timeout $n_timeout"