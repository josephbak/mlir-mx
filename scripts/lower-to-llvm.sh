#!/usr/bin/env bash
# Lower an mx-dialect .mlir file all the way to LLVM dialect and (optionally) run it.
#
# Pipeline derived + verified against local llvm-build, Week 5 (2026-06).
# Usage:
#   ./docs/lower-to-llvm.sh test/MX/run-roundtrip.mlir          # lower only, print LLVM-dialect IR
#   ./docs/lower-to-llvm.sh test/MX/run-roundtrip.mlir --run    # lower AND execute via mlir-runner
#
# Requires: mx-opt, mlir-runner on PATH (direnv puts llvm-build/bin there);
#           libmlir_runner_utils.dylib in $HOME/dev/llvm-build/lib.

set -euo pipefail

INPUT="${1:?usage: lower-to-llvm.sh <input.mlir> [--run]}"
MODE="${2:-}"
RUNNER_LIB="$HOME/dev/llvm-build/lib/libmlir_runner_utils.dylib"

# --- Pass ordering rationale (do not reorder casually) ---
#  mx-to-linalg                         : your dialect -> linalg on tensors (1:N split)
#  one-shot-bufferize (fn-boundaries=1) : tensors -> memrefs, in-place into acc (copy-free)
#  convert-linalg-to-loops              : linalg.generic -> scf.for + memref.load/store
#                                         (BORN HERE: affine.apply for the k floordiv 32 scale index)
#  lower-affine                         : affine.apply -> arith/scf  (MUST follow linalg-to-loops)
#  convert-scf-to-cf                    : scf.for -> cf branches (scf only exists after linalg-to-loops)
#  expand-strided-metadata              : memref.expand_shape/cast -> analyzable strided form
#                                         (MUST precede finalize-memref-to-llvm)
#  arith-expand include-f8e8m0=true     : emulate f8E8M0FNU extf/truncf (backend can't represent it)
#                                         (MUST precede convert-arith-to-llvm; emits arith/cmpi ops)
#  convert-math-to-llvm                 : math.{log,exp,floor,absf} -> llvm intrinsics
#  finalize-memref-to-llvm              : memref.{alloc,load,store} -> llvm descriptors
#  convert-arith-to-llvm                : arith.* -> llvm
#  convert-cf-to-llvm                   : cf -> llvm
#  convert-func-to-llvm                 : func -> llvm
#  reconcile-unrealized-casts           : erase leftover cast glue (LAST)
#
# KNOWN LIMITATION (v1.5): arith.extf/truncf on f8E4M3FN do NOT lower on CPU.
#   They survive to LLVM dialect; mlir-runner's LLVM backend is expected to handle
#   them but this is unverified. The E8M0 scale path and all f32 paths lower cleanly.

PIPELINE=(
  --mx-to-linalg
  --one-shot-bufferize="bufferize-function-boundaries=1"
  --convert-linalg-to-loops
  --lower-affine
  --convert-scf-to-cf
  --expand-strided-metadata
  --arith-expand="include-f8e8m0=true"
  --convert-math-to-llvm
  --finalize-memref-to-llvm
  --convert-arith-to-llvm
  --convert-cf-to-llvm
  --convert-func-to-llvm
  --reconcile-unrealized-casts
)

if [[ "$MODE" == "--run" ]]; then
  mx-opt "$INPUT" "${PIPELINE[@]}" \
    | mlir-runner -e main --entry-point-result=void --shared-libs="$RUNNER_LIB"
else
  mx-opt "$INPUT" "${PIPELINE[@]}"
fi