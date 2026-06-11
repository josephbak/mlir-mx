# MX-Quantization Dialect

An out-of-tree MLIR (Multi-Level Intermediate Representation) dialect for Microscaling (MX) and sub-INT8 (sub-8-bit integer) quantization formats. Built against current upstream LLVM/MLIR. The dialect lowers to `linalg` + `arith` and bufferizes to memrefs; Transform-dialect scheduling, execution via `mlir-runner`, and reproducible benchmarks are in progress. It implements a full dialect and lowering stack — the design/frontend half of the compiler.

## What's covered

The completed project covers, end to end:

- Modern C++ + TableGen (ODS — Operation Definition Specification)
- Dialect design: op set, parameterized types, verification, op interfaces
- Pattern rewriting and canonicalization (`OpRewritePattern`)
- Dialect conversion / progressive lowering (`OpConversionPattern`, `TypeConverter`)
- Bufferization of block-scaled types (value-semantic tensors → multi-memref payload)
- LLVM dialect lowering, execution, and performance measurement
- Transform-dialect scheduling applied to the lowered `linalg` form

## Why MX-quantization

Upstream MLIR's `quant` dialect defines quantized types and utility ops — intentionally thin. Much of the active low-bit compiler work in 2026 sits above it:

- **MX formats** — OCP (Open Compute Project) spec, used by NVIDIA Blackwell and AMD MI3xx
- **Sub-INT8 weight quantization** — INT4/INT2 schemes (GPTQ, AWQ)
- **FP8 (8-bit floating point)** — standard on H100/B200/MI300; heavily used in hardware but thinly supported at the IR (Intermediate Representation) level
- **KV (Key-Value) cache quantization** — central to long-context LLM (Large Language Model) serving

This surface is thinly covered upstream and broadly relevant across the industry:

- **Cloud LLM serving** — agentic long-context workloads make low-bit KV-cache quantization critical
- **Edge / physical AI** — VLA (Vision-Language-Action) models need aggressive sub-INT8 to fit on robotic SoCs (Systems-on-Chip)
- **Accelerator hardware** — MX formats target the precision/efficiency trade-off these chips are built around

## v1 scope (4 ops)

| Op | Purpose |
|---|---|
| `mx.quantize_block` | FP32 tensor → block-scaled mantissa + scale tensors |
| `mx.dequantize_block` | Inverse |
| `mx.block_matmul` | Matrix multiply on block-scaled operands; FP32 accumulator |
| `mx.fold_scale` | Canonicalization seed: collapse adjacent scale-only operations |

**Types:** `!mx.tensor<MxNxFP8, block_size=32, scale_type=E8M0>` — parameterized type carrying format metadata.

**Lowering target:** `linalg.generic` over INT/FP element types with explicit scale management via `arith` ops.

**Bufferization strategy:** value-semantic `!mx.tensor` decomposes into two memrefs (mantissa + scale) at the bufferization boundary. Full rationale in `docs/PLANNING.md`.

**Benchmark plan:** the benchmark targets the memory-centric thesis (quantization attacks data movement, not arithmetic), so it measures throughput vs FP32 baseline *and* bytes moved, memory footprint reduction, and the operational-intensity shift on the roofline. Target deliverable: a roofline plot showing the workload moving from memory-bound toward compute-bound.

## Design rationale

A dialect is a set of co-designed decisions, not just a list of ops — abstraction level, op-set granularity, type parameterization, bufferization, lowering target, canonicalization, and more. Each was a deliberate trade-off; the full reasoning across every axis is in `docs/PLANNING.md`.

## Transform dialect scheduling

The dialect lowers to `linalg`, then a Transform schedule tiles and vectorizes the linalg form. Illustrative target schedule (exact op syntax and handle arity verified during implementation against the local LLVM checkout):

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %matmul = transform.structured.match ops{["linalg.generic"]} in %root
    // tile N and K by 32; leave M whole (M = block size, so a 32-tile is a trip-1 loop)
    %tiled, %loops:2 = transform.structured.tile_using_for %matmul tile_sizes [0, 32, 32]
    transform.structured.vectorize %tiled vector_sizes [16, 16]
    transform.yield
  }
}
```

The schedule is data, not compiled code, which makes it autotunable — the same Transform-dialect approach IREE codegen uses. The K (reduction) tile is block-aligned to the block size (32) for vectorization reasons detailed in `docs/PLANNING.md`.

## Build and run

```bash
# Prerequisites: LLVM/MLIR built locally (see docs/PLANNING.md § Development setup)
cd ~/dev/mlir-mx
mkdir build && cd build
cmake -G Ninja .. \
  -DMLIR_DIR=$HOME/dev/llvm-build/lib/cmake/mlir \
  -DLLVM_DIR=$HOME/dev/llvm-build/lib/cmake/llvm
ninja
./bin/mx-opt --help
```

## Status and roadmap

Done: the dialect, its four ops and parameterized type, three canonicalizations, conversion to `linalg.generic` on tensors, and bufferization to memrefs via one-shot-bufferize — all tested.

In progress: a Transform-dialect schedule that tiles and vectorizes the lowered `linalg` form, end-to-end execution via `mlir-runner`, and a memory-traffic benchmark culminating in the roofline plot. A writeup and blog-post series follow.

## Deliverables

- This repo with reproducible benchmark
- Blog post series:
  1. "Designing an MX-quantization dialect in MLIR"
  2. "Bufferizing block-scaled types: multi-buffer vs. custom layout maps"
  3. "Lowering MX-matmul to linalg.generic with block-scale affine maps"

## Related work

I also contribute to IREE — in-progress work on convolution vectorization heuristics in the LLVMCPU backend. That work is on the backend/codegen side; this project covers the dialect-design and frontend-lowering side.
