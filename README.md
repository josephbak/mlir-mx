# MX-Quantization Dialect

An out-of-tree MLIR (Multi-Level Intermediate Representation) dialect for Microscaling (MX) and sub-INT8 (8-bit integer) quantization formats. Built against current upstream LLVM/MLIR. Lowers to `linalg` + `arith` and executes via `mlir-cpu-runner` with reproducible benchmarks. This repo complements my ongoing upstream IREE contributions by demonstrating ownership of a full dialect and lowering stack.

## What this project demonstrates

End-to-end ownership of an MLIR compiler component:

- Modern C++ + TableGen (ODS — Operation Definition Specification)
- Dialect design: op set, parameterized types, verification, op interfaces
- Pattern rewriting and canonicalization (`OpRewritePattern`)
- Dialect conversion / progressive lowering (`OpConversionPattern`, `TypeConverter`)
- Bufferization of block-scaled types (value-semantic tensors → multi-memref payload)
- LLVM dialect lowering, execution, and performance measurement
- Transform-dialect scheduling applied to the lowered `linalg` form

## Why MX-quantization

Upstream MLIR's `quant` dialect defines quantized types and utility ops — intentionally thin. A lot of modern compiler work in 2026 happens at:

- **MX formats** — OCP (Open Compute Project) Spec, used by NVIDIA Blackwell and AMD MI3xx
- **Sub-INT8 weight quantization** — INT4/INT2 schemes (GPTQ, AWQ)
- **FP8 (8-bit floating point)** — standard on H100/B200/MI300; widely used in accelerators but with thin IR (Intermediate Representation) support
- **KV (Key-Value) cache quantization** — central to long-context LLM (Large Language Model) serving

This surface is thinly covered upstream and broadly relevant to accelerator companies. The work complements existing IREE backend contributions by covering the dialect-design / frontend half of the stack.

## Industry alignment

- **Cloud LLM serving** — agentic long-context workloads make low-bit KV cache quantization critical
- **Edge / Physical AI** — VLA (Vision-Language-Action) models need aggressive sub-INT8 to fit on robotic SoCs (Systems-on-Chip)
- **Accelerator hardware** — MX formats are designed for the precision/efficiency trade-off modern silicon demands

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

**Benchmark plan:** the goal is to demonstrate the memory-centric thesis (quantization attacks data movement, not arithmetic), so the benchmark measures throughput vs FP32 baseline *and* bytes moved, memory footprint reduction, and the operational-intensity shift on the roofline. Target deliverable: a roofline plot showing the workload moving from memory-bound toward compute-bound.

## Dialect design as multi-axis decision-making

A dialect is not "a list of ops" — it's co-designed decisions across twelve axes. Each choice is interview-defensible:

| Axis | Question |
|---|---|
| Abstraction level | Where in the lowering tower — semantic, structured, or hardware-near? |
| Op set | What primitives? Granularity trade-off (coarse = inflexible; fine = bloated) |
| Type system | What types? Parameterized by what? Value or memory semantics? |
| Op interfaces | `LoopLikeOpInterface`? `MemoryEffectOpInterface`? `ViewLikeOpInterface`? |
| Region structure | Do ops carry regions, block args, yields? |
| Verification | What invariants must hold for well-formedness? |
| Canonical form | What's preferred? Which rewrites converge toward it? |
| Folding | What identities, constants, algebraic simplifications fire? |
| Lowering | To what dialect(s), through what intermediate steps? |
| Composability | Can `arith`/`tensor`/`memref` mix with these ops/types? |
| Side effects | Pure, side-effecting, or both (triggers bufferization decisions)? |
| Naming | Op prefix, dialect namespace, conventions |

## Transform dialect scheduling (planned, Week 5)

The dialect lowers to `linalg`, then a Transform schedule tiles and vectorizes the linalg form. The target schedule:

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %matmul = transform.structured.match ops{["linalg.generic"]} in %root
    %tiled, %loops = transform.structured.tile_using_for %matmul [32, 32, 32]
    transform.structured.vectorize %tiled vector_sizes [16, 16]
    transform.yield
  }
}
```

The schedule is data, not compiled code — autotunable and RL-learnable. Demonstrates fluency with the modern MLIR scheduling story used by IREE codegen and upstream auto-scheduling research. The K (reduction) tile is block-aligned (= block size, 32) so the block-scale index becomes loop-invariant and vectorizes to a load + broadcast; see `docs/PLANNING.md` for the rationale.

## Build and run

```bash
# Prerequisites: LLVM/MLIR built locally (see docs/PLANNING.md § Development setup)
cd ~/dev/my-dialect
mkdir build && cd build
cmake -G Ninja .. \
  -DMLIR_DIR=$HOME/dev/llvm-build/lib/cmake/mlir \
  -DLLVM_DIR=$HOME/dev/llvm-build/lib/cmake/llvm
ninja
./bin/my-opt --help
```

## Timeline

- **Week 1** — Project skeleton; `my-opt` round-trips a stub op
- **Week 2** — All 4 ops + parameterized type; parse, print, verify
- **Week 3** — `mx.fold_scale` plus two more canonicalizations
- **Week 4** — Dialect conversion to `linalg` + bufferization handling
- **Week 5** — Execute via `mlir-cpu-runner`; benchmark; add Transform schedule
- **Week 6** — Writeup; blog post series on `josephbak.github.io`

## Deliverables

- This repo with reproducible benchmark
- Blog post series:
  1. "Designing an MX-quantization dialect in MLIR"
  2. "Bufferizing block-scaled types: multi-buffer vs. custom layout maps"
  3. "Lowering MX-matmul to linalg.generic with block-scale affine maps"

## Related work

The author contributes to [IREE](https://github.com/iree-org/iree), with in-progress work on convolution vectorization heuristics in the LLVMCPU backend. This project demonstrates dialect design + frontend lowering, complementing that codegen-heuristics work upstream.

---

*For full design rationale, trend analysis, bufferization design notes, and v2 roadmap, see `docs/PLANNING.md`.*
