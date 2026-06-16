// KNOWN LIMITATION (v1.5): arith.extf on f8E4M3FN does not lower to
// executable code on CPU through the standard pipeline. The LLVM type
// converter knows f8E4M3FN (-> i8 storage), but no MLIR pass lowers the
// conversion *op*; the only op-lowerings are GPU (ArithToAMDGPU). The
// f8E8M0FNU scale path lowers via --arith-expand="include-f8e8m0=true".
// This file is a probe documenting the gap, not a passing test.
// See DECISIONS.md 2026-06-16.

func.func @main() {
  %c = arith.constant dense<2.0> : tensor<4xf8E4M3FN>
  %f = arith.extf %c : tensor<4xf8E4M3FN> to tensor<4xf32>
  %m = bufferization.to_buffer %f : tensor<4xf32> to memref<4xf32>
  %u = memref.cast %m : memref<4xf32> to memref<*xf32>
  call @printMemrefF32(%u) : (memref<*xf32>) -> ()
  return
}
func.func private @printMemrefF32(memref<*xf32>)

