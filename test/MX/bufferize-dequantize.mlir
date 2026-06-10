// RUN: mx-opt %s --mx-to-linalg --one-shot-bufferize="bufferize-function-boundaries=1" | FileCheck %s

func.func @dequantize(
    %in: !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>)
    -> tensor<32x64xf32> {
  %0 = mx.dequantize_block %in : !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU> -> tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}

// CHECK-LABEL: func.func @dequantize
// CHECK-SAME: %{{.*}}: memref<32x64xf8E4M3FN
// CHECK-SAME: %{{.*}}: memref<32x2xf8E8M0FNU
// CHECK-SAME: -> memref<32x64xf32
// CHECK: memref.alloc
// CHECK: linalg.generic
// CHECK-SAME: outs(%{{.*}} : memref
// CHECK: arith.extf
// CHECK: arith.extf
// CHECK: arith.mulf
// CHECK: return
// CHECK-NOT: bufferization.to_tensor