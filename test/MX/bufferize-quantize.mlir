// RUN: mx-opt %s --mx-to-linalg --one-shot-bufferize="bufferize-function-boundaries=1" | FileCheck %s

func.func @quantize(
    %in: tensor<32x64xf32>)
    -> !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU> {
  %0 = mx.quantize_block %in : tensor<32x64xf32> -> !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>
  return %0 : !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>
}

// CHECK-LABEL: func.func @quantize
// CHECK-SAME: %{{.*}}: memref<32x64xf32
// CHECK-SAME: -> (memref<32x64xf8E4M3FN
// CHECK-SAME: memref<32x2xf8E8M0FNU
// CHECK: linalg.fill
// CHECK: memref.expand_shape
// CHECK: math.log2
// CHECK: arith.truncf
// CHECK: return %{{.*}}, %{{.*}} :
// CHECK-NOT: bufferization.to_tensor