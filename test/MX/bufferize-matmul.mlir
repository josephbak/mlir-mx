// RUN: mx-opt %s --mx-to-linalg --one-shot-bufferize="bufferize-function-boundaries=1" | FileCheck %s

// CHECK-LABEL: func.func @block_matmul
// CHECK-SAME: memref<32x64xf8E4M3FN
// CHECK-SAME: memref<32x2xf8E8M0FNU
// CHECK-SAME: memref<64x64xf32
// CHECK-SAME: memref<32x64xf32

// CHECK: linalg.generic
// CHECK-SAME: outs(%{{.*}} : memref
// CHECK: arith.addf
// CHECK: return %{{.*}} : memref

// CHECK-NOT: memref.copy
// CHECK-NOT: bufferization.to_tensor

func.func @block_matmul(
    %lhs: !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>,
    %rhs: tensor<64x64xf32>,
    %acc: tensor<32x64xf32>) -> tensor<32x64xf32> {
  %0 = mx.block_matmul %lhs, %rhs, %acc : !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>, tensor<64x64xf32>, tensor<32x64xf32> -> tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}