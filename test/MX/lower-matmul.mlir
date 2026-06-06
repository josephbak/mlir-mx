// RUN: mx-opt %s --mx-to-linalg | FileCheck %s

// CHECK-DAG: #[[$MANT:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$SCALE:.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 32)>
// CHECK-DAG: #[[$B:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$ACC:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func.func @block_matmul
// CHECK-SAME:  %[[ARG0:.+]]: tensor<32x64xf8E4M3FN>
// CHECK-SAME:  %[[ARG1:.+]]: tensor<32x2xf8E8M0FNU>
// CHECK-SAME:  %[[ARG2:.+]]: tensor<64x64xf32>
// CHECK-SAME:  %[[ARG3:.+]]: tensor<32x64xf32>

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[$MANT]], #[[$SCALE]], #[[$B]], #[[$ACC]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]], %[[ARG2]]
// CHECK-SAME: outs(%[[ARG3]]

// CHECK: arith.extf
// CHECK: arith.extf
// CHECK: %[[AREAL:.+]] = arith.mulf
// CHECK: %[[PROD:.+]] = arith.mulf %[[AREAL]],
// CHECK: arith.addf
// CHECK: linalg.yield

func.func @block_matmul(
    %lhs: !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>,        // ← fill in your real MX type
    %rhs: tensor<64x64xf32>,
    %acc: tensor<32x64xf32>) -> tensor<32x64xf32> {
  %0 = mx.block_matmul %lhs, %rhs, %acc : !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>, tensor<64x64xf32>, tensor<32x64xf32> -> tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}