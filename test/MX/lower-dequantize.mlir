// RUN: mx-opt --mx-to-linalg %s | FileCheck %s

// CHECK: #[[$IDENTITY:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$SCALE_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1 floordiv 32)>

// CHECK-LABEL: func.func @dequantize_basic
// CHECK-SAME: (%[[MANT:.*]]: tensor<32x64xf8E4M3FN>, %[[SCALE:.*]]: tensor<32x2xf8E8M0FNU>) -> tensor<32x64xf32>
func.func @dequantize_basic(
    %mx_in: !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>
) -> tensor<32x64xf32> {
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<32x64xf32>
  // CHECK: %[[RES:.*]] = linalg.generic
  // CHECK-SAME: indexing_maps = [#[[$IDENTITY]], #[[$SCALE_MAP]], #[[$IDENTITY]]]
  // CHECK-SAME: iterator_types = ["parallel", "parallel"]
  // CHECK-SAME: ins(%[[MANT]], %[[SCALE]]
  // CHECK-SAME: outs(%[[EMPTY]]
  // CHECK:   arith.extf %{{.*}} : f8E4M3FN to f32
  // CHECK:   arith.extf %{{.*}} : f8E8M0FNU to f32
  // CHECK:   arith.mulf
  // CHECK:   linalg.yield
  // CHECK: return %[[RES]] : tensor<32x64xf32>

  %r = mx.dequantize_block %mx_in
      : !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>
      -> tensor<32x64xf32>
  return %r : tensor<32x64xf32>
}