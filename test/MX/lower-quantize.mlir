// RUN: mx-opt --mx-to-linalg %s | FileCheck %s

// CHECK-LABEL: func.func @quantize_basic
// CHECK-SAME: (%[[IN:.*]]: tensor<32x64xf32>) -> (tensor<32x64xf8E4M3FN>, tensor<32x2xf8E8M0FNU>)

// max-abs reduction
// CHECK: %[[MAXEMPTY:.*]] = tensor.empty() : tensor<32x2xf32>
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[MAXINIT:.*]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[MAXEMPTY]]
// CHECK: %[[EXP:.*]] = tensor.expand_shape %[[IN]] {{\[\[}}0], [1, 2]] output_shape [32, 2, 32]
// CHECK: %[[MAXES:.*]] = linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK: math.absf
// CHECK: arith.maximumf

// scale computation
// CHECK: %[[SCALE:.*]] = linalg.generic
// CHECK: math.log2
// CHECK: math.floor
// CHECK: arith.subf
// CHECK: math.exp2
// CHECK: arith.truncf %{{.*}} : f32 to f8E8M0FNU

// mantissa computation
// CHECK: %[[MANT:.*]] = linalg.generic
// CHECK-SAME: ins(%[[IN]], %[[SCALE]]
// CHECK: arith.extf %{{.*}} : f8E8M0FNU to f32
// CHECK: arith.divf
// CHECK: arith.truncf %{{.*}} : f32 to f8E4M3FN

// CHECK: return %[[MANT]], %[[SCALE]]

func.func @quantize_basic(
    %input: tensor<32x64xf32>
) -> !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU> {
  %r = mx.quantize_block %input
      : tensor<32x64xf32>
      -> !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>
  return %r : !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>
}