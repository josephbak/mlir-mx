// RUN: mx-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @dequant_of_fold_scale
// CHECK: %[[C:.*]] = arith.constant dense<4.000000e+00> : tensor<32x64xf32>
// CHECK: %[[D:.*]] = mx.dequantize_block %arg0
// CHECK: %[[R:.*]] = arith.mulf %[[D]], %[[C]]
// CHECK-NOT: mx.fold_scale
// CHECK: return %[[R]]

// CHECK-LABEL: func.func @dequant_fold_scale_runtime_alpha
// CHECK: %[[D:.*]] = mx.dequantize_block %arg0
// CHECK: %[[S:.*]] = tensor.splat %arg1
// CHECK: %[[R:.*]] = arith.mulf %[[D]], %[[S]]
// CHECK-NOT: mx.fold_scale
// CHECK: return %[[R]]

func.func @dequant_of_fold_scale(
    %t: !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>)
    -> tensor<32x64xf32> {
        //ops
    %alpha = arith.constant 4.0 : f32
    %s = mx.fold_scale %t, %alpha : !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>, f32 -> !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>
    %result = mx.dequantize_block %s : !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU> -> tensor<32x64xf32>
    return %result: tensor<32x64xf32>
}

func.func @dequant_fold_scale_runtime_alpha(
    %t: !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>,
    %alpha: f32) -> tensor<32x64xf32> {
    %s = mx.fold_scale %t, %alpha : !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>, f32 -> !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>
    %result = mx.dequantize_block %s : !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU> -> tensor<32x64xf32>
    return %result : tensor<32x64xf32>
}