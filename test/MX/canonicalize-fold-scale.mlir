// RUN: mx-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @fold_adjacent_scales
// CHECK: %[[C:.*]] = arith.constant 3.200000e+01 : f32
// CHECK: %[[R:.*]] = mx.fold_scale %arg0, %[[C]]
// CHECK: return %[[R]]

func.func @fold_adjacent_scales(
    %t: !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>)
    -> !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU> {
  %alpha1 = arith.constant 4.0 : f32
  %alpha2 = arith.constant 8.0 : f32
  %inner = mx.fold_scale %t, %alpha1 : !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>, f32 -> !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>
  %outer = mx.fold_scale %inner, %alpha2 : !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>, f32 -> !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>
  return %outer : !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>
}

// CHECK-LABEL: func.func @no_fold_single
// CHECK: %[[C:.*]] = arith.constant 4.000000e+00 : f32
// CHECK: %[[R:.*]] = mx.fold_scale %arg0, %[[C]]
// CHECK: return %[[R]]

func.func @no_fold_single(
    %t: !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>)
    -> !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU> {
  %alpha = arith.constant 4.0 : f32
  %result = mx.fold_scale %t, %alpha : !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>, f32 -> !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>
  return %result : !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>
}

// CHECK-LABEL: func.func @fold_three_levels
// CHECK: %[[C:.*]] = arith.constant 6.400000e+01 : f32
// CHECK: %[[R:.*]] = mx.fold_scale %arg0, %[[C]]
// CHECK: return %[[R]]

func.func @fold_three_levels(
    %t: !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>)
    -> !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU> {
  %a1 = arith.constant 2.0 : f32
  %a2 = arith.constant 4.0 : f32
  %a3 = arith.constant 8.0 : f32
  %s1 = mx.fold_scale %t, %a1 : !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>, f32 -> !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>
  %s2 = mx.fold_scale %s1, %a2 : !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>, f32 -> !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>
  %s3 = mx.fold_scale %s2, %a3 : !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>, f32 -> !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>
  return %s3 : !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>
}