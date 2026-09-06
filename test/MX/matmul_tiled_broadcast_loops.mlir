// HAND-EDITED, not pipeline output. The full two-loop tiled nest from
// schedule.mlir, with one edit: the scale slice rank-reduced to tensor<32> and
// its indexing map rewritten from (d0, d2 floordiv 32) to (d0), to demonstrate
// the broadcast form structured.vectorize produces. The automatic
// floordiv->broadcast rewrite is v1.5 (see DECISIONS 2026-06-15).
#map = affine_map<(d0) -> (d0 floordiv 32)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @block_matmul(%arg0: tensor<32x64xf8E4M3FN>, %arg1: tensor<32x2xf8E8M0FNU>, %arg2: tensor<64x64xf32>, %arg3: tensor<32x64xf32>) -> tensor<32x64xf32> {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %0 = scf.for %arg4 = %c0 to %c64 step %c32 iter_args(%arg5 = %arg3) -> (tensor<32x64xf32>) {
      %1 = scf.for %arg6 = %c0 to %c64 step %c32 iter_args(%arg7 = %arg5) -> (tensor<32x64xf32>) {
        %2 = affine.apply #map(%arg6)
        %extracted_slice = tensor.extract_slice %arg0[0, %arg6] [32, 32] [1, 1] : tensor<32x64xf8E4M3FN> to tensor<32x32xf8E4M3FN>
        %extracted_slice_3 = tensor.extract_slice %arg1[0, %2] [32, 1] [1, 1] : tensor<32x2xf8E8M0FNU> to tensor<32xf8E8M0FNU>
        %extracted_slice_4 = tensor.extract_slice %arg2[%arg6, %arg4] [32, 32] [1, 1] : tensor<64x64xf32> to tensor<32x32xf32>
        %extracted_slice_5 = tensor.extract_slice %arg7[0, %arg4] [32, 32] [1, 1] : tensor<32x64xf32> to tensor<32x32xf32>
        %3 = linalg.generic {indexing_maps = [#map1, #map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_3, %extracted_slice_4 : tensor<32x32xf8E4M3FN>, tensor<32xf8E8M0FNU>, tensor<32x32xf32>) outs(%extracted_slice_5 : tensor<32x32xf32>) {
        ^bb0(%in: f8E4M3FN, %in_6: f8E8M0FNU, %in_7: f32, %out: f32):
          %4 = arith.extf %in : f8E4M3FN to f32
          %5 = arith.extf %in_6 : f8E8M0FNU to f32
          %6 = arith.mulf %4, %5 : f32
          %7 = arith.mulf %6, %in_7 : f32
          %8 = arith.addf %out, %7 : f32
          linalg.yield %8 : f32
        } -> tensor<32x32xf32>
        %inserted_slice = tensor.insert_slice %3 into %arg7[0, %arg4] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<32x64xf32>
        scf.yield %inserted_slice : tensor<32x64xf32>
      }
      scf.yield %1 : tensor<32x64xf32>
    }
    return %0 : tensor<32x64xf32>
  }
}