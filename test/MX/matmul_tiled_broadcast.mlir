// HAND-EDITED, not pipeline output. Single tile (loops stripped), scale rewritten
// to tensor<32> + map (d0) to demonstrate the broadcast form structured.vectorize
// produces. The automatic floordiv->broadcast rewrite is v1.5 (see DECISIONS 2026-06-15).
#mantissa = affine_map<(d0, d1, d2) -> (d0, d2)>
#scale    = affine_map<(d0, d1, d2) -> (d0)>
#b        = affine_map<(d0, d1, d2) -> (d2, d1)>
#acc      = affine_map<(d0, d1, d2) -> (d0, d1)>

module {
  func.func @matmul_tile(
      %mant: tensor<32x32xf8E4M3FN>,
      %scale: tensor<32xf8E8M0FNU>,
      %b: tensor<32x32xf32>,
      %acc: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = linalg.generic {
      indexing_maps = [#mantissa, #scale, #b, #acc],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%mant, %scale, %b : tensor<32x32xf8E4M3FN>, tensor<32xf8E8M0FNU>, tensor<32x32xf32>)
      outs(%acc : tensor<32x32xf32>) {
    ^bb0(%in: f8E4M3FN, %in_s: f8E8M0FNU, %in_b: f32, %out: f32):
      %1 = arith.extf %in   : f8E4M3FN to f32
      %2 = arith.extf %in_s : f8E8M0FNU to f32
      %3 = arith.mulf %1, %2 : f32
      %4 = arith.mulf %3, %in_b : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}