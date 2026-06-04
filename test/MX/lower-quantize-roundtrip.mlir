// RUN: mx-opt --mx-to-linalg %s | FileCheck %s

func.func @quantize_roundtrip(
    %input: tensor<32x64xf32>
) -> tensor<32x64xf32> {
  %q = mx.quantize_block %input
      : tensor<32x64xf32>
      -> !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>
  %d = mx.dequantize_block %q
      : !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>
      -> tensor<32x64xf32>
  return %d : tensor<32x64xf32>
}