// RUN: mx-opt --mx-to-linalg %s | FileCheck %s

func.func @dequantize_basic(
    %mx_in: !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>
) -> tensor<32x64xf32> {
  %r = mx.dequantize_block %mx_in
      : !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>
      -> tensor<32x64xf32>
  return %r : tensor<32x64xf32>
}