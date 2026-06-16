func.func @main() {
    // Known input: all 2.0 -> exact round-trip expected
    %input = arith.constant dense<2.0> : tensor<32x64xf32>

    %mx = mx.quantize_block %input
    : tensor<32x64xf32>
    -> !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>

    %out = mx.dequantize_block %mx
        : !mx.tensor<32x64xf8E4M3FN, block_size=32, scale_type=f8E8M0FNU>
        -> tensor<32x64xf32>

    // tensor -> memref (so it has a memory location to print)
    %out_memref = bufferization.to_buffer %out : tensor<32x64xf32> to memref<32x64xf32>
    // ranked -> unranked (printMemrefF32's signature)
    %unranked = memref.cast %out_memref : memref<32x64xf32> to memref<*xf32>
    // call the runtime print helper
    call @printMemrefF32(%unranked) : (memref<*xf32>) -> ()
    return
}

// declare the external runtime function (defined in libmlir_runner_utils)
func.func private @printMemrefF32(memref<*xf32>)