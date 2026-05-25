//===- MXOps.td - MX dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MX_OPS
#define MX_OPS

include "MX/MXTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def MX_FooOp : MX_Op<"foo", [Pure, SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `mx.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = arith.constant 2 : i32
        // Apply the foo operation to %0
        %1 = mx.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def MX_QuantizeBlockOp : MX_Op<"quantize_block", [Pure]> {
    let summary = "Quantize an FP32 tensor into block-scaled MX form";
    let description = [{ ... longer prose, optional ... }];


    let arguments = (ins StaticShapeTensorOf<[F32]>:$input);
    let results   = (outs MX_TensorType:$result);

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  let hasVerifier = 1;
}

def MX_DeQuantizeBlockOp : MX_Op<"dequantize_block", [Pure]> {
    let summary = "Dequantize a block-scaled MX tensor into an FP32 tensor";
    let description = [{ ... longer prose, optional ... }];


    let arguments   = (ins MX_TensorType:$input);
    let results = (outs StaticShapeTensorOf<[F32]>:$result);
    let assemblyFormat = [{
        $input attr-dict `:` type($input) `->` type($result)
    }];
  let hasVerifier = 1;
}

def MX_BlockMatmulOp : MX_Op<"block_matmul", [Pure]> {
    let summary = "Perform block matrix multiplication on MX tensor and float32 tensor to generate float32 output";
    let description = [{ ... longer prose, optional ... }];

    let arguments = (ins
        MX_TensorType:$lhs,
        StaticShapeTensorOf<[F32]>:$rhs,
        StaticShapeTensorOf<[F32]>:$acc
    );
    let results = (outs StaticShapeTensorOf<[F32]>:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $acc attr-dict `:` type($lhs) `,` type($rhs) `,` type($acc) `->` type($result)
  }];
  let hasVerifier = 1;
}

#endif // MX_OPS
