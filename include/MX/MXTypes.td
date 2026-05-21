//===- MXTypes.td - MX dialect types -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MX_TYPES
#define MX_TYPES

include "mlir/IR/AttrTypeBase.td"
include "MX/MXDialect.td"

//===----------------------------------------------------------------------===//
// MX type definitions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// !mx.tensor type
// Syntax: !mx.tensor<32x64xf8E4M3FN, block_size = 32, scale_type = f8E8M0FNU>
// Decomposes to (mantissa memref, scale memref) at bufferization.
//===----------------------------------------------------------------------===//

class MX_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MX_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def MX_TensorType : MX_Type<"MxTensor", "tensor"> {
  let summary = "Block-scaled MX quantized tensor";
  let description = [{
    Carries mantissa data and block-scale metadata as a single value-semantic
    type. Bufferizes to a (mantissa, scale) memref pair via 1:N TypeConverter.

    Parameters:
    - shape      : tensor dimensions (e.g. [32, 64])
    - elementType: mantissa element type (e.g. f8E4M3FN)
    - blockSize  : number of elements per scale block (must be > 0)
    - scaleType  : shared scale element type (e.g. f8E8M0FNU)
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "::mlir::Type":$elementType,
    "int64_t":$blockSize,
    "::mlir::Type":$scaleType
  );

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    // Returns the number of scale blocks along the last axis.
    int64_t getNumScaleBlocks() const {
      auto s = getShape();
      return (s.back() + getBlockSize() - 1) / getBlockSize();
    }
  }];
}

#endif // MX_TYPES
