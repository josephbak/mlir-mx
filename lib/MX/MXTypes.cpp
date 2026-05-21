//===- MXTypes.cpp - MX dialect types -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MX/MXTypes.h"

#include "MX/MXDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::mx;

#define GET_TYPEDEF_CLASSES
#include "MX/MXOpsTypes.cpp.inc"

::mlir::Type MxTensorType::parse(::mlir::AsmParser &parser) {
  if (parser.parseLess())
    return {};

  // Parses "32x64x" → shape=[32,64], consumes trailing 'x'
  llvm::SmallVector<int64_t> shape;
  if (parser.parseDimensionList(shape, /*allowDynamic=*/false))
    return {};

  // Parses "f8E4M3FN"
  mlir::Type elemType;
  if (parser.parseType(elemType))
    return {};

  // Parses ", block_size = 32"
  int64_t blockSize;
  if (parser.parseComma() || parser.parseKeyword("block_size") ||
      parser.parseEqual() || parser.parseInteger(blockSize))
    return {};

  // Parses ", scale_type = f8E8M0FNU"
  mlir::Type scaleType;
  if (parser.parseComma() || parser.parseKeyword("scale_type") ||
      parser.parseEqual() || parser.parseType(scaleType))
    return {};

  if (parser.parseGreater())
    return {};

  return MxTensorType::get(parser.getContext(), shape, elemType,
                           blockSize, scaleType);
}

void MxTensorType::print(::mlir::AsmPrinter &printer) const {
  printer << '<';
  for (int64_t dim : getShape())
    printer << dim << 'x';
  printer << getElementType();
  printer << ", block_size = " << getBlockSize();
  printer << ", scale_type = " << getScaleType();
  printer << '>';
}

void MXDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "MX/MXOpsTypes.cpp.inc"
      >();
}
