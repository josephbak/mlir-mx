//===- MXOps.cpp - MX dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MX/MXOps.h"
#include "MX/MXDialect.h"
#include "MX/MXTypes.h"

using namespace mlir;
using namespace mlir::mx; 

LogicalResult QuantizeBlockOp::verify() {
  auto inputType = llvm::cast<RankedTensorType>(getInput().getType());
  auto resultType = llvm::cast<MxTensorType>(getResult().getType());

  if (inputType.getShape() != resultType.getShape())
    return emitOpError("input shape doesn't match result shape");

  return success();
}

LogicalResult DeQuantizeBlockOp::verify() {
  auto inputType = llvm::cast<MxTensorType>(getInput().getType());
  auto resultType = llvm::cast<RankedTensorType>(getResult().getType());

  if (inputType.getShape() != resultType.getShape())
    return emitOpError("input shape doesn't match result shape");

  return success();
}

#define GET_OP_CLASSES
#include "MX/MXOps.cpp.inc"
