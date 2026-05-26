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

LogicalResult BlockMatmulOp::verify() {
    auto lhsType = llvm::cast<MxTensorType>(getLhs().getType());
    auto rhsType = llvm::cast<RankedTensorType>(getRhs().getType());
    auto accType = llvm::cast<RankedTensorType>(getAcc().getType());
    auto resultType = llvm::cast<RankedTensorType>(getResult().getType());

    // 1. Rank checks FIRST — before indexing into shape
    if (lhsType.getShape().size() != 2 ||
        rhsType.getShape().size() != 2 ||   // was missing () on getShape
        accType.getShape().size() != 2)      // same
      return emitOpError("all operands must be 2D");

    int64_t M = lhsType.getShape()[0];
    int64_t K = lhsType.getShape()[1];
    int64_t K2 = rhsType.getShape()[0];
    int64_t N  = rhsType.getShape()[1];
    int64_t M2 = accType.getShape()[0];
    int64_t N2 = accType.getShape()[1];

    // 2. Matmul dimension compatibility
    if (K != K2)
        return emitOpError("reduction dimension mismatch: lhs has K=")
             << K << " but rhs has K=" << K2;
    if (M != M2)
        return emitOpError("M dimension mismatch: lhs has M=")
             << M << " but acc has M=" << M2;
    if (N != N2)
        return emitOpError("N dimension mismatch: rhs has N=")
             << N << " but acc has N=" << N2;

    // 3. Result shape == acc shape
    if (accType.getShape() != resultType.getShape())
        return emitOpError("acc shape doesn't match result shape");

    return success();
}

LogicalResult FoldScaleOp::verify() {
  if (getInput().getType() != getResult().getType()) {
    return emitOpError("input and output types must be the same.");
  }

  return success();
}

#define GET_OP_CLASSES
#include "MX/MXOps.cpp.inc"
