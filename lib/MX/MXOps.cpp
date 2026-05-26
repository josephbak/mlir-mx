//===- MXOps.cpp - MX dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"            // matchPattern, m_ConstantFloat
#include "mlir/IR/PatternMatch.h"        // OpRewritePattern, PatternRewriter
#include "mlir/Dialect/Arith/IR/Arith.h" // arith::ConstantOp

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

// (1) The pattern: this is where Steps 1–2 + the rewrite live
struct FoldScalePow2 : public OpRewritePattern<FoldScaleOp> {
  using OpRewritePattern<FoldScaleOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(FoldScaleOp op, PatternRewriter &rewriter) const override {
    auto innerOp = op.getInput().getDefiningOp<FoldScaleOp>();

    // Step 1: constant?      m_ConstantFloat -> else return failure();
    // Step 2: power of two?  getExactLog2    -> else return failure();
    // rewrite: e += a, replace op

    // Step 1: Is our input produced by another fold_scale?
    if (!innerOp)
      return failure();  // input came from something else — pattern doesn't apply 
    
    // Step 2: Are both alphas compile-time constants?
    APFloat alpha1(0.0f), alpha2(0.0f);
    if (!matchPattern(innerOp.getAlpha(), m_ConstantFloat(&alpha1)))
      return failure();  // inner α is runtime — can't fold
    if (!matchPattern(op.getAlpha(), m_ConstantFloat(&alpha2)))
      return failure();  // outer α is runtime — can't fold 

    // Step 3: Compute product α1 · α2
    APFloat product = alpha1;
    product.multiply(alpha2, APFloat::rmNearestTiesToEven); 

   // Step 4: Build new constant and replace outer op
    Value newAlpha = arith::ConstantOp::create(rewriter, op.getLoc(),
      rewriter.getFloatAttr(rewriter.getF32Type(), product));

    rewriter.replaceOpWithNewOp<FoldScaleOp>(
        op,                    // the op being replaced (outer fold_scale)
        op.getType(),          // result type: same !mx.tensor type
        innerOp.getInput(),    // %t — the original input, skipping the inner fold_scale
        newAlpha               // the combined α1·α2
    ); 

    return success();
  }
};

// (2) The generated hook's body: just register the pattern(s)
void FoldScaleOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<FoldScalePow2>(context);
}

#define GET_OP_CLASSES
#include "MX/MXOps.cpp.inc"
