//===- MXPasses.cpp - MX passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "MX/MXPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::mx {
#define GEN_PASS_DEF_CONVERTMXTOLINALG
#include "MX/MXPasses.h.inc"

namespace {
struct ConvertMXToLinalgPass : public impl::ConvertMXToLinalgBase<ConvertMXToLinalgPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    ConversionTarget target(*ctx);
    target.addIllegalDialect<mx::MXDialect>();
    target.addLegalDialect<linalg::LinalgDialect,
                          arith::ArithDialect,
                          tensor::TensorDialect,
                          func::FuncDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion(
    [](mx::MxTensorType mxType, SmallVectorImpl<Type> &results) -> LogicalResult {
      // mantissa tensor
      results.push_back(
          RankedTensorType::get(mxType.getShape(), mxType.getElementType()));

      // scale tensor: same shape but last dim / block_size
      SmallVector<int64_t> scaleShape(mxType.getShape());
      scaleShape.back() /= mxType.getBlockSize();
      results.push_back(
          RankedTensorType::get(scaleShape, mxType.getScaleType()));

      return success();
    });

    RewritePatternSet patterns(ctx);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
} // namespace mlir::mx