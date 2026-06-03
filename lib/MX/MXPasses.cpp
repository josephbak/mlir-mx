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
#include "mlir/IR/AffineMap.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"

#include "mlir/Dialect/Func/Transforms/FuncConversions.h"


namespace mlir::mx {
#define GEN_PASS_DEF_CONVERTMXTOLINALG
#include "MX/MXPasses.h.inc"

namespace {

  struct DequantizeBlockLowering : public OpConversionPattern<DeQuantizeBlockOp> {
  using OpConversionPattern<DeQuantizeBlockOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      DeQuantizeBlockOp op,
      OneToNOpAdaptor adaptor, // was OpAdaptor
      ConversionPatternRewriter &rewriter) const override {
    // ... build linalg.generic, replace op ...

    // llvm::errs() << "Pattern fired. Adaptor operands:\n";
    // for (ValueRange vr : adaptor.getOperands()) {
    //   llvm::errs() << "  operand group:\n";
    //   for (Value v : vr) {
    //     llvm::errs() << "    " << v << "  (type: " << v.getType() << ")\n";
    //   }
    // }

        // auto convertedInputs = adaptor.getOperands();  // ArrayRef<ValueRange>
        // ValueRange inputPair = convertedInputs[0];     // ValueRange for the $input operand
        // Value mantissa = inputPair[0];
        // Value scale    = inputPair[1];

        // Value mantissa = adaptor.getOperands()[0][0];
        // Value scale    = adaptor.getOperands()[0][1];

        // ValueRange inputPair = adaptor.getInput();  // ValueRange of {mantissa, scale}
        // Value mantissa = inputPair[0];
        // Value scale    = inputPair[1];

        ValueRange mantissaScale = adaptor.getInput();     // now correctly ValueRange
        Value mantissa = mantissaScale[0];
        Value scale    = mantissaScale[1];

        auto resultType = cast<RankedTensorType>(op.getResult().getType());
        Value emptyResult = tensor::EmptyOp::create(
            rewriter,
            op.getLoc(),
            resultType.getShape(),
            resultType.getElementType());
        
        // step 3
        // Recover block size from the original op's MX type
        auto mxType = cast<MxTensorType>(op.getInput().getType());
        int64_t blockSize = mxType.getBlockSize();

        // Build indexing maps
        MLIRContext *ctx = op.getContext();
        AffineExpr i, j;
        bindDims(ctx, i, j);

        auto mantissaMap = AffineMap::get(/*dims=*/2, /*syms=*/0, {i, j}, ctx);
        auto scaleMap    = AffineMap::get(/*dims=*/2, /*syms=*/0, {i, j.floorDiv(blockSize)}, ctx);
        auto resultMap   = AffineMap::get(/*dims=*/2, /*syms=*/0, {i, j}, ctx);

        SmallVector<AffineMap, 3> indexingMaps{mantissaMap, scaleMap, resultMap};
        SmallVector<utils::IteratorType, 2> iteratorTypes{
            utils::IteratorType::parallel,
            utils::IteratorType::parallel};

        auto genericOp = linalg::GenericOp::create(rewriter, op.getLoc(),
          /*resultTensorTypes=*/TypeRange{emptyResult.getType()},
          /*inputs=*/ValueRange{mantissa, scale},
          /*outputs=*/ValueRange{emptyResult},
          indexingMaps,
          iteratorTypes,
          [&](OpBuilder &builder, Location loc, ValueRange args) {
            // args[0] = mantissa element (f8E4M3FN)
            // args[1] = scale element    (f8E8M0FNU)
            // args[2] = out element      (f32)  ← unused: pure elementwise
            Value mF32 = arith::ExtFOp::create(
                builder, loc, builder.getF32Type(), args[0]);
            Value sF32 = arith::ExtFOp::create(
                builder, loc, builder.getF32Type(), args[1]);
            Value prod = arith::MulFOp::create(builder, loc, mF32, sF32);
            linalg::YieldOp::create(builder, loc, prod);
          });

      rewriter.replaceOp(op, genericOp.getResult(0));


    return success();
    }
  };

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
    typeConverter.addConversion([](Type t) -> Type { return t; });

    typeConverter.addConversion(
    [](mx::MxTensorType mxType, SmallVectorImpl<Type> &results) -> std::optional<LogicalResult> {
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
    patterns.add<DequantizeBlockLowering>(typeConverter, ctx);

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
            typeConverter.isLegal(&op.getBody());
    });
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return typeConverter.isLegal(op.getOperandTypes());
    });
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return typeConverter.isLegal(op);
    });


    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
} // namespace mlir::mx