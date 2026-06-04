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

#include "mlir/IR/AffineExpr.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"

#include "mlir/Dialect/Func/Transforms/FuncConversions.h"

#include "mlir/Dialect/Math/IR/Math.h"


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

  struct QuantizeBlockLowering : public OpConversionPattern<QuantizeBlockOp> {
  using OpConversionPattern<QuantizeBlockOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      QuantizeBlockOp op,
      OneToNOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

      // step 1 — output setup for Generic 1
      auto inputType = cast<RankedTensorType>(op.getInput().getType());
      ArrayRef<int64_t> inputShape = inputType.getShape();
      int64_t M = inputShape[0];
      int64_t N = inputShape[1];

      auto resultMxType = cast<MxTensorType>(op.getResult().getType());
      int64_t blockSize = resultMxType.getBlockSize();
      int64_t numBlocks = N / blockSize;

      Value input = op.getInput();
      Location loc = op.getLoc();

      // max-abs accumulator: tensor<M x N/B x f32>
      Value maxEmpty = tensor::EmptyOp::create(
          rewriter, loc,
          SmallVector<int64_t>{M, numBlocks},
          rewriter.getF32Type());

      Value zero = arith::ConstantOp::create(rewriter, loc, rewriter.getF32FloatAttr(0.0)); // 0.0 constant
      Value maxInit = linalg::FillOp::create(
          rewriter, loc,
          /*inputs=*/ValueRange{zero},
          /*outputs=*/ValueRange{maxEmpty}).getResult(0);


      // step 2 — Generic 1: indexing maps + iterator types

      // reshape input: tensor<MxNxf32> -> tensor<M x numBlocks x blockSize x f32>
      SmallVector<int64_t> expandedShape{M, numBlocks, blockSize};
      auto expandedType = RankedTensorType::get(expandedShape, rewriter.getF32Type());

      SmallVector<ReassociationIndices> reassoc = {{0}, {1, 2}};

      Value input3D = tensor::ExpandShapeOp::create(
          rewriter, loc, expandedType, input, reassoc);


      MLIRContext *ctx = op.getContext();
      AffineExpr ii, bb, kk;
      bindDims(ctx, ii, bb, kk);


      auto inputMap = AffineMap::get(/*dims=*/3, /*syms=*/0,
                               {ii, bb, kk}, ctx);   // was {ii, bb*blockSize + kk}
      auto maxMap   = AffineMap::get(/*dims=*/3, /*syms=*/0,
                                {ii, bb}, ctx);        // unchanged

      SmallVector<AffineMap, 2> g1IndexingMaps{inputMap, maxMap};
      SmallVector<utils::IteratorType, 3> g1IteratorTypes{
          utils::IteratorType::parallel,
          utils::IteratorType::parallel,
          utils::IteratorType::reduction
        };

      // step 3 — build Generic 1
      auto maxOp = linalg::GenericOp::create(rewriter, loc,
          /*resultTensorTypes=*/TypeRange{maxInit.getType()},
          /*inputs=*/ValueRange{input3D},
          /*outputs=*/ValueRange{maxInit},
          g1IndexingMaps,
          g1IteratorTypes,
          [&](OpBuilder &builder, Location loc, ValueRange args) {
            // args[0] = current input element (f32)
            // args[1] = current accumulator value (f32)
            Value absX = math::AbsFOp::create(builder, loc, args[0]);
            Value newMax = arith::MaximumFOp::create(builder, loc, absX, args[1]);
            linalg::YieldOp::create(builder, loc, newMax);
          });

      Value perBlockMax = maxOp.getResult(0);

      // step 4 — post-processing generic
      // 4a: empty scale tensor<M x N/B x f32>
      Value scaleEmpty = tensor::EmptyOp::create(
          rewriter, loc,
          SmallVector<int64_t>{M, numBlocks},
          resultMxType.getScaleType());

      // 4b: indexing maps + iterator types for this generic
      AffineExpr pi, pb;
      bindDims (ctx, pi, pb);

      auto scaleInMap = AffineMap::get(/*dims=*/2, /*syms=*/0, {pi, pb}, ctx);
      auto scaleOutMap    = AffineMap::get(/*dims=*/2, /*syms=*/0, {pi, pb}, ctx);

      SmallVector<AffineMap, 2> ppIndexingMaps{scaleInMap, scaleOutMap};
      SmallVector<utils::IteratorType, 2> ppIteratorTypes{
          utils::IteratorType::parallel,
          utils::IteratorType::parallel};

      // 4c: build the post-processing generic itself
      auto scaleOp = linalg::GenericOp::create(
        rewriter, loc,
        /*resultTensorTypes=*/TypeRange{scaleEmpty.getType()},
        /*inputs=*/ValueRange{perBlockMax},
        /*outputs=*/ValueRange{scaleEmpty},
        ppIndexingMaps,
        ppIteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          // args[0] = per-block max-abs (f32)
          // args[1] = output slot (f8E8M0FNU) — unused, elementwise
          //
          // TODO: compute scale = 2^(floor(log2(args[0])) - 8), truncate to f8E8M0FNU
          // then linalg.yield it
          Value logVal = math::Log2Op::create(builder,loc, args[0]);
          Value floorVal = math::FloorOp::create(builder, loc, logVal);
          Value eMax = arith::ConstantOp::create(builder, loc, builder.getF32FloatAttr(8.0));
          Value expVal = arith::SubFOp::create(builder, loc, floorVal, eMax);
          Value scaleF32 = math::Exp2Op::create(builder, loc, expVal);
          Value scaleQ  = arith::TruncFOp::create(builder, loc, resultMxType.getScaleType() , scaleF32);
          linalg::YieldOp::create(builder, loc, scaleQ);
        });

      Value scaleTensor = scaleOp.getResult(0);

    // step 5 — output setup for Generic 2
    Value mantissaEmpty = tensor::EmptyOp::create(
        rewriter, loc,
        SmallVector<int64_t>{M, N},
        resultMxType.getElementType());

    // step 6 — indexing maps + iterator types for Generic 2
      AffineExpr mi, mj;
      bindDims (ctx, mi, mj);

      auto mantInMap = AffineMap::get(/*dims=*/2, /*syms=*/0, {mi, mj}, ctx);
      auto mantScaleMap = AffineMap::get(/*dims=*/2, /*syms=*/0, {mi, mj.floorDiv(blockSize)}, ctx);
      auto mantOutMap = AffineMap::get(/*dims=*/2, /*syms=*/0, {mi, mj}, ctx);

      SmallVector<AffineMap, 3> g2IndexingMaps{mantInMap, mantScaleMap, mantOutMap};
      SmallVector<utils::IteratorType, 2> g2IteratorTypes{
          utils::IteratorType::parallel,
          utils::IteratorType::parallel};

    // step 7 — build Generic 2
      auto mantissaOp = linalg::GenericOp::create(
        rewriter, loc,
        /*resultTensorTypes=*/TypeRange{mantissaEmpty.getType()},
        /*inputs=*/ValueRange{input, scaleTensor},
        /*outputs=*/ValueRange{mantissaEmpty},
        g2IndexingMaps,
        g2IteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          Value scaleF32 = arith::ExtFOp::create(builder, loc, builder.getF32Type(), args[1]); // extf the scale ele. (f8E8M0FNU -> f32)
          Value quotient = arith::DivFOp::create(builder, loc, args[0], scaleF32);
          Value mantQ  = arith::TruncFOp::create(builder, loc, args.back().getType() , quotient);
          linalg::YieldOp::create(builder, loc, mantQ);
        });

      Value mantTensor = mantissaOp.getResult(0);

    // step 8 — replaceOp with (mantissa, scale)
    // rewriter.replaceOp(op, ValueRange{mantTensor, scaleTensor});
    rewriter.replaceOpWithMultiple(op, {{mantTensor, scaleTensor}});

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
                          func::FuncDialect,
                          math::MathDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type t) -> Type { return t; });

    typeConverter.addConversion(
    [](mx::MxTensorType mxType, SmallVectorImpl<Type> &results) -> std::optional<LogicalResult> {
      // mantissa tensor <- pushed first
      results.push_back(
          RankedTensorType::get(mxType.getShape(), mxType.getElementType()));

      // scale tensor: same shape but last dim / block_size <- pushed second
      SmallVector<int64_t> scaleShape(mxType.getShape());
      scaleShape.back() /= mxType.getBlockSize();
      results.push_back(
          RankedTensorType::get(scaleShape, mxType.getScaleType()));

      return success();
    });

    RewritePatternSet patterns(ctx);
    patterns.add<DequantizeBlockLowering>(typeConverter, ctx);
    patterns.add<QuantizeBlockLowering>(typeConverter, ctx);

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