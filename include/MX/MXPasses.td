//===- MXPasses.td - MX dialect passes -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MX_PASS
#define MX_PASS

include "mlir/Pass/PassBase.td"

def ConvertMXToLinalg : Pass<"mx-to-linalg", "::mlir::ModuleOp"> {
  let summary = "Lower MX ops to linalg + arith on tensors";
  let dependentDialects = [
    "::mlir::linalg::LinalgDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::math::MathDialect"
  ];
}

#endif // MX_PASS
