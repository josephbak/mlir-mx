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

void MXDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "MX/MXOpsTypes.cpp.inc"
      >();
}
