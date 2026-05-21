//===- MXDialect.cpp - MX dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MX/MXDialect.h"
#include "MX/MXOps.h"
#include "MX/MXTypes.h"

using namespace mlir;
using namespace mlir::mx;

#include "MX/MXOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// MX dialect.
//===----------------------------------------------------------------------===//

void MXDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "MX/MXOps.cpp.inc"
      >();
  registerTypes();
}
