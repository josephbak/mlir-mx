//===- MXPasses.h - MX passes  ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MX_MXPASSES_H
#define MX_MXPASSES_H

#include "MX/MXDialect.h"
#include "MX/MXOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace mx {
#define GEN_PASS_DECL
#include "MX/MXPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "MX/MXPasses.h.inc"
} // namespace mx
} // namespace mlir

#endif
