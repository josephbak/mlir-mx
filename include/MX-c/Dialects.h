//===- Dialects.h - CAPI for dialects -----------------------------*- C -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MX_C_DIALECTS_H
#define MX_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(MX, mx);

MLIR_CAPI_EXPORTED MlirType mlirMXCustomTypeGet(MlirContext ctx,
                                                        MlirStringRef value);

MLIR_CAPI_EXPORTED bool mlirMXTypeIsACustomType(MlirType t);

MLIR_CAPI_EXPORTED MlirTypeID mlirMXCustomTypeGetTypeID(void);

#ifdef __cplusplus
}
#endif

#endif // MX_C_DIALECTS_H
