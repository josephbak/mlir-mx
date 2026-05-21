//===- Dialects.cpp - CAPI for dialects -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MX-c/Dialects.h"

#include "MX/MXDialect.h"
#include "MX/MXTypes.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(MX, mx,
                                      mlir::mx::MXDialect)

MlirType mlirMXCustomTypeGet(MlirContext ctx, MlirStringRef value) {
  return wrap(mlir::mx::CustomType::get(unwrap(ctx), unwrap(value)));
}

bool mlirMXTypeIsACustomType(MlirType t) {
  return llvm::isa<mlir::mx::CustomType>(unwrap(t));
}

MlirTypeID mlirMXCustomTypeGetTypeID() {
  return wrap(mlir::mx::CustomType::getTypeID());
}
