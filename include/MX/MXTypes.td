//===- MXTypes.td - MX dialect types -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MX_TYPES
#define MX_TYPES

include "mlir/IR/AttrTypeBase.td"
include "MX/MXDialect.td"

//===----------------------------------------------------------------------===//
// MX type definitions
//===----------------------------------------------------------------------===//

class MX_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MX_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def MX_CustomType : MX_Type<"Custom", "custom"> {
    let summary = "MX custom type";
    let description = "Custom type in mx dialect";
    let parameters = (ins StringRefParameter<"the custom value">:$value);
    let assemblyFormat = "`<` $value `>`";
}

#endif // MX_TYPES
