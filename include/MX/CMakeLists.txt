add_mlir_dialect(MXOps mx)
add_mlir_doc(MXDialect MXDialect MX/ -gen-dialect-doc)
add_mlir_doc(MXOps MXOps MX/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS MXPasses.td)
mlir_tablegen(MXPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRMXPassesIncGen)
