# Here we create a single aggregate shared library with the parts of the CAPI
# that we want to bundle together. Then we link a simple C executable
# against it to demonstrate that it does have the fully self contained
# core MLIR library and our own mx dialect.
add_mlir_aggregate(MXCAPITestLib
  SHARED
  EMBED_LIBS
  MLIRCAPIIR
  MLIRCAPIArith
  MXCAPI
)

add_llvm_executable(mx-capi-test
  mx-capi-test.c
)
llvm_update_compile_flags(mx-capi-test)
target_link_libraries(mx-capi-test
  PRIVATE MXCAPITestLib)
