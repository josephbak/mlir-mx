set(LIBS
    MLIRArithDialect
    MLIROptLib
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
    MLIRMX
  )
add_llvm_executable(mx-opt mx-opt.cpp)

llvm_update_compile_flags(mx-opt)
target_link_libraries(mx-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(mx-opt)
