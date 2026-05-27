add_mlir_dialect_library(MLIRMX
        MXTypes.cpp
        MXDialect.cpp
        MXOps.cpp
        MXPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/MX

        DEPENDS
        MLIRMXOpsIncGen
        MLIRMXPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        MLIRArithDialect
        MLIRTensorDialect
        )
