add_llvm_library(MXPlugin
        # BUILDTREE_ONLY is only for testing purposes
        MODULE BUILDTREE_ONLY
        mx-plugin.cpp

        DEPENDS
        MLIRMX
        PLUGIN_TOOL
        mlir-opt

        LINK_LIBS
        MLIRMX
        )
target_include_directories(
        MXPlugin
        PRIVATE
        "${MX_BINARY_DIR}/include"
)
