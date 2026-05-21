llvm_canonicalize_cmake_booleans(
        MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(MX_TEST_DEPENDS
        FileCheck count not
        mx-capi-test
        mx-opt
        mx-translate
        )
if(NOT WIN32)
  list(APPEND MX_TEST_DEPENDS MXPlugin)
endif()
if(MLIR_ENABLE_BINDINGS_PYTHON)
  list(APPEND MX_TEST_DEPENDS MXPythonModules)
endif()

add_lit_testsuite(check-mx "Running the mx regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${MX_TEST_DEPENDS}
        )
set_target_properties(check-mx PROPERTIES FOLDER "Tests")

add_lit_testsuites(MX ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${MX_TEST_DEPENDS})

add_subdirectory(CAPI)
