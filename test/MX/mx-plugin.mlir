// UNSUPPORTED: system-windows
// RUN: mlir-opt %s --load-dialect-plugin=%mx_libs/MXPlugin%shlibext --pass-pipeline="builtin.module(mx-switch-bar-foo)" | FileCheck %s

module {
  // CHECK-LABEL: func @foo()
  func.func @bar() {
    return
  }

  // CHECK-LABEL: func @mx_types(%arg0: !mx.custom<"10">)
  func.func @mx_types(%arg0: !mx.custom<"10">) {
    return
  }
}
