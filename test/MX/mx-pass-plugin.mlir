// UNSUPPORTED: system-windows
// RUN: mlir-opt %s --load-pass-plugin=%mx_libs/MXPlugin%shlibext --pass-pipeline="builtin.module(mx-switch-bar-foo)" | FileCheck %s

module {
  // CHECK-LABEL: func @foo()
  func.func @bar() {
    return
  }

  // CHECK-LABEL: func @abar()
  func.func @abar() {
    return
  }
}
