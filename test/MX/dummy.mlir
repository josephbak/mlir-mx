// RUN: mx-opt %s | mx-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = mx.foo %{{.*}} : i32
        %res = mx.foo %0 : i32
        return
    }

    // CHECK-LABEL: func @mx_types(%arg0: !mx.custom<"10">)
    func.func @mx_types(%arg0: !mx.custom<"10">) {
        return
    }
}
