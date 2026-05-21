# RUN: %python %s 2>&1 | FileCheck %s
import sys

# CHECK: Testing mlir_mx package
print("Testing mlir_mx package", file=sys.stderr)

import mlir_mx.ir
from mlir_mx.dialects import mx_nanobind as mx_d

with mlir_mx.ir.Context():
    mx_d.register_dialects()
    mx_module = mlir_mx.ir.Module.parse(
        """
    %0 = arith.constant 2 : i32
    %1 = mx.foo %0 : i32
    """
    )
    # CHECK: %[[C2:.*]] = arith.constant 2 : i32
    # CHECK: mx.foo %[[C2]] : i32
    print(str(mx_module), file=sys.stderr)

    custom_type = mx_d.CustomType.get("foo")
    # CHECK: !mx.custom<"foo">
    print(custom_type, file=sys.stderr)

    # CHECK: this is a fp16 type
    mx_d.print_fp_type(mlir_mx.ir.F16Type.get(), sys.stderr)
    # CHECK: this is a fp32 type
    mx_d.print_fp_type(mlir_mx.ir.F32Type.get(), sys.stderr)
    # CHECK: this is a fp64 type
    mx_d.print_fp_type(mlir_mx.ir.F64Type.get(), sys.stderr)


# CHECK: Testing mlir package
print("Testing mlir package", file=sys.stderr)

from mlir.ir import *

# CHECK-NOT: RuntimeWarning: nanobind: type '{{.*}}' was already registered!
