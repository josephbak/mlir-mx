#!/usr/bin/env python3
"""
Analytical roofline for mx.block_matmul (M=32, N=64, K=64, block_size=32).

All numbers are computed from the BUFFERIZED MEMREF TYPES, not measured.
Operational intensity I = FLOPs / Bytes, Bytes = sum over operands of
(num_elements * element_byte_width), capacity model (each buffer once).

Points: f32 baseline (all f32) vs mx block_matmul (A = f8 mantissa + f8 scale,
B and acc f32). Ridge is REPRESENTATIVE: BW sourced (120 GB/s LPDDR5X base M4),
P-core peak rests on a reverse-engineered FMA-units factor -> drawn as a band.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- constants ---
M, N, K = 32, 64, 64
BLOCK = 32
F8, F32 = 1, 4  # storage byte widths: f8E4M3FN/f8E8M0FNU -> 1, f32 -> 4

# --- FLOPs ---
points = M * N * K
FLOPS_PER_POINT_MX  = 3   # 1 MAC (2) + 1 dequant mul (1)
FLOPS_PER_POINT_F32 = 2   # 1 MAC
flops_mx  = points * FLOPS_PER_POINT_MX
flops_f32 = points * FLOPS_PER_POINT_F32

# --- bytes (capacity model: each memref once) ---
bytes_mx = (        # 2048 + 64 + 16384 + 8192 (A_mant, A_scale, B, acc)
    M*K*F8              # A mantissa
    + (M*K//BLOCK)*F8   # A scale
    + K*N*F32           # B
    + M*N*F32           # acc/C
)
bytes_f32 = (       # 8192 + 16384 + 8192 (A, B, acc)
    M*K*F32             # A
    + K*N*F32           # B
    + M*N*F32           # acc/C
)

# --- intensity ---
I_mx  = flops_mx  / bytes_mx
I_f32 = flops_f32 / bytes_f32

# --- machine ceilings (base M4) ---
BW_GBs = 120.0          # SOURCED: base M4 LPDDR5X unified memory bandwidth
PEAK_GFLOPS = 560.0     # REPRESENTATIVE: ~4 P-core x 4 FMA x 4 lanes x 2 x 4.4GHz (reverse-engineered)
I_ridge = PEAK_GFLOPS / BW_GBs

def main():
    I = np.logspace(-1, 2.5, 500)
    roofline = np.minimum(PEAK_GFLOPS, BW_GBs * I)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(I, roofline, color="black", lw=2,
            label="Roofline (base M4, representative)")

    ridge_lo, ridge_hi = I_ridge * 0.7, I_ridge * 1.3
    ax.axvspan(ridge_lo, ridge_hi, color="gray", alpha=0.15,
               label=f"Ridge ~{I_ridge:.1f} FLOPs/byte (representative)")

    ax.scatter([I_f32], [min(PEAK_GFLOPS, BW_GBs * I_f32)],
               color="tab:blue", zorder=5, s=70,
               label=f"f32 baseline  (I={I_f32:.2f})")
    ax.scatter([I_mx], [min(PEAK_GFLOPS, BW_GBs * I_mx)],
               color="tab:red", zorder=5, s=70,
               label=f"mx block_matmul  (I={I_mx:.2f})")

    ax.annotate("", xy=(I_mx, PEAK_GFLOPS * 0.55),
                xytext=(I_f32, PEAK_GFLOPS * 0.55),
                arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.5))
    ax.text((I_f32 * I_mx) ** 0.5, PEAK_GFLOPS * 0.6,
            f"{I_mx / I_f32:.2f}x", ha="center", color="tab:red", fontsize=11)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Operational intensity  I  (FLOPs / byte)")
    ax.set_ylabel("Attainable performance  (GFLOP/s)")
    ax.set_title("Analytical roofline: mx.block_matmul vs f32 baseline\n"
                 "M=32, N=64, K=64, block_size=32 (capacity model)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig("roofline.png", dpi=150)
    print(f"I_f32 = {I_f32:.4f}  I_mx = {I_mx:.4f}  shift = {I_mx/I_f32:.3f}x")
    print(f"bytes_f32 = {bytes_f32}  bytes_mx = {bytes_mx}")
    print(f"byte-ratio (equal-FLOPs data-movement win) = {bytes_f32/bytes_mx:.3f}x")
    print(f"I_ridge = {I_ridge:.2f} FLOPs/byte")
    print("Saved roofline.png")

if __name__ == "__main__":
    main()