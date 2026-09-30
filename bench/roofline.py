#!/usr/bin/env python3
"""
Analytical roofline for mx.block_matmul.

All numbers are computed from the BUFFERIZED MEMREF TYPES, not measured.
Operational intensity I = FLOPs / Bytes, Bytes = sum over operands of
(num_elements * element_byte_width), capacity model (each buffer once).

Two shapes are plotted:
  M=32  the test shape -- compute-bound, both variants right of the ridge
  M=1   decode (one token at a time) -- memory-bound, and where quantizing
        A alone buys almost nothing because B dominates the byte count

Each shape gives an f32 baseline (all f32) and an mx point (A = f8 mantissa +
f8 scale, B and acc f32). Ridge is REPRESENTATIVE: BW sourced (120 GB/s LPDDR5X
base M4), P-core peak rests on a reverse-engineered FMA-units factor -> band.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- format constants ---
BLOCK = 32
F8, F32 = 1, 4  # storage byte widths: f8E4M3FN/f8E8M0FNU -> 1, f32 -> 4

FLOPS_PER_POINT_MX = 3   # 1 MAC (2) + 1 dequant mul (1)
FLOPS_PER_POINT_F32 = 2  # 1 MAC

# --- machine ceilings (base M4) ---
BW_GBs = 120.0        # SOURCED: base M4 LPDDR5X unified memory bandwidth
PEAK_GFLOPS = 560.0   # REPRESENTATIVE: ~4 P-core x 4 FMA x 4 lanes x 2 x 4.4GHz
I_RIDGE = PEAK_GFLOPS / BW_GBs


class Variant:
    """One (shape, precision) point: FLOPs, bytes, and their ratio."""

    def __init__(self, label, flops, byte_parts):
        self.label = label
        self.flops = flops
        self.byte_parts = byte_parts          # dict: operand -> bytes
        self.bytes = sum(byte_parts.values())
        self.intensity = flops / self.bytes

    def attainable(self):
        return min(PEAK_GFLOPS, BW_GBs * self.intensity)

    def seconds(self):
        """Wall-clock under the same model: whichever ceiling binds."""
        return max(self.bytes / (BW_GBs * 1e9), self.flops / (PEAK_GFLOPS * 1e9))


def f32_variant(M, N, K, label):
    return Variant(
        label,
        flops=FLOPS_PER_POINT_F32 * M * N * K,
        byte_parts={"A": M * K * F32, "B": K * N * F32, "acc": M * N * F32},
    )


def mx_variant(M, N, K, label, block=BLOCK):
    return Variant(
        label,
        flops=FLOPS_PER_POINT_MX * M * N * K,
        byte_parts={
            "A_mant": M * K * F8,
            "A_scale": (M * K // block) * F8,
            "B": K * N * F32,
            "acc": M * N * F32,
        },
    )


class Shape:
    """A matmul shape with its f32 and mx variants side by side."""

    def __init__(self, M, N, K, name):
        self.M, self.N, self.K, self.name = M, N, K, name
        self.f32 = f32_variant(M, N, K, f"f32 {name}")
        self.mx = mx_variant(M, N, K, f"mx {name}")

    @property
    def intensity_shift(self):
        return self.mx.intensity / self.f32.intensity

    @property
    def byte_ratio(self):
        """Equal-FLOPs data-movement win. The number that is actually a win."""
        return self.f32.bytes / self.mx.bytes

    @property
    def flop_inflation(self):
        return self.mx.flops / self.f32.flops

    @property
    def time_ratio(self):
        """mx time / f32 time. Above 1.0 means the mx path is slower."""
        return self.mx.seconds() / self.f32.seconds()

    def report(self):
        print(f"--- {self.name}  (M={self.M}, N={self.N}, K={self.K}) ---")
        print(f"  bytes    f32 {self.f32.bytes:>7,}   mx {self.mx.bytes:>7,}")
        print(f"  FLOPs    f32 {self.f32.flops:>7,}   mx {self.mx.flops:>7,}")
        print(f"  I        f32 {self.f32.intensity:>7.4f}   mx {self.mx.intensity:>7.4f}")
        print(f"  shift    {self.intensity_shift:.3f}x"
              f"  =  bytes {self.byte_ratio:.3f}x"
              f"  x  FLOPs {self.flop_inflation:.3f}x")
        print(f"  time     f32 {self.f32.seconds():.3e}s  mx {self.mx.seconds():.3e}s"
              f"   mx/f32 = {self.time_ratio:.3f}x")


def plot_roofline(shapes, path_stem="roofline"):
    """Intensity against attainable rate, with each shape's shift decomposed."""
    fig, ax = plt.subplots(figsize=(7.4, 5.0))

    I = np.logspace(-1, 2.5, 500)
    ax.plot(I, np.minimum(PEAK_GFLOPS, BW_GBs * I), color="black", lw=1.8)
    ax.axvspan(I_RIDGE * 0.7, I_RIDGE * 1.3, color="gray", alpha=0.12, lw=0)
    ax.text(I_RIDGE, 12, f"ridge ~{I_RIDGE:.1f}", ha="center",
            fontsize=10, color="gray")

    markers = ["o", "s"]
    for shape, marker in zip(shapes, markers):
        for variant, color in ((shape.f32, "tab:blue"), (shape.mx, "tab:red")):
            ax.scatter([variant.intensity], [variant.attainable()],
                       color=color, marker=marker, zorder=5, s=64,
                       edgecolor="white", linewidth=0.6)
        y = shape.f32.attainable()
        ax.annotate("", xy=(shape.mx.intensity, y * 0.62),
                    xytext=(shape.f32.intensity, y * 0.62),
                    arrowprops=dict(arrowstyle="->", color="dimgray", lw=1.2))
        ax.text(shape.f32.intensity, y * 0.42,
                f"{shape.intensity_shift:.2f}x  =  {shape.byte_ratio:.2f}x bytes"
                f"  x  {shape.flop_inflation:.2f}x FLOPs",
                ha="left", va="top", fontsize=10, color="dimgray")

    for shape, marker in zip(shapes, markers):
        for variant, color in ((shape.f32, "tab:blue"), (shape.mx, "tab:red")):
            kind = "f32" if color == "tab:blue" else "mx"
            ax.scatter([], [], color=color, marker=marker, s=48,
                       label=f"{kind}, {shape.name}")
    ax.legend(loc="upper left", fontsize=10, frameon=False)
    ax.tick_params(labelsize=10)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Operational intensity  I  (FLOPs / byte)")
    ax.set_ylabel("Attainable performance  (GFLOP/s)")
    ax.grid(True, which="major", ls=":", alpha=0.35)
    ax.set_title("Analytical roofline\nN=64, K=64, block_size=32, each buffer counted once",
             fontsize=12)

    fig.tight_layout()
    fig.savefig(f"{path_stem}.png", dpi=150, facecolor="white")
    fig.savefig(f"{path_stem}.svg", facecolor="white")
    return fig, ax


def plot_time(shapes, path_stem="roofline-time"):
    """The same model read as a duration: whichever ceiling binds."""
    fig, bx = plt.subplots(figsize=(7.4, 4.0))

    labels, times, colors = [], [], []
    for shape in shapes:
        for variant, color in ((shape.f32, "tab:blue"), (shape.mx, "tab:red")):
            labels.append(f"{shape.name}\n{'f32' if color == 'tab:blue' else 'mx'}")
            times.append(variant.seconds() * 1e9)
            colors.append(color)

    bx.bar(range(len(times)), times, color=colors, width=0.62)
    bx.set_xticks(range(len(labels)))
    bx.set_xticklabels(labels, fontsize=10)
    bx.tick_params(axis="y", labelsize=10)
    bx.set_ylabel("Time per call  (ns, lower bound)")
    bx.grid(True, axis="y", ls=":", alpha=0.35)
    bx.set_ylim(0, max(times) * 1.15)
    for shape, i in zip(shapes, (0, 2)):
        bx.text(i + 0.5, max(times) * 1.05, f"{shape.time_ratio:.2f}x",
                ha="center", fontsize=10, color="dimgray")
    bx.set_title("Same model, wall-clock", fontsize=12)

    fig.tight_layout()
    fig.savefig(f"{path_stem}.png", dpi=150, facecolor="white")
    fig.savefig(f"{path_stem}.svg", facecolor="white")
    return fig, bx


def main():
    shapes = [
        Shape(32, 64, 64, "M=32"),
        Shape(1, 64, 64, "decode M=1"),
    ]
    for s in shapes:
        s.report()
    print(f"I_ridge = {I_RIDGE:.2f} FLOPs/byte")
    plot_roofline(shapes)
    plot_time(shapes)
    print("Saved roofline.{png,svg} and roofline-time.{png,svg}")


if __name__ == "__main__":
    main()