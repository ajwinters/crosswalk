"""
Plot the CPU-vs-GPU matching benchmark (time vs dataset size), log-log.

Data are the measured results from benchmark_1m.py (run 2026-06-26, RTX 4070 Ti
SUPER 16 GB, in-container). CPU is measured where it can run (10k, 25k) and
EXTRAPOLATED from its measured throughput beyond that (the CPU cannot actually
run >=100k -- it OOMs materializing the candidate-pair index).

  docker run --rm -v "<repo>:/workspace/crosswalk" crosswalk-gpu \
      python /workspace/crosswalk/scripts/plot_benchmark.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- measured results (seconds) ---
records = [10_000, 25_000, 100_000, 500_000, 1_000_000]
gpu = [0.3, 0.5, 2.9, 59.9, 237.8]

cpu_x_meas = [10_000, 25_000]
cpu_y_meas = [7.0, 35.5]
# extrapolated from measured CPU throughput (~0.4M pairs/s); cannot actually run
cpu_x_extra = [25_000, 100_000, 500_000, 1_000_000]
cpu_y_extra = [35.5, 570.0, 14238.0, 56946.0]  # 9.5 / 237.3 / 949.1 min

speedups = {10_000: "21x", 25_000: "79x", 100_000: "196x",
            500_000: "238x", 1_000_000: "239x"}

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_REPO, "docs", "benchmark_cpu_vs_gpu.png")


def main():
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(records, gpu, "o-", color="#1f77b4", lw=2.2, ms=8,
            label="GPU (streaming, measured)", zorder=3)
    ax.plot(cpu_x_meas, cpu_y_meas, "s-", color="#d62728", lw=2.2, ms=8,
            label="CPU (measured)", zorder=3)
    ax.plot(cpu_x_extra, cpu_y_extra, "s--", color="#d62728", lw=1.8, ms=8,
            mfc="white", label="CPU (extrapolated — cannot run, OOM)", zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("records per dataset")
    ax.set_ylabel("wall time, seconds (Indexing + Comparing + Classifier)")
    ax.set_title("CPU vs GPU record-linkage matching — time vs dataset size")
    ax.grid(True, which="both", ls=":", alpha=0.5)

    # speedup labels above each GPU point
    for x, yg in zip(records, gpu):
        ax.annotate(speedups[x], (x, yg), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=9, color="#1f77b4")

    # human-readable callouts at 1M
    ax.annotate("GPU: 238 s (~4 min)", (1_000_000, 237.8),
                textcoords="offset points", xytext=(-150, 18), fontsize=9,
                color="#1f77b4",
                arrowprops=dict(arrowstyle="->", color="#1f77b4"))
    ax.annotate("CPU: ~56,900 s (~15.8 h)\nand OOMs", (1_000_000, 56946.0),
                textcoords="offset points", xytext=(-170, -5), fontsize=9,
                color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728"))

    # secondary axis: human time units
    secax = ax.secondary_yaxis("right")
    secax.set_yticks([1, 60, 600, 3600, 36000])
    secax.set_yticklabels(["1 s", "1 min", "10 min", "1 h", "10 h"])

    ax.legend(loc="upper left", framealpha=0.95)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
