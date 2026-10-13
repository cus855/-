"""Generate numerical checks and figures for Eqs. (12) and (15)."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core import eq12, integrate_eq15


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    s, u0, tau = 1.0, 0.001, 1.0
    times = np.linspace(0.0, 30.0, 30001)
    exact = eq12(times, s, u0, tau)
    trajectories = {depth: integrate_eq15(times, s, u0, depth, tau)
                    for depth in (3, 5, 10, 30, 100)}
    max_abs_error = float(np.max(np.abs(trajectories[3] - exact)))
    if max_abs_error > 1e-8:
        raise RuntimeError("Eq. (15) at depth 3 did not reproduce Eq. (12)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(times, exact, label="Eq. (12), exact", linewidth=2)
    axes[0].plot(times[::100], trajectories[3][::100], ".", ms=2,
                 label="Eq. (15), RK4 (3 layers)")
    axes[0].set_title("Three-layer mode")
    for depth, values in trajectories.items():
        axes[1].plot(times, values, label=f"{depth} layers")
    axes[1].set_title("Balanced deep modes: Eq. (15)")
    for ax in axes:
        ax.set(xlabel="Continuous time t", ylabel="End-to-end mode strength u")
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(args.output / "equations_12_15.png", dpi=180)
    plt.close(fig)

    with (args.output / "equations_12_15.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "eq12_exact"] + [f"eq15_depth_{d}" for d in trajectories])
        for i in range(0, len(times), 100):
            t = times[i]
            writer.writerow([f"{t:.6f}", f"{exact[i]:.12g}"] +
                            [f"{v[i]:.12g}" for v in trajectories.values()])
    print(f"Eq. (12) versus Eq. (15), depth 3: max absolute error = {max_abs_error:.3e}")
    print(f"Wrote {args.output / 'equations_12_15.png'}")


if __name__ == "__main__":
    main()
