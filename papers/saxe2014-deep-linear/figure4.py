"""Reproduce the depth trend of Fig. 4 of Saxe et al. (2014).

The paper's aligned initialization confines every supervised gradient to a
10-dimensional hidden subspace. We evolve that active subspace rather than
materialize the inert 1000-dimensional orthogonal complement. This is an
exact reduction of Eq. (13) for the stated initialization, not a narrower
trainable model approximation. The input-facing factor remains 10 x 784, so
it can learn input directions outside the initial task SVD subspace.
"""

import argparse
import csv
import gzip
import hashlib
import json
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PAPER_DEPTHS = (3, 5, 8, 10, 14, 20, 28, 36, 44, 54, 64, 74, 84, 94, 100)
IMAGE_FILE = "train-images-idx3-ubyte.gz"
LABEL_FILE = "train-labels-idx1-ubyte.gz"


def read_idx_gzip(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        magic, count = struct.unpack(">II", stream.read(8))
        if magic == 2051:
            rows, cols = struct.unpack(">II", stream.read(8))
            data = np.frombuffer(stream.read(), dtype=np.uint8)
            if len(data) != count * rows * cols:
                raise ValueError(f"Truncated MNIST images: {path}")
            return data.reshape(count, rows * cols)
        if magic == 2049:
            data = np.frombuffer(stream.read(), dtype=np.uint8)
            if len(data) != count:
                raise ValueError(f"Truncated MNIST labels: {path}")
            return data
    raise ValueError(f"Unexpected IDX magic number {magic}: {path}")


def load_statistics(data_dir: Path, count: int = 50000):
    raw = data_dir / "MNIST" / "raw"
    images = read_idx_gzip(raw / IMAGE_FILE)
    labels = read_idx_gzip(raw / LABEL_FILE)
    if len(images) < count or len(labels) < count:
        raise ValueError("Insufficient training samples")
    # The paper does not specify further preprocessing. Use pixel / 255,
    # without centering or whitening, and document this choice in README.
    x = images[:count].astype(np.float32) / 255.0
    y = np.eye(10, dtype=np.float32)[labels[:count]]
    xx = x.T @ x
    yx = y.T @ x
    return xx, yx, count


def initial_factors(depth, rates_count, yx, u0, device):
    u, _, vt = np.linalg.svd(yx, full_matrices=False)
    u = torch.as_tensor(u.copy(), dtype=torch.float32, device=device)
    vt = torch.as_tensor(vt.copy(), dtype=torch.float32, device=device)
    scale = u0 ** (1.0 / (depth - 1))
    eye = torch.eye(10, dtype=torch.float32, device=device)
    return [
        (scale * vt).unsqueeze(0).expand(rates_count, -1, -1).clone(),
        *[(scale * eye).unsqueeze(0).expand(rates_count, -1, -1).clone()
          for _ in range(depth - 3)],
        (scale * u).unsqueeze(0).expand(rates_count, -1, -1).clone(),
    ]


def one_step(weights, xx, yx, y_norm_sq, rates, active):
    """Full-batch Eq. (13), simultaneously for all learning rates."""
    prefix = [None] * len(weights)
    prefix[1] = weights[0]
    for layer in range(2, len(weights)):
        prefix[layer] = torch.bmm(weights[layer - 1], prefix[layer - 1])
    prediction = torch.bmm(weights[-1], prefix[-1])
    prediction_xx = torch.matmul(prediction, xx)
    error = 0.5 * (y_norm_sq - 2 * (prediction * yx).sum((1, 2))
                   + (prediction * prediction_xx).sum((1, 2)))

    residual = yx.unsqueeze(0) - prediction_xx
    upper = torch.eye(yx.shape[0], device=xx.device).unsqueeze(0).expand(len(rates), -1, -1)
    gradients = [None] * len(weights)
    for layer in range(len(weights) - 1, -1, -1):
        partial = torch.bmm(upper.transpose(1, 2), residual)
        gradients[layer] = (partial if layer == 0 else
                            torch.bmm(partial, prefix[layer].transpose(1, 2)))
        upper = torch.bmm(upper, weights[layer])
    step_sizes = (rates * active.to(rates.dtype)).reshape(-1, 1, 1)
    for layer in range(len(weights)):
        weights[layer] = weights[layer] + step_sizes * gradients[layer]
    return error


def run_depth(depth, rates, xx, yx, y_norm_sq, u0, threshold, max_epochs, device):
    weights = initial_factors(depth, len(rates), yx.cpu().numpy(), u0, device)
    rate_tensor = torch.tensor(rates, dtype=torch.float32, device=device)
    crossing = [None] * len(rates)
    failure = [False] * len(rates)
    history = []
    for epoch in range(max_epochs + 1):
        active = torch.tensor([c is None and not f for c, f in zip(crossing, failure)],
                              dtype=torch.bool, device=device)
        errors = one_step(weights, xx, yx, y_norm_sq, rate_tensor, active)
        values = errors.detach().cpu().numpy().tolist()
        history.append([float(value) if np.isfinite(value) else None for value in values])
        for i, value in enumerate(values):
            if crossing[i] is None and not failure[i]:
                if not np.isfinite(value) or value > 1e9:
                    failure[i] = True
                elif value < threshold:
                    crossing[i] = epoch
        if all(c is not None or f for c, f in zip(crossing, failure)):
            break
    eligible = [(epoch, i) for i, epoch in enumerate(crossing) if epoch is not None]
    if not eligible:
        return {"depth": depth, "best_epochs": None, "best_rate": None,
                "crossings": crossing, "failed": failure, "history": history}
    best_epochs, best_index = min(eligible)
    return {"depth": depth, "best_epochs": best_epochs,
            "best_rate": float(rates[best_index]), "crossings": crossing,
            "failed": failure, "history": history, "best_index": best_index}


def plot_results(results, out_dir):
    valid = [r for r in results if r["best_epochs"] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    axes[0].plot([r["depth"] for r in valid], [r["best_epochs"] for r in valid], "o-")
    axes[0].set(xlabel="Number of layers", ylabel="Iterations to error threshold",
                title="Figure 4: learning time")
    axes[1].plot([r["depth"] for r in valid], [r["best_rate"] for r in valid], "o-")
    axes[1].set(xlabel="Number of layers", ylabel="Selected learning rate",
                title="Figure 4: learning rate")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(out_dir / "figure4_reproduction.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--depths", type=int, nargs="+", default=PAPER_DEPTHS)
    parser.add_argument("--rates", type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    xx_np, yx_np, count = load_statistics(args.data)
    device = torch.device(args.device)
    xx = torch.as_tensor(xx_np, device=device)
    yx = torch.as_tensor(yx_np, device=device)
    rates = np.logspace(-7, -4, args.rates)
    threshold, u0 = 1.3e4, 0.001
    results = []
    for depth in args.depths:
        result = run_depth(depth, rates, xx, yx, float(count), u0,
                           threshold, args.max_epochs, device)
        results.append(result)
        print(f"depth={depth:3d} epochs={result['best_epochs']} "
              f"rate={result['best_rate']}", flush=True)

    plot_results(results, args.output)
    with (args.output / "figure4_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("depth", "best_epochs", "best_rate"))
        writer.writerows((r["depth"], r["best_epochs"], r["best_rate"]) for r in results)
    compact_results = []
    for result in results:
        compact = {key: value for key, value in result.items() if key != "history"}
        if result["best_epochs"] is not None:
            last = result["best_epochs"]
            best = result["best_index"]
            compact["best_curve"] = [
                [epoch, row[best]] for epoch, row in enumerate(result["history"])
                if epoch <= last and (epoch % 5 == 0 or epoch == last)
            ]
        compact_results.append(compact)
    with (args.output / "figure4_details.json").open("w", encoding="utf-8") as f:
        raw = args.data / "MNIST" / "raw"
        json.dump({"configuration": {"samples": count, "depths": args.depths,
                   "rates": rates.tolist(), "threshold": threshold, "u0": u0,
                   "max_epochs": args.max_epochs, "device": args.device,
                   "input_scaling": "pixel / 255, no centering or whitening",
                   "dtype": "float32", "torch_version": torch.__version__,
                   "numpy_version": np.__version__,
                   "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                   "input_md5": {
                       name: hashlib.md5((raw / name).read_bytes()).hexdigest()
                       for name in (IMAGE_FILE, LABEL_FILE)}},
                   "results": compact_results}, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
