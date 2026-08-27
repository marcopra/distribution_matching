"""Sample integer time steps by approximately solving gamma**t = r."""

import argparse

import numpy as np


def sample_time_steps(
    gamma: float,
    num_samples: int,
    seed: int | None = None,
    horizon: int | np.ndarray | None = None,
    rng=None,
) -> np.ndarray:
    """Sample t using r ~ U(gamma**horizon, 1), or U(0, 1) if unset.

    ``horizon`` may be one scalar or one value per sample. Passing ``rng``
    lets callers share an RNG across trajectory selection and time sampling.
    """
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must be strictly between 0 and 1")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if seed is not None and rng is not None:
        raise ValueError("seed and rng are mutually exclusive")

    horizons = None
    if horizon is not None:
        horizons = np.asarray(horizon)
        if horizons.ndim > 1:
            raise ValueError("horizon must be a scalar or one-dimensional")
        if horizons.ndim == 1 and horizons.shape[0] != num_samples:
            raise ValueError("vector horizon must contain num_samples values")
        if not np.issubdtype(horizons.dtype, np.integer):
            if not np.all(np.equal(horizons, np.floor(horizons))):
                raise ValueError("horizon values must be integers")
        horizons = horizons.astype(np.int64, copy=False)
        if np.any(horizons < 0):
            raise ValueError("horizon must be non-negative")

    rng = np.random.default_rng(seed) if rng is None else rng
    random_min = np.power(gamma, horizons) if horizons is not None else 0.0
    random_values = rng.uniform(random_min, 1.0, num_samples)

    # Avoid log(0) in the extremely unlikely event that the RNG returns 0.
    random_values = np.maximum(random_values, np.nextafter(0.0, 1.0))
    real_time_steps = np.log(random_values) / np.log(gamma)

    # Exact equality generally gives a non-integer t. Compare both neighboring
    # integers and keep the one whose discounted value is closest to r.
    lower = np.floor(real_time_steps).astype(np.int64)
    upper = np.ceil(real_time_steps).astype(np.int64)
    lower_error = np.abs(np.power(gamma, lower) - random_values)
    upper_error = np.abs(np.power(gamma, upper) - random_values)
    time_steps = np.where(lower_error <= upper_error, lower, upper)
    if horizons is not None:
        time_steps = np.minimum(time_steps, horizons)
    return time_steps


def plot_histogram(
    time_steps: np.ndarray, gamma: float, output: str | None = None
) -> None:
    """Plot a discrete histogram of sampled time steps."""
    import matplotlib.pyplot as plt

    minimum = int(time_steps.min())
    maximum = int(time_steps.max())
    bins = np.arange(minimum - 0.5, maximum + 1.5)

    plt.figure(figsize=(10, 6))
    plt.hist(time_steps, bins=bins, edgecolor="black", alpha=0.8)
    plt.xlabel("Integer time step t")
    plt.ylabel("Count")
    plt.title(f"Samples from gamma^t = r (gamma={gamma})")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150)
        print(f"Histogram saved to {output}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample integer t values from gamma**t = r, with r uniform in (0, 1)."
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Restrict t to [0, horizon] using r in [gamma**horizon, 1].",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="histogram.png",
        help="Save plot to this path instead of opening a window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    time_steps = sample_time_steps(
        args.gamma, args.num_samples, args.seed, args.horizon
    )
    plot_histogram(time_steps, args.gamma, args.output)


if __name__ == "__main__":
    main()
